"""Equal-storage students for predictive-state compilation."""
from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from representation_experiments.common import parameter_count


class StudentBase(nn.Module):
    def __init__(self, vocab_size: int, width: int, max_context: int):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.width = int(width)
        self.max_context = int(max_context)
        self.embedding = nn.Embedding(vocab_size, width)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=width ** -0.5)
        self.final_norm = nn.LayerNorm(width)

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.linear(self.final_norm(hidden), self.embedding.weight)

    def stored_state_elements(self, batch_size: int = 1, context: int = 512) -> int:
        raise NotImplementedError

    def output_head_macs_per_token(self) -> int:
        return self.width * self.vocab_size

    def estimated_macs_per_token(self, context: int = 512) -> int:
        raise NotImplementedError


class TransformerStudent(StudentBase):
    kind = "transformer"

    def __init__(self, vocab_size: int, width: int, max_context: int,
                 layers: int, heads: int, expansion: int = 4,
                 position_encoding: str = "learned"):
        super().__init__(vocab_size, width, max_context)
        if width % heads:
            raise ValueError("transformer width must divide evenly into heads")
        if position_encoding not in ("learned", "sinusoidal"):
            raise ValueError("position_encoding must be learned or sinusoidal")
        self.layers = int(layers)
        self.heads = int(heads)
        self.expansion = int(expansion)
        self.position_encoding = position_encoding
        self.position = (nn.Embedding(max_context, width)
                         if position_encoding == "learned" else None)
        block = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=expansion * width,
            activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(block, num_layers=layers)

    def forward(self, input_ids: torch.Tensor):
        length = input_ids.shape[1]
        if length > self.max_context:
            input_ids = input_ids[:, -self.max_context:]
            length = self.max_context
        positions = torch.arange(length, device=input_ids.device)
        if self.position is not None:
            position_values = self.position(positions)
        else:
            half = self.width // 2
            frequencies = torch.exp(
                -math.log(10000.0) * torch.arange(
                    half, device=input_ids.device, dtype=torch.float32)
                / max(half - 1, 1))
            angles = positions.float().unsqueeze(1) * frequencies.unsqueeze(0)
            position_values = torch.cat((angles.sin(), angles.cos()), dim=-1)
            if position_values.shape[-1] < self.width:
                position_values = F.pad(position_values, (0, 1))
            position_values = position_values.to(self.embedding.weight.dtype)
        hidden = self.embedding(input_ids) + position_values.unsqueeze(0)
        mask = torch.ones(length, length, device=input_ids.device, dtype=torch.bool).triu(1)
        hidden = self.blocks(hidden, mask=mask, is_causal=True)
        return SimpleNamespace(logits=self.logits(hidden), hidden_states=hidden)

    def stored_state_elements(self, batch_size: int = 1, context: int = 512) -> int:
        # Standard autoregressive KV cache, two tensors per layer.
        return 2 * batch_size * min(context, self.max_context) * self.width * self.layers

    def estimated_macs_per_token(self, context: int = 512) -> int:
        feedforward = 2 * self.expansion * self.width * self.width
        projections = 4 * self.width * self.width
        attention = 2 * min(context, self.max_context) * self.width
        return (self.output_head_macs_per_token()
                + self.layers * (feedforward + projections + attention))


class RecurrentLogitStudent(StudentBase):
    kind = "recurrent_logits"

    def __init__(self, vocab_size: int, width: int, max_context: int, layers: int):
        super().__init__(vocab_size, width, max_context)
        self.layers = int(layers)
        self.cells = nn.ModuleList([
            nn.GRUCell(width, width) for _ in range(layers)
        ])

    def initial_state(self, batch_size: int, device: torch.device,
                      dtype: torch.dtype | None = None):
        dtype = dtype or self.embedding.weight.dtype
        return [torch.zeros(batch_size, self.width, device=device, dtype=dtype)
                for _ in range(self.layers)]

    def step(self, token_ids: torch.Tensor, state=None):
        if state is None:
            state = self.initial_state(token_ids.shape[0], token_ids.device)
        hidden = self.embedding(token_ids)
        new_state = []
        for cell, previous in zip(self.cells, state):
            hidden = cell(hidden, previous)
            new_state.append(hidden)
        return self.logits(hidden), new_state, hidden

    def transition(self, token_ids: torch.Tensor, state=None):
        if state is None:
            state = self.initial_state(token_ids.shape[0], token_ids.device)
        hidden = self.embedding(token_ids)
        new_state = []
        for cell, previous in zip(self.cells, state):
            hidden = cell(hidden, previous)
            new_state.append(hidden)
        return hidden, new_state

    def forward(self, input_ids: torch.Tensor):
        state = None
        hidden_values = []
        for index in range(input_ids.shape[1]):
            hidden, state = self.transition(input_ids[:, index], state)
            hidden_values.append(hidden)
        if not hidden_values:
            raise ValueError("forward requires at least one input token")
        stacked_hidden = torch.stack(hidden_values, dim=1)
        return SimpleNamespace(
            logits=self.logits(stacked_hidden),
            hidden_states=stacked_hidden,
            final_state=state)

    def last_logits(self, input_ids: torch.Tensor):
        state = None
        token_logits = None
        for index in range(input_ids.shape[1]):
            token_logits, state, _ = self.step(input_ids[:, index], state)
        if token_logits is None:
            raise ValueError("last_logits requires at least one input token")
        return token_logits.unsqueeze(1)

    def stored_state_elements(self, batch_size: int = 1, context: int = 512) -> int:
        del context
        return batch_size * self.width * self.layers

    def estimated_macs_per_token(self, context: int = 512) -> int:
        del context
        gru = 6 * self.width * self.width
        return self.output_head_macs_per_token() + self.layers * gru


class HierarchicalPredictiveStudent(StudentBase):
    kind = "hierarchical_predictive"

    def __init__(self, vocab_size: int, width: int, max_context: int,
                 atoms: int, segment_length: int, document_segments: int,
                 future_horizons: list[int]):
        super().__init__(vocab_size, width, max_context)
        self.atoms = int(atoms)
        self.segment_length = int(segment_length)
        self.document_segments = int(document_segments)
        self.future_horizons = tuple(int(value) for value in future_horizons)
        self.fast_cell = nn.GRUCell(3 * width, width)
        self.segment_cell = nn.GRUCell(width, width)
        self.document_cell = nn.GRUCell(width, width)
        self.combine = nn.Sequential(
            nn.Linear(3 * width, 2 * width), nn.SiLU(), nn.Linear(2 * width, width))
        self.atom_head = nn.Linear(width, atoms)
        self.future_heads = nn.ModuleList([
            nn.Linear(width, atoms) for _ in self.future_horizons
        ])

    def initial_state(self, batch_size: int, device: torch.device,
                      dtype: torch.dtype | None = None):
        dtype = dtype or self.embedding.weight.dtype
        zero = lambda: torch.zeros(batch_size, self.width, device=device, dtype=dtype)
        return {
            "fast": zero(), "segment": zero(), "document": zero(),
            "segment_sum": zero(), "segment_tokens": 0, "segments": 0,
        }

    def step(self, token_ids: torch.Tensor, state=None):
        hidden, new_state = self.transition(token_ids, state)
        return {
            "logits": self.logits(hidden),
            "atom_logits": self.atom_head(hidden),
            "future_atom_logits": [head(hidden) for head in self.future_heads],
            "hidden": hidden,
        }, new_state

    def transition(self, token_ids: torch.Tensor, state=None):
        if state is None:
            state = self.initial_state(token_ids.shape[0], token_ids.device)
        token = self.embedding(token_ids)
        fast_input = torch.cat((token, state["segment"], state["document"]), dim=-1)
        fast = self.fast_cell(fast_input, state["fast"])
        segment_sum = state["segment_sum"] + fast
        segment_tokens = int(state["segment_tokens"]) + 1
        segment = state["segment"]
        document = state["document"]
        segments = int(state["segments"])
        if segment_tokens == self.segment_length:
            segment = self.segment_cell(segment_sum / segment_tokens, segment)
            segment_sum = torch.zeros_like(segment_sum)
            segment_tokens = 0
            segments += 1
            if segments == self.document_segments:
                document = self.document_cell(segment, document)
                segments = 0
        combined = self.combine(torch.cat((fast, segment, document), dim=-1))
        new_state = {
            "fast": fast, "segment": segment, "document": document,
            "segment_sum": segment_sum, "segment_tokens": segment_tokens,
            "segments": segments,
        }
        return combined, new_state

    def forward(self, input_ids: torch.Tensor):
        state = None
        hidden_values = []
        for index in range(input_ids.shape[1]):
            hidden, state = self.transition(input_ids[:, index], state)
            hidden_values.append(hidden)
        if not hidden_values:
            raise ValueError("forward requires at least one input token")
        stacked_hidden = torch.stack(hidden_values, dim=1)
        return SimpleNamespace(
            logits=self.logits(stacked_hidden),
            atom_logits=self.atom_head(stacked_hidden),
            future_atom_logits=[head(stacked_hidden) for head in self.future_heads],
            hidden_states=stacked_hidden,
            final_state=state)

    def stored_state_elements(self, batch_size: int = 1, context: int = 512) -> int:
        del context
        # fast, segment, document, and the active segment accumulator.
        return 4 * batch_size * self.width

    def estimated_macs_per_token(self, context: int = 512) -> int:
        del context
        fast = 12 * self.width * self.width
        slower = (6 * self.width * self.width / self.segment_length
                   + 6 * self.width * self.width
                   / (self.segment_length * self.document_segments))
        combine = 8 * self.width * self.width
        probes = self.width * self.atoms * (1 + len(self.future_horizons))
        return int(self.output_head_macs_per_token() + fast + slower + combine + probes)


class HierarchicalDistributionStudent(StudentBase):
    """Three-timescale recurrent state without the failed discrete atom heads."""

    kind = "hierarchical_distribution"

    def __init__(self, vocab_size: int, width: int, max_context: int,
                 segment_length: int, document_segments: int):
        super().__init__(vocab_size, width, max_context)
        self.segment_length = int(segment_length)
        self.document_segments = int(document_segments)
        self.fast_cell = nn.GRUCell(3 * width, width)
        self.segment_cell = nn.GRUCell(width, width)
        self.document_cell = nn.GRUCell(width, width)
        self.combine = nn.Sequential(
            nn.Linear(3 * width, 2 * width), nn.SiLU(), nn.Linear(2 * width, width))

    def initial_state(self, batch_size: int, device: torch.device,
                      dtype: torch.dtype | None = None):
        dtype = dtype or self.embedding.weight.dtype
        zero = lambda: torch.zeros(batch_size, self.width, device=device, dtype=dtype)
        return {
            "fast": zero(), "segment": zero(), "document": zero(),
            "segment_sum": zero(), "segment_tokens": 0, "segments": 0,
        }

    def transition(self, token_ids: torch.Tensor, state=None):
        if state is None:
            state = self.initial_state(token_ids.shape[0], token_ids.device)
        token = self.embedding(token_ids)
        fast_input = torch.cat((token, state["segment"], state["document"]), dim=-1)
        fast = self.fast_cell(fast_input, state["fast"])
        segment_sum = state["segment_sum"] + fast
        segment_tokens = int(state["segment_tokens"]) + 1
        segment = state["segment"]
        document = state["document"]
        segments = int(state["segments"])
        if segment_tokens == self.segment_length:
            segment = self.segment_cell(segment_sum / segment_tokens, segment)
            segment_sum = torch.zeros_like(segment_sum)
            segment_tokens = 0
            segments += 1
            if segments == self.document_segments:
                document = self.document_cell(segment, document)
                segments = 0
        combined = self.combine(torch.cat((fast, segment, document), dim=-1))
        new_state = {
            "fast": fast, "segment": segment, "document": document,
            "segment_sum": segment_sum, "segment_tokens": segment_tokens,
            "segments": segments,
        }
        return combined, new_state

    def step(self, token_ids: torch.Tensor, state=None):
        hidden, new_state = self.transition(token_ids, state)
        return self.logits(hidden), new_state, hidden

    def forward(self, input_ids: torch.Tensor):
        state = None
        hidden_values = []
        for index in range(input_ids.shape[1]):
            hidden, state = self.transition(input_ids[:, index], state)
            hidden_values.append(hidden)
        if not hidden_values:
            raise ValueError("forward requires at least one input token")
        stacked_hidden = torch.stack(hidden_values, dim=1)
        return SimpleNamespace(
            logits=self.logits(stacked_hidden),
            hidden_states=stacked_hidden,
            final_state=state)

    def last_logits(self, input_ids: torch.Tensor):
        state = None
        hidden = None
        for index in range(input_ids.shape[1]):
            hidden, state = self.transition(input_ids[:, index], state)
        if hidden is None:
            raise ValueError("last_logits requires at least one input token")
        return self.logits(hidden).unsqueeze(1)

    def stored_state_elements(self, batch_size: int = 1, context: int = 512) -> int:
        del context
        return 4 * batch_size * self.width

    def estimated_macs_per_token(self, context: int = 512) -> int:
        del context
        fast = 12 * self.width * self.width
        slower = (6 * self.width * self.width / self.segment_length
                   + 6 * self.width * self.width
                   / (self.segment_length * self.document_segments))
        combine = 8 * self.width * self.width
        return int(self.output_head_macs_per_token() + fast + slower + combine)


class CompressedLocalAttention(nn.Module):
    """Exact causal softmax attention over a fixed compressed local cache."""

    def __init__(self, width: int, local_width: int, heads: int, window: int,
                 gate_bias: float):
        super().__init__()
        if min(width, local_width, heads, window) <= 0:
            raise ValueError("local attention dimensions must be positive")
        if local_width % heads:
            raise ValueError("local attention width must divide evenly into heads")
        self.local_width = int(local_width)
        self.heads = int(heads)
        self.head_width = self.local_width // self.heads
        self.window = int(window)
        self.input_norm = nn.LayerNorm(width)
        self.query = nn.Linear(width, self.local_width, bias=False)
        self.key = nn.Linear(width, self.local_width, bias=False)
        self.value = nn.Linear(width, self.local_width, bias=False)
        self.output = nn.Linear(self.local_width, width, bias=False)
        self.output_norm = nn.LayerNorm(width)
        self.gate = nn.Parameter(torch.tensor(float(gate_bias)))

    def forward(self, hidden: torch.Tensor, keys: torch.Tensor | None,
                values: torch.Tensor | None):
        batch = hidden.shape[0]
        normalized = self.input_norm(hidden)
        query = self.query(normalized).view(
            batch, self.heads, self.head_width)
        key = self.key(normalized).view(
            batch, self.heads, 1, self.head_width)
        value = self.value(normalized).view(
            batch, self.heads, 1, self.head_width)
        keys = key if keys is None else torch.cat((keys, key), dim=2)
        values = value if values is None else torch.cat((values, value), dim=2)
        keys = keys[:, :, -self.window:]
        values = values[:, :, -self.window:]
        scores = torch.einsum("bhd,bhtd->bht", query, keys)
        scores = scores / math.sqrt(self.head_width)
        probabilities = torch.softmax(scores.float(), dim=-1).to(values.dtype)
        context = torch.einsum("bht,bhtd->bhd", probabilities, values)
        context = context.reshape(batch, self.local_width)
        result = self.output_norm(
            hidden + torch.sigmoid(self.gate) * self.output(context))
        return result, keys, values

    def stored_state_elements(self, batch_size: int = 1) -> int:
        return 2 * batch_size * self.window * self.local_width

    def estimated_macs_per_token(self, model_width: int) -> int:
        projections = 4 * model_width * self.local_width
        attention = 2 * self.window * self.local_width
        return projections + attention


class HierarchicalLocalDistributionStudent(HierarchicalDistributionStudent):
    """Hierarchical recurrence plus an exact, fixed-size local attention cache."""

    kind = "hierarchical_local_distribution"

    def __init__(self, vocab_size: int, width: int, max_context: int,
                 segment_length: int, document_segments: int,
                 local_attention: dict):
        super().__init__(
            vocab_size, width, max_context, segment_length, document_segments)
        self.local_attention = CompressedLocalAttention(
            width=width,
            local_width=int(local_attention["width"]),
            heads=int(local_attention["heads"]),
            window=int(local_attention["window"]),
            gate_bias=float(local_attention["gate_bias"]),
        )

    def transition(self, token_ids: torch.Tensor, state=None):
        hidden, new_state = super().transition(token_ids, state)
        keys = None if state is None else state.get("local_keys")
        values = None if state is None else state.get("local_values")
        hidden, keys, values = self.local_attention(hidden, keys, values)
        new_state["local_keys"] = keys
        new_state["local_values"] = values
        return hidden, new_state

    def stored_state_elements(self, batch_size: int = 1, context: int = 512) -> int:
        del context
        return (4 * batch_size * self.width
                + self.local_attention.stored_state_elements(batch_size))

    def estimated_macs_per_token(self, context: int = 512) -> int:
        return (super().estimated_macs_per_token(context)
                + self.local_attention.estimated_macs_per_token(self.width))


def clamped_cubic_design(points: int, controls: int,
                         dtype=torch.float32) -> torch.Tensor:
    """Open-uniform cubic B-spline controls sampled at virtual depths."""
    degree = 3
    if points < 1:
        raise ValueError("virtual depth must be positive")
    if controls <= degree:
        raise ValueError("a cubic spline needs at least four controls")
    interior_count = controls - degree - 1
    if interior_count:
        interior = torch.arange(
            1, interior_count + 1, dtype=dtype, device="cpu") / (
                interior_count + 1)
    else:
        interior = torch.empty(0, dtype=dtype, device="cpu")
    knots = torch.cat((
        torch.zeros(degree + 1, dtype=dtype, device="cpu"), interior,
        torch.ones(degree + 1, dtype=dtype, device="cpu")))
    depths = (torch.linspace(0, 1, points, dtype=dtype, device="cpu")
              if points > 1 else torch.zeros(1, dtype=dtype, device="cpu"))
    rows = []
    for index, depth in enumerate(depths):
        basis = ((depth >= knots[:-1]) & (depth < knots[1:])).to(dtype)
        if index + 1 == points:
            basis.zero_()
            basis[controls - 1] = 1
        for order in range(1, degree + 1):
            count = basis.numel() - 1
            left_den = knots[order:order + count] - knots[:count]
            right_den = (
                knots[order + 1:order + 1 + count] - knots[1:1 + count])
            left = torch.where(
                left_den > 0,
                (depth - knots[:count]) / left_den.clamp_min(1e-30),
                torch.zeros_like(left_den))
            right = torch.where(
                right_den > 0,
                (knots[order + 1:order + 1 + count] - depth)
                / right_den.clamp_min(1e-30),
                torch.zeros_like(right_den))
            basis = left * basis[:count] + right * basis[1:1 + count]
        rows.append(basis)
    design = torch.stack(rows)
    if design.shape != (points, controls):
        raise AssertionError("cubic design shape mismatch")
    if not torch.allclose(
            design.sum(1), torch.ones(
                points, dtype=dtype, device="cpu"), atol=1e-6):
        raise AssertionError("cubic design partition of unity failed")
    return design


def periodic_cubic_design(points: int, controls: int,
                          dtype=torch.float32) -> torch.Tensor:
    """Uniform cubic B-spline controls sampled around a periodic circle."""
    if points < 1:
        raise ValueError("periodic design needs at least one sample")
    if controls < 4:
        raise ValueError("periodic cubic design needs at least four controls")
    design = torch.zeros(points, controls, dtype=dtype, device="cpu")
    for point in range(points):
        coordinate = point * controls / points
        base = math.floor(coordinate)
        fraction = coordinate - base
        weights = (
            (1.0 - fraction) ** 3 / 6.0,
            (3.0 * fraction ** 3 - 6.0 * fraction ** 2 + 4.0) / 6.0,
            (-3.0 * fraction ** 3 + 3.0 * fraction ** 2
             + 3.0 * fraction + 1.0) / 6.0,
            fraction ** 3 / 6.0,
        )
        for offset, weight in zip((-1, 0, 1, 2), weights):
            design[point, (base + offset) % controls] += weight
    if not torch.allclose(
            design.sum(1),
            torch.ones(points, dtype=dtype, device="cpu"), atol=1e-6):
        raise AssertionError("periodic cubic design partition of unity failed")
    return design


class SmoothFunctionalAffine(nn.Module):
    """Dense private map plus a spline-generated low-rank residual family."""

    def __init__(self, input_size: int, output_size: int, rank: int,
                 controls: int, virtual_depth: int):
        super().__init__()
        if min(input_size, output_size, rank) <= 0:
            raise ValueError("functional affine dimensions must be positive")
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.controls = int(controls)
        self.virtual_depth = int(virtual_depth)
        self.base = nn.Linear(input_size, output_size)
        self.control_left = nn.Parameter(
            torch.empty(controls, input_size, rank))
        self.control_right = nn.Parameter(
            torch.empty(controls, rank, output_size))
        self.register_buffer(
            "design", clamped_cubic_design(virtual_depth, controls),
            persistent=True)
        nn.init.normal_(self.control_left, std=input_size ** -0.5)
        nn.init.normal_(self.control_right, std=1e-3)

    def forward(self, values: torch.Tensor, phase: int) -> torch.Tensor:
        coefficients = self.design[int(phase) % self.virtual_depth].to(
            device=values.device, dtype=values.dtype)
        left_matrix = self.control_left.permute(1, 0, 2).reshape(
            self.input_size, -1)
        coordinates = torch.matmul(values, left_matrix)
        coordinates = coordinates.view(
            values.shape[0], self.controls, self.control_left.shape[-1])
        weighted_coordinates = coordinates * coefficients.view(1, -1, 1)
        residual = torch.matmul(
            weighted_coordinates.flatten(1), self.control_right.flatten(0, 1))
        return self.base(values) + residual

    def estimated_macs(self) -> int:
        rank = self.control_left.shape[-1]
        return int(
            self.input_size * self.output_size
            + self.controls * rank * (self.input_size + self.output_size)
            + self.controls * self.output_size)


class SmoothFunctionalResidual(nn.Module):
    """Spline-controlled low-rank residual without a private dense map."""

    def __init__(self, width: int, rank: int, controls: int,
                 virtual_depth: int, right_init_std: float = 1e-3,
                 basis: str = "open_uniform_cubic"):
        super().__init__()
        if min(width, rank) <= 0:
            raise ValueError("functional residual dimensions must be positive")
        self.width = int(width)
        self.controls = int(controls)
        self.virtual_depth = int(virtual_depth)
        self.control_left = nn.Parameter(
            torch.empty(controls, width, rank))
        self.control_right = nn.Parameter(
            torch.empty(controls, rank, width))
        designs = {
            "open_uniform_cubic": clamped_cubic_design,
            "periodic_cubic": periodic_cubic_design,
        }
        if basis not in designs:
            raise ValueError(f"unknown functional residual basis: {basis}")
        self.register_buffer(
            "design", designs[basis](virtual_depth, controls), persistent=True)
        nn.init.normal_(self.control_left, std=width ** -0.5)
        nn.init.normal_(self.control_right, std=float(right_init_std))

    def forward(self, values: torch.Tensor, phase: int) -> torch.Tensor:
        coefficients = self.design[int(phase) % self.virtual_depth].to(
            device=values.device, dtype=values.dtype)
        left_matrix = self.control_left.permute(1, 0, 2).reshape(
            self.width, -1)
        coordinates = torch.matmul(values, left_matrix).view(
            values.shape[0], self.controls, self.control_left.shape[-1])
        weighted_coordinates = coordinates * coefficients.view(1, -1, 1)
        return torch.matmul(
            weighted_coordinates.flatten(1), self.control_right.flatten(0, 1))

    def estimated_macs(self) -> int:
        rank = self.control_left.shape[-1]
        return int(
            self.controls * rank * (2 * self.width)
            + self.controls * self.width)


class GeometryRoutedInnovationResidual(nn.Module):
    """A bounded low-rank residual routed by local state-space geometry."""

    def __init__(self, width: int, rank: int, charts: int,
                 correction_scale: float, right_init_std: float = 0.0,
                 gate_bias: float = 0.0):
        super().__init__()
        if min(width, rank, charts) <= 0:
            raise ValueError("innovation residual dimensions must be positive")
        if correction_scale <= 0:
            raise ValueError("innovation residual scale must be positive")
        self.width = int(width)
        self.rank = int(rank)
        self.charts = int(charts)
        self.correction_scale = float(correction_scale)
        self.norm = nn.LayerNorm(width)
        self.router = nn.Linear(width, charts)
        self.control_left = nn.Parameter(torch.empty(charts, width, rank))
        self.control_right = nn.Parameter(torch.empty(charts, rank, width))
        self.gate = nn.Parameter(torch.tensor(float(gate_bias)))
        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)
        nn.init.normal_(self.control_left, std=width ** -0.5)
        nn.init.normal_(self.control_right, std=float(right_init_std))

    def forward(self, proposal: torch.Tensor,
                previous: torch.Tensor) -> torch.Tensor:
        innovation = self.norm(proposal - previous)
        routing = torch.softmax(self.router(innovation).float(), dim=-1).to(
            innovation.dtype)
        left_matrix = self.control_left.permute(1, 0, 2).reshape(
            self.width, self.charts * self.rank)
        coordinates = torch.matmul(innovation, left_matrix).view(
            innovation.shape[0], self.charts, self.rank)
        weighted = coordinates * routing.unsqueeze(-1)
        residual = torch.matmul(
            weighted.flatten(1), self.control_right.flatten(0, 1))
        amplitude = self.correction_scale * torch.sigmoid(self.gate)
        return proposal + amplitude * torch.tanh(residual)

    def estimated_macs(self) -> int:
        factors = 2 * self.charts * self.rank * self.width
        routing = self.width * self.charts + self.charts * self.rank
        normalization_and_bounding = 6 * self.width
        return int(factors + routing + normalization_and_bounding)


class HierarchicalResidualFunctionWeightsStudent(HierarchicalDistributionStudent):
    """Winning dense hierarchy plus a small phase-dependent functional correction."""

    kind = "hierarchical_residual_function_weights"

    def __init__(self, vocab_size: int, width: int, max_context: int,
                 segment_length: int, document_segments: int,
                 function_weights: dict):
        super().__init__(
            vocab_size, width, max_context, segment_length, document_segments)
        self.residual_norm = nn.LayerNorm(width)
        self.residual_function = SmoothFunctionalResidual(
            width=width,
            rank=int(function_weights["rank"]),
            controls=int(function_weights["controls"]),
            virtual_depth=int(function_weights["virtual_depth"]),
        )
        self.residual_gate = nn.Parameter(torch.tensor(
            float(function_weights["gate_bias"])))

    def transition(self, token_ids: torch.Tensor, state=None):
        position = 0 if state is None else int(state.get("position", 0))
        hidden, new_state = super().transition(token_ids, state)
        correction = self.residual_function(
            self.residual_norm(hidden), position)
        hidden = hidden + torch.sigmoid(self.residual_gate) * correction
        new_state["position"] = position + 1
        return hidden, new_state

    def estimated_macs_per_token(self, context: int = 512) -> int:
        return int(
            super().estimated_macs_per_token(context)
            + self.residual_function.estimated_macs()
            + 2 * self.width)


class HierarchicalStateResidualFunctionWeightsStudent(
        HierarchicalDistributionStudent):
    """Dense hierarchy with a shared functional residual in recurrent state."""

    kind = "hierarchical_state_residual_function_weights"

    def __init__(self, vocab_size: int, width: int, max_context: int,
                 segment_length: int, document_segments: int,
                 function_weights: dict):
        super().__init__(
            vocab_size, width, max_context, segment_length, document_segments)
        self.virtual_depth = int(function_weights["virtual_depth"])
        self.state_norms = nn.ModuleList([
            nn.LayerNorm(width), nn.LayerNorm(width), nn.LayerNorm(width)])
        self.state_function = SmoothFunctionalResidual(
            width=width,
            rank=int(function_weights["rank"]),
            controls=int(function_weights["controls"]),
            virtual_depth=self.virtual_depth,
            right_init_std=float(function_weights.get("right_init_std", 0.0)),
            basis=str(function_weights.get("basis", "open_uniform_cubic")),
        )
        gate_bias = float(function_weights.get("gate_bias", 0.0))
        self.state_gates = nn.Parameter(torch.full((3,), gate_bias))

    def initial_state(self, batch_size: int, device: torch.device,
                      dtype: torch.dtype | None = None):
        state = super().initial_state(batch_size, device, dtype)
        state.update({
            "position": 0,
            "segment_updates": 0,
            "document_updates": 0,
        })
        return state

    def _correct_state(self, value: torch.Tensor, scale: int,
                       phase: int) -> torch.Tensor:
        correction = self.state_function(
            self.state_norms[scale](value), phase)
        return value + torch.sigmoid(self.state_gates[scale]) * correction

    def transition(self, token_ids: torch.Tensor, state=None):
        if state is None:
            state = self.initial_state(token_ids.shape[0], token_ids.device)
        position = int(state["position"])
        segment_updates = int(state["segment_updates"])
        document_updates = int(state["document_updates"])

        token = self.embedding(token_ids)
        fast_input = torch.cat(
            (token, state["segment"], state["document"]), dim=-1)
        fast = self.fast_cell(fast_input, state["fast"])
        fast = self._correct_state(fast, 0, position)

        segment_sum = state["segment_sum"] + fast
        segment_tokens = int(state["segment_tokens"]) + 1
        segment = state["segment"]
        document = state["document"]
        segments = int(state["segments"])
        if segment_tokens == self.segment_length:
            segment = self.segment_cell(segment_sum / segment_tokens, segment)
            segment = self._correct_state(segment, 1, segment_updates)
            segment_updates += 1
            segment_sum = torch.zeros_like(segment_sum)
            segment_tokens = 0
            segments += 1
            if segments == self.document_segments:
                document = self.document_cell(segment, document)
                document = self._correct_state(
                    document, 2, document_updates)
                document_updates += 1
                segments = 0

        combined = self.combine(torch.cat((fast, segment, document), dim=-1))
        new_state = {
            "fast": fast,
            "segment": segment,
            "document": document,
            "segment_sum": segment_sum,
            "segment_tokens": segment_tokens,
            "segments": segments,
            "position": position + 1,
            "segment_updates": segment_updates,
            "document_updates": document_updates,
        }
        return combined, new_state

    def estimated_macs_per_token(self, context: int = 512) -> int:
        del context
        update_rate = (
            1.0
            + 1.0 / self.segment_length
            + 1.0 / (self.segment_length * self.document_segments))
        return int(
            super().estimated_macs_per_token(512)
            + update_rate * (
                self.state_function.estimated_macs() + 2 * self.width))


class HierarchicalInnovationResidualStudent(HierarchicalDistributionStudent):
    """Hierarchy with scale-private residual charts over recurrent innovations."""

    kind = "hierarchical_innovation_residual"

    def __init__(self, vocab_size: int, width: int, max_context: int,
                 segment_length: int, document_segments: int,
                 innovation_residual: dict):
        super().__init__(
            vocab_size, width, max_context, segment_length, document_segments)
        arguments = {
            "width": width,
            "rank": int(innovation_residual["rank"]),
            "charts": int(innovation_residual["charts"]),
            "correction_scale": float(
                innovation_residual["correction_scale"]),
            "right_init_std": float(
                innovation_residual.get("right_init_std", 0.0)),
            "gate_bias": float(innovation_residual.get("gate_bias", 0.0)),
        }
        self.innovation_residuals = nn.ModuleList([
            GeometryRoutedInnovationResidual(**arguments) for _ in range(3)
        ])

    def transition(self, token_ids: torch.Tensor, state=None):
        if state is None:
            state = self.initial_state(token_ids.shape[0], token_ids.device)

        token = self.embedding(token_ids)
        fast_input = torch.cat(
            (token, state["segment"], state["document"]), dim=-1)
        fast_proposal = self.fast_cell(fast_input, state["fast"])
        fast = self.innovation_residuals[0](fast_proposal, state["fast"])

        segment_sum = state["segment_sum"] + fast
        segment_tokens = int(state["segment_tokens"]) + 1
        segment = state["segment"]
        document = state["document"]
        segments = int(state["segments"])
        if segment_tokens == self.segment_length:
            segment_proposal = self.segment_cell(
                segment_sum / segment_tokens, segment)
            segment = self.innovation_residuals[1](
                segment_proposal, segment)
            segment_sum = torch.zeros_like(segment_sum)
            segment_tokens = 0
            segments += 1
            if segments == self.document_segments:
                document_proposal = self.document_cell(segment, document)
                document = self.innovation_residuals[2](
                    document_proposal, document)
                segments = 0

        combined = self.combine(torch.cat((fast, segment, document), dim=-1))
        new_state = {
            "fast": fast,
            "segment": segment,
            "document": document,
            "segment_sum": segment_sum,
            "segment_tokens": segment_tokens,
            "segments": segments,
        }
        return combined, new_state

    def estimated_macs_per_token(self, context: int = 512) -> int:
        del context
        fast = self.innovation_residuals[0].estimated_macs()
        segment = (
            self.innovation_residuals[1].estimated_macs()
            / self.segment_length)
        document = (
            self.innovation_residuals[2].estimated_macs()
            / (self.segment_length * self.document_segments))
        return int(
            super().estimated_macs_per_token(512) + fast + segment + document)


class AdaptiveFunctionMemory(nn.Module):
    """Shared chart library for stable, content-adaptive slow memories.

    Each memory is a causal exponential update whose rate is bounded by a
    configured horizon interval.  A single low-rank chart library supplies a
    bounded proposal correction at every scale; only normalization, routing
    bias, and write-rate heads are scale private.
    """

    def __init__(self, width: int, scales: list[dict], rank: int, charts: int,
                 correction_scale: float, right_init_std: float = 0.0):
        super().__init__()
        if min(width, rank, charts) <= 0:
            raise ValueError("adaptive function-memory dimensions must be positive")
        if correction_scale <= 0:
            raise ValueError("adaptive function-memory scale must be positive")
        if not scales:
            raise ValueError("adaptive function memory needs at least one slow scale")
        self.width = int(width)
        self.rank = int(rank)
        self.charts = int(charts)
        self.correction_scale = float(correction_scale)
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in scales])
        self.rate_heads = nn.ModuleList([nn.Linear(width, 1) for _ in scales])
        self.router = nn.Linear(width, charts)
        self.scale_routing_bias = nn.Parameter(torch.zeros(len(scales), charts))
        self.control_left = nn.Parameter(torch.empty(charts, width, rank))
        self.control_right = nn.Parameter(torch.empty(charts, rank, width))

        minimum_rates = []
        maximum_rates = []
        for index, spec in enumerate(scales):
            base_horizon = float(spec["base_horizon"])
            minimum_horizon = float(spec["minimum_horizon"])
            maximum_horizon = float(spec["maximum_horizon"])
            if not 0 < minimum_horizon <= base_horizon <= maximum_horizon:
                raise ValueError(
                    "adaptive memory horizons must satisfy "
                    "0 < minimum <= base <= maximum")
            minimum_rate = 1.0 / maximum_horizon
            maximum_rate = 1.0 / minimum_horizon
            base_rate = 1.0 / base_horizon
            normalized_rate = (
                (base_rate - minimum_rate) / (maximum_rate - minimum_rate))
            normalized_rate = min(max(normalized_rate, 1e-6), 1.0 - 1e-6)
            nn.init.zeros_(self.rate_heads[index].weight)
            nn.init.constant_(
                self.rate_heads[index].bias,
                math.log(normalized_rate / (1.0 - normalized_rate)))
            minimum_rates.append(minimum_rate)
            maximum_rates.append(maximum_rate)

        self.register_buffer(
            "minimum_rates", torch.tensor(minimum_rates, dtype=torch.float32),
            persistent=True)
        self.register_buffer(
            "maximum_rates", torch.tensor(maximum_rates, dtype=torch.float32),
            persistent=True)
        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)
        nn.init.normal_(self.control_left, std=width ** -0.5)
        nn.init.normal_(self.control_right, std=float(right_init_std))

    def forward(self, source: torch.Tensor, previous: torch.Tensor,
                scale: int) -> tuple[torch.Tensor, torch.Tensor]:
        scale = int(scale)
        innovation = self.norms[scale](source - previous)
        routing_logits = (
            self.router(innovation) + self.scale_routing_bias[scale])
        routing = torch.softmax(routing_logits.float(), dim=-1).to(
            innovation.dtype)
        left_matrix = self.control_left.permute(1, 0, 2).reshape(
            self.width, self.charts * self.rank)
        coordinates = torch.matmul(innovation, left_matrix).view(
            innovation.shape[0], self.charts, self.rank)
        weighted = coordinates * routing.unsqueeze(-1)
        correction = torch.matmul(
            weighted.flatten(1), self.control_right.flatten(0, 1))
        proposal = source + self.correction_scale * torch.tanh(correction)

        normalized_rate = torch.sigmoid(self.rate_heads[scale](innovation))
        minimum_rate = self.minimum_rates[scale].to(
            device=source.device, dtype=source.dtype)
        maximum_rate = self.maximum_rates[scale].to(
            device=source.device, dtype=source.dtype)
        rate = minimum_rate + (maximum_rate - minimum_rate) * normalized_rate
        updated = previous + rate * (proposal - previous)
        return updated, rate

    def estimated_macs(self, scale_count: int) -> int:
        # Per scale: normalization, routing and rate heads, shared low-rank
        # factor application, proposal bounding, and the stable interpolation.
        per_scale = (
            2 * self.width
            + self.width * self.charts
            + self.width
            + 2 * self.charts * self.rank * self.width
            + self.charts * self.rank
            + 5 * self.width)
        return int(scale_count * per_scale)


class HierarchicalAdaptiveFunctionMemoryStudent(StudentBase):
    """Boundary-free hierarchy with a shared functional slow-memory library.

    The fast GRU remains the token-level communication engine.  Unlike the
    hard-boundary hierarchy, every token can update the two slower memories,
    with stable write rates centered on configured horizons.  This removes the
    active segment accumulator and the two expensive private slow GRUs.
    """

    kind = "hierarchical_adaptive_function_memory"

    def __init__(self, vocab_size: int, width: int, max_context: int,
                 memory_scales: list[dict], function_memory: dict):
        super().__init__(vocab_size, width, max_context)
        if len(memory_scales) != 2:
            raise ValueError(
                "adaptive hierarchy currently requires exactly two slow scales")
        self.memory_scales = tuple(dict(value) for value in memory_scales)
        self.fast_cell = nn.GRUCell(3 * width, width)
        self.function_memory = AdaptiveFunctionMemory(
            width=width,
            scales=list(self.memory_scales),
            rank=int(function_memory["rank"]),
            charts=int(function_memory["charts"]),
            correction_scale=float(function_memory["correction_scale"]),
            right_init_std=float(function_memory.get("right_init_std", 0.0)),
        )
        self.combine = nn.Sequential(
            nn.Linear(3 * width, 2 * width), nn.SiLU(),
            nn.Linear(2 * width, width))

    def initial_state(self, batch_size: int, device: torch.device,
                      dtype: torch.dtype | None = None):
        dtype = dtype or self.embedding.weight.dtype
        zero = lambda: torch.zeros(
            batch_size, self.width, device=device, dtype=dtype)
        return {"fast": zero(), "segment": zero(), "document": zero()}

    def transition(self, token_ids: torch.Tensor, state=None):
        if state is None:
            state = self.initial_state(token_ids.shape[0], token_ids.device)
        token = self.embedding(token_ids)
        fast_input = torch.cat(
            (token, state["segment"], state["document"]), dim=-1)
        fast = self.fast_cell(fast_input, state["fast"])
        segment, segment_rate = self.function_memory(
            fast, state["segment"], 0)
        document, document_rate = self.function_memory(
            segment, state["document"], 1)
        hidden = self.combine(torch.cat((fast, segment, document), dim=-1))
        new_state = {
            "fast": fast,
            "segment": segment,
            "document": document,
        }
        self._last_write_rates = (segment_rate, document_rate)
        return hidden, new_state

    def step(self, token_ids: torch.Tensor, state=None):
        hidden, new_state = self.transition(token_ids, state)
        return self.logits(hidden), new_state, hidden

    def forward(self, input_ids: torch.Tensor):
        state = None
        hidden_values = []
        for index in range(input_ids.shape[1]):
            hidden, state = self.transition(input_ids[:, index], state)
            hidden_values.append(hidden)
        if not hidden_values:
            raise ValueError("forward requires at least one input token")
        stacked_hidden = torch.stack(hidden_values, dim=1)
        return SimpleNamespace(
            logits=self.logits(stacked_hidden),
            hidden_states=stacked_hidden,
            final_state=state)

    def last_logits(self, input_ids: torch.Tensor):
        state = None
        hidden = None
        for index in range(input_ids.shape[1]):
            hidden, state = self.transition(input_ids[:, index], state)
        if hidden is None:
            raise ValueError("last_logits requires at least one input token")
        return self.logits(hidden).unsqueeze(1)

    def stored_state_elements(self, batch_size: int = 1,
                              context: int = 512) -> int:
        del context
        return 3 * batch_size * self.width

    def estimated_macs_per_token(self, context: int = 512) -> int:
        del context
        fast = 12 * self.width * self.width
        combine = 8 * self.width * self.width
        adaptive_memory = self.function_memory.estimated_macs(
            len(self.memory_scales))
        return int(
            self.output_head_macs_per_token()
            + fast + combine + adaptive_memory)


class SmoothFunctionGRUCell(nn.Module):
    """GRU whose phase-specific maps come from smooth functional bases."""

    def __init__(self, input_size: int, hidden_size: int, rank: int,
                 controls: int, virtual_depth: int):
        super().__init__()
        self.input_function = SmoothFunctionalAffine(
            input_size, 3 * hidden_size, rank, controls, virtual_depth)
        self.hidden_function = SmoothFunctionalAffine(
            hidden_size, 3 * hidden_size, rank, controls, virtual_depth)

    def forward(self, values: torch.Tensor, previous: torch.Tensor,
                phase: int) -> torch.Tensor:
        input_gates = self.input_function(values, phase)
        hidden_gates = self.hidden_function(previous, phase)
        input_reset, input_update, input_new = input_gates.chunk(3, dim=-1)
        hidden_reset, hidden_update, hidden_new = hidden_gates.chunk(3, dim=-1)
        reset = torch.sigmoid(input_reset + hidden_reset)
        update = torch.sigmoid(input_update + hidden_update)
        candidate = torch.tanh(input_new + reset * hidden_new)
        return candidate + update * (previous - candidate)

    def estimated_macs(self) -> int:
        return (self.input_function.estimated_macs()
                + self.hidden_function.estimated_macs())


class HierarchicalFunctionWeightsStudent(StudentBase):
    """Three-timescale recurrent memory with functional transitions only."""

    kind = "hierarchical_function_weights"

    def __init__(self, vocab_size: int, width: int, max_context: int,
                 segment_length: int, document_segments: int,
                 function_weights: dict):
        super().__init__(vocab_size, width, max_context)
        self.segment_length = int(segment_length)
        self.document_segments = int(document_segments)
        self.virtual_depth = int(function_weights["virtual_depth"])
        arguments = {
            "hidden_size": width,
            "rank": int(function_weights["rank"]),
            "controls": int(function_weights["controls"]),
            "virtual_depth": self.virtual_depth,
        }
        self.fast_cell = SmoothFunctionGRUCell(
            input_size=3 * width, **arguments)
        self.segment_cell = SmoothFunctionGRUCell(
            input_size=width, **arguments)
        self.document_cell = SmoothFunctionGRUCell(
            input_size=width, **arguments)
        self.cross_scale = SmoothFunctionalAffine(
            3 * width, 2 * width, int(function_weights["rank"]),
            int(function_weights["controls"]), self.virtual_depth)
        self.readout = SmoothFunctionalAffine(
            2 * width, width, int(function_weights["rank"]),
            int(function_weights["controls"]), self.virtual_depth)

    def initial_state(self, batch_size: int, device: torch.device,
                      dtype: torch.dtype | None = None):
        dtype = dtype or self.embedding.weight.dtype
        zero = lambda: torch.zeros(batch_size, self.width, device=device, dtype=dtype)
        return {
            "fast": zero(), "segment": zero(), "document": zero(),
            "segment_sum": zero(), "segment_tokens": 0, "segments": 0,
            "position": 0, "segment_updates": 0, "document_updates": 0,
        }

    def transition(self, token_ids: torch.Tensor, state=None):
        if state is None:
            state = self.initial_state(token_ids.shape[0], token_ids.device)
        position = int(state["position"])
        token = self.embedding(token_ids)
        fast_input = torch.cat(
            (token, state["segment"], state["document"]), dim=-1)
        fast = self.fast_cell(fast_input, state["fast"], position)
        segment_sum = state["segment_sum"] + fast
        segment_tokens = int(state["segment_tokens"]) + 1
        segment = state["segment"]
        document = state["document"]
        segments = int(state["segments"])
        segment_updates = int(state["segment_updates"])
        document_updates = int(state["document_updates"])
        if segment_tokens == self.segment_length:
            segment = self.segment_cell(
                segment_sum / segment_tokens, segment, segment_updates)
            segment_sum = torch.zeros_like(segment_sum)
            segment_tokens = 0
            segment_updates += 1
            segments += 1
            if segments == self.document_segments:
                document = self.document_cell(
                    segment, document, document_updates)
                document_updates += 1
                segments = 0
        combined = torch.cat((fast, segment, document), dim=-1)
        hidden = self.readout(
            F.silu(self.cross_scale(combined, position)), position)
        new_state = {
            "fast": fast, "segment": segment, "document": document,
            "segment_sum": segment_sum, "segment_tokens": segment_tokens,
            "segments": segments, "position": position + 1,
            "segment_updates": segment_updates,
            "document_updates": document_updates,
        }
        return hidden, new_state

    def step(self, token_ids: torch.Tensor, state=None):
        hidden, new_state = self.transition(token_ids, state)
        return self.logits(hidden), new_state, hidden

    def forward(self, input_ids: torch.Tensor):
        state = None
        hidden_values = []
        for index in range(input_ids.shape[1]):
            hidden, state = self.transition(input_ids[:, index], state)
            hidden_values.append(hidden)
        if not hidden_values:
            raise ValueError("forward requires at least one input token")
        stacked_hidden = torch.stack(hidden_values, dim=1)
        return SimpleNamespace(
            logits=self.logits(stacked_hidden),
            hidden_states=stacked_hidden,
            final_state=state)

    def last_logits(self, input_ids: torch.Tensor):
        state = None
        hidden = None
        for index in range(input_ids.shape[1]):
            hidden, state = self.transition(input_ids[:, index], state)
        if hidden is None:
            raise ValueError("last_logits requires at least one input token")
        return self.logits(hidden).unsqueeze(1)

    def stored_state_elements(self, batch_size: int = 1, context: int = 512) -> int:
        del context
        return 4 * batch_size * self.width

    def estimated_macs_per_token(self, context: int = 512) -> int:
        del context
        transitions = (
            self.fast_cell.estimated_macs()
            + self.segment_cell.estimated_macs() / self.segment_length
            + self.document_cell.estimated_macs()
            / (self.segment_length * self.document_segments))
        return int(
            self.output_head_macs_per_token()
            + transitions
            + self.cross_scale.estimated_macs()
            + self.readout.estimated_macs())


class MultiTimescaleHierarchicalFunctionWeightsStudent(StudentBase):
    """Functional recurrent memory with configurable fixed update horizons."""

    kind = "hierarchical_multiscale_function_weights"

    def __init__(self, vocab_size: int, width: int, max_context: int,
                 timescales: list[int], function_weights: dict):
        super().__init__(vocab_size, width, max_context)
        self.timescales = tuple(int(value) for value in timescales)
        if not self.timescales:
            raise ValueError("multi-timescale hierarchy needs at least one timescale")
        if any(value <= 0 for value in self.timescales):
            raise ValueError("multi-timescale horizons must be positive")
        if tuple(sorted(set(self.timescales))) != self.timescales:
            raise ValueError("multi-timescale horizons must be unique and increasing")
        self.virtual_depth = int(function_weights["virtual_depth"])
        arguments = {
            "hidden_size": width,
            "rank": int(function_weights["rank"]),
            "controls": int(function_weights["controls"]),
            "virtual_depth": self.virtual_depth,
        }
        scale_count = len(self.timescales)
        self.fast_cell = SmoothFunctionGRUCell(
            input_size=(1 + scale_count) * width, **arguments)
        self.scale_cells = nn.ModuleList([
            SmoothFunctionGRUCell(input_size=width, **arguments)
            for _ in self.timescales
        ])
        self.cross_scale = SmoothFunctionalAffine(
            (1 + scale_count) * width, 2 * width,
            int(function_weights["rank"]), int(function_weights["controls"]),
            self.virtual_depth)
        self.readout = SmoothFunctionalAffine(
            2 * width, width, int(function_weights["rank"]),
            int(function_weights["controls"]), self.virtual_depth)

    def initial_state(self, batch_size: int, device: torch.device,
                      dtype: torch.dtype | None = None):
        dtype = dtype or self.embedding.weight.dtype
        zero = lambda: torch.zeros(batch_size, self.width, device=device, dtype=dtype)
        count = len(self.timescales)
        return {
            "fast": zero(),
            "scale_states": [zero() for _ in range(count)],
            "scale_sums": [zero() for _ in range(count)],
            "scale_tokens": [0 for _ in range(count)],
            "scale_updates": [0 for _ in range(count)],
            "position": 0,
        }

    def transition(self, token_ids: torch.Tensor, state=None):
        if state is None:
            state = self.initial_state(token_ids.shape[0], token_ids.device)
        position = int(state["position"])
        token = self.embedding(token_ids)
        fast_input = torch.cat((token, *state["scale_states"]), dim=-1)
        fast = self.fast_cell(fast_input, state["fast"], position)
        scale_states = []
        scale_sums = []
        scale_tokens = []
        scale_updates = []
        for index, (horizon, cell) in enumerate(
                zip(self.timescales, self.scale_cells)):
            accumulated = state["scale_sums"][index] + fast
            tokens = int(state["scale_tokens"][index]) + 1
            updates = int(state["scale_updates"][index])
            value = state["scale_states"][index]
            if tokens == horizon:
                value = cell(accumulated / tokens, value, updates)
                accumulated = torch.zeros_like(accumulated)
                tokens = 0
                updates += 1
            scale_states.append(value)
            scale_sums.append(accumulated)
            scale_tokens.append(tokens)
            scale_updates.append(updates)
        combined = torch.cat((fast, *scale_states), dim=-1)
        hidden = self.readout(
            F.silu(self.cross_scale(combined, position)), position)
        new_state = {
            "fast": fast, "scale_states": scale_states,
            "scale_sums": scale_sums, "scale_tokens": scale_tokens,
            "scale_updates": scale_updates, "position": position + 1,
        }
        return hidden, new_state

    def step(self, token_ids: torch.Tensor, state=None):
        hidden, new_state = self.transition(token_ids, state)
        return self.logits(hidden), new_state, hidden

    def forward(self, input_ids: torch.Tensor):
        state = None
        hidden_values = []
        for index in range(input_ids.shape[1]):
            hidden, state = self.transition(input_ids[:, index], state)
            hidden_values.append(hidden)
        if not hidden_values:
            raise ValueError("forward requires at least one input token")
        stacked_hidden = torch.stack(hidden_values, dim=1)
        return SimpleNamespace(
            logits=self.logits(stacked_hidden),
            hidden_states=stacked_hidden,
            final_state=state)

    def last_logits(self, input_ids: torch.Tensor):
        state = None
        hidden = None
        for index in range(input_ids.shape[1]):
            hidden, state = self.transition(input_ids[:, index], state)
        if hidden is None:
            raise ValueError("last_logits requires at least one input token")
        return self.logits(hidden).unsqueeze(1)

    def stored_state_elements(self, batch_size: int = 1, context: int = 512) -> int:
        del context
        # Fast state plus one state and one accumulator for every horizon.
        return (1 + 2 * len(self.timescales)) * batch_size * self.width

    def estimated_macs_per_token(self, context: int = 512) -> int:
        del context
        transitions = self.fast_cell.estimated_macs()
        transitions += sum(
            cell.estimated_macs() / horizon
            for horizon, cell in zip(self.timescales, self.scale_cells))
        return int(
            self.output_head_macs_per_token()
            + transitions
            + self.cross_scale.estimated_macs()
            + self.readout.estimated_macs())


class HierarchicalLocalFunctionWeightsStudent(StudentBase):
    """Local hierarchical memory with smooth function-generated transitions."""

    kind = "hierarchical_local_function_weights"

    def __init__(self, vocab_size: int, width: int, max_context: int,
                 segment_length: int, document_segments: int,
                 local_attention: dict, function_weights: dict):
        super().__init__(vocab_size, width, max_context)
        self.segment_length = int(segment_length)
        self.document_segments = int(document_segments)
        self.virtual_depth = int(function_weights["virtual_depth"])
        arguments = {
            "hidden_size": width,
            "rank": int(function_weights["rank"]),
            "controls": int(function_weights["controls"]),
            "virtual_depth": self.virtual_depth,
        }
        self.fast_cell = SmoothFunctionGRUCell(
            input_size=3 * width, **arguments)
        self.segment_cell = SmoothFunctionGRUCell(
            input_size=width, **arguments)
        self.document_cell = SmoothFunctionGRUCell(
            input_size=width, **arguments)
        self.cross_scale = SmoothFunctionalAffine(
            3 * width, 2 * width, int(function_weights["rank"]),
            int(function_weights["controls"]), self.virtual_depth)
        self.readout = SmoothFunctionalAffine(
            2 * width, width, int(function_weights["rank"]),
            int(function_weights["controls"]), self.virtual_depth)
        self.local_attention = CompressedLocalAttention(
            width=width,
            local_width=int(local_attention["width"]),
            heads=int(local_attention["heads"]),
            window=int(local_attention["window"]),
            gate_bias=float(local_attention["gate_bias"]),
        )

    def initial_state(self, batch_size: int, device: torch.device,
                      dtype: torch.dtype | None = None):
        dtype = dtype or self.embedding.weight.dtype
        zero = lambda: torch.zeros(batch_size, self.width, device=device, dtype=dtype)
        return {
            "fast": zero(), "segment": zero(), "document": zero(),
            "segment_sum": zero(), "segment_tokens": 0, "segments": 0,
            "position": 0, "segment_updates": 0, "document_updates": 0,
            "local_keys": None, "local_values": None,
        }

    def transition(self, token_ids: torch.Tensor, state=None):
        if state is None:
            state = self.initial_state(token_ids.shape[0], token_ids.device)
        position = int(state["position"])
        token = self.embedding(token_ids)
        fast_input = torch.cat(
            (token, state["segment"], state["document"]), dim=-1)
        fast = self.fast_cell(fast_input, state["fast"], position)
        segment_sum = state["segment_sum"] + fast
        segment_tokens = int(state["segment_tokens"]) + 1
        segment = state["segment"]
        document = state["document"]
        segments = int(state["segments"])
        segment_updates = int(state["segment_updates"])
        document_updates = int(state["document_updates"])
        if segment_tokens == self.segment_length:
            segment = self.segment_cell(
                segment_sum / segment_tokens, segment, segment_updates)
            segment_sum = torch.zeros_like(segment_sum)
            segment_tokens = 0
            segment_updates += 1
            segments += 1
            if segments == self.document_segments:
                document = self.document_cell(
                    segment, document, document_updates)
                document_updates += 1
                segments = 0
        combined = torch.cat((fast, segment, document), dim=-1)
        hidden = self.readout(
            F.silu(self.cross_scale(combined, position)), position)
        hidden, local_keys, local_values = self.local_attention(
            hidden, state["local_keys"], state["local_values"])
        new_state = {
            "fast": fast, "segment": segment, "document": document,
            "segment_sum": segment_sum, "segment_tokens": segment_tokens,
            "segments": segments, "position": position + 1,
            "segment_updates": segment_updates,
            "document_updates": document_updates,
            "local_keys": local_keys, "local_values": local_values,
        }
        return hidden, new_state

    def step(self, token_ids: torch.Tensor, state=None):
        hidden, new_state = self.transition(token_ids, state)
        return self.logits(hidden), new_state, hidden

    def forward(self, input_ids: torch.Tensor):
        state = None
        hidden_values = []
        for index in range(input_ids.shape[1]):
            hidden, state = self.transition(input_ids[:, index], state)
            hidden_values.append(hidden)
        if not hidden_values:
            raise ValueError("forward requires at least one input token")
        stacked_hidden = torch.stack(hidden_values, dim=1)
        return SimpleNamespace(
            logits=self.logits(stacked_hidden),
            hidden_states=stacked_hidden,
            final_state=state)

    def last_logits(self, input_ids: torch.Tensor):
        state = None
        hidden = None
        for index in range(input_ids.shape[1]):
            hidden, state = self.transition(input_ids[:, index], state)
        if hidden is None:
            raise ValueError("last_logits requires at least one input token")
        return self.logits(hidden).unsqueeze(1)

    def stored_state_elements(self, batch_size: int = 1, context: int = 512) -> int:
        del context
        return (4 * batch_size * self.width
                + self.local_attention.stored_state_elements(batch_size))

    def estimated_macs_per_token(self, context: int = 512) -> int:
        del context
        transitions = (
            self.fast_cell.estimated_macs()
            + self.segment_cell.estimated_macs() / self.segment_length
            + self.document_cell.estimated_macs()
            / (self.segment_length * self.document_segments))
        return int(
            self.output_head_macs_per_token()
            + transitions
            + self.cross_scale.estimated_macs()
            + self.readout.estimated_macs()
            + self.local_attention.estimated_macs_per_token(self.width))


def build_student(kind: str, vocab_size: int, width: int, max_context: int,
                  config: dict) -> StudentBase:
    if kind == "transformer":
        return TransformerStudent(
            vocab_size, width, max_context, int(config["layers"]),
            int(config["heads"]), int(config.get("expansion", 4)),
            str(config.get("position_encoding", "learned")))
    if kind == "recurrent_logits":
        return RecurrentLogitStudent(
            vocab_size, width, max_context, int(config["layers"]))
    if kind in ("hierarchical_predictive", "hierarchical_logit_control"):
        return HierarchicalPredictiveStudent(
            vocab_size, width, max_context, int(config["atoms"]),
            int(config["segment_length"]), int(config["document_segments"]),
            list(config["future_horizons"]))
    if kind == "hierarchical_distribution":
        return HierarchicalDistributionStudent(
            vocab_size, width, max_context, int(config["segment_length"]),
            int(config["document_segments"]))
    if kind == "hierarchical_residual_function_weights":
        return HierarchicalResidualFunctionWeightsStudent(
            vocab_size, width, max_context, int(config["segment_length"]),
            int(config["document_segments"]),
            dict(config["function_weights"]))
    if kind == "hierarchical_state_residual_function_weights":
        return HierarchicalStateResidualFunctionWeightsStudent(
            vocab_size, width, max_context, int(config["segment_length"]),
            int(config["document_segments"]),
            dict(config["function_weights"]))
    if kind == "hierarchical_innovation_residual":
        return HierarchicalInnovationResidualStudent(
            vocab_size, width, max_context, int(config["segment_length"]),
            int(config["document_segments"]),
            dict(config["innovation_residual"]))
    if kind == "hierarchical_adaptive_function_memory":
        return HierarchicalAdaptiveFunctionMemoryStudent(
            vocab_size, width, max_context,
            list(config["memory_scales"]), dict(config["function_memory"]))
    if kind == "hierarchical_local_distribution":
        return HierarchicalLocalDistributionStudent(
            vocab_size, width, max_context, int(config["segment_length"]),
            int(config["document_segments"]), dict(config["local_attention"]))
    if kind == "hierarchical_local_function_weights":
        return HierarchicalLocalFunctionWeightsStudent(
            vocab_size, width, max_context, int(config["segment_length"]),
            int(config["document_segments"]), dict(config["local_attention"]),
            dict(config["function_weights"]))
    if kind == "hierarchical_function_weights":
        return HierarchicalFunctionWeightsStudent(
            vocab_size, width, max_context, int(config["segment_length"]),
            int(config["document_segments"]), dict(config["function_weights"]))
    if kind == "hierarchical_multiscale_function_weights":
        return MultiTimescaleHierarchicalFunctionWeightsStudent(
            vocab_size, width, max_context, list(config["timescales"]),
            dict(config["function_weights"]))
    raise ValueError(f"unknown student kind {kind}")


@dataclass
class BudgetedStudent:
    model: StudentBase
    width: int
    parameters: int
    target_parameters: int


def build_to_budget(kind: str, vocab_size: int, max_context: int, config: dict,
                    target_parameters: int) -> BudgetedStudent:
    """Choose the widest multiple-of-32 model that does not exceed the budget."""
    widths = list(range(int(config["minimum_width"]),
                        int(config["maximum_width"]) + 1,
                        int(config.get("width_step", 32))))
    if kind == "transformer":
        heads = int(config["heads"])
        widths = [width for width in widths if width % heads == 0]
    chosen = None
    for width in widths:
        with torch.device("meta"):
            candidate = build_student(kind, vocab_size, width, max_context, config)
        count = parameter_count(candidate)
        if count <= target_parameters:
            chosen = (width, count)
        else:
            break
    if chosen is None:
        raise ValueError(f"{kind} cannot fit minimum width under parameter budget")
    width, count = chosen
    model = build_student(kind, vocab_size, width, max_context, config)
    actual = parameter_count(model)
    if actual != count:
        raise RuntimeError("meta-device parameter count disagrees with real model")
    return BudgetedStudent(model, width, actual, int(target_parameters))
