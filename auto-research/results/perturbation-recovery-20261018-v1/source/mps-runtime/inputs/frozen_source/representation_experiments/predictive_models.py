"""Equal-storage students for predictive-state compilation."""
from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from representation_experiments.common import parameter_count
from representation_experiments.nanogpt_control import NanoGPTStudent


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
            # The spline row weights the ``controls * rank`` coordinate
            # tensor before the right projection.  It does not weight one
            # value per output feature.
            + self.controls * rank)


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
    state_schema = "native_full_width_v1"
    persistent_tensor_keys = ("fast", "segment", "document", "segment_sum")
    persistent_control_keys = (
        "segment_tokens", "segments", "position", "segment_updates",
        "document_updates")
    control_metadata_schema = "five_unsigned_64_bit_integers_v1"
    control_metadata_bytes = 5 * 8

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

    def control_metadata_accounting(self, runtime_state=None) -> dict:
        """Audit the five non-tensor clocks separately from tensor payload.

        Python object overhead is deliberately not presented as a portable
        model-state byte count.  The logical serialization contract is five
        unsigned 64-bit integers: 40 bytes for one recurrent stream.  Batched
        training advances every sequence in lockstep and therefore stores one
        shared control record, while an independently resumed sequence needs
        its own 40-byte record.
        """
        result = {
            "schema": self.control_metadata_schema,
            "keys": list(self.persistent_control_keys),
            "logical_encoding": "little_endian_uint64",
            "logical_bytes_per_recurrent_stream": self.control_metadata_bytes,
            "logical_bytes_per_independent_sequence": self.control_metadata_bytes,
            "batched_runtime_uses_one_lockstep_shared_record": True,
            "python_object_overhead_is_implementation_dependent": True,
            "python_object_overhead_included_in_logical_bytes": False,
            "audited": runtime_state is not None,
        }
        if runtime_state is None:
            return result
        tensor_keys = {
            key for key, value in runtime_state.items() if torch.is_tensor(value)
        }
        control_keys = set(runtime_state) - tensor_keys
        expected = set(self.persistent_control_keys)
        if control_keys != expected:
            raise RuntimeError(
                "runtime control metadata keys drifted: "
                f"expected={sorted(expected)} actual={sorted(control_keys)}")
        values = {}
        for key in self.persistent_control_keys:
            value = runtime_state[key]
            if type(value) is not int or not 0 <= value < 2 ** 64:
                raise RuntimeError(
                    f"runtime control metadata {key} must be uint64-compatible")
            values[key] = value
        result["values"] = values
        result["all_values_uint64_compatible"] = True
        return result

    def state_accounting(self, batch_size: int = 1, context: int = 512,
                         runtime_state=None) -> dict:
        del context
        elements = self.stored_state_elements(batch_size)
        accounting = {
            "batch_size": int(batch_size),
            "state_schema": self.state_schema,
            "persistent_tensor_keys": list(self.persistent_tensor_keys),
            "persistent_tensor_shapes": {
                name: [int(batch_size), self.width]
                for name in self.persistent_tensor_keys
            },
            "persistent_elements": elements,
            "nominal_bf16_bytes": elements * 2,
            "nominal_bf16_bytes_per_sequence": 4 * self.width * 2,
            "runtime_dtype": "float32",
            "runtime_bytes": elements * 4,
            "runtime_bytes_per_sequence": 4 * self.width * 4,
            "runtime_state_audited": runtime_state is not None,
        }
        control_metadata = self.control_metadata_accounting(runtime_state)
        accounting.update({
            "tensor_payload_excludes_control_metadata": True,
            "control_metadata": control_metadata,
            "nominal_bf16_tensor_plus_logical_control_bytes_per_sequence": (
                4 * self.width * 2 + self.control_metadata_bytes),
            "fp32_tensor_plus_logical_control_bytes_per_sequence": (
                4 * self.width * 4 + self.control_metadata_bytes),
        })
        if runtime_state is None:
            return accounting
        tensor_state = {
            key: value for key, value in runtime_state.items()
            if torch.is_tensor(value)
        }
        if set(tensor_state) != set(self.persistent_tensor_keys):
            raise RuntimeError("native HFW runtime persistent tensor keys drifted")
        component_runtime = {}
        actual_runtime_bytes = 0
        for key in self.persistent_tensor_keys:
            value = tensor_state[key]
            if tuple(value.shape) != (int(batch_size), self.width):
                raise RuntimeError(
                    f"native HFW runtime state shape mismatch for {key}")
            if value.dtype != torch.float32:
                raise RuntimeError(
                    f"native HFW runtime state {key} must be FP32")
            if not torch.isfinite(value).all():
                raise FloatingPointError(
                    f"native HFW runtime state {key} is nonfinite")
            component_bytes = value.numel() * value.element_size()
            actual_runtime_bytes += component_bytes
            component_runtime[key] = {
                "shape": list(value.shape), "dtype": "float32",
                "bytes": component_bytes, "finite": True,
            }
        if actual_runtime_bytes != accounting["runtime_bytes"]:
            raise RuntimeError("native HFW runtime bytes differ from contract")
        accounting.update({
            "actual_component_runtime": component_runtime,
            "actual_runtime_bytes": actual_runtime_bytes,
        })
        return accounting

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


class TiedBiasFreeStateCodec(nn.Module):
    """Component-private hidden/state projection with a tied transpose decode.

    The sole learned tensor maps hidden values into persistent state. Decoding
    uses its transpose, so there is no decoder parameter and a native all-zero
    state decodes to an exact all-zero hidden value.
    """

    def __init__(self, hidden_width: int, state_width: int):
        super().__init__()
        if min(hidden_width, state_width) <= 0:
            raise ValueError("codec dimensions must be positive")
        if state_width > hidden_width:
            raise ValueError("state_width cannot exceed hidden_width")
        self.hidden_width = int(hidden_width)
        self.state_width = int(state_width)
        self.weight = nn.Parameter(torch.empty(state_width, hidden_width))
        if self.weight.device.type != "meta":
            nn.init.orthogonal_(self.weight)

    def encode(self, hidden: torch.Tensor) -> torch.Tensor:
        # Persistent state is deliberately FP32 even when the surrounding
        # transition executes under CUDA autocast. This makes runtime bytes a
        # tested storage contract instead of an autocast side effect.
        device_type = hidden.device.type
        if device_type in ("cpu", "cuda"):
            with torch.autocast(device_type=device_type, enabled=False):
                return F.linear(hidden.float(), self.weight.float())
        return F.linear(hidden.float(), self.weight.float())

    def decode(self, persistent: torch.Tensor) -> torch.Tensor:
        return F.linear(persistent, self.weight.transpose(0, 1))

    def estimated_macs(self) -> int:
        return self.hidden_width * self.state_width

    def row_orthogonality_error(self) -> torch.Tensor:
        identity = torch.eye(
            self.state_width, device=self.weight.device,
            dtype=self.weight.dtype)
        return (self.weight @ self.weight.transpose(0, 1) - identity).abs().max()


class BottleneckHierarchicalFunctionWeightsStudent(
        HierarchicalFunctionWeightsStudent):
    """Function-weight hierarchy with four compressed persistent components.

    Hidden-width transition values are ephemeral. The returned recurrent state
    contains exactly four tensors of ``state_width`` values per sample; clocks
    and counters are Python integers and therefore are not learned or
    tensor-valued memory. Each component has its own bias-free tied codec.
    """

    kind = "bottleneck_hierarchical_function_weights"
    persistent_tensor_keys = ("fast", "segment", "document", "segment_sum")

    def __init__(self, vocab_size: int, width: int, max_context: int,
                 segment_length: int, document_segments: int,
                 state_width: int, function_weights: dict):
        super().__init__(
            vocab_size, width, max_context, segment_length,
            document_segments, function_weights)
        self.state_width = int(state_width)
        if self.state_width <= 0 or self.state_width > self.width:
            raise ValueError("state_width must be in [1, width]")
        self.identity_codec_mode = self.state_width == self.width
        self.state_schema = (
            "identity_full_width_v1" if self.identity_codec_mode
            else "lossy_tied_component_codec_v1")
        self.state_codecs = nn.ModuleDict(
            {} if self.identity_codec_mode else {
                name: TiedBiasFreeStateCodec(width, self.state_width)
                for name in self.persistent_tensor_keys
            })

    def initial_state(self, batch_size: int, device: torch.device,
                      dtype: torch.dtype | None = None):
        if self.identity_codec_mode:
            return super().initial_state(batch_size, device, dtype)
        zero = lambda: torch.zeros(
            batch_size, self.state_width, device=device, dtype=torch.float32)
        return {
            "fast": zero(), "segment": zero(), "document": zero(),
            "segment_sum": zero(), "segment_tokens": 0, "segments": 0,
            "position": 0, "segment_updates": 0, "document_updates": 0,
        }

    def transition(self, token_ids: torch.Tensor, state=None):
        if self.identity_codec_mode:
            return super().transition(token_ids, state)
        if state is None:
            state = self.initial_state(token_ids.shape[0], token_ids.device)
        position = int(state["position"])
        token = self.embedding(token_ids)

        # These decoded values are local temporaries and are never returned as
        # recurrent state.
        previous_fast = self.state_codecs["fast"].decode(state["fast"])
        previous_segment = self.state_codecs["segment"].decode(state["segment"])
        previous_document = self.state_codecs["document"].decode(state["document"])
        previous_segment_sum = self.state_codecs["segment_sum"].decode(
            state["segment_sum"])

        fast = self.fast_cell(
            torch.cat((token, previous_segment, previous_document), dim=-1),
            previous_fast, position)
        completed_segment_sum = previous_segment_sum + fast
        segment_tokens = int(state["segment_tokens"]) + 1
        segments = int(state["segments"])
        segment_updates = int(state["segment_updates"])
        document_updates = int(state["document_updates"])
        segment = previous_segment
        document = previous_document
        segment_persistent = state["segment"]
        document_persistent = state["document"]

        if segment_tokens == self.segment_length:
            segment = self.segment_cell(
                completed_segment_sum / segment_tokens,
                previous_segment, segment_updates)
            segment_persistent = self.state_codecs["segment"].encode(segment)
            segment_sum_persistent = torch.zeros_like(state["segment_sum"])
            segment_tokens = 0
            segment_updates += 1
            segments += 1
            if segments == self.document_segments:
                document = self.document_cell(
                    segment, previous_document, document_updates)
                document_persistent = self.state_codecs["document"].encode(
                    document)
                document_updates += 1
                segments = 0
        else:
            segment_sum_persistent = self.state_codecs["segment_sum"].encode(
                completed_segment_sum)

        hidden = self.readout(F.silu(self.cross_scale(torch.cat(
            (fast, segment, document), dim=-1), position)), position)
        new_state = {
            "fast": self.state_codecs["fast"].encode(fast),
            "segment": segment_persistent,
            "document": document_persistent,
            "segment_sum": segment_sum_persistent,
            "segment_tokens": segment_tokens,
            "segments": segments,
            "position": position + 1,
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

    def stored_state_elements(self, batch_size: int = 1,
                              context: int = 512) -> int:
        del context
        return len(self.persistent_tensor_keys) * batch_size * self.state_width

    def persistent_state_accounting(
            self, batch_size: int = 1,
            runtime_dtype: torch.dtype = torch.float32) -> dict:
        elements = self.stored_state_elements(batch_size)
        runtime_bytes_per_element = torch.empty(
            (), dtype=runtime_dtype).element_size()
        return {
            "batch_size": int(batch_size),
            "persistent_tensor_keys": list(self.persistent_tensor_keys),
            "persistent_tensor_shapes": {
                name: [int(batch_size), self.state_width]
                for name in self.persistent_tensor_keys
            },
            "persistent_elements": elements,
            "nominal_bf16_bytes": elements * 2,
            "nominal_bf16_bytes_per_sequence": (
                len(self.persistent_tensor_keys) * self.state_width * 2),
            "runtime_dtype": str(runtime_dtype).removeprefix("torch."),
            "runtime_bytes": elements * runtime_bytes_per_element,
            "runtime_bytes_per_sequence": (
                len(self.persistent_tensor_keys) * self.state_width
                * runtime_bytes_per_element),
            "decoded_hidden_width": self.width,
            "decoded_tensors_persistent": self.identity_codec_mode,
            "state_schema": self.state_schema,
        }

    def state_accounting(self, batch_size: int = 1, context: int = 512,
                         runtime_state=None) -> dict:
        del context
        accounting = self.persistent_state_accounting(
            batch_size, torch.float32)
        control_metadata = self.control_metadata_accounting(runtime_state)
        accounting.update({
            "tensor_payload_excludes_control_metadata": True,
            "control_metadata": control_metadata,
            "nominal_bf16_tensor_plus_logical_control_bytes_per_sequence": (
                accounting["nominal_bf16_bytes_per_sequence"]
                + self.control_metadata_bytes),
            "fp32_tensor_plus_logical_control_bytes_per_sequence": (
                accounting["runtime_bytes_per_sequence"]
                + self.control_metadata_bytes),
        })
        accounting["runtime_state_audited"] = runtime_state is not None
        if runtime_state is None:
            return accounting
        tensor_state = {
            key: value for key, value in runtime_state.items()
            if torch.is_tensor(value)
        }
        expected_keys = set(self.persistent_tensor_keys)
        if set(tensor_state) != expected_keys:
            raise RuntimeError(
                "runtime persistent tensors differ from declared state: "
                f"expected={sorted(expected_keys)} actual={sorted(tensor_state)}")
        component_runtime = {}
        actual_runtime_bytes = 0
        for key in self.persistent_tensor_keys:
            value = tensor_state[key]
            expected_shape = (int(batch_size), self.state_width)
            if tuple(value.shape) != expected_shape:
                raise RuntimeError(
                    f"runtime state shape mismatch for {key}: "
                    f"expected={expected_shape} actual={tuple(value.shape)}")
            if not self.identity_codec_mode and value.dtype != torch.float32:
                raise RuntimeError(
                    f"compressed runtime state {key} must be FP32, got "
                    f"{value.dtype}")
            component_bytes = value.numel() * value.element_size()
            actual_runtime_bytes += component_bytes
            component_runtime[key] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype).removeprefix("torch."),
                "bytes": component_bytes,
                "finite": bool(torch.isfinite(value).all().item()),
            }
            if not component_runtime[key]["finite"]:
                raise FloatingPointError(
                    f"nonfinite runtime persistent state component {key}")
        if (not self.identity_codec_mode
                and actual_runtime_bytes != accounting["runtime_bytes"]):
            raise RuntimeError(
                "compressed runtime bytes differ from declared FP32 contract: "
                f"expected={accounting['runtime_bytes']} "
                f"actual={actual_runtime_bytes}")
        accounting.update({
            "actual_component_runtime": component_runtime,
            "actual_runtime_bytes": actual_runtime_bytes,
        })
        return accounting

    def codec_macs_per_token(self) -> float:
        if self.identity_codec_mode:
            return 0.0
        # Four decodes, a fast encode, a segment-sum encode on non-boundary
        # tokens, a segment encode per segment, and a document encode per doc.
        projection = self.width * self.state_width
        return projection * (
            6.0 + 1.0 / (self.segment_length * self.document_segments))

    @torch.no_grad()
    def codec_conditioning_audit(
            self, maximum_condition_number: float,
            minimum_singular_value: float) -> dict:
        if self.identity_codec_mode:
            return {
                "identity_codec_mode": True,
                "components": {},
                "all_components_finite": True,
                "all_components_full_row_rank": True,
                "all_components_within_thresholds": True,
                "pass": True,
            }
        if maximum_condition_number <= 0 or minimum_singular_value <= 0:
            raise ValueError("codec conditioning thresholds must be positive")
        records = {}
        for name, codec in self.state_codecs.items():
            weight = codec.weight.detach().float()
            singular = torch.linalg.svdvals(weight)
            finite = bool(torch.isfinite(singular).all().item())
            minimum = float(singular.min().item()) if finite else float("nan")
            maximum = float(singular.max().item()) if finite else float("nan")
            condition = (
                maximum / minimum
                if finite and minimum > 0.0 else float("inf"))
            gram = weight @ weight.transpose(0, 1)
            identity = torch.eye(
                self.state_width, device=gram.device, dtype=gram.dtype)
            roundtrip_error = float((gram - identity).abs().max().item())
            full_row_rank = bool(
                finite and int(torch.count_nonzero(
                    singular >= minimum_singular_value).item())
                == self.state_width)
            within = bool(
                finite and full_row_rank
                and minimum >= minimum_singular_value
                and condition <= maximum_condition_number)
            records[name] = {
                "minimum_singular_value": minimum,
                "maximum_singular_value": maximum,
                "condition_number": condition,
                "latent_roundtrip_operator_max_abs_error": roundtrip_error,
                "full_row_rank_at_minimum_singular_value": full_row_rank,
                "finite": finite,
                "within_thresholds": within,
            }
        all_finite = all(row["finite"] for row in records.values())
        all_full_rank = all(
            row["full_row_rank_at_minimum_singular_value"]
            for row in records.values())
        all_within = all(row["within_thresholds"] for row in records.values())
        return {
            "identity_codec_mode": False,
            "maximum_condition_number": float(maximum_condition_number),
            "minimum_singular_value": float(minimum_singular_value),
            "components": records,
            "all_components_finite": all_finite,
            "all_components_full_row_rank": all_full_rank,
            "all_components_within_thresholds": all_within,
            "pass": bool(all_finite and all_full_rank and all_within),
        }

    def estimated_macs_per_token(self, context: int = 512) -> int:
        if self.identity_codec_mode:
            return super().estimated_macs_per_token(context)
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
            + self.codec_macs_per_token())


class _CoherentTemporalRead(torch.autograd.Function):
    """Bounded temporal residual with an exact-zero identity backward.

    Head and history are deliberately separate graph inputs.  At exact-zero
    initialization the forward is the head object and the backward sends the
    incoming gradient directly to that head, sends no gradient edge into the
    recurrent history, and still computes the mathematically live coefficient
    gradient.  This avoids adding materialized zeros through old recurrent
    heads, which can perturb CUDA gradient accumulation order.
    """

    @staticmethod
    def forward(ctx, head: torch.Tensor, history: torch.Tensor,
                raw_coefficients: torch.Tensor,
                exact_zero_mode: bool) -> torch.Tensor:
        older = history
        scale = math.sqrt(older.shape[1])
        coefficients = torch.tanh(raw_coefficients.to(head.dtype))
        differences = older - head.unsqueeze(1)
        preactivation = (
            differences * coefficients.unsqueeze(0)
        ).sum(dim=1) / scale
        bounded = torch.tanh(preactivation)
        ctx.save_for_backward(differences, coefficients, bounded)
        ctx.scale = scale
        ctx.raw_dtype = raw_coefficients.dtype
        ctx.exact_zero = bool(exact_zero_mode)
        return head if ctx.exact_zero else head + bounded

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        differences, coefficients, bounded = ctx.saved_tensors
        bounded_derivative = 1 - bounded.square()
        preactivation_gradient = gradient * bounded_derivative
        coefficient_gradient = (
            preactivation_gradient.unsqueeze(1)
            * differences / ctx.scale).sum(dim=0)
        raw_gradient = coefficient_gradient * (1 - coefficients.square())
        if ctx.exact_zero:
            return gradient, None, raw_gradient.to(ctx.raw_dtype), None
        # This form deliberately multiplies the incoming gradient by exactly
        # one when coefficients are zero.  It avoids an add-zero accumulation
        # node and therefore preserves original shared-parameter gradients.
        head_multiplier = (
            1 - bounded_derivative
            * coefficients.sum(dim=0) / ctx.scale)
        head_gradient = gradient * head_multiplier
        older_gradient = (
            preactivation_gradient.unsqueeze(1)
            * coefficients.unsqueeze(0) / ctx.scale)
        return head_gradient, older_gradient, raw_gradient.to(ctx.raw_dtype), None


class _CoherentIdentityRead(torch.autograd.Function):
    """Keep an unfilled coefficient region live without altering head backward."""

    @staticmethod
    def forward(ctx, head: torch.Tensor,
                raw_coefficients: torch.Tensor) -> torch.Tensor:
        ctx.raw_shape = raw_coefficients.shape
        ctx.raw_dtype = raw_coefficients.dtype
        ctx.raw_device = raw_coefficients.device
        return head

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        return gradient, torch.zeros(
            ctx.raw_shape, dtype=ctx.raw_dtype, device=ctx.raw_device)


class CoherentTemporalTapeFunctionWeightsStudent(
        HierarchicalFunctionWeightsStudent):
    """The original hierarchy with deterministic, age-ordered state history.

    There is still exactly one fast, segment, document, and segment-accumulator
    head.  Older values are retained as temporal taps rather than independent
    memories.  Every transition evaluates the original cells exactly once.  A
    zero-initialized, featurewise temporal residual lets those cells learn to
    consult older values without content selection or a second recurrent clock.

    All factors instantiate the maximum coefficient tensor.  Consequently the
    learned parameter graph is identical across the capacity sweep; only the
    number of active temporal taps and returned recurrent-state bytes changes.
    """

    kind = "coherent_temporal_tape_function_weights"
    state_schema = "separate_head_history_v1"
    _SUPPORTED_STATE_FACTORS = (1, 4, 16, 64)
    _COMPONENTS = ("fast", "segment", "document", "segment_sum")
    _MAX_STATE_FACTOR = 64

    def __init__(self, vocab_size: int, width: int, max_context: int,
                 segment_length: int, document_segments: int,
                 function_weights: dict, state_factor: int):
        super().__init__(
            vocab_size, width, max_context, segment_length,
            document_segments, function_weights)
        self.state_factor = int(state_factor)
        if self.state_factor not in self._SUPPORTED_STATE_FACTORS:
            raise ValueError(
                f"state_factor must be one of {self._SUPPORTED_STATE_FACTORS}")
        self.history_coefficients = nn.Parameter(torch.zeros(
            len(self._COMPONENTS), self._MAX_STATE_FACTOR - 1, self.width))

    def _wrap_original_state(self, state: dict) -> dict:
        return {
            **{
                name: state[name]
                for name in self._COMPONENTS
            },
            **{
                f"{name}_history": state[name].new_empty(
                    state[name].shape[0], 0, state[name].shape[1])
                for name in self._COMPONENTS
            },
            **{
                name: state[name]
                for name in (
                    "segment_tokens", "segments", "position",
                    "segment_updates", "document_updates")
            },
        }

    def _unwrap_original_state(self, state: dict) -> dict:
        return {
            **{
                name: state[name]
                for name in self._COMPONENTS
            },
            **{
                name: state[name]
                for name in (
                    "segment_tokens", "segments", "position",
                    "segment_updates", "document_updates")
            },
        }

    def initial_state(self, batch_size: int, device: torch.device,
                      dtype: torch.dtype | None = None):
        dtype = dtype or self.embedding.weight.dtype
        zero_head = lambda: torch.zeros(
            batch_size, self.width, device=device, dtype=dtype)
        zero_history = lambda: torch.zeros(
            batch_size, self.state_factor - 1, self.width,
            device=device, dtype=dtype)
        return {
            **{name: zero_head() for name in self._COMPONENTS},
            **{
                f"{name}_history": zero_history()
                for name in self._COMPONENTS
            },
            "segment_tokens": 0,
            "segments": 0,
            "position": 0,
            "segment_updates": 0,
            "document_updates": 0,
        }

    def _coherent_read(self, head: torch.Tensor, history: torch.Tensor,
                       component_index: int,
                       valid_taps: int,
                       exact_zero_mode: bool) -> torch.Tensor:
        valid_taps = min(self.state_factor, max(1, int(valid_taps)))
        if valid_taps == 1:
            # Keep the fixed parameter graph live while excluding every
            # unfilled temporal position from values and gradients.
            return _CoherentIdentityRead.apply(
                head, self.history_coefficients[component_index, :1])
        if self.state_factor == 1:
            return head
        return _CoherentTemporalRead.apply(
            head, history[:, :valid_taps - 1],
            self.history_coefficients[
                component_index, :valid_taps - 1], exact_zero_mode)

    @staticmethod
    def _shift_state(head: torch.Tensor, history: torch.Tensor,
                     new_head: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if history.shape[1] == 0:
            return new_head, history
        return new_head, torch.cat((head.unsqueeze(1), history[:, :-1]), dim=1)

    @staticmethod
    def _component_tape(state: dict, name: str) -> torch.Tensor:
        """Materialize a diagnostic tape without using it in recurrence."""
        return torch.cat((state[name].unsqueeze(1), state[f"{name}_history"]), dim=1)

    def _history_exact_zero(self) -> bool:
        return bool(
            torch.count_nonzero(self.history_coefficients.detach()).item() == 0)

    def transition(self, token_ids: torch.Tensor, state=None,
                   exact_zero_mode: bool | None = None):
        # This explicit branch is the scientific anchor.  It calls the frozen
        # original implementation, so factor-one hidden values, head state,
        # operation ordering, and shared-parameter gradients are bitwise exact.
        if self.state_factor == 1:
            original_state = (
                HierarchicalFunctionWeightsStudent.initial_state(
                    self, token_ids.shape[0], token_ids.device)
                if state is None else self._unwrap_original_state(state))
            hidden, new_original_state = super().transition(
                token_ids, original_state)
            # Keep the fixed-shape capacity parameter in the DDP graph without
            # changing the factor-one value.  The multiplication is exact zero
            # and gives the scientific anchor an explicit all-zero gradient.
            hidden = hidden + (
                self.history_coefficients.reshape(-1)[0].to(hidden.dtype) * 0.0)
            return hidden, self._wrap_original_state(new_original_state)

        if state is None:
            state = self.initial_state(token_ids.shape[0], token_ids.device)
        if exact_zero_mode is None:
            exact_zero_mode = self._history_exact_zero()
        position = int(state["position"])
        token = self.embedding(token_ids)
        segment_updates = int(state["segment_updates"])
        document_updates = int(state["document_updates"])
        valid_taps = {
            "fast": min(self.state_factor, position),
            "segment": min(self.state_factor, max(1, segment_updates)),
            "document": min(self.state_factor, max(1, document_updates)),
        }
        reads = {
            name: self._coherent_read(
                state[name], state[f"{name}_history"], component_index,
                valid_taps[name], exact_zero_mode)
            for component_index, name in enumerate(
                ("fast", "segment", "document"))
        }

        fast_input = torch.cat(
            (token, reads["segment"], reads["document"]), dim=-1)
        fast = self.fast_cell(fast_input, reads["fast"], position)
        # segment_sum[:, 0] is the one coherent in-progress accumulator.
        # Older positions retain completed segment sums and are consulted only
        # at the native segment boundary; partial sums are never mixed.
        completed_segment_sum = state["segment_sum"] + fast
        segment_sum = completed_segment_sum
        segment_tokens = int(state["segment_tokens"]) + 1
        segment = reads["segment"]
        document = reads["document"]
        segments = int(state["segments"])
        segment_changed = False
        document_changed = False
        if segment_tokens == self.segment_length:
            effective_completed_mean = self._coherent_read(
                completed_segment_sum / segment_tokens,
                state["segment_sum_history"] / segment_tokens, 3,
                min(self.state_factor, segment_updates + 1), exact_zero_mode)
            segment = self.segment_cell(
                effective_completed_mean,
                reads["segment"], segment_updates)
            segment_sum = torch.zeros_like(segment_sum)
            segment_tokens = 0
            segment_updates += 1
            segments += 1
            segment_changed = True
            if segments == self.document_segments:
                document = self.document_cell(
                    segment, reads["document"], document_updates)
                document_updates += 1
                segments = 0
                document_changed = True

        fast_head, fast_history = self._shift_state(
            state["fast"], state["fast_history"], fast)
        if segment_changed:
            segment_head, segment_history = self._shift_state(
                state["segment"], state["segment_history"], segment)
            document_head, document_history = (
                self._shift_state(
                    state["document"], state["document_history"], document)
                if document_changed else (
                    state["document"], state["document_history"]))
            segment_sum_head = torch.zeros_like(completed_segment_sum)
            segment_sum_history = torch.cat((
                completed_segment_sum.unsqueeze(1),
                state["segment_sum_history"][:, :-1]), dim=1)
        else:
            segment_head, segment_history = (
                state["segment"], state["segment_history"])
            document_head, document_history = (
                state["document"], state["document_history"])
            segment_sum_head, segment_sum_history = (
                completed_segment_sum, state["segment_sum_history"])
        new_state = {
            "fast": fast_head,
            "fast_history": fast_history,
            "segment": segment_head,
            "segment_history": segment_history,
            "document": document_head,
            "document_history": document_history,
            "segment_sum": segment_sum_head,
            "segment_sum_history": segment_sum_history,
            "segment_tokens": segment_tokens,
            "segments": segments,
            "position": position + 1,
            "segment_updates": segment_updates,
            "document_updates": document_updates,
        }
        # Read the newly advanced coherent state, not merely its raw heads.
        # This makes older fast/segment/document values available both to the
        # next transition and to the current readout.  Zero coefficients reduce
        # these reads exactly to the original newly computed heads.
        output_fast = self._coherent_read(
            new_state["fast"], new_state["fast_history"], 0,
            min(self.state_factor, position + 1), exact_zero_mode)
        output_segment = (
            self._coherent_read(
                new_state["segment"], new_state["segment_history"], 1,
                min(self.state_factor, max(1, segment_updates)), exact_zero_mode)
            if segment_changed else reads["segment"])
        output_document = (
            self._coherent_read(
                new_state["document"], new_state["document_history"], 2,
                min(self.state_factor, max(1, document_updates)), exact_zero_mode)
            if document_changed else reads["document"])
        combined = torch.cat(
            (output_fast, output_segment, output_document), dim=-1)
        hidden = self.readout(
            F.silu(self.cross_scale(combined, position)), position)
        return hidden, new_state

    def forward(self, input_ids: torch.Tensor):
        exact_zero_mode = self._history_exact_zero()
        state = None
        hidden_values = []
        for index in range(input_ids.shape[1]):
            hidden, state = self.transition(
                input_ids[:, index], state, exact_zero_mode=exact_zero_mode)
            hidden_values.append(hidden)
        if not hidden_values:
            raise ValueError("forward requires at least one input token")
        stacked_hidden = torch.stack(hidden_values, dim=1)
        return SimpleNamespace(
            logits=self.logits(stacked_hidden),
            hidden_states=stacked_hidden,
            final_state=state)

    def last_logits(self, input_ids: torch.Tensor):
        exact_zero_mode = self._history_exact_zero()
        state = None
        hidden = None
        for index in range(input_ids.shape[1]):
            hidden, state = self.transition(
                input_ids[:, index], state, exact_zero_mode=exact_zero_mode)
        if hidden is None:
            raise ValueError("last_logits requires at least one input token")
        return self.logits(hidden).unsqueeze(1)

    def stored_state_elements(self, batch_size: int = 1,
                              context: int = 512) -> int:
        del context
        return (
            len(self._COMPONENTS) * batch_size
            * self.state_factor * self.width)

    def temporal_tape_accounting(self, context: int) -> dict:
        tokens = max(0, int(context))
        updates = {
            "fast": tokens,
            "segment": tokens // self.segment_length,
            "document": (
                tokens // (self.segment_length * self.document_segments)),
            # At an exact boundary the head is the reset in-progress sum and
            # completed sums occupy positions1..; otherwise the live partial
            # sum is head plus the same completed history.
            "segment_sum": min(
                self.state_factor - 1,
                tokens // self.segment_length)
                + int(tokens % self.segment_length != 0),
        }
        valid = {
            name: min(self.state_factor, count)
            for name, count in updates.items()
        }
        total = len(self._COMPONENTS) * self.state_factor
        return {
            "valid_temporal_taps": valid,
            "active_temporal_taps": self.state_factor,
            "maximum_temporal_taps": self._MAX_STATE_FACTOR,
            "valid_taps_total": sum(valid.values()),
            "tape_positions_total": total,
            "utilization": sum(valid.values()) / total,
            "fixed_age_order": True,
            "content_dependent_selection": False,
            "native_shift_clocks": {
                "fast": "token",
                "segment": f"every_{self.segment_length}_tokens",
                "document": (
                    "every_"
                    f"{self.segment_length * self.document_segments}_tokens"),
                "segment_sum": (
                    "partial_head_each_token_completed_history_every_"
                    f"{self.segment_length}_tokens"),
            },
        }

    def max_exercised_history_lags(self, context: int) -> dict[str, int]:
        tokens = max(0, int(context))
        maximum = max(0, self.state_factor - 1)
        segment_updates = tokens // self.segment_length
        document_updates = (
            tokens // (self.segment_length * self.document_segments))
        return {
            "fast": min(maximum, max(0, tokens - 1)),
            "segment": min(maximum, max(0, segment_updates - 1)),
            "document": min(maximum, max(0, document_updates - 1)),
            "segment_sum": min(maximum, max(0, segment_updates - 1)),
        }

    def state_accounting(self, batch_size: int = 1,
                         context: int = 512,
                         runtime_state: dict | None = None) -> dict:
        elements = self.stored_state_elements(batch_size, context)
        components = None
        actual_runtime_bytes = None
        if runtime_state is not None:
            components = {
                name: {
                    "dtype": str(runtime_state[name].dtype).removeprefix(
                        "torch."),
                    "head_shape": list(runtime_state[name].shape),
                    "history_shape": list(
                        runtime_state[f"{name}_history"].shape),
                    "elements": (
                        runtime_state[name].numel()
                        + runtime_state[f"{name}_history"].numel()),
                    "element_bytes": runtime_state[name].element_size(),
                    "bytes": (
                        (runtime_state[name].numel()
                         + runtime_state[f"{name}_history"].numel())
                        * runtime_state[name].element_size()),
                }
                for name in self._COMPONENTS
            }
            actual_runtime_bytes = sum(
                item["bytes"] for item in components.values())
        return {
            "tensor_payload_elements": elements,
            "nominal_bf16_inference_payload_bytes": 2 * elements,
            "actual_runtime_tensor_payload_bytes": actual_runtime_bytes,
            "actual_runtime_component_accounting": components,
            "runtime_measurement_required": runtime_state is None,
            "control_metadata": (
                "five caller-owned Python integer clocks/counters; integer "
                "object size is runtime-dependent and excluded from tensor "
                "payload"),
            "context_length_invariant_payload": True,
            "state_factor": self.state_factor,
            "vectors_per_temporal_tap": len(self._COMPONENTS),
            "temporal_tape": self.temporal_tape_accounting(context),
        }

    def temporal_mixing_macs_per_token(self) -> int:
        # Fast, segment, and document are read before the native transition;
        # fast is reread after every transition, while segment/document are
        # reread only when their native clocks advance. Completed segment sums
        # are read only at the native segment boundary. One
        # multiply-accumulate is counted per older feature.  The analytic
        # convention excludes subtraction, tanh, and memory traffic. This is a
        # steady-state upper bound with the full configured tape populated;
        # early contexts execute fewer active lags.
        return int(
            (4 + 2 / self.segment_length
             + 1 / (self.segment_length * self.document_segments))
            * (self.state_factor - 1) * self.width)

    def temporal_compute_accounting(self) -> dict:
        return {
            "steady_state_full_tape_mixing_macs_per_token_upper_bound": (
                self.temporal_mixing_macs_per_token()),
            "active_lag_runtime_is_native_clock_bounded": True,
            "uncounted_operations": (
                "subtraction, tanh, concatenation, slicing, and memory traffic"),
        }

    def estimated_macs_per_token(self, context: int = 512) -> int:
        return (
            super().estimated_macs_per_token(context)
            + self.temporal_mixing_macs_per_token())


class ClockAwareCoherentTemporalTapeFunctionWeightsStudent(
        CoherentTemporalTapeFunctionWeightsStudent):
    """One coherent hierarchy with capacities allocated by native clocks."""

    kind = "clock_aware_coherent_temporal_tape_function_weights"
    _SUPPORTED_STATE_FACTORS = (1, 4, 16, 64)
    _MAX_CAPACITIES = (222, 16, 2, 16)
    _MAX_OLDER_LAGS = tuple(value - 1 for value in _MAX_CAPACITIES)
    _MAX_TOTAL_OLDER_LAGS = sum(_MAX_OLDER_LAGS)

    @classmethod
    def derive_capacities(cls, state_factor: int, context: int,
                          segment_length: int,
                          document_segments: int) -> tuple[int, int, int, int]:
        factor = int(state_factor)
        if factor not in cls._SUPPORTED_STATE_FACTORS:
            raise ValueError(
                f"state_factor must be one of {cls._SUPPORTED_STATE_FACTORS}")
        total = len(cls._COMPONENTS) * factor
        quarter = total // len(cls._COMPONENTS)
        segment = min(quarter, int(context) // int(segment_length))
        document = min(
            quarter,
            int(context) // (int(segment_length) * int(document_segments)))
        segment_sum = min(quarter, int(context) // int(segment_length))
        fast = total - segment - document - segment_sum
        return fast, segment, document, segment_sum

    def __init__(self, vocab_size: int, width: int, max_context: int,
                 segment_length: int, document_segments: int,
                 function_weights: dict, state_factor: int,
                 training_context: int,
                 state_capacities: list[int] | tuple[int, ...] | None = None):
        HierarchicalFunctionWeightsStudent.__init__(
            self, vocab_size, width, max_context, segment_length,
            document_segments, function_weights)
        self.state_factor = int(state_factor)
        self.training_context = int(training_context)
        derived = self.derive_capacities(
            self.state_factor, self.training_context,
            self.segment_length, self.document_segments)
        declared = derived if state_capacities is None else tuple(
            int(value) for value in state_capacities)
        if declared != derived:
            raise ValueError(
                f"state_capacities {declared} differ from derived {derived}")
        self.state_capacities = derived
        maximum = self.derive_capacities(
            max(self._SUPPORTED_STATE_FACTORS), self.training_context,
            self.segment_length, self.document_segments)
        if maximum != self._MAX_CAPACITIES:
            raise ValueError(
                f"training clocks derive unsupported maximum {maximum}")
        self._coefficient_offsets = {}
        offset = 0
        for name, maximum_lags in zip(self._COMPONENTS, self._MAX_OLDER_LAGS):
            self._coefficient_offsets[name] = (offset, offset + maximum_lags)
            offset += maximum_lags
        self.history_coefficients = nn.Parameter(torch.zeros(
            self._MAX_TOTAL_OLDER_LAGS, self.width))

    def _component_capacity(self, name_or_index) -> int:
        index = (
            self._COMPONENTS.index(name_or_index)
            if isinstance(name_or_index, str) else int(name_or_index))
        return self.state_capacities[index]

    def _component_coefficients(self, component_index: int) -> torch.Tensor:
        name = self._COMPONENTS[int(component_index)]
        start, stop = self._coefficient_offsets[name]
        return self.history_coefficients[start:stop]

    def component_history_gradient_regions(
            self, gradient: torch.Tensor) -> dict[str, torch.Tensor]:
        if tuple(gradient.shape) != tuple(self.history_coefficients.shape):
            raise ValueError("history gradient shape differs from coefficient graph")
        return {
            name: gradient[start:stop]
            for name, (start, stop) in self._coefficient_offsets.items()
        }

    def component_declared_history_lags(self) -> dict[str, int]:
        return {
            name: self._component_capacity(name) - 1
            for name in self._COMPONENTS
        }

    def initial_state(self, batch_size: int, device: torch.device,
                      dtype: torch.dtype | None = None):
        dtype = dtype or self.embedding.weight.dtype
        return {
            **{
                name: torch.zeros(
                    batch_size, self.width, device=device, dtype=dtype)
                for name in self._COMPONENTS
            },
            **{
                f"{name}_history": torch.zeros(
                    batch_size, self._component_capacity(name) - 1,
                    self.width, device=device, dtype=dtype)
                for name in self._COMPONENTS
            },
            "segment_tokens": 0,
            "segments": 0,
            "position": 0,
            "segment_updates": 0,
            "document_updates": 0,
        }

    def _coherent_read(self, head: torch.Tensor, history: torch.Tensor,
                       component_index: int, valid_taps: int,
                       exact_zero_mode: bool) -> torch.Tensor:
        capacity = self._component_capacity(component_index)
        valid_taps = min(capacity, max(1, int(valid_taps)))
        coefficients = self._component_coefficients(component_index)
        if valid_taps == 1:
            return _CoherentIdentityRead.apply(head, coefficients[:1])
        return _CoherentTemporalRead.apply(
            head, history[:, :valid_taps - 1],
            coefficients[:valid_taps - 1], exact_zero_mode)

    def _direct_temporal_transition(self, token_ids: torch.Tensor, state=None,
                                    exact_zero_mode: bool = False):
        if self.state_factor == 1:
            original_state = (
                HierarchicalFunctionWeightsStudent.initial_state(
                    self, token_ids.shape[0], token_ids.device)
                if state is None else self._unwrap_original_state(state))
            hidden, new_original_state = (
                HierarchicalFunctionWeightsStudent.transition(
                    self, token_ids, original_state))
            hidden = hidden + self.history_coefficients.reshape(-1)[0].to(
                hidden.dtype) * 0.0
            return hidden, self._wrap_original_state(new_original_state)
        if state is None:
            state = self.initial_state(token_ids.shape[0], token_ids.device)
        position = int(state["position"])
        segment_updates = int(state["segment_updates"])
        document_updates = int(state["document_updates"])
        token = self.embedding(token_ids)
        counts = {
            "fast": min(self._component_capacity("fast"), max(1, position)),
            "segment": min(
                self._component_capacity("segment"), max(1, segment_updates)),
            "document": min(
                self._component_capacity("document"), max(1, document_updates)),
        }
        reads = {
            name: self._coherent_read(
                state[name], state[f"{name}_history"], index, counts[name],
                exact_zero_mode)
            for index, name in enumerate(("fast", "segment", "document"))
        }
        fast = self.fast_cell(
            torch.cat((token, reads["segment"], reads["document"]), dim=-1),
            reads["fast"], position)
        completed_segment_sum = state["segment_sum"] + fast
        segment_sum = completed_segment_sum
        segment_tokens = int(state["segment_tokens"]) + 1
        segment = reads["segment"]
        document = reads["document"]
        segments = int(state["segments"])
        segment_changed = False
        document_changed = False
        if segment_tokens == self.segment_length:
            effective_completed_mean = self._coherent_read(
                completed_segment_sum / segment_tokens,
                state["segment_sum_history"] / segment_tokens, 3,
                min(self._component_capacity("segment_sum"),
                    segment_updates + 1), exact_zero_mode)
            segment = self.segment_cell(
                effective_completed_mean, reads["segment"], segment_updates)
            segment_sum = torch.zeros_like(segment_sum)
            segment_tokens = 0
            segment_updates += 1
            segments += 1
            segment_changed = True
            if segments == self.document_segments:
                document = self.document_cell(
                    segment, reads["document"], document_updates)
                document_updates += 1
                segments = 0
                document_changed = True
        fast_head, fast_history = self._shift_state(
            state["fast"], state["fast_history"], fast)
        if segment_changed:
            segment_head, segment_history = self._shift_state(
                state["segment"], state["segment_history"], segment)
            document_head, document_history = (
                self._shift_state(
                    state["document"], state["document_history"], document)
                if document_changed else (
                    state["document"], state["document_history"]))
            segment_sum_head = torch.zeros_like(completed_segment_sum)
            segment_sum_history = torch.cat((
                completed_segment_sum.unsqueeze(1),
                state["segment_sum_history"][:, :-1]), dim=1)
        else:
            segment_head, segment_history = (
                state["segment"], state["segment_history"])
            document_head, document_history = (
                state["document"], state["document_history"])
            segment_sum_head, segment_sum_history = (
                completed_segment_sum, state["segment_sum_history"])
        new_state = {
            "fast": fast_head,
            "fast_history": fast_history,
            "segment": segment_head,
            "segment_history": segment_history,
            "document": document_head,
            "document_history": document_history,
            "segment_sum": segment_sum_head,
            "segment_sum_history": segment_sum_history,
            "segment_tokens": segment_tokens,
            "segments": segments,
            "position": position + 1,
            "segment_updates": segment_updates,
            "document_updates": document_updates,
        }
        output_fast = self._coherent_read(
            new_state["fast"], new_state["fast_history"], 0,
            min(self._component_capacity("fast"), position + 1),
            exact_zero_mode)
        output_segment = (
            self._coherent_read(
                new_state["segment"], new_state["segment_history"], 1,
                min(self._component_capacity("segment"),
                    max(1, segment_updates)), exact_zero_mode)
            if segment_changed else reads["segment"])
        output_document = (
            self._coherent_read(
                new_state["document"], new_state["document_history"], 2,
                min(self._component_capacity("document"),
                    max(1, document_updates)), exact_zero_mode)
            if document_changed else reads["document"])
        hidden = self.readout(F.silu(self.cross_scale(torch.cat(
            (output_fast, output_segment, output_document), dim=-1),
            position)), position)
        return hidden, new_state

    def transition(self, token_ids: torch.Tensor, state=None):
        if not self._history_exact_zero():
            return self._direct_temporal_transition(
                token_ids, state, exact_zero_mode=False)

        original_state = (
            None if state is None else self._unwrap_original_state(state))
        hidden, primary_state = HierarchicalFunctionWeightsStudent.transition(
            self, token_ids, original_state)
        return hidden, primary_state

    def stored_state_elements(self, batch_size: int = 1,
                              context: int = 512) -> int:
        del context
        return batch_size * sum(self.state_capacities) * self.width

    def max_exercised_history_lags(self, context: int) -> dict[str, int]:
        tokens = max(0, int(context))
        natural = {
            "fast": max(0, tokens - 1),
            "segment": max(0, tokens // self.segment_length - 1),
            "document": max(
                0, tokens // (
                    self.segment_length * self.document_segments) - 1),
            "segment_sum": max(0, tokens // self.segment_length - 1),
        }
        return {
            name: min(self._component_capacity(name) - 1, natural[name])
            for name in self._COMPONENTS
        }

    def temporal_tape_accounting(self, context: int) -> dict:
        tokens = max(0, int(context))
        valid = {
            "fast": min(self._component_capacity("fast"), tokens),
            "segment": min(
                self._component_capacity("segment"),
                tokens // self.segment_length),
            "document": min(
                self._component_capacity("document"), tokens // (
                    self.segment_length * self.document_segments)),
            "segment_sum": min(
                self._component_capacity("segment_sum") - 1,
                tokens // self.segment_length)
                + int(tokens % self.segment_length != 0),
        }
        total = sum(self.state_capacities)
        return {
            "component_capacities": dict(zip(
                self._COMPONENTS, self.state_capacities)),
            "valid_temporal_taps": valid,
            "valid_taps_total": sum(valid.values()),
            "tape_positions_total": total,
            "utilization": sum(valid.values()) / total,
            "fixed_age_order": True,
            "content_dependent_selection": False,
            "native_shift_clocks": {
                "fast": "token",
                "segment": f"every_{self.segment_length}_tokens",
                "document": (
                    f"every_{self.segment_length * self.document_segments}_tokens"),
                "segment_sum": (
                    "partial_head_each_token_completed_history_every_"
                    f"{self.segment_length}_tokens"),
            },
        }

    def temporal_mixing_macs_per_token(self) -> int:
        older = dict(zip(self._COMPONENTS, (
            value - 1 for value in self.state_capacities)))
        return int((
            2 * older["fast"] + older["segment"] + older["document"]
            + older["segment"] / self.segment_length
            + older["document"] /
                (self.segment_length * self.document_segments)
            + older["segment_sum"] / self.segment_length) * self.width)

    def state_accounting(self, batch_size: int = 1,
                         context: int = 512,
                         runtime_state: dict | None = None) -> dict:
        result = super().state_accounting(
            batch_size, context, runtime_state)
        result.update({
            "state_capacities": dict(zip(
                self._COMPONENTS, self.state_capacities)),
            "total_state_vectors_per_sequence": sum(self.state_capacities),
            "history_coefficient_vectors": self._MAX_TOTAL_OLDER_LAGS,
        })
        result.pop("vectors_per_temporal_tap", None)
        result["temporal_tape"] = self.temporal_tape_accounting(context)
        return result

    def _original_forward(self, input_ids: torch.Tensor):
        """Evaluate the protected hierarchy without temporal graph wrappers."""
        state = None
        hidden_values = []
        for index in range(input_ids.shape[1]):
            if state is None:
                state = HierarchicalFunctionWeightsStudent.initial_state(
                    self, input_ids.shape[0], input_ids.device)
            hidden, state = HierarchicalFunctionWeightsStudent.transition(
                self, input_ids[:, index], state)
            hidden_values.append(hidden)
        if not hidden_values:
            raise ValueError("forward requires at least one input token")
        stacked_hidden = torch.stack(hidden_values, dim=1)
        return SimpleNamespace(
            logits=self.logits(stacked_hidden),
            hidden_states=stacked_hidden,
            final_state=state)

    def _direct_temporal_forward(self, input_ids: torch.Tensor):
        """Evaluate the v7 temporal recurrence, including its true derivative."""
        state = None
        hidden_values = []
        for index in range(input_ids.shape[1]):
            hidden, state = self._direct_temporal_transition(
                input_ids[:, index], state, exact_zero_mode=False)
            hidden_values.append(hidden)
        if not hidden_values:
            raise ValueError("forward requires at least one input token")
        stacked_hidden = torch.stack(hidden_values, dim=1)
        return SimpleNamespace(
            logits=self.logits(stacked_hidden),
            hidden_states=stacked_hidden,
            final_state=state)

    def _detached_shared_temporal_sidecar(self, input_ids: torch.Tensor):
        """Run temporal dynamics with only history coefficients trainable."""
        parameters = {
            name: (value if name == "history_coefficients" else value.detach())
            for name, value in self.named_parameters()
        }
        buffers = {
            name: value.detach() for name, value in self.named_buffers()
        }
        return torch.func.functional_call(
            self, (parameters, buffers), (input_ids,),
            {"_v9_direct_temporal": True}, strict=True)

    def coefficient_sidecar(self, input_ids: torch.Tensor):
        """Public coefficient-only derivative path for a second backward.

        Callers must first finish the primary objective backward, then build
        the identical objective from this output and differentiate only with
        respect to ``history_coefficients``.  No tensor from this sidecar is
        attached to the primary output graph.
        """
        if not self._history_exact_zero():
            raise RuntimeError(
                "coefficient_sidecar is only valid at exact-zero coefficients")
        return self._detached_shared_temporal_sidecar(input_ids)

    def forward(self, input_ids: torch.Tensor,
                _v9_direct_temporal: bool = False):
        if _v9_direct_temporal:
            return self._direct_temporal_forward(input_ids)
        if not self._history_exact_zero():
            return self._direct_temporal_forward(input_ids)

        # Exact-zero mode evaluates only the protected original recurrence.
        # Temporal state/derivatives are produced only through the explicit
        # sidecar API and never implicitly attached to ordinary forward.
        primary = self._original_forward(input_ids)
        return SimpleNamespace(
            logits=primary.logits,
            hidden_states=primary.hidden_states,
            final_state=primary.final_state)

    def last_logits(self, input_ids: torch.Tensor):
        if not self._history_exact_zero():
            return self._direct_temporal_forward(input_ids).logits[:, -1:]
        # Do not use the inherited wrapped transition at the exact-zero
        # boundary: branching callers receive the same protected original graph
        # as the full forward path.
        return self._original_forward(input_ids).logits[:, -1:]


class _PackedSparseRead(torch.autograd.Function):
    """One packed gather with four separate graph inputs and outputs."""

    @staticmethod
    def forward(ctx, fast, segment, document, segment_sum,
                probabilities, read_indices):
        banks = (fast, segment, document, segment_sum)
        if len({(value.dtype, value.device) for value in banks}) != 1:
            raise ValueError("packed read requires identical bank dtype/device")
        packed = torch.stack(banks, dim=0)
        gather_indices = read_indices.view(
            1, read_indices.shape[0], -1, 1).expand(
                4, -1, -1, fast.shape[-1])
        values = packed.gather(2, gather_indices)
        outputs = tuple(
            (probabilities.unsqueeze(-1) * values[component]).sum(dim=1)
            for component in range(4))
        ctx.save_for_backward(
            fast, segment, document, segment_sum,
            probabilities, read_indices, values)
        return outputs

    @staticmethod
    def backward(ctx, *gradients):
        fast, segment, document, segment_sum, probabilities, read_indices, \
            values = ctx.saved_tensors
        banks = (fast, segment, document, segment_sum)
        gather_indices = read_indices.unsqueeze(-1).expand(
            -1, -1, fast.shape[-1])
        bank_gradients = []
        probability_parts = []
        for component, (bank, gradient) in enumerate(zip(banks, gradients)):
            value_gradient = (
                gradient.unsqueeze(1) * probabilities.unsqueeze(-1))
            bank_gradients.append(
                torch.zeros_like(bank).scatter_add(
                    1, gather_indices, value_gradient))
            probability_parts.append((
                gradient.unsqueeze(1) * values[component]
            ).sum(dim=2).to(probabilities.dtype))
        probability_gradient = probability_parts[-1]
        for component in range(2, -1, -1):
            probability_gradient = (
                probability_gradient + probability_parts[component])
        return (*bank_gradients, probability_gradient, None)


class _PackedSparseReplacement(torch.autograd.Function):
    """One packed hard replacement with separate graph inputs and outputs."""

    @staticmethod
    def forward(ctx, fast, segment, document, segment_sum,
                fast_proposal, segment_proposal, document_proposal,
                segment_sum_proposal, probabilities, selected_indices):
        banks = (fast, segment, document, segment_sum)
        proposals = (
            fast_proposal, segment_proposal,
            document_proposal, segment_sum_proposal)
        if len({(value.dtype, value.device) for value in banks}) != 1:
            raise ValueError("packed replacement requires identical bank dtype/device")
        if len({(value.dtype, value.device) for value in proposals}) != 1:
            raise ValueError(
                "packed replacement requires identical proposal dtype/device")
        packed_bank = torch.stack(banks, dim=0)
        packed_proposal = torch.stack(proposals, dim=0)
        if packed_bank.dtype != packed_proposal.dtype:
            raise ValueError("packed replacement bank/proposal dtype mismatch")
        indices = selected_indices.view(1, -1, 1, 1).expand(
            4, -1, 1, fast.shape[-1])
        packed_output = packed_bank.scatter(
            2, indices, packed_proposal.unsqueeze(2))
        ctx.save_for_backward(*banks, *proposals, probabilities)
        return tuple(packed_output[component] for component in range(4))

    @staticmethod
    def backward(ctx, *gradients):
        saved = ctx.saved_tensors
        banks = saved[:4]
        proposals = saved[4:8]
        probabilities = saved[8]
        weights = probabilities.to(banks[0].dtype).unsqueeze(-1)
        bank_gradients = []
        proposal_gradients = []
        probability_parts = []
        for bank, proposal, gradient in zip(banks, proposals, gradients):
            weighted_gradient = gradient * weights
            bank_gradients.append(gradient - weighted_gradient)
            proposal_gradients.append(weighted_gradient.sum(dim=1))
            probability_parts.append((
                gradient * (proposal.unsqueeze(1) - bank)
            ).sum(dim=2).to(probabilities.dtype))
        probability_gradient = probability_parts[-1]
        for component in range(2, -1, -1):
            probability_gradient = (
                probability_gradient + probability_parts[component])
        return (*bank_gradients, *proposal_gradients, probability_gradient, None)


class HierarchicalRoutedRecurrentBankStudent(HierarchicalFunctionWeightsStudent):
    """Exact function-weight hierarchy over a sparse routed state bank.

    Each slot is one complete four-vector hierarchy state: fast, segment,
    document, and the active segment accumulator.  The learned module graph is
    exactly the original function-weight model.  A token content-routes a top-2
    aggregate, the original transition is evaluated once, and one complete
    slot is replaced by that proposal.  This is routed memory rather than K
    independent clock-coherent hierarchies: every proposal uses the global
    segment/document clock and a top-k aggregate.  Hard-forward writes touch
    one slot; dense-soft replacement supplies router gradients during training.
    """

    kind = "hierarchical_routed_recurrent_bank"
    _SUPPORTED_STATE_KIB = (10, 40, 160, 640)

    def __init__(self, vocab_size: int, width: int, max_context: int,
                 segment_length: int, document_segments: int,
                 function_weights: dict, state_kib: int, read_topk: int = 2,
                 routing_temperature: float = 0.25,
                 routing_diagnostics: bool = False):
        super().__init__(
            vocab_size, width, max_context, segment_length,
            document_segments, function_weights)
        self.state_kib = int(state_kib)
        self.read_topk = int(read_topk)
        self.routing_temperature = float(routing_temperature)
        self.routing_diagnostics = bool(routing_diagnostics)
        if self.state_kib not in self._SUPPORTED_STATE_KIB:
            raise ValueError(
                f"state_kib must be one of {self._SUPPORTED_STATE_KIB}")
        if self.state_kib % 10:
            raise ValueError("state_kib must be an integral multiple of 10")
        self.bank_slots = self.state_kib // 10
        if self.read_topk != 2:
            raise ValueError("routed recurrent bank requires read_topk=2")
        if (not math.isfinite(self.routing_temperature)
                or self.routing_temperature <= 0):
            raise ValueError("routing_temperature must be finite and positive")
        exact_bytes = 4 * self.bank_slots * self.width * 2
        if exact_bytes != self.state_kib * 1024:
            raise ValueError(
                "state_kib contract requires width=1280 so each complete "
                "hierarchy slot occupies exactly 10 KiB")

        # Diagnostic-only buffers: non-persistent, non-predictive, and absent
        # from parameter/state checkpoints.  They never affect route selection.
        self.register_buffer(
            "_routing_write_counts", torch.zeros(
                self.bank_slots, dtype=torch.long), persistent=False)
        self.register_buffer(
            "_routing_read_entropy", torch.zeros(()), persistent=False)
        self.register_buffer(
            "_routing_write_entropy", torch.zeros(()), persistent=False)
        self.register_buffer(
            "_routing_read_events", torch.zeros((), dtype=torch.long),
            persistent=False)
        self.register_buffer(
            "_routing_write_events", torch.zeros((), dtype=torch.long),
            persistent=False)
        self.register_buffer(
            "_routing_cold_write_events", torch.zeros((), dtype=torch.long),
            persistent=False)
        self.register_buffer(
            "_routing_score_gradient_l1", torch.zeros(()), persistent=False)
        self.register_buffer(
            "_routing_score_gradient_events", torch.zeros((), dtype=torch.long),
            persistent=False)
        self.register_buffer(
            "_routing_score_gradient_nonfinite", torch.zeros((), dtype=torch.long),
            persistent=False)
        self._last_write_indices = None
        self._last_read_indices = None

    @staticmethod
    def _entropy(probabilities: torch.Tensor) -> torch.Tensor:
        values = probabilities.float().clamp_min(torch.finfo(torch.float32).tiny)
        return -(values * values.log()).sum(dim=-1)

    @staticmethod
    def _content_scores(query: torch.Tensor,
                        fast_bank: torch.Tensor) -> torch.Tensor:
        normalized_query = F.normalize(query.float(), dim=-1)
        normalized_keys = F.normalize(fast_bank.float(), dim=-1)
        return torch.einsum("bw,bkw->bk", normalized_query, normalized_keys)

    def _route(self, query: torch.Tensor, state: dict,
               use_packed_read: bool = False):
        batch = query.shape[0]
        position = int(state["position"])
        occupied = min(position, self.bank_slots)
        if occupied == 0:
            scores = torch.zeros(
                batch, self.bank_slots, device=query.device,
                dtype=torch.float32)
            full_probabilities = (
                torch.full_like(scores, 1.0 / self.bank_slots)
                if self.routing_diagnostics else None)
            read_indices = torch.zeros(
                batch, 1, device=query.device, dtype=torch.long)
            read_probabilities = torch.ones(
                batch, 1, device=query.device, dtype=query.dtype)
            aggregate = {
                name: torch.zeros_like(query)
                for name in ("fast", "segment", "document", "segment_sum")
            }
            if self.routing_diagnostics:
                self._last_read_indices = read_indices.detach().clone()
            return aggregate, scores, full_probabilities
        if self.bank_slots == 1:
            scores = torch.zeros(
                batch, 1, device=query.device, dtype=torch.float32)
            full_probabilities = (
                torch.ones_like(scores) if self.routing_diagnostics else None)
            read_indices = torch.zeros(
                batch, 1, device=query.device, dtype=torch.long)
            read_probabilities = torch.ones(
                batch, 1, device=query.device, dtype=query.dtype)
        else:
            # Score the full bank exactly as frozen v2. Restricting the
            # contraction to occupied slots changes einsum kernel rounding,
            # even though empty slots are mathematically zero.
            scores = self._content_scores(query, state["fast_bank"])
            if occupied == self.bank_slots:
                masked_scores = scores
                full_probabilities = (
                    torch.softmax(
                        scores / self.routing_temperature, dim=-1)
                    if self.routing_diagnostics else None)
            else:
                occupied_mask = torch.arange(
                    self.bank_slots, device=query.device).unsqueeze(0) < occupied
                masked_scores = scores.masked_fill(
                    ~occupied_mask, float("-inf"))
                # Production writes are deterministic during cold fill.  The
                # v3 full-bank softmax was returned only to diagnostics and
                # then discarded by transition; do not build it in silent
                # timed mode.  The top-k read softmax below remains exact.
                full_probabilities = (
                    torch.softmax(
                        masked_scores / self.routing_temperature, dim=-1)
                    if self.routing_diagnostics else None)
            selected = min(self.read_topk, occupied)
            top_scores, read_indices = torch.topk(
                masked_scores, k=selected, dim=-1, sorted=True)
            read_probabilities = torch.softmax(
                top_scores / self.routing_temperature, dim=-1).to(query.dtype)
        if self.bank_slots == 1:
            aggregate = {
                name: state[f"{name}_bank"].squeeze(1)
                for name in ("fast", "segment", "document", "segment_sum")
            }
        else:
            # One packed gather/reduction replaces four independent gather,
            # multiply, and sum launch chains while preserving each component.
            if use_packed_read:
                names = ("fast", "segment", "document", "segment_sum")
                outputs = _PackedSparseRead.apply(
                    *(state[f"{name}_bank"] for name in names),
                    read_probabilities, read_indices)
                aggregate = dict(zip(names, outputs))
            else:
                # Fail closed to the frozen v2 component-wise path when a
                # future runtime supplies mixed component dtypes/devices.
                gather_indices = read_indices.unsqueeze(-1).expand(
                    -1, -1, self.width)
                aggregate = {}
                for name in (
                        "fast", "segment", "document", "segment_sum"):
                    values = state[f"{name}_bank"].gather(
                        1, gather_indices)
                    aggregate[name] = (
                        read_probabilities.unsqueeze(-1) * values).sum(dim=1)
        if self.routing_diagnostics:
            with torch.no_grad():
                self._routing_read_entropy.add_(
                    self._entropy(read_probabilities).sum())
                self._routing_read_events.add_(batch)
            self._last_read_indices = read_indices.detach().clone()
        return aggregate, scores, full_probabilities

    def _capture_score_gradient(self, gradient: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            finite = torch.isfinite(gradient)
            self._routing_score_gradient_l1.add_(
                gradient.masked_fill(~finite, 0).abs().sum())
            self._routing_score_gradient_events.add_(gradient.numel())
            self._routing_score_gradient_nonfinite.add_((~finite).sum())
        return gradient

    def _write_probabilities(self, scores: torch.Tensor, position: int,
                             selected_indices: torch.Tensor) -> torch.Tensor:
        if self.bank_slots == 1:
            return torch.ones_like(scores)
        if position < self.bank_slots:
            # Cold allocation is a deterministic control decision.  Its
            # surrogate must target the same empty slot so the stored proposal
            # remains connected to the training loss.
            return F.one_hot(
                selected_indices, num_classes=self.bank_slots).to(scores.dtype)
        # Diagnostic hooks are sampled deterministically; they never alter the
        # route or learned gradient.  One first-overwrite sample plus every
        # 128th global token is enough to prove recurring finite/nonzero route
        # gradients without one Python hook per token.
        if (self.routing_diagnostics and scores.requires_grad and (
                position == self.bank_slots or position % 128 == 0)):
            scores.register_hook(self._capture_score_gradient)
        # Recompute here, as frozen v2 does. Reusing the numerically identical
        # read-side tensor changes autograd node creation/traversal order and
        # introduces ulp-scale learned-gradient drift over long unrolls.
        return torch.softmax(scores / self.routing_temperature, dim=-1)

    @staticmethod
    def _sparse_straight_through_innovation(
            bank: torch.Tensor, proposal: torch.Tensor,
            aggregate: torch.Tensor, probabilities: torch.Tensor,
            selected_indices: torch.Tensor) -> torch.Tensor:
        """Hard selected replacement with a dense-soft backward surrogate."""
        if bank.shape[:2] != probabilities.shape:
            raise ValueError("routing probabilities do not match bank")
        if proposal.shape != aggregate.shape or proposal.shape != (
                bank.shape[0], bank.shape[2]):
            raise ValueError("proposal/aggregate does not match bank")
        if bank.shape[1] == 1:
            # Algebraically and numerically exact original K=1 transition.
            return proposal.unsqueeze(1)
        del aggregate
        batch = torch.arange(bank.shape[0], device=bank.device)
        with torch.no_grad():
            hard = bank.detach().clone()
            hard[batch, selected_indices] = proposal.detach()
        if not torch.is_grad_enabled():
            return hard
        surrogate = bank + probabilities.to(bank.dtype).unsqueeze(-1) * (
            proposal.unsqueeze(1) - bank)
        # Parentheses are intentional: surrogate-surrogate.detach() is an
        # exact forward zero.  Left-associative addition/subtraction can round
        # unchanged slots and violate the selected-only forward contract.
        return hard + (surrogate - surrogate.detach())

    def _select_write(self, scores: torch.Tensor, position: int) -> torch.Tensor:
        # Deterministic cold allocation fills every initially empty slot once.
        # Afterwards writes are parameter-free content argmax decisions.
        if position < self.bank_slots:
            if self.routing_diagnostics:
                self._routing_cold_write_events.add_(scores.shape[0])
            return torch.full(
                (scores.shape[0],), position, device=scores.device,
                dtype=torch.long)
        return scores.argmax(dim=-1)

    def initial_state(self, batch_size: int, device: torch.device,
                      dtype: torch.dtype | None = None):
        dtype = dtype or self.embedding.weight.dtype
        zero_bank = lambda: torch.zeros(
            batch_size, self.bank_slots, self.width,
            device=device, dtype=dtype)
        return {
            "fast_bank": zero_bank(),
            "segment_bank": zero_bank(),
            "document_bank": zero_bank(),
            "segment_sum_bank": zero_bank(),
            # Autoregressive position is caller-owned control metadata.  Every
            # slower clock/occupancy fact is derived from it, so no per-slot
            # mask, key, age, or counter consumes recurrent payload.
            "position": 0,
        }

    def transition(self, token_ids: torch.Tensor, state=None):
        if state is None:
            state = self.initial_state(token_ids.shape[0], token_ids.device)
        position = int(state["position"])
        token = self.embedding(token_ids)
        names = ("fast", "segment", "document", "segment_sum")
        bank_components = tuple(state[f"{name}_bank"] for name in names)
        bank_dtype_devices = {
            (component.dtype, component.device) for component in bank_components}
        use_packed_read = self.bank_slots > 1 and len(bank_dtype_devices) == 1
        aggregate, scores, _ = self._route(
            token, state, use_packed_read)

        fast_input = torch.cat((
            token, aggregate["segment"], aggregate["document"]), dim=-1)
        proposal = dict(aggregate)
        proposal["fast"] = self.fast_cell(
            fast_input, aggregate["fast"], position)
        proposal["segment_sum"] = (
            aggregate["segment_sum"] + proposal["fast"])
        completed_tokens = position + 1
        if completed_tokens % self.segment_length == 0:
            segment_update = completed_tokens // self.segment_length - 1
            proposal["segment"] = self.segment_cell(
                proposal["segment_sum"] / self.segment_length,
                aggregate["segment"], segment_update)
            proposal["segment_sum"] = torch.zeros_like(
                proposal["segment_sum"])
            if segment_update % self.document_segments == self.document_segments - 1:
                document_update = (
                    completed_tokens
                    // (self.segment_length * self.document_segments) - 1)
                proposal["document"] = self.document_cell(
                    proposal["segment"], aggregate["document"],
                    document_update)

        combined = torch.cat((
            proposal["fast"], proposal["segment"], proposal["document"]),
            dim=-1)
        hidden = self.readout(
            F.silu(self.cross_scale(combined, position)), position)

        selected_indices = self._select_write(scores, position)
        write_probabilities = self._write_probabilities(
            scores, position, selected_indices)
        new_state = {"position": completed_tokens}
        if self.bank_slots == 1:
            # Preserve the original exact K=1 path and its numerical ordering.
            for name in names:
                storage_bank = state[f"{name}_bank"].to(proposal[name].dtype)
                new_state[f"{name}_bank"] = (
                    self._sparse_straight_through_innovation(
                        storage_bank, proposal[name], aggregate[name],
                        write_probabilities, selected_indices))
        else:
            proposal_components = tuple(proposal[name] for name in names)
            proposal_dtype_devices = {
                (component.dtype, component.device)
                for component in proposal_components}
            can_pack = (
                use_packed_read
                and len(proposal_dtype_devices) == 1
                and next(iter(bank_dtype_devices))[1]
                == next(iter(proposal_dtype_devices))[1])
            if can_pack:
                proposal_dtype = proposal_components[0].dtype
                storage_components = tuple(
                    bank.to(proposal_dtype) for bank in bank_components)
                outputs = _PackedSparseReplacement.apply(
                    *storage_components, *proposal_components,
                    write_probabilities, selected_indices)
                for name, output in zip(names, outputs):
                    new_state[f"{name}_bank"] = output
            else:
                # Mixed component precision is semantically meaningful under
                # autocast. Never silently promote it through torch.stack.
                for name in names:
                    storage_bank = state[f"{name}_bank"].to(
                        proposal[name].dtype)
                    new_state[f"{name}_bank"] = (
                        self._sparse_straight_through_innovation(
                            storage_bank, proposal[name], aggregate[name],
                            write_probabilities, selected_indices))
        if self.routing_diagnostics:
            with torch.no_grad():
                self._routing_write_counts.scatter_add_(
                    0, selected_indices.detach(), torch.ones_like(
                        selected_indices,
                        dtype=self._routing_write_counts.dtype))
                self._routing_write_entropy.add_(
                    self._entropy(write_probabilities).sum())
                self._routing_write_events.add_(token_ids.shape[0])
            self._last_write_indices = selected_indices.detach().clone()
        return hidden, new_state

    def stored_state_elements(self, batch_size: int = 1,
                              context: int = 512) -> int:
        del context
        return 4 * batch_size * self.bank_slots * self.width

    def state_accounting(self, batch_size: int = 1,
                         context: int = 512,
                         runtime_state: dict | None = None) -> dict:
        elements = self.stored_state_elements(batch_size, context)
        runtime_dtype_bytes = None
        actual_runtime_bytes = None
        if runtime_state is not None:
            banks = {
                name: runtime_state[f"{name}_bank"]
                for name in ("fast", "segment", "document", "segment_sum")
            }
            runtime_dtype_bytes = {
                name: {
                    "dtype": str(value.dtype).removeprefix("torch."),
                    "elements": value.numel(),
                    "element_bytes": value.element_size(),
                    "bytes": value.numel() * value.element_size(),
                }
                for name, value in banks.items()
            }
            actual_runtime_bytes = sum(
                item["bytes"] for item in runtime_dtype_bytes.values())
        diagnostic_buffer_bytes = (
            self._routing_write_counts.numel()
            * self._routing_write_counts.element_size()
            + sum(getattr(self, name).element_size() for name in (
                "_routing_read_entropy", "_routing_write_entropy",
                "_routing_read_events", "_routing_write_events",
                "_routing_cold_write_events", "_routing_score_gradient_l1",
                "_routing_score_gradient_events",
                "_routing_score_gradient_nonfinite")))
        return {
            "tensor_payload_elements": elements,
            "nominal_bf16_inference_payload_bytes": 2 * elements,
            "actual_runtime_tensor_payload_bytes": actual_runtime_bytes,
            "actual_runtime_component_accounting": runtime_dtype_bytes,
            "runtime_measurement_required": runtime_state is None,
            "precision_confound_warning": (
                "GPU preflight must compare each component dtype with the "
                "protected K=1 runtime; capacity screens are blocked on a "
                "component-specific precision mismatch"),
            "control_metadata": (
                "caller-owned Python integer position in returned state; "
                "implementation object size is runtime-dependent"),
            "canonical_uint64_control_serialization_bytes": 8,
            "required_external_control_serialization_bytes": 8,
            "persistent_per_slot_mask_counter_or_key_bytes": 0,
            "diagnostic_buffer_bytes": diagnostic_buffer_bytes,
            "diagnostic_state_bytes": diagnostic_buffer_bytes,
            "diagnostic_buffers_are_nonpersistent_and_nonpredictive": True,
            "diagnostics_persist_across_checkpoint_resume": False,
            "bank_slots": self.bank_slots,
            "vectors_per_slot": 4,
            "context_length_invariant": True,
        }

    def routing_compute_accounting(self) -> dict:
        if self.bank_slots == 1:
            logical = 0
            dense_training = 0
        else:
            selected = min(self.read_topk, self.bank_slots)
            # MAC convention: cosine dot-product scoring is 2*K*W and four
            # weighted top-k reads are 4*topk*W.  Norms/top-k/softmax are
            # disclosed separately as uncounted scalar operations.
            logical = self.width * (
                2 * self.bank_slots + 4 * selected)
            # Autograd additionally materializes a dense soft replacement for
            # all four banks: four K-by-W multiply-adds.
            dense_training = logical + 4 * self.bank_slots * self.width
        dense_elements = 4 * self.bank_slots * self.width
        return {
            "logical_sparse_inference_routing_macs_per_token": int(logical),
            "actual_dense_training_routing_macs_per_token_estimate": int(
                dense_training),
            "dense_surrogate_bank_elements_per_token": int(dense_elements),
            "hard_clone_bank_elements_per_token": int(dense_elements),
            "assumptions": (
                "analytic MAC convention excludes normalization, top-k, "
                "softmax, indexing, clone, scatter, and memory traffic"),
        }

    def estimated_routing_macs_per_token(self) -> int:
        return int(self.routing_compute_accounting()[
            "logical_sparse_inference_routing_macs_per_token"])

    def estimated_macs_per_token(self, context: int = 512) -> int:
        return (super().estimated_macs_per_token(context)
                + self.estimated_routing_macs_per_token())

    def reset_routing_metrics(self) -> None:
        for name in (
                "_routing_write_counts", "_routing_read_entropy",
                "_routing_write_entropy", "_routing_read_events",
                "_routing_write_events", "_routing_cold_write_events",
                "_routing_score_gradient_l1",
                "_routing_score_gradient_events",
                "_routing_score_gradient_nonfinite"):
            getattr(self, name).zero_()

    def set_routing_diagnostics(self, enabled: bool,
                                reset: bool = True) -> bool:
        """Switch non-predictive audit instrumentation outside timed work."""
        previous = self.routing_diagnostics
        self.routing_diagnostics = bool(enabled)
        if reset:
            self.reset_routing_metrics()
        self._last_write_indices = None
        self._last_read_indices = None
        return previous

    def routing_metrics(self, reset: bool = False) -> dict:
        counts = self._routing_write_counts
        read_events = int(self._routing_read_events.detach().cpu())
        write_events = int(self._routing_write_events.detach().cpu())
        metrics = {
            "routing_bank_slots": self.bank_slots,
            "routing_read_topk": min(self.read_topk, self.bank_slots),
            "routing_temperature": self.routing_temperature,
            "routing_diagnostics_enabled": self.routing_diagnostics,
            "routing_metrics_scope": "rank-local since process start",
            "routing_diagnostics_persist_across_checkpoint_resume": False,
            "routing_read_events": read_events,
            "routing_write_events": write_events,
            "routing_cold_write_events": int(
                self._routing_cold_write_events.detach().cpu()),
            "routing_write_counts": [
                int(value) for value in counts.detach().cpu().tolist()],
            "routing_occupied_slots": int(
                (counts > 0).sum().detach().cpu()),
            "routing_effective_write_slots": float(
                torch.exp(self._entropy(
                    (counts.float() / counts.sum().clamp_min(1)).unsqueeze(0)
                )[0]).detach().cpu()) if write_events else 0.0,
            "routing_max_slot_load": float(
                counts.max().float().div(counts.sum().clamp_min(1)).detach().cpu())
                if write_events else 0.0,
            "routing_slot_utilization": float(
                (counts > 0).float().mean().detach().cpu()),
            "routing_read_entropy": (
                float(self._routing_read_entropy.detach().cpu()) / read_events
                if read_events else 0.0),
            "routing_write_entropy": (
                float(self._routing_write_entropy.detach().cpu()) / write_events
                if write_events else 0.0),
            "routing_score_gradient_l1": float(
                self._routing_score_gradient_l1.detach().cpu()),
            "routing_score_gradient_events": int(
                self._routing_score_gradient_events.detach().cpu()),
            "routing_score_gradient_nonfinite": int(
                self._routing_score_gradient_nonfinite.detach().cpu()),
            "routing_score_gradient_sampling": (
                "first content overwrite and global positions divisible by 128"),
        }
        if reset:
            self.reset_routing_metrics()
        return metrics


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
    if kind == "nanogpt":
        return NanoGPTStudent(
            vocab_size, width, max_context, int(config["layers"]),
            int(config["heads"]), float(config.get("dropout", 0.0)),
            bool(config.get("bias", True)))
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
    if kind == "bottleneck_hierarchical_function_weights":
        return BottleneckHierarchicalFunctionWeightsStudent(
            vocab_size, width, max_context, int(config["segment_length"]),
            int(config["document_segments"]), int(config["state_width"]),
            dict(config["function_weights"]))
    if kind == "coherent_temporal_tape_function_weights":
        return CoherentTemporalTapeFunctionWeightsStudent(
            vocab_size, width, max_context, int(config["segment_length"]),
            int(config["document_segments"]), dict(config["function_weights"]),
            int(config["state_factor"]))
    if kind == "clock_aware_coherent_temporal_tape_function_weights":
        return ClockAwareCoherentTemporalTapeFunctionWeightsStudent(
            vocab_size, width, max_context, int(config["segment_length"]),
            int(config["document_segments"]), dict(config["function_weights"]),
            int(config["state_factor"]), int(config["training_context"]),
            tuple(config["state_capacities"]))
    if kind == "hierarchical_routed_recurrent_bank":
        return HierarchicalRoutedRecurrentBankStudent(
            vocab_size, width, max_context, int(config["segment_length"]),
            int(config["document_segments"]), dict(config["function_weights"]),
            int(config["state_kib"]), int(config.get("read_topk", 2)),
            float(config.get("routing_temperature", 0.25)),
            bool(config.get("routing_diagnostics", False)))
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
    if kind == "nanogpt":
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
