"""Atlas V2: depth-grouped, routed, signed functional attention.

Atlas V2 keeps dense attention only as a frozen training bypass.  At alpha=1
the forward path contains only the deployable functional operator, coordinate
adapters, and the original residual/MLP stack.
"""
from __future__ import annotations

import math
import weakref
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from representation_experiments.operator_attention import _causal_wavelet, _local_exact_attention


def _causal_signed_operator(query: torch.Tensor, key: torch.Tensor,
                            value: torch.Tensor) -> torch.Tensor:
    """Linear-time signed associative memory for [B,T,H,F] features."""
    query_float = query.float()
    key_float = key.float()
    value_float = value.float()
    key_value = torch.einsum(
        "bthf,bthd->bthfd", key_float, value_float).cumsum(dim=1)
    key_mass = key_float.abs().cumsum(dim=1)
    numerator = torch.einsum("bthf,bthfd->bthd", query_float, key_value)
    denominator = torch.einsum(
        "bthf,bthf->bth", query_float.abs(), key_mass).unsqueeze(-1)
    return (numerator / denominator.clamp_min(1e-5)).to(value.dtype)


def _multi_window_local(query: torch.Tensor, key: torch.Tensor,
                        value: torch.Tensor, windows: list[int]) -> torch.Tensor:
    """Apply exact local attention independently per functional head."""
    batch, sequence, heads, feature = query.shape
    value_dim = value.shape[-1]
    flat_query = query.permute(0, 2, 1, 3).reshape(batch * heads, sequence, feature)
    flat_key = key.permute(0, 2, 1, 3).reshape(batch * heads, sequence, feature)
    flat_value = value.permute(0, 2, 1, 3).reshape(
        batch * heads, sequence, value_dim)
    messages = [
        _local_exact_attention(flat_query, flat_key, flat_value, int(window))
        for window in windows
    ]
    combined = torch.stack(messages, dim=0).mean(dim=0)
    return combined.reshape(batch, heads, sequence, value_dim).permute(0, 2, 1, 3)


class ResidualCoordinateAdapter(nn.Module):
    def __init__(self, width: int, rank: int):
        super().__init__()
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, width, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.up(F.silu(self.down(hidden)))


class AtlasV2Bank(nn.Module):
    def __init__(self, hidden_size: int, heads: int, charts: int, basis: int,
                 topk_charts: int, value_dim: int, local_features: int,
                 local_windows: list[int], latent: int, wavelet_scales: int,
                 depth_basis: int):
        super().__init__()
        if not 1 <= topk_charts <= charts:
            raise ValueError("topk_charts must be between one and charts")
        self.hidden_size = int(hidden_size)
        self.heads = int(heads)
        self.charts = int(charts)
        self.basis = int(basis)
        self.topk_charts = int(topk_charts)
        self.value_dim = int(value_dim)
        self.local_features = int(local_features)
        self.local_windows = [int(value) for value in local_windows]
        self.latent = int(latent)
        self.depth_basis = int(depth_basis)

        self.chart_gate = nn.Linear(hidden_size, heads * charts, bias=False)
        self.basis_feature = nn.Linear(hidden_size, heads * basis, bias=False)
        self.value = nn.Linear(hidden_size, heads * value_dim, bias=False)
        self.local_query = nn.Linear(hidden_size, heads * local_features, bias=False)
        self.local_key = nn.Linear(hidden_size, heads * local_features, bias=False)
        self.wavelet_down = nn.Linear(hidden_size, latent, bias=False)
        self.wavelet_up = nn.Linear(latent, heads * value_dim, bias=False)
        self.wavelet_gains = nn.Parameter(torch.zeros(wavelet_scales, latent))
        self.token_mix = nn.Linear(hidden_size, heads * 3, bias=False)
        self.depth_chart = nn.Linear(depth_basis, heads * charts, bias=False)
        self.depth_mix = nn.Linear(depth_basis, heads * 3, bias=False)
        self.output = nn.Linear(heads * value_dim, hidden_size, bias=False)
        nn.init.zeros_(self.output.weight)
        self._auxiliary: list[dict[str, torch.Tensor]] = []

    def depth_features(self, depth: float, device: torch.device) -> torch.Tensor:
        position = torch.tensor(float(depth), device=device, dtype=torch.float32)
        orders = torch.arange(self.depth_basis, device=device, dtype=torch.float32)
        return torch.cos(math.pi * orders * position)

    def clear_auxiliary(self) -> None:
        self._auxiliary.clear()

    def auxiliary_losses(self) -> dict[str, torch.Tensor]:
        if not self._auxiliary:
            zero = self.output.weight.sum() * 0.0
            return {"chart_balance": zero, "mix_balance": zero}
        return {
            key: torch.stack([entry[key] for entry in self._auxiliary]).mean()
            for key in self._auxiliary[0]
        }

    def forward(self, hidden: torch.Tensor, depth: float) -> torch.Tensor:
        batch, sequence, _ = hidden.shape
        depth_feature = self.depth_features(depth, hidden.device)
        chart_logits = self.chart_gate(hidden).view(
            batch, sequence, self.heads, self.charts)
        chart_logits = chart_logits.float() + self.depth_chart(depth_feature).view(
            1, 1, self.heads, self.charts)
        selected_value, selected_index = torch.topk(
            chart_logits, self.topk_charts, dim=-1)
        selected_probability = torch.softmax(selected_value, dim=-1)
        chart = torch.zeros_like(chart_logits).scatter(
            -1, selected_index, selected_probability)
        basis = torch.tanh(self.basis_feature(hidden).float()).view(
            batch, sequence, self.heads, self.basis)
        feature = (chart.unsqueeze(-1) * basis.unsqueeze(-2)).flatten(-2)

        value = self.value(hidden).view(
            batch, sequence, self.heads, self.value_dim)
        global_message = _causal_signed_operator(feature, feature, value)
        query = self.local_query(hidden).view(
            batch, sequence, self.heads, self.local_features)
        key = self.local_key(hidden).view(
            batch, sequence, self.heads, self.local_features)
        local_message = _multi_window_local(
            query, key, value, self.local_windows)
        wavelet_message = self.wavelet_up(_causal_wavelet(
            self.wavelet_down(hidden), self.wavelet_gains)).view(
                batch, sequence, self.heads, self.value_dim)

        mix_logits = self.token_mix(hidden).view(batch, sequence, self.heads, 3)
        mix_logits = mix_logits.float() + self.depth_mix(depth_feature).view(
            1, 1, self.heads, 3)
        mixture = torch.softmax(mix_logits, dim=-1).to(value.dtype)
        message = (
            mixture[..., 0:1] * global_message
            + mixture[..., 1:2] * local_message
            + mixture[..., 2:3] * wavelet_message)

        chart_usage = chart.mean(dim=(0, 1, 2))
        mix_usage = mixture.float().mean(dim=(0, 1, 2))
        self._auxiliary.append({
            "chart_balance": self.charts * (
                chart_usage - 1.0 / self.charts).square().mean(),
            "mix_balance": 3.0 * (mix_usage - 1.0 / 3.0).square().mean(),
        })
        return self.output(message.flatten(-2))

    def estimated_flops(self, sequence: int) -> int:
        t = int(sequence)
        d = self.hidden_size
        projected = self.heads * self.value_dim
        features = self.heads * self.charts * self.basis
        local = sum(min(t, window) for window in self.local_windows)
        projections = 2 * t * d * (
            self.heads * self.charts + self.heads * self.basis
            + projected + 2 * self.heads * self.local_features
            + self.latent + self.heads * 3)
        communication = 4 * t * features * self.value_dim
        local_cost = 4 * t * local * self.heads * (
            self.local_features + self.value_dim)
        wavelet = 4 * t * d * self.latent
        output = 2 * t * projected * d
        return projections + communication + local_cost + wavelet + output


class AtlasV2Attention(nn.Module):
    def __init__(self, dense_attention: nn.Module, bank: AtlasV2Bank,
                 layer_index: int, layer_count: int, adapter_rank: int):
        super().__init__()
        self.dense_attention = dense_attention
        self._bank_reference = weakref.ref(bank)
        width = bank.hidden_size
        self.pre_adapter = ResidualCoordinateAdapter(width, adapter_rank)
        self.post_adapter = ResidualCoordinateAdapter(width, adapter_rank)
        self.layer_index = int(layer_index)
        self.layer_count = int(layer_count)
        self.register_buffer("functional_alpha", torch.tensor(0.1), persistent=True)

    def bank(self) -> AtlasV2Bank:
        bank = self._bank_reference()
        if bank is None:
            raise RuntimeError("Atlas V2 bank was released")
        return bank

    def set_alpha(self, value: float) -> None:
        self.functional_alpha.fill_(float(value))

    def forward(self, hidden_states: torch.Tensor, *args: Any, **kwargs: Any):
        depth = self.layer_index / max(self.layer_count - 1, 1)
        functional = self.bank()(self.pre_adapter(hidden_states), depth)
        functional = self.post_adapter(functional)
        alpha = float(self.functional_alpha.item())
        if alpha >= 1.0 - 1e-8:
            return functional, None
        dense_result = self.dense_attention(hidden_states, *args, **kwargs)
        dense = dense_result[0] if isinstance(dense_result, tuple) else dense_result
        return (1.0 - alpha) * dense + alpha * functional, None


class NormWithAdapter(nn.Module):
    def __init__(self, norm: nn.Module, width: int, rank: int):
        super().__init__()
        self.norm = norm
        self.adapter = ResidualCoordinateAdapter(width, rank)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.adapter(self.norm(hidden))


@dataclass
class AtlasV2Installed:
    banks: list[AtlasV2Bank]
    attentions: list[AtlasV2Attention]
    dense_attention_parameters: int
    deployable_parameters: int
    estimated_dense_flops: int
    estimated_operator_flops: int


def install_atlas_v2(model: nn.Module, config: dict, context: int) -> AtlasV2Installed:
    layers = list(model.model.layers)
    hidden = int(model.config.hidden_size)
    reference_parameter = next(model.parameters())
    model_device = reference_parameter.device
    model_dtype = reference_parameter.dtype
    groups = int(config["groups"])
    if not 1 <= groups <= len(layers):
        raise ValueError("groups must be between one and the model layer count")
    dense_parameters = sum(
        parameter.numel()
        for layer in layers for parameter in layer.self_attn.parameters())
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    banks = nn.ModuleList([
        AtlasV2Bank(
            hidden, int(config["heads"]), int(config["charts"]),
            int(config["basis"]), int(config["topk_charts"]),
            int(config["value_dim"]), int(config["local_features"]),
            [int(value) for value in config["local_windows"]],
            int(config["latent"]), int(config["wavelet_scales"]),
            int(config["depth_basis"]))
        for _ in range(groups)
    ]).to(device=model_device, dtype=model_dtype)
    model.add_module("_atlas_v2_banks", banks)
    attentions = []
    for index, layer in enumerate(layers):
        group = min(groups - 1, index * groups // len(layers))
        replacement = AtlasV2Attention(
            layer.self_attn, banks[group], index, len(layers),
            int(config["adapter_rank"])).to(
                device=model_device, dtype=model_dtype)
        layer.self_attn = replacement
        attentions.append(replacement)
        for norm in (layer.input_layernorm, layer.post_attention_layernorm):
            for parameter in norm.parameters():
                parameter.requires_grad_(True)

    model.model.norm = NormWithAdapter(
        model.model.norm, hidden, int(config["adapter_rank"])).to(
            device=model_device, dtype=model_dtype)
    for parameter in model.model.norm.parameters():
        parameter.requires_grad_(True)

    deployable_ids = set()
    deployable_parameters = 0
    modules: list[nn.Module] = list(banks)
    modules.extend(attention.pre_adapter for attention in attentions)
    modules.extend(attention.post_adapter for attention in attentions)
    modules.append(model.model.norm)
    for module in modules:
        for parameter in module.parameters():
            if parameter.requires_grad and id(parameter) not in deployable_ids:
                deployable_ids.add(id(parameter))
                deployable_parameters += parameter.numel()

    dense_flops_per_layer = int(
        6 * context * hidden * hidden + 4 * context * context * hidden)
    operator_flops = sum(
        banks[min(groups - 1, index * groups // len(layers))].estimated_flops(context)
        for index in range(len(layers)))
    adapter_flops = len(layers) * 8 * context * hidden * int(config["adapter_rank"])
    return AtlasV2Installed(
        banks=list(banks), attentions=attentions,
        dense_attention_parameters=dense_parameters,
        deployable_parameters=deployable_parameters,
        estimated_dense_flops=dense_flops_per_layer * len(layers),
        estimated_operator_flops=operator_flops + adapter_flops)


def set_functional_alpha(installed: AtlasV2Installed, value: float) -> None:
    for attention in installed.attentions:
        attention.set_alpha(value)


def clear_auxiliary(installed: AtlasV2Installed) -> None:
    for bank in installed.banks:
        bank.clear_auxiliary()


def auxiliary_losses(installed: AtlasV2Installed) -> dict[str, torch.Tensor]:
    values = [bank.auxiliary_losses() for bank in installed.banks]
    return {
        key: torch.stack([value[key] for value in values]).mean()
        for key in values[0]
    }
