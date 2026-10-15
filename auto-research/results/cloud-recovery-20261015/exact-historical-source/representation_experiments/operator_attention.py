"""Functional replacements for Qwen self-attention.

Every operator consumes and returns ``[batch, sequence, hidden]``.  The Qwen
adapter at the bottom preserves the decoder-layer calling contract while
intentionally ignoring Qwen's QKV projections and RoPE.  The four arms therefore
test complete communication-operator replacement, not a different softmax
approximation wrapped around the original projections.
"""
from __future__ import annotations

import math
import weakref
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _causal_positive_operator(query: torch.Tensor, key: torch.Tensor,
                              value: torch.Tensor) -> torch.Tensor:
    query = F.softplus(query.float()) + 1e-4
    key = F.softplus(key.float()) + 1e-4
    value_float = value.float()
    key_value = torch.einsum("btf,btd->btfd", key, value_float).cumsum(dim=1)
    key_sum = key.cumsum(dim=1)
    numerator = torch.einsum("btf,btfd->btd", query, key_value)
    denominator = torch.einsum("btf,btf->bt", query, key_sum).unsqueeze(-1)
    return (numerator / denominator.clamp_min(1e-5)).to(value.dtype)


def _local_exact_attention(query: torch.Tensor, key: torch.Tensor,
                           value: torch.Tensor, window: int) -> torch.Tensor:
    """Causal sliding-window attention computed in O(T * window), not O(T^2)."""
    if window < 1:
        return torch.zeros_like(value)
    batch, sequence, feature = query.shape
    width = min(window, sequence)
    key_pad = F.pad(key, (0, 0, width - 1, 0))
    value_pad = F.pad(value, (0, 0, width - 1, 0))
    key_windows = key_pad.unfold(1, width, 1).permute(0, 1, 3, 2)
    value_windows = value_pad.unfold(1, width, 1).permute(0, 1, 3, 2)
    scores = torch.einsum("btf,btwf->btw", query.float(), key_windows.float())
    scores = scores / math.sqrt(feature)
    positions = torch.arange(sequence, device=query.device).unsqueeze(1)
    offsets = torch.arange(width - 1, -1, -1, device=query.device).unsqueeze(0)
    valid = positions - offsets >= 0
    scores = scores.masked_fill(~valid.unsqueeze(0), torch.finfo(scores.dtype).min)
    probability = torch.softmax(scores, dim=-1).to(value.dtype)
    return torch.einsum("btw,btwd->btd", probability, value_windows)


def _causal_spectral_filter(hidden: torch.Tensor, real: torch.Tensor,
                            imag: torch.Tensor) -> torch.Tensor:
    """Causal convolution whose impulse response is parameterized spectrally."""
    batch, sequence, rank = hidden.shape
    modes = min(real.shape[0], sequence)
    # CUDA FFT does not accept BF16.  Autocast can cast the matrix product that
    # constructs the kernel back to BF16 even when its operands were promoted,
    # so keep the complete spectral path explicitly outside autocast.
    with torch.autocast(device_type=hidden.device.type, enabled=False):
        lag = torch.arange(sequence, device=hidden.device, dtype=torch.float32)
        mode = torch.arange(
            modes, device=hidden.device, dtype=torch.float32).unsqueeze(1)
        phase = 2.0 * math.pi * mode * lag.unsqueeze(0) / max(sequence, 1)
        kernel = (
            real[:modes].float().t() @ torch.cos(phase)
            - imag[:modes].float().t() @ torch.sin(phase)
        ) / math.sqrt(max(modes, 1))
        fft_size = 1 << max(1, (2 * sequence - 1).bit_length())
        signal_fft = torch.fft.rfft(
            hidden.float().transpose(1, 2), n=fft_size)
        kernel_fft = torch.fft.rfft(kernel.float(), n=fft_size)
        result = torch.fft.irfft(
            signal_fft * kernel_fft.unsqueeze(0), n=fft_size)
    return result[..., :sequence].transpose(1, 2).to(hidden.dtype)


def _causal_wavelet(hidden: torch.Tensor, gains: torch.Tensor) -> torch.Tensor:
    """Fixed Haar-like causal filters with learned channel/scale gains."""
    batch, sequence, rank = hidden.shape
    source = hidden.transpose(1, 2)
    result = torch.zeros_like(source)
    for scale_index in range(gains.shape[0]):
        half = 1 << scale_index
        width = 2 * half
        kernel = torch.cat((
            -torch.ones(half, device=hidden.device),
            torch.ones(half, device=hidden.device),
        )) / math.sqrt(width)
        weight = kernel.view(1, 1, width).repeat(rank, 1, 1)
        filtered = F.conv1d(F.pad(source, (width - 1, 0)), weight, groups=rank)
        result = result + filtered * gains[scale_index].view(1, rank, 1)
    return result.transpose(1, 2)


class FunctionalOperator(nn.Module):
    kind = "base"

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = int(hidden_size)

    def estimated_flops(self, sequence: int) -> int:
        raise NotImplementedError


class PositiveKernelOperator(FunctionalOperator):
    kind = "positive_kernel"

    def __init__(self, hidden_size: int, features: int):
        super().__init__(hidden_size)
        self.features = int(features)
        self.query = nn.Linear(hidden_size, features, bias=False)
        self.key = nn.Linear(hidden_size, features, bias=False)
        self.value = nn.Linear(hidden_size, hidden_size, bias=False)
        self.output = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden: torch.Tensor, depth: float = 0.0) -> torch.Tensor:
        del depth
        mixed = _causal_positive_operator(
            self.query(hidden), self.key(hidden), self.value(hidden))
        return self.output(mixed)

    def estimated_flops(self, sequence: int) -> int:
        d, f, t = self.hidden_size, self.features, int(sequence)
        return 4 * t * d * d + 4 * t * d * f


class FourierLocalOperator(FunctionalOperator):
    kind = "fourier_local"

    def __init__(self, hidden_size: int, rank: int, local_features: int,
                 local_window: int, modes: int):
        super().__init__(hidden_size)
        self.rank = int(rank)
        self.local_features = int(local_features)
        self.local_window = int(local_window)
        self.modes = int(modes)
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=False)
        self.query = nn.Linear(hidden_size, local_features, bias=False)
        self.key = nn.Linear(hidden_size, local_features, bias=False)
        self.value = nn.Linear(hidden_size, hidden_size, bias=False)
        self.output = nn.Linear(hidden_size, hidden_size, bias=False)
        self.spectral_real = nn.Parameter(torch.randn(modes, rank) * 0.02)
        self.spectral_imag = nn.Parameter(torch.randn(modes, rank) * 0.02)
        self.mix = nn.Parameter(torch.tensor([0.5, 0.5]))

    def forward(self, hidden: torch.Tensor, depth: float = 0.0) -> torch.Tensor:
        del depth
        spectral = self.up(_causal_spectral_filter(
            self.down(hidden), self.spectral_real, self.spectral_imag))
        local = _local_exact_attention(
            self.query(hidden), self.key(hidden), self.value(hidden),
            self.local_window)
        weights = torch.softmax(self.mix, dim=0)
        return self.output(weights[0] * spectral + weights[1] * local)

    def estimated_flops(self, sequence: int) -> int:
        d, r, f, t, w = (self.hidden_size, self.rank, self.local_features,
                         int(sequence), self.local_window)
        return 4 * t * d * d + 4 * t * d * (r + f) + 4 * t * w * (d + f)


class WaveletRBFOperator(FunctionalOperator):
    kind = "wavelet_rbf"

    def __init__(self, hidden_size: int, latent: int, charts: int,
                 wavelet_scales: int):
        super().__init__(hidden_size)
        self.latent = int(latent)
        self.charts = int(charts)
        self.wavelet_scales = int(wavelet_scales)
        self.coordinates = nn.Linear(hidden_size, latent, bias=False)
        self.centers = nn.Parameter(torch.randn(charts, latent) / math.sqrt(latent))
        self.log_bandwidth = nn.Parameter(torch.zeros(charts))
        self.value = nn.Linear(hidden_size, hidden_size, bias=False)
        self.wavelet_down = nn.Linear(hidden_size, latent, bias=False)
        self.wavelet_up = nn.Linear(latent, hidden_size, bias=False)
        self.wavelet_gains = nn.Parameter(torch.zeros(wavelet_scales, latent))
        self.output = nn.Linear(hidden_size, hidden_size, bias=False)
        self.mix = nn.Parameter(torch.tensor([0.5, 0.5]))

    def features(self, hidden: torch.Tensor) -> torch.Tensor:
        coordinate = self.coordinates(hidden).float()
        distance = (
            coordinate.unsqueeze(-2) - self.centers.float().view(1, 1, self.charts, self.latent)
        ).square().mean(dim=-1)
        bandwidth = self.log_bandwidth.float().exp().view(1, 1, self.charts)
        return torch.exp(-distance / bandwidth.clamp_min(1e-4)).to(hidden.dtype)

    def forward(self, hidden: torch.Tensor, depth: float = 0.0) -> torch.Tensor:
        del depth
        feature = self.features(hidden)
        global_message = _causal_positive_operator(
            feature, feature, self.value(hidden))
        wavelet = self.wavelet_up(_causal_wavelet(
            self.wavelet_down(hidden), self.wavelet_gains))
        weights = torch.softmax(self.mix, dim=0)
        return self.output(weights[0] * global_message + weights[1] * wavelet)

    def estimated_flops(self, sequence: int) -> int:
        d, r, c, t = self.hidden_size, self.latent, self.charts, int(sequence)
        return 4 * t * d * d + 4 * t * d * r + 4 * t * c * (d + r)


class AtlasOperator(FunctionalOperator):
    kind = "atlas"

    def __init__(self, hidden_size: int, charts: int, basis: int, latent: int,
                 local_features: int, local_window: int, wavelet_scales: int,
                 depth_basis: int):
        super().__init__(hidden_size)
        self.charts = int(charts)
        self.basis = int(basis)
        self.latent = int(latent)
        self.local_features = int(local_features)
        self.local_window = int(local_window)
        self.wavelet_scales = int(wavelet_scales)
        self.depth_basis = int(depth_basis)
        self.chart_gate = nn.Linear(hidden_size, charts, bias=False)
        self.basis_feature = nn.Linear(hidden_size, basis, bias=False)
        self.value = nn.Linear(hidden_size, hidden_size, bias=False)
        self.query = nn.Linear(hidden_size, local_features, bias=False)
        self.key = nn.Linear(hidden_size, local_features, bias=False)
        self.wavelet_down = nn.Linear(hidden_size, latent, bias=False)
        self.wavelet_up = nn.Linear(latent, hidden_size, bias=False)
        self.wavelet_gains = nn.Parameter(torch.zeros(wavelet_scales, latent))
        self.output = nn.Linear(hidden_size, hidden_size, bias=False)
        self.depth_gain = nn.Linear(depth_basis, charts * basis, bias=False)
        self.mix = nn.Parameter(torch.tensor([0.34, 0.33, 0.33]))

    def depth_features(self, depth: float, device: torch.device) -> torch.Tensor:
        depth_tensor = torch.tensor(float(depth), device=device)
        orders = torch.arange(self.depth_basis, device=device, dtype=torch.float32)
        return torch.cos(math.pi * orders * depth_tensor)

    def forward(self, hidden: torch.Tensor, depth: float = 0.0) -> torch.Tensor:
        chart = torch.softmax(self.chart_gate(hidden).float(), dim=-1)
        basis = F.softplus(self.basis_feature(hidden).float())
        feature = (chart.unsqueeze(-1) * basis.unsqueeze(-2)).flatten(-2)
        gain = self.depth_gain(self.depth_features(depth, hidden.device))
        feature = feature * torch.sigmoid(gain).view(1, 1, -1) * 2.0
        value = self.value(hidden)
        global_message = _causal_positive_operator(
            feature.to(hidden.dtype), feature.to(hidden.dtype), value)
        local_message = _local_exact_attention(
            self.query(hidden), self.key(hidden), value,
            self.local_window)
        wavelet_message = self.wavelet_up(_causal_wavelet(
            self.wavelet_down(hidden), self.wavelet_gains))
        weights = torch.softmax(self.mix, dim=0)
        return self.output(
            weights[0] * global_message
            + weights[1] * local_message
            + weights[2] * wavelet_message)

    def estimated_flops(self, sequence: int) -> int:
        d, c, r, z, f, t, w = (
            self.hidden_size, self.charts, self.basis, self.latent,
            self.local_features, int(sequence), self.local_window)
        feature = c * r
        return (4 * t * d * d + 4 * t * d * (z + f + c + r)
                + 4 * t * feature * d + 4 * t * w * (d + f))


class QwenOperatorAttention(nn.Module):
    """Adapter from a functional operator to the Qwen attention call contract."""

    def __init__(self, operator: FunctionalOperator, layer_index: int,
                 layer_count: int, register_operator: bool = True):
        super().__init__()
        if register_operator:
            self.operator = operator
            self._operator_reference = None
        else:
            self._operator_reference = weakref.ref(operator)
        self.layer_index = int(layer_index)
        self.layer_count = int(layer_count)

    def _operator(self) -> FunctionalOperator:
        if self._operator_reference is None:
            return self.operator
        operator = self._operator_reference()
        if operator is None:
            raise RuntimeError("shared Atlas operator was released")
        return operator

    def forward(self, hidden_states: torch.Tensor, *args: Any, **kwargs: Any):
        del args, kwargs
        depth = self.layer_index / max(self.layer_count - 1, 1)
        return self._operator()(hidden_states, depth=depth), None


@dataclass
class InstalledOperators:
    modules: list[FunctionalOperator]
    dense_attention_parameters: int
    operator_parameters: int
    estimated_dense_flops: int
    estimated_operator_flops: int


def _make_operator(kind: str, hidden_size: int, config: dict) -> FunctionalOperator:
    if kind == "positive_kernel":
        return PositiveKernelOperator(hidden_size, config["features"])
    if kind == "fourier_local":
        return FourierLocalOperator(
            hidden_size, config["rank"], config["local_features"],
            config["local_window"], config["modes"])
    if kind == "wavelet_rbf":
        return WaveletRBFOperator(
            hidden_size, config["latent"], config["charts"],
            config["wavelet_scales"])
    if kind == "atlas":
        return AtlasOperator(
            hidden_size, config["charts"], config["basis"], config["latent"],
            config["local_features"], config["local_window"],
            config["wavelet_scales"], config["depth_basis"])
    raise ValueError(f"unknown operator kind {kind}")


def install_qwen_operators(model: nn.Module, kind: str, config: dict,
                           context: int) -> InstalledOperators:
    layers = list(model.model.layers)
    hidden = int(model.config.hidden_size)
    dense_parameters = sum(
        parameter.numel()
        for layer in layers for parameter in layer.self_attn.parameters())
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    modules: list[FunctionalOperator] = []
    if kind == "atlas":
        shared = _make_operator(kind, hidden, config)
        model.add_module("_functional_attention_bank", shared)
        modules.append(shared)
        for index, layer in enumerate(layers):
            layer.self_attn = QwenOperatorAttention(
                shared, index, len(layers), register_operator=False)
    else:
        for index, layer in enumerate(layers):
            operator = _make_operator(kind, hidden, config)
            modules.append(operator)
            layer.self_attn = QwenOperatorAttention(operator, index, len(layers))

    operator_parameters = sum(
        parameter.numel() for module in modules for parameter in module.parameters())
    dense_flops_per_layer = int(6 * context * hidden * hidden
                                + 4 * context * context * hidden)
    operator_flops = (
        modules[0].estimated_flops(context) * len(layers)
        if kind == "atlas" else
        sum(module.estimated_flops(context) for module in modules)
    )
    return InstalledOperators(
        modules=modules,
        dense_attention_parameters=dense_parameters,
        operator_parameters=operator_parameters,
        estimated_dense_flops=dense_flops_per_layer * len(layers),
        estimated_operator_flops=operator_flops,
    )
