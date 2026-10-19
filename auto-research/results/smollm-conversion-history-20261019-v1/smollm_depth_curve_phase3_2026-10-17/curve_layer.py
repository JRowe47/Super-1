"""Lossless cardinal-cubic coordinates for ONE Llama transformer block.

This changes parameter representation, not the block's computational graph.
Integer knots are exact; off-knot interpolation is a chosen extension with no
claim of learned semantic geometry. All seven projections and both RMSNorm
scales live in one scalar curve. Later blocks and embeddings are untouched.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import save_file, load_file


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(memoryview(tensor.detach().cpu().contiguous().numpy()).cast('B')).hexdigest()


class CardinalCurve(nn.Module):
    """Uniform Catmull-Rom (cubic Hermite) interpolant through scalar ordinates.

    Domain is [0, N-1]. At knot j, C(j)=ordinates[j] exactly. Slopes are
    (y[j+1]-y[j-1])/2 with clamped endpoint neighbors. Slopes are implicit,
    so no second coefficient array is stored. This is C1, not generally C2.
    Integer-domain evaluation avoids floating-point rescaling of knot indices.
    """
    def __init__(self, ordinates: torch.Tensor, layout: dict):
        super().__init__()
        self.register_buffer('ordinates', ordinates.detach().clone().contiguous())
        self.layout = copy.deepcopy(layout)

    def sample(self, coordinates: torch.Tensor) -> torch.Tensor:
        x = coordinates.to(device=self.ordinates.device, dtype=torch.float64)
        if not torch.isfinite(x).all() or (x < 0).any() or (x > self.ordinates.numel()-1).any():
            raise ValueError('Coordinates must be finite and inside the curve domain.')
        j = torch.floor(x).long()
        t = (x-j).to(self.ordinates.dtype)
        if bool((t == 0).all()):
            return self.ordinates[j]
        n = self.ordinates.numel()
        p0 = self.ordinates[(j-1).clamp(0,n-1)]
        p1 = self.ordinates[j]
        p2 = self.ordinates[(j+1).clamp(0,n-1)]
        p3 = self.ordinates[(j+2).clamp(0,n-1)]
        m1, m2 = (p2-p0)*0.5, (p3-p1)*0.5
        t2, t3 = t*t, t*t*t
        return (2*t3-3*t2+1)*p1 + (t3-2*t2+t)*m1 + (-2*t3+3*t2)*p2 + (t3-t2)*m2

    def tensor(self, name: str, start_row: int = 0, end_row: int | None = None,
               phase: float = 0.0) -> torch.Tensor:
        item = self.layout[name]
        shape = tuple(item['shape'])
        end_row = shape[0] if end_row is None else end_row
        stride = shape[1] if len(shape)==2 else 1
        start = item['offset'] + start_row*stride
        count = (end_row-start_row)*stride
        target_shape = (end_row-start_row, *shape[1:])
        if phase == 0:
            # Exact cardinal evaluation at a contiguous run of integer knots.
            # This is the simplified cubic basis at t=0, not a dense fallback.
            return self.ordinates.narrow(0, start, count).view(target_shape)
        coords = torch.arange(start, start+count, dtype=torch.float64, device=self.ordinates.device)+phase
        return self.sample(coords).view(target_shape)

    def save(self, folder: Path):
        folder.mkdir(parents=True, exist_ok=True)
        save_file({'ordinates': self.ordinates.cpu()}, str(folder/'first_block_curve.safetensors'))
        info = {'format': 'cardinal-cubic-first-block-v1', 'interpolant': 'uniform Catmull-Rom',
                'domain': [0, self.ordinates.numel()-1], 'dtype': str(self.ordinates.dtype),
                'layout': self.layout, 'ordinates_sha256': tensor_sha256(self.ordinates)}
        (folder/'curve_layout.json').write_text(json.dumps(info, indent=2)+'\n')

    @classmethod
    def load(cls, folder: Path):
        info = json.loads((folder/'curve_layout.json').read_text())
        ordinates = load_file(str(folder/'first_block_curve.safetensors'))['ordinates']
        if tensor_sha256(ordinates) != info['ordinates_sha256']:
            raise ValueError('Curve checksum mismatch')
        return cls(ordinates, info['layout'])


class CurveLinear(nn.Module):
    def __init__(self, curve: CardinalCurve, name: str, rows_per_chunk: int = 0):
        super().__init__()
        # The containing transformer block owns/registers the shared curve once.
        object.__setattr__(self, '_curve', curve)
        self.name, self.rows_per_chunk, self.phase = name, rows_per_chunk, 0.0
        self.out_features, self.in_features = curve.layout[name]['shape']

    @property
    def weight(self):
        return self._curve.tensor(self.name, phase=self.phase)

    def forward(self, hidden_states):
        if not self.rows_per_chunk:
            return F.linear(hidden_states, self.weight)
        pieces = [F.linear(hidden_states, self._curve.tensor(
            self.name, i, min(i+self.rows_per_chunk, self.out_features), phase=self.phase))
            for i in range(0, self.out_features, self.rows_per_chunk)]
        return torch.cat(pieces, dim=-1)


class CurveRMSNorm(nn.Module):
    def __init__(self, curve: CardinalCurve, name: str, eps: float):
        super().__init__()
        object.__setattr__(self, '_curve', curve)
        self.name, self.variance_epsilon = name, eps

    @property
    def weight(self):
        return self._curve.tensor(self.name)

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def encode_first_block(block: nn.Module) -> CardinalCurve:
    layout, arrays, offset = {}, [], 0
    for name, param in block.named_parameters():
        if not name.endswith('.weight') or param.ndim not in (1,2):
            raise ValueError(f'Unsupported parameter: {name} {tuple(param.shape)}')
        layout[name] = {'offset': offset, 'shape': list(param.shape), 'count': param.numel(),
                        'source_sha256': tensor_sha256(param)}
        arrays.append(param.detach().flatten())
        offset += param.numel()
    return CardinalCurve(torch.cat(arrays), layout)


def replace_first_block(model: nn.Module, curve: CardinalCurve, rows_per_chunk: int = 0):
    """Mutate ONLY model.model.layers[0]; preserve attention/MLP/residual code."""
    block = model.model.layers[0]
    if getattr(model.config, 'pretraining_tp', 1) != 1:
        raise ValueError('This prototype requires pretraining_tp=1.')
    for name in curve.layout:
        path = name.removesuffix('.weight')
        parent_path, _, attr = path.rpartition('.')
        parent = block.get_submodule(parent_path) if parent_path else block
        old = getattr(parent, attr)
        if isinstance(old, nn.Linear):
            if old.bias is not None:
                raise ValueError('Bias not implemented in this experiment.')
            new = CurveLinear(curve, name, rows_per_chunk)
        elif old.__class__.__name__ == 'LlamaRMSNorm':
            new = CurveRMSNorm(curve, name, old.variance_epsilon)
        else:
            raise ValueError(f'Unexpected module: {type(old)}')
        setattr(parent, attr, new)
    block.add_module('parameter_curve', curve)
    return model


def set_chunk_rows(model: nn.Module, rows: int):
    for module in model.model.layers[0].modules():
        if isinstance(module, CurveLinear):
            module.rows_per_chunk = rows
