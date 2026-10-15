"""Packed execution of the separately tested cross-scale rotation candidate.

This remains a sequential nonlinear recurrence, not a fused/parallel scan.
"""
import importlib.util
from pathlib import Path
import sys
import torch
from torch.nn import functional as F

source = Path(__file__).resolve().with_name('cross_scale_rotation_canary.py')
spec = importlib.util.spec_from_file_location('rotation_slow_reference', source)
reference = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = reference
spec.loader.exec_module(reference)

class PackedRotation(reference.CrossScaleRotation):
    def forward(self, tokens, state=None, return_state=False):
        core = self.base.core
        embedded = core.embedding(tokens)
        if state is None:
            state = self.base.initial_state(tokens.shape[0])
        self.base._validate_state(state, embedded)
        input_terms = F.linear(embedded, torch.cat([core.input_update.weight, core.input_gate.weight]), torch.cat([core.input_update.bias, core.input_gate.bias]))
        inputs_update, inputs_gate = input_terms.split(self.base.width, -1)
        recurrent_weight = torch.cat([core.state_update.weight, core.state_gate.weight])
        scale = core.scale[None]
        means = []
        for index in range(tokens.shape[1]):
            recurrent_update, recurrent_gate = F.linear(state, recurrent_weight).split(self.base.width, -1)
            coupled = core.coupling(state.mean(1, keepdim=True))
            update = torch.tanh(inputs_update[:, index, None] + recurrent_update + coupled + scale)
            gate = torch.sigmoid(inputs_gate[:, index, None] + recurrent_gate - scale)
            state = gate * state + (1-gate) * update
            pairs = state.reshape(*state.shape[:-1], -1, 2)
            other = pairs.roll(1, dims=1) if self.cross_scale else pairs
            cross = pairs[..., 0]*other[..., 1] - pairs[..., 1]*other[..., 0]
            dot = (pairs*other).sum(-1)
            query = cross / torch.sqrt(pairs.square().sum(-1)*other.square().sum(-1)+1e-12) if self.cross_scale else dot/(1+dot)
            angle = self.rotation_gain[None, :, None]*torch.tanh(query)
            c, s = torch.cos(angle), torch.sin(angle)
            state = torch.stack([c*pairs[..., 0]-s*pairs[..., 1], s*pairs[..., 0]+c*pairs[..., 1]], -1).flatten(-2)
            means.append(state.mean(1))
        logits = self.base.output(core.norm(torch.stack(means, 1)))
        return (logits, state) if return_state else logits
