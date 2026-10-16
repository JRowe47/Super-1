"""Isolated angular versus radial fast-state modulation; native recurrence retained."""
import torch
from torch import nn

class FastModulation(nn.Module):
    def __init__(self, native, kind, epsilon):
        super().__init__()
        self.native, self.kind, self.epsilon = native, kind, epsilon
        self.hidden_size, self.input_size = native.hidden_size, native.input_size
        assert self.hidden_size % 2 == 0 and self.input_size == 3 * self.hidden_size
        assert kind in ('rotation', 'radial') and epsilon > 0
        self.phase = nn.Parameter(torch.zeros(self.hidden_size // 2))
        self.gain = nn.Parameter(torch.zeros(self.hidden_size // 2))

    def forward(self, inputs, previous):
        proposal = self.native(inputs, previous)
        width = self.hidden_size
        context = inputs[..., width:2*width] + inputs[..., 2*width:3*width]
        p = proposal.reshape(*proposal.shape[:-1], -1, 2)
        q = context.reshape(*context.shape[:-1], -1, 2)
        alignment = (p*q).sum(-1) / torch.sqrt(p.square().sum(-1)*q.square().sum(-1) + self.epsilon)
        amount = self.phase + torch.tanh(self.gain)*torch.tanh(alignment)
        if self.kind == 'radial':
            return ((1 + torch.tanh(amount)).unsqueeze(-1)*p).flatten(-2)
        c, s = amount.cos(), amount.sin()
        return torch.stack((c*p[..., 0]-s*p[..., 1], s*p[..., 0]+c*p[..., 1]), -1).flatten(-2)

def install_fast_rotation(model, kind, epsilon):
    model.fast_cell = FastModulation(model.fast_cell, kind, epsilon)
    return model
