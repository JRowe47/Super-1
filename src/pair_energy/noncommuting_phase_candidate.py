"""Exploratory overlapping-plane rotations; not a claim of research novelty.

Independent pair rotations cannot exchange information between pairs. A second
rotation bank uses shifted pairs, producing order-sensitive transport without
materializing a dense recurrent matrix. Its identity initialization preserves
the prior PhaseLM exactly. No training experiment is launched by this file.
"""
import json
import torch
from torch import nn
import phase_transition_language as pilot


def rotate(state, theta, offset):
    shifted = torch.roll(state, shifts=offset, dims=-1)
    pair = shifted.reshape(len(state), -1, 2)
    cosine, sine = theta.cos(), theta.sin()
    transformed = torch.stack((cosine * pair[..., 0] - sine * pair[..., 1],
                               sine * pair[..., 0] + cosine * pair[..., 1]), -1).flatten(1)
    return torch.roll(transformed, shifts=-offset, dims=-1)


class OverlappingPhaseLM(pilot.PhaseLM):
    def __init__(self, width, dynamic):
        super().__init__(width, dynamic)
        self.second_angle = nn.Parameter(torch.zeros(width // 2))
        self.second_gain = nn.Parameter(torch.zeros(width // 2)) if dynamic else None

    def angle_for(self, state, drive, angle, gain, offset):
        if gain is None:
            return angle
        pair = torch.roll(state, shifts=offset, dims=-1).reshape(len(state), -1, 2)
        target = torch.roll(drive, shifts=offset, dims=-1).reshape(len(state), -1, 2)
        cross = pair[..., 0] * target[..., 1] - pair[..., 1] * target[..., 0]
        denom = torch.sqrt(pair.square().sum(-1) * target.square().sum(-1) + 1e-6)
        return angle + gain * torch.tanh(cross / denom)

    def forward(self, tokens):
        embedded = self.embedding(tokens)
        drive = torch.tanh(self.drive(embedded))
        retention = torch.sigmoid(self.write(embedded) + self.retention)
        state = embedded.new_zeros((len(tokens), self.d))
        history = []
        for step in range(tokens.shape[1]):
            theta = self.angle_for(state, drive[:, step], self.angle, self.gain, 0)
            transported = rotate(state, theta, 0)
            theta2 = self.angle_for(transported, drive[:, step], self.second_angle, self.second_gain, 1)
            transported = rotate(transported, theta2, 1)
            state = retention[:, step] * transported + (1 - retention[:, step]) * drive[:, step]
            history.append(state)
        return self.norm(torch.stack(history, 1)) @ self.embedding.weight.T


def checks():
    torch.manual_seed(41)
    base = pilot.PhaseLM(16, True).double()
    with torch.no_grad():
        base.gain.fill_(0.4)
    candidate = OverlappingPhaseLM(16, True).double()
    missing = candidate.load_state_dict(base.state_dict(), strict=False).missing_keys
    assert set(missing) == {'second_angle', 'second_gain'}
    tokens = torch.randint(256, (2, 8))
    assert torch.equal(base(tokens), candidate(tokens)), 'Second bank must start exactly at identity'
    state = torch.randn(2, 16, dtype=torch.float64)
    angle = torch.randn(8, dtype=torch.float64)
    second = torch.randn(8, dtype=torch.float64)
    ab = rotate(rotate(state, angle, 0), second, 1)
    ba = rotate(rotate(state, second, 1), angle, 0)
    assert not torch.allclose(ab, ba)
    assert torch.allclose(state.square().sum(-1), ab.square().sum(-1), atol=1e-12)
    candidate(tokens).square().sum().backward()
    assert torch.isfinite(candidate.second_gain.grad).all()
    assert candidate.second_gain.grad.abs().sum() > 0
    with torch.no_grad():
        candidate.second_angle.fill_(0.3)
        candidate.second_gain.fill_(0.2)
    changed = tokens.clone()
    changed[:, 4:] = torch.randint(256, (2, 4))
    assert torch.equal(candidate(tokens)[:, :4], candidate(changed)[:, :4])
    return {'identity_extension_exact': True, 'noncommuting_banks': True,
            'transport_norm_preserved': True, 'second_gain_gradient_nonzero': True,
            'causal_prefix': True, 'parameters': pilot.count(candidate),
            'trained': False, 'novelty_established': False}


if __name__ == '__main__':
    torch.set_num_threads(1)
    print(json.dumps(checks()))
