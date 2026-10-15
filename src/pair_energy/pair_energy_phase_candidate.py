"""Identity-initialized relative pair-energy signal for geometric transport.

A mechanism proposal, not a novelty, stability, or benchmark claim.
The existing oriented cross query remains; pair-energy contrast is learned
inside the same angle function, without a materialized recurrent matrix.
"""
import json

import torch
from torch import nn
import noncommuting_phase_candidate as source


class PairEnergyPhaseLM(source.OverlappingPhaseLM):
    def __init__(self, width, dynamic=True):
        super().__init__(width, dynamic)
        self.energy_gain = nn.Parameter(torch.zeros(width // 2))
        self.second_energy_gain = nn.Parameter(torch.zeros(width // 2))

    def angle_for(self, state, drive, angle, gain, offset):
        if gain is None:
            return angle
        pair = torch.roll(state, shifts=offset, dims=-1).reshape(len(state), -1, 2)
        target = torch.roll(drive, shifts=offset, dims=-1).reshape(len(state), -1, 2)
        state_energy = pair.square().sum(-1)
        drive_energy = target.square().sum(-1)
        cross = pair[..., 0] * target[..., 1] - pair[..., 1] * target[..., 0]
        denominator = torch.sqrt(state_energy * drive_energy + 1e-6)
        contrast = (torch.log(state_energy + 1e-6) - torch.log(drive_energy + 1e-6)).tanh()
        weight = self.energy_gain if offset == 0 else self.second_energy_gain
        return angle + gain * (cross / denominator + weight * contrast).tanh()


def checks():
    torch.set_num_threads(1)
    torch.manual_seed(3313)
    base = source.OverlappingPhaseLM(16, True).double()
    with torch.no_grad():
        base.gain.fill_(.4)
        base.second_gain.fill_(.3)
        base.second_angle.fill_(.2)
    model = PairEnergyPhaseLM(16).double()
    missing = model.load_state_dict(base.state_dict(), strict=False).missing_keys
    assert set(missing) == {'energy_gain', 'second_energy_gain'}
    tokens = torch.randint(256, (2, 9))
    expected, actual = base(tokens), model(tokens)
    assert torch.equal(expected, actual)
    expected.square().sum().backward()
    actual.square().sum().backward()
    error = max((parameter.grad - dict(model.named_parameters())[name].grad).abs().max().item()
                for name, parameter in base.named_parameters())
    assert error < 1e-12
    assert all(torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
               for p in [model.energy_gain, model.second_energy_gain])
    with torch.no_grad():
        model.energy_gain.fill_(.5)
        model.second_energy_gain.fill_(-.3)
    state = torch.randn(2, 16, dtype=torch.float64)
    drive = torch.randn_like(state)
    theta = model.angle_for(state, drive, model.angle, model.gain, 0)
    common = torch.randn(8, dtype=torch.float64)
    invariant = model.angle_for(source.rotate(state, common, 0), source.rotate(drive, common, 0),
                                model.angle, model.gain, 0)
    assert torch.allclose(theta, invariant, atol=1e-12, rtol=1e-12)
    transported = source.rotate(state, theta, 0)
    assert torch.allclose(transported.square().sum(-1), state.square().sum(-1), atol=1e-12)
    # Parallel pairs have zero cross product but different relative energy.
    low = model.angle_for(state, state * .25, model.angle, model.gain, 0)
    high = model.angle_for(state, state * 4, model.angle, model.gain, 0)
    assert not torch.allclose(low, high)
    changed = tokens.clone()
    changed[:, 4:] = torch.randint(256, (2, 5))
    assert torch.equal(model(tokens)[:, :4], model(changed)[:, :4])
    zeros = torch.zeros_like(state, requires_grad=True)
    model.angle_for(zeros, zeros, model.angle, model.gain, 0).sum().backward()
    assert torch.isfinite(zeros.grad).all()
    return {'zero_energy_gain_forward_exact': True, 'native_gradient_max_error': error,
            'both_energy_gradients_nonzero': True, 'relative_energy_distinguished': True,
            'common_pair_rotation_invariance': True, 'transport_norm_preserved': True,
            'causal_prefix_exact': True, 'zero_state_gradients_finite': True,
            'trained': False, 'novelty_established': False,
            'limitations': 'No extra independent information or contraction guarantee. Not the original hierarchy. Performance requires a fresh registered comparison.'}


if __name__ == '__main__':
    print(json.dumps(checks()))
