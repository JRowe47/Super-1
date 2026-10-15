"""Learn a reference phase for relative state/input geometric queries.

This is a mechanism experiment, not a novelty or stability claim. A zero key
recovers the existing normalized-cross query exactly. Learned nonzero keys
can distinguish parallel and antiparallel vectors through their dot product.
"""
import json

import torch
from torch import nn
import noncommuting_phase_candidate as source


class RelativePhaseKeyLM(source.OverlappingPhaseLM):
    def __init__(self, width, dynamic=True):
        super().__init__(width, dynamic)
        self.phase_key = nn.Parameter(torch.zeros(width // 2))
        self.second_phase_key = nn.Parameter(torch.zeros(width // 2))

    def angle_for(self, state, drive, angle, gain, offset):
        if gain is None:
            return angle
        pair = torch.roll(state, shifts=offset, dims=-1).reshape(len(state), -1, 2)
        target = torch.roll(drive, shifts=offset, dims=-1).reshape(len(state), -1, 2)
        cross = pair[..., 0] * target[..., 1] - pair[..., 1] * target[..., 0]
        dot = (pair * target).sum(-1)
        denominator = torch.sqrt(pair.square().sum(-1) * target.square().sum(-1) + 1e-6)
        key = self.phase_key if offset == 0 else self.second_phase_key
        query = (cross * key.cos() + dot * key.sin()) / denominator
        return angle + gain * query.tanh()


def checks():
    torch.set_num_threads(1)
    torch.manual_seed(1307)
    base = source.OverlappingPhaseLM(16, True).double()
    with torch.no_grad():
        base.gain.fill_(0.4)
        base.second_gain.fill_(0.3)
        base.second_angle.fill_(0.2)
    model = RelativePhaseKeyLM(16).double()
    missing = model.load_state_dict(base.state_dict(), strict=False).missing_keys
    assert set(missing) == {'phase_key', 'second_phase_key'}
    tokens = torch.randint(256, (2, 9))
    expected, actual = base(tokens), model(tokens)
    assert torch.equal(expected, actual), 'Zero key must exactly recover prior geometry'
    expected.square().sum().backward()
    actual.square().sum().backward()
    error = max((parameter.grad-dict(model.named_parameters())[name].grad).abs().max().item()
                for name, parameter in base.named_parameters())
    assert error < 1e-12
    assert all(torch.isfinite(key.grad).all() and key.grad.abs().sum() > 0
               for key in [model.phase_key, model.second_phase_key])
    with torch.no_grad():
        model.phase_key.fill_(0.7)
        model.second_phase_key.fill_(-0.4)
    state = torch.randn(2, 16, dtype=torch.float64)
    drive = torch.randn_like(state)
    theta = model.angle_for(state, drive, model.angle, model.gain, 0)
    rotated = source.rotate(state, theta, 0)
    assert torch.allclose(rotated.square().sum(-1), state.square().sum(-1), atol=1e-12)
    # Common rotation leaves relative cross/dot geometry unchanged.
    common = torch.randn(8, dtype=torch.float64)
    invariant = model.angle_for(source.rotate(state, common, 0), source.rotate(drive, common, 0),
                                model.angle, model.gain, 0)
    assert torch.allclose(theta, invariant, atol=1e-12, rtol=1e-12)
    aligned = model.angle_for(state, state, model.angle, model.gain, 0)
    antiparallel = model.angle_for(state, -state, model.angle, model.gain, 0)
    assert not torch.allclose(aligned, antiparallel)
    changed = tokens.clone()
    changed[:, 4:] = torch.randint(256, (2, 5))
    assert torch.equal(model(tokens)[:, :4], model(changed)[:, :4])
    zeros = torch.zeros_like(state, requires_grad=True)
    model.angle_for(zeros, zeros, model.angle, model.gain, 0).sum().backward()
    assert torch.isfinite(zeros.grad).all()
    return {'zero_key_forward_exact': True, 'native_gradient_max_error': error,
            'both_key_gradients_nonzero': True, 'aligned_antiparallel_distinguished': True,
            'common_pair_rotation_invariance': True, 'transport_norm_preserved': True,
            'zero_state_gradients_finite': True, 'causal_prefix_exact': True,
            'novelty_established': False, 'trained': False,
            'limitation': 'State-dependent query can expand the Jacobian; norm preservation does not imply contraction. Not the original hierarchy.'}


if __name__ == '__main__':
    print(json.dumps(checks()))
