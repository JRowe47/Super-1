"""Synthetic identity-safe cross-scale function-weight test, not a benchmark."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'experiments/function_weight_wt2_original_multiscale_decisive/models.py'
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == '3dbe4fa50a4e7e0a9f1d1dec72b8e209e66ae842cb164e00d93cb4fe56d854a8'
spec = importlib.util.spec_from_file_location('rotation_reference', SOURCE)
reference = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = reference
spec.loader.exec_module(reference)

class CrossScaleRotation(nn.Module):
    def __init__(self, base, cross_scale=True):
        super().__init__()
        if base.width % 2 or base.state_scales < 2:
            raise ValueError('Even width and at least two scales required')
        self.base = base
        self.cross_scale = cross_scale
        self.rotation_gain = nn.Parameter(torch.zeros(base.state_scales))

    def forward(self, tokens, state=None, return_state=False):
        if state is None:
            state = self.base.initial_state(tokens.shape[0])
        outputs = []
        for index in range(tokens.shape[1]):
            # Use the preserved original transition without rewriting its gates,
            # learned timescales, recurrent matrices or global coupling.
            _, state = self.base(tokens[:, index:index+1], state, return_state=True)
            pairs = state.reshape(*state.shape[:-1], -1, 2)
            other = pairs.roll(1, dims=1) if self.cross_scale else pairs
            cross = pairs[..., 0] * other[..., 1] - pairs[..., 1] * other[..., 0]
            dot = (pairs * other).sum(-1)
            if self.cross_scale:
                query = cross / torch.sqrt(pairs.square().sum(-1) * other.square().sum(-1) + 1e-12)
            else:
                # Same parameter count, local content-dependent angle control.
                query = dot / (1 + dot)
            angle = self.rotation_gain[None, :, None] * torch.tanh(query)
            c, s = torch.cos(angle), torch.sin(angle)
            state = torch.stack([c*pairs[..., 0]-s*pairs[..., 1], s*pairs[..., 0]+c*pairs[..., 1]], -1).flatten(-2)
            outputs.append(self.base.output(self.base.core.norm(state.mean(1))))
        logits = torch.stack(outputs, 1)
        return (logits, state) if return_state else logits

def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(20261003)
    base = reference.PersistentMultiScaleLM(37, 16, 4)
    wrapped_base = reference.PersistentMultiScaleLM(37, 16, 4)
    wrapped_base.load_state_dict(base.state_dict())
    wrapped = CrossScaleRotation(wrapped_base)
    tokens = torch.randint(37, (2, 32))
    a, b = base(tokens), wrapped(tokens)
    identity_drift = (a-b).abs().max().item()
    assert torch.allclose(a, b, atol=2e-6, rtol=2e-6)
    a.square().mean().backward()
    b.square().mean().backward()
    inherited = {name: (param.grad-dict(wrapped.base.named_parameters())[name].grad).abs().max().item() for name, param in base.named_parameters()}
    assert max(inherited.values()) < 2e-6
    gain_grad = wrapped.rotation_gain.grad.detach().tolist()
    assert torch.isfinite(wrapped.rotation_gain.grad).all() and wrapped.rotation_gain.grad.abs().sum() > 0
    optimizer = torch.optim.AdamW(wrapped.parameters(), lr=0.001, foreach=False)
    for _ in range(3):
        optimizer.zero_grad()
        wrapped(tokens).square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in wrapped.parameters())
        optimizer.step()
    with torch.no_grad():
        whole, final = wrapped(tokens, return_state=True)
        first, state = wrapped(tokens[:, :11], return_state=True)
        last, carried = wrapped(tokens[:, 11:], state, return_state=True)
        assert torch.equal(whole, torch.cat([first, last], 1)) and torch.equal(final, carried)
        assert torch.equal(whole[:, :11], wrapped(tokens[:, :11]))
        alone = wrapped(tokens[:1])
        assert torch.allclose(alone, whole[:1], atol=2e-6, rtol=2e-6)
    report = dict(identity_logits_max_drift=identity_drift, inherited_gradient_max_drifts=inherited, initial_gain_gradient=gain_grad, joint_synthetic_updates=3, chunk_carry_exact=True, causal_prefix_exact=True, added_parameters=4, added_fp32_parameter_bytes=16, task_data_reads=0, official_test_reads=0, scope='Structural canary only; token-wise original transition preserved, not fused, no accuracy or novelty claim', design='Neighbor-scale oriented pair angle generates norm-preserving state rotations after each original transition; zero gain is mathematical identity', caveat='Token-wise GEMM changes FP32 operation ordering. Pair norm preservation does not guarantee stability of the entire recurrent system')
    with (ROOT / 'CROSS_SCALE_ROTATION_CANARY.json').open('x') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
