"""Small isolated mechanism test, not a reproduction of the Super-1 hierarchy.

The base uses linear-minGRU interpolation plus a common token-conditioned
pair rotation. Adaptive coefficients change update logits and rotation angles.
The parameter/state-matched control inserts the same fields into candidates.
The coefficient updater is learned but fixed at inference: full self-modifying
updater weights are NOT implemented in this first experiment.
"""
import math
import torch
from torch import nn

ARMS = ('fixed48', 'wide64', 'context64', 'adaptive64', 'gru64')
VOCAB = 52

def cosine_basis(points, modes=8, dtype=torch.float32):
    s = (torch.arange(points, dtype=dtype) + .5) / points
    k = torch.arange(modes, dtype=dtype)
    basis = torch.cos(math.pi * s[:, None] * k[None])
    basis[:, 1:] *= math.sqrt(2)
    return basis / math.sqrt(modes)

class CurveCell(nn.Module):
    def __init__(self, arm, seed=0, chunk=8):
        super().__init__()
        assert arm in ARMS
        torch.manual_seed(seed)
        self.arm, self.chunk = arm, chunk
        self.width = 64 if arm in ('wide64', 'gru64') else 48
        self.has_p = arm in ('context64', 'adaptive64')
        self.embed = nn.Embedding(VOCAB, 16)
        nn.init.normal_(self.embed.weight, std=.3)
        h = self.width
        self.read = nn.Sequential(nn.Linear(h + 16, 64), nn.Tanh(), nn.Linear(64, 8))
        if arm == 'gru64':
            self.gru = nn.GRUCell(16, h)
        else:
            self.proj = nn.Linear(16, 2*h + h//2)
            with torch.no_grad():
                nn.init.xavier_uniform_(self.proj.weight)
                self.proj.bias.zero_()
                self.proj.bias[h:2*h].fill_(-2.5)
                self.proj.weight[2*h:].zero_()
            self.register_buffer('basis', cosine_basis(h//2))
        if self.has_p:
            self.updater = nn.Linear(16 + h + 16, 32)
            with torch.no_grad():
                self.updater.weight.zero_()
                self.updater.bias.zero_()
                self.updater.bias[16:].fill_(math.log(.1/.9))

    def initial(self, batch):
        w = self.embed.weight
        return (w.new_zeros(batch, self.width), w.new_zeros(batch, 16 if self.has_p else 0), 0)

    def step(self, token, state, intervention='none'):
        h, p, clock = state
        x = self.embed(token)
        if self.arm == 'gru64':
            h = self.gru(x, h)
        else:
            raw = self.proj(x)
            v, gate, angle = torch.split(raw, [self.width, self.width, self.width//2], -1)
            if self.has_p:
                active = torch.zeros_like(p) if intervention == 'zero_p' else p
                fields = active.reshape(-1, 2, 8) @ self.basis.T
                if self.arm == 'adaptive64':
                    if intervention != 'no_gate':
                        gate = gate + 2*fields[:, 0].repeat_interleave(2, -1)
                    if intervention != 'no_mix':
                        angle = angle + fields[:, 1]
                else:
                    v = v + fields.transpose(1, 2).reshape(-1, self.width)
            theta = .25*torch.tanh(angle)
            cs, sn = torch.cos(theta), torch.sin(theta)
            pair = h.reshape(-1, self.width//2, 2)
            moved = torch.stack((cs*pair[..., 0]-sn*pair[..., 1],
                                 sn*pair[..., 0]+cs*pair[..., 1]), -1).flatten(1)
            z = torch.sigmoid(gate)
            h = (1-z)*moved + z*v
        clock += 1
        if self.has_p and clock == self.chunk:
            # Causal: new coefficients affect only the next input and beyond.
            proposal, eta = self.updater(torch.cat((x, torch.tanh(h), p), -1)).chunk(2, -1)
            eta = torch.sigmoid(eta)
            p = (1-eta)*p + eta*torch.tanh(proposal)
            if intervention == 'reset_p':
                p = torch.zeros_like(p)
        if clock == self.chunk:
            clock = 0
        logits = self.read(torch.cat((h, x), -1))
        return logits, (h, p, clock)

    def forward(self, tokens, state=None, intervention='none', trace=False):
        if state is None:
            state = self.initial(tokens.shape[0])
        outputs, hs, ps = [], [], []
        for token in tokens.unbind(1):
            logits, state = self.step(token, state, intervention)
            outputs.append(logits)
            if trace:
                hs.append(state[0]); ps.append(state[1])
        out = torch.stack(outputs, 1)
        if trace:
            return out, state, torch.stack(hs, 1), torch.stack(ps, 1)
        return out, state

    def counts(self):
        return dict(parameters=sum(p.numel() for p in self.parameters()),
                    state_scalars=self.width+(16 if self.has_p else 0),
                    state_bytes_fp32=4*(self.width+(16 if self.has_p else 0)),
                    shared_clock_integer=1, basis_buffer_scalars=0 if self.arm=='gru64' else self.basis.numel())
