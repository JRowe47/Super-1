"""Single-layer linear minGRU, normalized linear minLSTM, native LSTM.

Reference: Feng et al., Were RNNs All We Needed?, arXiv:2410.01201v3 §3.
These isolated cells have no curve rotation, hierarchy, or positive candidate
transform. Signed candidates use an affine scan, not a positive log scan.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

ARMS = {'mingru64': ('mingru', 64), 'minlstm64': ('minlstm', 64),
        'lstm32': ('lstm', 32), 'lstm64': ('lstm', 64)}

def affine_scan(a, b, initial):
    """Inclusive Hillis–Steele composition; O(T log T) work, log T stages.

    Composition (a2,b2)∘(a1,b1)=(a2*a1,b2+a2*b1). No division by
    cumulative products. Nonzero initial states and signed b are supported.
    """
    offset = 1
    while offset < a.shape[1]:
        b = torch.cat((b[:, :offset], b[:, offset:] + a[:, offset:]*b[:, :-offset]), 1)
        a = torch.cat((a[:, :offset], a[:, offset:]*a[:, :-offset]), 1)
        offset *= 2
    return b + a*initial[:, None]

class Model(nn.Module):
    def __init__(self, arm):
        super().__init__()
        self.kind, self.width = ARMS[arm]
        self.arm = arm
        h = self.width
        self.embed = nn.Embedding(52, 32)
        self.readout = nn.Linear(h, 4*8)
        if self.kind == 'lstm':
            self.cell = nn.LSTM(32, h, batch_first=True)
            with torch.no_grad():
                self.cell.bias_ih_l0.zero_(); self.cell.bias_hh_l0.zero_()
                # Input, forget, output gates all start input/state independent.
                for j in (0, 1, 3):
                    self.cell.weight_ih_l0[j*h:(j+1)*h].zero_()
                    self.cell.weight_hh_l0[j*h:(j+1)*h].zero_()
                self.cell.bias_ih_l0[:h].fill_(math.log(.05/.95))
                self.cell.bias_ih_l0[h:2*h].fill_(math.log(.95/.05))
        else:
            self.proj = nn.Linear(32, h*(2 if self.kind == 'mingru' else 3))
            with torch.no_grad():
                self.proj.weight[h:].zero_()
                if self.kind == 'mingru':
                    self.proj.bias[h:].fill_(math.log(.05/.95))
                else:
                    # f=.5, i=.5*.05/.95 => normalized carry=.95, write=.05.
                    self.proj.bias[h:2*h].zero_()
                    raw_i = .5*.05/.95
                    self.proj.bias[2*h:].fill_(math.log(raw_i/(1-raw_i)))

    def initial(self, batch, like):
        h = like.new_zeros(batch, self.width)
        return (h, h.clone()) if self.kind == 'lstm' else h

    def recurrent(self, embedded, state=None, sequential=False):
        if state is None:
            state = self.initial(len(embedded), embedded)
        if self.kind == 'lstm':
            if not sequential:
                out, (h, c) = self.cell(embedded, (state[0][None], state[1][None]))
                return out, (h[0], c[0])
            # Independent scalar-time reference with native LSTM weights.
            h, c = state
            rows = []
            for x in embedded.unbind(1):
                gates = F.linear(x, self.cell.weight_ih_l0, self.cell.bias_ih_l0)
                gates = gates + F.linear(h, self.cell.weight_hh_l0, self.cell.bias_hh_l0)
                i, f, g, o = gates.chunk(4, -1)
                c = f.sigmoid()*c + i.sigmoid()*g.tanh()
                h = o.sigmoid()*c.tanh()
                rows.append(h)
            return torch.stack(rows, 1), (h, c)
        parts = self.proj(embedded).chunk(2 if self.kind == 'mingru' else 3, -1)
        candidate = parts[0]
        if self.kind == 'mingru':
            write = parts[1].sigmoid()
        else:
            # Algebraically i/(f+i), with stable log sigmoid evaluations.
            write = torch.sigmoid(F.logsigmoid(parts[2])-F.logsigmoid(parts[1]))
        a, b = 1-write, write*candidate
        if sequential:
            h, rows = state, []
            for at, bt in zip(a.unbind(1), b.unbind(1)):
                h = at*h + bt; rows.append(h)
            out = torch.stack(rows, 1)
        else:
            out = affine_scan(a, b, state)
        return out, out[:, -1]

    def forward(self, tokens, state=None, sequential=False):
        h, state = self.recurrent(self.embed(tokens), state, sequential)
        heads = self.readout(h).reshape(*tokens.shape, 4, 8)
        # Visible query identity selects one head; no answer or lock state is
        # supplied. Non-query positions have no loss and select head 0..3 safely.
        q = (tokens-32).clamp(0, 3)
        logits = heads.gather(2, q[..., None, None].expand(*q.shape, 1, 8)).squeeze(2)
        return logits, state

    def accounting(self):
        h = self.width
        return dict(parameters=sum(p.numel() for p in self.parameters()),
                    recurrent_parameters=sum(p.numel() for n,p in self.named_parameters()
                                             if n.startswith(('cell.', 'proj.'))),
                    persistent_state_floats=h*(2 if self.kind=='lstm' else 1),
                    persistent_state_bytes_fp32=4*h*(2 if self.kind=='lstm' else 1),
                    recurrent_dense_macs_per_token=(4*h*(32+h) if self.kind=='lstm'
                          else (2 if self.kind=='mingru' else 3)*32*h),
                    note='State excludes weights, optimizer, activations and allocator. MACs exclude gates, scan and readout.')
