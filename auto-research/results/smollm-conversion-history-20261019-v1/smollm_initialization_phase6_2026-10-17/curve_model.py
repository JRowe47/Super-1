"""Depth-indexed Hermite weight curves with immutable original knots.

CPU float32 research prototype. A curve is over homologous parameter tensors,
not over tokens. Embeddings and the output head are not interpolated.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.func import functional_call
from transformers.models.llama.modeling_llama import LlamaDecoderLayer


def key(name):
    return name.replace('.', '__')


class HermiteBank(nn.Module):
    def __init__(self, blocks, trainable=True):
        super().__init__()
        # Frozen views avoid duplicating the complete original checkpoint.
        self.knots = [{n: p.detach() for n, p in b.named_parameters()} for b in blocks]
        self.names = list(self.knots[0])
        assert all(list(row) == self.names for row in self.knots)
        self.slopes = nn.ParameterDict()
        if trainable:
            for name in self.names:
                self.slopes[key(name)] = nn.Parameter(torch.stack([
                    self.initial_slope(name, j) for j in range(len(self.knots))]))

    def initial_slope(self, name, j):
        # Clamped Catmull-Rom initialization: duplicated boundary neighbors.
        left, right = max(0, j-1), min(len(self.knots)-1, j+1)
        return .5 * (self.knots[right][name] - self.knots[left][name])

    def sample(self, t):
        if not 0 <= t <= len(self.knots)-1:
            raise ValueError('Curve coordinate is outside the original knot interval')
        if abs(t-round(t)) < 1e-12:
            return self.knots[round(t)]
        j = math.floor(t); u = t-j
        h = (2*u**3-3*u**2+1, u**3-2*u**2+u, -2*u**3+3*u**2, u**3-u**2)
        out = {}
        for name in self.names:
            m0 = self.slopes[key(name)][j] if self.slopes else self.initial_slope(name,j)
            m1 = self.slopes[key(name)][j+1] if self.slopes else self.initial_slope(name,j+1)
            out[name] = h[0]*self.knots[j][name]+h[1]*m0+h[2]*self.knots[j+1][name]+h[3]*m1
        return out


class CurveBlock(nn.Module):
    def __init__(self, config, bank, t, scale, layer_idx, materialized=False):
        super().__init__()
        # The optimizer owns the bank explicitly; do not register it repeatedly.
        object.__setattr__(self, 'bank', bank)
        with torch.device('meta'):
            self.template = LlamaDecoderLayer(config, layer_idx)
        self.template.requires_grad_(False)
        self.t, self.scale = float(t), float(scale)
        self.last_output = None
        self.capture = False
        self.sampled = None
        if materialized:
            with torch.no_grad():
                self.sampled = {n:p.clone() for n,p in bank.sample(t).items()}

    def forward(self, hidden_states, **kwargs):
        params = self.sampled if self.sampled is not None else self.bank.sample(self.t)
        hooks = []
        if self.scale != 1:
            hooks.append(self.template.self_attn.register_forward_hook(
                lambda m,a,o: (o[0]*self.scale, *o[1:])))
            hooks.append(self.template.mlp.register_forward_hook(lambda m,a,o: o*self.scale))
        try:
            # All nine parameters replaced. The unused template rotary buffer is
            # meta: the native outer model supplies real position_embeddings.
            result = functional_call(self.template, params, (hidden_states,), kwargs)
        finally:
            for hook in hooks: hook.remove()
        self.last_output = result[0] if self.capture else None
        return result


def coordinates(n):
    return [(i+.5)*4/n-.5 for i in range(n)]


def restore(model, original):
    model.model.layers = nn.ModuleList(original)
    for i,b in enumerate(original): b.self_attn.layer_idx = i


def install(model, original, specs, materialized=False, capture=False):
    """specs: {start: {'bank': bank, 'steps': n}} or {'keep': offsets}."""
    layers=[]; active={}; i=0
    while i < len(original):
        if i not in specs:
            b=original[i]; b.self_attn.layer_idx=len(layers); layers.append(b); i+=1
            continue
        spec=specs[i]
        if 'keep' in spec:
            for offset in spec['keep']:
                b=original[i+offset]; b.self_attn.layer_idx=len(layers); layers.append(b)
        else:
            active[i]=[]; n=spec['steps']
            for t in coordinates(n):
                b=CurveBlock(model.config,spec['bank'],t,4/n,len(layers),materialized)
                b.capture=capture; layers.append(b); active[i].append(b)
        i+=4
    model.model.layers=nn.ModuleList(layers)
    return active


def window_forward(model, bank, anchor, steps, return_path=True):
    positions=torch.arange(anchor.shape[1])[None]
    mask=torch.triu(torch.full((anchor.shape[1],anchor.shape[1]),torch.finfo(anchor.dtype).min),diagonal=1)[None,None]
    rope=model.model.rotary_emb(anchor,positions)
    state=anchor; path=[]
    for i,t in enumerate(coordinates(steps)):
        block=CurveBlock(model.config,bank,t,4/steps,i)
        state=block(state,attention_mask=mask,position_ids=positions,
                    position_embeddings=rope,use_cache=False)[0]
        path.append(state)
    return path if return_path else state


def path_loss(predictions, teacher, floor):
    """Per-token displacement-relative error, with linear teacher path targets.

    Intermediate teacher states at fractional depth are a proxy target, not an
    assertion that the original discrete network has a known continuous path.
    """
    anchor=teacher[:,0]; losses=[]; n=len(predictions)
    for i,prediction in enumerate(predictions):
        t=4*(i+1)/n; lo=min(math.floor(t),4); hi=min(lo+1,4); u=t-lo
        target=teacher[:,lo]*(1-u)+teacher[:,hi]*u
        energy=(target-anchor).square().mean(-1).clamp_min(floor)
        losses.append(((prediction-target).square().mean(-1)/energy).mean())
    return .5*losses[-1]+.5*torch.stack(losses).mean()
