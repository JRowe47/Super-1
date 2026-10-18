"""Fixed-step sampling of the existing cubic weight curve; no block skipping."""
import importlib.util
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'src/hierarchy/experiments/function_weight_wt2_original_transformer_port/model.py'
NANO_SOURCE = ROOT / 'artifacts/nanogpt-10m-20260815/package/source/official_nanogpt/third_party/nanogpt/model.py'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


original = load('original_curve', SOURCE)
nano = load('pinned_nano', NANO_SOURCE)


class SampledCurve(nn.Module):
    def __init__(self, c, mode):
        super().__init__()
        self.mode, self.depth = mode, c['layers']
        self.base = original.GPT(c['vocab'], c['width'], c['heads'], c['layers'],
                                 c['context'], mode='curve', r=c['bases'])
        self.router = nn.Sequential(nn.Linear(c['width'] + 1, c['router_hidden']),
                                    nn.Tanh(), nn.Linear(c['router_hidden'], 1))
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)

    @staticmethod
    def projection(layer, x, t):
        powers = torch.stack([t ** j for j in range(layer.DEG + 1)], dim=-1)
        coefficients = powers @ layer.p.view(-1, layer.DEG + 1).T
        # All curve arms execute every basis; no claim of matched dense FLOPs.
        return sum((x @ layer.B[k]) * coefficients[..., k:k+1]
                   for k in range(layer.B.shape[0]))

    def forward(self, idx, positions=False, uniform=False):
        x = self.base.tok(idx)
        cos, sin = original.rope_cache(idx.shape[1], x.shape[-1] // self.base.blocks[0].nh,
                                       x.device, x.dtype)
        t = x.new_zeros(idx.shape)
        sampled = []
        for i, block in enumerate(self.base.blocks):
            h = block.ln1(x)
            if 0 < i < self.depth - 1:
                condition = h if self.mode == 'token' else torch.zeros_like(h)
                delta = self.router(torch.cat((condition, t.unsqueeze(-1)), dim=-1)).squeeze(-1)
                if self.mode == 'fixed' or uniform:
                    delta = delta * 0
                # Uniform spacing at delta=0; positive increments keep ordered depth.
                fraction = torch.sigmoid(delta - x.new_tensor(self.depth - i - 1).log())
                t = t + (1 - t) * fraction
            elif i == self.depth - 1:
                t = torch.ones_like(t)
            sampled.append(t)
            batch, length, width = x.shape
            def heads(layer):
                return self.projection(layer, h, t).view(batch, length, block.nh, -1).transpose(1, 2)
            q, k, v = heads(block.q), heads(block.k), heads(block.v)
            q, k = original.apply_rope(q, cos, sin), original.apply_rope(k, cos, sin)
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            x = x + self.projection(block.o, a.transpose(1, 2).reshape(batch, length, width), t)
            x = x + self.projection(block.f2, F.gelu(self.projection(block.f1, block.ln2(x), t)), t)
        logits = self.base.head(self.base.lnf(x))
        return (logits, torch.stack(sampled, dim=-1)) if positions else logits


def build(c, arm, seed):
    torch.manual_seed(seed)
    if arm == 'nanogpt':
        return nano.GPT(nano.GPTConfig(block_size=c['context'], vocab_size=c['vocab'],
                        n_layer=c['layers'], n_head=c['heads'], n_embd=c['width'], dropout=0, bias=False))
    if arm == 'dense':
        return original.GPT(c['vocab'], c['width'], c['heads'], c['layers'], c['context'], mode='full')
    return SampledCurve(c, arm)


def logits(model, arm, x, y):
    return model(x.contiguous(), y.contiguous())[0] if arm == 'nanogpt' else model(x)
