"""Experimental CPU packed-weight inference; no dense weight masters or KV cache.

Int4 is symmetric groupwise quantization in the original tensor's flat order.
Inference decodes bounded output tiles; activations and attention remain FP32.
This is a storage/quality reference, not a fast integer kernel.
"""
import math
import torch
from torch.nn import functional as F


def quantized(name, tensor, arm):
    if arm == 'curve_b4':
        return name.endswith('.B')
    if arm == 'curve_be4':
        return name.endswith('.B') or name == 'base.tok.weight'
    if arm == 'nano_q4':
        return tensor.ndim == 2
    if arm in ('curve_fp16', 'nano_fp16'):
        return False
    raise ValueError(arm)


def quant_values(w, group):
    flat = w.detach().float().reshape(-1)
    padded = F.pad(flat, (0, (-flat.numel()) % group)).view(-1, group)
    scale = (padded.abs().amax(-1) / 7).clamp_min(torch.finfo(torch.float16).tiny).half()
    q = (padded / scale.float()[:, None]).round().clamp(-7, 7)
    return q, scale


def fake_parameters(model, arm, group):
    result = {}
    for name, w in model.named_parameters():
        if quantized(name, w, arm):
            q, scale = quant_values(w, group)
            rounded = (q * scale.float()[:, None]).flatten()[:w.numel()].view_as(w)
        else:
            rounded = w.detach().half().float()
        result[name] = w + (rounded - w).detach()
    return result


def export(model, config, arm, group=64, tile=16):
    if group <= 0 or group % 2 or tile <= 0:
        raise ValueError('Positive even group size and positive tile size required')
    entries, aliases, seen = {}, {}, {}
    for name, w in model.named_parameters(remove_duplicate=False):
        if id(w) in seen:
            aliases[name] = seen[id(w)]
            continue
        seen[id(w)] = name
        if quantized(name, w, arm):
            q, scale = quant_values(w, group)
            nibbles = (q.flatten().to(torch.int16) + 8).to(torch.uint8)
            data = nibbles[::2] | (nibbles[1::2] << 4)
            entries[name] = dict(shape=list(w.shape), data=data, scale=scale, bits=4)
        else:
            entries[name] = dict(shape=list(w.shape), data=w.detach().half().clone(), bits=16)
    return dict(format_version=1, config=dict(config), arm=arm, group=group,
                tile=tile, entries=entries, aliases=aliases)


class PackedModel:
    """Own only packed/FP16 tensors; dequantization scratch is transient."""
    def __init__(self, bundle):
        if bundle['format_version'] != 1:
            raise ValueError('Unsupported format')
        self.bundle = bundle
        self.c = bundle['config']
        self.arm = bundle['arm']
        self.tile = bundle['tile']
        self.max_decoded_values = 0
        self.depth = self.c['layers']
        self.parameter_bytes = sum(t.numel() * t.element_size()
                                   for e in bundle['entries'].values()
                                   for t in e.values() if isinstance(t, torch.Tensor))
        assert all(t.dtype in (torch.uint8, torch.float16)
                   for e in bundle['entries'].values() for t in e.values()
                   if isinstance(t, torch.Tensor))

    def entry(self, name):
        return self.bundle['entries'][self.bundle['aliases'].get(name, name)]

    def get(self, name, indices=None):
        e = self.entry(name)
        if indices is None:
            # Whole small vectors only; major matrices must use tiled access.
            if len(e['shape']) > 1:
                raise ValueError('Dense matrix materialization is forbidden')
            indices = torch.arange(math.prod(e['shape']))
        self.max_decoded_values = max(self.max_decoded_values, indices.numel())
        if e['bits'] == 16:
            return e['data'].reshape(-1)[indices].float()
        data = e['data'][indices // 2]
        q = ((data >> ((indices % 2) * 4)) & 15).float() - 8
        return q * e['scale'][indices // self.bundle['group']].float()

    def embedding(self, name, idx):
        width = self.entry(name)['shape'][1]
        # Decode at most tile requested rows at once; returned embedding is activation.
        flat = idx.flatten()
        out = torch.empty((flat.numel(), width))
        for start in range(0, len(flat), self.tile):
            ix = flat[start:start+self.tile, None] * width + torch.arange(width)[None, :]
            out[start:start+self.tile] = self.get(name, ix)
        return out.view(*idx.shape, width)

    def linear(self, name, x, bias=None, basis=None):
        shape = self.entry(name)['shape']
        if basis is None:
            dout, din = shape
        else:
            _, din, dout = shape
        result = torch.empty((*x.shape[:-1], dout), dtype=x.dtype)
        for start in range(0, dout, self.tile):
            rows = torch.arange(start, min(start+self.tile, dout))
            if basis is None:
                ix = rows[:, None] * din + torch.arange(din)[None, :]
            else:
                ix = basis * din * dout + torch.arange(din)[None, :] * dout + rows[:, None]
            result[..., start:start+len(rows)] = F.linear(x, self.get(name, ix))
        if bias is not None:
            result += self.get(bias)
        return result

    def norm(self, name, x):
        return F.layer_norm(x, (x.shape[-1],), self.get(name+'.weight'), self.get(name+'.bias'), 1e-5)

    def projection(self, name, x, t):
        p = self.get(name+'.p').view(-1, 4)
        coefficients = torch.stack([t**j for j in range(4)], -1) @ p.T
        return sum(self.linear(name+'.B', x, basis=k)*coefficients[..., k:k+1]
                   for k in range(p.shape[0]))

    def __call__(self, idx, positions=False, depth=None):
        if idx.device.type != 'cpu':
            raise ValueError('This reference implementation supports CPU only')
        if self.arm.startswith('nano'):
            if positions or depth is not None:
                raise ValueError('Depth/t sampling belongs to the curve model')
            return self.nano(idx)
        depth = self.depth if depth is None else depth
        if not 2 <= depth <= self.c['layers']:
            raise ValueError('Depth outside trained block range')
        x = self.embedding('base.tok.weight', idx)
        batch, length, width = x.shape
        heads = self.c['heads']; dim = width // heads
        # Exactly the original RoPE convention, without loading a dense model.
        theta = 10000.0 ** (-torch.arange(0, dim, 2).float() / dim)
        angles = torch.outer(torch.arange(length).float(), theta)
        cos, sin = angles.cos()[None, None], angles.sin()[None, None]
        def rope(q):
            even, odd = q[..., ::2], q[..., 1::2]
            return torch.stack((even*cos-odd*sin, even*sin+odd*cos), -1).flatten(-2)
        t = torch.zeros_like(idx, dtype=torch.float32); sampled = []
        for i in range(depth):
            prefix = f'base.blocks.{i}'
            h = self.norm(prefix+'.ln1', x)
            if 0 < i < depth-1:
                condition = torch.cat((h, t.unsqueeze(-1)), -1)
                z = self.linear('router.0.weight', condition, 'router.0.bias').tanh()
                delta = self.linear('router.2.weight', z, 'router.2.bias').squeeze(-1)
                t = t + (1-t)*torch.sigmoid(delta - x.new_tensor(depth-i-1).log())
            elif i == depth-1:
                t = torch.ones_like(t)
            sampled.append(t)
            def proj(n):
                return self.projection(prefix+'.'+n,h,t).view(batch,length,heads,dim).transpose(1,2)
            q,k,v = proj('q'),proj('k'),proj('v')
            a = F.scaled_dot_product_attention(rope(q),rope(k),v,is_causal=True)
            x = x+self.projection(prefix+'.o',a.transpose(1,2).reshape(batch,length,width),t)
            x = x+self.projection(prefix+'.f2',F.gelu(self.projection(prefix+'.f1',self.norm(prefix+'.ln2',x),t)),t)
        logits = self.linear('base.head.weight',self.norm('base.lnf',x))
        return (logits,torch.stack(sampled,-1)) if positions else logits

    def nano(self, idx):
        if idx.shape[1] > self.c['context']:
            raise ValueError('nanoGPT position limit exceeded')
        x = self.embedding('transformer.wte.weight',idx)
        x += self.embedding('transformer.wpe.weight',torch.arange(idx.shape[1]))
        batch,length,width=x.shape; heads=self.c['heads']
        for i in range(self.c['layers']):
            p=f'transformer.h.{i}'
            h=self.norm(p+'.ln_1',x)
            q,k,v=self.linear(p+'.attn.c_attn.weight',h,p+'.attn.c_attn.bias').split(width,-1)
            q,k,v=[a.view(batch,length,heads,width//heads).transpose(1,2) for a in (q,k,v)]
            a=F.scaled_dot_product_attention(q,k,v,is_causal=True).transpose(1,2).reshape(batch,length,width)
            x=x+self.linear(p+'.attn.c_proj.weight',a,p+'.attn.c_proj.bias')
            h=self.norm(p+'.ln_2',x)
            h=F.gelu(self.linear(p+'.mlp.c_fc.weight',h,p+'.mlp.c_fc.bias'))
            x=x+self.linear(p+'.mlp.c_proj.weight',h,p+'.mlp.c_proj.bias')
        return self.linear('lm_head.weight',self.norm('transformer.ln_f',x))
