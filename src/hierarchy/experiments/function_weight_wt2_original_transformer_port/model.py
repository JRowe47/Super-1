"""Function-parameterized GPT for scaled experiments.

Modes:
  full           - standard nn.Linear weights (control)
  tucker         - W_l = U_type @ diag(z_layer) @ V_type; U,V shared across layers per matrix type
  tucker_noshare - W_l = U_layer @ diag(z_layer) @ V_layer; same form/rank/FLOPs, NO sharing.
                   Controlled baseline: identical architecture and compute; differs from
                   `tucker` only in whether U,V are shared across layers.
  addlr          - W_l = W_shared_type + U_layer @ V_layer
  basis          - W_l = sum_k c_layer[k] * B_type[k]; K full-rank shared basis
                   matrices per type, each layer a K-vector of coefficients.
                   (tucker is the special case where the basis is r rank-1 terms.)
  dyncurve       - curve + input-conditioned coefficients: c = c(t) + g(x_token).
                   Same shared basis; a tiny zero-init per-layer gate maps each
                   token's activation to a K-vector delta, so effective weights
                   vary PER TOKEN at static-weight storage. Costs ~K x block
                   FLOPs at train time; decode bandwidth unchanged (streams only B).
  curve          - W(t) = sum_k c_k(t) * B_type[k], c(t) = P @ [1,t,t^2,t^3],
                   t = depth position in [0,1]. NO per-layer parameters at all:
                   the optimizer must place layers on a learned curve. Depth is
                   continuous -- the same trained curve can be sampled at any L.

All linears are bias-free. Pre-LN blocks, RoPE, GELU MLP (4x), tied embedding head.
Function-form linears compute x@U -> *z -> @V and never materialize W.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _uv_pair(din, dout, r):
    # Effective W = U @ diag(1) @ V has elementwise std ~0.02, matching the
    # dense control's init (was din**-0.5 ~ 0.036 at d=768 -- a confound).
    return (nn.Parameter(torch.randn(din, r) * din ** -0.5),
            nn.Parameter(torch.randn(r, dout) * (0.02 * din ** 0.5 * r ** -0.5)))


class TuckerLinear(nn.Module):
    def __init__(self, bank, key, din, dout, r, share=True):
        super().__init__()
        if share:
            if key not in bank:
                bank[key] = _uv_pair(din, dout, r)
            self.U, self.V = bank[key]
        else:
            self.U, self.V = _uv_pair(din, dout, r)  # per-layer, unshared
        self.z = nn.Parameter(torch.ones(r))

    def forward(self, x):
        return ((x @ self.U) * self.z) @ self.V  # function-form: W never materialized


class BasisLinear(nn.Module):
    def __init__(self, bank, key, din, dout, K):
        super().__init__()
        if key not in bank:
            bank[key] = (nn.Parameter(torch.randn(K, din, dout) * 0.02),)
        self.B = bank[key][0]
        self.c = nn.Parameter(torch.full((K,), K ** -0.5))  # W std 0.02 at init

    def forward(self, x):
        W = torch.einsum("k,kio->io", self.c, self.B)
        return x @ W


class CurveLinear(nn.Module):
    DEG = 3

    def __init__(self, bank, key, din, dout, K, t):
        super().__init__()
        if key not in bank:
            B = nn.Parameter(torch.randn(K, din, dout) * 0.02)
            # stored 1-D so the "dim<2 -> no weight decay" rule protects it;
            # init: constant term K**-0.5, higher orders 0 -> all layers start
            # identical (like tucker z=1) and the curve grows during training.
            p = torch.zeros(K, self.DEG + 1)
            p[:, 0] = K ** -0.5
            bank[key] = (B, nn.Parameter(p.reshape(-1)))
        self.B, self.p = bank[key]
        self.register_buffer(
            "tpow", torch.tensor([t ** i for i in range(self.DEG + 1)]), persistent=False)

    def coeffs(self):
        return self.p.view(-1, self.DEG + 1) @ self.tpow

    def forward(self, x):
        W = torch.einsum("k,kio->io", self.coeffs(), self.B)
        return x @ W


class DynCurveLinear(nn.Module):
    DEG = 3
    GATE_H = 8

    def __init__(self, bank, key, din, dout, K, t):
        super().__init__()
        if key not in bank:
            B = nn.Parameter(torch.randn(K, din, dout) * 0.02)
            p = torch.zeros(K, self.DEG + 1); p[:, 0] = K ** -0.5
            bank[key] = (B, nn.Parameter(p.reshape(-1)))
        self.B, self.p = bank[key]
        self.register_buffer(
            "tpow", torch.tensor([t ** i for i in range(self.DEG + 1)]), persistent=False)
        # per-layer input gate, ZERO-init -> starts identical to curve mode
        self.g1 = nn.Parameter(torch.randn(din, self.GATE_H) * din ** -0.5)
        self.g2 = nn.Parameter(torch.zeros(self.GATE_H, self.B.shape[0]))

    def coeffs(self):  # static part (for analysis tools)
        return self.p.view(-1, self.DEG + 1) @ self.tpow

    def forward(self, x):  # x: (B, T, din)
        c0 = self.coeffs()                                   # (K,)
        dc = torch.tanh(x @ self.g1) @ self.g2               # (B, T, K) per-token delta
        c = c0 + dc
        y = None
        for k in range(self.B.shape[0]):                     # sum_k (c_k * x) @ B_k
            yk = (x * c[..., k:k + 1]) @ self.B[k]
            y = yk if y is None else y + yk
        return y


class AddLRLinear(nn.Module):
    def __init__(self, bank, key, din, dout, r):
        super().__init__()
        if key not in bank:
            bank[key] = (nn.Parameter(torch.randn(din, dout) * 0.02),)  # match dense init
        self.Ws = bank[key][0]
        self.U = nn.Parameter(torch.randn(din, r) * 0.01)
        self.V = nn.Parameter(torch.randn(r, dout) * 0.01)

    def forward(self, x):
        return x @ self.Ws + (x @ self.U) @ self.V


def rope_cache(ctx, hd, device, dtype, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, hd, 2, device=device).float() / hd))
    t = torch.arange(ctx, device=device).float()
    freqs = torch.outer(t, inv)
    return torch.cos(freqs).to(dtype), torch.sin(freqs).to(dtype)


def apply_rope(x, cos, sin):
    # x: (B, nh, T, hd)
    T = x.shape[2]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c, s = cos[:T].unsqueeze(0).unsqueeze(0), sin[:T].unsqueeze(0).unsqueeze(0)
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * c - x2 * s
    out[..., 1::2] = x1 * s + x2 * c
    return out


class Block(nn.Module):
    def __init__(self, d, nh, layer, mode, bank, r):
        super().__init__()
        self.nh, self.d = nh, d
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)

        def lin(key, din, dout):
            if mode in ("tucker", "tucker_noshare"):
                return TuckerLinear(bank, key, din, dout, r, share=(mode == "tucker"))
            if mode == "addlr":
                return AddLRLinear(bank, key, din, dout, r)
            if mode == "basis":
                return BasisLinear(bank, key, din, dout, r)  # r = K here
            if mode == "curve":
                return CurveLinear(bank, key, din, dout, r, layer)  # layer = t in [0,1]
            if mode == "dyncurve":
                return DynCurveLinear(bank, key, din, dout, r, layer)
            return nn.Linear(din, dout, bias=False)

        self.q = lin("q", d, d); self.k = lin("k", d, d)
        self.v = lin("v", d, d); self.o = lin("o", d, d)
        self.f1 = lin("f1", d, 4 * d); self.f2 = lin("f2", 4 * d, d)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        h = self.ln1(x)
        q = self.q(h).view(B, T, self.nh, -1).transpose(1, 2)
        k = self.k(h).view(B, T, self.nh, -1).transpose(1, 2)
        v = self.v(h).view(B, T, self.nh, -1).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.o(a.transpose(1, 2).reshape(B, T, C))
        x = x + self.f2(F.gelu(self.f1(self.ln2(x))))
        return x


class GPT(nn.Module):
    def __init__(self, vocab, d, nh, L, ctx, mode="full", r=0):
        super().__init__()
        self.ctx = ctx
        self.tok = nn.Embedding(vocab, d)
        bank = {}
        self.blocks = nn.ModuleList([
            Block(d, nh, l / max(L - 1, 1), mode, bank, r) for l in range(L)])
        self.bank = nn.ParameterList([p for tup in bank.values() for p in tup])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight
        self.register_buffer("_cos", torch.empty(0), persistent=False)
        self.register_buffer("_sin", torch.empty(0), persistent=False)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Linear) and m is not getattr(self, "head", None):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx):
        B, T = idx.shape
        if self._cos.numel() == 0 or self._cos.shape[0] < T:
            hd = self.blocks[0].d // self.blocks[0].nh
            cos, sin = rope_cache(self.ctx, hd, idx.device, self.tok.weight.dtype)
            self._cos, self._sin = cos, sin
        h = self.tok(idx)
        for b in self.blocks:
            h = b(h, self._cos, self._sin)
        return self.head(self.lnf(h))

    def block_params(self):
        seen, tot = set(), 0
        for b in self.blocks:
            for p in b.parameters():
                if id(p) not in seen:
                    seen.add(id(p)); tot += p.numel()
        return tot
