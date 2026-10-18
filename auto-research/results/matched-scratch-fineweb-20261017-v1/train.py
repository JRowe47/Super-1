"""Matched scratch-CE benchmark: pinned nanoGPT vs the exact original hierarchy.

The hierarchy arm uses an execution-optimized but equation-identical implementation
(`FastHierarchy`) whose state_dict is interchangeable with the recovered
`HierarchicalDistributionStudent`. Parity is checked by `parity.py`.
"""
from __future__ import annotations

import argparse, json, math, os, time, hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB = 50304  # GPT-2 50257 padded (nanoGPT convention)


# ----------------------------------------------------------------------------- data
def load_bin(path):
    header = np.fromfile(path, dtype=np.int32, count=256)
    assert header[0] == 20240520, path
    n = int(header[2])
    return np.memmap(path, dtype=np.uint16, mode="r", offset=1024, shape=(n,))


class Data:
    def __init__(self, root, train_shards):
        self.train = np.concatenate([np.asarray(load_bin(f"{root}/fineweb_train_{i:06d}.bin")) for i in range(1, train_shards + 1)])
        val = np.asarray(load_bin(f"{root}/fineweb_val_000000.bin"))
        half = len(val) // 2
        self.selection = val[:half]
        self.confirmation = val[half:]

    def batch(self, rng, B, T):
        o = rng.integers(0, len(self.train) - T - 1, size=B)
        xs = np.stack([self.train[i:i + T + 1] for i in o]).astype(np.int64)
        t = torch.from_numpy(xs).pin_memory().cuda(non_blocking=True)
        return t[:, :-1].contiguous(), t[:, 1:].contiguous()


class Lanes:
    """B contiguous lanes over the train stream; state is carried across adjacent chunks."""

    def __init__(self, data, B, T, seed):
        rng = np.random.default_rng(seed)
        lane = len(data.train) // B
        self.starts = np.arange(B) * lane + rng.integers(0, 1000, size=B)
        self.data, self.T, self.pos = data, T, 0

    def batch(self):
        xs = np.stack([self.data.train[s + self.pos:s + self.pos + self.T + 1] for s in self.starts]).astype(np.int64)
        self.pos += self.T
        t = torch.from_numpy(xs).pin_memory().cuda(non_blocking=True)
        return t[:, :-1].contiguous(), t[:, 1:].contiguous()


# ----------------------------------------------------------------------------- models
class FastHierarchy(nn.Module):
    """Equation-identical, segment-chunked execution of HierarchicalDistributionStudent.

    Within a segment the slow states are constant, so the fast GRUCell recurrence over
    the segment's tokens is executed as one cuDNN GRU call with the same weights.
    Parameter names match the recovered class exactly.
    """

    def __init__(self, vocab_size, width, segment_length=32, document_segments=8):
        super().__init__()
        self.vocab_size, self.width = vocab_size, width
        self.segment_length, self.document_segments = segment_length, document_segments
        self.embedding = nn.Embedding(vocab_size, width)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=width ** -0.5)
        self.final_norm = nn.LayerNorm(width)
        self.fast_cell = nn.GRUCell(3 * width, width)
        self.segment_cell = nn.GRUCell(width, width)
        self.document_cell = nn.GRUCell(width, width)
        self.combine = nn.Sequential(
            nn.Linear(3 * width, 2 * width), nn.SiLU(), nn.Linear(2 * width, width))

    def initial_state(self, B, device, dtype=torch.float32):
        z = lambda: torch.zeros(B, self.width, device=device, dtype=dtype)
        return {"fast": z(), "segment": z(), "document": z(), "segments": 0}

    def _gru(self, x, h):
        c = self.fast_cell
        params = [c.weight_ih, c.weight_hh, c.bias_ih, c.bias_hh]
        out, hn = torch._VF.gru(x, h.unsqueeze(0), params, True, 1, 0.0, self.training, False, True)
        return out, hn.squeeze(0)

    def hidden(self, idx, state=None):
        B, T = idx.shape
        L = self.segment_length
        assert T % L == 0, "context must be a multiple of the segment length"
        if state is None:
            state = self.initial_state(B, idx.device, self.embedding.weight.dtype)
        emb = self.embedding(idx)
        fast, seg, doc, segments = state["fast"], state["segment"], state["document"], state["segments"]
        outs, segc, docc = [], [], []
        for k in range(T // L):
            e = emb[:, k * L:(k + 1) * L]
            x = torch.cat((e, seg.unsqueeze(1).expand(-1, L, -1).to(e.dtype),
                           doc.unsqueeze(1).expand(-1, L, -1).to(e.dtype)), dim=-1)
            out, fast = self._gru(x, fast.to(x.dtype))
            seg_before, doc_before = seg, doc
            seg = self.segment_cell(out.sum(1) / L, seg)
            segments += 1
            if segments == self.document_segments:
                doc = self.document_cell(seg, doc)
                segments = 0
            sc = torch.cat((seg_before.unsqueeze(1).expand(-1, L - 1, -1), seg.unsqueeze(1)), 1)
            dc = torch.cat((doc_before.unsqueeze(1).expand(-1, L - 1, -1), doc.unsqueeze(1)), 1)
            outs.append(out); segc.append(sc); docc.append(dc)
        out = torch.cat(outs, 1)
        h = self.combine(torch.cat((out, torch.cat(segc, 1).to(out.dtype), torch.cat(docc, 1).to(out.dtype)), -1))
        new_state = {"fast": fast, "segment": seg, "document": doc, "segments": segments}
        return h, new_state

    def forward(self, idx, targets=None, state=None):
        h, st = self.hidden(idx, state)
        logits = F.linear(self.final_norm(h), self.embedding.weight)
        loss = None
        if targets is not None:
            lg = logits if logits.dtype == torch.float64 else logits.float()
            loss = F.cross_entropy(lg.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss, st


class HierMix(nn.Module):
    """The original hierarchy cell equations (fast/segment/document GRUs + combine) used as a token mixer."""

    def __init__(self, width, segment_length=32, document_segments=8):
        super().__init__()
        self.width, self.segment_length, self.document_segments = width, segment_length, document_segments
        self.fast_cell = nn.GRUCell(3 * width, width)
        self.segment_cell = nn.GRUCell(width, width)
        self.document_cell = nn.GRUCell(width, width)
        self.combine = nn.Sequential(nn.Linear(3 * width, 2 * width), nn.SiLU(), nn.Linear(2 * width, width))

    _gru = FastHierarchy._gru

    def forward(self, e, state=None):
        B, T, d = e.shape
        L = self.segment_length
        if state is None:
            z = lambda: torch.zeros(B, d, device=e.device, dtype=torch.float32)
            state = {"fast": z(), "segment": z(), "document": z(), "segments": 0}
        fast, seg, doc, segments = state["fast"], state["segment"], state["document"], state["segments"]
        outs, segc, docc = [], [], []
        for k in range(T // L):
            x = torch.cat((e[:, k * L:(k + 1) * L], seg.unsqueeze(1).expand(-1, L, -1).to(e.dtype),
                           doc.unsqueeze(1).expand(-1, L, -1).to(e.dtype)), -1)
            out, fast = self._gru(x, fast.to(x.dtype))
            sb, db = seg, doc
            seg = self.segment_cell(out.sum(1) / L, seg)
            segments += 1
            if segments == self.document_segments:
                doc = self.document_cell(seg, doc); segments = 0
            segc.append(torch.cat((sb.unsqueeze(1).expand(-1, L - 1, -1), seg.unsqueeze(1)), 1))
            docc.append(torch.cat((db.unsqueeze(1).expand(-1, L - 1, -1), doc.unsqueeze(1)), 1))
            outs.append(out)
        out = torch.cat(outs, 1)
        h = self.combine(torch.cat((out, torch.cat(segc, 1).to(out.dtype), torch.cat(docc, 1).to(out.dtype)), -1))
        return h, {"fast": fast, "segment": seg, "document": doc, "segments": segments}


class DeepHierarchy(nn.Module):
    """NEW ARCHITECTURE ARM (not the original): K pre-LN residual blocks, each = original hierarchy
    cell as token mixer + 4x GELU MLP, tied embeddings. Separately labeled from the reference."""

    def __init__(self, vocab_size, width, layers, mlp_mult=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, width)
        nn.init.normal_(self.embedding.weight, std=0.02)
        self.blocks = nn.ModuleList()
        for _ in range(layers):
            blk = nn.Module()
            blk.ln1, blk.mix, blk.ln2 = nn.LayerNorm(width), HierMix(width), nn.LayerNorm(width)
            blk.mlp = nn.Sequential(nn.Linear(width, mlp_mult * width), nn.GELU(), nn.Linear(mlp_mult * width, width))
            nn.init.normal_(blk.mlp[2].weight, std=0.02 / math.sqrt(2 * layers))
            nn.init.normal_(blk.mix.combine[2].weight, std=0.02 / math.sqrt(2 * layers))
            self.blocks.append(blk)
        self.final_norm = nn.LayerNorm(width)

    def forward(self, idx, targets=None, state=None):
        x = self.embedding(idx)
        new = []
        for i, b in enumerate(self.blocks):
            h, st = b.mix(b.ln1(x), None if state is None else state[i])
            x = x + h
            x = x + b.mlp(b.ln2(x))
            new.append(st)
        logits = F.linear(self.final_norm(x), self.embedding.weight)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss, new


class CausalSelfAttention(nn.Module):
    """nanoGPT-style causal attention (flash SDPA), no positional embedding: order comes from the recurrence."""

    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.c_attn = nn.Linear(width, 3 * width)
        self.c_proj = nn.Linear(width, width)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.heads, C // self.heads).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C))


class HybridHierarchy(nn.Module):
    """NEW ARCHITECTURE ARM: pre-LN residual stack alternating original-hierarchy mixer blocks and
    nanoGPT causal-attention blocks (pattern string, e.g. 'HAHAHA'); each block has a 4x GELU MLP."""

    def __init__(self, vocab_size, width, pattern, heads, mlp_mult=4):
        super().__init__()
        self.pattern = pattern
        self.embedding = nn.Embedding(vocab_size, width)
        nn.init.normal_(self.embedding.weight, std=0.02)
        self.blocks = nn.ModuleList()
        n = len(pattern)
        for ch in pattern:
            blk = nn.Module()
            blk.ln1, blk.ln2 = nn.LayerNorm(width), nn.LayerNorm(width)
            blk.mix = HierMix(width) if ch == "H" else CausalSelfAttention(width, heads)
            blk.mlp = nn.Sequential(nn.Linear(width, mlp_mult * width), nn.GELU(), nn.Linear(mlp_mult * width, width))
            nn.init.normal_(blk.mlp[2].weight, std=0.02 / math.sqrt(2 * n))
            out = blk.mix.combine[2] if ch == "H" else blk.mix.c_proj
            nn.init.normal_(out.weight, std=0.02 / math.sqrt(2 * n))
            self.blocks.append(blk)
        self.final_norm = nn.LayerNorm(width)

    def forward(self, idx, targets=None, state=None):
        x = self.embedding(idx)
        for ch, b in zip(self.pattern, self.blocks):
            if ch == "H":
                h, _ = b.mix(b.ln1(x))
            else:
                h = b.mix(b.ln1(x))
            x = x + h
            x = x + b.mlp(b.ln2(x))
        logits = F.linear(self.final_norm(x), self.embedding.weight)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss, None


def detach_state(st):
    if isinstance(st, list):
        return [detach_state(x) for x in st]
    return {k: (v.detach() if torch.is_tensor(v) else v) for k, v in st.items()}


def hier_params(V, d):
    return V * d + 32 * d * d + 23 * d


def build(arm, cfg):
    if arm.startswith("nanogpt"):
        from nanogpt_model import GPT, GPTConfig
        m = GPT(GPTConfig(block_size=cfg["context"], vocab_size=VOCAB, n_layer=cfg["layers"],
                          n_head=cfg["heads"], n_embd=cfg["width"], dropout=0.0, bias=True))
        return m
    if arm.startswith("hybrid"):
        return HybridHierarchy(VOCAB, cfg["width"], cfg["pattern"], cfg["heads"], cfg.get("mlp_mult", 4))
    if arm.startswith("deep_hier"):
        return DeepHierarchy(VOCAB, cfg["width"], cfg["layers"], cfg.get("mlp_mult", 4))
    return FastHierarchy(VOCAB, cfg["width"], cfg.get("segment_length", 32), cfg.get("document_segments", 8))


# ----------------------------------------------------------------------------- eval
@torch.no_grad()
def evaluate(model, arm, split, T, tokens, stateful, B=32):
    model.eval()
    n = (tokens // (B * T)) * B * T
    data = torch.from_numpy(np.asarray(split[:n + 1]).astype(np.int64)).cuda()
    total, count = 0.0, 0
    if stateful:
        lane = n // B
        xs = torch.stack([data[b * lane:(b + 1) * lane + 1] for b in range(B)])
        st = None
        for o in range(0, lane - T + 1, T):
            x, y = xs[:, o:o + T], xs[:, o + 1:o + T + 1]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _, st = model(x, None, st)
            total += F.cross_entropy(logits.float().view(-1, VOCAB), y.reshape(-1), reduction="sum").item()
            count += y.numel()
    else:
        w = data[:n].view(-1, T)
        tgt = data[1:n + 1].view(-1, T)
        for i in range(0, w.shape[0], B):
            x, y = w[i:i + B], tgt[i:i + B]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(x, y) if arm.startswith("nanogpt") else model(x, None)
                logits = out[0]
            total += F.cross_entropy(logits.float().view(-1, VOCAB), y.reshape(-1), reduction="sum").item()
            count += y.numel()
    model.train()
    return total / count


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)  # JSON string or path
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text()) if os.path.exists(a.config) else json.loads(a.config)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if (out / "result.json").exists():
        print("exists"); return
    (out / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True))
    seed, arm, T, B = cfg["seed"], cfg["arm"], cfg["context"], cfg["batch"]
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    data = Data(cfg.get("data_root", "/work/data"), cfg.get("train_shards", 10))
    model = build(arm, cfg).cuda()
    nparams = sum(p.numel() for p in model.parameters())
    stateful = bool(cfg.get("stateful", False))
    # identical optimizer recipe for all arms (nanoGPT's: decay 2D+ params only)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    nodecay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg["weight_decay"]},
                             {"params": nodecay, "weight_decay": 0.0}],
                            lr=cfg["lr"], betas=tuple(cfg["betas"]), fused=True)
    steps = cfg["tokens"] // (B * T)
    warm = max(1, int(cfg.get("warmup_frac", 0.02) * steps))
    lr_at = lambda s: cfg["lr"] * (s + 1) / warm if s < warm else cfg["lr"] * (
        cfg["min_lr_ratio"] + (1 - cfg["min_lr_ratio"]) * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    fwd = model
    if arm.startswith("nanogpt") and cfg.get("compile", True):
        fwd = torch.compile(model)
    rng = np.random.default_rng(seed)
    lanes = Lanes(data, B, T, seed) if stateful else None
    st = None
    log = open(out / "train_log.jsonl", "a")
    eval_every = cfg.get("eval_every", max(1, steps // 10))
    curve = []
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time(); tok_time = 0.0
    for s in range(steps):
        x, y = lanes.batch() if stateful else data.batch(rng, B, T)
        for g in opt.param_groups:
            g["lr"] = lr_at(s)
        torch.cuda.synchronize(); ts = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if arm.startswith("nanogpt"):
                _, loss = fwd(x, y)
            else:
                _, loss, st = fwd(x, y, st if stateful else None)
                if stateful:
                    st = detach_state(st)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); tok_time += time.time() - ts
        if s % 50 == 0 or s == steps - 1:
            rec = {"step": s, "tokens": (s + 1) * B * T, "loss": loss.item(), "lr": lr_at(s), "gnorm": gn.item(),
                   "elapsed": time.time() - t0, "tok_per_s": (s + 1) * B * T / tok_time}
            log.write(json.dumps(rec) + "\n"); log.flush(); print(rec, flush=True)
        if (s + 1) % eval_every == 0 and s + 1 < steps:
            v = evaluate(model, arm, data.selection, T, cfg.get("curve_eval_tokens", 2_000_000), stateful)
            curve.append({"step": s + 1, "tokens": (s + 1) * B * T, "selection_ce": v})
            log.write(json.dumps({"curve": curve[-1]}) + "\n"); log.flush(); print(curve[-1], flush=True)
    train_time = time.time() - t0
    peak = torch.cuda.max_memory_allocated()
    sel = evaluate(model, arm, data.selection, T, cfg["eval_tokens"], stateful)
    res = {"arm": arm, "seed": seed, "context": T, "params": nparams, "tokens": steps * B * T, "steps": steps,
           "selection_ce": sel, "curve": curve + [{"step": steps, "tokens": steps * B * T, "selection_ce": sel}],
           "train_seconds": train_time, "train_tok_per_s": steps * B * T / tok_time,
           "peak_train_mem_bytes": peak, "stateful": stateful, "config": cfg}
    if cfg.get("confirm", False):
        res["confirmation_ce"] = evaluate(model, arm, data.confirmation, T, cfg["eval_tokens"], stateful)
        if not arm.startswith("nanogpt"):
            res["confirmation_ce_reset_windows"] = evaluate(model, arm, data.confirmation, T, cfg["eval_tokens"], False)
            res["confirmation_ce_streaming"] = evaluate(model, arm, data.confirmation, T, cfg["eval_tokens"], True)
    ck = out / "final.pt"
    torch.save(model.state_dict(), ck)
    res["checkpoint_sha256"] = hashlib.sha256(ck.read_bytes()).hexdigest()
    (out / "result.json").write_text(json.dumps(res, indent=2))
    print("RESULT", json.dumps({k: v for k, v in res.items() if k not in ("curve", "config")}), flush=True)


if __name__ == "__main__":
    main()
