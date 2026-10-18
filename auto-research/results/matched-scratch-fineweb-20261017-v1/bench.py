"""Measured prefill latency / peak memory on the same H100, bf16 autocast, same harness for all arms.
Not decode latency (pinned nanoGPT has no KV cache). Analytic KV/state bytes reported separately."""
import json, sys, time, torch
sys.path.insert(0, "/work/src")
import train
rows = []
for name in sys.argv[2:]:
    cfg = json.load(open(f"/work/runs/{name}/config.json"))
    m = train.build(cfg["arm"], cfg).cuda().eval()
    m.load_state_dict(torch.load(f"/work/runs/{name}/final.pt", map_location="cuda", weights_only=True))
    weight_bytes = sum(p.numel() * p.element_size() for p in m.parameters())
    for B in (1, 16):
        x = torch.randint(0, 50257, (B, 1024), device="cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(5): m(x)
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            ts = []
            for _ in range(30):
                torch.cuda.synchronize(); t = time.perf_counter(); m(x); torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
        ts.sort()
        rows.append({"run": name, "arm": cfg["arm"], "batch": B, "context": 1024, "p50_ms": 1e3 * ts[15], "p95_ms": 1e3 * ts[28],
                     "prefill_tok_per_s": B * 1024 / ts[15], "peak_alloc_bytes": torch.cuda.max_memory_allocated(),
                     "peak_minus_resident_bytes": torch.cuda.max_memory_allocated() - base, "fp32_weight_bytes": weight_bytes})
        print(rows[-1], flush=True)
    del m; torch.cuda.empty_cache()
json.dump(rows, open(sys.argv[1], "w"), indent=2)
