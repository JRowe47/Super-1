"""Float64 equation-identity parity (declared after FP32 row 2 missed the 1e-5 state tolerance; FP32 failure retained).
Tolerance declared before running: every logit/loss/grad/state max|d| <= 1e-9."""
import json, sys, torch
sys.path.insert(0, "/work/src")
from representation_experiments.predictive_models import HierarchicalDistributionStudent
from train import FastHierarchy
rows = []
for seed, B, T, d, scale in [(1, 3, 256, 64, 1.0), (2, 2, 512, 96, 3.0), (3, 4, 320, 48, 0.3), (4, 1, 1024, 32, 2.0)]:
    torch.manual_seed(seed)
    ref = HierarchicalDistributionStudent(1000, d, 4096, 32, 8).cuda()
    with torch.no_grad():
        for n, p in ref.named_parameters():
            if "cell" in n or "combine" in n: p.mul_(scale)
    ref = ref.double()
    fast = FastHierarchy(1000, d).cuda().double(); fast.load_state_dict(ref.state_dict(), strict=True)
    x = torch.randint(0, 1000, (B, T), device="cuda"); y = torch.randint(0, 1000, (B, T), device="cuda")
    o = ref(x); l1 = torch.nn.functional.cross_entropy(o.logits.reshape(-1, 1000), y.reshape(-1)); l1.backward()
    lf, l2, st = fast(x, y); l2.backward()
    g = max((a.grad - b.grad).abs().max().item() for a, b in zip(ref.parameters(), fast.parameters()))
    s = max((o.final_state[k] - st[k]).abs().max().item() for k in ("fast", "segment", "document"))
    r = {"seed": seed, "T": T, "scale": scale, "logit": (o.logits - lf).abs().max().item(), "loss": abs(l1.item() - l2.item()), "grad": g, "state": s}
    r["pass"] = max(r["logit"], r["loss"], g, s) <= 1e-9; rows.append(r); print(r)
json.dump({"tolerance": 1e-9, "rows": rows, "all_pass": all(r["pass"] for r in rows)}, open(sys.argv[1], "w"), indent=2)
print("ALL_PASS", all(r["pass"] for r in rows))
