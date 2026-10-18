"""Exact-equation parity: FastHierarchy vs recovered HierarchicalDistributionStudent (FP32, TF32 off).
Preregistered tolerances: logits max|d| <= 1e-4, loss |d| <= 1e-5, every grad max|d| <= 1e-4 * max(1, max|g_ref|),
final fast/segment/document state max|d| <= 1e-5. Nonzero (random, scaled) weights; several seeds/lengths/batches."""
import json, sys, torch
sys.path.insert(0, "/work/src")
from representation_experiments.predictive_models import HierarchicalDistributionStudent
from train import FastHierarchy
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
rows = []
for seed, B, T, d, scale in [(1, 3, 256, 64, 1.0), (2, 2, 512, 96, 3.0), (3, 4, 320, 48, 0.3), (4, 1, 1024, 32, 2.0)]:
    torch.manual_seed(seed)
    ref = HierarchicalDistributionStudent(1000, d, 4096, 32, 8).cuda().double().float()
    with torch.no_grad():
        for n, p in ref.named_parameters():
            if "cell" in n or "combine" in n:
                p.mul_(scale)
    fast = FastHierarchy(1000, d).cuda()
    fast.load_state_dict(ref.state_dict(), strict=True)
    x = torch.randint(0, 1000, (B, T), device="cuda"); y = torch.randint(0, 1000, (B, T), device="cuda")
    o = ref(x); lr_ = torch.nn.functional.cross_entropy(o.logits.reshape(-1, 1000), y.reshape(-1)); lr_.backward()
    lf, loss_f, st = fast(x, y); loss_f.backward()
    g = max((p1.grad - p2.grad).abs().max().item() / max(1.0, p1.grad.abs().max().item())
            for p1, p2 in zip(ref.parameters(), fast.parameters()))
    s = max((o.final_state[k] - st[k]).abs().max().item() for k in ("fast", "segment", "document"))
    r = {"seed": seed, "B": B, "T": T, "d": d, "scale": scale,
         "logit_maxabs": (o.logits - lf).abs().max().item(), "loss_absdiff": abs(lr_.item() - loss_f.item()),
         "grad_rel_maxabs": g, "state_maxabs": s, "segments_counter_equal": o.final_state["segments"] == st["segments"]}
    r["pass"] = r["logit_maxabs"] <= 1e-4 and r["loss_absdiff"] <= 1e-5 and g <= 1e-4 and s <= 1e-5 and r["segments_counter_equal"]
    rows.append(r); print(r, flush=True)
json.dump({"tolerances": {"logits": 1e-4, "loss": 1e-5, "grad_rel": 1e-4, "state": 1e-5}, "rows": rows,
           "all_pass": all(r["pass"] for r in rows)}, open(sys.argv[1], "w"), indent=2)
print("ALL_PASS", all(r["pass"] for r in rows))
