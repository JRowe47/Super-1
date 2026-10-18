"""Print per-seed confirmation CE, paired gaps vs nanoGPT, and perplexity from runs/*/result.json."""
import json, glob, math, os, sys
root = sys.argv[1] if len(sys.argv) > 1 else "runs"
rows = {}
for f in glob.glob(f"{root}/main_*/result.json"):
    name = os.path.basename(os.path.dirname(f)); r = json.load(open(f))
    arm, seed = name[len("main_"):].rsplit("_s", 1)
    rows.setdefault(arm, {})[int(seed)] = r["confirmation_ce"]
nano = rows.get("nanogpt", {})
for arm, d in sorted(rows.items()):
    m = sum(d.values()) / len(d)
    line = f"{arm:10s} " + " ".join(f"s{s}={v:.4f}" for s, v in sorted(d.items())) + f"  mean={m:.4f} ppl={math.exp(m):.2f}"
    gaps = [d[s] - nano[s] for s in d if s in nano]
    if arm != "nanogpt" and gaps:
        g = sum(gaps) / len(gaps)
        line += f"  vs nanoGPT {g:+.4f} nats ({100 * g / (sum(nano[s] for s in d if s in nano) / len(gaps)):+.2f}%), {sum(x < 0 for x in gaps)}/{len(gaps)} seeds better"
    print(line)
