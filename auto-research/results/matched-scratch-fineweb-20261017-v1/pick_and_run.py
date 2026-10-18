import json, glob, subprocess, sys
arm, prefix = sys.argv[1], sys.argv[2]
res = [json.load(open(f)) for f in glob.glob(f"/persist/runs/lr_{prefix}_*/result.json")]
best = min(res, key=lambda r: r["selection_ce"])
print("selected lr", best["config"]["lr"], best["selection_ce"], flush=True)
cfg = {k: v for k, v in best["config"].items()}
cfg.update(tokens=7680*65536, eval_tokens=33554432, curve_eval_tokens=4194304, confirm=True, eval_every=768)
jobs = []
for s in [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "1,2,3").split(",")]:
    jobs.append({**cfg, "seed": s, "name": f"main_{prefix}_s{s}"})
json.dump(jobs, open(f"/work/q_main_{prefix}_{len(jobs)}{jobs[0]['seed']}.json", "w"))
subprocess.run([sys.executable, "/work/runq.py", f"/work/q_main_{prefix}_{len(jobs)}{jobs[0]['seed']}.json"])
