"""Merge batch-1 (transcribed, files lost) and batch-2 (synced result.json) into METRICS.json."""
import json, glob, pathlib
H = pathlib.Path(__file__).resolve().parent
b1 = json.loads((H / "RECOVERED_BATCH1.json").read_text())
M = {"main": {k: list(v) for k, v in b1["main"].items()}, "lr_trials": {k: list(v) for k, v in b1["lr_trials"].items()},
     "lr_trial_curves": b1["lr_trial_curves"], "throughput_tok_per_s_H100_bf16": dict(b1["throughput_tok_per_s_H100_bf16"]),
     "params": {"nanogpt": 124475904, "hier": 125442240, "hier_stateful": 125442240, "deep_hier": 123784896, "hybrid": 121624736},
     "sources": {"batch1": "RECOVERED_BATCH1.json (transcribed)", "batch2": "batch2/runs/*/result.json"}}
for f in sorted(glob.glob(str(H / "batch2/runs/*/result.json"))):
    r = json.loads(open(f).read()); name = pathlib.Path(f).parent.name
    cfg = r["config"]; arm = "hybrid1" if cfg.get("pattern") == "HAAAAAAAAA" else r["arm"]
    M["params"][arm] = r["params"]; M["throughput_tok_per_s_H100_bf16"][arm] = round(r["train_tok_per_s"])
    if name.startswith("lr_"):
        M["lr_trials"].setdefault(arm, []).append({"lr": cfg["lr"], "selection_ce": r["selection_ce"], "checkpoint_sha256": r["checkpoint_sha256"]})
        M["lr_trial_curves"][f"{arm}_{cfg['lr']:g}"] = r["curve"]
    else:
        row = {k: r[k] for k in ("seed", "confirmation_ce", "selection_ce", "curve", "checkpoint_sha256", "train_seconds", "peak_train_mem_bytes") if k in r}
        M["main"].setdefault(arm, []).append(row)
for arm in M["main"]:
    M["main"][arm].sort(key=lambda x: x["seed"])
(H / "METRICS.json").write_text(json.dumps(M, indent=1))
for arm, rows in M["main"].items():
    print(arm, [(x["seed"], round(x["confirmation_ce"], 4)) for x in rows])
