"""Render charts for matched-scratch-fineweb-20261017-v1 from METRICS.json only (no training or data access)."""
import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

HERE = pathlib.Path(__file__).resolve().parent
M = json.loads((HERE.parents[1] / "results/matched-scratch-fineweb-20261017-v1/METRICS.json").read_text())
COL = {"nanogpt": "#2a78d6", "hier": "#eb6834", "deep_hier": "#1baf7a", "hybrid": "#eda100", "hier_stateful": "#e87ba4", "hybrid1": "#6250d6"}
NAME = {"nanogpt": "nanoGPT (pinned)", "hier": "Original hierarchy", "deep_hier": "Deep hierarchy (new arm)",
        "hybrid": "Hierarchy + attention hybrid (new arm)", "hier_stateful": "Original hierarchy, stateful lanes", "hybrid1": "1 hierarchy + 9 attention blocks (new arm)"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "#fcfcfb",
                     "axes.facecolor": "#fcfcfb"})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(HERE / f"{name}.svg"); fig.savefig(HERE / f"{name}.png", dpi=160); plt.close(fig)


# 1) main-run learning curves: every seed thin, mean bold, direct labels
fig, ax = plt.subplots(figsize=(8, 4.6))
labels_end = []
for arm, runs in M["main"].items():
    full = [r for r in runs if len(r["curve"]) >= 5]
    for r in full:
        xs = [c["tokens"] / 1e6 for c in r["curve"]]; ys = [c["selection_ce"] for c in r["curve"]]
        ax.plot(xs, ys, color=COL[arm], lw=2 if len(full) == 1 else 1, alpha=1 if len(full) == 1 else 0.45,
                label=f"{NAME[arm]} (seed {r['seed']}): {r['curve'][-1]['selection_ce']:.3f}" if len(full) == 1 else None)
    if len(full) > 1:
        n = min(len(r["curve"]) for r in full)
        xs = [full[0]["curve"][i]["tokens"] / 1e6 for i in range(n)]
        ys = [sum(r["curve"][i]["selection_ce"] for r in full) / len(full) for i in range(n)]
        ax.plot(xs, ys, color=COL[arm], lw=2, label=f"{NAME[arm]} (mean of {len(full)} seeds): {ys[-1]:.3f}")
    if full:
        last = full[0]["curve"][-1] if len(full) == 1 else {"tokens": xs[-1] * 1e6, "selection_ce": ys[-1]}
        labels_end.append((last["selection_ce"], last["tokens"] / 1e6, COL[arm]))

ax.set_xlabel("Training tokens (millions)"); ax.set_ylabel("Selection cross-entropy (nats, lower is better)")
ax.set_ylim(3.4, 5.2); ax.grid(axis="y", color=GRID, lw=0.8); ax.legend(frameon=False, loc="upper right")
ax.set_title("FineWeb, ~124M params, context 1024 (train = eval); seeds with full saved curves", color=INK, loc="left", fontsize=11)
save(fig, "learning_curves")

# 2) per-seed confirmation CE
fig, ax = plt.subplots(figsize=(7, 3.6))
arms = list(M["main"])
for i, arm in enumerate(arms):
    vals = [r["confirmation_ce"] for r in M["main"][arm]]
    for j, v in enumerate(vals):
        ax.scatter(v, i + (j - 1) * 0.12, s=64, color=COL[arm], edgecolor="#fcfcfb", linewidth=2, zorder=3)
    mean = sum(vals) / len(vals)
    ax.plot([mean, mean], [i - 0.3, i + 0.3], color=INK, lw=2)
    ax.annotate(f"mean {mean:.4f}", (max(vals), i), xytext=(10, 0), textcoords="offset points", va="center", color=INK, fontsize=9)
ax.set_yticks(range(len(arms))); ax.set_yticklabels([NAME[a] for a in arms])
ax.set_xlabel("Confirmation cross-entropy (nats, lower is better), 33.6M held-out tokens")
ax.grid(axis="x", color=GRID, lw=0.8); ax.invert_yaxis()
ax.set_title("Final held-out CE per seed (dots) and mean (bar)", color=INK, loc="left", fontsize=11)
save(fig, "confirmation_per_seed")

# 3) LR trials
fig, ax = plt.subplots(figsize=(7, 3.6))
for arm, rows in M["lr_trials"].items():
    rows = sorted([r for r in rows if r["selection_ce"] is not None], key=lambda r: r["lr"])
    ax.plot([r["lr"] for r in rows], [r["selection_ce"] for r in rows], marker="o", ms=7, lw=2, color=COL[arm], label=NAME[arm])
ax.set_xscale("log"); ax.set_xlabel("Peak learning rate"); ax.set_ylabel("Selection CE @105M tokens")
ax.set_ylim(4.4, 6.0); ax.set_xticks([3e-4, 1e-3, 3e-3]); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.set_xticklabels(["3e-4", "1e-3", "3e-3"]); ax.grid(axis="y", color=GRID, lw=0.8); ax.legend(frameon=False, fontsize=8)
ax.set_title("Equal LR-trial budget: 3 trials per arm, seed 1 (values >6 clipped)", color=INK, loc="left", fontsize=11)
save(fig, "lr_trials")
# 4) paired difference vs nanoGPT, same seed
fig, ax = plt.subplots(figsize=(8.5, 3.0))
nano = {r["seed"]: r["confirmation_ce"] for r in M["main"]["nanogpt"]}
arms = [a for a in M["main"] if a not in ("nanogpt", "hier")]
for i, arm in enumerate(arms):
    diffs = [(r["seed"], r["confirmation_ce"] - nano[r["seed"]]) for r in M["main"][arm] if r["seed"] in nano]
    for seed, v in diffs:
        ax.scatter(v, i, s=64, color=COL[arm], edgecolor="#fcfcfb", linewidth=2, zorder=3)
    mean = sum(v for _, v in diffs) / len(diffs)
    ax.annotate(f"mean {mean:+.4f} (n={len(diffs)})", (max(v for _, v in diffs), i), xytext=(10, 0),
                textcoords="offset points", va="center", color=INK, fontsize=9)
ax.axvline(0, color=MUTED, lw=1); ax.axvline(-0.02, color=MUTED, lw=1, ls=":")
ax.annotate("min. useful effect -0.02", (-0.02, -0.35), xytext=(-4, 0), textcoords="offset points", ha="right", color=MUTED, fontsize=8)
ax.set_yticks(range(len(arms))); ax.set_yticklabels([NAME[a] for a in arms])
ax.set_xlim(-0.035, 0.14); ax.set_ylim(len(arms) - 0.5, -0.5)
ax.set_xlabel("Confirmation CE minus nanoGPT (nats; negative = beats nanoGPT). Original hierarchy: +0.568, off-scale")
ax.grid(axis="x", color=GRID, lw=0.8)
ax.set_title("Gap to nanoGPT, same seed, 500M tokens", color=INK, loc="left", fontsize=11)
save(fig, "paired_gap")
print("rendered")
