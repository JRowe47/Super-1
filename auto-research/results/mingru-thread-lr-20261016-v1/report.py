"""Plot all declared trials and report the small thread/LR screen."""
import gzip, json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent; REPO = ROOT.parents[2]
p = json.loads((ROOT/'PROTOCOL.json').read_text()); s = json.loads((ROOT/'SUMMARY.json').read_text())
rows = s['rows']; by_id = {r['run_id']:r for r in rows}; best = min(rows, key=lambda r:r['ce'])
base = by_id['b1_t1_lr002']
def label(r): return f"B{r['batch']} / T{r['threads']} / LR {r['peak_lr']:g}"
palette = {(1,1):'#414b62', (1,2):'#087f8c', (1,4):'#34843c', (4,2):'#bb650c', (16,4):'#8151ad'}
out = REPO/'auto-research/graphs'/ROOT.name; out.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False, 'svg.fonttype':'none'})
fig, axes = plt.subplots(1, 2, figsize=(12, 5.1), gridspec_kw={'width_ratios':[1.2,1]})
for r in sorted(rows, key=lambda r:(r['batch'],r['threads'],r['peak_lr'])):
    path = ROOT/'execution'/r['run_id']/'curves.json'
    c = json.loads(path.read_text()) if path.exists() else json.loads(gzip.decompress(path.with_suffix('.json.gz').read_bytes()))
    axes[0].plot([x['train_wall_seconds'] for x in c['monitor']], [x['validation_ce'] for x in c['monitor']],
        color=palette[(r['batch'],r['threads'])], ls='--' if r['peak_lr']>.002 else '-', marker='o', markersize=3, lw=2, label=label(r))
axes[0].set(xlabel='Training wall seconds', ylabel='Monitoring CE · lower is better', xlim=(0,61), title='Learning within the same time allowance')
axes[0].legend(fontsize=8, frameon=False, ncol=1); axes[0].grid(alpha=.18)
for i, r in enumerate(rows):
    axes[1].scatter(r['ce'], i, s=65, color=palette[(r['batch'],r['threads'])], marker='D' if r['peak_lr']>.002 else 'o')
    axes[1].annotate(f"{r['ce']:.3f} · {r['updates']:,} updates", (r['ce'],i), xytext=(8,0), textcoords='offset points', va='center', fontsize=9)
axes[1].set(yticks=range(len(rows)), yticklabels=[label(r) for r in rows], xlabel='Final validation CE · lower is better',
    title='Final loss and completed updates', xlim=(min(r['ce'] for r in rows)-.08,max(r['ce'] for r in rows)+.4))
axes[1].invert_yaxis(); axes[1].grid(axis='x', alpha=.18)
fig.suptitle('Linear minGRU · 996,320 parameters · 60-second CPU training allowance', fontsize=13, fontweight='bold')
fig.text(.5,.025,'One reused seed. Monitoring: 4,096 targets; final ranking: 16,384 separate targets. Dashed / diamond = increased peak LR.',ha='center',fontsize=9,color='#505965')
fig.tight_layout(rect=(0,.06,1,.94)); fig.savefig(out/'loss_curves.png',dpi=170); fig.savefig(out/'loss_curves.svg'); plt.close(fig)
table = '\n'.join(f"| {r['batch']} / {r['threads']} | {r['peak_lr']:g} | {r['updates']:,} | {r['training_targets']:,} | {r['train_wall_seconds']:.2f} | {r['train_cpu_seconds']:.1f} | {r['ce']:.6f} |" for r in rows)
contrasts = '\n'.join(f"- **{label(by_id[c['trial']])} vs {label(by_id[c['control']])}:** CE {c['ce_change']:+.4f}; perplexity {c['perplexity_change_percent']:+.1f}%; {c['throughput_ratio']:.2f}× target throughput, {c['update_ratio']:.2f}× optimizer updates, {c['cpu_ratio']:.2f}× aggregate training CPU time." for c in s['contrasts'])
drift = '\n'.join(f"- {label(by_id[d['run_id']])}: previous CE {d['previous_ce']:.4f}, fresh CE {d['fresh_ce']:.4f}; fresh/previous throughput {d['throughput_ratio']:.3f}×." for d in s['fresh_control_vs_previous'])
failures = 'No declared trial failed.' if not s['failures'] else '\n'.join(f"- {f['run_id']}: {f['error_type']} — {f['error']}" for f in s['failures'])
text = f'''# minGRU: small batches, threads and learning rate

**Best measured endpoint: {label(best)}, CE {best['ce']:.6f}.** The fresh B1/T1/LR0.002 reference reaches CE {base['ce']:.6f}. The winning configuration changes perplexity by {100*(best['perplexity']/base['perplexity']-1):+.1f}% versus that reference within the same 60-second training-wall allowance. This is a seven-trial explorer screen with one reused seed, not a tuned optimum or a seed replication.

Research PR target: **JRowe47/Super-1 → main**, using the existing draft PR. Upstream submission requires the user's explicit go-ahead.

## Results

| Batch / threads | Peak LR | Updates | Target exposures | Training wall s | Training CPU s | Final CE ↓ |
|---|---:|---:|---:|---:|---:|---:|
{table}

All rows use the unchanged 996,320-parameter linear minGRU hierarchy, initialization seed 101603, data-order seed 101612, and the same fixed 32-window / 16,384-target final validation region as the preceding study. Validation has been repeatedly viewed; it is not an untouched test. {failures}

![Loss curves and endpoint ranking](../graphs/{ROOT.name}/loss_curves.png)

The learning curves use a separate eight-window / 4,096-target monitoring region. Table and right-panel endpoints use the larger final region. A curve endpoint and its table loss are therefore different measurements. All declared trials are reported; no best-checkpoint selection.

## Planned contrasts

{contrasts}

These comparisons isolate the declared setting changes within a common implementation and time allowance, but different runtime speeds also change the number of updates and consumed windows. They do not separately identify data efficiency, optimizer-step efficiency or a mathematical scaling law.

Fresh controls compared with the preceding batch/thread study:

{drift}

Use the fresh within-study controls for the main comparisons. The two reruns help expose timing drift but do not estimate seed variance. Ordinary GRU and nanoGPT were not retrained in this pass, so the new ranking is among minGRU configurations only. The preceding [batch/thread report](gru-batch-wall-20261016-v1.md) contains the architecture comparisons.

## What was held fixed

- **Data:** exact hash-bound uint16 arrays and train-only 2048-BPE tokenizer from the preceding study. Each example predicts the next token over a reset 512-token window with full BPTT. Each batch takes the next B consecutive windows of the same frozen stream. Batch size changes grouping, exposures per update and total exposures within the clock.
- **Model:** signed linear candidate and input-only gate, affine scan within 32-token chunks, with the same segment/document GRUs and readout. No new recurrent, attention or scan implementation. The existing parent logits/gradient parity checks remain applicable; this pass changes configuration and schedule only.
- **Optimizer:** AdamW betas 0.9/0.95, epsilon 1e-8, weight decay 0.1 with the same exclusions, global gradient clip 1. Peak LR is declared per row. Warmup occupies 5% of measured training time with initial factor floor 0.04; cosine decay then reaches 10% of the peak at 60 seconds. The 2× and 4× LR choices are quick trials, not a claimed Adam batch-scaling rule.
- **Execution:** one worker at a time; thread 1 pinned to CPU0, thread 2 to CPUs0–1, thread 4 to CPUs0–3. CPU FP32, deterministic PyTorch algorithms, one interop thread. Training time includes data construction, forward, CE, backward, clipping and AdamW. Startup, evaluation and checkpoint IO are outside this clock. Separate `run_wall_seconds` / `run_cpu_seconds` fields cover initial monitoring through final evaluation, including checkpoint IO; they exclude process startup, data verification and model/optimizer construction. The inherited protocol phrase “full-run” is broader than these actual fields: they are not end-to-end launcher latency. Equal wall allowance with more threads consumes more aggregate CPU, as shown above; it is not equal compute.
- **Stopping:** stop before the next update when remaining time is at most max(0.2 seconds, twice the largest of the last ten updates). Actual training times are shown. Validation does not affect stopping. The initial estimate reuses the preceding benchmark for that batch/thread cell; no benchmark weights are reused.
- **Ordering:** all seven fresh trials and their seeded shuffled order were frozen before training. Frozen source hashes bind the parent helpers and architecture. No new paid resources, GPU run or official-test evaluation.

## Checks and artifacts

Final checkpoints replay every final-window loss within 5e-6. The analysis reconciles data/tokenizer/source hashes, stream identity, parameter counts, all update and target counts, the learning rate at every step, accumulated wall/CPU time, finite losses/gradients and the stopping margin. Checkpoints retain model weights, AdamW state, Torch RNG and complete training traces. No extra model-parity grid was run because the forward/backward implementation is unchanged from the preceding checked study.

- [Frozen protocol](../results/{ROOT.name}/PROTOCOL.json), [registration](../results/{ROOT.name}/REGISTRATION.json), [summary](../results/{ROOT.name}/SUMMARY.json), [checks](../results/{ROOT.name}/CHECKS.json), [terminal state](../results/{ROOT.name}/TERMINAL.json).
- [Runner](../results/{ROOT.name}/experiment.py), [sequential launcher](../results/{ROOT.name}/launch.py), [analysis](../results/{ROOT.name}/analyze.py), [plot/report generator](../results/{ROOT.name}/report.py). The runner imports unchanged helpers from the preceding batch/thread study, which imports the original GRU model. Keep those sibling directories together.
- Checkpoint-only archive `mingru_thread_lr_checkpoints_20261016.zip`; exact arrays remain in `gru_extended_checkpoints_20261016.zip` (SHA256 c90bb924b3b69e49bf217582776e7add78292d8c72e8a2fa5e82bd28a6db80a6). These contributor-held payloads are outside Git. See [preservation metadata](../results/{ROOT.name}/PRESERVATION.json).

```bash
python auto-research/results/{ROOT.name}/report.py
# Training replay: separate copy/output directory, retaining frozen protocol and dependencies,
# with no completed execution artifacts. Do not overwrite the published run.
python auto-research/results/{ROOT.name}/launch.py --data-dir /path/to/exact/data
```

The time-based schedule depends on measured hardware speed, so a training replay need not reproduce the identical trajectory. Saved checkpoint evaluation is the exact replay check; complete observed traces are published as `curves.json.gz`.
'''
(REPO/'auto-research/reports'/f'{ROOT.name}.md').write_text(text)
print('REPORT', best['run_id'], best['ce'])
