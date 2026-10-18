"""Plot every declared three-minute recipe and document the measured trade-offs."""
import gzip, json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent; REPO=ROOT.parents[2]
p=json.loads((ROOT/'PROTOCOL.json').read_text()); s=json.loads((ROOT/'SUMMARY.json').read_text())
rows=s['rows']; by_id={r['run_id']:r for r in rows}; best=min(rows,key=lambda r:r['ce'])
names={'mingru_linear':'Linear minGRU','native':'Ordinary GRU','nanogpt':'nanoGPT'}
colors={'mingru_linear':'#087f8c','native':'#af6418','nanogpt':'#8051ad'}
def label(r): return f"{names[r['arm']]} · B{r['batch']}/T{r['threads']} · LR {r['peak_lr']:g}"
def dashed(r): return r['batch']==16 or (r['arm']=='nanogpt' and r['peak_lr']==.004)
curves={}
for r in rows:
    path=ROOT/'execution'/r['run_id']/'curves.json'
    curves[r['run_id']]=json.loads(path.read_text()) if path.exists() else json.loads(gzip.decompress(path.with_suffix('.json.gz').read_bytes()))
out=REPO/'auto-research/graphs'/ROOT.name; out.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
fig,axes=plt.subplots(1,2,figsize=(13,5.4),gridspec_kw={'width_ratios':[1.2,1]})
for r in sorted(rows,key=lambda r:(r['arm'],r['batch'],r['peak_lr'])):
    c=curves[r['run_id']]['monitor']
    axes[0].plot([x['train_wall_seconds'] for x in c],[x['validation_ce'] for x in c],
        color=colors[r['arm']],ls='--' if dashed(r) else '-',marker='o',markersize=3,lw=2,label=label(r))
axes[0].set(xlabel='Training wall seconds',ylabel='Monitoring CE · lower is better',xlim=(0,183),title='Learning within the same time allowance')
axes[0].legend(fontsize=7.5,frameon=False); axes[0].grid(alpha=.18)
for i,r in enumerate(rows):
    axes[1].scatter(r['ce'],i,s=65,color=colors[r['arm']],marker='D' if dashed(r) else 'o')
    axes[1].annotate(f"{r['ce']:.3f} · {r['updates']:,} updates",(r['ce'],i),xytext=(8,0),textcoords='offset points',va='center',fontsize=9)
axes[1].set(yticks=range(len(rows)),yticklabels=[label(r) for r in rows],xlabel='Final validation CE · lower is better',
    title='Final loss and completed updates',xlim=(min(r['ce'] for r in rows)-.08,max(r['ce'] for r in rows)+.5))
axes[1].tick_params(axis='y',labelsize=8)
axes[1].invert_yaxis(); axes[1].grid(axis='x',alpha=.18)
fig.suptitle('Longer CPU comparison · ~1M parameters · 180-second training allowance',fontsize=14,fontweight='bold')
fig.text(.5,.025,'Fresh seed 101604. Monitoring: 4,096 targets; final ranking: 16,384 separate reused validation targets.',ha='center',fontsize=9,color='#505965')
fig.tight_layout(rect=(0,.06,1,.94)); fig.savefig(out/'loss_curves.png',dpi=170); fig.savefig(out/'loss_curves.svg'); plt.close(fig)

fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
for r in sorted(rows,key=lambda r:(r['arm'],r['batch'],r['peak_lr'])):
    c=curves[r['run_id']]['monitor']
    ax.plot([x['train_cpu_seconds'] for x in c],[x['validation_ce'] for x in c],color=colors[r['arm']],
        ls='--' if dashed(r) else '-',marker='o',markersize=3,lw=2,label=label(r))
ax.set(xlabel='Aggregate training CPU seconds',ylabel='Monitoring CE · lower is better',title='Same trajectories viewed by consumed CPU time')
ax.legend(fontsize=8,frameon=False); ax.grid(alpha=.18)
fig.savefig(out/'cpu_curves.png',dpi=170); fig.savefig(out/'cpu_curves.svg'); plt.close(fig)

table='\n'.join(f"| {names[r['arm']]} | {r['batch']} / {r['threads']} | {r['peak_lr']:g} | {r['updates']:,} | {r['training_targets']:,} | {r['train_wall_seconds']:.2f} | {r['train_cpu_seconds']:.1f} | {r['ce']:.6f} |" for r in rows)
contrasts='\n'.join(f"- **{label(by_id[c['trial']])} versus {label(by_id[c['control']])}:** CE {c['ce_change']:+.4f}, perplexity {c['perplexity_change_percent']:+.1f}%; {c['throughput_ratio']:.2f}× target throughput, {c['update_ratio']:.2f}× updates and {c['cpu_ratio']:.2f}× aggregate CPU time." for c in s['contrasts'])
family='\n'.join(f"- **{names[a]}:** {label(by_id[rid])}, CE {by_id[rid]['ce']:.6f}." for a,rid in s['family_winners'].items())
tradeoffs=[]
for arm in ('native','mingru_linear'):
    small=next((r for r in rows if r['arm']==arm and r['batch']==4),None)
    large=next((r for r in rows if r['arm']==arm and r['batch']==16),None)
    if small and large:
        tradeoffs.append(f"- **{names[arm]}:** switching from B4/T2/LR 0.004 to B16/T4/LR 0.008 changes final CE by {large['ce']-small['ce']:+.4f}, while consuming {large['train_cpu_seconds']/small['train_cpu_seconds']:.2f}× aggregate CPU time and processing {large['training_targets']/small['training_targets']:.2f}× targets.")
tradeoffs='\n'.join(tradeoffs)
failed='No declared trial failed.' if not s['failures'] else '\n'.join(f"- {f['run_id']}: {f['error_type']} — {f['error']}" for f in s['failures'])
text=f'''# Longer wall-time comparison: minGRU, ordinary GRU and nanoGPT

**Best measured endpoint: {label(best)}, CE {best['ce']:.6f}.** Six fresh recipes each receive a 180-second training-wall allowance, extending the previous 60-second exploration. This is a practical recipe comparison with two choices per architecture. It does not establish the best possible recipe for any architecture.

This contribution updates the existing draft PR in **JRowe47/Super-1 → main**. No upstream submission or automatic merge is requested.

## Results

| Model | Batch / threads | Peak LR | Updates | Target exposures | Training wall s | Training CPU s | Final CE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
{table}

Parameter counts: ordinary GRU 995,925; linear minGRU 996,320; nanoGPT 1,004,880 including position embeddings. nanoGPT's constructor log prints0.94M because its default display excludes the position table; the counts here include every trainable parameter. All rows use seed 101604, data-order seed 101613, FP32, context 512, and the same final 32-window / 16,384-target region. The validation region has been repeatedly used in this project. {failed}

![Learning curves and final ranking](../graphs/{ROOT.name}/loss_curves.png)

Curves use a separate fixed eight-window / 4,096-target monitoring region. The final table and right panel use the larger final region, so curve endpoints and final ranking losses are different measurements. Every declared recipe is shown, using the final checkpoint rather than a validation-selected checkpoint.

## Planned comparisons

{contrasts}

Best endpoint among the two declared recipes in each family:

{family}

The resource trade-offs between the two GRU recipes are:

{tradeoffs}

Tiny endpoint differences should be read as close results in a single-seed screen, rather than reliable separation. The exact ranking and the resource cost are both retained.

The two GRU-family comparisons match batch, threads and peak learning rate between architectures. At a fixed elapsed allowance, their cheaper or slower updates produce different data exposure and optimizer-step counts. The nanoGPT pair holds batch and threads fixed and changes peak LR. Within either GRU family, B4/T2/LR 0.004 versus B16/T4/LR 0.008 changes three settings together; it cannot isolate any one setting's causal effect.

![Monitoring loss by aggregate CPU consumption](../graphs/{ROOT.name}/cpu_curves.png)

CPU time is the process's aggregate CPU consumption across its threads. More threads can spend more CPU seconds inside the same wall allowance. This second plot re-expresses the recorded trajectories; it is not a separate equal-CPU training experiment, and each recipe's LR still follows its original wall-time schedule. Neither clock is a direct FLOP measurement.

## Why these six recipes

The preceding [minGRU thread/LR screen](mingru-thread-lr-20261016-v1.md) favored B4/T2/LR 0.004, followed by B16/T4/LR 0.008. Those two recipes are carried forward and also applied to ordinary GRU, giving matched configuration comparisons. nanoGPT keeps its previously measured B4/T4 throughput setting and receives peak LR 0.002 and 0.004. These are two plausible choices per family, not equal or exhaustive hyperparameter optimization; the GRU choices are informed by minGRU's prior results.

This pass uses a new initialization seed and a new common training-window stream. Its final losses therefore cannot be compared with the previous study as an isolated effect of extending time. The 180-second cosine schedule also has a nine-second warmup; a checkpoint around 60 seconds is not equivalent to a completed 60-second run.

## Data, implementation and timing

- The exact FineWeb-Edu uint16 arrays and train-only 2048-BPE tokenizer are unchanged and hash-checked. Examples use 512 next-token targets with a reset state and full BPTT. Consecutive entries in the common random-window stream are regrouped into batches; the number of consumed entries depends on runtime speed. Official test data is untouched.
- The existing model implementations are reused without architecture changes. Linear minGRU retains the signed linear candidate, input-only gate and 32-token affine scan, plus the same segment/document GRUs. Ordinary GRU uses the same hierarchy. nanoGPT uses the corrected wrapper that consumes its returned CE once; the old duplicated CE overhead is absent.
- AdamW betas 0.9/0.95, epsilon 1e-8, weight decay 0.1 with existing exclusions, gradient norm clip 1. The declared peak LR has 5% training-wall warmup (factor floor 0.04) then cosine decay to 10% of peak over 180 seconds. No adaptive LR search or validation-dependent stopping.
- Every run starts fresh. Workers execute sequentially on the existing CPU host, pinned to CPUs 0–1 for two threads or 0–3 for four threads, with one interop thread and deterministic PyTorch algorithms. Other host activity is uncontrolled. No GPU or new paid resources.
- The training timer covers batch construction, forward, CE, backward, clipping and AdamW. It excludes setup, monitoring and checkpoint IO. `run_wall_seconds` / `run_cpu_seconds` cover initial monitoring through final evaluation, including checkpoint IO. Separate setup timers start before Torch import; `LAUNCH_RESULT.json` additionally records end-to-end worker wall time from before process creation through exit. This clarifies the previous harness's overly broad “full-run” wording.
- Stopping uses a reserve of max(0.2 seconds, twice the largest recent update). Actual training times appear in the table; the initial estimate uses the earlier benchmark for that configuration. A failure is retained and reported without replacement. Per-process CPU and wall caps were raised to accommodate the longer run before training began.

## Checks and replay

Checkpoint evaluation reproduces every saved final-window loss within 5e-6. The analysis reconciles source/data/tokenizer hashes, the new stream digest, parameter counts, every learning rate, update and target counts, finite training values, accumulated wall/CPU times, stopping margins and launcher timing. Existing parent logits/gradient parity checks cover the unchanged model implementation; that grid is not repeated here.

- [Protocol](../results/{ROOT.name}/PROTOCOL.json), [registration](../results/{ROOT.name}/REGISTRATION.json), [summary](../results/{ROOT.name}/SUMMARY.json), [checks](../results/{ROOT.name}/CHECKS.json), [terminal state](../results/{ROOT.name}/TERMINAL.json).
- [Runner](../results/{ROOT.name}/experiment.py), [launcher](../results/{ROOT.name}/launch.py), [analysis](../results/{ROOT.name}/analyze.py), [report generator](../results/{ROOT.name}/report.py). Keep the hash-bound sibling batch/thread study and original GRU implementation directories together.
- Final model/AdamW/RNG/full traces: `gru_longer_wall_checkpoints_20261016.zip`. Exact data remains in `gru_extended_checkpoints_20261016.zip`, SHA256 c90bb924b3b69e49bf217582776e7add78292d8c72e8a2fa5e82bd28a6db80a6. These contributor-held payloads remain outside Git; see [preservation metadata](../results/{ROOT.name}/PRESERVATION.json).

```bash
python auto-research/results/{ROOT.name}/report.py
# Training replay: separate copy/output directory, retaining frozen protocol and dependencies,
# with no completed execution artifacts. Preserve this published run.
python auto-research/results/{ROOT.name}/launch.py --data-dir /path/to/exact/data
```

Time-based schedules depend on hardware speed, so a new training replay need not follow the identical trajectory. The exact saved-checkpoint evaluation and complete compressed traces provide the record of what ran. These single-seed, small-model CPU observations guide the next experiment; they are not an official nanoGPT GPU speedrun or a large-model scaling result.
'''
(REPO/'auto-research/reports'/f'{ROOT.name}.md').write_text(text)
print('REPORT',best['run_id'],best['ce'])
