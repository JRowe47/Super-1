"""Render the measured throughput grid and all predeclared quality runs."""
import gzip,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[2]
p=json.loads((ROOT/'PROTOCOL.json').read_text());s=json.loads((ROOT/'SUMMARY.json').read_text());q=json.loads((ROOT/'SELECTION.json').read_text())
labels={'native':'Ordinary GRU','mingru_linear':'Linear minGRU','nanogpt':'nanoGPT'}
colors={'native':'#293e65','mingru_linear':'#07816c','nanogpt':'#ad6146'}
grid=[r for t in p['threads'] for r in json.loads((ROOT/'benchmark'/f'threads{t}.json').read_text())['rows']]
out=REPO/'auto-research/graphs'/ROOT.name;out.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
fig,axes=plt.subplots(1,3,figsize=(12,4.2),layout='constrained')
vmax=max(x['tokens_per_wall_second'] for x in grid)/1000
for arm,ax in zip(p['arms'],axes):
    a=np.array([[next(x['tokens_per_wall_second'] for x in grid if x['arm']==arm and x['batch']==b and x['threads']==t)/1000 for t in p['threads']] for b in p['batches']])
    im=ax.imshow(a,vmin=0,vmax=vmax,cmap='YlGnBu')
    for i in range(3):
        for j in range(3):ax.text(j,i,f'{a[i,j]:.1f}',ha='center',va='center',color='white' if a[i,j]>vmax*.55 else '#172c3b',fontweight='bold')
    ax.set(xticks=range(3),xticklabels=p['threads'],yticks=range(3),yticklabels=p['batches'],xlabel='CPU threads',ylabel='Batch size',title=labels[arm])
fig.colorbar(im,ax=axes,label='Thousands of target tokens / training wall-second',shrink=.8)
fig.suptitle('Batching and CPU threads · full training-update throughput',fontsize=14,fontweight='bold')
fig.savefig(out/'throughput.png',dpi=170);fig.savefig(out/'throughput.svg');plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(12,4.8),gridspec_kw={'width_ratios':[1.35,1]})
for r in sorted(s['rows'],key=lambda x:(p['arms'].index(x['arm']),x['role'])):
    cpath=ROOT/'execution'/r['run_id']/'curves.json'
    c=json.loads(cpath.read_text()) if cpath.exists() else json.loads(gzip.decompress(cpath.with_suffix('.json.gz').read_bytes()))
    label=f"{labels[r['arm']]} · B{r['batch']}/T{r['threads']}"
    axes[0].plot([x['train_wall_seconds'] for x in c['monitor']],[x['validation_ce'] for x in c['monitor']],
        color=colors[r['arm']],ls='-' if r['role']=='baseline' else '--',marker='o',markersize=3,label=label,lw=2)
axes[0].set(xlabel='Training wall seconds',ylabel='Monitoring CE · lower is better',title='Learning during the same time allowance',xlim=(0,61))
axes[0].legend(fontsize=8,frameon=False);axes[0].grid(alpha=.15)
for role,offset,hatch in [('baseline',-.18,''),('throughput',.18,'//')]:
    values=[]
    for arm in p['arms']:
        r=next((x for x in s['rows'] if x['arm']==arm and x['role']==role),next(x for x in s['rows'] if x['arm']==arm and x['role']=='baseline'))
        values.append(r['ce'])
    bars=axes[1].bar(np.arange(3)+offset,values,width=.34,color=[colors[x] for x in p['arms']],alpha=.85,hatch=hatch,
                    label='B1 / T1' if role=='baseline' else 'Fastest measured setting')
    for bar,value in zip(bars,values):axes[1].text(bar.get_x()+bar.get_width()/2,value+.035,f'{value:.3f}',ha='center',fontsize=9)
axes[1].set(xticks=range(3),xticklabels=['GRU','Linear minGRU','nanoGPT'],ylabel='Endpoint CE · lower is better',title='Final loss on the same 16,384 targets',ylim=(0,max(r['ce'] for r in s['rows'])+.55))
axes[1].legend(fontsize=8,frameon=False,loc='lower right');axes[1].grid(axis='y',alpha=.15)
fig.suptitle('Equal training wall time · ~1M parameters · fresh seed 101603',fontsize=14,fontweight='bold')
fig.text(.5,.02,'Monitoring uses 4,096 separate targets. Dashed / hatched = throughput-selected setup; solid = original B1 / T1.',ha='center',fontsize=9,color='#505965')
fig.tight_layout(rect=(0,.06,1,.93));fig.savefig(out/'loss_curves.png',dpi=170);fig.savefig(out/'loss_curves.svg');plt.close(fig)

table='\n'.join(f"| {labels[r['arm']]} | {r['batch']} / {r['threads']} | {r['updates']:,} | {r['training_targets']:,} | {r['train_wall_seconds']:.2f} | {r['train_cpu_seconds']:.1f} | {r['ce']:.6f} |" for r in sorted(s['rows'],key=lambda x:(p['arms'].index(x['arm']),x['role'])))
bench='\n'.join(f"| {labels[arm]} | {q['selected'][arm]['batch']} / {q['selected'][arm]['threads']} | {q['selected'][arm]['tokens_per_wall_second']:,.0f} |" for arm in p['arms'])
observations='\n'.join(f"- **{labels[c['arm']]}:** selected setup achieves {c['observed_throughput_ratio']:.2f}x training throughput, {c['target_exposure_ratio']:.2f}x target exposures and {c['optimizer_update_ratio']:.2f}x optimizer updates versus B1/T1; final CE changes by {c['ce_change']:+.4f} ({c['baseline_ce']:.4f} to {c['selected_ce']:.4f})." for c in s['contrasts'])
best=min(s['rows'],key=lambda r:r['ce'])
text=f'''# Batching, threads and equal-wall-time learning

The best endpoint in this short screen is **{labels[best['arm']]} at batch {best['batch']} / {best['threads']} CPU threads: CE {best['ce']:.6f}**. All six fresh runs receive a 60-second training-wall allowance, with their actual stopping times below. Throughput and learning efficiency are measured separately: faster token processing does not guarantee the lowest loss under the common optimizer recipe.

This independent explorer study and its PR target **JRowe47/Super-1 → main**. Upstream submission requires the user's explicit go-ahead. The PR also contains the preceding unmerged longer-GRU and equal-CPU studies through its ancestry.

## Results

| Model | Batch / threads | Updates | Target exposures | Training wall s | Training CPU s | Final CE ↓ |
|---|---:|---:|---:|---:|---:|---:|
{table}

Every row uses approximately 1M parameters, initialization seed 101603, a common ordered FineWeb-Edu training-window stream, context 512, FP32, and the same 32 final validation windows / 16,384 targets. Batching regroups consecutive windows; shorter runs see prefixes of the same stream. Batch size changes targets per optimizer step, so update counts alone are not data exposure. Native and linear have 995,925 and 996,320 parameters; nanoGPT has 1,004,880 including its position table. The nanoGPT constructor log excludes position embeddings by default and prints 0.94M; the counts here include every trainable parameter. Validation has been reused across explorations and is not an untouched test.

{observations}

![Loss trajectories](../graphs/{ROOT.name}/loss_curves.png)

The curves use a separate fixed 8-window / 4,096-target monitoring region. Final table and bars use the 32-window region; their CE values need not match the curve endpoints. All arms and both settings are shown, without best-checkpoint selection.

## Throughput screen

| Model | Selected batch / threads | Benchmark target tokens/s |
|---|---:|---:|
{bench}

![Throughput grid](../graphs/{ROOT.name}/throughput.png)

Each of 27 cells includes two warmup updates and three measured full updates: data slicing, forward, CE, backward, gradient clipping, AdamW and model constraints. Fresh disposable weights are used per cell. The order is seeded; thread counts run in separate sequential processes. Select the maximum median tokens per wall-second within each architecture, then freeze the selected settings before quality training. Also train B1/T1 references. The choices therefore use throughput alone, without validation-based tuning. The full training runs provide a separate observed-throughput check; short microbenchmarks can be noisy.

At the best measured settings linear minGRU / GRU throughput is {q['selected']['mingru_linear']['tokens_per_wall_second']/q['selected']['native']['tokens_per_wall_second']:.2f}x. The original B1/T1 ratio in this screen is {next(x['tokens_per_wall_second'] for x in grid if x['arm']=='mingru_linear' and x['batch']==1 and x['threads']==1)/next(x['tokens_per_wall_second'] for x in grid if x['arm']=='native' and x['batch']==1 and x['threads']==1):.2f}x. Batching and threading can change relative advantages; extrapolating the earlier one-thread ratio would be misleading.

## Training and accounting

- **Time:** 60 seconds of measured training-update wall time per row. Startup, validation, checkpoint IO and orchestration are excluded from this clock. Actual full-run wall is also recorded in each RESULT.json and SUMMARY.json. The stopping margin is max(0.2s, twice the largest recent update); actual budgets differ by less than roughly two updates. No validation-dependent stopping.
- **Resources:** sequential model runs on the same existing CPU host, up to four threads each. Thread 1 runs are pinned to CPU 0; thread 2 to CPUs 0–1; thread 4 to CPUs 0–3. Equal elapsed time with four allowed cores is not equal CPU-seconds or FLOPs; the table exposes the extra CPU consumption. Other host activity remains uncontrolled. No GPU or new paid resources.
- **Optimizer:** common AdamW, peak LR 0.002, decay 0.1 with tied token tables and one-dimensional parameters excluded, betas 0.9/0.95, epsilon 1e-8, norm clip 1. Warm up over 5% of training time (initial factor floor 0.04), then cosine decay to 0.0002 at the time limit. No batch-specific LR scaling or tuning. This can limit large-batch learning and is part of the tested recipe, not an architecture limit.
- **History:** full gradients within each 512-token example; reset recurrent state for every example. Hierarchical GRUs still feed segment/document state back every 32 tokens / 8 segments. Linear scan parallelism remains confined to 32-token chunks. No long-context or GPU scaling inference follows.
- **Freshness:** all six quality runs start fresh at seed 101603; benchmark weights are discarded. The 60-second horizon, fresh seed/data order and warmup differ from the preceding 256-second comparison. Do not interpret cross-study loss differences as an isolated batching effect; use the paired rows here.

## nanoGPT wrapper correction

Inspection found that the old wrapper called `model(x, targets)`, which already computes CE, discarded that loss, and computed the same CE again. The new harness uses the returned loss directly. The duplicate forward computation is removed for all new timings; the objective and gradients are unchanged. The preflight confirms the returned and recomputed CE agree exactly. Earlier nanoGPT CPU numbers included the redundant work, so the older 1.80x minGRU/nanoGPT throughput ratio should not be treated as a clean implementation comparison.

The vendored nanoGPT model is unchanged, with its original MIT license. It uses the available PyTorch scaled-dot-product attention path. These ~1M CPU runs with a 2048-token vocabulary are not an optimized public nanoGPT GPU speedrun reproduction.

## Checks and replay

Initial and trained batched logits/gradients agree with averaging individual examples. Every saved checkpoint replays all final-window losses within 5e-6, including multi-thread-to-single-thread evaluation rounding. Source/data hashes, stream digest, parameter counts, time budgets, complete training traces and target accounting are reconciled. Checkpoint artifacts retain all model weights, optimizer moments, Torch RNG and traces. See [CHECKS.json](../results/{ROOT.name}/CHECKS.json), [PROTOCOL.json](../results/{ROOT.name}/PROTOCOL.json), [SELECTION.json](../results/{ROOT.name}/SELECTION.json) and [SUMMARY.json](../results/{ROOT.name}/SUMMARY.json).

Source is in this fork; the parent hierarchy is hash-bound under `../gru-extended-20261016-v1`. Exact train/validation arrays remain in `gru_extended_checkpoints_20261016.zip` (SHA256 c90bb924b3b69e49bf217582776e7add78292d8c72e8a2fa5e82bd28a6db80a6). Final checkpoints are held in `gru_batch_wall_checkpoints_20261016.zip`; see [PRESERVATION.json](../results/{ROOT.name}/PRESERVATION.json). These contributor-held large payloads are outside Git.

```bash
python auto-research/results/{ROOT.name}/report.py
# For training replay, use a separate copy/output directory with no completed execution artifacts:
python auto-research/results/{ROOT.name}/launch.py --data-dir /path/to/exact/data
```

Keep the frozen protocol and selected settings for a training replay. The clock schedule makes training trajectories depend on measured hardware speed; the exact observed traces are saved as `curves.json.gz`. Do not overwrite this completed run. Independent one-seed observations guide the next quick experiment; they do not establish a scaling law or a tuned optimum.
'''
(REPO/'auto-research/reports'/f'{ROOT.name}.md').write_text(text)
print('REPORT',best['run_id'],best['ce'])
