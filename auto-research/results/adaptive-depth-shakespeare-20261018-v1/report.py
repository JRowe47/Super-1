"""Render report and scientific plots from verified saved metrics only."""
import hashlib,json,statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=Path(__file__).resolve().parent; ROOT=H.parents[2]; O=H/'execution'
r=json.loads((O/'RESULT.json').read_text()); t=json.loads((O/'TERMINAL.json').read_text()); c=json.loads((H/'CONFIG.json').read_text())
rows=r['rows']; arms=c['arms']; labels={'fixed':'Uniform curve','global':'Global positions','token':'Token positions','dense':'Dense RoPE','nanogpt':'Upstream nanoGPT'}
by={a:[v for v in rows if v['arm']==a] for a in arms}
mean=lambda a,k:statistics.mean(v[k] for v in by[a])
ce={a:statistics.mean(v['confirmation']['ce'] for v in by[a]) for a in arms}
bm=lambda a,k:statistics.mean(v['benchmark'][k] for v in by[a])
G=ROOT/'auto-research/graphs'/H.name; G.mkdir(parents=True,exist_ok=True)
colors=['#8796a5','#ba8b47','#266fc0','#6b56a8','#2d8d71']
fig,axs=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
for i,a in enumerate(arms):
    axs[0].scatter([i]*len(by[a]),[v['confirmation']['ce'] for v in by[a]],color=colors[i],s=28)
    axs[0].plot([i-.22,i+.22],[ce[a]]*2,color=colors[i],lw=3)
axs[0].set(xticks=range(len(arms)),xticklabels=[labels[a] for a in arms],ylabel='Confirmation CE (nats / byte)',title='All three seeds; line = mean')
axs[0].tick_params(axis='x',rotation=30)
for i,a in enumerate(arms):
    axs[1].scatter([bm(a,'decode_ms_per_token_p50')],[ce[a]],s=90,color=colors[i],label=labels[a])
axs[1].set(xlabel='Uncached decode median (ms / byte)',ylabel='Mean confirmation CE',title='Lower and left are better')
axs[1].legend(fontsize=8)
for seed in c['seeds']:
    row=next(v for v in by['token'] if v['seed']==seed)
    axs[2].errorbar(range(c['layers']),row['selection_positions_mean'],yerr=row['selection_positions_std'],marker='o',capsize=3,label=str(seed))
axs[2].plot(range(c['layers']),[i/(c['layers']-1) for i in range(c['layers'])],'k--',label='Uniform')
axs[2].set(xlabel='Executed block',ylabel='Depth coordinate',title='Token positions: mean and token SD',xticks=range(c['layers']))
axs[2].legend(fontsize=8)
fig.suptitle('Fixed-step adaptive depth: Tiny Shakespeare CPU pilot',fontsize=14)
fig.savefig(G/'results.png',dpi=160); fig.savefig(G/'results.pdf'); plt.close(fig)
(G/'METRICS.json').write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
lines=[f'# Fixed-step adaptive depth pilot: {H.name}', '', f'Status: COMPLETED. Decision: **{r["outcome"]}**.', '',
'## Question and controls','',
'Does choosing ordered weight-curve sampling positions from each token improve held-out byte prediction with the same number of executed blocks? Uniform positions are the primary comparator. Learned global positions isolate whether token conditioning itself matters. Dense RoPE and byte-exact upstream nanoGPT provide external controls. This extends the cubic projection curve; it does not change the historical recurrent hierarchy.', '',
'## What actually ran','',
f'User authorization: current-chat “ok do it” following the proposed fixed-step experiment. Preregistration commit: `{json.loads((O/"START.json").read_text())["commit"]}`. Protocol SHA-256: `{hashlib.sha256((H/"PROTOCOL.json").read_bytes()).hexdigest()}`.', '',
f'Five arms x three paired seeds ({", ".join(map(str,c["seeds"]))}), each with {c["updates"]} AdamW updates and {c["updates"]*c["batch"]*c["context"]:,} next-byte targets; {len(rows)*c["updates"]*c["batch"]*c["context"]:,} total training targets. Width {c["width"]}, {c["layers"]} blocks, {c["heads"]} heads, training/evaluation context {c["context"]} bytes. One LR recipe and final-checkpoint selection for every arm. CPU FP32, one intra-op and inter-op thread, no cloud or GPU.', '',
'Tiny Shakespeare uses a fixed 256-byte vocabulary. Contiguous source regions are 80% training, 10% selection, 10% confirmation. The endpoint scores the first 256 nonoverlapping 65-byte chunks of confirmation (16,384 targets); monitoring scores 4,096 selection targets. It does not score the whole reserved region. The public corpus snapshot, exact boundaries, source/data hashes and runtime are included. No teacher or official benchmark test was used.', '',
'All 15 final checkpoints were frozen before confirmation scoring. Every endpoint was evaluated twice with exact FP32 CE and logit hashes, then independently replayed by verify_results.py. All arms improved monitoring CE from initialization. No LR sweep, early-stop selection, retraining, or confirmation-based repair occurred.', '',
'## Results','',
'Lower CE is better. These are byte losses, not word perplexity or accuracy.', '',
'| Arm | Parameters | Seed 73 | Seed 191 | Seed 307 | Mean CE |', '|---|---:|---:|---:|---:|---:|']
for a in arms:
    vals=[next(v['confirmation']['ce'] for v in by[a] if v['seed']==s) for s in c['seeds']]
    lines.append(f'| {labels[a]} | {by[a][0]["parameters"]:,} | '+ ' | '.join(f'{v:.6f}' for v in vals)+f' | {ce[a]:.6f} |')
lines+=['','Differences below are token-conditioned minus comparator; negative favors token conditioning. The prespecified useful effect is at least 0.02 nats/byte, with all three gaps negative and a descriptive paired-t 95% upper bound below zero. The n=3 intervals have unverified normality and cover seed variability on one shared text region, not corpus-generalization uncertainty.','',
'| Comparator | Seed gaps | Mean gap | Descriptive 95% interval |','|---|---|---:|---|']
for a,z in r['contrasts'].items(): lines.append(f'| {labels[a]} | '+', '.join(f'{v:+.6f}' for v in z['token_minus_control_nats'])+f' | {z["mean"]:+.6f} | [{z["descriptive_paired_t95"][0]:+.6f}, {z["descriptive_paired_t95"][1]:+.6f}] |')
lines+=['',f'Primary fixed comparison passed: **{r["primary_fixed_comparison_passed"]}**. Token-specific control passed: **{r["token_specific_control_passed"]}**. Combined dense/nanoGPT gate passed: **{r["nanoGPT_comparison_passed"]}**.','',
'## Deployment measurements','',
'CPU measurements use fresh processes per checkpoint, batch 1, 5 warmups and 30 repetitions. Prefill scores 64 bytes. Decode generates 16 bytes from a common 32-byte prompt by recomputing the entire prefix each time; no KV cache is implemented. Table shows means across seeds of within-process medians/p95. Concurrent unrelated host work can affect wall latency; these are implementation-specific pilot measurements.','',
'| Arm | Weight KiB | Prefill p50 / p95 ms | Decode p50 / p95 ms per byte | Process peak MiB | Training CPU s |','|---|---:|---:|---:|---:|---:|']
for a in arms:
    lines.append(f'| {labels[a]} | {bm(a,"weight_bytes")/1024:.1f} | {bm(a,"prefill_ms_p50"):.3f} / {bm(a,"prefill_ms_p95"):.3f} | {bm(a,"decode_ms_per_token_p50"):.3f} / {bm(a,"decode_ms_per_token_p95"):.3f} | {bm(a,"process_peak_rss_bytes")/2**20:.1f} | {mean(a,"training_cpu_seconds_including_selection"):.2f} |')
lines+=['',
'Process peaks include Python/PyTorch, checkpoint loading, data and workspaces, not just model memory. Initial process peaks are retained in raw metrics. Separate activation-only/allocator peaks were not measured. There is no quantization or GPU memory result. Curve arms execute two basis matmuls per projection; dense controls execute one. The fixed arm deliberately uses the same basis execution path for mechanism comparison, rather than an optimized prematerialized static weight matrix. Equal block count and fewer weights are not equal FLOPs or optimized inference.', '',
'## Mechanism diagnostics and limitations','',
'Endpoints remain 0 and 1; internal increments are positive. The router starts with zero output so all three curve arms begin with the same uniform positions and base weights. Per-block normalization remains unchanged. Fixed includes an inert router; global lacks token input, so equal stored counts do not mean equal effective conditional capacity. Token-specific claim requires beating that global control.','',
'| Seed | Token positions mean | Token positions SD | CE after forcing uniform positions |','|---:|---|---|---:|']
for v in by['token']:
    lines.append(f'| {v["seed"]} | '+', '.join(f'{x:.6f}' for x in v['selection_positions_mean'])+' | '+', '.join(f'{x:.6f}' for x in v['selection_positions_std'])+f' | {v["uniform_intervention_diagnostic"]["ce"]:.6f} |')
lines+=['',
'Uniform-position intervention is a post-training diagnostic declared before training; it is not an independently trained control and does not replace the primary endpoint. The current short-budget, single-domain byte task and 64-byte context cannot establish broader language quality, reasoning ability, transfer, convergence, novelty or SOTA. Models are not parameter- or FLOP-matched across curve and dense families; shared width/depth and exposure isolate the within-curve comparison.', '',
'## Decision and next experiment','',
f'The frozen decision is **{r["outcome"]}**. Preserve these outcomes without promoting the adaptive model or changing thresholds. No extra seeds, training, cloud resources or automation are started by this result.', '',
'A follow-up should first determine whether a well-trained nonconstant curve provides a useful depth landscape: compare token-conditioned and global routing while freezing the same pretrained curve, on a new preregistered confirmation region. That would separate weak curve differentiation during scratch training from routing usefulness; it is a proposal only, not evidence of the explanation. Measure its cost before a larger language run.', '',
'## Integrity and cleanup','',
f'Training/evaluation parent process: {t["cpu_seconds"]:.2f} CPU seconds and {t["wall_seconds"]:.2f} wall seconds. Isolated benchmark subprocess CPU totals: {sum(v["benchmark"]["cpu_seconds"] for v in rows):.2f} seconds, separately accounted. Resource bounds were 900 CPU seconds, 1,800 wall seconds and 100 MiB new run artifacts. Terminal receipt records actual bytes at completion.', '',
'The initial FP32 finite-difference canary suffered cancellation (zero numerical estimate). Before training, it was strengthened to FP64 and passed at absolute error below 1e-8; both receipts are retained. The original reference passes forward, gradient and AdamW-step parity with nonconstant coefficients. Nonzero routing also passes exact causal-prefix checks. No frozen training source changed after the preregistration freeze; reporting and independent verification tools were added afterward.', '',
'All checkpoints, corpus snapshot, immutable protocol, source manifest, raw progress, duplicate results, independent verification and figure inputs are preserved. Disposable runtime/font caches are excluded from Git and stored only on the SSD. Original workspace changes and historical evidence are preserved. No cloud resources were created; cleanup receipt is local-only.', '',
'## Reproduction','',
'From the repository root with Python 3.9 and PyTorch 2.8.0:', '',
'```bash',f'python3 -B auto-research/results/{H.name}/verify_results.py',
f'python3 -B auto-research/results/{H.name}/run.py --output "/Volumes/SP AI 01_16/CodexWorkspaces/YOUR_FRESH_REPLAY_DIRECTORY"','```','',
'The second command retrains and therefore requires an authorized fresh resource envelope and an unused SSD destination. Source hashes are checked; outputs refuse overwrite. Training order is shuffled deterministically per seed, and source/config/checkpoint identities are preserved.', '',
f'![All-seed results and position diagnostics](../graphs/{H.name}/results.png)','']
report=ROOT/'auto-research/reports'/f'{H.name}.md'
with report.open('x') as f:f.write('\n'.join(lines))
print(report)
