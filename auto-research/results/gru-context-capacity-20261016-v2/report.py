"""Render the complete descriptive screen from saved results."""
import json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parent
s=json.loads((ROOT/'SUMMARY.json').read_text());rows=s['rows'];lookup={r['run_id']:r for r in rows}
profiles=json.loads((ROOT/'PREFIX_PROFILE.json').read_text())['rows']
def name(r):
    n='nanoGPT' if r['arm']=='nanogpt' else 'Ordinary GRU' if r['arm']=='native' else 'Linear minGRU'
    return f"{n} w{r['width']}"+(' / chunk64' if r['segment_length']==64 else '')
best=rows[0]
lines=['# Context length and recurrent capacity: recovered CPU screen','',
    f"**{len(rows)} of 10 trials completed; {len(s['failures'])} failed.** Best observed endpoint: **{name(best)}, context {best['context']}, CE {best['ce']:.6f}**. Each declared recipe receives a 90-second training-wall allowance. Every final score uses the same final 512 targets from 16 validation windows (8,192 targets). One seed, descriptive exploration.",'',
    'This fresh v2 follows an environment reset. The interrupted v1 result remains separate and is not included in the ranking. Exact data and pinned model sources were restored; source/protocol were published before training, and checkpoints were archived between trials. See [recovery details](../results/'+ROOT.name+'/RECOVERY.md).','',
    '## Every endpoint','',
    '| Model | Context | Updates | Target exposures | Train wall s | CPU s | Final CE ↓ | PPL ↓ | Targets/s | Prefix benefit CE |',
    '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:
    benefit='—' if r['prefix_benefit_ce'] is None else f"{r['prefix_benefit_ce']:+.5f}"
    lines.append(f"| {name(r)} | {r['context']} | {r['updates']:,} | {r['training_targets']:,} | {r['train_wall_seconds']:.2f} | {r['train_cpu_seconds']:.1f} | {r['ce']:.6f} | {r['perplexity']:.2f} | {r['tokens_per_wall_second']:,.0f} | {benefit} |")
lines+=['','Prefix benefit = CE after swapping history before the last 512 input tokens minus CE with authentic history. Positive means the authentic prefix helped this probe. Its first scored prediction is adjacent to the altered prefix, so the whole-suffix average mixes near and distant effects. Current suffix, targets and positions remain identical. The post-hoc PREFIX_PROFILE.json separates 64-token bins and the final prediction; only that final prediction isolates a prefix beyond the latest 512 inputs, and it has only 16 targets. This is not a retrieval benchmark.','',
    '![Context and throughput](../graphs/'+ROOT.name+'/context_tradeoffs.png)','','## Declared contrasts','',
    '| Change | CE change ↓ | Perplexity change | Throughput ratio |','|---|---:|---:|---:|']
contrasts=[('minGRU160:512 → 1024','linear_w160_c512_s32','linear_w160_c1024_s32'),
    ('minGRU160:512 → 2048','linear_w160_c512_s32','linear_w160_c2048_s32'),
    ('nanoGPT:512 → 1024','nano_w108_c512_s32','nano_w108_c1024_s32'),
    ('nanoGPT:512 → 2048','nano_w108_c512_s32','nano_w108_c2048_s32'),
    ('Width 160 → 224 at 512','linear_w160_c512_s32','linear_w224_c512_s32'),
    ('Width 160 → 224 at 2048','linear_w160_c2048_s32','linear_w224_c2048_s32'),
    ('Chunk32 → 64 at 2048','linear_w160_c2048_s32','linear_w160_c2048_s64'),
    ('Ordinary GRU → minGRU160 at 2048','native_w147_c2048_s32','linear_w160_c2048_s32')]
for title,a,b in contrasts:
    if a not in lookup or b not in lookup:continue
    a,b=lookup[a],lookup[b]
    lines.append(f"| {title} | {b['ce']-a['ce']:+.5f} | {100*(b['perplexity']/a['perplexity']-1):+.2f}% | {b['tokens_per_wall_second']/a['tokens_per_wall_second']:.3f}× |")
if len(rows)==10:
    base=lookup['linear_w160_c512_s32'];long=lookup['linear_w160_c2048_s32'];chunk=lookup['linear_w160_c2048_s64']
    native=lookup['native_w147_c2048_s32'];nano=lookup['nano_w108_c2048_s32']
    lines+=['','## Observations','',
        f"- Longer context did not pay back its time cost for the small minGRU in this screen: 512 → 2,048 changes CE by {long['ce']-base['ce']:+.4f} and throughput by {100*(long['tokens_per_wall_second']/base['tokens_per_wall_second']-1):+.1f}%. Both lengths score the identical target tails, but longer sequences have fewer resets and different optimization dynamics.",
        f"- The 64-token chunk at 2K improves CE by {long['ce']-chunk['ce']:.4f} and throughput by {100*(chunk['tokens_per_wall_second']/long['tokens_per_wall_second']-1):.1f}% relative to chunk32. This CE gap is below the declared 0.05 follow-up signal and combines scan size with slower feedback clocks.",
        f"- At 2K, the small chunk32 minGRU reaches CE {long['ce']:.4f}, ordinary GRU {native['ce']:.4f}, and nanoGPT {nano['ce']:.4f}. Its throughput is {long['tokens_per_wall_second']/native['tokens_per_wall_second']:.2f}× ordinary GRU and {long['tokens_per_wall_second']/nano['tokens_per_wall_second']:.2f}× nanoGPT on this CPU implementation.",
        f"- Widening minGRU changes CE by {lookup['linear_w224_c512_s32']['ce']-base['ce']:+.4f} at 512 and {lookup['linear_w224_c2048_s32']['ce']-long['ce']:+.4f} at 2K. That is a result for the shared short-run recipe, not evidence that larger state cannot help after longer training or tuning.",
        '- The earlier 180-second results use a different horizon, seed, data order and evaluation windows. Their lower absolute CE values are not a regression comparison with this screen.']
lines+=['','These are paired configurations within one seed, not uncertainty estimates. CE differences of 0.05 were declared useful leads for follow-up; they do not establish confirmation. A wall-time comparison permits different update and target counts.','',
    '![All loss curves](../graphs/'+ROOT.name+'/loss_curves.png)','',
    'Curves score four separate monitoring windows (2,048 targets), so their endpoints differ from the final table. Every declared final checkpoint is reported; no best-checkpoint selection.','',
    '## Position-specific prefix diagnostic','',
    'This post-hoc view is descriptive. Each 64-target bin contains 1,024 losses; the last-target column contains only 16. Values are swapped-prefix CE minus authentic-prefix CE. The first bin is close to the changed history; only the final prediction guarantees the entire changed prefix lies beyond its latest 512 inputs.','',
    '| Model | Context | First 64 suffix targets | Last 64 suffix targets | Final target only |',
    '|---|---:|---:|---:|---:|']
for r in rows:
    if r['run_id'] not in profiles:continue
    v=profiles[r['run_id']]
    lines.append(f"| {name(r)} | {r['context']} | {v['bins64_benefit_ce'][0]:+.6f} | {v['bins64_benefit_ce'][-1]:+.6f} | {v['last_target_benefit_ce']:+.6f} |")
lines+=['',
    'Eleven of the 16 final windows contain no EOS in their latest 512 inputs. The other five end within 512 inputs of an EOS. This composition check uses existing windows and does not create a new test split. Prefix swapping also changes the distribution of history, so these differences do not establish retrieval competence.','',
    'For the small recurrent models, most of the positive whole-suffix effect occurs in the first 128 targets. Their last 64-target bins are close to zero, and their final-target estimates are negative. With only 16 final-target observations, this gives no convincing evidence of a useful distant-prefix benefit; it does not prove the state contains no distant information. The nanoGPT 2K profile is more distributed, but its tiny final-target estimate is also inconclusive.','',
    'All six fixed-prompt sample continuations are retained. They contain word fragments and occasional local phrases but are not coherent answers. The loss ranking should not be mistaken for usable generation quality.','',
    '## Implementation and limits','',
    '- Exact preserved FineWeb-Edu uint16 arrays, train-only 2048-BPE tokenizer, seed 101605/data-order 101614. Reused validation; no official test access.',
    '- All lengths tile the same ordered 2048-target training blocks. Recurrent arms use 8,192 targets/update (batch 16/8/4 at context 512/1024/2048); nanoGPT uses 2,048 (batch 4/2/1). Same ordered target content at a matched consumed-block count, but faster runs consume more blocks.',
    '- Full BPTT and a fresh recurrent state for each example; EOS is a learned token and does not reset state mid-example. Longer context also means fewer independent reset sequences per update. This is not a persistent-lane streaming training test.',
    '- AdamW 0.9/0.95, epsilon 1e-8, clip 1, decay 0.1 for matrices except 2048-row embedding tables. **Both nanoGPT token and position tables receive zero decay.** V2 explicitly records the inherited helper behavior that v1 prose omitted. Peak LR 0.008 for recurrent arms, 0.002 for nanoGPT; 5% wall warmup and cosine to 10%. One recipe per arm, not an exhaustive tuning comparison.',
    '- Linear minGRU retains input-only gates, signed linear candidates, an unfused affine scan inside each feedback chunk, and nonlinear segment/document GRUs. Native and chunk32 linear arms keep the same 32/8 hierarchy timing. Chunk64 changes segment feedback AND document period 256 → 512 tokens; its result cannot isolate scan speed from memory timescales.',
    '- Width 160 → 224 increases persistent FP32 sequence state from 2,560 to 3,584 bytes (+40%), while parameters rise 996,320 → 1,767,584 (+77.4%). All hidden projections/readout also widen. This does not isolate state capacity at fixed weights or compute.',
    '- Ordinary GRU has 995,925 parameters and 2,352 persistent FP32 state bytes per sequence. These state counts exclude model weights, training activations, workspaces and allocator overhead; they are not total-memory measurements.',
    '- All nanoGPT arms have 1,008,072 stored parameters, width 108, four layers/heads and a fixed 2048-position table. At context 512/1024/2048, respectively 842,184/897,480/1,008,072 parameters are reached; unused position rows still count toward stored parameters. This choice fixes the architecture across length arms. The inherited startup log excludes position embeddings and prints 0.79M; the table counts all stored parameters. Every primary evaluation uses its training length; no learned-position extrapolation.',
    '- Four CPU threads; sequential workers. Training clock includes sampling through AdamW, excludes monitoring/checkpoint IO. Aggregate CPU, launcher latency and peak process RSS are retained; RSS is not a deployment-memory comparison. No GPU speed claim follows.',
    '- Recovery repeats the interrupted seed/configurations on a restored runtime. Prior console-only results are not pooled, and cross-study differences do not isolate a single cause.','',
    '## Checks and artifacts','',
    'Preflight verifies identical training target content across lengths, identical scored validation tails, finite forward/backward/AdamW, and 2K state carry through a nonboundary split. Post-run analysis reconciles every target count, LR and clock and replays final/prefix-swap window losses from saved checkpoints. All replay differences are exactly zero on this runtime. Two truncated backup archives were detected after successful upload receipts; both were rebuilt byte-for-byte from intact checkpoints and replaced, with repair provenance retained.','',
    '- [Protocol](../results/'+ROOT.name+'/PROTOCOL.json), [registration](../results/'+ROOT.name+'/REGISTRATION.json), [summary](../results/'+ROOT.name+'/SUMMARY.json), [checks](../results/'+ROOT.name+'/CHECKS.json), [terminal](../results/'+ROOT.name+'/TERMINAL.json).',
    '- [Source/runner](../results/'+ROOT.name+'/experiment.py), [launcher](../results/'+ROOT.name+'/launch.py), [analysis](../results/'+ROOT.name+'/analyze.py), [report generator](../results/'+ROOT.name+'/report.py).',
    '- [Checkpoint hashes](../results/'+ROOT.name+'/CHECKPOINT_MANIFEST.json), [archive preservation](../results/'+ROOT.name+'/CHECKPOINT_ARCHIVES.json), [repair provenance](../results/'+ROOT.name+'/ARCHIVE_REPAIRS.json), [unfiltered samples](../results/'+ROOT.name+'/SAMPLES.json). Large weights/AdamW/RNG/full traces are in contributor-held checkpoint archives, outside Git.','',
    '- [Position-specific prefix profile](../results/'+ROOT.name+'/PREFIX_PROFILE.json), [document-boundary composition](../results/'+ROOT.name+'/DATA_CONTEXT.json).','',
    '## Next small experiment','',
    'Test state carry between 512-token training chunks with detached gradients, matched contiguous data, and explicit document-boundary behavior, beside a reset-state control. This could expose longer history while retaining short backpropagation chunks. A four-token causal input mixer is a separate cheap variant to sample. These are proposed follow-ups, not implemented or queued; no larger model or GPU allocation is needed to ask the next question.','',
    'Every declared failure is retained in SUMMARY.json. No automatic promotion, upstream submission or merge is requested.']
dest=ROOT.parents[1]/'reports'/f'{ROOT.name}.md';dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text('\n'.join(lines)+'\n')
print(dest)
