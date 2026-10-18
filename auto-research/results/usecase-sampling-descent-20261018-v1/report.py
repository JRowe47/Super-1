"""Plot the complete optimization trajectory and report verified outcomes."""
import json,statistics,hashlib
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=Path(__file__).resolve().parent;ROOT=H.parents[2];O=H/'execution'
c=json.loads((H/'CONFIG.json').read_text());r=json.loads((O/'RESULT.json').read_text());hist=json.loads((O/'SAMPLING_HISTORY.json').read_text());terminal=json.loads((O/'TERMINAL.json').read_text());freeze=json.loads((O/'FREEZE.json').read_text())
labels={'uniform':'Frozen uniform','shared_token':'Shared token sampler','global':'Use-case global','token':'Use-case token'}
colors={'uniform':'#88939d','shared_token':'#be8937','global':'#6f5aa6','token':'#2676bb'}
G=ROOT/'auto-research/graphs'/H.name;G.mkdir(parents=True,exist_ok=True)
fig,axs=plt.subplots(2,3,figsize=(14,8),layout='constrained')
for j,task in enumerate(c['tasks']):
 for kind in ['shared_token','global','token']:
  name=kind if kind=='shared_token' else task+'_'+kind
  vals=[]
  for round_id in range(c['rounds']+1):
   vals.append(statistics.mean(z['selection'][task]['ce'] for z in hist if z['candidate']==name and z['round']==round_id))
  axs[0,j].plot(range(c['rounds']+1),vals,marker='o',color=colors[kind],label=labels[kind])
 axs[0,j].set(title=task.title(),xlabel='Gradient-descent round',ylabel='Selection CE (nats / target)',xticks=range(c['rounds']+1))
 for i,kind in enumerate(['uniform','shared_token','global','token']):
  name=kind if kind in ['uniform','shared_token'] else task+'_'+kind
  values=[z['confirmation']['ce'] for z in r['rows'] if z['task']==task and z['candidate']==name]
  axs[1,j].scatter([i]*len(values),values,color=colors[kind],s=25);axs[1,j].plot([i-.2,i+.2],[statistics.mean(values)]*2,color=colors[kind],lw=3)
 axs[1,j].set(xticks=range(4),xticklabels=['Uniform','Shared','Global','Per-case token'],ylabel='Confirmation CE (nats / target)')
 axs[1,j].tick_params(axis='x',rotation=20)
axs[0,0].legend(fontsize=8)
fig.suptitle('Sampling-only gradient descent: model weights frozen across three rounds\nText = bytes; retrieval and arithmetic = synthetic answer targets',fontsize=14)
fig.savefig(G/'trajectories.png',dpi=150);fig.savefig(G/'trajectories.pdf');plt.close(fig)
(G/'METRICS.json').write_text(json.dumps({'results':r,'history':hist},indent=2,sort_keys=True)+'\n')
lines=[f'# Per-use-case sampling descent: {H.name}','','Status: COMPLETED. Three paired seeds, three gradient-descent rounds, one final confirmation per selected checkpoint (with exact duplicates).','','## What changed','',
'The optimization loop benchmarks each workload, freezes the shared model weights, optimizes sampling-policy parameters with Adam, evaluates selection data, retains the best checkpoint and repeats. This is iterative research and tuning, not a recursively executing inference model. Every inference still executes four blocks. Only two internal ordered sampling coordinates can move; endpoints stay fixed.', '',
'The controller is trained by gradients from task loss. The caller explicitly selects a workload-specific policy; there are no keyword rules for routing or hand-selected positions. The original cubic projection weight function and per-block normalization remain unchanged.','',
'## Workloads and controls','',
'- Text: next-byte continuation on Tiny Shakespeare.\n- Retrieval: synthetic four-key dictionary lookup with one answer byte.\n- Arithmetic: synthetic four-operand addition modulo eight with one answer byte.', '',
'These are narrow benchmark use cases; the synthetic tasks are not production retrieval or general mathematics. A fresh shared backbone is trained with 600 updates per task on each seed, then remains frozen throughout adaptation. Four variants are compared: frozen uniform sampling; one shared token-conditioned policy; one global-position policy per use case; and one token-conditioned policy per use case.', '',
'The shared policy receives 200 updates per task in each of three rounds. Each specialist receives the same 200 batches for its task. The shared policy is selected once using equal-weight mean selection CE across tasks; specialists use their own task CE. Both have four selection opportunities including the untuned baseline. A fixed ReduceLROnPlateau schedule reduces learning rate when validation stalls. Every round and optimizer state is preserved, including rejected checkpoints.', '',
'## Scope, provenance and execution','',
f'User authorized iterative per-use-case sampling optimization by gradient descent in the current chat. Preregistration commit: `{json.loads((O/"START.json").read_text())["commit"]}`; protocol SHA-256: `{hashlib.sha256((H/"PROTOCOL.json").read_bytes()).hexdigest()}`.', '',
f'Seeds: {c["seeds"]}. CPU FP32, one thread, width {c["width"]}, {c["layers"]} blocks, {c["bases"]} bases, context {c["context"]} bytes, batch {c["batch"]}. Initial sampling LR {c["sampling_lr"]}, with no hyperparameter sweep or confirmation tuning. Backbone: 68,496 parameters. Active router: 409 parameters. Three specialists store 1,227 router parameters altogether, versus 409 shared; per-request active compute and parameter count remain the same.', '',
'Text train/tune/selection/confirmation regions are disjoint in this run. Confirmation begins after the region scored by the prior pilot; the text tuning/selection regions were exposed to earlier research and are not claimed globally unseen. No prior checkpoint weights transfer. Synthetic prompts are regenerated exactly from the seed and are disjoint across four splits. Each confirmation endpoint uses 256 examples: 16,384 byte targets for text or 256 answer targets for synthetic tasks. All labels and causal masking were checked before training.', '',
'## Results','',
'Lower CE is better. Do not compare text CE directly to synthetic answer CE. Tables show all seeds and the mean; accuracy on text is next-byte accuracy.','']
for task in c['tasks']:
 lines += [f'### {task.title()}','', '| Policy | Seed 73 CE | Seed 191 CE | Seed 307 CE | Mean CE | Mean accuracy |','|---|---:|---:|---:|---:|---:|']
 for kind in ['uniform','shared_token','global','token']:
  name=kind if kind in ['uniform','shared_token'] else task+'_'+kind
  rows=sorted([z for z in r['rows'] if z['task']==task and z['candidate']==name],key=lambda z:c['seeds'].index(z['seed']))
  vals=[z['confirmation']['ce'] for z in rows]
  lines.append(f'| {labels[kind]} | '+' | '.join(f'{v:.6f}' for v in vals)+f' | {statistics.mean(vals):.6f} | {statistics.mean(z["confirmation"]["accuracy"] for z in rows):.2%} |')
 lines+=['','Token-specialist minus comparator differences; negative favors the specialist. Intervals use df=2 with Bonferroni adjustment across three workloads.','', '| Comparator | Paired seed gaps | Mean gap | Familywise interval |','|---|---|---:|---|']
 for comparator,z in r['contrasts'][task].items():
  lines.append(f'| {comparator} | '+', '.join(f'{v:+.6f}' for v in z['gaps_nats_per_target'])+f' | {z["mean"]:+.6f} | [{z["familywise_interval"][0]:+.6f}, {z["familywise_interval"][1]:+.6f}] |')
 lines+=['',f'Frozen decisions: `{json.dumps(r["decisions"][task],sort_keys=True)}`.','']
lines+=['## Interpreting the result','',
'The useful-effect gate requires at least 0.02 nats/target mean gain, favorable signs on every seed and an adjusted interval below zero. Specialization must beat both uniform and shared policies; claiming token-conditioned benefit also requires beating the global-position control. The n=3 intervals are descriptive and depend on unverified normality; one shared holdout does not establish domain generalization.', '',
'Task qualification is checked before interpreting a sampler gain: text must improve selection loss by at least 10% from initialization on every seed; synthetic accuracy must reach 22.5% on every seed, versus 12.5% answer-class chance. This is only a weak learning check, not proof a task is solved. A failed qualification means the frozen backbone did not learn enough for the routing hypothesis to be meaningfully promoted.', '',
'## Selection and gradient evidence','',
'| Seed | Candidate | Selected round | Selection criterion | Mean gradient norm, round 1 to 3 |','|---:|---|---:|---:|---|']
for z in freeze['selected']:
 rows=sorted([v for v in hist if v['seed']==z['seed'] and v['candidate']==z['candidate'] and v['round']>0],key=lambda v:v['round'])
 lines.append(f'| {z["seed"]} | {z["candidate"]} | {z["best_round"]} | {z["selection_criterion"]:.6f} | '+', '.join(f'{v["gradient_norm_mean"]:.6g}' for v in rows)+' |')
lines+=['','All base state fingerprints remain exactly unchanged during sampling. Sampling gradients, learning-rate changes, per-round samplers and all checkpoints are recorded. Matching per-task batch digests ensure shared and specialist policies see identical tuning examples, in the same task-local order.','',
'## Cost and limitations','',
f'The complete run took {terminal["cpu_seconds"]:.2f} CPU seconds and {terminal["wall_seconds"]:.2f} wall seconds. It retained {terminal["base_checkpoints"]} backbone checkpoints, {terminal["sampler_round_checkpoints"]} sampler/optimizer checkpoints and {terminal["confirmation_rows"]} endpoint rows. No cloud, GPU or dependency installation. All child temporary files stayed on the SSD.', '',
'Batch-one full-forward medians/p95 and parameter bytes are in the raw results. No cached decode, total-process-memory, GPU speed, broad language-quality or nanoGPT superiority claim is made. Dense/nanoGPT comparisons from the previous pilot are separate. Specialized controllers have more total stored routing capacity; this study does not remove that confound.', '',
'## Integrity and next decision','',
'All candidates were frozen before reserved confirmation. Every saved score/logit hash reproduced exactly twice and in independent replay. The verifier also reconstructs synthetic data, checks selection winners, all checkpoint/source hashes, frozen-backbone fingerprints, matched tuning samplers and statistical decisions. Initial canary verifies nonzero sampler gradients by FP64 finite differences and exact causal-prefix equality.', '',
'This bounded research loop stops after the three preregistered rounds. Failed use cases and rejected policies remain in the report. Any follow-up must use a new run and reserved confirmation; do not tune directly on this result or turn a failed learning check into a sampler-success claim.', '',
'## Reproduction','',
'```bash',f'python3 -B auto-research/results/{H.name}/verify.py','```','',
'For an authorized training replay, use run.py --output with a fresh directory on the mounted SSD and direct TMPDIR there. Outputs refuse overwrite and the resource monitor covers the actual output directory.', '',
f'![Optimization trajectories](../graphs/{H.name}/trajectories.png)','']
p=ROOT/'auto-research/reports'/f'{H.name}.md'
with p.open('x') as f:f.write('\n'.join(lines))
print(p)
