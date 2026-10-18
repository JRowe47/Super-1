import json, time, re
from pathlib import Path
import numpy as np
from experiment import ROOT, save, sha
from analyze import LABELS

def main():
    s=json.loads((ROOT/'SUMMARY.json').read_text())
    p=json.loads((ROOT/'PROTOCOL.json').read_text())
    interventions=json.loads((ROOT/'INTERVENTIONS.json').read_text())
    def row(arm,mode,delay):return next(r for r in s['aggregates'] if r['arm']==arm and r['mode']==mode and r['delay']==delay)
    lines=['# Persistent curve parameters: short-horizon gains, failed retention gate','',
      '**Decision: no promotion.** All 15 main runs completed and all saved primary-delay endpoints replayed exactly. The adaptive parameter-curve arm lost to the exactly parameter/state-matched ordinary-context control on all three training seeds. This is a negative result for this specific first design, not a rejection of curve representations or self-modifying models.','',
      'The tested model makes 16 persistent coefficients control gate biases and pair mixing. The updater weights themselves remain fixed at inference; full self-reference, cross-episode learning, variable depth, consolidation, and language retention were not tested.','',
      '## Primary endpoint: conflicting writes after 256 intervening tokens','',
      '| Arm | Seed 101701 | Seed 101702 | Seed 101703 | Mean | Trainable parameters | FP32 state bytes |',
      '|---|---:|---:|---:|---:|---:|---:|']
    for arm in LABELS:
        r=row(arm,'conflict',256);c=s['counts'][arm]
        lines.append('| '+LABELS[arm]+' | '+' | '.join(f'{v*100:.2f}%' for v in r['per_seed_accuracy'])+f" | {r['mean_accuracy']*100:.2f}% | {c['parameters']:,} | {c['state_bytes_fp32']} |")
    lines += ['',f"Adaptive minus matched context: **{s['primary']['mean_difference']*100:.2f} percentage points**, with paired differences "+', '.join(f'{v*100:+.2f}' for v in s['primary']['paired_seed_differences'])+' points. The predeclared +5-point/all-seeds-positive rule failed. Three training seeds support a consistent result within this experiment; they do not establish broad architecture superiority. Chance accuracy is12.5%.','',
      'The matched context control retained locked values at26.04%; the adaptive arm reached14.06%. Neither solved the task reliably. Inadequate learning of the conditional write rule and the short training horizon limit the strength of architectural conclusions.','',
      '## Delay and replacement results','',
      '| Condition | Matched context | Adaptive curves |', '|---|---:|---:|']
    for mode,label,delay in [('retention','No conflicting writes,32 tokens',32),('retention','No conflicting writes,256 tokens',256),('retention','No conflicting writes,1,024 tokens',1024),('conflict','Conflicting writes,256 tokens',256),('reopen','Reopen and replace,256 tokens',256)]:
        lines.append(f"| {label} | {row('context64',mode,delay)['mean_accuracy']*100:.2f}% | {row('adaptive64',mode,delay)['mean_accuracy']*100:.2f}% |")
    lines += ['', 'Training delays were8/16/32 tokens. The32-token endpoint is within the trained range;256 and1,024 are extrapolation. Short-delay no-conflict recall favored adaptive curves over the matched context arm, but the wider fixed cell was close (64.39% versus65.56%). Thus the short gain does not establish that persistent parameter modulation is necessary.','',
      'The short reopened-key accuracy gap favored adaptive curves by12.50 points; this passed the secondary replacement safeguard but did not rescue the failed primary retention gate. Reopen-table accuracy averages all four final keys; the safeguard uses only the explicitly reopened key.','',
      f'![All arms and all seeds](../graphs/{ROOT.name}/retention.png)','',
      '## Registered mechanism interventions','',
      'These are evaluation-time changes to trained adaptive models, not retrained ablations. They are diagnostic and distribution-shifting. The base token-conditioned mixer remains active in the no-mixing-modulation intervention.','',
      '| Adaptive model intervention | No conflicting writes,256 | Conflicting writes,256 | Reopen/replace,256 |','|---|---:|---:|---:|']
    original=[row('adaptive64',mode,256)['mean_accuracy'] for mode in ('retention','conflict','reopen')]
    lines.append('| None | '+' | '.join(f'{v*100:.2f}%' for v in original)+' |')
    for key,label in [('zero_p','Zero both coefficient fields'),('no_gate','Disable adaptive gate modulation'),('no_mix','Disable adaptive mixing modulation')]:
        vals=[np.mean([next(q for q in r['scores'] if q['mode']==mode)['accuracy'] for r in interventions if r['intervention']==key]) for mode in ('retention','conflict','reopen')]
        lines.append('| '+label+' | '+' | '.join(f'{v*100:.2f}%' for v in vals)+' |')
    lines += ['', 'Gate modulation is the more concerning path for conflicting writes in this run: disabling it raises accuracy from22.59% to26.89%; disabling mixing modulation alone leaves the rounded aggregate unchanged. Zeroing both gives27.86%. This does not show that a gate-only model trained from scratch would behave the same, nor that mixing is universally harmless. For no-conflict retention, all removal interventions reduce the already-low adaptive score.','',
      f'![Example coefficient curves](../graphs/{ROOT.name}/curve_evolution.png)','',
      'The example stream and first training seed were fixed before plotting, not selected for correctness. The fields are smooth cosine functions over hidden-coordinate pairs. They are not the whole model, a token-time Fourier spectrum, or a continuous depth coordinate.','',
      '## Fidelity, resources, and preservation','',
      '- Zero adaptive updates reproduce the fixed48 model exactly; the ordinary-context extension does too.',
      '- Nonzero-controller chunk carry and causal-prefix comparisons match exactly. A float64 finite-difference check of an active updater weight agrees within1.98e-11 absolute gradient error.',
      '- An independent scalar simulator confirms the task labels. All five arms consume exactly identical examples, query targets, and input-token counts within each training seed.',
      '- Every main checkpoint completed3,000 optimizer updates. All15 primary-delay evaluations replay exactly from saved FP32 checkpoints, including per-episode outcomes.',
      f"- Main training consumed{s['training_cpu_seconds']/60:.2f} aggregate CPU minutes. Main launch including evaluation took{json.loads((ROOT/'main/LAUNCH_TERMINAL.json').read_text())['wall_seconds']/60:.2f} wall minutes with at most3 workers and1 PyTorch thread per worker. No deployment-speed claim is made.",
      '- The primary pair shares10,144 trainable parameters and64 FP32 state scalars. The state-byte count excludes weights, basis buffers, activations, and workspaces; a shared integer clock is additional. Standard GRU has more parameters. Compute and wall time were not matched.',
      '- All five250-update pilot outcomes are retained separately. Pilot and main evaluation seeds differ. Protocol/source hashes were frozen locally before main training; subsequent Git publication is not independent preregistration.',
      '- Source, metrics, reports and figures are submitted to the user’s own fork. All20 pilot/main checkpoints, every full-step trace, and the example curve trace are in the companion archive; hashes and preservation receipt identify it.',
      '- The original local macOS SSD instruction belongs to a different host. This run used the explicitly supplied conversation workspace and durable companion artifact; it does not claim to have mounted that SSD.',
      '- No official test, external corpus, teacher, GPU, paid provisioning, or language-model evaluation was used. No process remains running after completion.','',
      '## Next experiment','',
      'Do not add variable depth or full recursive weight updates yet. First establish reliable conditional-write learning with a stronger task-learning control, then test a bounded residual gate update initialized from an already competent fixed model. Pair that with ordinary context feedback, longer no-overwrite training intervals, and a replay/functional-drift penalty. Train gate-only and mixing-only changes separately rather than inferring their training behavior from the interventions here.','',
      'This follow-up is a proposal, not a running job. A small matched synthetic comparison can use the same CPU envelope; natural-language retention requires a separate transfer study.','',
      f'Protocol SHA-256: `{sha(ROOT/"PROTOCOL.json")}`. See [reproduction and mechanism](../results/{ROOT.name}/README.md), [raw summary](../results/{ROOT.name}/SUMMARY.json), and [replay checks](../results/{ROOT.name}/REPLAY_CHECKS.json).','']
    report=ROOT.parents[1]/'reports'/f'{ROOT.name}.md';report.parent.mkdir(exist_ok=True)
    text='\n'.join(lines)
    text=re.sub(r'\b(is|at|from|to|all|All|within|with|five|every|exactly|shares|after|completed|consumed|took|of|both|and|than|only)(?=\d)',r'\1 ',text)
    text=re.sub(r'(\d)(updates|tokens|points|intervening|primary|trainable|optimizer|paired|query|FP32|CPU)',r'\1 \2',text)
    report.write_text(text)
    registry_path=ROOT.parents[1]/'RUN_REGISTRY.json';registry=json.loads(registry_path.read_text())
    entry=next(r for r in registry['runs'] if r['run_id']==ROOT.name)
    entry.update(status='COMPLETE_NO_PROMOTION',metrics='auto-research/results/'+ROOT.name+'/SUMMARY.json',
                 terminal='auto-research/results/'+ROOT.name+'/TERMINAL.json',arm_seed_rows=15,
                 decision='Primary paired gain -7.88pp;all3 negative. Limited synthetic evidence;no architecture promotion.')
    save(registry_path,registry)
    save(ROOT/'TERMINAL.json',dict(status='COMPLETE_NO_PROMOTION',main_runs=15,pilot_runs=5,
        main_steps_per_run=3000,all_primary_replays_exact=True,primary=s['primary'],
        cleanup={'training_or_analysis_processes_left_running':0,'external_resources_created':0},
        authorization=p['authorization']['user_approval_reference'],
        utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    print(report)

if __name__=='__main__':main()
