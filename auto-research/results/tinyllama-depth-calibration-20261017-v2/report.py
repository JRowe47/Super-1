"""Verify recorded measurements and render the calibration experiment report."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[2]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(name):
    return json.loads((ROOT/name).read_text())


def save(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def paired(a,b):
    d=np.asarray(a)-np.asarray(b)
    return {'mean':float(d.mean()),'window_se':float(d.std(ddof=1)/math.sqrt(len(d))),
            'min':float(d.min()),'max':float(d.max()),'windows_improved':int((d<0).sum())}


def main():
    p,r,t=read('PROTOCOL.json'),read('RESULTS.json'),read('TERMINAL.json')
    assert t['status']=='COMPLETED'
    assert sha(ROOT/'PROTOCOL.json')==read('REGISTRATION.json')['protocol_sha256']
    assert sha(ROOT/'experiment.py')==p['design']['original_source_hashes']['experiment.py']
    assert sha(ROOT/'CANARY.json')==read('REGISTRATION.json')['canary_sha256']
    assert read('CANARY.json')['passed']
    assert sha(ROOT/'INPUT_MOMENTS.npz')==read('CALIBRATION.json')['input_moments_sha256']
    assert [x['id'] for x in r['rows']]==[x['id'] for x in p['design']['arms']]
    rows={x['id']:x for x in r['rows']}
    teacher=rows['teacher']
    comparisons={}
    for row in r['rows']:
        assert row['anchor_byte_exact']
        for region,m in row['metrics'].items():
            assert len(m['window_ce'])==16
            assert abs(float(np.mean(m['window_ce']))-m['ce'])<1e-12
            assert abs(math.exp(m['ce'])-m['perplexity'])<1e-8
            assert abs(float(np.mean(m['window_kl']))-m['forward_kl'])<1e-12
            assert abs(float(np.mean(m['window_agreement']))-m['teacher_top1_agreement'])<1e-12
            assert np.isfinite(m['window_ce']).all()
            comparisons[f"{row['id']}_vs_teacher_{region}"]=paired(m['window_ce'],teacher['metrics'][region]['window_ce'])
    for k in [19,15]:
        train=read(f'k{k}_TRAINING.json')
        assert train['steps']==1344 and len(train['epochs'])==4
        assert len(set(rows[f'k{k}_{kind}']['analytical_stored_scalars'] for kind in ['raw','activation','output']))==1
        for region in ['fresh_validation','reused_validation']:
            for control in ['raw','activation']:
                comparisons[f'k{k}_output_vs_{control}_{region}']=paired(
                    rows[f'k{k}_output']['metrics'][region]['window_ce'],rows[f'k{k}_{control}']['metrics'][region]['window_ce'])
    for ck in read('CHECKPOINTS.json')['checkpoints']:
        assert sha(ROOT/ck['file'])==ck['sha256']
        assert (ROOT/ck['file']).stat().st_size==ck['bytes']
    check=read('CHECKS.json')['checks']
    assert len(check)==4
    assert next(c for c in check if c['kind']=='teacher_duplicate')['max_logit_abs']<=1e-6
    assert next(c for c in check if c['kind']=='teacher_parent_replay')['max_window_ce_abs']<=2e-6
    for c in [x for x in check if x['kind']=='output_merge_and_curve']:
        assert max(c['per_layer_merge_relative_squared_error'])<=1e-9
        assert max(c['curve_relative_squared_residual'].values())<=1e-10
    primary={kind:rows[f'k19_{kind}']['metrics']['fresh_validation']['ce'] for kind in ['raw','activation','output']}
    teacher_ce=teacher['metrics']['fresh_validation']['ce']
    gates={'gain_vs_raw_at_least_0_10':primary['raw']-primary['output']>=.10,
           'gain_vs_activation_at_least_0_10':primary['activation']-primary['output']>=.10,
           'teacher_gap_at_most_0_30':primary['output']-teacher_ce<=.30}
    passed=all(gates.values())
    summary={'primary_gate_passed':passed,'gate_components':gates,'paired_window_changes':comparisons,
        'teacher_fresh_ce':teacher_ce,'teacher_reused_ce':teacher['metrics']['reused_validation']['ce'],
        'all_seven_arms_complete':True,'source_protocol_checkpoint_bindings_verified':True,
        'single_seed':True,'independent_confirmation':False,'official_test_accessed':False,
        'analytical_weight_savings':{str(k):rows[f'k{k}_raw']['analytical_weight_saving_fraction'] for k in [19,15]},
        'measurement_wall_seconds':t['wall_seconds'],'measurement_cpu_seconds':t['cpu_seconds']}
    save(ROOT/'SUMMARY.json',summary)
    save(ROOT/'REPORT_CHECKS.json',{'passed':True,'checks':['Frozen source and protocol hashes','Synthetic math and optimizer canary',
        'All seven declared arms in order','Every metric reconciled from 16 windows per region','Matched final storage per K',
        'All 1344 updates per output arm','Exact adapter checkpoint hashes','Teacher replay and duplicate',
        'First FFN byte-exact for every arm','All-layer merge fidelity and full weight depth-subspace residual'],
        'results_sha256':sha(ROOT/'RESULTS.json'),'checkpoints_manifest_sha256':sha(ROOT/'CHECKPOINTS.json')})
    graphs=REPO/'auto-research/graphs'/ROOT.name
    graphs.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
        'svg.fonttype':'none','svg.hashsalt':ROOT.name})
    colors={'raw':'#667085','activation':'#c67b1d','output':'#067d91'}
    names={'raw':'Weight fit','activation':'Input-aware fit','output':'Output recovery'}
    fig,axes=plt.subplots(1,2,figsize=(11.6,4.7),sharey=True)
    for ax,region,title in zip(axes,['fresh_validation','reused_validation'],['New frozen windows','Previously used windows']):
        for j,kind in enumerate(colors):
            xs=np.array([0,1])+(j-1)*.23
            ys=[rows[f'k{k}_{kind}']['metrics'][region]['ce'] for k in [19,15]]
            bars=ax.bar(xs,ys,width=.21,color=colors[kind],label=names[kind])
            ax.bar_label(bars,labels=[f'{y:.3f}' for y in ys],fontsize=8,padding=3)
        baseline=teacher['metrics'][region]['ce']
        ax.axhline(baseline,color='#273346',linestyle='--',linewidth=1.4,label=f'Teacher {baseline:.3f}')
        ax.set_xticks([0,1],['19 controls + exact first FFN\n6.29% fewer weight scalars','15 controls + exact first FFN\n18.87% fewer weight scalars'])
        ax.set_title(title);ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)
        ax.legend(frameon=False,fontsize=8,loc='upper left')
    axes[0].set_ylabel('Next-token cross-entropy (nats/token; lower is better)')
    fig.suptitle('Does calibration recover a pretrained model represented by depth curves?',fontsize=13,y=1.01)
    fig.text(.5,-.04,'All 22 blocks execute. FFNs only. 2,048 targets per panel, reset context 128. One training seed. Storage is analytical.',ha='center',fontsize=9,color='#475467')
    fig.tight_layout()
    fig.savefig(graphs/'calibration_quality.png',dpi=180,bbox_inches='tight')
    fig.savefig(graphs/'calibration_quality.svg',bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11.6,4.1),sharey=True)
    for ax,k in zip(axes,[19,15]):
        for kind in colors:
            vals=read(f'k{k}_{kind}_LOCAL.json')['per_layer_relative_mse']
            ax.semilogy(np.arange(2,23),np.maximum(vals,1e-12),marker='.',color=colors[kind],label=names[kind])
        ax.set(title=f'{k} controls + exact first FFN',xlabel='Executed block (1-based; exact block 1 omitted)',xticks=[2,6,10,14,18,22])
        ax.grid(alpha=.18);ax.legend(frameon=False,fontsize=8)
    axes[0].set_ylabel('Relative squared FFN-output error\non 1,024 training positions')
    fig.tight_layout();fig.savefig(graphs/'local_recovery.png',dpi=170,bbox_inches='tight');fig.savefig(graphs/'local_recovery.svg',bbox_inches='tight');plt.close(fig)
    save(graphs/'inputs.json',{'results_sha256':sha(ROOT/'RESULTS.json'),'plot_source':str((ROOT/'report.py').relative_to(REPO)),
        'plot_source_sha256':sha(__file__),'local_inputs':{f'k{k}_{kind}_LOCAL.json':sha(ROOT/f'k{k}_{kind}_LOCAL.json') for k in [19,15] for kind in colors}})
    lines=['| Fit | Controls + exact FFN | Weight scalars saved | New-window CE | Reused-window CE | New-window teacher KL | New-window top-1 agreement |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in r['rows']:
        f=row['metrics']['fresh_validation'];v=row['metrics']['reused_validation']
        label='Original teacher' if row['id']=='teacher' else names[row['kind']]
        count='22 original' if row['id']=='teacher' else f"{row['controls']} + 1"
        lines.append(f"| {label} | {count} | {100*row['analytical_weight_saving_fraction']:.2f}% | {f['ce']:.6f} | {v['ce']:.6f} | {f['forward_kl']:.6f} | {100*f['teacher_top1_agreement']:.2f}% |")
    effects=[]
    for k in [19,15]:
        raw=rows[f'k{k}_raw']['metrics']['fresh_validation']['ce'];out=rows[f'k{k}_output']['metrics']['fresh_validation']['ce']
        local_before=read(f'k{k}_activation_LOCAL.json')['mean'];local_after=read(f'k{k}_output_LOCAL.json')['mean']
        comp=comparisons[f'k{k}_output_vs_raw_fresh_validation']
        effects.append(f"- **K={k}:** output recovery changes fresh-window CE by {out-raw:+.6f} nats/token versus the raw fit (paired-window descriptive SE {comp['window_se']:.6f}; improves {comp['windows_improved']}/16 windows). Final teacher gap {out-teacher_ce:.6f}. Mean normalized training FFN-output error falls from {local_before:.6f} to {local_after:.6f}.")
    ck=read('CHECKPOINTS.json')['checkpoints']
    report=f'''# TinyLlama anchored depth-curve calibration

The predeclared K19 continuation gate **{'passed' if passed else 'failed'}**. This is an exploratory, single-seed local-output recovery experiment, not an independent confirmation or a deployed compression result. All seven declared arms and all 2,688 optimizer updates completed.

## Question and chronology

Earlier experiments found little raw cross-layer weight structure and a disproportionate benefit from keeping the first FFN exact. This run was designed after those outcomes, then source, data positions, hyperparameters, checks and decision thresholds were frozen before measurement. It compares raw fitting, diagonal-input-aware fitting, and local FFN-output recovery with identical final curve-control storage at each K.

The protocol was published to the existing user-fork draft branch before measurement. It uses the same pinned public TinyLlama 1.1B 3T checkpoint and tokenizer. The official WikiText test was not accessed. This work does not alter the project's historical recurrent architecture or claim to have converted TinyLlama attention into that recurrence.

## Results

{chr(10).join(lines)}

![Calibration quality](../graphs/{ROOT.name}/calibration_quality.png)

{chr(10).join(effects)}

The primary gate required K19 output recovery to improve CE by at least 0.10 nats/token over **both** raw and input-aware fits, and to finish within 0.30 nats/token of the teacher. Components: `{json.dumps(gates,sort_keys=True)}`. These thresholds are continuation criteria, not statistical significance tests. Every arm is retained; no checkpoint or hyperparameter was selected using validation.

## Representation and fitting

For each of gate, up and down projection families, blocks 2–22 follow a matrix-valued clamped cubic spline with K controls; block 1 keeps its original FFN exactly. All original 22 blocks, attention, residuals and norms still execute in order. The depth parameter does not remove composition or supply a variable-depth execution rule.

Let Q be an orthonormal basis for the sampled spline space. The raw fit is Q Q^T X across depth. The input-aware fit separately minimizes a diagonal-input proxy per input column, using teacher training activations: Q(Q^T D_c Q)^-1 Q^T D_c X_c. A small predeclared floor keeps D_c positive. For the down projection, these moments come from the **original teacher** gate/up intermediate, so they do not account exactly for the nonlinear error introduced when all three matrices change together.

The recovery arm starts from the input-aware fit. Each shared control matrix receives a rank-8 update 2 A_k B_k; the executed matrix receives the fixed spline mixture of these updates. Training minimizes each block's FFN-output MSE divided by that block's teacher-output mean square, on cached teacher inputs. The first FFN remains frozen. Updates are shared through the spline basis, not stored as independent adapters per executed layer. After fitting, updates are folded into the same K controls. The experiment evaluates equivalent materialized weights.

Low-rank factors are a computationally limited optimization parameterization, following the general [LoRA idea](https://arxiv.org/abs/2106.09685); this experiment places them in shared depth controls. This is not a claim that the method or depth compression is novel.

## Data, training and controls

- Calibration: 8 fixed training windows, 1,024 token positions, 128-token reset context. Nonempty raw lines joined with two newlines; windows may cross article boundaries.
- Primary evaluation: 16 new frozen validation windows, 2,048 next-token targets. Positions do not overlap the earlier 16 windows. The containing validation split has already been used in this project, so this is not globally untouched confirmation.
- Secondary evaluation: exactly the earlier 16 windows, another 2,048 targets. Teacher CE reproduces the parent to the frozen tolerance.
- Output recovery: four epochs, 64 token inputs for one block per update, all 21 non-anchor blocks per minibatch; **1,344 updates per K**, one seed, one fixed learning rate/schedule, final checkpoint only. Each block sees each calibration input four times. Cached teacher inputs omit accumulated student-state drift.
- Raw/input-aware arms have no optimizer updates. Final storage is matched, fitting compute is not. Training is local FFN distillation, not end-to-end token-loss or logit-KL distillation.

![Local output recovery](../graphs/{ROOT.name}/local_recovery.png)

The exact source and protocol are under `auto-research/results/{ROOT.name}/`. `RESULTS.json` preserves per-window CE, full-vocabulary teacher-forward KL and top-1 agreement. `SUMMARY.json` includes paired differences and descriptive standard errors. Standard errors over 16 windows do not justify population claims, and only one training seed was tested.

## Fidelity and storage

Synthetic checks cover independent weighted least squares, a full-rank identity, factored-versus-dense forward/input/parameter gradients, and a complete AdamW step. Full-size checks cover the byte-exact first FFN, teacher replay/duplicate, all-layer factor-merge agreement, and the residual of the entire fitted weight stack outside the intended spline space. All passed their frozen tolerances. The report checks source, protocol, input-moment and checkpoint hashes and recomputes every aggregate.

The unchanged teacher has {r['teacher_parameters']:,} scalars; each FFN has {r['one_ffn_parameters']:,}. K19 plus one exact FFN saves 69,206,016 scalars (6.29% of model weights); K15 plus one exact FFN saves 207,618,048 (18.87%). These counts retain all non-FFN weights. They exclude fixed geometry metadata but include every learned dense control. There are no extra learned depth coefficients. Rank-8 training factors have {ck[0]['trainable_scalars']:,} and {ck[1]['trainable_scalars']:,} scalars respectively and fold into the counted controls.

Exact FP32 factors are preserved as `k19_adapter.safetensors` and `k15_adapter.safetensors`, with sizes/hashes in `CHECKPOINTS.json`. `INPUT_MOMENTS.npz` plus the pinned original model deterministically defines the base fit. A compact full checkpoint and compact inference kernel were **not** exported. Actual inference weights are materialized; analytical savings are not measured RAM or speed gains.

Measurement wall time: {t['wall_seconds']:.1f} seconds; process CPU: {t['cpu_seconds']:.1f} seconds, four CPU threads. No GPU or paid resources were provisioned. Timings include fitting/evaluation and are not deployment latency benchmarks.

## Decision and limits

{'The fixed small-recovery gate passed; replication and a stronger data budget are warranted before any deployment claim.' if passed else 'The fixed small-recovery gate failed. Do not promote this as a successful quality-preserving conversion. Local output recovery and end-to-end prediction changes should be judged separately.'}

This isolates a bounded input-aware fit and low-rank local-output recovery within fixed smooth depth spaces. A failure does not rule out full-control optimization, better neuron alignment, different curve geometry or end-to-end distillation. Those require separate controls and budgets. The next choice should follow the measured gap and calibration-vs-end-to-end mismatch; no new run is automatically launched.
'''
    report_path=REPO/'auto-research/reports'/f'{ROOT.name}.md'
    report_path.write_text(report)
    registry_path=REPO/'auto-research/RUN_REGISTRY.json'
    registry=json.loads(registry_path.read_text())
    entry=next(x for x in registry['runs'] if x['run_id']==ROOT.name)
    entry.update(status='COMPLETED_EXPLORATORY_GATE_PASS' if passed else 'COMPLETED_EXPLORATORY_GATE_FAIL',
        terminal=str((ROOT/'TERMINAL.json').relative_to(REPO)),decision=f"K19 continuation gate {'passed' if passed else 'failed'}; all seven arms complete; single seed, no independent confirmation",
        training_updates=2688,checkpoints=str((ROOT/'CHECKPOINTS.json').relative_to(REPO)))
    registry_path.write_text(json.dumps(registry,indent=2)+'\n')
    print(json.dumps({'primary_gate_passed':passed,'teacher_fresh_ce':teacher_ce,'primary':primary,'gates':gates,'report':str(report_path)},indent=2))


if __name__=='__main__':main()
