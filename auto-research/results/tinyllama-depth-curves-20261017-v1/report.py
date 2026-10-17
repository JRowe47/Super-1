"""Verify saved outcomes, render figures, and write both completed reports."""
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
ANCHOR=ROOT.parent/'tinyllama-depth-anchor-20261017-v1'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def main():
    p=json.loads((ROOT/'PROTOCOL.json').read_text())
    s=json.loads((ROOT/'STRUCTURE.json').read_text())
    f=json.loads((ROOT/'FUNCTIONAL.json').read_text())
    t=json.loads((ROOT/'TERMINAL.json').read_text())
    ap=json.loads((ANCHOR/'PROTOCOL.json').read_text())
    af=json.loads((ANCHOR/'RESULTS.json').read_text())
    at=json.loads((ANCHOR/'TERMINAL.json').read_text())
    assert t['status']==at['status']=='COMPLETED'
    assert [r['id'] for r in f['rows']]==[r['id'] for r in p['design']['arms']]
    assert [r['id'] for r in af['rows']]==[r['id'] for r in ap['arms']]
    assert not f['failures']
    for folder in [ROOT,ANCHOR]:
        assert sha(folder/'PROTOCOL.json')==json.loads((folder/'REGISTRATION.json').read_text())['protocol_sha256']
    for name,digest in json.loads((ROOT/'SOURCES.json').read_text())['files'].items():
        assert sha(ROOT/name)==digest
    assert sha(ANCHOR/'anchor_probe.py')==ap['source_sha256']
    assert sha(ROOT/'FUNCTIONAL.json')==ap['parent_functional_sha256']
    assert sha(ROOT/'PROJECTORS.json')==ap['parent_projectors_sha256']
    by_id={r['id']:r for r in f['rows']}
    ab={r['id']:r for r in af['rows']}
    teacher=by_id['teacher']
    deltas={}
    for r in f['rows']+af['rows']:
        assert len(r['window_ce'])==16
        assert abs(np.mean(r['window_ce'])-r['validation_ce'])<1e-12
        assert abs(math.exp(r['validation_ce'])-r['perplexity'])<1e-9
        assert all(np.isfinite(r['window_ce']))
        delta=np.array(r['window_ce'])-np.array(teacher['window_ce'])
        deltas[r['id']]={'mean_ce_change':float(delta.mean()),'window_se':float(delta.std(ddof=1)/4)}
    assert by_id['spline22_roundtrip']['roundtrip_ce_abs']<=2e-6
    assert ab['teacher']['max_window_ce_replay_abs']<=2e-6
    assert ab['pca20_replay']['max_window_ce_replay_abs']<=2e-6
    weights=np.array([np.trace(np.array(s[k]['gram'])) for k in s])
    structure=[]
    for kind in ['pca','spline','cosine']:
        for rank in p['design']['structure_ranks']:
            entries=[next(r for r in s[k]['rows'] if r['kind']==kind and r['rank']==rank) for k in s]
            structure.append({'kind':kind,'rank':rank,
                'relative_squared_weight_error':float(np.average([e['relative_squared_weight_error'] for e in entries],weights=weights))})
    other=ap['other_block']
    first=ab['pca20_restore_block0'];control=ab[f'pca20_restore_block{other}']
    effect=control['validation_ce']-first['validation_ce']
    primary_pass=by_id['spline8']['validation_ce']-teacher['validation_ce']<=.10 and by_id['spline8']['validation_ce']-by_id['pca8']['validation_ce']<=.05
    summary={'primary_gate_passed':primary_pass,'diagnostic_gate_passed':effect>=1.,
             'restoring_first_vs_other_ce_advantage':effect,'other_restored_block':other,
             'weighted_structure':structure,'paired_window_changes':deltas,
             'original_parameters':f['teacher_parameters'],'original_ffn_parameters':f['teacher_ffn_parameters'],
             'validation_targets_per_arm':2048,'training_updates':0,'primary_arms':len(f['rows']),'diagnostic_arms':len(af['rows']),
             'primary_protocol_sha256':sha(ROOT/'PROTOCOL.json'),'diagnostic_protocol_sha256':sha(ANCHOR/'PROTOCOL.json'),
             'reconstruction_ce_abs':by_id['spline22_roundtrip']['roundtrip_ce_abs'],
             'reconstruction_max_logit_abs':by_id['spline22_roundtrip']['roundtrip_first_window_max_logit_abs'],
             'source_manifest_verified':True,'all_outcomes_complete':True,'official_test_accessed':False,
             'memory_note':'psutil reported a constant implausible 2,125,824 RSS bytes even for the loaded 1.1B model. Do not treat that field as a measured model footprint. ru_maxrss is retained as OS-reported process high-water mark, not compact deployment memory.'}
    save(ROOT/'SUMMARY.json',summary)
    save(ROOT/'CHECKS.json',{'passed':True,'checks':['frozen protocol hashes','frozen measurement source hashes','parent bindings for diagnostic','all declared arms present in order','no primary failures','per-window aggregate and perplexity reconciliation','finite scores','full-rank reconstruction','fresh teacher and PCA20 replay'],
         'primary_functional_sha256':sha(ROOT/'FUNCTIONAL.json'),'diagnostic_results_sha256':sha(ANCHOR/'RESULTS.json')})
    save(ANCHOR/'SUMMARY.json',{'diagnostic_gate_passed':effect>=1.,'block0_vs_other_ce_advantage':effect,'other_block':other,
         'source_and_parent_bindings_verified':True,'independent_confirmation':False})
    save(ANCHOR/'CHECKS.json',{'passed':True,'teacher_max_window_ce_replay_abs':ab['teacher']['max_window_ce_replay_abs'],
         'pca20_max_window_ce_replay_abs':ab['pca20_replay']['max_window_ce_replay_abs'],'protocol_sha256':sha(ANCHOR/'PROTOCOL.json'),
         'source_sha256':sha(ANCHOR/'anchor_probe.py')})
    plots=REPO/'auto-research/graphs'/ROOT.name
    anchor_plots=REPO/'auto-research/graphs'/ANCHOR.name
    plots.mkdir(parents=True,exist_ok=True);anchor_plots.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'tinyllama-depth-20261017','font.family':'DejaVu Sans'})
    colors={'pca':'#3064b3','spline':'#bf5b19','cosine':'#7c54a4'}
    names={'pca':'Shared PCA bases','spline':'Cubic depth spline','cosine':'Cosine depth curve'}
    fig,axes=plt.subplots(1,2,figsize=(12.6,5.3))
    for kind in colors:
        r=[r for r in structure if r['kind']==kind]
        axes[0].plot([v['rank'] for v in r],[100*(1-v['relative_squared_weight_error']) for v in r],marker='o',color=colors[kind],label=names[kind],lw=2)
    axes[0].plot([0,22],[0,100],':',color='#7d858e',label='K / 22 reference')
    axes[0].set(xlabel='Stored matrix bases per FFN projection',ylabel='Original squared weight energy retained (%)',xlim=(3,22.5),ylim=(10,103),title='Raw weights show little shared depth structure')
    axes[0].legend(frameon=False,fontsize=8.5);axes[0].grid(alpha=.18)
    for kind in colors:
        r=sorted([r for r in f['rows'] if r['kind']==kind and r['rank']<22],key=lambda x:x['analytical_weight_saving_fraction'])
        axes[1].plot([100*v['analytical_weight_saving_fraction'] for v in r],[v['validation_ce'] for v in r],marker='o',lw=2,color=colors[kind],label=names[kind])
        for v in r:
            shift=(5,-14) if kind=='pca' else (5,8)
            axes[1].annotate(f"K={v['rank']}",(100*v['analytical_weight_saving_fraction'],v['validation_ce']),xytext=shift,textcoords='offset points',fontsize=8,color=colors[kind])
    axes[1].axhline(teacher['validation_ce'],color='#25303b',ls='--',lw=1.5,label=f"TinyLlama reference: {teacher['validation_ce']:.3f}")
    axes[1].set(xlabel='Analytical whole-model FP32 weight saving (%)',ylabel='Validation cross-entropy (nats/token; lower is better)',title='Compression errors damage language predictions',xlim=(0,49))
    axes[1].legend(frameon=False,fontsize=8.5,loc='upper left');axes[1].grid(alpha=.18)
    fig.suptitle('TinyLlama 1.1B · FFN-only depth conversion · all 22 blocks still execute',fontsize=13,fontweight='bold')
    fig.text(.5,.025,'No recovery training. Same 2,048 validation targets at context 128. Storage is analytical; evaluation materializes all weights.',ha='center',fontsize=9,color='#515c68')
    fig.tight_layout(rect=(0,.07,1,.94))
    fig.savefig(plots/'depth_curves.png',dpi=180);fig.savefig(plots/'depth_curves.svg',metadata={'Date':None});plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12.6,5.3))
    for ident,kind in [('pca20','pca'),('spline20','spline')]:
        errs=by_id[ident]['local_ffn_diagnostics']['train']['per_layer_relative_squared_output_error']
        axes[0].plot(np.arange(22)+1,errs,marker='o',markersize=4,lw=2,color=colors[kind],label=names[kind]+' · K=20')
    axes[0].set(xlabel='Decoder block (1-based)',ylabel='Relative squared FFN output error',title='PCA sacrifices particular blocks',xticks=[1,4,8,12,16,20,22])
    axes[0].legend(frameon=False,fontsize=9);axes[0].grid(alpha=.18)
    bars=[ab['teacher'],ab['pca20_replay'],control,first]
    labels=['Original TinyLlama','PCA20',f'Restore block {other+1}','Restore block 1']
    palette=['#6d7784','#3064b3','#9a6db8','#167f76']
    axes[1].barh(labels,[r['validation_ce'] for r in bars],color=palette,height=.62)
    axes[1].invert_yaxis()
    for i,r in enumerate(bars):axes[1].text(r['validation_ce']+.08,i,f"{r['validation_ce']:.3f}",va='center',fontsize=10)
    axes[1].set(xlabel='Validation cross-entropy (nats/token; lower is better)',xlim=(0,max(r['validation_ce'] for r in bars)+.8),title='Equal-size restorations have unequal effects')
    axes[1].grid(axis='x',alpha=.18);axes[1].set_axisbelow(True)
    fig.suptitle('Diagnostic: does the first FFN carry unusually consequential computation?',fontsize=13,fontweight='bold')
    fig.text(.5,.025,f'Post-hoc follow-up, registered before intervention. Block {other+1} selected by training-window error. Both restorations add 34.6M weights.',ha='center',fontsize=9,color='#515c68')
    fig.tight_layout(rect=(0,.07,1,.94))
    fig.savefig(anchor_plots/'anchor_probe.png',dpi=180);fig.savefig(anchor_plots/'anchor_probe.svg',metadata={'Date':None});plt.close(fig)
    for svg in [plots/'depth_curves.svg',anchor_plots/'anchor_probe.svg']:
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    save(plots/'inputs.json',{'weighted_structure':structure,'functional_sha256':sha(ROOT/'FUNCTIONAL.json')})
    save(anchor_plots/'inputs.json',{'results_sha256':sha(ANCHOR/'RESULTS.json'),'parent_functional_sha256':sha(ROOT/'FUNCTIONAL.json')})
    table='\n'.join(f"| {r['id']} | {r['rank']} | {100*r['analytical_weight_saving_fraction']:.2f}% | {r['validation_ce']:.6f} | {r['forward_kl']:.6f} | {100*r['teacher_top1_agreement']:.2f}% |" for r in f['rows'])
    stable='\n'.join(f"| {family} | {s[family]['adjacent_cosine_mean']:.6f} | "+' | '.join(f"{100*next(x['relative_squared_weight_error'] for x in s[family]['rows'] if x['kind']==kind and x['rank']==8):.2f}%" for kind in ['pca','spline','cosine'])+' |' for family in s)
    report=f'''# TinyLlama: pretrained FFN weights as curves through depth

Status: **completed exploratory screen; the K=8 conversion gate failed**. All nine declared evaluations completed. No recovery training, architecture promotion, or independent confirmation is claimed.

The user authorized exploring the proposed TinyLlama FFN-only shared-basis/depth-curve conversion. Source and protocol were frozen locally before measurement; the initial protocol tree was published to JRowe47/Super-1 in commit `200efe83c8ff6dae81a7f2523e90a71f97701bcc`. Functional measurement began before that remote publication completed. Protocol SHA256: `{sha(ROOT/'PROTOCOL.json')}`.

## Question and controls

Can corresponding FFN weights of a pretrained 22-block language model be represented by far fewer shared matrix bases, with coefficients constrained to smooth functions of depth?

The checkpoint is public TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T, revision `59f6f375b26bde864a6ca194a9a3044570490064`, safetensors SHA256 `9dc1c1a8c6b29e153dea7aaf2ed6c4d242163489add0e7c14c23889745437be4`. It has {f['teacher_parameters']:,} parameters, including {f['teacher_ffn_parameters']:,} in the three FFN projections across 22 blocks.

For each projection family, flatten corresponding matrices into X with 22 rows. Fit Xhat = Q Qᵀ X. PCA uses a constant depth mode and the leading K−1 directions of the centered layer Gram matrix. Splines use K uniformly spaced, clamped cubic B-spline controls. Cosines use K low-frequency DCT-II modes. All use K **full matrices** per family; this does not impose low matrix rank within a layer. Free PCA coefficient storage is counted separately. A rank22 spline is a reconstruction control. Sixty-four fixed random depth permutations provide a structural ordering control.

All attention, normalization, embeddings, output weights, residual connections, RoPE and execution depth remain the pretrained architecture. There is no recurrent conversion or variable-depth evaluation. No neuron/head alignment was performed; this is raw pretrained coordinate space.

## What actually ran

CPU FP32 inference, four threads; FP64 Gram matrices and depth fits. PyTorch {p['execution']['runtime_versions']['torch']}, Transformers {p['execution']['runtime_versions']['transformers']}. The CPU identifies as AMD EPYC 9V74. Setup used an already available CPU PyTorch runtime and a separate environment for the remaining dependencies. No GPU or paid service was used.

Exact input hashes, pinned dataset revision, tokenizer identity and sampling positions are in [INPUTS.json](../results/{ROOT.name}/INPUTS.json) and [PROTOCOL.json](../results/{ROOT.name}/PROTOCOL.json). WikiText-2 raw **train and validation only** were downloaded. Sixteen fixed windows evenly span the tokenized validation stream; each has 128 inputs and 128 next-token targets, totaling 2,048 targets. Windows reset, may cross concatenated article boundaries, and provide at most 128-token history. Text joins nonempty raw lines with two newlines. These are TinyLlama-token metrics, not standard word-level WikiText perplexity.

Four train windows were frozen; only the first is used for the local FFN diagnostic in this pass. The first validation window supplies a second local diagnostic. All fits minimize weight error, without activation-based fitting, gradient updates, or distillation. Validation has been used elsewhere in the project and is not an untouched confirmation split. Pretraining contamination is unaudited.

## Results

| Arm | Matrix bases per projection | Analytical total weight saving | CE ↓ (nats/token) | Forward KL ↓ | Teacher top-1 agreement ↑ |
|---|---:|---:|---:|---:|---:|
{table}

![Structural fits and language loss](../graphs/{ROOT.name}/depth_curves.png)

The full-rank spline has exactly zero observed logit/CE drift in the declared reconstruction checks. The original teacher's repeated first window also has zero logit drift. Thus the poor compressed scores are not explained by the tested reconstruction or inference adapter.

At K=8, the primary gate required <=0.10 nats/token degradation from TinyLlama and <=0.05 nats/token spline disadvantage relative to PCA. Neither condition holds. The stored K=8 approximation would analytically remove about 44% of total weight scalars, but is not a useful converted model.

| Projection | Adjacent raw-weight cosine | PCA8 discarded energy | Spline8 discarded energy | Cosine8 discarded energy |
|---|---:|---:|---:|---:|
{stable}

The spectrum is relatively flat and natural ordering gives only small improvements over shuffled ordering in raw weight error; all shuffle ranges are retained in STRUCTURE.json. This does **not** establish that the model lacks functional redundancy, that all tensor factorizations fail, or that alignment/distillation cannot help.

At K=20, PCA minimizes the tested centered weight-space objective better than the spline, yet its language loss is much worse. Its depth projector gives the first block only roughly 0.056–0.064 diagonal self-contribution across the three projections. Its local train-window FFN output error is nearly 1.0 at block0 and block7. This motivated the separately registered, post-hoc [anchor diagnostic](tinyllama-depth-anchor-20261017-v1.md); those results are not pooled into the original gate.

Per-window CE values, changes and descriptive standard errors are retained in FUNCTIONAL.json and SUMMARY.json. There is one fixed pretrained model and no training-seed replication. Standard errors over these 16 selected windows are not population-level confidence bounds.

## Deployment and cost

The compact parameter counts are **analytical**: shared full matrix bases, small free PCA coefficients, and all unchanged non-FFN parameters. No compact checkpoint or optimized curve kernel was deployed. Functional evaluation materializes the reconstructed 22-layer weights, using the original checkpoint to compute them. Weight savings therefore do not describe the memory used by this experiment.

Structure measurement took {json.loads((ROOT/'STRUCTURE_CHECKS.json').read_text())['wall_seconds']:.2f}s. Functional execution, including model loading, local diagnostics and reconstructions, took {t['wall_seconds']:.1f}s wall / {t['cpu_seconds']:.1f}s process CPU. These are run costs, not prefill/decode or deployment-speed comparisons. Peak compact deployment RAM/VRAM, KV state, allocator reserve and p50/p95 latency are unmeasured. Recorded psutil RSS is implausible and excluded from conclusions; the OS-reported ru_maxrss high-water mark is {t['max_rss_kib']:,} KiB for this materialized experiment.

## Decision and next experiment

**Do not promote an untrained raw-weight curve conversion.** The useful next question is how to preserve functionally important computation while learning a shared representation. The follow-up tests whether a small exact exception at the input end explains the PCA failure. Subsequent alternatives include function-preserving FFN neuron alignment and activation/output-aware fitting with explicit endpoint protection. Neither was tested in this primary screen.

## Integrity, artifacts and reproduction

- [Frozen source](../results/{ROOT.name}/experiment.py), [input acquisition](../results/{ROOT.name}/acquire.py), [source hashes](../results/{ROOT.name}/SOURCES.json), [registration](../results/{ROOT.name}/REGISTRATION.json).
- [All structural results](../results/{ROOT.name}/STRUCTURE.json), [exact depth projectors](../results/{ROOT.name}/PROJECTORS.json), [functional outcomes](../results/{ROOT.name}/FUNCTIONAL.json), [checks](../results/{ROOT.name}/CHECKS.json), [terminal receipt](../results/{ROOT.name}/TERMINAL.json).
- [Reproduction notes](../results/{ROOT.name}/README.md). Inputs are external public artifacts, recoverable by revision and verified SHA256. Large downloaded weights and corpus are not committed. No trained or transformed checkpoint requiring separate preservation was created.

All source/protocol/aggregate checks pass; nine of nine arms completed, zero failed arms. The official test was never downloaded or evaluated. No cloud resources, schedules, or follow-up jobs are running.
'''
    report_dir=REPO/'auto-research/reports'
    (report_dir/f'{ROOT.name}.md').write_text(report)
    anchor_table='\n'.join(f"| {r['id']} | {100*r['analytical_weight_saving_fraction']:.2f}% | {r['validation_ce']:.6f} | {r['forward_kl']:.6f} | {100*r['teacher_top1_agreement']:.2f}% |" for r in af['rows'])
    anchor_report=f'''# TinyLlama: restoring a damaged FFN after depth compression

Status: **completed post-hoc diagnostic**, four of four arms. This follow-up was chosen after observing the complete [primary conversion screen]({ROOT.name}.md). It is not an independent confirmation. Its protocol was frozen before the restoration evaluations; SHA256 `{sha(ANCHOR/'PROTOCOL.json')}`.

## Question and equal-size control

The parent PCA20 fit gave nearly unit relative FFN output error at zero-based blocks0 and7, despite preserving more weight energy than the spline. Does the location of the damaged computation matter?

Keep the same PCA20 approximation and restore all three original FFN matrices in exactly one block. Compare block0 with block{other}, selected before intervention by the **largest parent training-window local error among nonzero blocks**. Each restoration adds exactly 34,603,008 weight scalars while retaining all shared bases and coefficients. This is an explicit original-layer exception, not a pure smooth curve. It does not create an extra executed layer.

## Results

Same pinned teacher, original 22-block execution, FP32 runtime, and repeated 2,048 validation targets as the parent. No training or tuning.

| Arm (0-based block identifiers) | Analytical total weight saving | CE ↓ (nats/token) | Forward KL ↓ | Teacher top-1 agreement ↑ |
|---|---:|---:|---:|---:|
{anchor_table}

![Damage and restoration diagnostic](../graphs/{ANCHOR.name}/anchor_probe.png)

Plots use **1-based** block labels for readability; JSON/arm names use **0-based** indices. The first block is block0 / plot block1. The control is block{other} / plot block{other+1}.

Restoring the first FFN improves CE by {ab['pca20_replay']['validation_ce']-first['validation_ce']:.6f} nats/token. Restoring the comparison FFN improves it by {ab['pca20_replay']['validation_ce']-control['validation_ce']:.6f}. The first-block advantage is **{effect:.6f} nats/token**. The predeclared diagnostic threshold was at least 1.0 nat/token; it **{'passed' if effect>=1. else 'did not pass'}**.

The teacher and unrepaired PCA20 were rerun and reproduce every parent window CE exactly. The effect is therefore measured against freshly replayed controls, not a changed evaluation setup.

## Interpretation and limits

This tests the consequence of restoring two particular FFNs at equal stored-parameter cost. It does not establish a universal ranking of layer importance, a general causal explanation of the pretraining process, or a successful compressed deployment. Both blocks were selected after the parent run; selection used local training-window error, and validation is reused. Other alignments, activations, rank allocations and recovered-training configurations remain untested.

The result supports using function-sensitive preservation criteria when building a depth representation. A small global weight reconstruction error can hide a very damaging local change. The next bounded experiment should keep the first FFN exact and compare activation/output-aware basis fitting with the same storage-budget raw fit; neuron permutation alignment is a separate control.

## Execution and integrity

Four CPU threads, no GPU, zero optimizer updates. Total {at['wall_seconds']:.1f}s wall / {at['cpu_seconds']:.1f}s process CPU, including loading, reconstruction and evaluation. Compact deployment memory and latency are unmeasured. All source, parent-result, projector and protocol bindings verified. All four arms completed, official test untouched, no paid resources or continued jobs.

[Source](../results/{ANCHOR.name}/anchor_probe.py) · [Protocol](../results/{ANCHOR.name}/PROTOCOL.json) · [Registration](../results/{ANCHOR.name}/REGISTRATION.json) · [All results](../results/{ANCHOR.name}/RESULTS.json) · [Checks](../results/{ANCHOR.name}/CHECKS.json) · [Terminal receipt](../results/{ANCHOR.name}/TERMINAL.json).
'''
    (report_dir/f'{ANCHOR.name}.md').write_text(anchor_report)
    registry_path=REPO/'auto-research/RUN_REGISTRY.json'
    registry=json.loads(registry_path.read_text())
    for run in registry['runs']:
        if run['run_id']==ROOT.name:run.update(status='COMPLETED_NEGATIVE',decision='K8 functional conversion gate failed; all outcomes retained')
        if run['run_id']==ANCHOR.name:run.update(status='COMPLETED_DIAGNOSTIC',decision=f'First-block restoration advantage {effect:.6f} nats/token; post-hoc, not confirmation')
    registry_path.write_text(json.dumps(registry,indent=2)+'\n')
    save(ROOT/'REPORT_INPUTS.json',{'script_sha256':sha(__file__),'functional_sha256':sha(ROOT/'FUNCTIONAL.json'),
        'structure_sha256':sha(ROOT/'STRUCTURE.json'),'diagnostic_sha256':sha(ANCHOR/'RESULTS.json')})
    print(json.dumps({'primary_gate':primary_pass,'anchor_advantage':effect,'teacher_ce':teacher['validation_ce'],'restored_first_ce':first['validation_ce']}))


if __name__=='__main__':main()
