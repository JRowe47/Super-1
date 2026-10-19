"""Render measured results, figures, and a standalone reproduction bundle."""
import base64,csv,hashlib,html,json,math,zipfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import ROOT,OUT,ARMS,dump
from evaluation import cluster_ci

LABELS={'initial_interpolated':'Interpolated · initial','initial_retained':'Retained · initial',
        'phase5_dense':'Phase 5 dense','trained_interpolated':'Interpolated · trained','trained_retained':'Retained · trained'}
COLORS={'initial_interpolated':'#8baccb','initial_retained':'#84b2a4','phase5_dense':'#898f9b',
        'trained_interpolated':'#2169ad','trained_retained':'#138065'}

def read(name):return json.loads((OUT/name).read_text())

def summarize():
    rows=read('audit.json');audit={r['configuration']['id']:r['metrics'] for r in rows}
    manifest=json.loads((ROOT/'data/dataset_manifest.json').read_text())
    docs=[r['document_index'] for r in manifest['audit_windows_manifest']]
    comparisons={}
    for left,right in [('trained_retained','trained_interpolated'),
            ('trained_retained','phase5_dense'),('trained_interpolated','phase5_dense'),
            ('trained_retained','initial_retained'),('trained_interpolated','initial_interpolated')]:
        d=np.asarray(audit[left]['per_window_delta_nll'])-np.asarray(audit[right]['per_window_delta_nll'])
        comparisons[left+'_minus_'+right]={'delta_nll':float(d.mean()),'relative_perplexity':float(np.exp(d.mean())),
            'document_bootstrap_95':cluster_ci(d,docs)}
    interactions={}
    for kind in ARMS:
        d=np.asarray(audit['trained_'+kind]['per_window_delta_nll'])
        for s in [4,12]:d=d-np.asarray(audit[f'trained_{kind}_only{s}']['per_window_delta_nll'])
        interactions[kind]={'delta_nll':float(d.mean()),'document_bootstrap_95':cluster_ci(d,docs)}
    long=[]
    for row in read('long_context.json'):
        label=row['configuration']['id'];m=row['metrics']
        long.append({'id':label,'short_context_ppl_ratio':audit[label]['perplexity_ratio_to_original'],
            'long_context_ppl_ratio':m['perplexity_ratio_to_original'],'long_context_teacher_kl':m['mean_kl'],
            'same_target_tokens':True})
    result={'phase':6,'audit_documents':32,'audit_target_tokens':8192,'decoder_blocks':28,
        'audit_source':'Fresh document-held-out official-train articles, excluding phase-2/3 training and phase-5 audit articles.',
        'primary_metrics':{label:{k:m[k] for k in ['delta_nll','perplexity_ratio_to_original','mean_kl','top1_agreement']} for label,m in audit.items()},
        'comparisons':comparisons,'interactions':interactions,'long_context':long,
        'selection':read('selection.json'),'checks':read('checks.json'),
        'native_serialization':read('native_serialization.json'),
        'global_curve_feasibility':read('global_curve_feasibility.json'),
        'exploratory_norm_control':[{k:r[k] for k in ['id','residual_scale','metrics']} for r in read('exploratory_norm_control.json')]}
    dump(OUT/'summary.json',result);return result,rows

def figures(summary,rows):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(15.6,4.9),gridspec_kw={'width_ratios':[.85,1.05,1.05]})
    initials=read('initial_validation.json');x=np.arange(2);width=.34
    for shift,scale,color,label in [(-width/2,1.,'#3c789b','Residual ×1'),(width/2,4/3,'#b9c8d8','Residual ×4/3')]:
        values=[next(r['metrics']['perplexity_ratio_to_original'] for r in initials if r['initialization']==k and abs(r['residual_scale']-scale)<1e-8) for k in ARMS]
        axes[0].bar(x+shift,values,width,color=color,label=label)
    axes[0].set_xticks(x,['Interpolated','Retained']);axes[0].set_ylim(0,2.)
    axes[0].axhline(1,color='#555',ls=':',lw=1);axes[0].set_ylabel('Perplexity / original')
    axes[0].set_title('Initial scaling · 32 validation windows',loc='left',fontweight='bold',fontsize=10)
    axes[0].legend(frameon=False,fontsize=8)
    for kind in ARMS:
        history=read(f'training_{kind}.json')['history']
        axes[1].plot([r['update'] for r in history],[r['validation']['mean_kl'] for r in history],
            '-o',color=COLORS['trained_'+kind],label=kind.title(),markersize=4)
    axes[1].set_title('Matched training · 8 validation windows',loc='left',fontweight='bold',fontsize=10)
    axes[1].set_xlabel('Optimizer updates');axes[1].set_ylabel('Teacher KL');axes[1].grid(alpha=.15)
    axes[1].legend(frameon=False,fontsize=8)
    indices={r['configuration']['id']:r['metrics'] for r in rows}
    for i,label in enumerate(LABELS):
        m=indices[label];ci=m['delta_nll_document_bootstrap_95'];value=m['perplexity_ratio_to_original']
        low,high=math.exp(ci['lower']),math.exp(ci['upper'])
        axes[2].errorbar(value,i,xerr=[[max(0,value-low)],[max(0,high-value)]],fmt='o',
            color=COLORS[label],capsize=3,markersize=6)
    axes[2].set_yticks(range(len(LABELS)),list(LABELS.values()),fontsize=8)
    axes[2].invert_yaxis();axes[2].axvline(1,color='#555',ls=':',lw=1)
    axes[2].set_xlabel('Perplexity / original · 95% document interval')
    axes[2].set_title('Fresh audit · all candidates have 28 blocks',loc='left',fontweight='bold',fontsize=10)
    axes[2].grid(axis='x',alpha=.15)
    fig.suptitle('SmolLM2-135M · Initialization and longer distillation',x=.04,ha='left',fontsize=17,fontweight='bold')
    fig.text(.04,.015,'1,800 updates per trained arm · identical batches · 460,800 input tokens per arm · 8,192 fresh audit targets · one seed',fontsize=9,color='#555')
    fig.tight_layout(rect=[.01,.06,1,.90]);fig.savefig(OUT/'phase6_results.png',dpi=160);fig.savefig(OUT/'phase6_results.svg');plt.close(fig)
    profiles=read('weight_profiles.json')['curves'];fig,axes=plt.subplots(2,2,figsize=(11,7.5),sharex='col')
    for col,s in enumerate(['4','12']):
        for kind in ['initial_interpolated','trained_interpolated','initial_retained','trained_retained']:
            row=profiles[kind][s];xs=[r['depth'] for r in row]
            style='--' if kind.startswith('initial') else '-'
            axes[0,col].plot(xs,[r['parameter_vector_norm'] for r in row],style,color=COLORS[kind],label=LABELS[kind],lw=1.7)
            axes[1,col].plot(xs,[r['departure_from_phase4_norm'] for r in row],style,color=COLORS[kind],lw=1.7)
        for ax in axes[:,col]:
            for j in range(int(s),int(s)+4):ax.axvline(j,color='#666',alpha=.2,ls=':')
            ax.grid(axis='y',alpha=.12);ax.set_xticks(range(int(s),int(s)+4))
        axes[0,col].set_title(f'Original blocks {s}–{int(s)+3}',loc='left',fontweight='bold')
        axes[1,col].set_xlabel('Continuous depth')
    axes[0,0].set_ylabel('Full parameter-vector norm');axes[1,0].set_ylabel('Distance from phase-4 curve')
    axes[0,0].legend(frameon=False,fontsize=8)
    fig.suptitle('Same original knots, different routes between them',x=.065,ha='left',fontsize=16,fontweight='bold')
    fig.text(.065,.013,'All nine parameter tensors enter these norms. Geometry alone does not measure language quality. Original knot values stay fixed.',fontsize=9,color='#555')
    fig.tight_layout(rect=[.02,.05,1,.93]);fig.savefig(OUT/'phase6_weight_curves.png',dpi=160);plt.close(fig)

def report(summary,rows):
    primary=summary['primary_metrics'];comparison=summary['comparisons']['trained_retained_minus_trained_interpolated']
    delta=comparison['delta_nll'];ci=comparison['document_bootstrap_95']
    retained=primary['trained_retained']['perplexity_ratio_to_original'];interpolated=primary['trained_interpolated']['perplexity_ratio_to_original']
    quality_rows=[]
    for row in rows:
        name=row['configuration']['id'];m=row['metrics'];interval=m['delta_nll_document_bootstrap_95']
        quality_rows.append(f'<tr><td>{html.escape(name)}</td><td>{row["decoder_blocks"]}</td><td>{m["perplexity_ratio_to_original"]:.4f}×</td><td>{math.exp(interval["lower"]):.4f}–{math.exp(interval["upper"]):.4f}</td><td>{m["mean_kl"]:.4f}</td><td>{m["top1_agreement"]:.1%}</td></tr>')
    comparisons=''.join(f'<tr><td>{name}</td><td>{row["delta_nll"]:+.5f}</td><td>{row["relative_perplexity"]:.5f}×</td><td>[{row["document_bootstrap_95"]["lower"]:+.5f}, {row["document_bootstrap_95"]["upper"]:+.5f}]</td></tr>' for name,row in summary['comparisons'].items())
    interactions=''.join(f'<li>{kind}: {row["delta_nll"]:+.5f} nats/token; document interval [{row["document_bootstrap_95"]["lower"]:+.5f}, {row["document_bootstrap_95"]["upper"]:+.5f}].</li>' for kind,row in summary['interactions'].items())
    longrows=''.join(f'<tr><td>{r["id"]}</td><td>{r["short_context_ppl_ratio"]:.4f}×</td><td>{r["long_context_ppl_ratio"]:.4f}×</td><td>{r["long_context_teacher_kl"]:.4f}</td></tr>' for r in summary['long_context'])
    timing=read('timing.json')
    timerows=''.join(f'<tr><td>{kind}</td><td>{v["native_blocks"]["median_paired_ratio"]:.3f}×</td><td>{v["streamed_curve"]["median_paired_ratio"]:.3f}×</td><td>{v["native_blocks"]["candidate_median_ms"]:.1f}</td></tr>' for kind,v in timing.items())
    checkrows=''.join(f'<tr><td>{kind}</td><td>{summary["checks"][kind]["all_sampled_weight_max_abs_difference"]:.3g}</td><td>{summary["checks"][kind]["curve_vs_native_max_logit_difference"]:.3g}</td><td>{summary["checks"][kind]["four_step_original_max_logit_difference"]:.1f}</td><td>{summary["native_serialization"][kind]["saved_model_reload_max_logit_difference"]:.1f}</td></tr>' for kind in ARMS)
    gates=summary['selection']['passed_gate'];passed=[k for k,v in gates.items() if v];pass_text=', '.join(passed) if passed else 'None'
    initialrows=''.join(f'<tr><td>{r["initialization"]}</td><td>{r["residual_scale"]:.4f}</td><td>{r["metrics"]["perplexity_ratio_to_original"]:.4f}×</td><td>{r["metrics"]["mean_kl"]:.4f}</td></tr>' for r in read('initial_validation.json'))
    normrows=''.join(f'<tr><td>{r["residual_scale"]:.4f}</td><td>{r["metrics"]["perplexity_ratio_to_original"]:.4f}×</td><td>{r["metrics"]["mean_kl"]:.4f}</td></tr>' for r in read('exploratory_norm_control.json'))
    historyrows=''.join(f'<tr><td>{kind}</td><td>{read("training_"+kind+".json")["best_update"]}</td><td>{read("training_"+kind+".json")["best_mean_validation_kl"]:.5f}</td></tr>' for kind in ARMS)
    samples=''.join(f'<div class="sample"><b>{html.escape(r["model"])}</b><p><i>{html.escape(r["prompt"])}</i>{html.escape(r["continuation"])}</p></div>' for r in read('generations.json'))
    im=lambda name:'data:image/png;base64,'+base64.b64encode((OUT/name).read_bytes()).decode()
    text=f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SmolLM2 — Phase 6</title><style>
    body{{font:16px/1.6 system-ui,sans-serif;color:#172536;background:#f5f7fa;margin:0}}main{{max-width:1120px;margin:auto;padding:42px 26px 72px}}h1{{font-size:36px;line-height:1.2}}h2{{margin-top:36px;font-size:23px}}.eyebrow{{letter-spacing:.12em;font-size:12px;color:#49627d}}.card,.sample{{padding:18px 23px;background:white;border:1px solid #dce4ee;border-radius:10px;margin:20px 0}}img{{width:100%;height:auto}}.table-wrap{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:12.5px;background:white}}td,th{{padding:9px 8px;border-bottom:1px solid #dde5ee;text-align:right}}td:first-child,th:first-child{{text-align:left}}th{{background:#eaf0f7}}code{{font-size:13px;background:#eaf0f7;padding:2px 5px}}.small{{font-size:13px;color:#536578}}pre{{white-space:pre-wrap;padding:16px;background:#eaf0f7;font-size:13px}}.sample p{{white-space:pre-wrap}}
    </style></head><body><main><div class="eyebrow">EXPERIMENT 06 · 17 OCTOBER 2026</div><h1>Initialization and longer distillation</h1>
    <p>Phase 5 found nearly identical quality for jointly trained curves and independent dense weights. Phase 6 therefore asks whether a different starting transformation, with more distillation, recovers the quality lost when reducing depth. This phase focuses on 28 blocks: two four-block regions each execute three blocks.</p>
    <div class="card"><b>Fresh audit.</b> Retained-native initialization reaches {retained:.4f}× original perplexity; interpolated initialization reaches {interpolated:.4f}×. The paired retained-minus-interpolated difference is {delta:+.5f} nats/token, with a 95% document-bootstrap interval [{ci['lower']:+.5f}, {ci['upper']:+.5f}]. Full-validation fidelity-gate passes: {pass_text}.</div>
    <img src="{im('phase6_results.png')}" alt="Initial scaling controls, matched training curves and fresh audit confidence intervals">
    <h2>The matched comparison</h2><p>Both arms train six ordinary decoder blocks with unit residual strength. The interpolated arm starts from the phase-4 Hermite samples at local coordinates 1/6, 1.5 and 17/6. The retained arm starts from original offsets [0,2,3] in each region: blocks [4,6,7] and [12,14,15], all zero indexed. Thus original blocks 5 and 13 are omitted during the reduced-depth forward pass. The retained policy was fixed before results: preserve endpoints and round the middle upward. We did not search for the best blocks to remove.</p>
    <p>Only these six cloned or reconstructed blocks train: 21,240,576 values in either arm. The other 22 blocks, embeddings, head and final norm remain frozen; gradients propagate through the complete model. The original 30-block parameter tensors remain untouched so the curve can retain its original knots. The interpolated starting point inherits earlier phase-4 slope training; it is not an untrained control.</p>
    <p>Each arm receives 1,800 identical ordered minibatches, two sequences per batch, context 128: 460,800 input tokens drawn from the same 1,024-window pool (131,072 input tokens). Phase 5 supplied 300 updates to its independent three-step dense weights, so this is six times the exposure for this resolution. The objective is full-distribution teacher-to-student next-token KL at temperature one. AdamW uses peak LR 0.0001, zero weight decay, global gradient clipping at one, 60 warmup updates and cosine decay to 20% of peak. There is one seed and no learning-rate search.</p>
    <p>Checkpoints minimize teacher KL on the first eight validation windows, checked at 0, 300, …, 1,800 updates. The full 32-window gate requires both ΔNLL ≤ 0.05 and KL ≤ 0.05. Both complete training runs are retained in the logs, and the released weight files are the selected checkpoints.</p>
    <table><tr><th>Arm</th><th>Selected update</th><th>Selection KL</th></tr>{historyrows}</table>
    <h2>Residual strength is a separate factor</h2><p>Before training, we evaluated unit scaling and the earlier 4/3 multiplier on all 32 validation windows. Multipliers are folded into attention output and MLP down-projection matrices. The multiplier helps interpolation but harms the chosen retained-block initialization in these initial measurements. Both trained arms nevertheless use the same prespecified unit convention. This is an initialization comparison under that convention, not a search for each arm's optimal residual scale.</p>
    <table><tr><th>Initialization</th><th>Residual multiplier</th><th>Validation PPL ratio</th><th>Teacher KL</th></tr>{initialrows}</table>
    <p>Comparisons with the phase-5 dense checkpoint combine more updates with a changed residual convention, schedule and training sequence. They do not isolate training duration. The matched phase-6 arms differ in initialization, with all subsequent settings held equal.</p>
    <h2>Exploratory matrix-norm control</h2><p>After the first 300-update checkpoints, we added a validation-only diagnostic. For each sampled projection matrix, restore its Frobenius norm to the square root of the linearly interpolated squared norms of the two adjacent original matrices. Leave both RMSNorm vectors unchanged. This checks a specific scale effect; it does not change either primary training run or the audit candidates.</p>
    <table><tr><th>Residual multiplier after matrix normalization</th><th>Validation PPL ratio</th><th>Teacher KL</th></tr>{normrows}</table>
    <p>At unit residual strength, norm restoration improves initial validation perplexity from 1.3541× to 1.2746×, close to the earlier 4/3 residual multiplier alone (1.2767×). Combining both raises the ratio to 1.4928×. This makes scale handling a concrete follow-up hypothesis. No normalized arm was trained in this phase, and these controls have no fresh-audit score.</p>
    <h2>New document-held-out audit</h2><p>The audit reserves 32 further WikiText-2 articles from the official <b>train</b> partition. It excludes every article sampled for phase-2/3 training (later training is a subset) and all 32 phase-5 audit articles. From each selected article we take two nonoverlapping 257-token spans. The short-context audit exposes the final 129 tokens and scores 128 predictions per span: 8,192 targets. Selection was frozen before training; no audit metric chose weights, residual scaling, pruning offsets or basis functions.</p>
    <p class="small">This is an experimental held-out partition, not the official benchmark test split. It does not establish absence from the reference model's original pretraining. All candidates below are scored on this same fresh audit, including the phase-5 reference; their numbers should not be directly compared with the previous audit's numbers.</p>
    <div class="table-wrap"><table><tr><th>Candidate</th><th>Blocks</th><th>PPL / original</th><th>95% interval</th><th>Teacher KL</th><th>Top-1 agreement</th></tr>{''.join(quality_rows)}</table></div>
    <p class="small">PPL ratio = exp(candidate NLL − original NLL). Intervals use 2,000 paired resamples of the 32 document clusters. They are pointwise and conditional on the fitted checkpoint and article sample; they do not capture training-seed uncertainty. Per-window rows in the JSON support all comparisons.</p>
    <div class="table-wrap"><table><tr><th>Left minus right</th><th>ΔNLL</th><th>Relative PPL</th><th>95% ΔNLL interval</th></tr>{comparisons}</table></div>
    <h2>Can the improved models still be curves?</h2><p>Yes: the three-sample derivative matrix A₃ has rank three. Define cardinal functions <code>l(t)=a(t) A₃⁺</code>, where a(t) is the four-component Hermite derivative basis. At the three fitted sample locations these functions form the identity matrix, and at all four original knots they vanish. For each tensor, add <code>Σ lᵢ(t) [Wᵢ − C₀(tᵢ)]</code> to the phase-4 base curve C₀. This interpolates arbitrary learned three-block weights while retaining every original knot.</p>
    <p>Phase 5 needed five sample points to cover both two-step and three-step schedules. This phase fits only three-step samples, so three cardinal components suffice. The public loader exposes only the trained three-step schedule and the original four-step schedule. Quality at two steps or other unseen schedules has not been established. This conversion uses the same 21,240,576 coefficient values as the six fitted blocks, plus the retained reference representation; it is a functionality construction without a storage-compression claim.</p>
    <table><tr><th>Arm</th><th>Max weight error</th><th>Max logit error</th><th>Original-knot logit error</th><th>Saved-model reload error</th></tr>{checkrows}</table>
    <p class="small">Sampled tensor checks cover every parameter in both windows. Dense-versus-curve logits cover 1,024 audit targets; original-knot checks use 128 input tokens. Cache checks use 32 tokens. Independent ordinary-model serialization and public-loader checks use 32 tokens. Logit tolerance is 0.001; sampled-weight tolerance is 0.000001. Exact-zero checks and original hashes are required where reported.</p>
    <img src="{im('phase6_weight_curves.png')}" alt="Weight-vector norms and distance from the previous curve, with fixed original knots">
    <p class="small">These plots use all nine parameter tensors per block. They visualize weight-space geometry, not an activation trajectory. Arc-length and second-derivative diagnostics are in weight_profiles.json; their relationship to useful information is not assumed.</p>
    <h2>Interaction between the two regions</h2><p>We run each trained region alone with the other region restored to original weights. Interaction is ΔNLL(both) − ΔNLL(region 4 only) − ΔNLL(region 12 only). Negative values indicate compensation relative to adding the two individual loss increases; positive values indicate amplification. These are ablations of fitted models, not a randomized test that joint training caused the interaction.</p><ul>{interactions}</ul>
    <h2>Longer context on the same targets</h2><p>Expose the preceding 128 tokens from each reserved span, then score exactly the same final 128 targets. This tests context 128 versus 256, not long-context deployment generally.</p>
    <table><tr><th>Candidate</th><th>Context 128 ratio</th><th>Context 256 ratio</th><th>Context 256 KL</th></tr>{longrows}</table>
    <h2>Runtime</h2><p>Both training workers ran concurrently with four CPU threads each; elapsed training times are not a controlled speed comparison. Inference measurements run after training completes. Native models contain 28 ordinary decoder blocks. Streamed curves construct the sampled weights during each forward pass.</p>
    <table><tr><th>Initialization</th><th>Native latency / original</th><th>Curve latency / original</th><th>Native prefill ms</th></tr>{timerows}</table>
    <p class="small">CPU FP32, four threads, batch one, context 128; 12 alternating paired measurements after two warmup pairs. Construction and curve fitting are excluded. These are prefill latencies, not decoding throughput or measurements on the user's laptop.</p>
    <h2>Fixed-prompt continuations</h2><p class="small">Two fixed prompts, up to 40 greedy tokens each, no sampling or selection. Illustrations of behavior, not benchmark scores.</p>{samples}
    <h2>Scope and next decision</h2><p>This experiment compares two concrete initialization policies at a single reduced depth under one residual convention and training seed. It tests whether each selected model can be expressed as a continuous curve while preserving the original discrete model. It does not reproduce GSA's input-conditioned coefficients, hierarchical recurrent state, expert bundling or from-scratch training.</p>
    <p>The next priority is whole-model adaptation at 28 blocks: clone and train every retained decoder block from the stronger phase-6 checkpoint, keeping the original 30-block model intact, then fit a global depth curve. That would directly test whether the frozen remainder limits recovery. Norm-restored interpolation remains a useful initialization control; its validation-only diagnostic does not predict its trained outcome. Further depth reduction needs its own evaluation and another reserved audit.</p>
    <h2>Whole-model feasibility: planning only</h2><p>A global 30-knot Hermite construction can retain the original model while fitting a separately trained 28-block model. The uniform 28-sample grid has no collisions with the original knots; its 28×30 derivative sampling matrix has rank 28 and condition number 6.92. Thus the same cardinal construction can represent arbitrary weights at all 28 fitted block positions. This is an algebraic feasibility result, not an evaluated whole-model curve.</p>
    <p>Training all 28 decoder blocks would expose 99,122,688 trainable values. Their FP32 weights, gradients and two Adam moments, plus all original model weights, total approximately 1.98 GiB. Activations, teacher caches, temporary buffers and framework overhead are additional; this is a lower bound, not a measured peak-memory claim.</p>
    <p>The same uniform 26-sample grid hits original knots 7 and 22, leaving rank 24 for corrections. A depth-only curve cannot both preserve those knot values and assign different weights at those identical coordinates. A 26-block follow-up must keep those samples fixed or change the grid or parameterization. The tested 28-block grid avoids this issue.</p>
    <h2>Related work and untested mechanisms</h2><p><a href="https://aclanthology.org/2025.findings-acl.1035/">ShortGPT (Men et al., ACL Findings 2025)</a> ranks layers using input/output activation similarity and treats generation with a separate strategy. Our inference is that calibration-based block selection is worth comparing with the fixed retention policy used here. This phase does not reproduce that method.</p>
    <p><a href="https://arxiv.org/abs/2209.04836v6">Git Re-Basin (Ainsworth, Hayase and Srinivasa, ICLR 2023)</a> uses function-preserving neuron permutations to align independently trained models for weight merging. Applying alignment to adjacent-layer curve construction is a separate hypothesis; these results do not identify misalignment as the cause of interpolation's losses. Both primary sources were checked on 17 October 2026.</p>
    <h2>Run and load</h2><pre>python -m venv venv
venv/bin/pip install -r requirements.txt
venv/bin/python download_model.py
venv/bin/python prepare.py
venv/bin/python train.py
venv/bin/python norm_control.py
venv/bin/python global_curve_feasibility.py
venv/bin/python evaluate_phase6.py
venv/bin/python verify_export.py
venv/bin/python build_report.py

from models import load_candidate
native = load_candidate('retained', as_curve=False, steps=3)
curve = load_candidate('retained', as_curve=True, steps=3)
original = load_candidate('retained', as_curve=True, steps=4)</pre>
    <p class="small">The loader accepts either initialization. The bundle includes selected blocks, phase-4 curve initialization, phase-5 reference blocks, prepared tokens, code, logs and results. Download the pinned original model or set SMOLLM_MODEL_DIR. Teacher caches regenerate and are excluded from the bundle. Default training is sequential; --arm interpolated and --arm retained can run in separate processes. Completed records skip retraining; move their records/checkpoints aside to run anew. Start a fresh KV cache when changing depth. The curve prototype requires this loader; native models support ordinary Transformers serialization. SHA256SUMS covers the archive's contents.</p>
    </main></body></html>'''
    (OUT/'phase6_report.html').write_text(text)

def package(rows):
    with (OUT/'audit_metrics.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['candidate','blocks','ppl_ratio','delta_nll','teacher_kl','top1_agreement','delta_nll_ci_lower','delta_nll_ci_upper'])
        for row in rows:
            m=row['metrics'];ci=m['delta_nll_document_bootstrap_95']
            writer.writerow([row['configuration']['id'],row['decoder_blocks'],m['perplexity_ratio_to_original'],m['delta_nll'],m['mean_kl'],m['top1_agreement'],ci['lower'],ci['upper']])
    (ROOT/'README.md').write_text('''# SmolLM2 initialization study — Phase 6

Open results/phase6_report.html for protocol, results, plots and limitations.

Two initializations, 1,800 matched updates each, N=3 only, 28 total decoder blocks. The retained policy keeps offsets [0,2,3] in original windows 4–7 and 12–15. Both primary arms have ordinary unit residual strength. Original 30-block weights remain immutable. The new audit excludes phase-5 audit documents.

Python 3.12, CPU FP32, four threads per worker. Install requirements, download_model.py (or set SMOLLM_MODEL_DIR), prepare.py, train.py, norm_control.py, global_curve_feasibility.py, evaluate_phase6.py, verify_export.py, build_report.py. Prepared tokens and selected checkpoints are included; the teacher caches and original pinned model are regenerated. train.py is sequential by default. Optional separate --arm interpolated / --arm retained workers have independent checkpoints and locks. Completed training records are skipped; move their records/checkpoints aside to retrain. Resume state is scratch-only and excluded from this archive.

from models import load_candidate
model = load_candidate('retained', as_curve=False, steps=3)
curve = load_candidate('retained', as_curve=True, steps=3)

Either initialization is supported. Curves expose the fitted N=3 or exact-original N=4 schedule. No N=2 quality claim. Curves are reconstructed deterministically from selected blocks plus reference/phase4_checkpoints; no separate curve checkpoint is needed. Original knots and the previous base curve add storage: coefficient count is not a compression ratio. Native models use ordinary Transformers serialization. Start a new KV cache whenever depth changes.

reference/phase5_dense_n3.safetensors contains the six reference native blocks compiled from the selected phase-5 dense model, with 4/3 residual scaling folded into output/down projections. export_phase5_reference.py records its provenance recipe. This reference is an evaluation control and never supplies phase-6 training weights.
''')
    paths=[p for p in ROOT.rglob('*') if p.is_file() and not any(part in ['cache','reference_model','__pycache__'] for part in p.relative_to(ROOT).parts) and p.name!='SHA256SUMS']
    (ROOT/'SHA256SUMS').write_text(''.join(hashlib.file_digest(p.open('rb'),'sha256').hexdigest()+'  '+str(p.relative_to(ROOT))+'\n' for p in sorted(paths)))
    paths.append(ROOT/'SHA256SUMS');archive=ROOT.parent/'smollm_initialization_phase6_2026-10-17.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(paths):z.write(p,p.relative_to(ROOT.parent))
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    print(json.dumps({'report':str(OUT/'phase6_report.html'),'archive':str(archive),'archive_bytes':archive.stat().st_size,'files':len(paths)}),flush=True)

def main():
    summary,rows=summarize();figures(summary,rows);report(summary,rows);package(rows)

if __name__=='__main__':main()
