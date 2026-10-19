"""Build the phase-5 report, figures and reproducible archive."""
import base64,csv,hashlib,html,json,math,zipfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import ROOT,OUT,ARMS,dump
from evaluation import cluster_ci
from joint_model import completed_basis,TIMES

LABELS={'initial':'Phase 4','raw_curve':'Original coordinates','orthogonal_curve':'Orthogonal coordinates',
        'dense':'Dense control','projected_dense':'Dense → four modes','lifted_dense':'Dense → five modes'}
COLORS={'initial':'#a3aab5','raw_curve':'#d98237','orthogonal_curve':'#1769b0','dense':'#7b4ab2',
        'projected_dense':'#ab684c','lifted_dense':'#27836b'}

def read(name):return json.loads((OUT/name).read_text())

def main():
    audit=read('audit.json');idx={r['configuration']['id']:r for r in audit};validation=read('validation.json')
    validx={r['configuration']['id']:r for r in validation};selection=read('selection.json');checks=read('checks.json')
    projection=read('projection.json');timing=read('timing.json');long=read('long_context.json')
    manifest=json.loads((ROOT/'data/dataset_manifest.json').read_text());docids=[r['document_index'] for r in manifest['audit_windows_manifest']]
    comparisons={}
    pairs=[('raw_curve','orthogonal_curve'),('orthogonal_curve','dense'),('projected_dense','dense'),('lifted_dense','dense')]
    for left,right in pairs:
        for n in [3,2]:
            a=idx[f'{left}_n{n}']['metrics'];b=idx[f'{right}_n{n}']['metrics']
            diff=np.asarray(a['per_window_delta_nll'])-np.asarray(b['per_window_delta_nll'])
            comparisons[f'{left}_minus_{right}_n{n}']={'delta_nll':float(diff.mean()),'relative_ppl_ratio':float(np.exp(diff.mean())),
                'document_bootstrap_95':cluster_ci(diff,docids)}
    interactions={}
    for kind in ['orthogonal_curve','dense']:
        both=idx[f'{kind}_n3']['metrics'];one=idx[f'{kind}_n3_only4']['metrics'];two=idx[f'{kind}_n3_only12']['metrics']
        diff=np.asarray(both['per_window_delta_nll'])-np.asarray(one['per_window_delta_nll'])-np.asarray(two['per_window_delta_nll'])
        interactions[kind]={'interaction_delta_nll':float(diff.mean()),'document_bootstrap_95':cluster_ci(diff,docids)}
    summary={'phase':5,'audit_tokens':8192,'audit_documents':32,'audit_source':'Previously unused official-train WikiText-2 articles, document-held out from phases 2–4 and phase-5 optimization/selection.',
        'primary_metrics':{f'{kind}_n{n}':{k:idx[f'{kind}_n{n}']['metrics'][k] for k in ['delta_nll','perplexity_ratio_to_original','mean_kl','top1_agreement']} for kind in LABELS for n in [3,2]},
        'validation_gates':selection['passed_gate'],'comparisons':comparisons,'interactions':interactions,
        'dense_increment_energy_outside_four_mode_space':projection['total_discarded_energy_fraction'],
        'dense_to_five_mode_curve_checks':checks['dense_to_curve_lift'],'exact_native_four_step_checks':checks['exact_four_step_logit_differences'],
        'native_serialization':read('native_serialization.json')}
    dump(OUT/'summary.json',summary)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,3,figsize=(15.5,4.8),gridspec_kw={'width_ratios':[1.08,1.1,1]})
    for kind in ARMS:
        history=read(f'training_{kind}.json')['history']
        axs[0].plot([r['update'] for r in history],[r['validation']['mean_kl'] for r in history],'-o',label=LABELS[kind],color=COLORS[kind],markersize=3)
    axs[0].set_title('Checkpoint validation · 8 windows',loc='left',fontweight='bold');axs[0].set_xlabel('Optimizer updates');axs[0].set_ylabel('Validation teacher KL (3/2-step mean)');axs[0].legend(frameon=False,fontsize=8);axs[0].grid(alpha=.16)
    kinds=['initial','raw_curve','orthogonal_curve','dense'];x=np.arange(len(kinds));width=.35
    for offset,n,alpha in [(-width/2,3,1),(width/2,2,.5)]:
        vals=[idx[f'{k}_n{n}']['metrics']['perplexity_ratio_to_original'] for k in kinds]
        axs[1].bar(x+offset,vals,width,color=[COLORS[k] for k in kinds],alpha=alpha,label=f'{n} steps per window')
    axs[1].set_xticks(x,['Phase 4','Raw\ncurve','Orthogonal\ncurve','Dense']);axs[1].set_ylim(0,max(idx[f'{k}_n2']['metrics']['perplexity_ratio_to_original'] for k in kinds)*1.15)
    axs[1].axhline(1,ls=':',color='#555555',lw=1);axs[1].set_title('Fresh document-held-out audit',loc='left',fontweight='bold');axs[1].set_ylabel('Perplexity / original');axs[1].legend(frameon=False,fontsize=8)
    kinds2=['dense','projected_dense','lifted_dense']
    for k in kinds2:
        vals=[idx[f'{k}_n{n}']['metrics']['perplexity_ratio_to_original'] for n in [3,2]]
        style={'linestyle':'--','marker':'s','markerfacecolor':'none','markersize':7} if k=='lifted_dense' else {'linestyle':'-','marker':'o','markersize':5}
        axs[2].plot([3,2],vals,color=COLORS[k],label=LABELS[k],lw=2,**style)
    axs[2].set_xticks([3,2]);axs[2].set_xlim(3.1,1.9);axs[2].set_title('What the curve constraint costs',loc='left',fontweight='bold');axs[2].set_xlabel('Evaluations per four-block window');axs[2].set_ylabel('Perplexity / original');axs[2].legend(frameon=False,fontsize=8);axs[2].grid(alpha=.16)
    fig.suptitle('SmolLM2-135M · Joint training and a complete curve basis',x=.04,ha='left',fontsize=17,fontweight='bold')
    fig.text(.04,.015,'600 updates per trained arm · 8,192 audit target tokens · 32 reserved articles · one seed · lower ratios are better',fontsize=9,color='#555555')
    fig.tight_layout(rect=[.02,.06,.99,.91]);fig.savefig(OUT/'phase5_results.png',dpi=160);fig.savefig(OUT/'phase5_results.svg');plt.close(fig)
    profiles=read('weight_profiles.json')['curves'];fig,axs=plt.subplots(1,3,figsize=(15.5,4.2))
    for ax,s in zip(axs[:2],['4','12']):
        for k in ['initial','orthogonal_curve','lifted_dense']:
            p=profiles[k][s];ax.plot([x['depth'] for x in p],[x['parameter_vector_norm'] for x in p],color=COLORS[k],label=LABELS[k],lw=1.8)
        for j in range(int(s),int(s)+4):ax.axvline(j,color='#555555',alpha=.2,ls=':')
        ax.set_xticks(range(int(s),int(s)+4));ax.set_title(f'Blocks {s}–{int(s)+3}',loc='left',fontweight='bold');ax.set_xlabel('Continuous depth');ax.set_ylabel('Full parameter-vector norm')
    axs[0].legend(frameon=False,fontsize=8)
    ts=np.linspace(0,3,301);axs[2].plot(ts,[completed_basis(t)[-1] for t in ts],color=COLORS['lifted_dense'],lw=2)
    axs[2].scatter(range(4),[0]*4,color='#343d4a',marker='s',s=30,label='Original knots',zorder=4)
    axs[2].scatter(TIMES[:3],[completed_basis(t)[-1] for t in TIMES[:3]],color=COLORS['orthogonal_curve'],s=32,label='3-step samples',zorder=4)
    axs[2].scatter(TIMES[3:],[completed_basis(t)[-1] for t in TIMES[3:]],color=COLORS['dense'],marker='^',s=38,label='2-step samples',zorder=4)
    axs[2].axhline(0,color='#777777',lw=.8,ls=':');axs[2].set_xticks(range(4));axs[2].set_xlabel('Local depth coordinate');axs[2].set_ylabel('Basis amplitude')
    axs[2].set_title('The added mode vanishes at every knot',loc='left',fontweight='bold',fontsize=10);axs[2].legend(frameon=False,fontsize=8)
    fig.suptitle('Different curves, identical original knots',x=.045,ha='left',fontsize=15)
    fig.text(.045,.01,'Left: exact norms of all nine parameter tensors, a scalar projection of each weight curve. Right: the added fifth basis function.',fontsize=9,color='#555555')
    fig.tight_layout(rect=[0,.05,1,.9]);fig.savefig(OUT/'phase5_weight_curves.png',dpi=160);plt.close(fig)
    with (OUT/'audit_metrics.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['candidate','decoder_blocks','ppl_ratio','delta_nll','kl','top1_agreement','delta_nll_ci_low','delta_nll_ci_high'])
        for r in audit:
            m=r['metrics'];ci=m['delta_nll_document_bootstrap_95']
            w.writerow([r['configuration']['id'],r['decoder_blocks'],m['perplexity_ratio_to_original'],m['delta_nll'],m['mean_kl'],m['top1_agreement'],ci['lower'],ci['upper']])
    build_html(summary,audit,long,timing,checks)
    package()


def build_html(summary,audit,long,timing,checks):
    primary=summary['primary_metrics'];gate=summary['validation_gates'];train=[read(f'training_{k}.json') for k in ARMS]
    rows=[]
    for r in audit:
        c=r['configuration'];m=r['metrics'];ci=m['delta_nll_document_bootstrap_95']
        rows.append(f'<tr><td>{html.escape(c["id"])}</td><td>{r["decoder_blocks"]}</td><td>{m["perplexity_ratio_to_original"]:.4f}×</td><td>{math.exp(ci["lower"]):.4f}–{math.exp(ci["upper"]):.4f}</td><td>{m["mean_kl"]:.4f}</td><td>{m["top1_agreement"]:.1%}</td></tr>')
    timerows=''.join(f'<tr><td>{name}</td><td>{v["streamed"]["median_paired_ratio"]:.3f}×</td><td>{v["native_export"]["median_paired_ratio"]:.3f}×</td><td>{v["native_export"]["candidate_median_ms"]:.1f}</td></tr>' for name,v in timing.items())
    longrows=''.join(f'<tr><td>{r["configuration"]["id"]}</td><td>{primary[r["configuration"]["id"]]["perplexity_ratio_to_original"]:.4f}×</td><td>{r["metrics"]["perplexity_ratio_to_original"]:.4f}×</td><td>{r["metrics"]["mean_kl"]:.4f}</td></tr>' for r in long)
    comparisons=''.join(f'<tr><td>{name}</td><td>{r["delta_nll"]:+.5f}</td><td>[{r["document_bootstrap_95"]["lower"]:+.5f}, {r["document_bootstrap_95"]["upper"]:+.5f}]</td></tr>' for name,r in summary['comparisons'].items())
    interactions=''.join(f'<li>{name}: {r["interaction_delta_nll"]:+.5f} nats/token, 95% document interval [{r["document_bootstrap_95"]["lower"]:+.5f}, {r["document_bootstrap_95"]["upper"]:+.5f}].</li>' for name,r in summary['interactions'].items())
    samples=''.join(f'<div class="sample"><b>{html.escape(r["model"])}</b><p><i>{html.escape(r["prompt"])}</i>{html.escape(r["continuation"])}</p></div>' for r in read('generations.json'))
    im=lambda f:'data:image/png;base64,'+base64.b64encode((OUT/f).read_bytes()).decode()
    passed=[k for k,v in gate.items() if v];gate_text=', '.join(passed) if passed else 'None'
    maxlift=max(v['dense_vs_lifted_max_logit_difference'] for v in checks['dense_to_curve_lift'].values())
    orth=primary['orthogonal_curve_n3']['perplexity_ratio_to_original'];dense=primary['dense_n3']['perplexity_ratio_to_original'];projection=summary['dense_increment_energy_outside_four_mode_space']
    content=f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SmolLM2 — Phase 5</title><style>
    body{{font:16px/1.6 system-ui,sans-serif;color:#172536;background:#f5f7fa;margin:0}}main{{max-width:1120px;margin:auto;padding:44px 28px 70px}}h1{{font-size:36px;line-height:1.2}}h2{{margin-top:36px;font-size:24px}}p{{max-width:1000px}}.eyebrow{{letter-spacing:.12em;font-size:12px;color:#49627d}}.card,.sample{{padding:18px 23px;background:white;border:1px solid #dce4ee;border-radius:10px;margin:20px 0}}img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:12.5px;background:white}}td,th{{padding:9px 8px;border-bottom:1px solid #dde5ee;text-align:right}}td:first-child,th:first-child{{text-align:left}}th{{background:#eaf0f7}}code{{font-size:13px;background:#eaf0f7;padding:2px 5px}}.small{{font-size:13px;color:#536578}}pre{{white-space:pre-wrap;padding:16px;background:#eaf0f7;font-size:13px}}.sample p{{white-space:pre-wrap}}
    </style></head><body><main><div class="eyebrow">EXPERIMENT 05 · 17 OCTOBER 2026</div><h1>Joint layer curves<br>Optimization, constraints, and exact conversion</h1>
    <p>Both windows (blocks 4–7 and 12–15, zero indexed) now train together through the complete frozen remainder of SmolLM2-135M. Three matched arms test original Hermite coordinates, an orthogonal reparameterization of the same curve space, and independent dense weights for each reduced resolution.</p>
    <div class="card"><b>Measured result.</b> At three steps per window, the orthogonal curve has {orth:.3f}× original perplexity and the dense control has {dense:.3f}× on the fresh audit. Validation-gate passes: {gate_text}. A deterministic five-mode curve reconstructed from the dense control matches it to within {maxlift:.3g} maximum logit difference on the 1,024-token equivalence check, while four-step outputs remain exactly native.</div>
    <img src="{im('phase5_results.png')}" alt="Matched training, fresh audit and projection controls">
    <h2>The useful algebraic distinction</h2><p>With original weight knots fixed, a four-knot Hermite curve has four learned derivative tensors per family. At the three evaluation points used by N=3, the derivative sampling matrix has rank three: it can represent arbitrary weights at those three points. The restriction appears when the same four derivative tensors must serve all five points used by N=3 and N=2 together. That combined sampling matrix has rank four, leaving one affine constraint per weight coordinate.</p>
    <p>The orthogonal arm changes coordinates, not expressivity: from the SVD of the 5×4 sampling matrix A, choose B so that AB has orthonormal columns. It begins with the exact phase-4 function. The dense control begins with precisely the same sampled weights, but can update all five samples independently. Its two resolutions have separate Parameters and optimizer state, so one cannot drift on the other's update.</p>
    <h2>A curve that exactly contains the dense control</h2><p>We added one basis function that vanishes at all four original knots. Start with the polynomial bubble <code>q(t)=∏(t−j)², j=0…3</code>, subtract its component in the four-mode sample space, and normalize the remainder. The resulting fifth mode spans the missing direction. The five modes form an orthogonal basis at the five tested reduced-depth sample points.</p>
    <p>After dense training and validation selection, multiplying its five learned increments by this basis transpose converts them into a continuous curve with no additional optimization. The interpolation algebra reproduces the dense weights at both reduced resolutions and preserves the original model at four steps. The measured FP32 tensor/logit differences are recorded in <code>checks.json</code>. This is a functionality construction with the same 35,400,960 trained degrees of freedom as the dense pair, not a parameter-compression result. Other sampling schedules and input-conditioned depth remain untested.</p>
    <p>The four-mode projection control takes the same selected dense solution and discards its component outside the original curve space. It removes {projection:.2%} of the learned increment energy. Its quality change measures the immediate effect of that projection on this solution; it is not a lower bound on the best retrained four-mode model.</p>
    <img src="{im('phase5_weight_curves.png')}" alt="Full-dimensional weight norm curves with original knots preserved">
    <p class="small">The weight norms nearly overlap at this scale. Equal or similar norms do not imply equal weights or equal language-model behavior.</p>
    <h2>Matched training protocol</h2><p>Each trained arm received 600 ordered minibatches of two 128-token sequences, alternating N=3 and N=2: 153,600 input tokens per arm, 76,800 per resolution. Both converted windows optimize jointly with next-token teacher KL at temperature one. The other 22 blocks, embeddings, output head and final norm remain frozen. Training draws from 1,024 phase-3 windows (131,072 input tokens). This is continuation from phase-4 weights, not training from scratch.</p>
    <p>AdamW uses zero weight decay, gradient clipping, 20 warmup updates and cosine decay to 20% of peak. Training-only one-step probes calibrate each arm's average initial sampled-weight RMS update to the dense control at LR 0.0001; those probe updates are discarded. This removes a large initial step-scale difference but does not make Adam coordinate invariant. Checkpoints minimize the equal mean of N=3/N=2 KL on the first eight validation windows, checked every 100 updates. Full validation uses all 32 windows with the existing ΔNLL ≤ 0.05 and KL ≤ 0.05 gate.</p>
    <p>The raw and orthogonal arms each train 28,320,768 coefficients. The dense control trains 35,400,960 across both resolutions, 25% more total, with only 21,240,576 active at N=3 or 14,160,384 at N=2. Data exposure and updates are matched; parameter storage and wall time are not. Dense training overlaps the curve worker on a separate four-thread process. Inference timing runs after all training workers finish. Comparing phase 4 with phase 5 combines joint training, more data, a different objective and more updates; it does not isolate the causal effect of joint training alone.</p>
    <h2>Fresh document-held-out audit</h2><p>We reserved two nonoverlapping 257-token spans from each of 32 articles unused in phase-2 and phase-3 training; phase 4 used a subset of phase 3. These articles come from the official WikiText-2 <b>train</b> partition and are held out at document level for this experiment. This is not the official benchmark test split and does not establish absence from the reference model's pretraining. No audit result selected weights, depths, learning rates or basis functions. Main scoring uses the last 129 tokens of each span: 8,192 predicted tokens at context 128.</p>
    <table><thead><tr><th>Candidate</th><th>Blocks</th><th>Audit PPL ratio</th><th>95% interval</th><th>Teacher KL</th><th>Top-1</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
    <p class="small">Ratios are exp(candidate NLL − native NLL), with 1 representing the reference. Paired document bootstrap uses 2,000 resamples of 32 articles. Intervals are pointwise, conditional on these checkpoints and articles; they do not capture seed uncertainty. Individual-window rows are diagnostic ablations of jointly trained weights.</p>
    <table><thead><tr><th>Paired comparison (left minus right)</th><th>ΔNLL</th><th>95% document interval</th></tr></thead><tbody>{comparisons}</tbody></table>
    <h2>Relationships between windows</h2><p>For each jointly trained candidate, we also execute one converted window at a time. The interaction is ΔNLL(both) − ΔNLL(window 4 only) − ΔNLL(window 12 only). Negative values indicate compensation relative to the additive loss prediction; positive values indicate amplification. These counterfactual ablations probe the fitted model, not a separate randomized test of joint versus independent training.</p><ul>{interactions}</ul>
    <h2>Longer context on identical target tokens</h2><p>These runs expose the extra 128 preceding tokens from each reserved span and score the same final 128 targets. This is a 128→256 context check, not evidence about long-context deployment in general.</p>
    <table><thead><tr><th>Candidate</th><th>Context 128 ratio</th><th>Context 256 ratio</th><th>Context 256 KL</th></tr></thead><tbody>{longrows}</tbody></table>
    <h2>Export, caches and cost</h2><p>Four-step curves preserve native logits exactly. DynamicCache correctness, dense-to-curve reconstruction, checkpoint reloads and frozen-original hashes are checked. A fixed-resolution native export samples the weights once and folds each residual multiplier into the attention output and MLP down-projection matrices. Its forward pass contains ordinary decoder blocks and no curve sampling. This export gives up runtime resolution changes until recompiled. The public five-mode loader was also exercised through an ordinary safetensors save and Transformers reload: the 28-block model reproduced all checked logits exactly (32 input tokens).</p>
    <table><thead><tr><th>Candidate</th><th>Streamed latency/native</th><th>Native export latency/native</th><th>Export prefill ms</th></tr></thead><tbody>{timerows}</tbody></table>
    <p class="small">12 alternating paired CPU measurements after warmup; FP32, four threads, batch 1, context 128. Native export construction is outside timing. These are prefill measurements, not decoding throughput. The research bundle retains the original model recipe, phase-4 initialization and learned increments; trainable-parameter counts are not compressed archive sizes.</p>
    <h2>Illustrative greedy continuations</h2><p class="small">Fixed prompts, 40 new tokens each, no sampling or cherry-picking. These are illustrations, not benchmark scores.</p>{samples}
    <h2>What this supports</h2><p>The difference between original and orthogonal coordinates measures optimization sensitivity within the same curve space. The dense projection measures a concrete restriction of the four-mode basis. The five-mode lift demonstrates that a continuous representation can exactly carry the fitted reduced-depth transformations at the tested resolutions while retaining the original knots. None of those facts by itself makes the reduced-depth model equivalent to the original 30-block model.</p>
    <p>The close curve/dense results make recovery of the reduced-depth model the next priority. A useful next comparison is retained-native-block initialization versus the present interpolated initialization, with a larger distillation budget. Any better dense variants can then be lifted into the complete curve without adding a representation error at the fitted resolutions. A separate set of depth sample points is needed to test interpolation beyond those resolutions; this experiment does not establish that generalization. This experiment does not implement the proposed GSA input conditioning, hierarchical state, expert bundling or from-scratch training.</p>
    <h2>Reproduce and load</h2><pre>python -m venv venv\nvenv/bin/pip install -r requirements.txt\nvenv/bin/python download_model.py\nvenv/bin/python prepare.py\nvenv/bin/python train.py\nvenv/bin/python run_evaluation.py\nvenv/bin/python verify_export.py\nvenv/bin/python build_report.py\n\nfrom load_candidate import load_candidate\nmodel = load_candidate('lifted_dense', steps=3, native_export=True)</pre><p class="small">The bundle includes prepared tokens, source, three selected checkpoints, phase-4 initialization, plots and machine-readable results. Teacher caches and the original pinned model are regenerated. Standard sequential training is the default; the parallel dense worker is optional. Remove completed training records/checkpoints to retrain. Source and artifact hashes are listed in SHA256SUMS.</p></main></body></html>'''
    (OUT/'phase5_report.html').write_text(content)


def package():
    (ROOT/'README.md').write_text('''# SmolLM2 joint curves — Phase 5\n\nOpen results/phase5_report.html. The report explains the matched protocol, fresh article-held-out audit, four-mode projection and five-mode exact dense-to-curve lift.\n\nPython 3.12, CPU FP32, four threads per worker. Install requirements, run download_model.py (or set SMOLLM_MODEL_DIR), prepare.py, train.py, run_evaluation.py, verify_export.py, build_report.py. Prepared data are included. Teacher caches are regenerated. Default training is sequential; train_dense_parallel.py is optional and must not overlap another dense worker.\n\n600 updates per arm, same recorded batches; one seed. The selected dense control and its five-mode lift have the same learned degrees of freedom, 25% more than the four-mode curve arms. No compression claim. Audit articles are reserved from the official train partition, not an official benchmark test split.\n\nload_candidate('orthogonal_curve', steps=3) returns the function-weight prototype. load_candidate('lifted_dense', steps=3, native_export=True) returns an ordinary native decoder with sampled weights and fixed residual scaling folded in. Supported resolutions are 2,3,4; the independent dense control supports 2,3. Start a new cache when changing resolution. The function-weight prototype needs this custom loader; native_export is suitable for ordinary model serialization.\n\nSelected checkpoints hold increments relative to reference/phase4_checkpoints. The four-mode projection and complete five-mode lift are regenerated deterministically from dense.safetensors. No separate projected/lifted checkpoint is necessary.\n\nInitial preparation calibration was superseded after a gradient-clearing correction; only calibration_verified.log and results/calibration.json supplied the frozen training learning rates. See reference/preparation_notes.json.\n''')
    paths=[p for p in ROOT.rglob('*') if p.is_file() and not any(x in p.relative_to(ROOT).parts for x in ['cache','reference_model','__pycache__']) and p.name!='SHA256SUMS']
    (ROOT/'SHA256SUMS').write_text(''.join(hashlib.file_digest(p.open('rb'),'sha256').hexdigest()+'  '+str(p.relative_to(ROOT))+'\n' for p in sorted(paths)))
    paths.append(ROOT/'SHA256SUMS');archive=ROOT.parent/'smollm_joint_curves_phase5_2026-10-17.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(paths):z.write(p,p.relative_to(ROOT.parent))
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    print(json.dumps({'report':str(OUT/'phase5_report.html'),'archive':str(archive),'archive_bytes':archive.stat().st_size,'files':len(paths)}),flush=True)


if __name__=='__main__':main()
