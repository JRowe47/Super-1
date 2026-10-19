"""Build standalone report and exact-data figures, then an integrity-checked ZIP."""
import base64,csv,hashlib,html,json,math,zipfile
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from common import ROOT,OUT,dump


def read(name):return json.loads((OUT/name).read_text())
def val(x,n=4):return f'{x:.{n}f}'


def main():
    test=read('test.json');validation=read('validation.json');selection=read('selection.json')
    byid={r['configuration']['id']:r for r in test};timing=read('timing.json')
    exact=read('exact_control.json');checks=read('cache_checks.json');manifest=json.loads((ROOT/'data/dataset_manifest.json').read_text())
    selected=[byid[r['selected']] for r in selection['per_window']];composite=byid['selected_composite']
    interaction=composite['metrics']['delta_nll']-sum(r['metrics']['delta_nll'] for r in selected)
    perwindow=np.array(composite['metrics']['per_window_delta_nll'])-sum(np.array(r['metrics']['per_window_delta_nll']) for r in selected)
    from evaluation import cluster_ci
    interaction_ci=cluster_ci(perwindow,[x['document_index'] for x in manifest['splits']['test']])
    summary={'phase':'4','exact_30_knot_max_logit_difference':exact['all_30_original_knots_max_logit_abs_difference'],
        'reduced_candidates_passing_validation_gate':[r['selected'] for r in selection['per_window'] if r['passed_gate']],
        'selected_per_window':[{'id':r['configuration']['id'],'steps':r['configuration']['windows'][0]['steps'],
            'test_ppl_ratio':r['metrics']['perplexity_ratio_to_original'],'test_kl':r['metrics']['mean_kl']} for r in selected],
        'composite':{'decoder_evaluations':composite['accounting']['decoder_evaluations'],
            'test_ppl_ratio':composite['metrics']['perplexity_ratio_to_original'],'test_kl':composite['metrics']['mean_kl'],
            'delta_nll_interaction':interaction,'interaction_document_bootstrap_95':interaction_ci},
        'all_cache_checks_passed':all(c['passed'] for c in checks.values()),
        'test_predicted_tokens':8192,'test_documents':len(set(x['document_index'] for x in manifest['splits']['test'])),
        'added_slope_parameters_per_window':14160384,'original_weights_retained':True}
    dump(OUT/'summary.json',summary)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,3,figsize=(15.5,4.6),gridspec_kw={'width_ratios':[1,1,1.12]})
    colors={'initial':'#a6adb8','trained':'#2066b0','drop':'#d97732'}
    for ax,start in zip(axs[:2],[4,12]):
        for kind,label in [('initial','Initial curve'),('trained','Trained curve'),('drop','Drop blocks')]:
            ys=[1]+[byid[f'w{start}_{kind}_n{n}']['metrics']['perplexity_ratio_to_original'] for n in [3,2]]
            ax.plot([4,3,2],ys,'o-',label=label,color=colors[kind],lw=2)
        ax.axhline(1,color='#333333',lw=.8,ls=':');ax.set_xticks([4,3,2]);ax.set_xlim(4.15,1.85)
        ax.set_xlabel('Evaluations replacing four blocks');ax.set_title(f'Blocks {start}–{start+3} (zero indexed)',loc='left',fontweight='bold')
        ax.set_ylabel('Test perplexity / original');ax.grid(axis='y',alpha=.18)
    axs[0].legend(frameon=False,fontsize=9)
    labels=['Original']+[f"{r['configuration']['windows'][0]['start']}–{r['configuration']['windows'][0]['start']+3}\n{r['configuration']['windows'][0]['steps']} steps" for r in selected]+['Both\nwindows']
    ratios=[1]+[r['metrics']['perplexity_ratio_to_original'] for r in selected]+[composite['metrics']['perplexity_ratio_to_original']]
    axs[2].bar(range(4),ratios,color=['#a6adb8','#2066b0','#3d8d89','#7546a1'],width=.65)
    for i,x in enumerate(ratios):axs[2].text(i,x+.02,f'{x:.3f}×',ha='center',fontsize=10)
    axs[2].set_xticks(range(4),labels);axs[2].set_ylim(0,max(ratios)*1.18)
    axs[2].set_title('Validation-selected composition',loc='left',fontweight='bold');axs[2].set_ylabel('Test perplexity / original')
    fig.suptitle('SmolLM2-135M · Exact knots, fewer decoder evaluations',x=.04,ha='left',fontsize=17,fontweight='bold')
    fig.text(.04,.015,'8,192 test tokens · 128-token context · one training seed · original weights retained plus learned slopes',fontsize=9,color='#555555')
    fig.tight_layout(rect=[.02,.06,.99,.91]);fig.savefig(OUT/'phase4_results.png',dpi=160);fig.savefig(OUT/'phase4_results.svg');plt.close(fig)
    diag=read('curve_diagnostics.json');fig,axs=plt.subplots(1,2,figsize=(11,3.9))
    for ax,s in zip(axs,[4,12]):
        p=diag[str(s)]['profile'];x=[r['depth'] for r in p];y=[r['weight_norm'] for r in p]
        ax.plot(x,y,color='#2066b0',lw=2);ax.set_title(f'Blocks {s}–{s+3}',loc='left',fontweight='bold')
        for j in range(s,s+4):ax.axvline(j,color='#777777',alpha=.25,ls=':')
        ax.set_xlabel('Continuous depth coordinate');ax.set_ylabel('Full parameter-vector norm');ax.set_xticks(range(s,s+4))
    fig.suptitle('Learned weight curves between immutable original knots',x=.06,ha='left',fontsize=14)
    fig.text(.06,.01,'Exact full-dimensional norms. This projection does not show all coordinate changes or establish functional smoothness.',fontsize=8,color='#555555')
    fig.tight_layout(rect=[0,.04,1,.9]);fig.savefig(OUT/'weight_curve_profiles.png',dpi=160);plt.close(fig)
    with (OUT/'test_metrics.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['candidate','decoder_evaluations','delta_nll','ppl_ratio','kl','top1_agreement','delta_nll_ci_low','delta_nll_ci_high'])
        for r in test:
            m=r['metrics'];ci=m['delta_nll_document_bootstrap_95']
            writer.writerow([r['configuration']['id'],r['accounting']['decoder_evaluations'],m['delta_nll'],m['perplexity_ratio_to_original'],m['mean_kl'],m['top1_agreement'],ci['lower'],ci['upper']])
    rows=[]
    for r in test:
        m=r['metrics'];ci=m['delta_nll_document_bootstrap_95'];name=r['configuration']['id']
        rows.append(f'<tr><td>{name}</td><td>{r["accounting"]["decoder_evaluations"]}</td><td>{m["perplexity_ratio_to_original"]:.4f}×</td><td>{math.exp(ci["lower"]):.4f}–{math.exp(ci["upper"]):.4f}</td><td>{m["mean_kl"]:.4f}</td><td>{m["top1_agreement"]:.1%}</td></tr>')
    pass_text='No reduced-depth curve passed both validation thresholds.' if not summary['reduced_candidates_passing_validation_gate'] else 'Passing reduced candidates: '+', '.join(summary['reduced_candidates_passing_validation_gate'])+'.'
    timingrows=''.join(f'<tr><td>{name}</td><td>{m["streamed"]["curve_median_ms"]:.1f}</td><td>{m["streamed"]["median_paired_ratio"]:.3f}×</td><td>{m["materialized"]["curve_median_ms"]:.1f}</td><td>{m["materialized"]["median_paired_ratio"]:.3f}×</td></tr>' for name,m in timing.items())
    im=lambda name:'data:image/png;base64,'+base64.b64encode((OUT/name).read_bytes()).decode()
    training=[read(f'training_window_{s}.json') for s in [4,12]]
    total_seconds=sum(t['elapsed_seconds'] for t in training)
    content=f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SmolLM2-135M — Joint Layer Curves, Phase 4</title><style>
    body{{font:16px/1.6 system-ui,sans-serif;color:#172536;background:#f5f7fa;margin:0}}main{{max-width:1100px;margin:auto;padding:44px 30px 70px}}h1{{font-size:36px;line-height:1.2}}h2{{margin-top:38px;font-size:24px}}p{{max-width:950px}}.eyebrow{{letter-spacing:.12em;font-size:12px;color:#49627d}}.card{{padding:20px 24px;background:white;border:1px solid #dce4ee;border-radius:10px;margin:22px 0}}img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:13px;background:white}}td,th{{padding:9px 10px;border-bottom:1px solid #dde5ee;text-align:right}}td:first-child,th:first-child{{text-align:left}}th{{background:#eaf0f7}}code{{font-size:13px;background:#eaf0f7;padding:2px 5px}}.small{{font-size:13px;color:#536578}}pre{{white-space:pre-wrap;padding:16px;background:#eaf0f7;font-size:13px}}a{{color:#2066b0}}
    </style></head><body><main><div class="eyebrow">EXPERIMENT 04 · 17 OCTOBER 2026</div><h1>SmolLM2-135M<br>Joint curves across layers</h1>
    <p>We moved from one layer to the entire 30-block decoder, then trained two four-block curves and evaluated each at four, three and two decoder steps. All original weights remain available, and four-step execution remains exactly equivalent to the native window.</p>
    <div class="card"><b>Measured result.</b> The all-30-knot representation has zero logit difference on the 256-token prefill check. {pass_text} The selected composite runs {summary['composite']['decoder_evaluations']} decoder blocks and has {summary['composite']['test_ppl_ratio']:.3f}× original test perplexity. These results concern this short-budget post-hoc conversion, not all possible continuous-depth architectures.</div>
    <img src="{im('phase4_results.png')}" alt="Measured quality versus decoder evaluations">
    <h2>What the representation does</h2><p>Each of nine homologous tensor families has a cubic Hermite curve across block depth. At integer depth we return the original frozen tensor directly. Between knots we learn four derivative tensors per window. The attention and MLP residual branches retain the native ordering and are scaled by 4/N when replacing four blocks with N evaluations. Evaluation coordinates are <code>(i+0.5)·4/N − 0.5</code>. N=4 lands exactly on the four original weights; N=3 and N=2 land between them.</p>
    <p>The full model keeps its embeddings, output head and final norm. This prototype adds 14,160,384 trained slope values per window: 56,641,536 bytes in FP32, on top of retained original weights. It is a functionality experiment, not parameter compression. The global representation is a family of tensor curves; a norm plot is only one projection of those curves.</p>
    <img src="{im('weight_curve_profiles.png')}" alt="Full-dimensional norms of learned weight curves">
    <h2>Training and evaluation</h2><p>Windows 4–7 and 12–15 were selected from the previous cross-layer survey (indices start at zero). Each received 300 local updates and 60 updates with gradients from the final next-token probabilities, alternating three and two evaluations. The local target includes intermediate teacher states, with linear interpolation for fractional boundaries; those interpolated states are a chosen training proxy. The final objective is teacher-to-student KL plus 0.1 times local path loss. All native parameters were frozen.</p>
    <p>Training used 256 WikiText-2 windows (32,768 input tokens), final validation 32 windows (4,096 tokens), and test 64 windows (8,192 predicted tokens across {summary['test_documents']} documents), all at context 128. The first eight validation windows selected checkpoints. The splits reuse phase-3 data: test was held out from this phase's optimization and selection, but is not a fresh benchmark. One seed was run. Total training time for both windows was {total_seconds/60:.1f} minutes on four CPU threads.</p>
    <p>The gate was fixed at validation ΔNLL ≤ 0.05 and KL ≤ 0.05. Per window we selected the fewest evaluations passing both, or the lowest-KL failed-gate representative if none passed. The composite combines those selections; test scores never chose its depths. Drop controls kept offsets [0,1,3] at three steps and [0,3] at two steps, with no residual rescaling.</p>
    <table><thead><tr><th>Candidate</th><th>Blocks</th><th>Test PPL ratio</th><th>95% interval</th><th>Teacher KL</th><th>Top-1 agreement</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
    <p class="small">PPL ratio = exp(candidate NLL − native NLL); lower is better, 1 is original. Intervals are paired document bootstrap, 2,000 resamples, pointwise and conditional on this corpus and fitted checkpoint. They do not capture training-seed uncertainty. Original test perplexity: {math.exp(test[0]['metrics']['reference_nll']):.3f}.</p>
    <h2>Interactions between windows</h2><p>The composite's ΔNLL minus the sum of its two constituent ΔNLLs is {interaction:+.4f} nats/token, with document-bootstrap 95% interval [{interaction_ci['lower']:+.4f}, {interaction_ci['upper']:+.4f}]. Negative values mean less degradation than the additive ΔNLL prediction; positive values mean more. This is a behavioral interaction measurement, not evidence of a shared low-dimensional weight basis or proof of interchangeability.</p>
    <h2>Cache correctness and measured cost</h2><p>All retained native attention blocks are reindexed after depth reduction. DynamicCache uses one KV entry per executed block. Cached versus full-prefill logits were checked on 32 tokens for each trained reduced configuration and the selected composite; {'all passed' if summary['all_cache_checks_passed'] else 'some failed'} the 0.001 absolute-logit threshold. Exported slopes reloaded with zero logit difference, and all original parameter hashes remained unchanged. A new cache is required if resolution changes.</p>
    <table><thead><tr><th>Candidate</th><th>Streamed ms</th><th>Latency/native</th><th>Pre-sampled ms</th><th>Latency/native</th></tr></thead><tbody>{timingrows}</tbody></table>
    <p class="small">12 alternating paired measurements after two warm-up pairs; CPU float32, batch 1, context 128. Pre-sampling occurs outside the timed region and consumes extra matrix storage. Streamed evaluation includes curve sampling each forward. Counts of fewer decoder projections are not a claim of measured speedup.</p>
    <h2>Interpretation and next experiment</h2><p>Exact knot conversion is viable across the full model in this environment. Reduced resolution is a separate optimization problem, and each window's quality must be measured in the complete model. Compare trained curves with the drop controls: improvement over interpolation alone does not establish an advantage over dropping native blocks.</p>
    <p>The next targeted comparison is a directly trained reduced-depth block stack at the same executed depth and distillation budget. That separates errors due to the curve constraint from errors due to having fewer attention/MLP steps. Function-preserving neuron/head alignment is another controlled hypothesis because this experiment interpolates raw, unaligned tensor coordinates. Further training, longer contexts and a new external test corpus would be needed before claiming a useful conversion.</p>
    <p>This implements depth-only, post-hoc Hermite curves. It does not reproduce the proposed GSA system's input-conditioned coefficients, from-scratch training, or hierarchical memory/cache.</p>
    <h2>Reproduce</h2><pre>python -m venv venv\nvenv/bin/pip install -r requirements.txt\nvenv/bin/python download_model.py\nvenv/bin/python prepare.py\nvenv/bin/python train.py\nvenv/bin/python run_evaluation.py\nvenv/bin/python build_report.py</pre><p class="small">Prepared token windows and selected checkpoints are included. The pinned original model is downloaded separately. Teacher caches are regenerated. Delete completed result files/checkpoints to retrain; evaluation resumes completed candidate rows. Model revision and source/data hashes accompany the bundle.</p></main></body></html>'''
    report=OUT/'phase4_report.html';report.write_text(content)
    readme='''# SmolLM2 joint curves — phase 4\n\nOpen results/phase4_report.html for methods, measurements and limitations.\n\nRun with Python 3.12 on CPU, at least 4 GiB available memory (8 GiB recommended). Install requirements, download the pinned original model, then run prepare.py, train.py, run_evaluation.py and build_report.py. Use SMOLLM_MODEL_DIR to point to existing model files. Token windows are supplied; teacher caches are regenerated.\n\nTo replay selected models, use load_candidate.load_candidate({4:3,12:3}); supported windows are 4 and 12 and supported resolutions are 2,3,4. This example is not a quality recommendation. Set the resolution before a sequence and start a fresh DynamicCache after changing it. All original model weights are retained. This CPU prototype requires its custom loader; Hugging Face save_pretrained does not export the external curve banks.\n\ncheckpoints/window_*.safetensors contain final selected slopes. *_local files are selected local-stage checkpoints. Coefficients are depth-only. This is not a GSA reproduction or a compressed-weight model. See results/protocol.json for prespecified settings and selection.\n\nThe earlier survey archive was recovered and rebuilt separately after a truncated stored ZIP was detected. Its complete entries passed CRC checks; the survey code, raw arrays and reports were then regenerated from the pinned original model.\n'''
    (ROOT/'README.md').write_text(readme)
    paths=[p for p in ROOT.rglob('*') if p.is_file() and not any(x in p.relative_to(ROOT).parts for x in ['cache','reference_model','__pycache__']) and p.name!='SHA256SUMS']
    (ROOT/'SHA256SUMS').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(ROOT))+'\n' for p in sorted(paths)))
    paths.append(ROOT/'SHA256SUMS');archive=ROOT.parent/'smollm_joint_curves_phase4_2026-10-17.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(paths):z.write(p,p.relative_to(ROOT.parent))
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    print(json.dumps({'report':str(report),'archive':str(archive),'archive_bytes':archive.stat().st_size,'summary':summary}),flush=True)


if __name__=='__main__':main()
