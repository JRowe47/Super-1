"""Render the completed diagnostic and a proposed, unrun toy experiment."""
from pathlib import Path
import base64,html,json,zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results'
d=json.loads((OUT/'diagnostics.json').read_text())
a=d['aggregate_projection_spectra'];family=d['family_spectra']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
    'axes.spines.right':False,'figure.facecolor':'#f6f7fb','axes.facecolor':'white',
    'axes.titleweight':'bold','axes.titlesize':12})
blue='#2563eb';orange='#e87924';green='#0d8b78';gray='#64748b'
fig,axs=plt.subplots(2,2,figsize=(12.6,8.2),layout='constrained')
k=np.arange(28)
ax=axs[0,0]
ax.plot(k,np.array(a['retained_initial']['dct_fraction'])*100,ls='--',color=gray,lw=2.8,label='Before distillation')
ax.plot(k,np.array(a['retained_final']['dct_fraction'])*100,color=blue,lw=1.6,label='After distillation')
ax.axhline(100/28,color='#94a3b8',ls=':',label='Equal energy: 3.57% per mode')
ax.set(title='Full weights: nearly flat; before/after overlap',xlabel='Depth cosine mode (0 is constant)',ylabel='Share of projection-weight energy (%)',ylim=(0,5))
ax.legend(fontsize=9,loc='lower left');ax.grid(alpha=.14)
ax=axs[0,1]
for arm,color,label in [('retained',blue,'Retained-layer start'),('interpolated',orange,'Interpolated start')]:
    e=np.array(a[arm+'_delta']['fft_fraction'])
    multiplicity=np.r_[1,np.full(13,2),1]
    ax.plot(np.arange(15),e/multiplicity*100,'o-',ms=3,color=color,label=label)
ax.axhline(100/28,color='#94a3b8',ls=':')
ax.set(title='Training updates: no dominant Fourier band',xlabel='Fourier frequency across 28 blocks',ylabel='Energy per real degree of freedom (%)',ylim=(0,5))
ax.legend(fontsize=9,loc='lower left');ax.grid(alpha=.14)
ax=axs[1,0]
for arm,color,label in [('retained',blue,'Retained final weights'),('interpolated',orange,'Interpolated final weights')]:
    ax.plot(np.arange(1,29),100*np.cumsum(a[arm+'_final']['dct_fraction']),color=color,label=label)
ax.plot(np.arange(1,29),np.arange(1,29)*100/28,color=gray,ls=':',label='Equal-energy reference')
ax.scatter([8],[100*a['retained_final']['dct_low8_fraction']],color=blue,zorder=5)
ax.annotate('8 modes retain 28.72%',(8,28.72),xytext=(3,62),arrowprops={'arrowstyle':'->','color':gray},fontsize=10)
ax.set(title='Few low modes do not reconstruct the weights',xlabel='Number of lowest cosine modes retained',ylabel='Cumulative weight energy (%)',xlim=(1,28),ylim=(0,103))
ax.legend(fontsize=9,loc='lower right');ax.grid(alpha=.14)
ax=axs[1,1]
names=['input_layernorm.weight','post_attention_layernorm.weight','self_attn.q_proj.weight','self_attn.k_proj.weight','self_attn.v_proj.weight','self_attn.o_proj.weight','mlp.gate_proj.weight','mlp.up_proj.weight','mlp.down_proj.weight']
labels=['Input RMSNorm','Post-attention RMSNorm','Attention Q','Attention K','Attention V','Attention O','MLP gate','MLP up','MLP down']
values=[100*family[n]['retained_final']['dct_low8_fraction'] for n in names]
ax.barh(labels,values,color=[green if 'layernorm' in n else blue for n in names]);ax.invert_yaxis()
for i,v in enumerate(values):ax.text(v+1,i,f'{v:.1f}',va='center',fontsize=9)
ax.axvline(100*8/28,color=gray,ls=':',lw=1)
ax.set(title='Normalization vectors are a distinct case',xlabel='Energy in lowest 8 cosine modes (%)',xlim=(0,112));ax.grid(axis='x',alpha=.14)
fig.suptitle('SmolLM2-135M · 28-block depth spectrum',fontsize=19,fontweight='bold')
fig.supxlabel('Post-hoc endpoint analysis. Only six blocks were trained. Spectral energy is not language-model importance.',fontsize=10,color='#475569')
fig.savefig(OUT/'smollm_depth_spectra.png',dpi=170)
fig.savefig(OUT/'smollm_depth_spectra.svg')
plt.close(fig)

toy={
 'status':'Proposed, not run',
 'question':'Does depth-structured training create useful smoothness or merely change optimizer coordinates?',
 'architecture':{'blocks':8,'width':128,'heads':4,'context':128,'vocabulary_target':8192,'tied_head':True,'approx_parameters':'2.6 million with a standard 4x MLP'},
 'data':'Fixed document-held-out ordinary-text corpus such as the already used WikiText data; fixed tokenizer. TinyStories is not the main experiment.',
 'seeds':3,
 'initialization':'Draw one dense initialization per seed; derive full DCT and random orthogonal coefficients from it to match initial functions exactly.',
 'stage1':['Dense block parameters','All 8 orthonormal cosine modes across depth','All 8 fixed random orthogonal depth modes'],
 'stage1_control':'First verify a few plain-SGD steps agree after mapping to weights, with no dropout and matching global clipping/decay conventions. Then compare AdamW with equal small validation LR-search budgets. Full-basis representation does not reduce capacity.',
 'stage2':['Dense updates on a shared base checkpoint','Four low-frequency depth modes for updates to the same fixed base','Four random orthogonal depth modes for updates to the same fixed base'],
 'stage2_control':'All arms start at exactly the same function using zero initial updates. The two subspace arms have matched trainable counts. A dense-vs-subspace gain cannot automatically be attributed to Fourier structure.',
 'measurements':['Validation cross-entropy versus tokens, updates and wall time','Per-family spectra of weights and actual updates at fixed checkpoints','Gradient energy and modest band attenuation / update-band removal with loss and teacher KL','SVD rank of the depth stack, family norms and fixed-probe residual activations','Parameter and optimizer-state bytes; peak memory; reconstruction overhead'],
 'depth_followup':'Only after fixed-depth controls, add an explicit multi-depth objective to a selected construction. Compare 4/6/8 block schedules and a held-out 7-block schedule; fix residual scaling and document sampling. Match training exposure and report extra compute. Fourier interpolation alone gives no depth-reduction guarantee.',
 'causal_rule':'A spectral change nominates a target; a matched intervention causing a reproducible held-out loss change establishes local usefulness. Neither identifies semantic circuitry exactly.',
}
(OUT/'toy_experiment_plan.json').write_text(json.dumps(toy,indent=2)+'\n')

def table(headers,rows):
    return '<table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'

energy_rows=[]
for key,label in [('retained_initial','Retained: before'),('retained_final','Retained: after'),('interpolated_initial','Interpolated: before'),('interpolated_final','Interpolated: after'),('retained_delta','Retained: update'),('interpolated_delta','Interpolated: update')]:
    r=a[key];energy_rows.append([label,f"{r['dct_low8_fraction']*100:.4f}%",r['dct_90pct_prefix'],r['dct_99pct_prefix'],r['numerical_rank']])
normrows=[[label,f'{value:.2f}%'] for label,value in zip(labels,values)]
nullrows=[]
for key,r in d['layer_order_nulls'].items():
    lo,hi=r['percentile_2_5_97_5']
    nullrows.append([key.replace('_',' '),f"{100*r['observed']:.3f}%",f"{100*r['mean']:.3f}%",f'{100*lo:.3f}–{100*hi:.3f}%'])
err=max(max(v.values()) for v in d['roundtrip_checks'].values())
image='data:image/png;base64,'+base64.b64encode((OUT/'smollm_depth_spectra.png').read_bytes()).decode()
report=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>SmolLM2 depth Fourier probe</title><style>
body{{margin:0;background:#f6f7fb;color:#17243b;font:17px/1.6 system-ui,sans-serif}}main{{max-width:1050px;margin:auto;padding:36px 24px 72px}}h1{{font-size:34px;line-height:1.15}}h2{{margin-top:36px}}.lede{{font-size:21px}}.card{{background:white;padding:20px 24px;border-radius:12px;margin:22px 0}}table{{width:100%;border-collapse:collapse;font-size:15px;background:white}}th,td{{padding:10px 12px;border-bottom:1px solid #dce1eb;text-align:left}}th{{background:#eaf0fb}}img{{width:100%;height:auto}}code,pre{{font-family:ui-monospace,monospace;font-size:14px}}pre{{white-space:pre-wrap;background:#eaf0fb;padding:18px}}small{{color:#536278}}a{{color:#1d4ed8}}</style></head><body><main>
<small>17 October 2026 · Read-only diagnostic · No new model training</small>
<h1>Fourier curves across the 28 blocks</h1>
<p class="lede">An exact decomposition is cheap. These checkpoints do not show a small set of dominant low-frequency depth modes in the projection weights.</p>
<div class="card"><strong>Measured:</strong> the seven projection families retain {100*a['retained_final']['dct_low8_fraction']:.2f}% of their squared weight norm in the lowest 8 of 28 cosine modes. An equal-energy reference retains {100*8/28:.2f}%. All 28 modes are needed to cross 99% in this frequency order. Training updates are also broadly distributed. This is a structural weight measurement, not a language-quality ablation.</div>
<img src="{image}" alt="Measured depth spectra and family comparisons">
<h2>What exactly is decomposed?</h2>
<p>For each corresponding tensor entry, collect its value across the 28 decoder blocks. Attention Q, K, V, O; MLP gate, up, down; and two RMSNorm vectors are kept as separate families. Tensor shapes match across blocks. Token embeddings, tied output head and final model norm are outside this depth stack.</p>
<p>Write the complete block-weight vector at depth ℓ as W(ℓ) = Σ φₖ(ℓ) Bₖ. Each Bₖ is a block-sized coefficient vector containing {d['parameters_per_block']:,} real values; φₖ is a scalar sine/cosine basis function. A mode is therefore a tensor-valued curve contribution, not one scalar storing the whole block.</p>
{table(['Interpretation','Exact reconstruction requirement'],[['DFT across 28 blocks','15 nonnegative frequency bands: DC + 13 sine/cosine pairs + Nyquist = 28 real coefficient tensors'],['DCT across 28 blocks','28 real cosine coefficient tensors'],['One flattened scalar sequence over the 28 blocks','99,122,688 real degrees of freedom, or 49,561,345 nonnegative complex FFT bins with endpoint constraints'],['Learned linear depth basis','Measured projection-stack rank is 28; fewer independent basis vectors cannot reconstruct it exactly at the recorded numerical tolerance']])}
<p>The exact depth transform retains {d['body28_parameters']:,} real coefficient values, about {4*d['body28_parameters']/2**20:.1f} MiB at FP32 before metadata. It is a change of coordinates. Low-mode truncation is a separate lossy experiment. Smooth sine waves can sum to an arbitrarily jagged discrete sequence. The DFT extends depth periodically, coupling the last block back to the first. A cosine transform is a useful boundary-condition control because its even extension avoids that wraparound assumption.</p>
<p>Exact here means exact in real arithmetic and within roundoff in the numeric implementation. Across every retained-final tensor, float64 FFT and DCT roundtrips had maximum absolute error {err:.3g}. The combined transforms and inverse checks took {d['fourier_plus_dct_roundtrip_seconds']:.2f} seconds; the full diagnostic took {d['total_analysis_seconds']:.2f} seconds on this CPU. These timings exclude downloading and loading a language runtime; they do not predict per-token speed.</p>
<p>A global Fourier interpolant through the 28 sampled weights does not recreate the previous Hermite curve between samples, and does not automatically preserve the original 30 layer knots. It also does not preserve the composed network when evaluated at fewer depths. More off-grid samples add no new information.</p>
<h2>Before versus after phase 6</h2>
{table(['Projection weights / updates','Energy in 8 lowest DCT modes','Modes for 90%','Modes for 99%','Depth-stack rank'],energy_rows)}
<p>Squared norms are pooled over the seven matrices. The maximum before/after change in any normalized DCT mode is only {100*np.max(np.abs(np.array(a['retained_initial']['dct_fraction'])-a['retained_final']['dct_fraction'])):.6f} percentage points for retained initialization, and {100*np.max(np.abs(np.array(a['interpolated_initial']['dct_fraction'])-a['interpolated_final']['dct_fraction'])):.6f} for interpolation. The absolute weight spectrum is dominated by the pretrained checkpoint, so actual updates are the more sensitive diagnostic.</p>
<p>Only final block positions 4, 5, 6, 11, 12, 13 changed. Their updates are zero elsewhere. The update stack therefore has rank at most six by design; observing rank six is not a newly discovered six-curve compression mechanism. Localized, nearly orthogonal updates can have a nearly flat Fourier spectrum. This outcome cannot rule out useful localized or aligned structure. Power spectra discard phase and sign information: a nearly unchanged energy plot does not mean an unchanged model. Transforming the actual before/after difference captures coefficient movement, but still does not identify semantic function.</p>
<h2>Family differences</h2>
{table(['Family','Retained-final energy in lowest 8 DCT modes'],normrows)}
<p>The normalization vectors are markedly smoother, but together they contain only 1,152 values per block ({100*1152/d['parameters_per_block']:.4f}% of block parameters). Their large constant component is partly a shared-scale effect. This is a candidate for a small functional ablation, not evidence that the main matrices can be safely smoothed.</p>
<h2>Layer-order control</h2>
<p>One thousand fixed-seed permutations of block order were applied to the Gram matrices. These are descriptive controls on spectra; the permuted networks were not evaluated and are not function-preserving. The statistic below excludes the constant mode and asks how much remaining energy lies in cosine modes 1–7.</p>
{table(['Stack','Actual order','Shuffle mean','Shuffle central 95%'],nullrows)}
<p>The trained raw weights show slight ordering structure, but the effect is small: roughly 26% versus a 25.9% shuffled reference, not strong low-frequency concentration. Update spectra lie comfortably within their shuffled ranges. These are exploratory comparisons, not preregistered significance tests.</p>
<h2>What this does and does not compare</h2>
<p>Phase 6 optimized ordinary block parameters and then lifted each selected checkpoint into a curve. At the fitted samples, native and curve forms represent the same weights; their spectra must agree within roundoff. They are not independently trained regimes. Only initial/final weights were saved, so this probe measures endpoint change, not the trajectory during training.</p>
<p>A full orthonormal DCT parameterization changes no expressivity. With matched initial weights, plain SGD steps map back identically in ideal arithmetic. AdamW uses coordinate-wise moments and generally follows a different path after an orthogonal transform. Therefore a curve-training advantage must be separated from an optimizer-coordinate effect. The random-basis control serves that purpose.</p>
<h2>Recommended toy experiment — not run</h2>
<p>Use 8 blocks, width 128, 4 heads, context 128 and an approximately 8K-token vocabulary with a tied output head: about 2.6M parameters. This gives more useful depth-frequency resolution than a 4-block model. Use document-held-out ordinary text, three paired seeds, identical minibatches and saved checkpoints; report tokens, updates and wall time.</p>
<ol><li><strong>Representation control:</strong> dense weights, all 8 DCT depth modes, and all 8 random orthogonal depth modes. Derive each from the same initialization. Verify a short plain-SGD equivalence run; then compare AdamW with equal learning-rate search budgets.</li>
<li><strong>Structural control:</strong> on the same fixed base model, learn updates in 4 low depth modes versus 4 random orthogonal modes, plus a dense-update reference. Zero updates give identical initial functions. This tests a smooth-depth prior without conflating it with different starting models.</li>
<li><strong>Interventions:</strong> attenuate individual bands or remove training-update bands, by tensor family, and measure held-out loss and teacher KL. Keep parameter displacement comparable. Frequency energy alone cannot identify which modes carry useful behavior.</li>
<li><strong>Depth functionality:</strong> after these controls, test an explicit multi-depth training objective, fixed sampling and residual scaling. Compare 4/6/8 steps and a held-out 7-step schedule. Smooth weights do not automatically imply stable reduced-depth computation.</li></ol>
<p>Track per-family weight, update and gradient spectra, coefficient drift (including phase/sign), depth-stack singular values, and residual activations on fixed text probes. Functional permutations of neurons/heads can alter weight spectra, so comparisons between independently trained models require symmetry-aware alignment or matched initialization. Arbitrarily sorting weights does not preserve the model.</p>
<h2>Related work and interpretation</h2>
<p><a href="https://arxiv.org/abs/2604.04440v2">Training Transformers in Cosine Coefficient Space (Bergach, 2026, v2)</a> reports a 4-layer, width-128 character model using within-matrix 2D DCT parameterization. Its revised ablation finds random orthogonal bases competitive with DCT. This is a small-model preprint, uses a different transform axis from our depth experiment, and does not establish a universal spectral advantage. It motivates the random-basis control here; its larger mechanistic claims are not adopted as established facts.</p>
<p><a href="https://arxiv.org/abs/1506.03767">Rippel et al. (2015)</a> study spectral CNN parameterization. <a href="https://arxiv.org/abs/2205.13599">VectorAdam (2022)</a> documents Adam's coordinate dependence. <a href="https://arxiv.org/abs/2209.04836">Git Re-Basin</a> demonstrates why neuron permutations matter to weight comparisons. Transform conventions follow the official <a href="https://numpy.org/doc/stable/reference/generated/numpy.fft.rfft.html">NumPy real FFT</a> and <a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.dct.html">SciPy DCT</a> documentation.</p>
<h2>Reproduce</h2>
<p>The small bundle contains the analysis, all aggregate/family statistics, Gram matrices, figure, and the proposed toy protocol. It does not duplicate model weights. Recover the phase-6 bundle and its pinned reference model, then run:</p>
<pre>OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python3 run_probe.py --phase6 /path/to/smollm_initialization_phase6
python3 build_report.py</pre>
<p>Reference revision: <code>{d['revision']}</code>. Input SHA-256 hashes and runtime versions are recorded in <code>results/diagnostics.json</code>. No new quality, latency, compression or training claim follows from this weight-only probe.</p>
</main></body></html>'''
(OUT/'smollm_fourier_probe.html').write_text(report)
(ROOT/'requirements.txt').write_text('numpy==2.3.5\nscipy==1.17.0\nmatplotlib\nthreadpoolctl\n')
(ROOT/'README.txt').write_text('SmolLM2 depth Fourier probe, 17 October 2026.\nOpen results/smollm_fourier_probe.html.\nRead-only post-hoc analysis; no new training or language-model evaluation.\nTo reproduce: recover the phase-6 archive and download its pinned model using its download_model.py, then run run_probe.py --phase6 PATH and build_report.py.\nThe archive intentionally omits model weights; input hashes appear in diagnostics.json.\nToy training plan is proposed only, not executed.\n')
archive=ROOT.parent/'smollm_fourier_probe_2026-10-17.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(ROOT.rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts:z.write(p,p.relative_to(ROOT.parent))
print(json.dumps({'report':str(OUT/'smollm_fourier_probe.html'),'archive':str(archive),'bytes':archive.stat().st_size}))
