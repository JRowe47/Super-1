import base64, hashlib, html, json, pathlib, zipfile, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *

def load(name): return json.loads((OUT/name).read_text())
S=load('spectra.json'); O=load('ordering_spectra.json'); B=load('behavior.json')
I=load('sparse_interventions.json'); A=load('head_ablation.json'); G=load('gaussian_controls.json')
V=load('validation.json')
BLUE='#2367a9'; TEAL='#16857b'; ORANGE='#cf6a28'; RED='#b6444b'; GRAY='#68768a'
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
 'axes.titleweight':'bold','axes.labelcolor':'#334155','text.color':'#17243b',
 'axes.edgecolor':'#bdc7d2','grid.alpha':.25,'font.family':'DejaVu Sans','savefig.facecolor':'white'})

def weighted(rows,key): return sum(r['energy']*r[key] for r in rows)/sum(r['energy'] for r in rows)
def sr(step,kind='interval'):return [r for r in S if r['step']==step and r['kind']==kind]
def ir(name,f):return next(r for r in I if r['variant']==name and r['fraction']==f)
def savefig(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=160,bbox_inches='tight')
    fig.savefig(OUT/(name+'.svg'),bbox_inches='tight')
    plt.close(fig)
def pic(name):
    return '<img alt="'+html.escape(name.replace('_',' '))+'" src="data:image/png;base64,'+base64.b64encode((OUT/(name+'.png')).read_bytes()).decode()+'">'
def table(headers,rows):
    return '<div class="table"><table><thead><tr>'+''.join('<th>'+str(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def pct(x):return f'{x*100:.1f}%'
def paired(a,b,key='kl_to_target'):
    d=np.array([x[key]-y[key] for x,y in zip(a['passages'],b['passages'])]); n=np.array([x['tokens'] for x in a['passages']])
    ix=np.random.default_rng(SEED).integers(0,len(n),(10000,len(n)))
    boot=np.sum(d[ix]*n[ix],axis=1)/np.sum(n[ix],axis=1)
    return {'difference':float(np.sum(d*n)/n.sum()),'ci95':[float(x) for x in np.quantile(boot,[.025,.975])],
       'clusters':len(n),'unit':'original prose passage','replicates':10000,'scope':'Conditional probe uncertainty; not variation across training seeds or model families.'}

def figures():
    steps=[r['step'] for r in B]; x=np.maximum(steps,64)
    fig,ax=plt.subplots(2,2,figsize=(13,8.8),constrained_layout=True)
    fig.suptitle('Pythia-70M: behavior changes faster than the broad Fourier spectrum',fontsize=17,fontweight='bold')
    fig.supxlabel('13 published checkpoints · 6 decoder blocks · inference and analysis only',fontsize=10,color=GRAY)
    a=ax[0,0]
    a.plot(x,[r['repeated']['second_top1']*100 for r in B],'-o',color=BLUE,ms=4,lw=2)
    a.axvspan(512,1000,color=TEAL,alpha=.12)
    a.annotate('0% → 38.5%',xy=(1000,38.5),xytext=(1400,12),arrowprops={'arrowstyle':'->','color':TEAL},color=TEAL)
    a.set_xscale('log'); a.set_ylim(-3,87);a.set_ylabel('Second-copy next-token accuracy (%)')
    a.set_xticks([64,512,1000,8000,64000,143000],['0/64','512','1k','8k','64k','143k']);a.tick_params(axis='x',labelrotation=25)
    a.set_title('A  Copying appears between steps 512 and 1,000',loc='left',fontsize=11)
    a.grid(axis='y');a.set_xlabel('Training checkpoint (log spacing; 0 shown at 64)')
    a=ax[0,1]; xx=np.arange(3); width=.19
    for j,(head,label,col) in enumerate([(None,'Unablated',BLUE),(6,'Head 6 removed',RED),(5,'Head 5 removed',GRAY),(7,'Head 7 removed',TEAL)]):
        y=[next(r['accuracy']*100 for r in A if r['step']==s and r['ablated_head']==head) for s in [1000,8000,143000]]
        a.bar(xx+(j-1.5)*width,y,width,label=label,color=col)
    a.set_xticks(xx,['1,000','8,000','143,000']);a.set_ylim(0,88);a.set_xlabel('Training checkpoint')
    a.set_ylabel('Copy accuracy on fresh sequences (%)');a.grid(axis='y');a.set_axisbelow(True)
    a.set_title('B  Layer 3, head 6 contributes causally',loc='left',fontsize=11)
    a.legend(loc='upper left',ncol=2,fontsize=8,frameon=False)
    a=ax[1,0]; st=STEPS[1:]
    a.plot(st,[weighted(sr(s),'top10')*100 for s in st],'-o',color=BLUE,label='Largest 10% of real Fourier coordinates',ms=3)
    a.plot(st,[weighted(sr(s),'low10')*100 for s in st],'-o',color=ORANGE,label='Lowest 10% of frequencies',ms=3)
    a.axhline(np.mean([r['top10'] for r in G])*100,ls='--',color=GRAY,label='Gaussian control: largest 10%')
    a.set_xscale('log');a.set_ylim(0,52);a.set_ylabel('Interval-update energy retained (%)');a.set_xlabel('End of sampled interval')
    a.set_title('C  Flat Fourier updates remain broad',loc='left',fontsize=11);a.grid(axis='y');a.legend(fontsize=8,loc='center left',frameon=False)
    a=ax[1,1]
    names=['spatial_random','spatial_top','fourier_native_top','fourier_pca_top']
    labels=['Random weight updates','Largest weight updates','Native Fourier','Frozen-order Fourier']
    vals=[ir(n,.25)['kl_recovery_fraction']*100 for n in names]
    a.barh(labels,vals,color=[GRAY,ORANGE,BLUE,TEAL]);a.invert_yaxis();a.set_xlim(0,100)
    for j,v in enumerate(vals):a.text(v+1,j,f'{v:.1f}%',va='center',fontsize=10)
    a.set_xlabel('Recovery of target output distribution (%)');a.grid(axis='x');a.set_axisbelow(True)
    a.set_title('D  A modest ordering benefit at a 25% budget',loc='left',fontsize=11)
    savefig(fig,'pythia70m_overview')
    fig,axs=plt.subplots(1,2,figsize=(13,4.9),constrained_layout=True)
    for a,field,labels in [(axs[0],'layer',list(range(6))),(axs[1],'family',FAMILIES)]:
        grid=np.array([[sum(r['energy'] for r in sr(s) if r[field]==lab)/sum(r['energy'] for r in sr(s)) for s in STEPS[1:]] for lab in labels])
        im=a.imshow(grid*100,aspect='auto',cmap='YlGnBu',vmin=0,vmax=80)
        a.set_yticks(range(len(labels)),[f'Layer {x}' if field=='layer' else {'q':'Query','k':'Key','v':'Value','o':'Output','mlp_in':'MLP in','mlp_out':'MLP out'}[x] for x in labels])
        a.set_xticks(range(len(STEPS)-1),[str(s) if s<1000 else f'{s//1000}k' for s in STEPS[1:]],rotation=50)
        for iy in range(grid.shape[0]):
            for ix in range(grid.shape[1]):
                a.text(ix,iy,f'{grid[iy,ix]*100:.0f}',ha='center',va='center',fontsize=8,color='white' if grid[iy,ix]>.4 else '#17243b')
        a.set_xlabel('End of interval (previous sampled checkpoint → this one)')
        a.set_title('Update energy by '+field,loc='left')
    fig.colorbar(im,ax=axs,shrink=.8,label='Share of decoder matrix update energy (%)')
    fig.suptitle('Where net parameter movement accumulates',fontsize=16,fontweight='bold')
    savefig(fig,'layer_and_tensor_updates')
    fig,ax=plt.subplots(1,2,figsize=(12.5,4.6),constrained_layout=True)
    for name,col,label in [('native',BLUE,'Native'),('random0',GRAY,'Random ordering'),('pca_frozen',TEAL,'PCA order fitted at 2k → 4k')]:
        for a,key in zip(ax,['low10','top10']):
            a.plot(STEPS[1:],[weighted([r for r in O if r['step']==s and r['ordering']==name],key)*100 for s in STEPS[1:]],'-o',ms=4,color=col,label=label)
            a.set_xscale('log');a.set_xlabel('End of sampled interval');a.grid(axis='y')
            a.axvline(4000,color='#999999',ls=':',lw=1)
    ax[0].set_title('Lowest 10% along the MLP neuron axis',loc='left');ax[1].set_title('Largest 10% along the same axis',loc='left')
    for a in ax:a.set_ylabel('Update energy retained (%)')
    ax[0].legend(fontsize=8,frameon=False)
    fig.suptitle('The fitted order transfers forward, with small energy changes',fontsize=15,fontweight='bold')
    savefig(fig,'ordering_comparison')
    fig,ax=plt.subplots(1,2,figsize=(12,4.9),constrained_layout=True)
    for a,l,fam in [(ax[0],3,'q'),(ax[1],3,'mlp_in')]:
        grid=np.array([next(r['frequency_tree']['6'] for r in sr(s) if r['layer']==l and r['family']==fam) for s in STEPS[1:]])*64
        im=a.imshow(grid,aspect='auto',cmap='coolwarm',vmin=.8,vmax=1.2)
        a.set_yticks(range(len(STEPS)-1),[str(s) for s in STEPS[1:]])
        a.set_xlabel('Frequency leaf, low → high (64 dyadic bands)');a.set_ylabel('Interval ending at checkpoint')
        a.set_title('Layer 3 / '+('query' if fam=='q' else 'MLP input'),loc='left')
    fig.colorbar(im,ax=ax,shrink=.8,label='Band energy / uniform-band expectation')
    fig.suptitle('Recursive frequency partitions retain the broad-spectrum picture',fontsize=14,fontweight='bold')
    savefig(fig,'recursive_bands')

def main():
    figures()
    comparisons={}
    for frac in [.1,.25]:
        for control in ['fourier_native_top','spatial_top']:
            comparisons[f'pca_minus_{control}_{frac}']=paired(ir('fourier_pca_top',frac),ir(control,frac))
    layers={}
    for step in STEPS[1:]:
        rr=sr(step);total=sum(r['energy'] for r in rr)
        layers[str(step)]={'layer_shares':[sum(r['energy'] for r in rr if r['layer']==l)/total for l in range(6)],
            'family_shares':{f:sum(r['energy'] for r in rr if r['family']==f)/total for f in FAMILIES}}
    summary={'checkpoints':len(STEPS),'steps':STEPS,'decoder_matrix_parameters':18874368,'mlp_weight_parameters':12582912,
      'optimizer_steps':0,'gaussian_top10':float(np.mean([r['top10'] for r in G])),
      'flat_final_weight_top10':weighted(sr(143000,'weights'),'top10'),
      'flat_final_weight_k95_mean':float(np.mean([r['k95_fraction'] for r in sr(143000,'weights')])),
      'copy_accuracy':{str(r['step']):r['repeated']['second_top1'] for r in B},
      'paired_kl_comparisons':comparisons,'update_energy_distribution':layers,
      'interpretation':'A pronounced copying transition and localized causal head contribution coexist with broad flattened Fourier spectra. Frozen ordering modestly improves retrospective 25% MLP update sufficiency on a small prose probe; at 10%, weight-coordinate selection wins. No sparse training or grokking established.'}
    dump('summary.json',summary)
    method_rows=[]
    for name,label in [('spatial_random','Random individual weight updates'),('spatial_top','Largest individual weight updates'),('fourier_native_top','Largest native Fourier update coefficients'),('fourier_pca_top','Largest Fourier update coefficients after frozen ordering'),('fourier_native_low','Lowest native frequencies'),('fourier_pca_low','Lowest frequencies after frozen ordering')]:
        method_rows.append([label,pct(ir(name,.1)['kl_recovery_fraction']),pct(ir(name,.25)['kl_recovery_fraction'])])
    random_top=[ir('fourier_random'+str(j)+'_top',.25)['kl_recovery_fraction'] for j in range(3)]
    random_low=[ir('fourier_random'+str(j)+'_low',.25)['kl_recovery_fraction'] for j in range(3)]
    behavior_rows=[[f"{r['step']:,}",pct(r['repeated']['second_top1']),f"{r['repeated']['second_nll']:.3f}",f"{r['prose']['nll']:.3f}"] for r in B]
    ablation_rows=[]
    for s in [1000,8000,143000]:
        vals=[next(r for r in A if r['step']==s and r['ablated_head']==h) for h in [None,6,5,7]]
        ablation_rows.append([f'{s:,}']+[pct(r['accuracy']) for r in vals])
    validations=load('validation.json'); timings=load('spectral_environment.json');bt=load('behavior_environment.json')
    ci=comparisons['pca_minus_fourier_native_top_0.25'];cs=comparisons['pca_minus_spatial_top_0.25']
    html_report=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pythia-70M spectral checkpoint study</title><style>
body{{margin:0;background:#eef2f6;color:#18253c;font:16px/1.65 system-ui,-apple-system,Segoe UI,sans-serif}}main{{max-width:1060px;margin:0 auto;background:white;padding:42px 48px 60px}}h1{{font-size:36px;line-height:1.18;margin:.3em 0}}h2{{font-size:24px;line-height:1.3;margin-top:2em}}h3{{font-size:19px}}p{{max-width:94ch}}.eyebrow{{color:#16857b;font-weight:700;letter-spacing:.08em;font-size:13px}}.lead{{font-size:19px;color:#3d506b}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:25px 0}}.card{{background:#edf4f8;padding:15px;border-radius:7px}}.card strong{{display:block;font-size:26px;color:#2367a9}}.card span{{font-size:13px}}img{{display:block;width:100%;height:auto;margin:25px 0}}table{{border-collapse:collapse;width:100%;font-size:14px}}th{{text-align:left;background:#eaf1f6}}td,th{{padding:10px 12px;border-bottom:1px solid #dce3ea}}.table{{overflow-x:auto;margin:20px 0}}.callout{{background:#eff8f5;border-left:4px solid #16857b;padding:15px 20px}}.small{{font-size:13px;color:#536378}}a{{color:#2367a9}}pre{{white-space:pre-wrap;background:#17243b;color:#e8f2f7;padding:18px;border-radius:6px;font-size:13px}}li{{margin:.5em 0}}@media(max-width:700px){{main{{padding:22px 17px}}h1{{font-size:29px}}.cards{{grid-template-columns:repeat(2,1fr)}}}}
</style></head><body><main>
<div class="eyebrow">GEOMETRIC SELF ATTENTION · EXPERIMENT · 17 OCTOBER 2026</div>
<h1>A real copying transition,<br>without a broad Fourier collapse</h1>
<p class="lead">Pythia-70M develops strong repeated-sequence copying between two early checkpoints. Its flattened weight and update spectra remain broad. A valid neuron ordering fitted on an earlier update gives a modest functional benefit at a 25% update budget.</p>
<div class="cards"><div class="card"><strong>13</strong><span>published checkpoints</span></div><div class="card"><strong>6</strong><span>decoder blocks tracked</span></div><div class="card"><strong>18.87M</strong><span>matrix entries per checkpoint</span></div><div class="card"><strong>0</strong><span>new training steps</span></div></div>
{pic('pythia70m_overview')}
<h2>What this adds to the previous experiment</h2>
<p>The recovered SmolLM study showed that coherent MLP permutations can change detailed spectra without changing outputs; its small guided-Fourier advantage over guided individual weights was not statistically resolved. This study preserves each Pythia layer and its native neuron identity across checkpoints, then tests an explicitly frozen alternative ordering. The October 15 project archive and October 17 FFT bundle are recorded by hash in <code>continuity_provenance.json</code>.</p>
<p>We analyzed <code>EleutherAI/pythia-70m</code>, standard data, original training seed, without mixing deduped or v0 checkpoints. Sampled steps: {', '.join(f'{s:,}' for s in STEPS)}. The model has six 512-wide decoder blocks, eight heads per block, and a 2,048-wide MLP. The main spectra cover query, key, value, attention output, MLP input and MLP output matrices. Embeddings, the vocabulary head, normalization and biases are excluded from spectral headline claims, but are present in every model inference.</p>

<h2>1. A copying transition with a causal head contribution</h2>
<p>On 32 seeded random sequences of 32 unique tokens repeated twice, second-copy next-token accuracy is <strong>0% at step 512</strong>, <strong>38.5% at step 1,000</strong>, and <strong>76.4% at step 8,000</strong>. It is not monotonic: the final checkpoint scores <strong>67.8%</strong> on these sequences. Each checkpoint uses the same 992 scored token positions. These are synthetic copying probes, not language benchmark scores.</p>
<p>At step 1,000, layer 3 / head 6 assigns 86.0% mean attention to the token following the earlier matching token. When the second copy is shuffled, the same content-following score remains 52.3%, versus 2.7% attention to the fixed-offset control. The result therefore goes beyond attending to a periodic position.</p>
<p>After observing that head, we specified an ablation addendum and tested <strong>64 fresh sequences / 1,984 scored positions</strong>. Removing its contribution before the attention output projection reduces step-1,000 accuracy from <strong>39.1% to 0.8%</strong>. Two fixed same-layer head controls have much smaller early effects. By later training, other heads also contribute; this is not a claim that one head explains all copying.</p>
{table(['Checkpoint','Unablated','Remove head 6','Remove head 5','Remove head 7'],ablation_rows)}
<p class="small">Layers and heads are zero-indexed. Zeroing a head is an out-of-distribution causal intervention; it establishes a contribution on this probe, not the complete circuit. Paired sequence bootstrap intervals and all individual sequence results are retained.</p>
<p>This is consistent with induction-style copying emergence. <strong>It is not established grokking.</strong> Grokking requires evidence of delayed generalization after fitting a training task; our checkpoint probes do not supply matched train/test learning curves. The induction-head literature motivates this probe, but the measurements and ablation above are new results from this run.</p>

<h2>2. Flat Fourier spectra do not isolate the behavioral transition</h2>
<p>For the interval ending at step 512, the largest 10% of real Fourier coordinates retain {pct(weighted(sr(512),'top10'))} of matrix-update energy. For 512 → 1,000, this is {pct(weighted(sr(1000),'top10'))}. The Gaussian control retains {pct(summary['gaussian_top10'])}. The lowest 10% retain {pct(weighted(sr(512),'low10'))} and {pct(weighted(sr(1000),'low10'))}, respectively. The striking behavioral change has no comparable collapse into a few flattened Fourier coordinates in these summaries.</p>
<p>At the final checkpoint, the largest 10% of flattened weight coefficients retain {pct(summary['flat_final_weight_top10'])} of energy. Reconstructing 95% of each tensor's energy requires about {pct(summary['flat_final_weight_k95_mean'])} of its real coordinates on average; the Gaussian reference is similar. This is a negative result for strong generic flat-Fourier compression. It does not rule out localized, activation-aware, phase-sensitive or better-aligned structure.</p>
{pic('recursive_bands')}
<p>Each tensor's complex rFFT was split recursively into 2, 4, 8, 16, 32 and 64 frequency bands. The figure shows two examples; the machine-readable results retain every band at every depth for all 36 matrices and 13 checkpoints. The color scale is clipped at 0.8–1.2 times equal-band energy to make small variation visible. The tree organizes the same coefficients; it does not create additional independent information. Repeatedly transforming the coefficients themselves is not the decomposition used here. Reconstruction retains phases.</p>

<h2>3. Layer and tensor identity reveal more informative localization</h2>
{pic('layer_and_tensor_updates')}
<p>During 512 → 1,000, layer 3 accounts for <strong>{pct(layers['1000']['layer_shares'][3])}</strong> of decoder-matrix update energy, versus {pct(layers['512']['layer_shares'][3])} in the preceding sampled interval. That is the layer containing the early copying head. This coincidence is descriptive; the head ablation provides the separate causal evidence.</p>
<p>During the broad late interval 64,000 → 143,000, layer 4 accounts for <strong>{pct(layers['143000']['layer_shares'][4])}</strong> of the measured update energy, and query matrices account for <strong>{pct(layers['143000']['family_shares']['q'])}</strong> across layers. A follow-up audit confirms that this query movement is distributed across many entries and several heads, not one isolated scalar. Parameter movement is not a measure of functional importance; query scaling and attention saturation are plausible follow-up questions, not established explanations.</p>
<p class="small">Every column is a net difference between adjacent sampled checkpoints. The intervals have different lengths. These are displacement energies, not per-step gradients or accumulated optimization path lengths. Cross-layer neuron coordinates are not treated as aligned.</p>

<h2>4. A valid ordering helps function slightly more than energy concentration</h2>
<p>For each MLP, we concatenated its incoming and outgoing neuron update signatures from steps 2,000 → 4,000, standardized the two matrix families, estimated the leading principal direction, and sorted neurons by their scores. This is a simple PCA ordering heuristic, not an optimal alignment algorithm. We froze that order before evaluating later updates. The incoming weight rows and biases move together with outgoing weight columns; residual coordinates and attention coordinates are unchanged. The existing shared-initialization correspondence remains the primary baseline.</p>
{pic('ordering_comparison')}
<p>On the subsequent 4,000 → 8,000 update, the largest 10% of neuron-axis Fourier coordinates retain <strong>44.83%</strong> of MLP weight-update energy in native order and <strong>44.87%</strong> after the frozen ordering. Low-frequency retention changes from <strong>10.32%</strong> to <strong>10.80%</strong>. This is a small concentration effect, despite the fit interval having preceded the evaluation interval. All earlier-than-fit values in the figure are descriptive and are not forward-transfer tests.</p>
<p>We then measured function. Start from the step-8,000 model, revert only its 12.58M MLP weight entries to step 4,000, and restore selected portions of that completed update. All other parameters, including MLP biases, remain at step 8,000. Score agreement with the full target distribution on 12 original prose passages, 520 next-token targets. A recovery of 100% means zero target KL; 0% is the reverted baseline; negative recovery is worse than reverting.</p>
{table(['Update coordinates retained','10% budget: KL recovery','25% budget: KL recovery'],method_rows)}
<p>At 25%, the frozen-order Fourier result is <strong>74.5%</strong>, versus <strong>71.6%</strong> for native Fourier and <strong>72.5%</strong> for individual weights. Three random neuron orderings span <strong>{pct(min(random_top))}–{pct(max(random_top))}</strong> for largest-coefficient Fourier selection. At 10%, individual weights recover <strong>46.7%</strong>, beating frozen-order Fourier's <strong>42.9%</strong>. There is therefore no general Fourier advantage established.</p>
<p>The low-frequency comparison is more sensitive to ordering: at 25%, frozen-order low frequencies recover 34.5%, native low frequencies recover −1.0%, and the three added random-order controls span {pct(min(random_low))}–{pct(max(random_low))}. That suggests the order changes which directions are useful, not just how much squared energy they carry. The random-low controls were an explicitly recorded follow-up after the primary result.</p>
<p>For the 25% largest-coefficient comparison, frozen ordering reduces KL by {abs(ci['difference']):.4f} nats versus native Fourier, with a paired passage-bootstrap 95% interval [{ci['ci95'][0]:.4f}, {ci['ci95'][1]:.4f}] for the difference; versus individual weights the difference is {cs['difference']:.4f}, interval [{cs['ci95'][0]:.4f}, {cs['ci95'][1]:.4f}]. These are conditional intervals on 12 short passages, not confidence across pretraining seeds, languages or tasks. The probe is too small to justify a general efficiency claim.</p>
<div class="callout"><strong>Crucial distinction:</strong> every selected coefficient comes from an already completed training update. This measures retrospective update sufficiency. It does not demonstrate training with only those coefficients, reduced training compute, storage compression, or faster inference.</div>

<h2>5. What the Fourier coordinates and budgets mean</h2>
<p>The flattened transform runs along C-order matrix entries, separately for each tensor. Ordering experiments instead transform the MLP hidden-neuron axis, preserving the other matrix axis. These are different bases and are labeled separately. Weight-index frequency is not token-position frequency or a semantic frequency.</p>
<p>We pack each real-valued length-N signal into N orthonormal real Fourier coordinates: DC, the scaled real and imaginary parts of interior rFFT bins, and Nyquist when present. Thus a 10% or 25% budget counts actual real degrees of freedom; conjugate duplicates are not counted as independent coefficients. Exactly the same number of real coordinates per matrix is retained across methods. Low-frequency masks traverse frequency first and may partially select the boundary frequency to match that count. All magnitude-selected masks are computed from the target update itself.</p>
<p>Singular-value diagnostics supply a separate, permutation-invariant check. For the 4,000 → 8,000 MLP update matrices, 95% energy requires ranks 413–437 of a maximum 512, and the leading 16 singular components retain only 8.2–10.8%. These updates are not dominated by a tiny low-rank subspace either. Earlier intervals contain more uneven directions; the full summaries are retained.</p>

<h2>Validation and reproducibility</h2>
<ul><li>All 13 downloaded checkpoint SHA-256 hashes still match after analysis. Immutable Hugging Face commit hashes and file sizes are in <code>checkpoint_manifest.json</code>.</li>
<li>Real Fourier Parseval, inverse-transform, dyadic-band sum, and eight-band reconstruction checks pass. Maximum tested relative reconstruction RMS is below {max(validations['max_axis_reconstruction_relative_rms'],validations['max_band_reconstruction_relative_rms']):.2g}.</li>
<li>Both tested coherent neuron orderings preserve all 260 next-token argmax choices. Relative RMS logit differences are below 8 × 10<sup>−7</sup>, consistent with floating-point accumulation changes.</li>
<li>Restoring the full MLP update exactly reproduces the target prose NLL and gives zero measured KL.</li>
<li>The spectral pass took {timings['seconds']:.1f} seconds; primary behavioral probes plus sparse interventions took {bt['seconds']:.1f} seconds on this CPU host. Downloads, follow-up controls, plotting and packaging are separate. These are diagnostic runtimes, not training or inference speedups.</li></ul>
<p>The archive contains executable scripts, frozen protocol, dated addenda, exact input tokens, per-passage and per-sequence outputs, band trees, orderings, metrics, plots, package versions, and manifests. The 3.66 GB of public checkpoint weights are not duplicated; the downloader retrieves pinned revisions for reproduction. No model training is required.</p>
<pre>python -m pip install -r requirements.txt
python download_checkpoints.py
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python analyze_spectra.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=3 python behavior.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=3 python transition_probe.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=3 python low_frequency_controls.py
python validate.py
python build_report.py</pre>
<details><summary>All checkpoint behavioral measurements</summary>
{table(['Step','Repeated-copy accuracy','Second-copy NLL','Original-prose NLL'],behavior_rows)}
<p class="small">NLL is measured in natural-log units per token. Uniform-random vocabulary sequences are strongly out of distribution. Copy advantage can rise when first-copy prediction becomes worse, so accuracy and absolute second-copy loss are retained instead of interpreting the advantage alone.</p></details>

<h2>Next experiment suggested by this result</h2>
<p>Focus on the 512 → 1,000 transition in layer 3, especially head 6 and its upstream contributors. Compare matched-budget interventions in native coordinates, head-local Fourier coordinates, and activation- or gradient-guided directions. Preserve the same content-copying control and include ordinary text loss. That would test whether a localized functional signal identifies useful updates that broad power spectra miss.</p>
<p>For ordering, repeat the frozen-order comparison on a second interval and another released Pythia seed before investing in a more expensive alignment search. Test the selected coordinates during training only after retrospective sufficiency is robust. Pythia's fixed depth does not test the earlier depth-change hypothesis; no layer deletion, insertion or depth interpolation was performed here.</p>

<h2>Primary sources</h2><ol>
<li><a href="https://github.com/EleutherAI/pythia">EleutherAI Pythia repository</a> and <a href="https://arxiv.org/abs/2304.01373">Pythia paper</a>: checkpoint design, architecture and training provenance.</li>
<li><a href="https://huggingface.co/EleutherAI/pythia-70m">Pythia-70M model repository</a>: actual checkpoint files and configuration; pinned revisions recorded with the results.</li>
<li><a href="https://arxiv.org/abs/2209.11895">Olsson et al., In-context Learning and Induction Heads</a>: motivation for content-following copying probes; it is not the source of our Pythia measurements.</li>
<li><a href="https://arxiv.org/abs/2201.02177">Power et al., Grokking</a>: delayed generalization distinction. We have not demonstrated that phenomenon in this study.</li></ol>
<p class="small">Single pretraining run; exploratory analysis; small diagnostic evaluation; some follow-ups chosen after initial observations. Measurements, prior results, interpretations and proposed next experiments are kept separate.</p>
</main></body></html>'''
    (ROOT/'pythia70m_spectral_report.html').write_text(html_report)
    (ROOT/'README.txt').write_text('Pythia-70M spectral checkpoint study — 17 October 2026\n\nOpen pythia70m_spectral_report.html for methods and results.\nNo model training is performed.\n\nReproduction:\n1. Install requirements.txt.\n2. Run download_checkpoints.py (about 3.66 GB public weights).\n3. Run analyze_spectra.py, behavior.py, transition_probe.py, low_frequency_controls.py, validate.py, build_report.py.\nUse OPENBLAS_NUM_THREADS=1 or 2 and OMP_NUM_THREADS=3 to avoid oversubscription.\n\nAll checkpoints have pinned revisions and SHA-256 hashes in checkpoint_manifest.json.\nThe initial download resolves branch refs only if no manifest is present.\nReport figures and machine-readable metrics are in results/.\nNo external code repository is modified.\n')
    (ROOT/'requirements.txt').write_text('numpy>=2.0\nscipy>=1.12\nmatplotlib>=3.8\ntorch>=2.2\ntransformers==4.57.1\nsafetensors>=0.4.5\n')
    files=[p for p in ROOT.rglob('*') if p.is_file() and 'checkpoints' not in p.relative_to(ROOT).parts and '__pycache__' not in p.parts and p.name!='SHA256SUMS']
    lines=[hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(ROOT)) for p in sorted(files)]
    (ROOT/'SHA256SUMS').write_text('\n'.join(lines)+'\n')
    bundle=ROOT.parent/'pythia70m_spectral_2026-10-17.zip'
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files+[ROOT/'SHA256SUMS']: z.write(p,ROOT.name+'/'+str(p.relative_to(ROOT)))
    print(json.dumps({'report':str(ROOT/'pythia70m_spectral_report.html'),'bundle':str(bundle),'bytes':bundle.stat().st_size,'comparisons':comparisons}),flush=True)

if __name__=='__main__':main()
