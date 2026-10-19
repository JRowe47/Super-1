import base64,hashlib,html,json,zipfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#f6f7fb','axes.facecolor':'white','axes.titlesize':12,'axes.titleweight':'bold'})
blue='#2563eb';green='#078775';orange='#e87924';gray='#64748b';red='#c74861'

def load(name):return json.loads((OUT/name).read_text())
def savefig(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=160);fig.savefig(OUT/(name+'.svg'));plt.close(fig)
def table(headers,rows):
    return '<table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'
def img(name):return '<img src="data:image/png;base64,'+base64.b64encode((OUT/name).read_bytes()).decode()+'">'

def spectral_figures():
    x=load('global_spectra.json');fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    for ax,key,title in zip(axes.flat,['retained_final','retained_delta','interpolated_delta','retained_delta_column_major'],['Trained 28-block waveform','Retained-start training update','Interpolated-start training update','Same retained update, different serialization']):
        r=x[key];grid=[]
        for level in range(1,7):
            row=sorted([n for n in r['nodes'] if n['level']==level],key=lambda n:n['index'])
            grid.append(np.repeat([n['fraction']/(n['real_dof']/r['n']) for n in row],64//len(row)))
        im=ax.imshow(grid,aspect='auto',origin='upper',extent=[0,.5,6.5,.5],cmap='RdBu_r',norm=TwoSlopeNorm(vmin=.90,vcenter=1,vmax=1.10))
        ax.set(title=title,xlabel='Frequency (cycles per serialized weight)',ylabel='Recursion level');ax.set_yticks(range(1,7),['1: 2 bands','2: 4 bands','3: 8 bands','4: 16 bands','5: 32 bands','6: 64 bands'])
    fig.colorbar(im,ax=axes,shrink=.8,label='Energy density relative to a flat spectrum (1.0)')
    fig.suptitle('Recursive Fourier decomposition of the entire weight waveform',fontsize=17,fontweight='bold')
    fig.supxlabel('Every level partitions the same complete spectrum. The color scale magnifies small differences; this is not functional importance.',fontsize=9,color=gray)
    savefig(fig,'recursive_spectrum')
    with np.load(OUT/'component_excerpt.npz') as z:
        components=z['components'][:,:128];original=z['original'][:128];summed=z['summed'][:128];index=z['indices'][:128]-z['indices'][0]
    fig,axes=plt.subplots(3,1,figsize=(12,9),layout='constrained',sharex=True)
    for ax,count in zip(axes,[2,4,8]):
        grouped=components.reshape(count,8//count,-1).sum(1)
        for i,c in enumerate(grouped):ax.plot(index,c,lw=.7,alpha=.9,label=f'Band {i+1}')
        ax.plot(index,grouped.sum(0),color='#111827',lw=1.3,alpha=.85,label='Sum')
        ax.set(title=f'{count} component curves — recursion level {int(np.log2(count))}',ylabel='Weight update');ax.grid(alpha=.15)
        ax.legend(ncol=count+1,fontsize=8,loc='upper right')
    axes[-1].set_xlabel('Consecutive weight index in a fixed 128-weight excerpt')
    fig.suptitle('The component curves add back to the learned update',fontsize=18,fontweight='bold')
    fig.supxlabel('Each curve comes from the full 99,122,688-value global FFT. Excerpt: retained model, block 4, attention Q. Black sums coincide.',fontsize=9,color=gray)
    savefig(fig,'component_curves')

def final_report():
    from layer_report import build_layer_section
    layer_section,layer_summary=build_layer_section(load,savefig,img,table)
    results=load('interventions.json');comparisons=load('comparisons.json');budget=load('coefficient_budget.json');checks=load('functional_checks.json');g=load('global_spectra.json');targeting=load('targeting_control.json')
    by={(r['depth'],r['variant']):r for r in results};cmp={(r['depth'],r['variant']):r for r in comparisons}
    by[28,'gradient_spatial']=targeting;cmp[28,'gradient_spatial']={'NLL_recovery_fraction':targeting['NLL_recovery_fraction']}
    labels={'initial':'Before recovery','full_update':'Full learned update','norm_only':'Norm updates only','low_quarter':'Lowest frequencies','high_quarter':'Highest frequencies','random_quarter':'Random spectral bands','energy_quarter':'Largest-energy bands','gradient_quarter':'Gradient-selected bands','random_norm_matched':'Random bands, matched norm','random_spatial':'Random individual weights','gradient_shift_one':'Selected bands, shifted phase'}
    labels['gradient_spatial']='Guided individual weights*'
    order=['initial','norm_only','low_quarter','high_quarter','random_quarter','energy_quarter','gradient_quarter','random_norm_matched','random_spatial','gradient_spatial','gradient_shift_one','full_update']
    fig,axes=plt.subplots(1,2,figsize=(13,6.3),layout='constrained')
    ax=axes[0];values=[by[28,v]['splits']['audit']['perplexity_ratio_to_original'] for v in order]
    colors=[green if v=='gradient_quarter' else orange if v=='full_update' else red if v=='gradient_shift_one' else blue for v in order]
    ax.barh([labels[v] for v in order],np.array(values)-1,left=1,color=colors);ax.invert_yaxis();ax.axvline(1,color=gray,lw=1)
    for i,val in enumerate(values):ax.text(val+.004,i,f'{val:.3f}×',va='center',fontsize=9)
    ax.set(title='28 blocks: useful recovery from selected coefficients',xlabel='Perplexity relative to original 30-block model ↓',xlim=(1,max(values)+.08));ax.grid(axis='x',alpha=.15)
    ax=axes[1];keys=['initial','random_quarter','gradient_quarter','full_update'];xx=np.arange(len(keys));width=.35
    for depth,color,offset in [(28,blue,-width/2),(26,orange,width/2)]:
        vals=[by[depth,v]['splits']['audit']['perplexity_ratio_to_original'] for v in keys]
        ax.bar(xx+offset,np.array(vals)-1,bottom=1,width=width,color=color,label=f'{depth} blocks')
        for x,val in zip(xx+offset,vals):ax.text(x,val+.015,f'{val:.3f}',ha='center',fontsize=9)
    ax.set_xticks(xx,['Before\nrecovery','Random\nbands','Gradient\nbands','Full\nupdate'])
    ax.set(title='Same update transferred to a smaller depth',ylabel='Perplexity / original ↓',ylim=(1,max(by[26,v]['splits']['audit']['perplexity_ratio_to_original'] for v in keys)*1.09));ax.legend();ax.grid(axis='y',alpha=.15)
    fig.suptitle('Spectral update interventions · SmolLM2-135M',fontsize=18,fontweight='bold')
    fig.supxlabel('Existing audit: 8,192 targets, 32 documents. Sparse masks use about 25% of update degrees of freedom. *Exploratory control added afterward.',fontsize=9,color=gray)
    savefig(fig,'spectral_interventions')
    local=load('local_regions.json');masks=load('masks.json');key='4.0.self_attn.q_proj.weight';r=local[key]
    energy=np.array(r['energy']);descent=np.array(r['descent']);chosen=np.array(masks['gradient_quarter'][key]['bands'])
    fig,axes=plt.subplots(1,2,figsize=(12,5.2),layout='constrained')
    im=axes[0].imshow(energy/energy.mean(1,keepdims=True),aspect='auto',cmap='viridis');fig.colorbar(im,ax=axes[0],label='Energy / row-group mean')
    limit=float(np.max(abs(descent)));im=axes[1].imshow(descent,aspect='auto',cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit));fig.colorbar(im,ax=axes[1],label='Predicted descent contribution')
    for group,bands in enumerate(chosen):axes[1].scatter(bands,np.full(4,group),marker='s',s=70,facecolors='none',edgecolors='black',linewidths=1)
    for ax,title in zip(axes,['Update energy by region','Loss-gradient score; selected cells outlined']):ax.set(title=title,xlabel='Frequency leaf (16 leaves)',ylabel='Contiguous group of 32 matrix rows')
    fig.suptitle('Where to spend an update: block 4, attention Q',fontsize=17,fontweight='bold');savefig(fig,'localized_update_map')
    rows=[]
    for depth in [28,26]:
        for v in (order if depth==28 else keys):
            m=by[depth,v]['splits']['audit'];rows.append([depth,labels[v],f"{m['perplexity_ratio_to_original']:.6f}",f"{m['mean_kl']:.6f}",f"{100*cmp[depth,v]['NLL_recovery_fraction']:.1f}%"])
    primary=by[28,'gradient_quarter']['splits']['audit'];random=by[28,'random_quarter']['splits']['audit'];full=by[28,'full_update']['splits']['audit'];shift=by[28,'gradient_shift_one']['splits']['audit'];ci=cmp[28,'gradient_quarter']['paired_ci_vs_random'];share=budget['fraction']['gradient_quarter']
    recovery=cmp[28,'gradient_quarter']['NLL_recovery_fraction'];ratio=primary['perplexity_ratio_to_original']/random['perplexity_ratio_to_original']
    spectrum_reconstruction=g['retained_delta']['eight_component_relative_rms_error'];target_ci=targeting['paired_ci_vs_gradient_spectral']
    protocol=json.loads((ROOT/'protocol.json').read_text())
    report=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Recursive spectral updates</title><style>
body{{margin:0;background:#f6f7fb;color:#17243b;font:17px/1.6 system-ui,sans-serif}}main{{max-width:1120px;margin:auto;padding:32px 24px 64px}}h1{{font-size:36px;line-height:1.15}}h2{{margin-top:32px}}.card{{padding:18px 24px;background:white;border-radius:12px;margin:22px 0}}img{{width:100%;height:auto;margin:14px 0}}table{{border-collapse:collapse;width:100%;background:white;font-size:14px}}th,td{{padding:9px 12px;text-align:left;border-bottom:1px solid #dce2ec}}th{{background:#e8effa}}small{{color:#52627a}}a{{color:#1d4ed8}}pre{{white-space:pre-wrap;background:#e8effa;padding:16px;font-size:13px}}code{{font-size:14px}}</style></head><body><main>
<small>17 October 2026 · Recursive Fourier study · Post-hoc interventions</small><h1>Finding useful spectral updates</h1>
<div class="card"><strong>Measured result:</strong> gradient-guided masks retained {share*100:.2f}% of the update's real spectral degrees of freedom and recovered {recovery*100:.1f}% of the full update's audit NLL improvement. Perplexity was {primary['perplexity_ratio_to_original']:.3f}× the original 30-block model, versus {random['perplexity_ratio_to_original']:.3f}× for random spectral bands and {full['perplexity_ratio_to_original']:.3f}× for the full update. These masks were extracted from an already trained update. No sparse model was trained in this study.</div>
{img('spectral_interventions.png')}
<h2>What was done</h2><p>The entire decoder body was serialized as a one-dimensional waveform: block order, then alphabetically ordered tensor family, then row-major entries. A global real FFT was partitioned recursively into 2, 4, 8, 16, 32 and 64 frequency bands. The original 30-block checkpoint, both phase-6 28-block initial/final checkpoints, and both training updates were measured. The embedding, output head and final normalization were excluded.</p>
<p>A second, localized representation was used for functional interventions: FFTs along matrix columns, grouped into contiguous sets of 32 rows, within each of the six trained blocks and seven projection families. This preserves the layer and parameter map needed for targeted edits. It is distinct from masking the global flattened waveform. The two RMSNorm vectors were carried over in full for every spectral-mask arm; a norm-only control measures their effect.</p>
<p>Each row group was divided into 16 frequency bands. Four bands were retained by low frequency, high frequency, random selection, update energy, or predicted descent. DC and Nyquist coefficients were always included. The exact selected fraction varies from {min(budget['fraction'].values())*100:.2f}% to {max(budget['fraction'].values())*100:.2f}% because of discrete band edges.</p>
<h2>Recursive decomposition</h2>{img('recursive_spectrum.png')}
<p>The global spectra are broadly distributed. The highest-energy leaf of the retained-final waveform contains {100*max(n['fraction'] for n in g['retained_final']['nodes'] if n['level']==6):.3f}% of total energy; a uniform 64-way division would have 1.563% per leaf. This does not identify a small dominant spectral branch.</p>
<p>Changing the serialization of the same tensor entries from row-major to column-major changes the measured spectrum. Decoding either order recovers identical matrices. Frequency is therefore a property of the chosen coordinate ordering, not an intrinsic semantic address.</p>
{img('component_curves.png')}
<p>The eight global components sum back to the complete retained-model update with relative RMS error {spectrum_reconstruction:.3g} in FP32. The picture shows the first 128 values of a fixed 512-weight excerpt so the oscillations remain visible; each component was reconstructed from the full 99,122,688-value FFT. Coarser 2- and 4-component curves are sums of these children. No frequency was discarded in this reconstruction. Sample points are joined for display; even a smooth Fourier component can oscillate rapidly.</p>
{layer_section}
<h2>Local loss information separates useful regions</h2>{img('localized_update_map.png')}
<p>Let ΔW be the complete learned update and G the gradient of teacher-to-student KL at the unadapted 28-block model. A region's score is −⟨G, ΔW<sub>region</sub>⟩, computed using the complex Fourier coefficients and correct real-spectrum endpoint weights. Positive scores predict a local loss decrease. Scores came only from the first eight training windows (1,024 targets), before validation/audit evaluation. This is a first-order approximation, not an assertion that contributions add independently after finite updates.</p>
<p>All masks were fixed before scoring the evaluation sets. The gradient-selected mask's perplexity relative to random bands was {ratio:.6f}×. The paired document-bootstrap 95% interval for NLL(selected) − NLL(random) was [{ci['lower']:.6f}, {ci['upper']:.6f}] nats. This is conditional on one training run and a reused audit; it does not measure seed uncertainty.</p>
<h2>Functional results</h2>{table(['Blocks','Intervention','PPL / original ↓','Teacher KL ↓','Full-update NLL improvement retained'],rows)}
<p>Recovery fraction is [NLL(initial) − NLL(masked)] / [NLL(initial) − NLL(full update)] at the same depth. It can be negative or exceed 100%; it is not the percentage of original-model quality restored. The reference model remains the original 30-block model in every PPL ratio.</p>
<p>The 26-block test kept the endpoints of each edited triplet, corresponding to original layers 4/7 and 12/15. It directly transferred the 28-block masks and learned updates without retraining, interpolation, or residual rescaling. It tests further block removal, not arbitrary continuous-depth resampling.</p>
<h2>Phase and magnitude controls</h2><p>Random spectral masks were also scaled per matrix to match the selected update's norm. A random individual-weight mask retained exactly the selected spectral mask's real degrees of freedom per matrix. These controls distinguish update scale and generic sparsity from frequency placement.</p>
<p>The phase control circularly shifted every row of the selected update by one column. Its Fourier magnitudes and update norms are preserved; only phases change. Its audit perplexity was {shift['perplexity_ratio_to_original']:.6f}× original versus {primary['perplexity_ratio_to_original']:.6f}× for the unshifted mask. A power spectrum alone cannot distinguish these updates.</p>
<h2>Exploratory control: is the benefit specifically Fourier?</h2><p>After the primary outcomes were available, one additional control applied the same −gradient×dense-update score to individual weight coordinates. Selection was within the same 32-row regions and retained exactly the same number of real degrees of freedom as the selected spectral mask, with identical full norm updates. This addendum is recorded separately and is not represented as part of the original prespecified comparison.</p>
<p>Guided individual weights recovered {100*targeting['NLL_recovery_fraction']:.1f}% of the full update's NLL improvement, at {targeting['splits']['audit']['perplexity_ratio_to_original']:.6f}× original perplexity. The paired NLL difference, spatial minus spectral, was {targeting['delta_nll_vs_gradient_spectral']:.6f} nats with interval [{target_ci['lower']:.6f}, {target_ci['upper']:.6f}]. The interval includes zero. This run supports loss-guided selection over random selection but does not establish a reliable advantage for Fourier over equally guided individual-weight selection.</p>
<h2>Verification and limits</h2><p>The two phase-6 reference models reproduce their previously recorded audit NLL/KL within 2×10⁻⁵. Every original model parameter hash is unchanged. Local forward/inverse Fourier and gradient Parseval checks passed. All intervention logits were finite. The global FFT collection and component reconstruction took {load('global_environment.json')['total_seconds']:.1f} seconds; functional preparation and scoring took {checks['total_seconds']:.1f} seconds. These are diagnostic costs, not inference-speed improvements.</p>
<p>These are the existing phase-6 sets: 32 validation windows and 64 audit windows across 32 documents, 128 targets each. The audit has been reported previously and is not a new independent benchmark. There is one source training seed and only six adapted blocks. All candidate updates inherit information from the full phase-6 training run. The selected coefficients have not been trained independently, so this demonstrates update sufficiency rather than saved training compute or a new sparse-training algorithm.</p>
<h2>Next experiment</h2><p>Train sparse spectral coordinates starting from zero update on the same base model. A deployable selector must use current gradients alone (or a short accounted-for training pilot); the present gradient-times-dense-update score knows the finished dense solution and is an optimistic reference. Compare current-gradient spectral selection, random spectral, and equally guided individual-weight supports at matched trainable count, initial function, data and tuning budget. Use multiple seeds and an untouched audit. Refresh selected regions only on training data and account for the cost of scoring them. Evaluate the same masks at 28 and 26 blocks; keep depth-conditioned selection separate from direct transfer.</p>
<p>For regions showing useful effects, refine their frequency tree while preserving a global coefficient budget. Compare parent-band interventions against their children before increasing resolution. This directly tests whether recursive allocation helps more than a fixed-resolution mask. Native versus curve-coefficient training is still a separate experiment.</p>
<h2>Related work</h2><p><a href="https://arxiv.org/abs/2209.04836">Git Re-Basin (ICLR 2023)</a> studies alignment modulo function-preserving hidden-unit permutations. That motivates alignment controls; it does not establish that different depth layers are equivalent functions. </p><p><a href="https://arxiv.org/abs/2405.03003">FourierFT (ICML 2024)</a> learns sparse Fourier coefficients for matrix updates. It supports the general direction but uses a different update construction and does not establish the results above. <a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ShortTimeFFT.html">SciPy's ShortTimeFFT documentation</a> describes localized Fourier analysis and invertible overlap-add reconstruction; that is a future tapered-window control, whereas this study uses explicit matrix rows and non-overlapping row groups.</p>
<h2>Reproduce</h2><p>Recover the existing phase-6 archive and pinned model. The code, protocol, masks, spectra, source hashes, plots and per-window results are included in this small companion bundle. The large model and teacher caches are not duplicated.</p>
<pre>python global_probe.py --phase6 /path/to/smollm_initialization_phase6
python interventions.py --phase6 /path/to/smollm_initialization_phase6
python targeting_control.py --phase6 /path/to/smollm_initialization_phase6
python layer_permutation_probe.py --phase6 /path/to/smollm_initialization_phase6
python build_report.py</pre><details><summary>Frozen primary protocol</summary><pre>{html.escape(json.dumps(protocol,indent=2))}</pre></details></main></body></html>'''
    (OUT/'recursive_spectral_report.html').write_text(report)
    summary={'gradient_selected_coefficient_fraction':share,'gradient_selected_recovery_fraction':recovery,'audit_gradient_ppl_ratio':primary['perplexity_ratio_to_original'],'audit_random_ppl_ratio':random['perplexity_ratio_to_original'],'audit_full_ppl_ratio':full['perplexity_ratio_to_original'],'gradient_vs_random_relative_ppl':ratio,'gradient_vs_random_nll_ci':ci,'scope':'Post-hoc update sufficiency, not sparse-model training','depth_transfer':{v:by[26,v]['splits']['audit']['perplexity_ratio_to_original'] for v in keys}}
    summary['layer_permutation_extension']=layer_summary
    summary['exploratory_guided_spatial']={'recovery_fraction':targeting['NLL_recovery_fraction'],'audit_ppl_ratio':targeting['splits']['audit']['perplexity_ratio_to_original'],'delta_nll_vs_spectral':targeting['delta_nll_vs_gradient_spectral'],'paired_ci':target_ci}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (ROOT/'README.txt').write_text('Recursive spectral update sufficiency study, 17 October 2026.\nOpen results/recursive_spectral_report.html.\nThis is post-hoc masking of an already trained model update, not a new training run.\nThe global waveform FFT and localized intervention FFT use explicitly different axes; see protocol.json.\nRequires the existing smollm_initialization_phase6 bundle and pinned original model.\nRun global_probe.py --phase6 PATH, interventions.py --phase6 PATH, targeting_control.py --phase6 PATH, layer_permutation_probe.py --phase6 PATH, then build_report.py.\nThe targeting control is an exploratory addendum after the primary outcomes; see protocol_addendum.json.\nThe layer/permutation extension is exploratory and has its own frozen protocol. It does not align different layers.\nNo large model checkpoints or teacher caches are duplicated here.\n')
    (ROOT/'requirements.txt').write_text('--extra-index-url https://download.pytorch.org/whl/cpu\ntorch==2.5.1+cpu\ntransformers==4.46.3\nsafetensors==0.4.5\ntokenizers==0.20.3\nhuggingface-hub==0.36.2\nnumpy==2.3.5\nscipy==1.17.0\nmatplotlib==3.10.8\n')
    files=[p for p in ROOT.rglob('*') if p.is_file() and 'cache' not in p.relative_to(ROOT).parts and '__pycache__' not in p.parts and p.name!='SHA256SUMS']
    (ROOT/'SHA256SUMS').write_text('\n'.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(ROOT)) for p in sorted(files))+'\n')
    archive=ROOT.parent/'smollm_recursive_spectral_2026-10-17.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(files+[ROOT/'SHA256SUMS']):z.write(p,p.relative_to(ROOT.parent))
    print(json.dumps({'report':str(OUT/'recursive_spectral_report.html'),'archive':str(archive),'bytes':archive.stat().st_size,'summary':summary}))

if __name__=='__main__':
    spectral_figures()
    if (OUT/'functional_checks.json').exists():final_report()
    else:print('Spectral figures rendered; functional tests still running.')
