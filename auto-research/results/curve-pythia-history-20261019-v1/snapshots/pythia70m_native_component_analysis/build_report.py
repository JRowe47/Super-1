import argparse,base64,hashlib,html,json,pathlib,zipfile,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from insitu import *

BLUE='#246aa5';TEAL='#17867d';RED='#ba4957';GRAY='#64748b'
plt.rcParams.update({'font.size':10,'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,
 'axes.titleweight':'bold','axes.edgecolor':'#bfc9d4','grid.alpha':.22,'text.color':'#17263d','savefig.facecolor':'white'})
def load(n):return json.loads((OUT/n).read_text())
def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=160,bbox_inches='tight');fig.savefig(OUT/(name+'.svg'),bbox_inches='tight');plt.close(fig)
def pic(n):return '<img alt="'+n.replace('_',' ')+'" src="data:image/png;base64,'+base64.b64encode((OUT/(n+'.png')).read_bytes()).decode()+'">'
def table(headers,rows):return '<div class="table"><table><tr>'+''.join('<th>'+str(x)+'</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in r)+'</tr>' for r in rows)+'</table></div>'

def make_case(checkpoints):
    tracks=load('component_trajectories.json')
    tr=next(r for r in tracks if r['layer']==3 and r['tensor']=='attention.query_key_value.weight' and r['basis']=='haar' and r['level']==6 and r['selected_interval']==[512,1000])
    data=np.stack([tensor(checkpoints,s,3,tr['tensor']).ravel()[tr['flat_start']:tr['flat_end_exclusive']] for s in STEPS])
    case={'steps':STEPS,'address':tr,'curves':data.astype(float).tolist(),'selection':'Largest absolute level-6 Haar coefficient change within layer3 QKV between step512 and step1000. Descriptive selection after observing changes.'}
    dump('native_window_case.json',case)
    before=data[STEPS.index(512)].astype(float);after=data[STEPS.index(1000)].astype(float)
    fig=plt.figure(figsize=(13,8.4),constrained_layout=True);gs=fig.add_gridspec(3,4)
    axes=[fig.add_subplot(gs[0,:]),fig.add_subplot(gs[1,:2]),fig.add_subplot(gs[1,2:])]+[fig.add_subplot(gs[2,i]) for i in range(4)]
    checks=[];curves=[]
    for ax,path in zip(axes,['','a','d','aa','ad','da','dd']):
        depth=len(path);pair=[]
        for x in [before,after]:
            nodes=packet(x,depth)
            only={k:(v if k==path else np.zeros_like(v)) for k,v in nodes.items()}
            rec=ipacket(only);pair.append(rec)
            checks.append(float(np.max(np.abs(ipacket(nodes)-x))))
        a,b=pair;curves.append({'path':path,'before':a.tolist(),'after':b.tolist(),'delta':(b-a).tolist()})
        ax.plot(np.arange(64),a,color=BLUE,lw=1,label='Step 512')
        ax.plot(np.arange(64),b,color=TEAL,lw=1,label='Step 1,000')
        ax.plot(np.arange(64),b-a,color=RED,lw=1.1,label='Signed change')
        ax.axhline(0,color=GRAY,lw=.5);ax.set_xlim(0,63);ax.grid(axis='y')
        ax.set_title('Native 64-weight curve' if not path else f'Component {path} · recursion depth {depth}',fontsize=10,loc='left')
        if depth==2:ax.set_xlabel('Native offset within the 64-entry interval')
        ax.set_ylabel('Weight contribution')
    axes[0].legend(ncol=3,frameon=False,fontsize=9)
    fig.suptitle('The same native curve, recursively decomposed',fontsize=17,fontweight='bold')
    fig.supxlabel('Layer 3 · QKV weight row 219, columns 448–511 · children sum back to their parent at every point',fontsize=11)
    save(fig,'native_recursive_components')
    dump('packet_example.json',{'curves':curves,'max_parent_reconstruction_error':max(checks),'native_window_length':64})
    ftr=next(r for r in tracks if r['layer']==3 and r['tensor']==tr['tensor'] and r['basis']=='fourier' and r['selected_interval']==[512,1000])
    htracks=[next(r for r in tracks if r['layer']==3 and r['tensor']==tr['tensor'] and r['basis']=='haar' and r['level']==j and r['selected_interval']==[512,1000]) for j in [1,6,9]]
    fig,ax=plt.subplots(1,3,figsize=(13,4.5),constrained_layout=True)
    ax[0].plot(ftr['real'],ftr['imag'],'-o',color=BLUE,ms=4)
    for s in [0,512,1000,143000]:
        i=STEPS.index(s);ax[0].annotate(str(s),(ftr['real'][i],ftr['imag'][i]),xytext=(5,4),textcoords='offset points',fontsize=8)
    ax[0].axhline(0,color=GRAY,lw=.6);ax[0].axvline(0,color=GRAY,lw=.6)
    ax[0].set_xlabel('Real coefficient');ax[0].set_ylabel('Imaginary coefficient');ax[0].set_title(f'Fourier mode {ftr["index"]}',loc='left',fontsize=11)
    x=np.maximum(STEPS,32)
    ax[1].plot(x,ftr['amplitude'],'-o',color=BLUE,ms=3,label='Amplitude')
    ax2=ax[1].twinx();ax2.spines['right'].set_visible(True);ax2.plot(x,ftr['phase'],'--o',color=RED,ms=3,label='Wrapped phase')
    ax[1].set_xscale('log');ax[1].set_xlabel('Checkpoint (0 plotted at 32)');ax[1].set_ylabel('Amplitude',color=BLUE);ax2.set_ylabel('Phase (radians)',color=RED)
    ax[1].set_title('Amplitude and phase both evolve',loc='left',fontsize=11);ax[1].grid(axis='y')
    for r,c in zip(htracks,[BLUE,TEAL,RED]):
        ax[2].plot(x,r['values'],'-o',ms=3,color=c,label=f'Level {r["level"]}, index {r["index"]}')
    ax[2].set_xscale('log');ax[2].set_xlabel('Checkpoint (0 plotted at 32)');ax[2].set_ylabel('Signed wavelet coefficient')
    ax[2].set_title('Fixed native components over training',loc='left',fontsize=11);ax[2].legend(fontsize=7.5,frameon=False);ax[2].grid(axis='y')
    fig.suptitle('Individual component trajectories reveal changes hidden by broad summaries',fontsize=15,fontweight='bold')
    save(fig,'native_component_trajectories')
    scales=load('wavelet_scales.json');fig,axes=plt.subplots(2,3,figsize=(13,7.5),constrained_layout=True)
    for l,ax in enumerate(axes.flat):
        grid=np.array([[next(r['rms_relative_to_tensor'] for r in scales if r['layer']==l and r['tensor']==tr['tensor'] and r['kind']=='detail' and r['level']==j and r['end']==s) for s in STEPS[1:]] for j in range(1,19)])
        im=ax.imshow(np.log2(np.maximum(grid,1e-6)),aspect='auto',origin='lower',cmap='coolwarm',vmin=-2,vmax=2)
        ax.set_yticks([0,5,8,11,14,17],[1,6,9,12,15,18]);ax.set_ylabel('Haar detail level')
        ax.set_xticks(range(12),[str(s) if s<1000 else f'{s//1000}k' for s in STEPS[1:]],rotation=55,fontsize=8)
        ax.set_title(f'Layer {l} · native QKV',loc='left',fontsize=11)
    fig.colorbar(im,ax=axes,shrink=.8,label='log₂ (change RMS per coefficient / whole-tensor change RMS)')
    fig.suptitle('Where changes occur across recursive scales',fontsize=16,fontweight='bold')
    fig.supxlabel('End of sampled interval; each coefficient at every scale is retained in the full data')
    save(fig,'native_multiscale_changes')
    return case,ftr,htracks

EXPLORER_JS=r'''
const D=CASE_DATA;
const beforeSel=document.querySelector('#before'),afterSel=document.querySelector('#after'),depthEl=document.querySelector('#depth'),pathSel=document.querySelector('#path');
for(const s of D.steps){for(const e of [beforeSel,afterSel]){const o=document.createElement('option');o.value=s;o.textContent=s.toLocaleString();e.append(o);}}
beforeSel.value=512;afterSel.value=1000;
function packet(x,depth){let m={'':x};for(let d=0;d<depth;d++){const q={};for(const [k,a] of Object.entries(m)){q[k+'a']=[];q[k+'d']=[];for(let i=0;i<a.length;i+=2){q[k+'a'].push((a[i]+a[i+1])/Math.SQRT2);q[k+'d'].push((a[i]-a[i+1])/Math.SQRT2);}}m=q;}return m;}
function inv(m){while(Object.keys(m)[0].length){let q={};for(const p of [...new Set(Object.keys(m).map(k=>k.slice(0,-1)))]){let a=m[p+'a'],d=m[p+'d'],x=[];for(let i=0;i<a.length;i++){x.push((a[i]+d[i])/Math.SQRT2,(a[i]-d[i])/Math.SQRT2);}q[p]=x;}m=q;}return m[''];}
function component(x,depth,path){let m=packet(x,depth);for(const k in m)if(k!==path)m[k]=m[k].map(()=>0);return inv(m);}
function paths(){pathSel.innerHTML='';for(let k of Object.keys(packet(D.curves[0],+depthEl.value))){let o=document.createElement('option');o.value=k;o.textContent=k||'whole curve';pathSel.append(o);}document.querySelector('#depthLabel').textContent=depthEl.value;draw();}
function draw(){let d=+depthEl.value,p=pathSel.value;let a=component(D.curves[D.steps.indexOf(+beforeSel.value)],d,p),b=component(D.curves[D.steps.indexOf(+afterSel.value)],d,p),delta=b.map((v,i)=>v-a[i]);let all=[...a,...b,...delta,0],lo=Math.min(...all),hi=Math.max(...all),span=hi-lo||1;lo-=span*.1;hi+=span*.1;let W=900,H=300,L=64,R=20,T=20,B=45,X=i=>L+(W-L-R)*i/63,Y=v=>T+(H-T-B)*(hi-v)/(hi-lo);
const esc=s=>s.replace(/&/g,'&amp;').replace(/</g,'&lt;');let svg=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Native component before, after and change"><rect width="900" height="300" fill="white"/>`;
for(let j=0;j<5;j++){let v=lo+(hi-lo)*j/4,y=Y(v);svg+=`<line x1="${L}" x2="${W-R}" y1="${y}" y2="${y}" stroke="#e2e8f0"/><text x="${L-8}" y="${y+4}" text-anchor="end" font-size="11" fill="#475569">${v.toFixed(4)}</text>`;}
for(const [v,c] of [[a,'#246aa5'],[b,'#17867d'],[delta,'#ba4957']]){let pts=v.map((q,i)=>`${X(i)},${Y(q)}`).join(' ');svg+=`<polyline points="${pts}" fill="none" stroke="${c}" stroke-width="1.7"/>`;}
svg+=`<text x="450" y="292" text-anchor="middle" font-size="12">Native offsets 0–63; component ${esc(p||'root')}</text></svg>`;document.querySelector('#explorerPlot').innerHTML=svg;
document.querySelector('#componentCount').textContent=`Depth ${d}: ${2**d} components, each with ${64/2**d} coefficients. All components sum to the original 64 samples.`;}
depthEl.addEventListener('input',paths);for(const e of [beforeSel,afterSel,pathSel])e.addEventListener('change',draw);paths();
'''

def main(checkpoints):
    case,ftr,htracks=make_case(checkpoints);env=load('analysis_environment.json');changes=load('fourier_component_changes.json')
    eligible=[r for r in changes if r['end']==1000 and r['tensor'].endswith('weight') and 'layernorm' not in r['tensor']]
    phase=sum(r['phase_change_energy'] for r in eligible)/sum(r['complex_change_energy'] for r in eligible)
    i0=STEPS.index(512);i1=STEPS.index(1000);phase_delta=float(np.angle(np.exp(1j*(ftr['phase'][i1]-ftr['phase'][i0]))))
    source_tables=[['Sparse Feature Circuits','Sparse, interpretable activation features can be linked into causal circuits for particular behaviors; the released code includes Pythia-70M examples.','https://arxiv.org/abs/2403.19647'],
      ['Towards Monosemanticity','Dictionary learning extracts interpretable directions from superposed activations. This is an activation representation, distinct from our fixed weight-coordinate transform.','https://transformer-circuits.pub/2023/monosemantic-features/index.html'],
      ['Sparse Crosscoders','Jointly learned dictionaries can track features across layers or models; useful context for connecting checkpoint component changes to representational changes. The original note reports mixed model-diffing results.','https://transformer-circuits.pub/2024/crosscoders/index.html'],
      ['Wavelet packet decomposition','Both approximation and detail branches can be recursively decomposed, giving a hierarchy of reconstructible components.','https://pywavelets.readthedocs.io/en/latest/ref/wavelet-packets.html']]
    summary={'orientation':'Native-coordinate component tracking and localization, not compression or reordering.',
      'checkpoints':13,'layers':6,'tensors':72,'learned_decoder_parameters':env['parameter_count'],
      'full_resolution_coefficients_retained':True,'training_steps':0,'permutations':0,
      'phase_fraction_of_matrix_change_512_to_1000':phase,
      'example_fourier_mode':{'layer':3,'tensor':ftr['tensor'],'index':ftr['index'],
        'amplitude_512':ftr['amplitude'][i0],'amplitude_1000':ftr['amplitude'][i1],'phase_change_radians':phase_delta},
      'example_local_component':case['address'],
      'catalog_records':len(load('component_target_catalog.json')),
      'limitations':'Descriptive component changes are candidate targeting information, not demonstrated causal or semantic feature identities. Global Fourier bins are not spatially localized. Haar locations are locations in the native parameter curve, not semantic or token positions.'}
    dump('summary.json',summary)
    functional=load('functional_baseline.json');exact=[r for r in functional['checks'] if r['coefficient_precision']==64]
    assert all(r['max_abs_logit_error']==0 for r in exact)
    rows=[]
    for r in htracks:
        rows.append([f"Haar level {r['level']}, coefficient {r['index']}",f"{r['flat_start']:,}–{r['flat_end_exclusive']-1:,}",f"{r['values'][i0]:+.6f}",f"{r['values'][i1]:+.6f}"])
    report=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Pythia native curve component atlas</title><style>
body{{margin:0;background:#eef3f6;font:16px/1.65 system-ui,sans-serif;color:#18263d}}main{{max-width:1080px;margin:auto;padding:40px 44px 65px;background:white}}h1{{font-size:36px;line-height:1.2;margin:.3em 0}}h2{{font-size:24px;margin-top:2em}}.eyebrow{{color:#17867d;font-size:13px;font-weight:700;letter-spacing:.06em}}.lead{{font-size:20px;color:#465d76}}img{{width:100%;height:auto;display:block;margin:22px 0}}.callout{{border-left:4px solid #17867d;background:#eef8f5;padding:16px 20px}}.table{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:10px;border-bottom:1px solid #d9e3eb;text-align:left}}th{{background:#edf3f8}}.small{{color:#607086;font-size:13px}}pre{{background:#17263d;color:#e5f3fa;padding:18px;border-radius:6px;font-size:13px;white-space:pre-wrap}}a{{color:#246aa5}}.controls{{display:flex;gap:18px;flex-wrap:wrap;align-items:center;background:#edf3f7;padding:15px}}select,input{{font:inherit;padding:4px;max-width:180px}}label{{font-size:14px}}.legend{{color:#64748b;font-size:13px}}@media(max-width:700px){{main{{padding:24px 16px}}h1{{font-size:29px}}}}
</style></head><body><main><div class="eyebrow">GEOMETRIC SELF ATTENTION · NATIVE COMPONENT ANALYSIS · 17 OCTOBER 2026</div>
<h1>The network in situ:<br>follow the components as training changes them</h1>
<p class="lead">An exact functional baseline, followed by full-resolution Fourier and wavelet comparisons in the network's existing coordinates.</p>
<div class="callout"><strong>Scope correction.</strong> The question here is which components change, how they change, at which recursive level, and where they sit in the native layer curve. Broad spectral concentration and ordering experiments did not answer that question. The earlier negative concentration result should not be read as evidence against component-level targeting.</div>
<p>This pass covers <strong>13 checkpoints, all six decoder blocks, all 72 learned block tensors, and {env['parameter_count']:,} parameters per checkpoint</strong>. Every complex Fourier coefficient and every signed Haar coefficient is retained at every checkpoint. No neurons are reordered. No coefficients are pruned. No training is performed.</p>
{pic('native_recursive_components')}
<h2>What the curve baseline means</h2>
<p>Each tensor retains its exact shape and native C-order scalar coordinates. A layer is represented by a fixed list of these tensor curves, with an explicit offset-to-parameter map. Transform boundaries stop at tensor boundaries. Query, key and value rows remain in GPT-NeoX's original packed order. Biases and normalization parameters are included.</p>
<p>A full Fourier encoding reconstructs the sampled weight curve; the original layer operations then consume those reconstructed parameters. The original nonlinearities, residual connections, attention equations and depth remain unchanged. We are not claiming that a scalar weight waveform alone replaces the layer's computation, or that samples between existing layers define a valid new depth.</p>
<p><strong>The functional baseline passes:</strong> at steps 1,000 and 143,000, reconstructing all six blocks with 64-bit Fourier or Haar coefficients produces <strong>exactly matching tested float32 logits and next-token choices</strong>. The saved step-1,000 full-precision decoder-curve archive can be loaded back into the pinned model.</p>
<p class="small">Precision is distinct from resolution. The complete time-series archives store complex64 / float32 analysis coefficients. At the final checkpoint, decoding those rounded coefficients changed 1 of 520 tested argmaxes despite a relative logit RMS error below 4×10⁻⁶. They are appropriate for component analysis but should not be substituted for the full-precision functional baseline. The full-precision baseline and the source-checkpoint regeneration path are provided explicitly.</p>
<h2>How the recursion is defined</h2>
{table(['View','Component identity','What it localizes'],[
 ['Fourier hierarchy','Layer, tensor, frequency interval; descend to an individual complex bin','Frequency within that tensor curve. A global Fourier component extends over the whole tensor.'],
 ['Haar multilevel wavelets','Layer, tensor, detail level j, position k','An exact native interval [k·2ʲ, (k+1)·2ʲ), down to two-entry details.'],
 ['Wavelet packet example','Native window, branch path a/d, coefficient position','Recursive splitting of both approximation and detail components, while preserving reconstruction.']])}
<p>The main Haar transform continues to the largest exactly divisible depth, without padding: up to 20 levels for the large MLP matrices. Packed QKV matrices end with three coarsest scaling coefficients. All fine-scale coefficients remain present. The Fourier tree can be refined to individual frequency bins using the extraction script; there is no fixed 64-band resolution limit.</p>
<p>For recursive Fourier analysis, a parent component is split into disjoint child-frequency ranges, each retaining its original complex phase and reconstructible curve. Simply applying FFT again to the coefficient vector would not create an independent multiscale hierarchy.</p>
<h2>Inspect a recursive component across training</h2>
<p>The example below follows the same 64 native weights through all 13 checkpoints: layer 3, QKV row 219, columns 448–511. This lies in head 1's query projection, row 27 within that projection. It was selected descriptively as the largest level-6 Haar change in this tensor during steps 512→1,000, not because its semantic function is known.</p>
<div class="controls"><label>Before <select id="before"></select></label><label>After <select id="after"></select></label><label>Packet depth <input id="depth" type="range" min="0" max="6" value="2"><span id="depthLabel">2</span></label><label>Component <select id="path"></select></label></div>
<div class="legend"><span style="color:#246aa5">Blue: before</span> · <span style="color:#17867d">Green: after</span> · <span style="color:#ba4957">Red: signed change</span></div><div id="explorerPlot"></div><p id="componentCount" class="small"></p>
<p class="small">Every displayed point corresponds to an original native coordinate. The full network atlas is retained separately; this browser control is one 64-entry microscope view. If the viewer disables scripts, the static figure above shows the same before/after decomposition through two recursive levels.</p>
<h2>Individual components reveal concrete, addressable changes</h2>
{pic('native_component_trajectories')}
<p>For native layer-3 QKV Fourier mode <strong>{ftr['index']:,}</strong>, amplitude changes from <strong>{ftr['amplitude'][i0]:.6f}</strong> to <strong>{ftr['amplitude'][i1]:.6f}</strong> between steps 512 and 1,000, while phase changes by <strong>{phase_delta:.3f} radians</strong>. Its later trajectory is not a simple monotonic strengthening. Those are specific complex-coefficient changes that a broad band-energy view does not describe.</p>
<p>The localized Haar components below have fixed native supports. They supply exact addresses for later component-wise tracing or masking:</p>
{table(['Native layer-3 QKV component','Original flat-coordinate support','Step 512','Step 1,000'],rows)}
<p>The level-6 component's signed coefficient changes from −0.028526 to −0.114385, and reaches −0.264832 by the final checkpoint. Within its 64-entry support, its 512→1,000 change contributes approximately −0.010732 to the first half and +0.010732 to the second half; the other components describe the remaining fine structure. This is a concrete targeting candidate, not a claim that it is an independent semantic feature.</p>
<p>Across the decoder matrices for 512→1,000, <strong>{phase*100:.1f}%</strong> of squared complex-coefficient movement is the phase-dependent term in the identity below. This is a geometric accounting identity in the fixed native Fourier basis, not a percentage of learned capability:</p>
<pre>|B·exp(iφ₁) − A·exp(iφ₀)|²
  = (B − A)² + 2AB·[1 − cos(φ₁ − φ₀)]</pre>
<p>Thus amplitude-only summaries omit substantial information about changes. The previous study's reconstructions retained phase, but its aggregate energy summaries did not track the individual phase trajectories that are recorded here.</p>
{pic('native_multiscale_changes')}
<p>The overview above indexes changes by layer, recursive level and training interval. It is a navigation map; the underlying coefficient arrays retain every position. The catalog contains {summary['catalog_records']:,} records pointing to the largest changes within each analyzed level and interval, while the full files allow inspecting any other component. Large movement is not a causal-importance score.</p>
<h2>How the sparse-feature literature connects</h2>
{table(['Primary source','Relevance'],[[f'<a href="{u}">{t}</a>',d] for t,d,u in source_tables])}
<p>The useful bridge is to test whether changes in a feature's activation or behavior track particular native curve components or groups of components across training. Sparse-feature work supports looking below whole layers and heads. It does not establish a one-to-one identification between a semantic feature and a Fourier or wavelet coefficient, or imply that an entire language model contains only a small number of features. We have not trained feature dictionaries in this pass.</p>
<h2>Files and reproducibility</h2>
<ul><li><strong>Full-precision functional baseline:</strong> <code>pythia70m_step1000_exact_decoder_curves.npz</code>, complex128, all six decoder blocks.</li>
<li><strong>Full-resolution time series:</strong> six <code>pythia70m_native_layerN_coefficients.npz</code> files, one per layer, with all coefficients for all 13 checkpoints and embedded tensor/scale metadata.</li>
<li><strong>Component index:</strong> <code>component_target_catalog.json</code>, <code>component_trajectories.json</code>, <code>tensor_mapping.json</code>, and scale/phase summaries.</li>
<li><strong>Exact local example:</strong> <code>native_window_case.json</code> contains all 64 original samples at each checkpoint. Packet children sum back to their parent at every point; maximum checked error is {load('packet_example.json')['max_parent_reconstruction_error']:.2g}.</li></ul>
<pre># Any single full-resolution Fourier component:
python inspect_components.py --layer 3 \
  --tensor attention.query_key_value.weight --start 512 --end 1000 \
  --basis fourier --low 334800 --high 334801 --output mode.npz

# Any individual localized detail component:
python inspect_components.py --layer 3 \
  --tensor attention.query_key_value.weight --start 512 --end 1000 \
  --basis haar --level 6 --index 1759 --output local_component.npz</pre>
<p>Both outputs contain before, after and signed-difference curves on the original sample grid. For a Fourier parent node, widen the half-open frequency interval; its disjoint children sum to the parent. Reproduction scripts accept <code>--checkpoints PATH</code>. Model revisions and source hashes are pinned in <code>checkpoint_manifest.json</code>.</p>
<h2>Research direction retained from the correction</h2>
<p>Keep the native coordinates fixed. Follow component trajectories, parent/child structure, local support and phase changes. Connect those observations to unchanged-network activation traces and, later, to narrowly targeted component interventions. Layer depth remains a separate coordinate to explore when a depth-parameterized curve model exists; it is not being inferred from the arbitrary serialization of tensor entries.</p>
<p class="small">Scope: one Pythia training seed, fixed architecture, published checkpoint intervals, descriptive selection of example components. No permutation search, coefficient-budget competition, retraining, depth editing, or semantic-feature identification was performed in this pass.</p>
<script>const CASE_DATA={json.dumps(case,separators=(',',':'))};
{EXPLORER_JS}</script></main></body></html>'''
    (ROOT/'pythia70m_native_component_report.html').write_text(report)
    (ROOT/'README.txt').write_text('Pythia native-coordinate component atlas, 17 October 2026\n\nOpen pythia70m_native_component_report.html.\n\nThe user corrected the research scope: preserve native order; convert each layer to an exact functional curve baseline; examine full-resolution recursively decomposed components and changes across training. Do not substitute permutation alignment or compression benchmarks for this objective.\n\nReproduction:\npython -m pip install -r requirements.txt\npython download_checkpoints.py\npython insitu.py --checkpoints ./checkpoints\npython baseline_check.py --checkpoints ./checkpoints\npython make_exact_baseline.py --checkpoints ./checkpoints\npython build_report.py --checkpoints ./checkpoints\n\nSet OPENBLAS_NUM_THREADS=1 for these analysis runs.\nThe separately supplied coefficient NPZ files belong in coefficients/.\nUse inspect_components.py for any full-resolution Fourier node or individual Haar detail.\nThe full-precision step1000 decoder baseline can be restored with make_exact_baseline.restore_decoder_from_curves(model, path), using the pinned step1000 model for the unchanged architecture, embeddings, final norm and output head.\n\nFull resolution and numerical precision are distinct: time-series files retain every coordinate at float32/complex64 precision. Use the complex128 baseline for function-preserving execution. See the report for the measured precision check.\n')
    files=[p for p in ROOT.rglob('*') if p.is_file() and not {'coefficients','checkpoints','__pycache__'}&set(p.relative_to(ROOT).parts) and p.name!='SHA256SUMS']
    (ROOT/'SHA256SUMS').write_text('\n'.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(ROOT)) for p in sorted(files))+'\n')
    bundle=ROOT.parent/'pythia70m_native_component_analysis_2026-10-17.zip'
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files+[ROOT/'SHA256SUMS']:z.write(p,ROOT.name+'/'+str(p.relative_to(ROOT)))
    print(json.dumps({'report':str(ROOT/'pythia70m_native_component_report.html'),'bundle':str(bundle),'bundle_bytes':bundle.stat().st_size,'phase_fraction':phase}),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoints',type=pathlib.Path,default=ROOT.parent/'pythia70m_spectral_2026-10-17'/'checkpoints')
    main(ap.parse_args().checkpoints)
