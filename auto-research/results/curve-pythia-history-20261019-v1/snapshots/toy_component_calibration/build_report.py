"""Create scientific figures, a standalone exploratory report, and an archive."""
import base64
import hashlib
import html
import json
import zipfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from calibrate import ROOT, OUT, MODELS, STEPS, X, TARGETS, WALSH, FEATURE_NAMES, projections

BLUE='#2563a6';ORANGE='#cf6232';GREEN='#168773';PURPLE='#8154a0';INK='#172637'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
    'axes.labelcolor':INK,'text.color':INK,'axes.titleweight':'bold','figure.facecolor':'white','axes.facecolor':'white',
    'savefig.facecolor':'white','axes.grid':False})

def load(name):return json.loads((OUT/name).read_text())
def finish(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=180,bbox_inches='tight');fig.savefig(OUT/(name+'.svg'),bbox_inches='tight');plt.close(fig)
def im(name):return '<img alt="'+name.replace('_',' ')+'" src="data:image/png;base64,'+base64.b64encode((OUT/(name+'.png')).read_bytes()).decode()+'">'

def figures():
    tr=load('training.json');s=load('summary.json');ex=s['example'];z=np.load(MODELS/'joint_seed1.npz')
    fig,ax=plt.subplots(1,3,figsize=(13,3.9),layout='constrained')
    for task,col in zip(['readout','parity','context','joint'],[BLUE,ORANGE,GREEN,PURPLE]):
        rows=[r for r in tr if r['task']==task]
        v=np.array([[np.mean(next(r['accuracy'] for r in rows if r['seed']==seed and r['step']==step)) for step in STEPS] for seed in range(5)])
        ax[0].plot(STEPS,v.mean(0)*100,'o-',color=col,label=task,ms=3)
        ax[0].fill_between(STEPS,v.min(0)*100,v.max(0)*100,color=col,alpha=.1)
    ax[0].set(xscale='symlog',xlabel='Training step',ylabel='Exact truth-table accuracy (%)',ylim=(20,102),title='All 20 fits reach 100%')
    ax[0].legend(frameon=False,fontsize=8,ncol=2);ax[0].grid(axis='y',alpha=.15)
    # Ground-truth function features, separate from the weight transform.
    joint=[r for r in tr if r['task']=='joint']
    for oi,fi,label,col in [(0,1,'readout: a',BLUE),(1,3,'parity: ab',ORANGE),(2,5,'routing: -ac',GREEN),(2,6,'routing: bc',PURPLE)]:
        sign=-1 if fi==5 else 1
        vals=np.array([[sign*next(r['function_coefficients'][oi][fi] for r in joint if r['seed']==seed and r['step']==step) for step in STEPS] for seed in range(5)])
        ax[1].plot(STEPS,vals.mean(0),'o-',color=col,ms=3,label=label)
        ax[1].fill_between(STEPS,vals.min(0),vals.max(0),color=col,alpha=.1)
    ax[1].set(xscale='symlog',xlabel='Training step',ylabel='Coefficient of signed probability',ylim=(-.1,1.05),title='Known behaviors appear in the function')
    ax[1].legend(frameon=False,fontsize=8);ax[1].grid(axis='y',alpha=.15)
    for a in ax[:2]:
        a.set_xlim(0,900);a.set_xticks([0,10,50,200,800],labels=['0','10','50','200','800'])
    ls=load('leaf_summaries.json')
    for ti,task in enumerate(['readout','parity','context','joint']):
        vals=[r['change_effect_spearman'] for r in ls if r['task']==task and r['basis']=='haar' and r['start']==0 and r['end']==800]
        ax[2].scatter(ti+np.linspace(-.12,.12,5),vals,color=[BLUE,ORANGE,GREEN,PURPLE][ti],s=32)
        ax[2].hlines(np.median(vals),ti-.22,ti+.22,color=INK,lw=1.5)
    ax[2].set(xticks=range(4),xticklabels=['Readout','Parity','Context','Joint'],ylabel='Spearman correlation',ylim=(.65,1),title='Change size is useful triage')
    ax[2].text(.02,.02,'Haar atoms: update norm vs. output effect\nOne dot per seed; interval 0 → 800',transform=ax[2].transAxes,fontsize=8)
    ax[2].grid(axis='y',alpha=.15)
    fig.suptitle('Toy calibration: known tasks, native coordinates, complete input enumeration',fontsize=15)
    finish(fig,'toy_calibration_overview')

    fig,ax=plt.subplots(2,2,figsize=(12.6,8.6),layout='constrained')
    ax[0,0].plot(STEPS,ex['coefficient_trajectory'],'o-',color=BLUE,ms=4)
    ax[0,0].axhline(ex['before_coefficient'],ls=':',color=ORANGE,label='Rollback value')
    ax[0,0].set(xscale='symlog',xlabel='Training step',ylabel='Signed Haar coefficient',title='A. One component, followed through learning')
    ax[0,0].set_xlim(0,900);ax[0,0].set_xticks([0,10,50,200,800],labels=['0','10','50','200','800'])
    ax[0,0].text(.04,.1,'Hidden layer 2: weights [1,2] and [1,3]\nHaar level 1, location 5 · native C order',transform=ax[0,0].transAxes,fontsize=9)
    ax[0,0].legend(frameon=False,fontsize=8)
    xx=np.arange(3)
    ax[0,1].bar(xx-.17,np.array(ex['baseline_accuracy'])*100,.34,label='Trained',color=BLUE)
    ax[0,1].bar(xx+.17,np.array(ex['rollback_accuracy'])*100,.34,label='One component rolled back',color=ORANGE)
    ax[0,1].set(xticks=xx,xticklabels=['Bit readout','Parity','Context routing'],ylim=(0,113),ylabel='Accuracy over all 16 inputs (%)',title='B. A selective behavioral effect')
    for j,v in enumerate(ex['rollback_accuracy']):ax[0,1].text(j+.17,v*100+2,f'{v*100:g}%',ha='center',fontsize=9)
    ax[0,1].legend(loc='upper left',frameon=False,fontsize=8,ncol=2)
    q=np.tanh(z['original_logits'][-1]/2);qr=np.tanh(z['rollback_logits'][-1,197]/2);diff=q-qr
    h=ax[1,0].imshow(diff,aspect='auto',cmap='RdBu_r',vmin=-2,vmax=2)
    labels=[' '.join('+' if t>0 else '−' for t in x) for x in X]
    ax[1,0].set(xticks=range(3),xticklabels=['Readout','Parity','Context'],yticks=range(16),yticklabels=labels,ylabel='Inputs: a  b  c  d',title='C. The complete behavioral footprint')
    ax[1,0].tick_params(axis='y',labelsize=8)
    fig.colorbar(h,ax=ax[1,0],label='Signed probability: trained − rollback',shrink=.8)
    for j,label,col in [(0,'Readout',BLUE),(1,'Parity',ORANGE),(2,'Context',GREEN)]:
        ax[1,1].plot([d['strength'] for d in ex['dose']],[d['target_margin_drop'][j] for d in ex['dose']],'o-',label=label,color=col)
    ax[1,1].set(xlabel='Fraction of this component’s training change rolled back',ylabel='Drop in mean target-aligned signed probability',title='D. The effect depends on intervention strength')
    ax[1,1].legend(frameon=False,fontsize=9);ax[1,1].grid(axis='y',alpha=.15)
    fig.suptitle('A two-weight component can affect routing while sparing other tasks',fontsize=15)
    fig.supxlabel('Illustrative joint model, seed 1; selected after the exhaustive scan. Different seeds use different native addresses.',fontsize=9)
    finish(fig,'toy_context_component')

    fig,ax=plt.subplots(1,2,figsize=(12,4.3),layout='constrained')
    rs=load('interaction_run_medians.json')
    for k,name in enumerate(['l0.weight','l1.weight','out.weight']):
        rows=[r for r in rs if r['tensor']==name]
        for t,task in enumerate(['readout','parity','context','joint']):
            vals=[r['median_relative'] for r in rows if r['task']==task]
            ax[0].scatter(k+np.linspace(-.19,.19,5)+(t-1.5)*.015,np.maximum(vals,1e-17),color=[BLUE,ORANGE,GREEN,PURPLE][t],s=20,label=task if k==0 else None)
    ax[0].set(xticks=range(3),xticklabels=['First hidden weights','Second hidden weights','Output weights'],yscale='log',ylabel='Relative sibling interaction in logits',title='Additive curves can have interacting effects')
    ax[0].set_ylim(1e-17,2);ax[0].legend(frameon=False,fontsize=8,ncol=2);ax[0].grid(axis='y',alpha=.15)
    ax[0].text(.02,.41,'Each dot: one model’s median across parent/child splits.\nOutput-layer logit interactions are numerical roundoff.',transform=ax[0].transAxes,fontsize=8)
    cols=[BLUE,ORANGE,GREEN,PURPLE]; tasklabels=['Readout','Parity','Context']
    ar=np.array([ex['baseline_initial_accuracy'],ex['transplant_accuracy'],ex['rollback_accuracy'],ex['baseline_accuracy']])*100
    for k,row in enumerate(ar):ax[1].bar(np.arange(3)+(k-1.5)*.19,row,.19,color=cols[k],label=['Initial model','Initial + component change','Trained − component change','Trained model'][k])
    ax[1].set(xticks=range(3),xticklabels=tasklabels,ylim=(0,128),ylabel='Accuracy over all 16 inputs (%)',title='The surrounding trained network matters')
    ax[1].legend(frameon=False,fontsize=8,ncol=2,loc='upper left')
    fig.suptitle('Two checks that keep component interpretation grounded',fontsize=15)
    finish(fig,'toy_interactions_and_context')

def explorer_data():
    z=np.load(MODELS/'joint_seed1.npz');cat=json.loads(str(z['catalog_json']))
    old=[z[f'p{k}'][0] for k in range(6)];new=[z[f'p{k}'][-1] for k in range(6)]
    curves=projections([b-a for a,b in zip(old,new)],cat)
    def rounded(a):return np.asarray(a).round(10).tolist()
    return dict(catalog=cat,curves=[rounded(c.ravel()) for c in curves],old=[rounded(p.ravel()) for p in old],new=[rounded(p.ravel()) for p in new],
        before_logits=rounded(z['original_logits'][0]),after_logits=rounded(z['original_logits'][-1]),rollback=rounded(z['rollback_logits'][-1]),transplant=rounded(z['transplant_logits'][-1]),
        inputs=X.tolist(),targets=z['targets'].tolist())

def report():
    s=load('summary.json');ex=s['example'];env=s['environment']
    top='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Toy calibration of native curve components</title><style>
    *{box-sizing:border-box}body{margin:0;background:#edf2f4;color:#172637;font:17px/1.6 system-ui,sans-serif}main{max-width:1120px;margin:auto;padding:42px 28px;background:white}h1{font-size:38px;line-height:1.15;max-width:900px}h2{font-size:26px;margin-top:44px}h3{font-size:20px}p{max-width:1000px}a{color:#185f9c}img{max-width:100%;height:auto;margin:18px 0}table{border-collapse:collapse;width:100%;margin:18px 0;font-size:15px}th,td{padding:10px 12px;border-bottom:1px solid #dce5e9;text-align:left}th{background:#edf3f6}code{background:#f0f3f5;padding:2px 5px;font-size:.9em}pre{background:#eef3f5;padding:18px;overflow:auto;font-size:13px}.tag{color:#477187;font-weight:700;letter-spacing:.09em;font-size:12px}.note{padding:15px 20px;background:#edf7f3;border-left:4px solid #168773}.small{font-size:14px;color:#4e6070}.controls{display:flex;flex-wrap:wrap;gap:14px;padding:18px;background:#edf3f6}label{font-size:14px;display:flex;flex-direction:column;gap:4px}select{padding:8px;max-width:100%;font-size:14px}canvas{width:100%;height:250px;border:1px solid #dce5e9;margin-top:16px}#explorer{border:1px solid #dce5e9;padding:18px}#truth{overflow:auto}#truth table{font-size:12px}#truth td{padding:5px 8px;font-variant-numeric:tabular-nums}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.stat{background:#edf3f6;padding:14px}.stat strong{display:block;font-size:25px}@media(max-width:650px){main{padding:25px 16px}h1{font-size:29px}.stats{grid-template-columns:repeat(2,1fr)}table{font-size:12px}th,td{padding:8px}}</style><main>
    <div class="tag">GEOMETRIC CURVE RESEARCH · 17 OCTOBER 2026</div><h1>What does a native curve component actually do?</h1>
    <p>Small, fully enumerable models provide the behavioral grounding missing from the descriptive Pythia checkpoint atlas. We train known tasks, retain every parameter-curve component, and directly test each component’s effect in its surrounding network.</p>'''
    body=f'''<div class="stats"><div class="stat"><strong>20</strong>small models · five seeds each</div><div class="stat"><strong>16</strong>inputs · complete truth table</div><div class="stat"><strong>{env['component_intervention_cases']:,}</strong>labelled intervention cases</div><div class="stat"><strong>{env['seconds']:.1f} seconds</strong>core fit-and-scan, one CPU thread</div></div>
    <h2>A minimal progression of known behaviors</h2><table><tr><th>Task</th><th>Exact target</th><th>What it isolates</th></tr><tr><td>Bit readout</td><td>a</td><td>One relevant input; b, c, d are distractors.</td></tr><tr><td>Two-bit parity</td><td>ab</td><td>A relation between two bits; equal signs give +1.</td></tr><tr><td>Context routing</td><td>a when c = −1; b when c = +1</td><td>The context changes which input matters.</td></tr><tr><td>Joint model</td><td>All three outputs</td><td>Shared hidden layers allow us to measure collateral effects.</td></tr></table>
    <p>Inputs are all 16 combinations of four signs. Models use two eight-unit tanh hidden layers and one output logit, or three logits for the joint task: 121 or 139 parameters. Each receives 800 full-batch Adam steps at learning rate 0.02. All 20 fits reach 100% exact truth-table accuracy. All inputs are trained; this is a mechanism calibration, with no held-out generalization or grokking claim.</p>
    {im('toy_calibration_overview')}
    <h2>The complete native-coordinate experiment</h2>
    <p>Every tensor is a separate curve in its original C order. We keep every Haar coefficient, every conjugate-paired Fourier bin and its binary interval ancestors, and every Haar wavelet-packet node at every divisible depth. Approximation and detail branches both split. No padding or permutations are used. Checkpoints are steps 0, 10, 25, 50, 100, 200, 400 and 800.</p>
    <p>For each of seven adjacent intervals and the full 0→800 interval, let Δθ be the training update and P<sub>c</sub> its projection onto component c. We measure both <strong>rollback</strong>, θ<sub>after</sub> − P<sub>c</sub>Δθ, and <strong>transplant</strong>, θ<sub>before</sub> + P<sub>c</sub>Δθ. Every case runs on every possible input. The archive stores all logits, all parameters, and full-precision Fourier/Haar coefficients—not just ranked examples. The count includes overlapping nodes and duplicate representations across bases; it is not a count of independent or deduplicated effects.</p>
    <p>Known input meanings are measured separately. Expanding the signed prediction q = 2p − 1 over Boolean products gives a complete behavioral fingerprint: readout is a, parity is ab, and routing is (a+b−ac+bc)/2. This input-function expansion does not reorder or change the weight curves.</p>
    <p class="note">Full curve reconstruction passes across every stored checkpoint and model: all choices match, and the largest logit difference is {s['functional_reconstruction_max_abs']:.2e}. Parent/child parameter curves add with maximum discrepancy {s['parameter_tree_additivity_max_abs']:.2e}. The selected intervention was also replayed directly in PyTorch.</p>
    <h2>A concrete targeting example</h2>
    <p>In joint seed 1, a level-1 Haar component of <code>l1.weight</code> spans flat entries 10–11: row 1, columns 2 and 3, using zero-based indices. Its coefficient changes from <strong>{ex['before_coefficient']:.6f}</strong> to <strong>{ex['after_coefficient']:.6f}</strong>. Rolling that component back changes exactly two weights and reduces routing accuracy from <strong>100% to 87.5%</strong>. Readout and parity remain at 100%; their mean target-aligned signed probabilities decrease by only {ex['target_margin_drop'][0]:.6f} and {ex['target_margin_drop'][1]:.6f}, versus {ex['target_margin_drop'][2]:.6f} for routing.</p>
    <p>The two mistakes are the same meaningful condition with either value of distractor d: <strong>a=+1, b=−1, c=+1</strong>, where the model should choose b. Accuracy in the c=−1 branch stays at 100%; the c=+1 branch falls to 75%. This identifies a specific behavior affected by a specific native component.</p>
    {im('toy_context_component')}
    <p>This example was selected after the exhaustive scan. Decision-selective hidden Haar components that damage routing while preserving both other task accuracies occur in <strong>3 of 5 joint-model seeds</strong>; the other two seeds still exhibit selective confidence changes. Addresses differ across seeds. The dose check is an explicitly labelled follow-up to this selected example: quarter-, half-, and three-quarter rollback retain perfect accuracy, with growing confidence effects; full rollback crosses the decision boundary.</p>
    <h2>What this changes about interpretation</h2>
    <p><strong>Change magnitude is a useful starting signal.</strong> Across the 20 models, the median rank correlation between Haar update norm and output-function perturbation is {s['haar_change_effect_spearman']['median']:.3f}, with a range of {s['haar_change_effect_spearman']['minimum']:.3f}–{s['haar_change_effect_spearman']['maximum']:.3f}. This descriptive comparison uses naturally sized updates; it does not equalize perturbation energy or establish one basis as superior. Magnitude says something about impact, but does not identify which behavior is affected.</p>
    <p><strong>Parent and child effects need separate measurement.</strong> Although the parameter projections add, the corresponding network effects generally do not. For full-interval Fourier and packet sibling splits, the median logit interaction is 22.4% of the sum of child-effect RMS magnitudes in the first hidden weight matrix and 6.1% in the second. Output-weight logit interactions are numerical roundoff, as expected for the final linear map. We use logits here to distinguish hidden-network interactions from the sigmoid’s own nonlinearity. These node-level medians are descriptive; the figure displays per-model medians.</p>
    <p><strong>The trained surroundings matter.</strong> Transplanting the same example component into the initial network gives only 50% routing accuracy, versus 37.5% initially and 100% in the trained network. A component that matters within a learned computation need not carry that computation by itself.</p>
    {im('toy_interactions_and_context')}
    <h2>Explore every component in the illustrative joint model</h2><p class="small">Joint seed 1, interval 0→800. The explorer includes all 519 declared components. Curves use full native resolution; displayed values are rounded to 10 decimals, while the archive retains float64 values. The complete other models and checkpoint intervals are in the bundle.</p>
    <div id="explorer"><div class="controls"><label>Tensor<select id="tensor"></select></label><label>Decomposition<select id="basis"><option value="haar">Haar coefficient</option><option value="fourier">Fourier interval / bin</option><option value="packet">Wavelet packet node</option></select></label><label>Component<select id="component"></select></label><label>Intervention<select id="direction"><option value="rollback">Rollback in trained model</option><option value="transplant">Transplant into initial model</option></select></label></div><p id="description"></p><canvas id="curve" width="1000" height="280"></canvas><p class="small">Blue: native tensor before intervention. Orange: after intervention. Green: selected component’s signed training update.</p><div id="scores"></div><div id="truth"></div></div>
    <h2>Anchor for the next Pythia pass</h2>
    <p>Keep the existing native component atlas. Associate candidate components with explicit behavioral probes, then compare rollback and transplant effects on that behavior and on unrelated outputs. Measure parent/child interactions in logits, and retain the full before/after response vectors so a small average loss change cannot hide a specific failure. For Pythia, the full prompt space cannot be enumerated, so candidate selection and confirmation should use separate prompt sets.</p>
    <p>This calibration supports testing component-level targeting. It does not establish a universal frequency-to-meaning map, prove that the Pythia 64-weight example is semantic, or demonstrate a curve-trained architecture advantage. All experiments here concern existing network parameters in situ.</p>
    <h2>Relation to the literature</h2><p><a href="https://arxiv.org/abs/2209.10652">Toy Models of Superposition</a> shows why small controlled systems can expose interpretable mechanisms. Our tasks are different and do not measure superposition directly. <a href="https://arxiv.org/abs/2403.19647">Sparse Feature Circuits</a> studies causally implicated activation features in language models. The relevant connection is to validate behavioral effects explicitly; a fixed weight-curve coefficient is not automatically an activation feature.</p>
    <h2>Reproduction and data</h2><pre>python calibrate.py
python summarize.py
python build_report.py

# Inspect any saved component; no retraining needed:
python inspect_component.py --task joint --seed 1 --component 197 --start 0 --end 800</pre>
    <p>Each <code>models/task_seedN.npz</code> contains checkpoint weights, every Fourier and Haar coefficient, the component catalog, all rollback and transplant logits, input vectors, labels, and update norms. JSON summaries include all seeds, numerical checks, sibling interactions and the labelled follow-up. The complete code-and-data ZIP is self-contained and includes SHA-256 checksums.</p>
    <p class="small">Scope: four Boolean task settings, five initializations, one small architecture family. All possible inputs are trained and tested. Exhaustive means every declared single component and sibling pair—not every possible subset, every perturbation direction, or every possible model. Core runtime excludes report construction and subsequent numerical replay.</p></main>'''
    js=(ROOT/'explorer.js').read_text()
    data=json.dumps(explorer_data(),separators=(',',':'))
    (ROOT/'toy_component_calibration_report.html').write_text(top+body+'<script>const DATA='+data+';</script><script>'+js+'</script></html>')

def bundle():
    files=sorted(p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ['SHA256SUMS','run.log'] and not p.name.endswith('.zip'))
    manifest='\n'.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.relative_to(ROOT).as_posix() for p in files)+'\n'
    (ROOT/'SHA256SUMS').write_text(manifest)
    dest=ROOT.parent/'toy_component_calibration_2026-10-17.zip'
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files+[ROOT/'SHA256SUMS']:z.write(p,ROOT.name+'/'+p.relative_to(ROOT).as_posix())
    print(json.dumps(dict(report=str(ROOT/'toy_component_calibration_report.html'),bundle=str(dest),bundle_bytes=dest.stat().st_size)))

if __name__=='__main__':figures();report();bundle()
