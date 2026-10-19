"""Self-contained report and reproducible code/data bundle."""
import base64,hashlib,html,json,zipfile
from pathlib import Path
from causal_probe import ROOT,OUT,read

def im(name):return '<img alt="'+name.replace('_',' ')+'" src="data:image/png;base64,'+base64.b64encode((OUT/(name+'.png')).read_bytes()).decode()+'">'
def ci(x,digits=3):return f"{x['estimate']:+.{digits}f} [{x['ci95'][0]:+.{digits}f}, {x['ci95'][1]:+.{digits}f}]"

def main():
    s=read('summary.json');effects={(r['phase'],r['id']):r for r in read('confirmation_effects.json')};leaf=s['selected_leaf']
    root=effects['rollback','H6_root'];fine=effects['rollback',leaf];trans=effects['transplant','H6_root'];late=effects['later','H6_root'];latefine=effects['later',leaf]
    cats={r['id']:r for r in read('component_metadata.json')};inter=read('behavioral_interactions.json')
    qkint=next(r for r in inter if r['phase']=='rollback' and r['probe']=='repeat' and r['parent']=='H6_root' and r['kind']=='native_spatial_split')
    names={'L3_fourier_atom':'Prior Fourier mode','L3_haar_j6':'Prior 64-weight Haar atom','L3_all':'Entire layer-3 QKV update','H6_root':'Head-6 Q/K interval','H6_query':'Head-6 query interval','H6_key':'Head-6 key interval','H5_root':'Neighbor head-5 Q/K','H7_root':'Neighbor head-7 Q/K',leaf:'Selected depth-6 packet group','head6_zero':'Zero head-6 output (control)'}
    rows=[]
    for cid in ['L3_fourier_atom','L3_haar_j6','H6_root','H6_query','H6_key',leaf,'H5_root','H7_root','head6_zero']:
        r=effects['rollback',cid]
        rows.append(f"<tr><td>{names[cid]}</td><td>{r['repeat']['accuracy']*100:.2f}%</td><td>{ci(r['repeat']['target_nll_delta'])}</td><td>{ci(r['scrambled']['target_nll_delta'])}</td><td>{ci(r['prose']['nll_delta'],4)}</td></tr>")
    fullrows=[]
    for r in read('confirmation_effects.json'):
        fullrows.append(f"<tr><td><code>{r['id']}</code></td><td>{r['phase']}</td><td>{ci(r['repeat']['target_nll_delta'],5)}</td><td>{r['repeat']['accuracy']*100:.3f}%</td><td>{ci(r['scrambled']['target_nll_delta'],5)}</td><td>{ci(r['prose']['nll_delta'],5)}</td></tr>")
    runtime=(s['discovery']['seconds']+s['confirmation']['seconds'])/60
    header='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Pythia native component causality</title><style>
    *{box-sizing:border-box}body{margin:0;background:#eef2f4;color:#172637;font:17px/1.62 system-ui,sans-serif}main{max-width:1160px;margin:auto;background:white;padding:45px 30px}h1{font-size:39px;line-height:1.14;max-width:1000px}h2{font-size:26px;margin-top:42px}p{max-width:1040px}a{color:#195f9b}img{width:100%;height:auto;margin:18px 0}code{background:#eef3f5;padding:2px 4px;font-size:.88em}pre{background:#eef3f5;padding:18px;overflow:auto;font-size:13px}.tag{font-size:12px;letter-spacing:.1em;color:#426b82;font-weight:700}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.stat{padding:16px;background:#edf3f6}.stat strong{display:block;font-size:25px}.note{padding:16px 20px;background:#edf7f3;border-left:4px solid #168773}.small{font-size:14px;color:#526371}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;font-size:13px;line-height:1.5}th,td{padding:9px 10px;border-bottom:1px solid #dbe5e9;text-align:left}th{background:#edf3f6}summary{cursor:pointer;font-weight:700;padding:14px;background:#edf3f6}li{margin:7px 0}@media(max-width:650px){main{padding:26px 16px}h1{font-size:29px}.stats{grid-template-columns:repeat(2,1fr)}}
    </style><main><div class="tag">GEOMETRIC CURVE RESEARCH · 17 OCTOBER 2026</div><h1>From component changes to causal behavior in Pythia-70M</h1>'''
    text=f'''<p>The toy calibration suggested that native curve changes need a behavioral response profile, including interactions and the surrounding checkpoint. This pass applies that procedure to the copying transition between Pythia steps 512 and 1,000, then revisits selected native addresses at step 8,000. No training or weight reordering is performed.</p>
    <div class="stats"><div class="stat"><strong>56</strong>discovery candidates</div><div class="stat"><strong>34</strong>locked confirmation candidates</div><div class="stat"><strong>48</strong>fresh prefix sequences</div><div class="stat"><strong>{runtime:.1f} minutes</strong>core inference · three CPU threads</div></div>
    <h2>The main measured result</h2>
    <p>At step 1,000, rolling back the native Q/K interval of block 3, head 6 reduces repeated-sequence successor accuracy from <strong>{root['repeat']['baseline_accuracy']*100:.2f}% to {root['repeat']['accuracy']*100:.2f}%</strong>. Scrambled-query accuracy falls from <strong>{root['scrambled']['baseline_accuracy']*100:.2f}% to {root['scrambled']['accuracy']*100:.2f}%</strong>. These use new token identities and sequence lengths 16, 32 and 48.</p>
    <p>The interval is exactly <code>[589824, 655360)</code> in the original layer-3 QKV tensor: 65,536 native entries, covering that head’s existing query and key projections. Block and head indices are zero-based. Its component coefficients are restored to step 512 while every other parameter remains at step 1,000. This is a checkpoint rollback of a defined curve interval, not deletion of a head.</p>
    <p class="note">This establishes a causal contribution to the tested copying/content-following behavior. It does not identify a natural-language concept, show that the interval is sufficient on its own, or establish preservation of general language ability.</p>
    {im('pythia_component_causality')}
    <h2>Discovery was separated from confirmation</h2>
    <p>Discovery used 16 length-32 prefixes and four original prose passages. Confirmation used 48 fresh prefixes, balanced across lengths 16, 32 and 48, and eight different prose passages. The two sequence sets have disjoint token identities drawn from one randomly shuffled vocabulary pool. The candidate list and selected recursive path were written to disk before confirmation; its checksum remained unchanged.</p>
    <p>Each prefix S is followed either by S without its final token, or by those same eligible query tokens in scrambled order. At each query we score the token that followed it in the original prefix. The scrambled condition checks content-following without the repeated sequence’s fixed positional pattern. This is a synthetic diagnostic of model behavior, not an instruction-following or natural-language benchmark.</p>
    <p>The scan included the largest Fourier-bin change and the largest Haar detail change at three levels in each of all six QKV tensors, full global Fourier and packet splits through depth two, and local packet splits in the head-6 Q/K window. The local path was chosen using discovery NLL only, evaluating both children at each step through depth six. Neighboring heads 5 and 7 use equal-sized native Q/K windows; their update norms differ, so this is not an energy-matched basis comparison.</p>
    <h2>Which components affect what?</h2>
    <p>Positive NLL changes mean worse target prediction. Brackets give 95% paired bootstrap intervals, stratified by sequence length. Prose intervals resample passages with token-weighted means. These describe uncertainty over the probe examples, conditional on one model seed, and are unadjusted for multiple comparisons.</p>
    <div class="scroll"><table><tr><th>Rollback at step 1,000</th><th>Repeat accuracy</th><th>Repeat ΔNLL</th><th>Scrambled ΔNLL</th><th>Prose ΔNLL</th></tr>{''.join(rows)}</table></div>
    <p>The two previously highlighted individual components have small behavioral effects: the original Fourier mode gives {ci(effects['rollback','L3_fourier_atom']['repeat']['target_nll_delta'],5)} nats on repeated queries, and the 64-weight Haar atom gives {ci(effects['rollback','L3_haar_j6']['repeat']['target_nll_delta'],5)}. Their visible training trajectories alone did not identify the main copying mechanism.</p>
    <p>The Q/K interval has a much larger copying effect, with prose NLL changing by {ci(root['prose']['nll_delta'],4)} nats and KL from the unchanged endpoint of {root['prose']['kl_from_baseline']:.6f} nats/token. Despite the small average loss change, <strong>{root['prose']['argmax_changed']*100:.2f}% of prose next-token choices change</strong>. The small prose set is a collateral check; it cannot certify that all other behaviors are preserved.</p>
    <h2>Recursive targeting retains a measurable effect</h2>
    <p>The locked local path ends at <code>{leaf}</code>, packet branch <code>{cats[leaf]['path']}</code>. Its repeated-query NLL change is <strong>{ci(fine['repeat']['target_nll_delta'],4)}</strong> nats; its scrambled-query change is <strong>{ci(fine['scrambled']['target_nll_delta'],4)}</strong>. Its prose NLL change is {ci(fine['prose']['nll_delta'],5)} nats, with {fine['prose']['argmax_changed']*100:.2f}% of prose next-token choices changing.</p>
    <p>This depth-six component contains <strong>{cats[leaf]['coefficient_count']:,} packet coefficients</strong>, reconstructed at all <strong>{cats[leaf]['native_support_count']:,} positions</strong> of the native window. It is a component group, not a single coefficient or a claim that only 1,024 weights changed. The decomposition is orthogonal and full resolution; the mask selects a node while retaining the exact within-node coefficients and phase/sign information.</p>
    {im('pythia_native_component_curves')}
    <h2>The learned surroundings and interactions matter</h2>
    <p>The reverse intervention adds the exact 512→1,000 component update to the step-512 network. For the Q/K interval, repeated-sequence top-choice accuracy is <strong>{trans['repeat']['accuracy']*100:.2f}%</strong>, versus {trans['repeat']['baseline_accuracy']*100:.2f}% before the transplant. Its NLL change is {ci(trans['repeat']['target_nll_delta'])} nats. The change has an effect on the distribution, but it does not by itself install top-choice copying in the earlier network.</p>
    <p>Parent projections equal the sum of their children to numerical precision. Their behavioral effects need not add. Splitting the native Q/K window into Q and K gives a centered-logit interaction magnitude of <strong>{qkint['interaction_fraction']*100:.1f}%</strong> of the sum of the two child-effect RMS magnitudes on repeated queries. The full interaction records include every measured packet/Fourier sibling pair, both intervention directions, and both query conditions.</p>
    <p>At step 8,000, restoring the same Q/K component coordinates to their step-512 values changes repeated-sequence accuracy from <strong>{late['repeat']['baseline_accuracy']*100:.2f}% to {late['repeat']['accuracy']*100:.2f}%</strong>. This later check uses the 512→8,000 coefficient difference; it does not subtract the earlier update from unrelated coordinates.</p>
    <p>The finer group shows an effect reversal under this anchored rollback: at step 8,000 its removal <strong>improves</strong> repeated-query NLL by a signed change of <strong>{ci(latefine['repeat']['target_nll_delta'],4)}</strong> nats, and scrambled-query NLL by {ci(latefine['scrambled']['target_nll_delta'],4)}. Both the selected coefficients and the surrounding network have changed since step 1,000. This does not isolate which change caused the reversal, and it is not evidence of a permanently beneficial or harmful frequency.</p>
    {im('pythia_interactions_later_context')}
    <h2>What the measurements do and do not support</h2>
    <p>The useful result is a behavioral map attached to an existing native hierarchy: a broad Q/K interval has a strong contribution to copying, smaller recursive component groups retain measurable effects, and isolated high-change coefficients need not have a large effect on that behavior. Both rollback and transplant are needed to understand the role of the trained surroundings.</p>
    <p>This pass does not establish a preferred transform, a sparse-training advantage, grokking, a semantic meaning for a Fourier frequency, or a transferable address across model seeds. Some candidate directions have larger natural update norms than others. The deeper path is discovery-selected; confirmation supports its measured effect but does not prove it is the best path in the tree.</p>
    <p>The next bounded experiment should compare behavior-guided recursive masks with same-depth sibling masks at matched update norm, then test controlled interpolation of the selected coefficients on independent prompts. Applying an identical update direction at multiple checkpoints would separate changes in the component itself from changes in its surroundings. The present pass stops at the locked confirmation and later-context checks.</p>
    <h2>Numerics and complete results</h2>
    <p>Transforms and interval differences use float64 values computed from original checkpoint weights; inference uses float32. Every intervention restores the exact original parameter tensor before the next case. Endpoint NLL checks reproduce the original scores exactly. Parent/child projection identities are recorded in <code>parameter_additivity.json</code>.</p>
    <p>For interaction measurement, we retain the full final hidden vector at the last query of every sequence. The fixed output-head Gram matrix reconstructs the RMS effect over all centered vocabulary logits, without truncating vocabulary directions. Interactions concern that predeclared final-query position; NLL and accuracy use every eligible query position.</p>
    <details><summary>All confirmation and later-context effects</summary><div class="scroll"><table><tr><th>Component</th><th>Direction/context</th><th>Repeat ΔNLL</th><th>Repeat accuracy</th><th>Scrambled ΔNLL</th><th>Prose ΔNLL</th></tr>{''.join(fullrows)}</table></div></details>
    <h2>Reproduce or inspect a component</h2><pre>python -m pip install -r requirements.txt
python download_checkpoints.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=3 python causal_probe.py discover
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=3 python causal_probe.py confirm
OPENBLAS_NUM_THREADS=1 python summarize.py
OPENBLAS_NUM_THREADS=1 python figures.py
python build_report.py

# Export the exact full-resolution curves and patch, without model inference:
python inspect_component.py {leaf} --output selected_component.npz</pre>
    <p>The bundle includes all six native QKV curves at steps 512 and 1,000, exact probe inputs and targets, all candidate addresses, the locked selection, per-sequence/per-passage scores, response hidden states, logit Gram matrices, complete local component curves, figures and code. Full language-model checkpoints are regenerated from immutable revisions and verified hashes in <code>checkpoint_manifest.json</code>.</p>
    <p class="small">Provenance: <a href="https://huggingface.co/EleutherAI/pythia-70m">EleutherAI/Pythia-70M</a>, standard published model, with the exact revisions supplied in the bundle. This report extends our native-component atlas and Boolean toy calibration. All reported experiment results are measurements from this run.</p></main></html>'''
    (ROOT/'pythia_component_causality_report.html').write_text(header+text)
    files=sorted(p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and 'checkpoints' not in p.parts and p.name!='SHA256SUMS' and not p.name.endswith(('.log','.zip')))
    (ROOT/'SHA256SUMS').write_text('\n'.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.relative_to(ROOT).as_posix() for p in files)+'\n')
    dest=ROOT.parent/'pythia70m_component_causality_2026-10-17.zip'
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files+[ROOT/'SHA256SUMS']:z.write(p,ROOT.name+'/'+p.relative_to(ROOT).as_posix())
    print(json.dumps(dict(report=str(ROOT/'pythia_component_causality_report.html'),bundle=str(dest),bytes=dest.stat().st_size)))

if __name__=='__main__':main()
