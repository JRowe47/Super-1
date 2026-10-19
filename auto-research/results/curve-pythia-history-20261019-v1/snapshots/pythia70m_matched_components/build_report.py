"""Self-contained measured report and reproducible native-curve bundle."""
import base64,hashlib,html,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results'
def read(n):return json.loads((OUT/n).read_text())
def image(name):return '<img alt="'+name.replace('_',' ')+'" src="data:image/png;base64,'+base64.b64encode((OUT/(name+'.png')).read_bytes()).decode()+'">'
def ci(s,n=4):return f"{s['estimate']:+.{n}f} [{s['ci95'][0]:+.{n}f}, {s['ci95'][1]:+.{n}f}]"
def find(rows,step,path='aaaaad',direction='early',lam=1):return next(r for r in rows if r['step']==step and r['path']==path and r['direction']==direction and abs(r['lam']-lam)<1e-7)
def main():
    s=read('summary.json');ce=read('confirmation_effects.json');se=read('scan_effects.json');sel=s['selection'];m=s['curves'];con=s['paired_contrasts']
    check=read('direct_scoring_verification.json');e1=find(ce,1000);e8=find(ce,8000);l1=find(ce,1000,direction='late');l8=find(ce,8000,direction='late')
    r=m['late_to_early_norm_ratio'];lnat=find(ce,8000,direction='late',lam=r);enat=find(ce,8000,lam=r)
    def contrast(name,kind='repeat',split='confirmation'):return next(r for r in con if r['contrast']==name and r['probe']==kind and r['split']==split)
    controls=[]
    for p in sel['paths']:
        x=find(ce,1000,path=p);label=p+(' (prior selected)' if p=='aaaaad' else ' (scan winner)' if p==sel['scan_winner'] else ' (sibling)' if p=='aaaaaa' else ' (largest natural norm)' if p==m['weight_only_max_norm_path'] else ' (fixed control)')
        controls.append(f"<tr><td><code>{label}</code></td><td>{ci(x['repeat']['target_nll_delta'])}</td><td>{ci(x['scrambled']['target_nll_delta'])}</td><td>{x['repeat']['accuracy']*100:.2f}%</td><td>{ci(x['prose']['nll_delta'],5)}</td><td>{x['prose']['changed_argmax_count']}/{x['prose']['tokens']}</td></tr>")
    contexts=[]
    for step in [512,1000,8000]:
        for direction in ['early','late']:
            x=find(ce,step,direction=direction)
            contexts.append(f"<tr><td>{step:,}</td><td>{direction}</td><td>{ci(x['repeat']['target_nll_delta'])}</td><td>{ci(x['scrambled']['target_nll_delta'])}</td><td>{x['repeat']['accuracy']*100:.2f}%</td><td>{x['prose']['kl_from_baseline']:.7f}</td></tr>")
    detailed=[]
    for x in ce:
        detailed.append(f"<tr><td>{x['step']:,}</td><td><code>{x['path']}</code></td><td>{x['direction']}</td><td>{x['lam']:+.4f}</td><td>{ci(x['repeat']['target_nll_delta'],5)}</td><td>{ci(x['scrambled']['target_nll_delta'],5)}</td><td>{ci(x['prose']['nll_delta'],5)}</td><td>{x['prose']['changed_argmax_count']}/{x['prose']['tokens']}</td></tr>")
    scanrows=[]
    for x in se:
        scanrows.append(f"<tr><td><code>{x['path']}</code></td><td>{m['early_norms'][x['path']]:.5f}</td><td>{ci(x['repeat']['target_nll_delta'],5)}</td><td>{ci(x['scrambled']['target_nll_delta'],5)}</td></tr>")
    base='; '.join(f"step {b['step']:,}: {b['repeat_accuracy']*100:.2f}% repeat, {b['scrambled_accuracy']*100:.2f}% scrambled" for b in s['baselines'])
    winnercontrast=contrast('selected_minus_'+sel['scan_winner']) if sel['scan_winner']!='aaaaad' else None
    if winnercontrast and winnercontrast['ci95'][0]<=0<=winnercontrast['ci95'][1]:
        winnertext='The fresh confirmation does not clearly distinguish the scan winner from the prior-selected branch: their paired repeat-NLL difference is '+ci(winnercontrast)+' nats. The earlier recursive descent found a competitive branch, but the precise rank should not be overinterpreted.'
    elif winnercontrast:
        winnertext='On fresh confirmation, prior-selected minus scan-winner repeat NLL is '+ci(winnercontrast)+' nats. Positive means the prior-selected rollback is more disruptive on these probes; negative means the scan winner is more disruptive.'
    else:winnertext='The scan selected the same branch as the prior recursive descent.'
    esign=e8['repeat']['target_nll_delta']['estimate'];lsign=l8['repeat']['target_nll_delta']['estimate']
    if esign*lsign<0:
        contexttext='At step 8,000, subtracting the early direction and subtracting the late direction have opposite signs despite equal norm and identical component coordinates. Thus a fixed address and an update magnitude do not determine the behavioral effect; the orientation of the coefficient change matters in this checkpoint.'
    else:
        contexttext='The same coefficient coordinates exhibit different measured responses as the direction and background checkpoint change. The controlled table, rather than the historical rollback alone, separates these dependencies.'
    header='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Pythia: matched native component analysis</title><style>
    *{box-sizing:border-box}body{margin:0;background:#edf1f3;color:#1b2a38;font:17px/1.62 system-ui,sans-serif}main{max-width:1180px;margin:auto;background:white;padding:42px 32px}h1{font-size:40px;line-height:1.15}h2{font-size:26px;margin-top:40px}img{width:100%;height:auto;margin:16px 0}code{font-size:.87em;background:#f0f4f6;padding:2px 4px}pre{background:#f0f4f6;padding:18px;font-size:13px;overflow:auto}.tag{font-size:12px;letter-spacing:.1em;color:#466b83;font-weight:700}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.stat{background:#edf4f7;padding:15px}.stat strong{display:block;font-size:27px}.note{padding:17px 20px;background:#eef7f3;border-left:4px solid #278571}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;font-size:13px;line-height:1.5}th,td{text-align:left;border-bottom:1px solid #dce5e9;padding:9px}th{background:#edf3f6}summary{padding:14px;background:#edf3f6;cursor:pointer;font-weight:700}.small{font-size:14px;color:#536678}a{color:#196caa}li{margin:6px 0}@media(max-width:700px){main{padding:24px 16px}h1{font-size:30px}.stats{grid-template-columns:repeat(2,1fr)}}
    </style><main><div class="tag">NATIVE CURVE RESEARCH · 17 OCTOBER 2026</div><h1>Matched component size, different behavioral effects</h1>'''
    body=f'''<p>This phase tests whether the earlier recursive targeting result survives equal-size controls, whether its effect varies smoothly with intervention strength, and what changes when an identical update vector is applied at different training checkpoints. It uses the existing Pythia-70M checkpoints without training.</p>
    <div class="stats"><div class="stat"><strong>64</strong>same-depth, equal-norm branches</div><div class="stat"><strong>48</strong>fresh confirmation sequences</div><div class="stat"><strong>3</strong>background checkpoints</div><div class="stat"><strong>{s['environment']['total_seconds']/60:.1f} min</strong>core experiment · three CPU threads</div></div>
    <h2>What survived the controls?</h2>
    <p>The previously selected branch <code>aaaaad</code> ranks <strong>{s['scan_ranks']['repeat']['selected_rank']} of 64</strong> for repeated-query disruption and <strong>{s['scan_ranks']['scrambled']['selected_rank']} of 64</strong> for scrambled-query disruption when every direction has the same L2 norm. These are descriptive ranks on the scan set. Its repeat-NLL effect exceeds the mean of the other 63 branches by <strong>{ci(contrast('selected_minus_other63_mean',split='scan'))}</strong> nats.</p>
    <p>On independent confirmation, subtracting this branch’s early update increases repeated-query loss by <strong>{ci(e1['repeat']['target_nll_delta'])}</strong> nats and scrambled-query loss by <strong>{ci(e1['scrambled']['target_nll_delta'])}</strong>. The response therefore survives a fresh probe set and the norm controls; the exact branch rank remains uncertain.</p>
    <p class="note">The useful targeting object is a native component address together with its coefficient direction, strength, background checkpoint and measured behavioral response. A branch’s energy or address alone is insufficient.</p>
    {image('pythia_matched_components')}
    <h2>A complete depth, then independent confirmation</h2>
    <p>The scan tests all 64 depth-six Haar-packet branches inside the original block-3, head-6 Q/K interval <code>[589824,655360)</code>. Each retains all <strong>1,024 coefficients</strong> and reconstructs an update over <strong>65,536 native weights</strong>. The source tensor remains in its original packed row-major order. No weights are permuted, aligned, padded, downsampled or truncated.</p>
    <p>Every scan direction is the corresponding projection of the step-512→1,000 update, rescaled to L2 norm <strong>{m['common_norm']:.6f}</strong>, the natural norm of <code>aaaaad</code>. Natural branch norms range from {min(m['early_norms'].values()):.6f} to {max(m['early_norms'].values()):.6f}. Matching norm controls magnitude but does not equalize Q/K balance, activation sensitivity or every other structural property.</p>
    <p>The scan uses 24 sequences, eight each at prefix lengths 16, 32 and 48, plus four original prose passages. Before any confirmation is scored, the procedure locks the prior branch, its sibling, a fixed all-detail branch, the branch with the largest natural update norm, and the scan winner <code>{sel['scan_winner']}</code>. Confirmation uses 48 new sequences and eight different prose passages. Token identities are disjoint across the sets and exclude the 2,048 random-prefix IDs used in the preceding phase.</p>
    <p>{winnertext}</p>
    <div class="scroll"><table><tr><th>Equal-norm branch at step 1,000</th><th>Repeat ΔNLL</th><th>Scrambled ΔNLL</th><th>Repeat accuracy</th><th>Prose ΔNLL</th><th>Prose argmax changes</th></tr>{''.join(controls)}</table></div>
    <p>Brackets are paired 95% bootstrap intervals, not multiplicity-adjusted hypothesis tests. Positive loss changes mean worse prediction. The confirmation baselines are {base}. These differ from the preceding report because the inputs are new.</p>
    <h2>Holding the direction fixed across checkpoints</h2>
    <p>Let E be the selected projection of θ<sub>1000</sub>−θ<sub>512</sub>, and L the projection of θ<sub>8000</sub>−θ<sub>512</sub>. The controlled experiment first gives both vectors the same norm, then evaluates <code>θ′ = θ<sub>s</sub> − λv</code>. Positive λ subtracts the vector; negative λ adds it. The exact same vector is reused at checkpoints 512, 1,000 and 8,000.</p>
    <p>The selected early and late directions have cosine <strong>{m['selected_early_late_cosine']:.4f}</strong>, an angle of <strong>{m['selected_angle_degrees']:.1f}°</strong>. The natural late vector is {r:.3f} times longer. Testing both at a common norm separates this orientation change from magnitude.</p>
    <div class="scroll"><table><tr><th>Checkpoint</th><th>Direction, λ=1</th><th>Repeat ΔNLL</th><th>Scrambled ΔNLL</th><th>Repeat accuracy</th><th>Prose KL</th></tr>{''.join(contexts)}</table></div>
    <p>{contexttext}</p>
    <p>For a fixed early vector, changing the background from step 1,000 to 8,000 changes its repeat-NLL effect by {ci(contrast('early_context8000_minus1000'))} nats. The corresponding late-vector change is {ci(contrast('late_context8000_minus1000'))}. The direction-by-checkpoint difference in differences is {ci(contrast('direction_by_context_difference_in_differences'))}.</p>
    <p>At the original late magnitude (λ={r:.3f}), step-8,000 loss changes by {ci(lnat['repeat']['target_nll_delta'])} for the late vector and {ci(enat['repeat']['target_nll_delta'])} for an equally long early vector. This directly relates the controlled comparison to the previous anchored rollback.</p>
    <p>“Background checkpoint” includes all baseline weights, including the existing values in this component. These tests do not localize the dependence specifically to parameters outside the group, nor assign a unique additive cause to the historical effect reversal.</p>
    <h2>Intervention strength and collateral effects</h2>
    {image('pythia_component_dose_response')}
    <p>The plotted values are independent evaluations at each strength, with no optimizer updates. At step 1,000, adding the early direction (λ=−1) changes repeat NLL by {ci(find(ce,1000,lam=-1)['repeat']['target_nll_delta'])}; subtracting it (λ=+1) changes NLL by {ci(e1['repeat']['target_nll_delta'])}. The saved second-difference contrasts quantify departure from a locally linear response.</p>
    <p>For the selected unit rollback at step 1,000, original-prose NLL changes by {ci(e1['prose']['nll_delta'],6)} nats. KL from the original endpoint is {e1['prose']['kl_from_baseline']:.7f}, and <strong>{e1['prose']['changed_argmax_count']}/{e1['prose']['tokens']}</strong> next-token argmax choices change. Small average prose loss does not establish that unrelated capabilities are preserved. Every variant’s collateral measurements are included below and in the raw records.</p>
    <h2>Full resolution and exact native coordinates</h2>
    {image('pythia_native_matched_curves')}
    <p>The recursive transform uses <code>a=(even+odd)/√2</code> and <code>d=(even−odd)/√2</code>. The complete depth-six tree reconstructs both training-update curves with maximum absolute errors {m['early_reconstruction_max_abs']:.2e} and {m['late_reconstruction_max_abs']:.2e}. Paths label binary-tree decisions; they are not a monotonic Fourier-frequency axis. Coefficients and reconstructed curves retain all signs and native positions.</p>
    <p>The archive includes original float32 curves for all three checkpoints, every float64 depth-six coefficient for both intervals, and explicit full-length selected update curves. The 512-position plot is a visual crop only. Analysis and interventions always use the complete interval.</p>
    <h2>Verification and limits</h2>
    <p>All {s['environment']['variants']} interventions restored the exact original parameter tensor. No edit changed a position outside the declared window. Restored baseline loss differences were zero. The maximum relative difference between intended and applied update norm was {s['verification']['max_relative_norm_error']:.2e}; this is the recorded float32 installation error after float64 projection.</p>
    <p>An independent ordinary full-model forward pass with direct target indexing checked both query types, all three lengths, and selected baseline/intervention cases at steps 1,000 and 8,000. Its largest NLL discrepancy was {check['maximum_nll_abs_error']:.2e}; all checked top-1 positions matched. The three checkpoint files match their pinned SHA-256 hashes. Selection and protocol hashes remained unchanged.</p>
    <p>This is one model training seed, a synthetic successor-copying/content-following assay, and a small prose collateral check. It does not establish natural-language concept identity, grokking, a universally best transform or cross-seed targeting. Conditional intervals resample sequences, stratified by length, 4,000 times; prose intervals resample passages with token-weighted means. Nominal scan ranking is not statistically resolved ordering.</p>
    <p>The next useful target is finer resolution <em>within</em> the confirmed groups: split their existing coefficients into native subgroups, map which subsequences respond, and compare child effects with the parent at controlled norms. This would test whether the effect is spread across the group or concentrated in a smaller native component. The present run stops at its fixed protocol.</p>
    <details><summary>Every confirmation intervention</summary><div class="scroll"><table><tr><th>Step</th><th>Path</th><th>Direction</th><th>λ</th><th>Repeat ΔNLL</th><th>Scrambled ΔNLL</th><th>Prose ΔNLL</th><th>Argmax changes</th></tr>{''.join(detailed)}</table></div></details>
    <details><summary>All 64 equal-norm scan branches</summary><div class="scroll"><table><tr><th>Path</th><th>Natural norm before matching</th><th>Repeat ΔNLL</th><th>Scrambled ΔNLL</th></tr>{''.join(scanrows)}</table></div></details>
    <h2>Reproduce and inspect</h2><pre>python -m pip install -r requirements.txt
python download_checkpoints.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=3 python run_experiment.py
OPENBLAS_NUM_THREADS=1 python analyze.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=3 python verify_scoring.py
OPENBLAS_NUM_THREADS=1 python figures.py
python build_report.py

# Full-resolution export without running the model:
python inspect_component.py aaaaad --direction early --matched --output early.npz</pre>
    <p>The bundle contains code, frozen protocol, checkpoint manifest, exact probe inputs, per-query scores, all native component coefficients, response hidden states, centered output-head Gram matrices, paired contrasts, figures and checksums. The full model files can be regenerated from immutable revisions. Model source: <a href="https://huggingface.co/EleutherAI/pythia-70m">EleutherAI/Pythia-70M</a>.</p>
    <p class="small">All numerical findings in this report are measurements from this run. Core runtime excludes report generation and the independent verification pass.</p></main></html>'''
    report=ROOT/'pythia_matched_components_report.html';report.write_text(header+body)
    files=sorted(p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and 'checkpoints' not in p.parts and p.name!='SHA256SUMS' and not p.name.endswith(('.log','.zip','.tmp')))
    (ROOT/'SHA256SUMS').write_text('\n'.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.relative_to(ROOT).as_posix() for p in files)+'\n')
    dest=ROOT.parent/'pythia70m_matched_components_2026-10-17.zip'
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files+[ROOT/'SHA256SUMS']:z.write(p,ROOT.name+'/'+p.relative_to(ROOT).as_posix())
    print(json.dumps(dict(report=str(report),bundle=str(dest),bytes=dest.stat().st_size,files=len(files)+1)))
if __name__=='__main__':main()
