"""Build a portable, self-contained HTML report from the measured data."""
import base64,html,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
read=lambda n:json.loads((ROOT/'results'/n).read_text())
S=read('summary.json');B=read('pythia_baseline.json')
labels={'native_adamw':'Native Adam','curve_transport_adamw':'Curve + transported Adam','curve_adamw':'Adam on curve coefficients',
 'native_sgd':'Native SGD','curve_sgd':'Curve SGD','native6':'Native six stages','curve6':'Full six-mode curve',
 'curve2':'Two-mode curve','curve2_multidepth':'Two modes, multi-depth training','curve6_exact':'Exact six-mode conversion',
 'curve2_hard':'Hard two-mode conversion','curve2_bridge':'Bridged two-mode conversion'}
def ci(c,precision=3):
    a,b=c['ci95'];return f"{c['estimate']:+.{precision}f} [{a:+.{precision}f}, {b:+.{precision}f}]"
def table(headers,rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+h+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(c)+'</td>' for c in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def figure(name,alt,caption):
    data=base64.b64encode((ROOT/'figures'/f'{name}.png').read_bytes()).decode()
    return '<figure><img src="data:image/png;base64,'+data+'" alt="'+html.escape(alt)+'"><figcaption>'+caption+'</figcaption></figure>'
final={(r['group'],r['arm']):r for r in S['pythia_final']}
adam=['native_adamw','curve_transport_adamw','curve_adamw']
rows=[['Starting checkpoint',f"{B['validation']['nll']:.6f}",'—',f"{100*B['retention']['repeat']['accuracy']:.2f}%",'—']]
for a in adam:
    r=final['original',a]
    rows.append([labels[a],f"{r['validation_nll']:.6f}",ci(r['validation_delta']),f"{100*r['repeat']['accuracy']:.2f}%",ci(r['prose']['nll_delta'])])
adamtable=table(['Arm, after 24 steps','Held-out NLL','Change in NLL [95% CI]','Copy accuracy','Prose ΔNLL [95% CI]'],rows)
retentiontable=table(['Arm','Copy accuracy Δ, pp [95% CI]','Scrambled-query accuracy','Prose KL from teacher','Prose top-token changes'],[
 [labels[a],ci(final['original',a]['repeat']['accuracy_delta'],2),f"{100*final['original',a]['scrambled']['accuracy']:.2f}%",f"{final['original',a]['prose']['kl']:.4f}",f"{100*final['original',a]['prose']['argmax_changed']:.2f}%"] for a in adam])
ct=S['paired_contrasts']
contrasttable=table(['Coefficient Adam minus native Adam','Estimate [95% CI]'],[
 ['Held-out NLL, nats',ci(ct['curve_minus_native_validation'],4)],
 ['Repeated-copy accuracy, percentage points',ci(ct['curve_minus_native_repeat_accuracy_pp'],2)],
 ['Scrambled-query accuracy, percentage points',ci(ct['curve_minus_native_scrambled_accuracy_pp'],2)]])
eqrows=[]
for name,rr in [('Transported Adam',S['adam_transport_equivalence']),('Stable-step SGD',S['stable_sgd_equivalence'])]:
    r=next(x for x in rr if x['step']==24)
    eqrows.append([name,f"{r['max_abs_parameter_difference']:.3g}",f"{r['parameter_rms_difference']:.3g}",f"{r['max_abs_logit_difference']:.5f}",f"{r['logit_rms_difference']:.6f}"])
eqtable=table(['Curve vs native, step 24','Max weight difference','RMS weight difference','Max logit difference','RMS logit difference'],eqrows)
sgdtable=table(['SGD condition','Native final NLL','Curve final NLL','Native / curve copy accuracy'],[
 ['Original LR 0.005 (unstable)',f"{final['original','native_sgd']['validation_nll']:.6f}",f"{final['original','curve_sgd']['validation_nll']:.6f}",'0% / 0%'],
 ['Added LR 2.60431694 × 10⁻⁶',f"{final['stable_sgd','native_sgd']['validation_nll']:.6f}",f"{final['stable_sgd','curve_sgd']['validation_nll']:.6f}",f"{100*final['stable_sgd','native_sgd']['repeat']['accuracy']:.2f}% / {100*final['stable_sgd','curve_sgd']['repeat']['accuracy']:.2f}%"]])
toyarms=['native6','curve6','curve2','curve2_multidepth','curve6_exact','curve2_hard','curve2_bridge']
toyrows=[]
for arm in toyarms:
    r=next(x for x in S['toy_final'] if x['arm']==arm and x['depth']==6)
    d=next(x for x in S['toy_final'] if x['arm']==arm and x['depth']==24)
    z=[x['accuracy']*100 for x in S['toy_zero_residual'] if x['arm']==arm]
    toyrows.append([labels[arm],r['active_parameters'],f"{100*r['accuracy']['mean']:.1f}%",f"{r['bce']['mean']:.3g}",f"{d['logit_rms_from_depth6']['mean']:.3f}",f"{min(z):.1f}–{max(z):.1f}%"])
toytable=table(['Arm','Active parameters','Six-step accuracy','Mean six-step BCE','24 vs 6 step logit RMS','No-stage accuracy range'],toyrows)
convrows=[]
for r in S['toy_conversion']:
    convrows.append([r['seed'],f"{100*r['discarded_weight_energy_fraction']:.2f}%",f"{r['curve2_hard']['initial_logit_rms_error']:.3f}",f"{100*r['curve2_hard']['gradient']['excluded_gradient_energy_fraction']:.2f}%"])
convtable=table(['Paired seed','Discarded residual matrix energy','Immediate hard-projection logit RMS','Excluded gradient energy after projection'],convrows)
rt=read('pythia_roundtrip_checks.json');mismatches=sum(r['float32_differing_entries'] for r in rt)
runtime=sum(S['runtime'].values())/60

body=f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Curve-space continuation: Pythia-70m and shared-depth toy controls</title>
<style>
:root{{--ink:#162735;--muted:#536b7b;--line:#dbe5eb;--blue:#1965b0;--teal:#078c83;--paper:#f3f7f9}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.65 system-ui,-apple-system,Segoe UI,sans-serif}}
main{{max-width:1180px;margin:0 auto;padding:56px 40px 80px;background:#fff}}h1{{font-size:38px;line-height:1.18;letter-spacing:-1px;max-width:1000px;margin:12px 0 18px}}h2{{font-size:25px;margin:46px 0 16px;line-height:1.3}}h3{{font-size:19px;margin-top:30px}}p{{max-width:1000px;margin:12px 0}}a{{color:var(--blue)}}.eyebrow{{color:var(--teal);font-weight:750;font-size:13px;letter-spacing:1.3px;text-transform:uppercase}}.lead{{font-size:20px;line-height:1.55;max-width:1020px}}.meta,.note,figcaption{{color:var(--muted);font-size:13px;line-height:1.6}}.callout{{padding:18px 22px;background:#eaf5f4;border-left:4px solid var(--teal);margin:24px 0}}.caution{{background:#fff5e8;border-left-color:#d47219}}nav{{display:flex;gap:20px;flex-wrap:wrap;border-top:1px solid var(--line);border-bottom:1px solid var(--line);padding:14px 0;margin:28px 0}}nav a{{text-decoration:none;font-size:14px}}figure{{margin:26px 0}}img{{width:100%;height:auto}}figcaption{{margin-top:8px}}table{{width:100%;border-collapse:collapse;font-size:13px;line-height:1.45;margin:18px 0}}th{{text-align:left;background:#eaf0f4;padding:12px 10px;vertical-align:bottom}}td{{padding:12px 10px;border-bottom:1px solid var(--line);vertical-align:top;font-variant-numeric:tabular-nums}}td:first-child{{font-weight:600}}.scroll{{overflow:auto}}code,pre{{font-family:ui-monospace,Consolas,monospace;font-size:13px}}pre{{white-space:pre-wrap;background:#f2f5f7;padding:18px;border:1px solid var(--line);line-height:1.6}}.equation{{font-family:Georgia,serif;font-size:20px;background:#f3f7f9;padding:18px 22px}}li{{margin:8px 0}}footer{{margin-top:44px;border-top:1px solid var(--line);padding-top:18px;color:var(--muted);font-size:13px}}@media(max-width:700px){{main{{padding:26px 18px}}h1{{font-size:29px}}.lead{{font-size:18px}}table{{min-width:650px}}}}@media print{{body{{background:white}}main{{max-width:none;padding:0}}nav{{display:none}}h2{{break-after:avoid}}figure,tr{{break-inside:avoid}}}}
</style></head><body><main>
<div class="eyebrow">Curve training compatibility · measured experiment · 17 October 2026</div>
<h1>Can a conventionally trained model keep learning in curve coordinates?</h1>
<p class="lead"><strong>Yes in the complete, native-order representation tested here.</strong> The important separation is between changing coordinates, changing the optimizer, and restricting the executable model. Pythia learned through full Haar coefficients. Shared-depth toys also learned, but their ceiling performance leaves the deeper GSA capacity question open.</p>
<p class="meta">Pythia-70m step 143000 · 18,914,304 trainable decoder parameters · 24 identical batches per arm · five paired toy seeds · approximately {runtime:.2f} minutes of measured experiment runtimes, excluding analysis and packaging.</p>
<nav><a href="#pythia">Pythia continuation</a><a href="#numerics">Equivalence and numerical drift</a><a href="#depth">Shared depth curves</a><a href="#meaning">Implications for GSA</a><a href="#reproduce">Methods and reproduction</a></nav>
<div class="callout"><strong>Practical conclusion.</strong> Preserve all curve modes and the native optimizer as the conversion baseline. Then test coefficient-space optimization, restrictions, and depth changes independently. The observed copying loss during adaptation occurs in native training too; it cannot be attributed solely to curve conversion.</div>

<h2 id="pythia">Pythia: real coefficient gradients, short continuation, paired controls</h2>
<p>Each of the 72 tensors in Pythia’s six decoder blocks is flattened in its existing C order and represented by a complete orthonormal Haar pyramid. Every coefficient is retained. Layer identities, attention, token order, nonlinearities and normalization are preserved. Embeddings, the final normalization and the output head stay fixed in every arm. This is a full-resolution parameter representation; it does not yet share curves across Pythia layers.</p>
<p>The curve arms backpropagate through an explicit inverse transform into trainable coefficients. The network runs in float32 using decoded dense matrices, while master weights or coefficients are float64. All Adam runs use learning rate 2 × 10⁻⁵, β = (0.9, 0.999), ε = 10⁻⁸ and zero weight decay, with fresh optimizer states. “Transported Adam” maps coefficient gradients back to native coordinates, maintains native Adam moments, then re-encodes each updated weight tensor.</p>
{figure('pythia_curve_continuation','Held-out loss, copying accuracy, update direction, and accumulated output differences across continuation steps.','Lines connect evaluations at steps 0, 1, 4, 12 and 24; intermediate behavior is not measured. Native and transported Adam almost overlap. Panel D compares one 96-token validation block across all vocabulary logits. Direct coefficient Adam is excluded from D because its optimizer is intentionally different.')}
{adamtable}
<p class="note">NLL is negative log-likelihood in nats per target token; lower is better. Confidence intervals are 4,000 paired bootstrap resamples of 24 validation blocks, or eight prose passages with token weighting. These intervals describe item variability within this single short run, not uncertainty across training seeds. The prose set is reused from the preceding phase.</p>
<p>Ordinary Adam and transported Adam improve held-out WikiText loss by approximately 0.367 nats. Coefficient Adam improves it by 0.331 nats. All three substantially reduce copying accuracy and increase loss on the small retained prose set. On prose, native Adam changes the most likely next token on 93 of 284 positions; coefficient Adam changes it on 96.</p>
{retentiontable}
<p class="note">Copy accuracy is the mean over 24 sequences, with eight sequences at each prefix length 16, 32 and 48. Accuracy-change intervals resample sequences within each length. Scrambled queries test the original prefix’s successor relation; they are not scored against adjacency in the scrambled query. Baseline scrambled-query accuracy is {100*B['retention']['scrambled']['accuracy']:.2f}%. KL is baseline-teacher-to-adapted-model KL, token-weighted over prose.</p>
<h3>Adam changes when the coordinates change</h3>
<p>The first native and coefficient Adam updates have almost the same Euclidean norm (0.0869715 and 0.0869720), yet their directional cosine is only <strong>0.6390</strong>. Transported Adam has cosine 1.0000 with native Adam. This directly connects the earlier component-direction observations to actual learning: equal update size does not imply the same functional intervention.</p>
{contrasttable}
<p>The fixed-rate coefficient optimizer performs slightly worse on held-out NLL in this run, with a small copying-retention tradeoff. The repeated-copy contrast is compatible with either sign. No learning-rate search was performed, so this is evidence that the trajectory changes, not a general ranking of native and curve optimizers.</p>

<h2 id="numerics">Exact mathematics, finite-precision implementation</h2>
<div class="equation">w = H<sup>T</sup>c &nbsp;&nbsp; and &nbsp;&nbsp; ∇<sub>c</sub>L = H ∇<sub>w</sub>L, &nbsp;&nbsp; H<sup>T</sup>H = I.</div>
<p>With the full orthonormal transform, unpreconditioned SGD commutes with the coordinate change in exact arithmetic. Adam’s coordinatewise second moments generally do not. The transported control preserves the native optimizer mathematically; it is a diagnostic control, not an efficient curve optimizer.</p>
<p>The independently calculated first-step gradient adjoint error is at most 2.67 × 10⁻¹⁵. Native versus curve SGD first updates differ by at most 1.20 × 10⁻¹⁴ per coordinate; native versus transported Adam differ by at most 2.91 × 10⁻¹⁴. The complete initial transform has a maximum float64 reconstruction error of {max(r['max_abs_roundtrip'] for r in rt):.3g}. There are {mismatches} float32 bitwise mismatches at very small or zero coordinates; tested starting losses and evaluation metrics agree.</p>
<p><strong>The full training trajectories are not bitwise identical.</strong> Transported Adam’s installed weights and checked logits match native Adam exactly at sampled steps 1, 4 and 12, but differ at step 24. The small-step SGD control also accumulates differences. These are measured numerical divergences between mathematically equivalent implementations; this study does not isolate their propagation mechanism.</p>
{eqtable}
<p>At step 24, transported Adam differs from native Adam in mean validation NLL by −0.0000276 nats and in mean copying accuracy by −0.1344 percentage points. The stable SGD pair differs by +0.0000105 nats, with identical mean copying accuracy. Aggregate agreement therefore coexists with nonzero individual logit differences.</p>
<p><strong>Saved-model verification:</strong> both saved native-Adam and coefficient-Adam decoders were independently loaded into fresh checkpoint models. Every recorded validation-block loss, checked retention metric and retained hidden vector reproduced exactly. Re-encoding their saved weights through Haar and back preserved all installed float32 weights and the logits on the checked validation block exactly. This verifies those saved artifacts, not all future update trajectories.</p>
<h3>The original SGD setting was unstable</h3>
<p>The predeclared SGD learning rate 0.005 produced a first update with norm 0.37824 and maximum coordinate change 0.03840. Native validation loss immediately rose from 4.741 to 7.392, and copying fell below 1%. Both native and curve arms were unstable, with later trajectories diverging strongly. Those runs are retained without replacing their results.</p>
{sgdtable}
{figure('sgd_stability_control','Original unstable SGD and the smaller-step SGD stability control.','The two panels deliberately use different y-axis ranges. The additional pair repeats the same data at a rate chosen from the initial training gradient. It is not an independent confirmation or an equal-learning-speed comparison with Adam.')}
<p>A disclosed addendum adds a second SGD pair, fixing its learning rate to 2 × 10⁻⁵ divided by the maximum absolute gradient on the first training batch: 2.60431694 × 10⁻⁶. This caps the largest first-step coordinate change at Adam’s first-step maximum. The trigger included observed instability; the rate itself was calculated from the training gradient, without searching validation scores. The original protocol and all original results remain intact. The extra pair uses 48 steps, bringing Pythia’s total to 168.</p>

<h2 id="depth">Shared curves across depth: an explicit toy construction</h2>
<p>The second experiment separates a curve <em>over layers</em> from the complete within-tensor representation. It uses all 16 inputs (a,b,c,d) ∈ {{−1,+1}}⁴ and six binary functions: a, ab, conditional selection between a and b by c, abcd, majority(a,b,c), and bcd. All 96 input-target decisions are trained and evaluated exhaustively. These tests give finite functional coverage, not held-out semantic generalization.</p>
<p>Each model has a 4→16 tanh input map, six residual stages and a six-output readout. Weight fields are W(s) = Σⱼ φⱼ(s)Cⱼ and b(s) = Σⱼ φⱼ(s)dⱼ, where φ₀ = 1/√6 and φⱼ = √(2/6) cos(πjs). All six modes exactly span the original layer weights at the six midpoint locations. Two-mode models retain only modes 0 and 1, keeping full resolution within each weight matrix.</p>
<div class="equation">h<sub>i+1</sub> = h<sub>i</sub> + tanh(W(s<sub>i</sub>)h<sub>i</sub> + b(s<sub>i</sub>)) / N, &nbsp; s<sub>i</sub> = (i + ½) / N.</div>
<p>The continuous depth field is executed at integer step counts N = 3, 4, 6, 8, 12 and 24. Native off-grid evaluations use the full six-mode interpolant through the trained native layers. This is an Euler-discretized residual model, not a reproduction of the project’s full GSA attention or memory architecture.</p>
{figure('toy_shared_depth_limits','Toy projection changes logits without changing classification, depth changes logits, and most labels remain correct after all residual stages are removed.','Panels B and C show the four from-scratch arms. B reports means over five paired seeds; full per-seed outputs are included. C is a disclosed post-hoc, no-training diagnostic: dots are seeds and bars are means. The other three continuation arms are listed below.')}
<p>All seven final arms reach 100% accuracy at six steps in every seed. All tested final depth/arm combinations also reach 100%, except the full six-mode curve trained from scratch at depth 3, which makes one error across 480 decisions. This apparent success is limited by the small, easy task set.</p>
{toytable}
<p class="note">Final models: 1,200 updates for from-scratch arms; 400 updates from a native step-800 teacher for conversion arms. All optimizers are float64 full-batch Adam, learning rate 0.01; scratch arms also reset optimizer state at step 800. Multi-depth training cycles 3/6/12, averaging seven executed stages per update versus six in the fixed-depth arms, so compute is not matched. Depths 4/8/24 are unseen execution counts for that arm. Inactive frozen bridge buffers are excluded from final active parameter counts.</p>
<h3>Conversion changes confidence before it changes decisions</h3>
<p>The exact six-mode conversion and the initial bridge match the teacher’s logits to below 10⁻¹¹. Hard two-mode projection discards 26–33% of the residual weight-matrix energy and changes logits substantially, yet already preserves 100% classification accuracy in all seeds at every tested depth, before any continuation. There is therefore no classification deficit here for the bridge to repair.</p>
{convtable}
<p class="note">Discarded energy above is calculated for residual weight matrices, excluding biases. The gradient diagnostic includes both residual matrices and biases, measured at the hard-projected model, and asks how much gradient energy lies outside the retained two-mode subspace. It is not a measure of uniquely lost capability: input/output maps and other retained directions can compensate.</p>
<p>The bridge starts with the discarded modes present as a frozen residual, anneals their multiplier from 1 to 0 over 200 updates, and continues for another 200. Its final architecture equals the hard-conversion model. This run provides no evidence that the bridge improves retention. Likewise, the small difference between two-mode fixed-depth and multi-depth logit drift does not establish a general benefit from multi-depth training.</p>
<div class="callout caution"><strong>Why the toy success is inconclusive.</strong> A post-hoc check removed every residual stage without retraining. The learned input and output maps still solve 87.5–100% of decisions across all final arms and seeds. The tests do not force substantial depth-dependent computation. They cannot settle whether a restricted GSA regime loses a capability available to vanilla models. Perfect classification also hides meaningful confidence and off-grid function changes.</div>

<h2 id="meaning">What this establishes for the curve-space goal</h2>
{table(['Question','Evidence from this phase','Remaining boundary'],[
 ['Does classical training make exact curve encoding incompatible?','No incompatibility appeared in the complete native-order representation. Real coefficient gradients support learning; independent reload checks pass.','This retains the original architecture and all degrees of freedom. It does not test a restricted GSA function class.'],
 ['Can continuing training damage retained behavior?','Yes. Ordinary and coefficient Adam both improve text loss while degrading copying and the prose probe.','Conversion alone does not explain the damage. There is no replay or teacher-preservation objective here.'],
 ['Does coefficient-space training behave differently?','Yes. Adam changes update direction despite almost identical first-step norm.','One transform, fixed learning rates and a single short data stream cannot identify an optimal curve optimizer.'],
 ['Do shared depth curves lose important functions?','The small two-mode models fit every finite toy label. Projection changes logits even when labels survive.','The toy tasks are too easy; no conclusion about long-context recall, compositional depth or full GSA expressivity follows.'],
 ['Does continuous parameter depth give interchangeable execution depths?','A defined depth field permits multiple execution counts, with high toy accuracy.','Exact function preservation is only built in at the original sample grid. Logits vary with execution depth; continuous adaptive stopping was not tested.']])}
<p>The complete representation is an appropriate starting point for curve-level training research. Imposing shared low-dimensional fields or changing the memory/attention operator is a separate architectural step. Failures there could reflect representational restrictions, insufficient optimization, or an unstable conversion path; this phase does not conflate those possibilities.</p>
<p><strong>The next informative experiment</strong> is a bounded retention-aware continuation comparison: native versus complete-curve updates with the same replay or teacher-preservation budget, then selected multiscale coefficient constraints. Judge new-task learning at comparable retained performance. In parallel, replace the saturated toy suite with sequence tasks that require explicit binding, composition or memory, including length generalization and a strong shallow control. Only then compare hard conversion, a residual bridge and training from scratch under matched parameter and execution budgets. These follow-up experiments have not been run in this bundle.</p>

<h2 id="reproduce">Methods, provenance and reproduction</h2>
<p>The starting model is <a href="https://huggingface.co/EleutherAI/pythia-70m/tree/de3e4e2d6cbb3b1a51f90fe153ea51fcd8c7c852">EleutherAI/Pythia-70m, immutable commit de3e4e2d6cbb3b1a51f90fe153ea51fcd8c7c852</a>, step 143000. The model file SHA-256 is <code>737de0ce1e32a9559925b452921cdea0b12d3932133b6b0b70b61a351b9f6272</code>. Other checkpoints were examined in preceding phases; this phase uses only the final checkpoint.</p>
<p>Continuation data comes from <a href="https://huggingface.co/datasets/Salesforce/wikitext/tree/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-2-raw-v1">WikiText-2 raw, pinned revision b08601e04326c79dfdd32d625aee71d232d685c3</a>. The deterministic selection contains 48 training blocks and 24 validation blocks of 96 tokens from distinct source rows, with seed 2026101709. Every arm sees the same 24 batches once: 4,608 token positions or 4,560 next-token targets. Validation contains 2,280 targets. WikiText is not verified absent from Pythia’s original pretraining.</p>
<p>The copying probes contain 24 newly generated sequences, excluding token IDs used by the preceding intervention probes. The eight original prose passages contain 284 next-token targets and are reused from that phase. Synthetic successor copying is a targeted behavioral probe, not a full language-capability evaluation. The initial checkpoint, dataset hashes, exact selected token IDs, probe IDs, per-item scores, trajectories, coefficients for the toy models, final native/curve-Adam decoder weights, and all runnable source are included or pinned.</p>
<p>Timing totals are {S['runtime']['pythia']:.2f} seconds for the original Pythia arms, {S['runtime']['stable_sgd']:.2f} seconds for the additional SGD pair and {S['runtime']['toy']:.2f} seconds for the toys. These are component wall times; some execution overlapped. Pythia uses three CPU threads and the toys one. There is no GPU run, no broad sweep and no claim of curve execution speedup: matrices are decoded and multiplied densely.</p>
<p>The environment used Python 3.12.14, PyTorch 2.14.1+cpu, NumPy 2.3.5, Transformers 4.57.1, Safetensors 0.8.0, PyArrow 19.0.1 and Matplotlib 3.10.8. Numerical identity across other software or hardware is not guaranteed. A rerun overwrites result files, so use a copy of the extracted bundle.</p>
<pre>python download_sources.py                 # pinned base model; exact token inputs are bundled
python pythia_train.py                     # original five arms, including unstable SGD
python stable_sgd.py                       # disclosed additional pair
python toy_depth.py
python analyze.py
python verify_saved.py
python make_figures.py
python build_report.py</pre>
<p class="note">Optional source-data reconstruction: run <code>python download_sources.py --dataset</code> and then <code>python prepare_data.py</code>. The ZIP includes SHA256SUMS.json and verify_bundle.py. It omits the downloadable 281.7 MB base checkpoint, raw WikiText parquet files and large temporary comparison snapshots. It retains final adapted decoder files and all reported numerical outputs.</p>
<p class="note">Data attribution: WikiText by Stephen Merity, Caiming Xiong, James Bradbury and Richard Socher; underlying text by Wikipedia contributors. The dataset card lists CC BY-SA 3.0 / GFDL. See the bundled DATA_ATTRIBUTION.txt and pinned row metadata for the included tokenized selections. Source model: EleutherAI/Pythia. This report’s empirical claims are derived from the accompanying measurements.</p>
<footer>Scope: a short continuation compatibility study and five-seed finite toy controls. No full GSA architecture reproduction, no general continual-learning guarantee, no demonstrated loss of an important vanilla-only capability, and no claim that sampled-depth accuracy means identical functions.</footer>
</main></body></html>'''
(ROOT/'curve_training_compatibility_report.html').write_text(body,encoding='utf-8')
print('Built curve_training_compatibility_report.html',len(body.encode()),'bytes')
