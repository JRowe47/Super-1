"""Build a self-contained, auditable scientific report from saved results."""
import base64,html,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def read(n):return json.loads((ROOT/'results'/n).read_text())
def table(headers,rows):return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+h+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def interval(x,d=3):return f"{x['estimate']:+.{d}f} [{x['ci95'][0]:+.{d}f}, {x['ci95'][1]:+.{d}f}]"
def figure(name,alt,caption):
    data=base64.b64encode((ROOT/'figures'/f'{name}.png').read_bytes()).decode()
    return f'<figure><img src="data:image/png;base64,{data}" alt="{html.escape(alt)}"><figcaption>{caption}</figcaption></figure>'
LABEL={'baseline':'Starting checkpoint','full':'All coefficients','targeted':'Targeted mask','random0':'Random mask 1','random1':'Random mask 2','random_mean':'Random masks, mean','opposite':'Opposite mask'}
def main():
    effects=read('effects.json');base=effects[0];final={r['arm']:r for r in effects if r['step']==24};cons=read('contrasts.json')
    get=lambda step,a,b:next(r['contrast'] for r in cons if r['step']==step and r['a']==a and r['b']==b)
    primary=get(24,'targeted','random_mean');early=get(1,'targeted','random_mean');counts=read('mask_calibration.json');env=read('environment.json');toyenv=read('transport_environment.json');checks=read('analysis_checks.json');audit=read('saved_model_audit.json')
    toy=read('transport_summary.json');oracle=read('transport_oracle.json');maskcount=counts['enabled']['targeted'];allcount=counts['total_coefficients']
    fnl=[base]+[final[a] for a in ['full','targeted','random0','random1','opposite']]
    endpoints=table(['Run','Text NLL ↓','Copy accuracy ↑','Copy NLL ↓','Long copy accuracy ↑','Reused prose ΔNLL ↓'],[[LABEL[r['arm']],f"{r['validation_nll']:.4f}",f"{100*r['repeat']['accuracy']:.2f}%",f"{r['repeat']['target_nll']:.3f}",f"{100*r['repeat_long']['accuracy']:.2f}%",f"{r['changes']['prose_nll']['estimate']:+.4f}"] for r in fnl])
    temporal=table(['Targeted − random mean','Copy accuracy, pp ↑','Copy NLL ↓','Text NLL ↓'],[[f'Update {s}',interval(get(s,'targeted','random_mean')['repeat']['accuracy'],2),interval(get(s,'targeted','random_mean')['repeat']['target_nll']),interval(get(s,'targeted','random_mean')['validation_nll'])] for s in [1,12,24]])
    extra=table(['Final contrast','Copy accuracy, pp ↑','Text NLL ↓','Scrambled copy accuracy, pp ↑'],[[f'{LABEL[a]} − {LABEL[b]}',interval(get(24,a,b)['repeat']['accuracy'],2),interval(get(24,a,b)['validation_nll']),interval(get(24,a,b)['scrambled']['accuracy'],2)] for a,b in [('targeted','full'),('targeted','opposite'),('opposite','random_mean'),('random_mean','full')]])
    toytable=table(['Learned curve','Seeds','MSE at grid 17, mean [min, max]','Max grid drift','Auxiliary energy, max'],[[LABEL_,5,f"{r['expected_mse']['mean']:.2e} [{r['expected_mse']['min']:.2e}, {r['expected_mse']['max']:.2e}]",f"{max(x['max_drift_from_12_pulse']['max'] for x in toy if x['arm']==r['arm']):.2e}",f"{r['auxiliary_energy']['max']:.2e}"] for LABEL_,r in [('12 independent controls',next(r for r in toy if r['arm']=='full12' and r['grid']==17)),('2 shared controls',next(r for r in toy if r['arm']=='shared2' and r['grid']==17))]])
    solverrows=[]
    for n in [12,17,24,31,96]:
        d={r['solver']:r for r in oracle if r['grid']==n};solverrows.append([n,d['exact']['rotation_operations'],f"{d['exact']['expected_mse']:.2e}",f"{d['midpoint']['expected_mse']:.4g}",f"{d['euler']['expected_mse']:.4g}"])
    solvertable=table(['Requested grid N','Exact local operations','Exact-rotation MSE','Midpoint-rotation MSE','Euler MSE'],solverrows)
    displacement={a:next(r['total_update_norm'] for r in read('haar_band_summary.json') if r['arm']==a) for a in ['full','targeted','random0','random1','opposite']}
    normtable=table(['Run','Final weight-displacement L2','Step multiplier, minimum–maximum'],[[LABEL[a],f'{displacement[a]:.5f}',f"{min(r['norm_scale'] for r in read('training.json') if r['arm']==a):.3f}–{max(r['norm_scale'] for r in read('training.json') if r['arm']==a):.3f}"] for a in displacement])
    css=(ROOT/'report_style.css').read_text()
    report=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Curve update masks and a constructive depth remedy</title><style>{css}</style></head><body><main>
<div class="eyebrow">Native-order curve research · continuation 04</div>
<h1>Local targeting works briefly.<br>A richer local flow fixes the depth toy.</h1>
<p class="lead">Static recursive masks do not preserve their initial advantage during continued Pythia training. Separately, a real two-channel construction preserves the depth-dependent routing computation that the previous scalar flow could not represent.</p>
<p class="meta">17 October 2026 · Pythia-70m, step143000 · 120 new model updates · 4,000 tiny toy updates · fixed controls, fresh primary evaluation inputs</p>
<nav><a href="#findings">Measured outcomes</a><a href="#masks">Mask experiment</a><a href="#depth">Depth construction</a><a href="#meaning">Implications</a><a href="#reproduce">Reproduce</a></nav>
<section id="findings"><h2>What this phase establishes</h2>
<div class="callout caution"><strong>The useful negative result:</strong> the targeted mask beats the random-mask mean by {early['repeat']['accuracy']['estimate']:.2f} percentage points of copying accuracy after one update, but trails it by {-primary['repeat']['accuracy']['estimate']:.2f} points after 24 updates. At the endpoint it also has higher text NLL ({primary['validation_nll']['estimate']:+.4f}). A useful local gradient signal is not a stable update policy.</div>
<p>The mask comparison controls both the number of enabled coefficients within every original tensor/wavelet band and the global size of each optimizer update. Every coefficient remains stored at full resolution. This rules out a simple “the targeted model changed fewer coefficients or took smaller steps” explanation for its difference from the random masks.</p>
{endpoints}
<p class="note">Copy accuracy is the mean over 48 fresh sequences of lengths 16, 32 and 48; long copying uses only six sequences and is descriptive. Text NLL is the mean over 48 fresh, equal-length 96-token blocks. Lower NLL is better. These new inputs differ from the preceding phase, so absolute scores should not be compared across phase reports.</p>
{figure('masked_update_tradeoff','Copying and text-loss trajectories for all-coefficient, targeted, random and opposite-mask training.','All runs start from the same checkpoint and use the same adaptation/teacher-replay stream. The random mean averages per-item metrics from two independently selected masks; it is not a model ensemble. All endpoints are reported.')}
<h3>The time course is the main result</h3>
{temporal}
<p class="note">Brackets are paired 95% bootstrap intervals: 4,000 resamples of 48 text blocks or48 copy sequences, stratified by sequence length. They condition on this single training stream and the two sampled random masks. They are not confidence intervals over training seeds or all possible masks. No multiple-comparison correction; the predeclared main comparison is the final targeted-minus-random mean.</p>
{extra}
<p>The opposite-ranked mask is a directional control. Its behavior helps distinguish initial sensitivity from persistent semantic specialization. None of these comparisons establishes that fixed wavelet locations have a universal semantic role.</p></section>
<section id="masks"><h2>Masking updates in complete native-order curves</h2>
<p>Each of the 72 decoder tensors is independently flattened in its original C order and represented by its complete orthonormal Haar pyramid: {allcount:,} coefficients in total. Embeddings, final normalization and the output head remain fixed. Float64 master coefficients decode into the ordinary float32 dense network. This tests curve-coordinate training; it does not replace Pythia attention with geometric self attention.</p>
<p>Within each tensor/detail band, consecutive 64-coefficient tiles are scored using two calibration gradients. The adaptation gradient averages the first six adaptation batches. The copying gradient averages 12 batches of fresh labeled copying probes, with one repeated and one scrambled-query sequence per batch. No held-out test is used to choose the mask.</p>
<div class="equation">score(tile) = Σ<sub>i in tile</sub> g<sub>copy,i</sub> · [−g<sub>adapt,i</sub> / (|g<sub>adapt,i</sub>| + 10<sup>−8</sup>)]</div>
<p>Lower scores predict a more favorable first-order change in copying NLL for a fresh-Adam-style adaptation proposal. The targeted mask allows the lowest-scoring half of tiles; the opposite mask allows the same number of highest-scoring tiles. Two random masks select exactly the same number in every band. Odd tile counts use floor-half. Bands smaller than 128 coefficients remain enabled in all arms.</p>
<p>Each masked arm enables {maskcount:,} coefficients ({100*maskcount/allcount:.3f}% of the complete representation). The {counts['scored_tiles']:,} scored tiles preserve native locations. A level j coefficient has support 2<sup>j</sup> original values; a64-coefficient tile spans 64×2<sup>j</sup> successive values inside its tensor. No neurons, channels or tensor rows are reordered.</p>
<h3>Training and control details</h3>
<p>All five arms take 24 AdamW updates, learning rate 2×10<sup>−5</sup>, β=(0.9,0.999), ε=10<sup>−8</sup>, no weight decay, no dropout and fresh optimizer states. Each sees the same 48 adaptation blocks,4,560 next-token targets, and the same 24 copying/24 text replay examples from the preceding phase. The frozen starting checkpoint supplies the teacher distributions.</p>
<div class="equation">L = L<sub>adaptation</sub> + ½ (KL<sub>copy</sub> + KL<sub>text</sub>)</div>
<p>KL is teacher-to-student over the full vocabulary at temperature 1. Copy replay covers all query positions; text replay uses eight fixed prediction positions per block. Gradients on disabled coefficients are set to zero. After Adam forms its masked proposal, the enabled update is scaled to the unrestricted arm’s L2 norm at that update number. Moments are not rescaled. The largest measured norm discrepancy is {checks['matching_step_norm_max_error']:.2e}; disabled float64 masters remain exactly unchanged.</p>
{normtable}
<p class="note">Matching each step norm does not match cumulative displacement, optimizer direction, or adaptation progress. Dense decoding also means this implementation is not a speed benchmark. The mask-construction method consumes labeled calibration probes; those labels are additional information used by the targeting rule.</p>
{figure('recursive_targeting_diagnostics','Calibration copying-gradient scores, observed first-update copying losses, and final normalized Haar update energies.','The score uses calibration mean gradients and mixed repeat/scrambled copying. The first-update measurement uses the actual first training batch and independent repeated-copy tests, so the two panels test direction rather than exact numerical calibration. Final spectral energy is normalized per coefficient; raw broad-band energy is not called semantic importance.')}
<h3>What the failure does and does not say</h3>
<p>The endpoint rejects this particular static mask as a retention improvement on this run. It does not reject recursive decomposition as a representation or as a source of local targeting information: the immediate held-out intervention shows that the initial signal is useful. A plausible explanation is that the locally useful directions change as the model and Adam state move, but this phase does not directly measure gradient-score aging or prove that refreshing masks will solve it.</p>
<p>Accuracy and target NLL are both included because preserving top-1 choices alone can conceal probability changes. Scrambled-copy probes and the eight reused prose passages are additional diagnostics, not broad semantic coverage. Evaluation rows and synthetic token pools are disjoint from mask selection and replay; WikiText absence from original Pythia pretraining is not established.</p></section>
<section id="depth"><h2>A constructive remedy for the depth failure</h2>
<p>The previous scalar family used h′(s)=[a(s)I+b(s)S]h, where S shifts each ring site to its successor. Its generators commute, so its continuous flow is exp(AI+BS). That real scalar family cannot equal the six-hop shift S<sup>6</sup> on the 12-site ring: unit gain in the zero and Nyquist modes forces A=B=0, leaving identity. The constant discrete oracle is exact at six Euler steps but has continuous-limit expected MSE 0.795814.</p>
<p>Here each site holds two ordinary real numbers, x and y, with y initially zero. Each hop alternates an onsite rotation and a nearest-neighbor rotation:</p>
<pre>Onsite:    xᵢ ← cos(a) xᵢ − sin(a) yᵢ
           yᵢ ← sin(a) xᵢ + cos(a) yᵢ

Neighbor:  xᵢ ← cos(b) xᵢ + sin(b) yᵢ₊₁
           yᵢ ← cos(b) yᵢ − sin(b) xᵢ₋₁

All right-hand sides within one rotation use its input state.
At a=b=π/2, the pair maps (x,y) → (Sx,S⁻¹y).
After six pairs: (x,0) → (S⁶x,0).</pre>
<p>There is no learned global input/readout shortcut, imposed state manifold or quaternion representation. The two generators do not commute. The construction changes both state capacity and the operator family; this experiment does not identify augmentation alone as the necessary ingredient.</p>
<p>For a continuous depth field, put each of the 12 rotation generators inside a disjoint smooth pulse. On interval j/12≤s≤(j+1)/12, let u=12s−j and amplitude g<sub>j</sub>(s)=24θ<sub>j</sub>sin²(πu), with zero amplitude outside. The pulse integrates to θ<sub>j</sub>, and its value and first derivative vanish at its boundaries. The generator curve is C1. Each partial pulse has analytic angle θ<sub>j</sub>[F(u₁)−F(u₀)], where F(u)=u−sin(2πu)/(2π).</p>
<div class="callout"><strong>Exact conversion of this toy operator:</strong> six discrete shifts become 12 local quarter-turn pulses. A solver that splits at pulse boundaries and applies each integrated rotation preserves the same operator on every tested grid, including irregular grids 17 and31. This is an explicit construction for this routing map, not a conversion algorithm for arbitrary Transformer layers.</div>
{figure('augmented_depth_transport','Exact, midpoint, and Euler errors for a two-channel local transport flow; learned solutions and operation counts.','The plotted error floor is for legibility only; raw floating-point errors are saved. A requested one-cell grid still executes all 12 local pulse operations. The construction preserves depth fidelity by respecting the ordered operators, not by obtaining six-hop communication in one local step.')}
<h3>The construction is learnable in this toy</h3>
<p>Five seeds train either 12 pulse angles or two angles shared across the six onsite/neighbor pairs. Both use 400 Adam updates at learning rate 0.05, batch 64 independent Gaussian rings, and fixed 12-pulse execution. Stagewise one-through-six-hop targets and zero-auxiliary targets provide training-only supervision. Initialization is Normal(1,0.1), paired across arms; the shared version uses per-type means. All ten runs are retained.</p>
{toytable}
<p>Evaluation reconstructs the exact linear operator from 12 impulses, giving expected output MSE for iid unit Gaussian inputs. An independent 4,096-example Gaussian set checks empirical MSE and sign accuracy. All ten learned models preserve their own output across every tested grid to a maximum absolute difference of {checks['max_toy_grid_drift']:.2e}. Small residual auxiliary energy is reported separately.</p>
<h3>Variable depth still requires a suitable execution rule</h3>
{solvertable}
<p>Uniform midpoint rotations miss or misintegrate narrow pulses on some grids. Boundary-aware forward Euler uses the correct integrated pulse areas but still distorts rotations severely at these step sizes. These are solver failures in a family that has an exact representation. The successful exact solver exploits the closed form of this toy; arbitrary learned operators may not have one.</p>
<p><strong>Cost:</strong> 12→24 real state values and 6→at least 12 local mixing operations. Irregular grid 17 requires 28 rotations because cells crossing pulse boundaries are split. This demonstrates representability and subdivision consistency, with no claim of equal computation, general adaptive-depth savings or faster inference.</p></section>
<section id="meaning"><h2>Implications for the curve-space goal</h2>
<p><strong>Representation, optimization and execution remain separate questions.</strong> Complete curve coordinates preserve an ordinary model’s stored function at conversion. Subsequent optimization can damage behavior even when every coefficient is retained; this phase adds a controlled example where a locally helpful mask becomes harmful over 24 updates. Changing the execution operator can also remove functions the original model used, as the prior scalar flow did.</p>
<p>The new construction shows that such a failure need not be fundamental to continuous curves. A richer real-state operator with noncommuting local actions can preserve the missing routing function under subdivision. It does not establish equivalence between the user’s GSA architecture and Pythia, or show that any specific pretrained language capability survives that architectural conversion.</p>
<p>The next targeted question is whether feedback can maintain useful update directions: recompute masks on separate calibration data, compare with masks frozen for the same total update budget, and evaluate teacher-distribution preservation rather than only copying-label gradients. The depth counterpart is to test richer operators with state-dependent actions and approximate solvers while charging their actual operation count. Neither improvement is assumed from the current evidence.</p>
<h3>Relationship to prior work</h3>
<p><a href="https://proceedings.mlr.press/v108/farajtabar20a.html">Orthogonal Gradient Descent (Farajtabar et al., 2020)</a> motivates constraining updates to limit interference. Our static wavelet-tile masks are a different method and do not reproduce its projection guarantees. <a href="https://arxiv.org/abs/1904.01681">Augmented Neural ODEs (Dupont et al., 2019)</a> studies how adding state dimensions can enlarge continuous-flow expressivity. The explicit ring construction and solver comparison here are our own small derivation and experiment.</p>
<p class="note">The original project archive uses recurrent bank coordinates and token-conditioned curve fields. Those coordinates are not equated with layer depth here. No full reproduction of the proposed GSA architecture is claimed.</p></section>
<section id="reproduce"><h2>Checks, provenance and reproduction</h2>
<p>The successful Pythia experiment took {env['seconds']:.1f}s with three CPU threads; the toy took {toyenv['seconds']:.1f}s with one thread. This excludes source preparation, analysis, figures, checkpoint reloads and an initial calibration-only attempt. That first attempt stopped before any training/evaluation when a QKV band had an odd number of 64-coefficient tiles. The floor-half rule was clarified before rerunning; details are preserved in execution_notes.json.</p>
<p>Both prespecified saved Pythia checkpoints were reloaded and all evaluated text/copy/prose metrics reproduced with maximum difference {max(max(r['validation_max_error'],r['copy_max_error'],r['prose_max_error']) for r in audit['checks']):.2e}. All per-band mask counts, step norms, frozen masters and 80 learned toy/grid measurements passed their checks. This is bounded verification of the reported experiment, not a claim of universal correctness.</p>
<pre>python verify_bundle.py
python download_sources.py
python run_masks.py --checkpoints checkpoints
python augmented_transport.py
python analyze_phase.py
python make_figures.py
python verify_saved.py --checkpoints checkpoints
python build_report.py</pre>
<p>The bundle includes protocol, source, pinned metadata, input selections, all raw measurements, packed masks, tile scores, two decoded Pythia endpoints, all ten toy models, PNG/SVG figures and SHA256 checksums. Final float64 masters and Adam states are not saved. The large base checkpoint and dataset parquet remain reproducible pinned downloads.</p>
<p class="note">Base model: <a href="https://huggingface.co/EleutherAI/pythia-70m/tree/de3e4e2d6cbb3b1a51f90fe153ea51fcd8c7c852">EleutherAI Pythia-70m</a>, commit de3e4e2d6cbb3b1a51f90fe153ea51fcd8c7c852, SHA256 737de0ce1e32a9559925b452921cdea0b12d3932133b6b0b70b61a351b9f6272. Data: <a href="https://huggingface.co/datasets/Salesforce/wikitext/tree/b08601e04326c79dfdd32d625aee71d232d685c3">WikiText-2 raw</a>, revision b08601e04326c79dfdd32d625aee71d232d685c3. See DATA_ATTRIBUTION.txt for authors and source licenses.</p></section>
<footer>Reproducible continuation of the native-order curve experiments. Claims are scoped to the measured probes, the fixed update policy and the explicitly defined local routing families.</footer>
</main></body></html>'''
    (ROOT/'curve_mask_transport_report.html').write_text(report)
    print('Wrote self-contained report',len(report),'characters')
if __name__=='__main__':main()
