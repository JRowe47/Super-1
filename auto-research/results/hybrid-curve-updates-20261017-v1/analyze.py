"""Deterministic descriptive analysis of frozen evidence; no model execution."""
import hashlib
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
ARMS = ['backprop', 'hybrid', 'gated_skip', 'confidence_only', 'always_hyper']
LABELS = ['Backprop', 'Guarded hybrid', 'Gated skip', 'Confidence only', 'Always predicted']

def read(path):
    return json.loads(path.read_text())

def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    protocol = read(ROOT / 'PROTOCOL.json')
    registration = read(ROOT / 'REGISTRATION.json')
    assert digest(ROOT / 'PROTOCOL.json') == registration['protocol_sha256']
    assert digest(ROOT / 'experiment.py') == registration['source_sha256']
    freeze = read(ROOT / 'CHECKPOINT_FREEZE.json')
    for name, expected in freeze['checkpoints'].items():
        assert digest(ROOT / name) == expected, name
    confirmation = read(ROOT / 'CONFIRMATION.json')
    assert confirmation['exact_replays']
    rows = []
    for score in confirmation['rows']:
        folder = ROOT / 'main' / f"s{score['seed']}"
        run = read(folder / f"{score['arm']}_summary.json")
        rows.append(dict(**score, **{k: v for k, v in run.items() if k not in score}))
    seeds = protocol['seeds']
    assert len(rows) == len(seeds) * len(ARMS)
    for seed in seeds:
        paired = [r for r in rows if r['seed'] == seed]
        assert len({r['data_sha256'] for r in paired}) == 1
        assert len({r['confirmation_input_sha256'] for r in paired}) == 1
    by = {(r['seed'], r['arm']): r for r in rows}
    aggregates = {}
    for arm in ARMS:
        a = [by[seed, arm] for seed in seeds]
        metrics = {k: [r[k] for r in a] for k in ['old_retention', 'new_atomic', 'old_ce', 'new_ce', 'main_cpu', 'lifecycle_cpu']}
        metrics.update({f'composition_{n}': [r['composition'][str(n)] for r in a] for n in [2, 4]})
        metrics['backward_reduction'] = [1 - r['counts']['backward'] / by[r['seed'], 'backprop']['counts']['backward'] for r in a]
        metrics['main_cpu_ratio'] = [r['main_cpu'] / by[r['seed'], 'backprop']['main_cpu'] for r in a]
        metrics['lifecycle_cpu_ratio'] = [r['lifecycle_cpu'] / by[r['seed'], 'backprop']['lifecycle_cpu'] for r in a]
        aggregates[arm] = {k: dict(mean=st.mean(v), minimum=min(v), maximum=max(v), per_seed=v) for k, v in metrics.items()}
        aggregates[arm]['total_counts'] = {k: sum(r['counts'][k] for r in a) for k in a[0]['counts']}
    calibration = []
    for seed in seeds:
        folder = ROOT / 'main' / f's{seed}'
        c = read(folder / 'calibration.json')
        audits = [r for r in read(folder / 'confidence_only_trace.json') if r['audit'] is not None]
        assert len(audits) == by[seed, 'confidence_only']['counts']['audits']
        calibration.append(dict(seed=seed, threshold=c['rule']['threshold'], loss_limit=c['rule']['loss_limit'],
            calibration_count=len(c['rows']), calibration_safe=sum(r['safe'] for r in c['rows']),
            calibration_confidence=st.mean(r['confidence'] for r in c['rows']),
            calibration_relative_update_error=st.mean(r['relative_update_error'] for r in c['rows']),
            audited_count=len(audits), audited_safe=sum(r['audit']['safe'] for r in audits),
            audited_confidence=st.mean(r['confidence'] for r in audits)))
    audited_count = sum(r['audited_count'] for r in calibration)
    audited_safe = sum(r['audited_safe'] for r in calibration)
    audited_confidence = sum(r['audited_confidence'] * r['audited_count'] for r in calibration) / audited_count
    h = aggregates['hybrid']
    tests = dict(
        composition_noninferiority=all(by[s, 'hybrid']['composition']['2'] >= by[s, 'backprop']['composition']['2'] - .02 for s in seeds),
        retention_noninferiority=all(by[s, 'hybrid']['old_retention'] >= by[s, 'backprop']['old_retention'] - .01 for s in seeds),
        backward_reduction=h['backward_reduction']['mean'] >= .25,
        main_cpu_reduction=h['main_cpu_ratio']['mean'] <= .9,
        added_value_over_skip=(h['composition_2']['mean'] - aggregates['gated_skip']['composition_2']['mean'] >= .02 or
                               aggregates['gated_skip']['new_ce']['mean'] - h['new_ce']['mean'] >= .02))
    assert not all(tests.values())
    cpu = sum(read(ROOT / 'main' / f's{s}' / 'TERMINAL.json')['cpu_seconds'] for s in seeds)
    summary = dict(run_id=ROOT.name, decision='NO_PROMOTION', status='COMPLETE', seeds=seeds,
        descriptive_only=True, aggregates=aggregates, rows=rows, calibration=calibration,
        confidence_shift=dict(mean_audited_confidence=audited_confidence, audited_pass_fraction=audited_safe / audited_count,
                              audited_pass_count=audited_safe, audited_count=audited_count),
        registered_criteria=tests, exact_checkpoint_replays=True, main_worker_total_cpu_seconds=cpu,
        notes=['Pilot excluded from main aggregation.', 'Ratios are means of paired per-seed ratios.',
               'Controller frozen after warmup; accepted main updates do not use autograd.',
               'Composition order is supplied by a fixed external executor.',
               'No confidence intervals or significance claims from three seeds.'])
    save(ROOT / 'SUMMARY.json', summary)
    save(ROOT / 'TERMINAL.json', dict(status='COMPLETE_NO_PROMOTION', decision='NO_PROMOTION',
        arm_seed_rows=len(rows), main_seeds=seeds, numerical_qualification='PASS_ORIGINAL_TOLERANCES',
        evaluation='15 final checkpoints; exact score replay', registered_criteria=tests,
        protocol_sha256=registration['protocol_sha256'], source_sha256=registration['source_sha256'],
        main_worker_total_cpu_seconds=cpu, resources='CPU only, 3 one-thread workers',
        publication_scope='Existing draft PR in JRowe47/Super-1; no merge'))

    graph = REPO / 'auto-research' / 'graphs' / ROOT.name
    graph.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    colors = ['#334155', '#0f766e', '#64748b', '#d97706', '#be123c']
    specifications = [('composition_2', 'Two-operation composition (%)', 100),
                      ('old_retention', 'Initial-fact retention (%)', 100),
                      ('backward_reduction', 'Main backward passes avoided (%)', 100),
                      ('main_cpu_ratio', 'Main CPU relative to backprop', 1)]
    for ax, (metric, title, scale) in zip(axes.flat, specifications):
        means = [aggregates[a][metric]['mean'] * scale for a in ARMS]
        ax.barh(LABELS, means, color=colors, alpha=.78)
        for i, arm in enumerate(ARMS):
            vals = np.array(aggregates[arm][metric]['per_seed']) * scale
            ax.scatter(vals, np.full(3, i) + [-.08, 0, .08], s=20, color='#111827', zorder=3)
            ax.text(max(float(max(vals)), means[i], .001) + (.025 if scale == 1 else 1.3), i, f'{means[i]:.2f}' if scale == 1 else f'{means[i]:.1f}', va='center', fontsize=9)
        ax.invert_yaxis(); ax.set_title(title, loc='left', fontweight='bold')
        ax.set_xlim(0, 2.5 if scale == 1 else 112)
        ax.grid(axis='x', alpha=.15); ax.set_axisbelow(True)
        if metric == 'main_cpu_ratio': ax.axvline(1, linestyle='--', color='#111827', lw=1)
    fig.suptitle('Full Fourier parameters: learned-update pilot follow-up\nThree paired worlds; dots show individual seeds', fontsize=15)
    fig.savefig(graph / 'comparison.png', dpi=170, metadata={'Description': 'Generated from SUMMARY.json; descriptive three-seed means and individual points.'})
    fig.savefig(graph / 'comparison.svg')
    plt.close(fig)

    mean_table = '| Arm | Two-step accuracy | Four-step accuracy | Old facts | New facts | Backward passes avoided | Main CPU / baseline | Lifecycle CPU / baseline |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
    for arm, label in zip(ARMS, LABELS):
        a = aggregates[arm]
        vals = [f"{100*a[k]['mean']:.2f}%" for k in ['composition_2', 'composition_4', 'old_retention', 'new_atomic', 'backward_reduction']]
        vals += [f"{a[k]['mean']:.2f}×" for k in ['main_cpu_ratio', 'lifecycle_cpu_ratio']]
        mean_table += '| ' + ' | '.join([label] + vals) + ' |\n'
    seed_table = '| Seed | Arm | Two-step | Four-step | Old facts | New-fact CE | Main CPU (s) | Backward passes | Predicted updates applied |\n|---|---|---:|---:|---:|---:|---:|---:|---:|\n'
    for r in rows:
        seed_table += f"| {r['seed']} | {r['arm']} | {100*r['composition']['2']:.2f}% | {100*r['composition']['4']:.2f}% | {100*r['old_retention']:.2f}% | {r['new_ce']:.5f} | {r['main_cpu']:.6f} | {r['counts']['backward']} | {r['counts']['applied']} |\n"
    cal_table = '| Seed | Calibration pass | Mean calibration confidence | Relative update error | Loss support cap | Shifted audit pass | Mean shifted confidence |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for c in calibration:
        cal_table += f"| {c['seed']} | {c['calibration_safe']}/{c['calibration_count']} | {100*c['calibration_confidence']:.3f}% | {c['calibration_relative_update_error']:.3f} | {c['loss_limit']:.6f} | {c['audited_safe']}/{c['audited_count']} | {100*c['audited_confidence']:.3f}% |\n"
    report = f'''# First full-Fourier / hybrid-update experiment

The complete Fourier representation passed its frozen fidelity checks. The learned updater did **not** meet the registered success rule. The guarded hybrid used backprop on every main step, preserved baseline quality, and cost {h['main_cpu_ratio']['mean']:.2f}× as much CPU. Dropping the loss-support guard avoided {100*aggregates['confidence_only']['backward_reduction']['mean']:.1f}% of main backward passes but reduced composition accuracy and still increased runtime. **Decision: no promotion.**

This is a deliberately narrow synthetic test of update prediction and qualification. It is not an LLM result or a test of the complete hierarchical-memory proposal.

## Design and scope

Two affine layers, 16→32→8 with tanh, contain 808 real parameters including biases. Each layer is represented by its **complete real orthonormal Fourier coordinates** (544 and 264). No frequencies are dropped, no smoothness regularizer is added, and no parameter family is removed. This establishes faithful coordinates in this model; it does not establish that FFTs make arbitrary dense layers cheaper or preserve all capabilities of an LLM.

Each seeded world contains eight random permutations of eight values. Warmup learns operators 0–3. Four main phases introduce operators 4–7, 250 steps each. Every batch of 64 contains 50% new-operator examples and 50% replay. Each arm sees the same 64,000 main examples, initial learner parameters and final evaluation inputs within its seed. SGD uses learning rate 0.5, with no momentum or weight decay.

An 898-parameter shared update network learns factors that assemble complete updates for both affine layers. Its inputs are forward activations, current output-layer weights, probabilities and observed target labels. True derivative factors supervise warmup but never enter predictor inputs. Predicted updates are norm-limited to 0.5. A 339-parameter gate predicts one-step acceptance, post-update query accuracy, and query loss change. It observes loss/entropy, norms, accuracy and two exponentially averaged audit-error summaries. The second summary measures disagreement between predicted and observed loss change—a limited operational version of dissonance. Post-update accuracy prediction is not a general calibrated state-uncertainty estimator; a separate learned surprise head is not implemented.

The learner, predictor and gate train together for 1,200 warmup steps. Calibration then uses 256 steps with the controller frozen and learner backprop active. The controller remains frozen during main evaluation to isolate transfer of update prediction. Ongoing controller training on fallback, token-context memory, multiple persistent memory timescales, learned hierarchy, variable depth and neural ODEs are deferred. The two audit EMAs are not a substitute for those components.

The pilot used seed 90917 and four arms. Inspection motivated one added diagnostic, confidence-only, before freezing the main protocol. The model, optimizer, tolerances and training lengths were unchanged. Main seeds 101721–101723 were fixed before execution. Registration was local, not an externally timestamped preregistration. Pilot evidence and its original source are preserved separately and excluded from main aggregates.

## Qualification and controls

A candidate passes a one-step query audit if its cross-entropy is at most 0.01 nats above the true SGD counterfactual on **both** a current-operator minibatch and an initial-operator minibatch. This labels tolerable effect, not exact gradient reconstruction or improvement over doing nothing. Queries are additional supervised computation and are included in measured cost; they are not free feedback or the final confirmation set.

Calibration selects the lowest of fixed thresholds 0.5/0.7/0.8/0.9/0.95/0.99 with at least 64 accepted examples and a one-sided Wilson lower bound ≥0.95. Temporally related samples make this a heuristic qualification, not an IID reliability guarantee. The guarded hybrid also requires training loss no larger than the maximum calibration loss. An independent 10% audit mask checks otherwise eligible steps.

- **Backprop:** ordinary learner SGD in the same complete Fourier coordinates.
- **Guarded hybrid:** confidence plus loss-support guard; sampled rejected audits fall back to SGD.
- **Gated skip:** identical predictor/gate cost, but an accepted step makes no update. It measures whether proposed edits add value over skipping.
- **Confidence only:** removes the loss-support guard, retaining confidence and audit fallback.
- **Always predicted:** applies every predicted update, even if a sampled audit rejects it. Audits are diagnostic and still charged.

Predicted main applications run without an autograd graph. Avoided backward passes count learner backward calls; they do not claim zero warmup training or zero supervision.

## Results

{mean_table}
Ratios are the mean of paired per-seed CPU ratios. Main CPU covers data generation, prediction, gating, audits, learner updates and progress collection. It excludes post-run checkpoint serialization and final confirmation. Lifecycle CPU adds the appropriate measured warmup: plain learner warmup for backprop, joint controller warmup for the other arms. Both startup paths produce exactly equal learner coefficients. Timing is descriptive: one thread per worker, three concurrent seed workers, fixed arm order, CPU only. The tiny learner makes controller overhead especially visible; these ratios do not predict GPU/LLM scaling.

![Comparison](../graphs/{ROOT.name}/comparison.png)

The guarded hybrid and gated-skip control had **zero eligible steps**, so both exactly match backprop's final metrics. They demonstrate fallback protection, not successful forward-only learning. Their zero acceptance also leaves the value of accepted updates versus no-op untested. The confidence-only arm retained 96.875% of initial facts but learned only 59.375% of new facts on average. Failures compound when those facts are chained. Always-predicted updates also substantially damaged old knowledge.

Confirmation scores all 64 atomic facts, all 512 two-operation combinations, and 2,048 seeded four-operation chains. Atomic facts overlap training by design; composed chains receive no training loss. A fixed external executor supplies the operation order. The model does not learn a parser, discover a semantic graph or choose its own reasoning depth. Baseline composition reaches a ceiling, so equal scores are not evidence of a quality improvement.

## Confidence failed under the curriculum shift

{cal_table}
Across the confidence-only audits, mean predicted acceptance was **{100*audited_confidence:.3f}%**, while only **{audited_safe}/{audited_count} ({100*audited_safe/audited_count:.2f}%)** met the registered one-step test. These are sampled audits across dependent trajectories, not an independent-binomial estimate.

All 256 calibration candidates passed in each seed, even though mean relative parameter-update error was roughly 1.0–1.2. Calibration took place near convergence, where a 0.01-nat tolerance can admit very inaccurate or nearly inert updates. That success did not predict ability to learn new operators. The loss-support guard correctly rejected the shifted regime, but it rejected every main step. Confidence alone did not detect the mismatch. A 10% fallback audit can replace a current bad proposal; it cannot undo accumulated unaudited errors.

This supports a concrete next hypothesis: train and calibrate over full learning trajectories and task changes, including confident mistakes and substantial beneficial updates; require measured usefulness versus no-op as well as retention relative to SGD. Compare a frozen controller with one that continues teacher training on fallback. These are follow-up proposals, not results of this run.

## Registered decision and integrity

The primary guarded-hybrid comparison had to stay within 2 percentage points of baseline composition and 1 point of old-fact retention in every seed, avoid at least 25% of main backward passes on average, and reduce mean main CPU by at least 10%. Added value over gated skip required a 2-point composition gain or 0.02-nat new-fact CE reduction. Quality limits passed; backward savings, CPU savings and added-value criteria failed. No thresholds were changed after main results.

Original FP32 absolute tolerance was 2e-6 and FP64 tolerance 2e-12. Checks covered odd/even FFT roundtrips, norm preservation, dense/curve forward outputs, input and parameter gradients, and one SGD update. Maximum FP32 roundtrip error was 7.153e-7; maximum parameter-gradient error 3.726e-8; maximum one-step weight error 1.491e-7. Teacher-factor assembly agreed with autograd to 6.520e-9. Nonzero predicted updates were also verified to change parameters with autograd disabled. All checks passed the original thresholds without relaxation.

All 18 main checkpoints (warmup plus five arms per seed) were hashed before confirmation. All 15 final checkpoints were reloaded and scored twice with identical metrics. This establishes reproducibility, not additional independent evidence. All main workers exited successfully, using {cpu:.3f} total process CPU seconds; the launcher took 17.214 wall seconds. Three seeds support descriptive paired results, not broad statistical or architectural claims.

A post-main diagnostic, `COMPUTE_DIAGNOSTIC.json`, compares dense and Fourier forward/backward, proposed SGD updates **and dense materialization of the updated curve**. Median curve/dense times at batches 1/64/256 were 0.550/0.181, 0.583/0.214 and 0.660/0.292 ms. This includes an extra re-export and is not the main training-step measurement. It shows no speed advantage for this unoptimized small-model route. No wavelet, sparse-band, basis-layout or variable-depth speedup was tested. Frequency in a flattened parameter vector should not be interpreted as semantic abstraction without separate evidence.

## Per-seed evidence

{seed_table}

Source, frozen protocols, calibration/audit traces, all scores and terminal receipts are in `auto-research/results/{ROOT.name}/`. The checkpoint companion contains the five pilot and eighteen main snapshots, with checksums; extract it into that result directory for evaluation replay. See that directory's `README.md` and `PRESERVATION.json`. Earlier curve/minLSTM studies and their caveats remain separate and unchanged.
'''
    report_path = REPO / 'auto-research' / 'reports' / f'{ROOT.name}.md'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)
    print(json.dumps(dict(status='ANALYZED', decision='NO_PROMOTION', main_cpu_seconds=cpu,
                          confidence=summary['confidence_shift'], report=str(report_path)), indent=2))

if __name__ == '__main__':
    main()
