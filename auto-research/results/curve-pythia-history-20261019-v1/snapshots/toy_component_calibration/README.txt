Native curve component calibration — 17 October 2026

Open toy_component_calibration_report.html. It is self-contained and includes
an interactive explorer for every declared component in the illustrative joint
model, seed 1, over the complete step 0 to 800 training interval.

Research orientation: preserve native parameter order and full resolution.
Use known toy behaviors to calibrate component changes and their causal effects.
No permutation optimization, compression benchmark, or curve-architecture
advantage is claimed. The four task settings are trained independently; this is
an incremental progression in task structure, not a sequential curriculum.

Reproduce:
python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python calibrate.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python summarize.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python build_report.py

Inspect a component without fitting:
python inspect_component.py --task joint --seed 1 --component 197 --start 0 --end 800

Files:
protocol.json: fixed experimental design and scope.
calibrate.py: training, all native transforms, all declared interventions.
summarize.py: full reconstruction checks, seed summary, selected dose follow-up.
build_report.py and explorer.js: report and scientific figures.
models/*.npz: all checkpoint weights, all Fourier and Haar coefficients,
             all intervention logits, inputs, labels, catalog and update norms.
results/*.json: every run and interaction, numerical checks, selected illustration.
results/*.png / *.svg: exportable scientific figures.
SHA256SUMS: integrity hashes for all bundled files except itself.

Array conventions:
p0..p5: (checkpoint, *native_tensor_shape), in order l0.weight, l0.bias,
        l1.weight, l1.bias, out.weight, out.bias.
fourier0..5: orthonormal complex rFFT, every bin.
haar0..5: all orthonormal Haar coefficients, ordering defined in calibrate.haar.
original_logits: (checkpoint, input, output).
rollback_logits / transplant_logits: (interval, component, input, output).
catalog_json: full addresses and native supports; component IDs index these arrays.
component_delta_norm: Euclidean norm of the component's interval update in weights.
walsh_basis: all products of input signs, with masks using a,b,c,d as bits 0..3.

All computation and stored results use float64/complex128. The interactive HTML
rounds display data to 10 decimals; authoritative NPZ arrays retain full precision.
Some hierarchy nodes have identical projections or overlap across bases; the
intervention count is labelled cases, not statistically independent effects.

Limitations: no natural-language task, held-out generalization, grokking, or
direct superposition measurement. All truth-table inputs are trained. Example
selection and the dose follow-up are post-hoc and explicitly labelled. Exact
component addresses are meaningful within a model and are not aligned across seeds.

Next Pythia application: preserve the component atlas, use explicit behavior
probes, compare rollback and transplant, measure collateral effects and parent/
child interactions, and confirm selected targets on separate prompt examples.
