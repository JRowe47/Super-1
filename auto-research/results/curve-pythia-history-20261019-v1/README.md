# Curve/Pythia experiment history: evidence-only publication

This is a retrospective, user-authorized import of ten completed October 17, 2026 stages, including both initial SmolLM Fourier probes. No training, model evaluation, official-test access, infrastructure provisioning, or architecture promotion was performed for publication.

See the [summary and stage index](SUMMARY.md). Each snapshot retains its original source, protocol, report, measurements, figures, and available historical verification receipts. Files are byte-identical to the saved experiment archives after lossless gzip decoding. Original paths and original/stored SHA-256 digests are in `IMPORT.json`; excluded dependency hashes are recorded there too. Historical README statements about no repository changes describe the original experiments; this explicit publication imports their evidence now.

## Verify without running models

From repository root:

    python auto-research/results/curve-pythia-history-20261019-v1/verify_import.py

This checks all published snapshot bytes, decodes compressed evidence, parses JSON and Python syntax without executing research code, against the scoped import manifest. It does not verify every unrelated repository file or independently replay the original model experiments.

## Inspect or reproduce a stage

Large JSON/log evidence is stored as `.gz` to keep the contribution compact. Use gzip tools to read it, or reconstruct the original directory layout in a fresh disposable location:

    python auto-research/results/curve-pythia-history-20261019-v1/restore_snapshot.py pythia70m_spectral /tmp/pythia-spectral-review

HTML reports, Python source, figures and smaller records remain directly browsable. Restore before running original report scripts because historical scripts expect uncompressed names. Their original absolute paths may need adapting to the new workspace.

Read each stage's original README, pinned checkpoint manifest, requirements and data attribution before reproduction. External pretrained weights, saved model checkpoints, full-resolution coefficient arrays, token arrays and other NumPy binary dependencies are deliberately omitted from Git. Their saved-archive identities and hashes appear in `IMPORT.json`; the archive itself is not distributed by this PR. Some original archives also omit dependencies, as documented by the original reports. A fully runnable standalone model replay is therefore not claimed. Do not use untrusted pickle/model loading merely to inspect this publication.

## Scope limits

The early Fourier and neuron-ordering studies are historical context, followed by native-order, full-resolution component work. Positive, negative, failed and limited toy results are preserved together. Complete coefficient encoding is a coordinate change, not compression. Synthetic copying is not general language competence; a successful augmented routing toy is not a complete GSA/Transformer conversion. Fresh evaluation regions, reused data, controls, optimizer changes and extra teacher-replay costs are identified in the original reports.

Only this ten-stage curve/Pythia sequence is included. The separate perturbation/annealing studies and unrelated research are not part of this snapshot.
