# SmolLM conversion and spectral research: compact recovered history

Nine completed stages are represented: exact first-block curve encoding, resolution/rank controls, learned depth curves, cross-layer survey, joint curves phases 4 and 5, initialization phase 6, Fourier diagnostics, and recursive spectral diagnostics. This is retrospective preservation, not a new experiment or model promotion.

The initial exact conversion is a lossless representation at the tested coordinates, not compression or faster inference. Later rank, depth and joint-training experiments test separate restricted approximations. The historical reports contain negative outcomes and single-seed/short-training/limited-evaluation caveats. Do not pool their differing evaluation sets into a leaderboard.

## What is preserved

238 original small source/report/measurement files retain their archive bytes. Each other artifact has a .txt placeholder recording archive filename, member, size and ZIP CRC32. Large HTML reports have readable text extracted into their placeholders; images and layout are omitted. INVENTORY.json distinguishes original bytes from derived placeholders and gives published SHA-256 hashes. Original archive filenames are recovery identifiers, not public download links. Ask the author for access to the originals.

This lighter format follows the owner's request to use text placeholders for bulky research artifacts. Raw dataset windows, checkpoints, token/coefficient arrays, images and unrelated conversational context are not republished. Python source and small quantitative results are retained where recovered. The two final spectral snapshots overlap the separate curve/Pythia history and are intentionally included here for chronological completeness.

## Known recovery gap

The original smollm_first_block_curve_experiment_2026-10-16.zip is invalid/truncated in the recovered copy, consistent with the earlier recovery report. Its original source and checkpoint recovery are incomplete. The separately saved first-block HTML report was recovered and converted to RECOVERED_REPORT.txt. No missing source, weights or evidence were invented.

## Validation and reproduction limits

The publisher read retained ZIP members using Python's ZIP reader (which checks member CRC), parsed retained JSON and Python syntax, and generated SHA-256 hashes for each published file. Run `python verify_publication.py` from this directory for the scoped published-byte check. This does not rerun historical models, recompute metrics, validate all archive members, or establish scientific reproducibility.

Original stage READMEs and scripts document the historical setup. Full reproduction requires original artifacts, public pretrained weights and dependencies, potentially unavailable portions of the first-stage archive, and datasets not provided here. Placeholders do not make those dependencies available to the public. Historical reports' local paths may require adaptation. No claim that all nine stages are self-contained or trivially reproducible is made.

No production code, training defaults, broad research registry or unrelated feedback experiments are changed.
