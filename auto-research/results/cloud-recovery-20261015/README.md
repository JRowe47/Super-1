# Cloud recovery: exact source and nanoGPT checkpoint

The historical nanoGPT final checkpoint is recovered and uploaded to release recovered-120m-checkpoints-20261015 as official-nanogpt-final.pt. Its local, recovered-disk and GitHub SHA-256 all match the historical completion receipt: 7de4e9d81b606ec6b0d7e9fbaf4dcdcb30aa55456e775a07371aaf65a6591241.

The exact deployed hierarchy source is under exact-historical-source/representation_experiments/. All four recorded deployed core hashes match. The earlier compatible snapshot remains preserved separately; do not overwrite or relabel it.

Full historical per-arm train logs, configs and results are under historical-evidence/. These are recovered old evidence, not new training.

For an exact-source rerun, use this recovered hierarchy module tree instead of the package's compatible historical snapshot. Read the recovered run_matched_hierarchy_suite.py and configs before execution; retain the existing corpus, teacher, checkpoint and cache pins from the replication-input release. The original launch paths are historical and must be mapped explicitly to the downloaded input locations. Do not claim a fresh reproduced score until the complete frozen evaluation is run and verified.

Preservation and cleanup are complete. RELEASES_VERIFIED.json verifies all 25 expected assets (17,777,363,648 bytes) across three private releases. CLEANUP.json confirms zero temporary recovery instances/disks and all original research GPU instances stopped. RUNTIME_REPLAY.md and EXPORTED_RUNTIME_SMOKE.json document the extracted runtime's limited CPU test; a full independent container/GPU benchmark replay remains unverified. No GPU or training workload was started.
