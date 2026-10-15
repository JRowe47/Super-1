# Recovered nanoGPT versus hierarchy paper artifacts

The exact supplied seven-page report, expanded 74-file hash-verified reproduction package, original frozen nanoGPT archive, historical hierarchy ZIP, and email-safe replication ZIP are included here. See [ARTIFACT_INDEX.json](ARTIFACT_INDEX.json) for a row-by-row inventory of the paper's provenance table.

The package includes full upstream nanoGPT revision, source/license, adapters, configuration, freeze manifest, resolved config, raw result/benchmark JSON, preflight, 246-record training log and both completion receipts. Original hierarchy result/benchmark/config records are included. File bytes have not been rewritten to match a claim.

## Still external or missing

The exact 4GB training corpus, 8MB validation corpus and 1.19GB teacher weights are present on the SSD and SHA-256 verified; their paths and hashes are in the index. They are not uploaded into Git. Large input distribution requires suitable artifact storage and redistribution review.

The nanoGPT final student model and hierarchy final checkpoint have not been located locally. Their expected hashes and original remote provenance are retained. A recorded hash is not a recovered checkpoint. No VM was provisioned/attached to retrieve them.

The hierarchy snapshot remains structurally compatible, not byte-identical to all deployed core files. See package/REPRODUCIBILITY.md. Exact original common.py was separately located in iCloud, but this does not resolve the other deployed files.

## Correct interpretation

The recovered raw receipts confirm hierarchy's longer-context CE advantage and 14,002.6x analytic persistent-state ratio at 2K. nanoGPT wins at the trained 512 context and on four-task mean (0.2863377193 vs 0.2802061404). The original in-house baseline task table is not a nanoGPT task table.

Use package/tools/verify_package.py and package/tools/compare_results.py only after reviewing their commands and output behavior. No training/evaluation was run for this upload. Existing one-shot official-test restrictions remain unchanged.

Third-party source/license notices are preserved; the repository MIT grant does not replace them.
