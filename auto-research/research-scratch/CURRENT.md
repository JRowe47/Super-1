# Current research handoff

Updated 2026-10-18 UTC.

## Active stage

`magnitude-feedback-20261018-v1` is frozen before execution. It tests adaptive feedback against a frozen direction with same-state adaptive update magnitudes, and includes a reciprocal normalization control. Six arms, three fresh main seeds701781/2/3, pilot96977; fixed prior retention gate. No results yet.

- [Protocol](../results/magnitude-feedback-20261018-v1/PROTOCOL.json)
- [Source hashes](../results/magnitude-feedback-20261018-v1/REGISTRATION.json)
- [Reproduction](../results/magnitude-feedback-20261018-v1/README.md)
- [Provenance](../results/magnitude-feedback-20261018-v1/SOURCE_PROVENANCE.json)

Exact source/protocol is being committed before verification and training. Resume by checking VERIFY/PILOT/MAIN receipts and seed RESULT.json before launching anything; never duplicate an active or completed run. Remaining compute has the same900-second aggregate CPU ceiling, at most3 single-thread workers. No GPU, cloud jobs or official-test access.

## Previous finding and preservation gap

The bounded study reported28.31% lower post-change CE with adaptive feedback, but old-fact retention failed in every seed. [Report](../reports/bounded-feedback-20261018-v1.md). Its source/registration/checkpoints did not survive the workspace reset; the Git records are transcribed findings, not original bytes. Do not claim the new implementation exactly reproduces that lost run.

The new source derives from the hash-verified preserved predecessor `adaptive-feedback-20261018-v1` in the existing `curve_feedback_experiments_2026-10-18.zip`. Only its small source files were extracted. The user requested small individual artifacts and this stable research scratch channel; do not create another cumulative archive.
