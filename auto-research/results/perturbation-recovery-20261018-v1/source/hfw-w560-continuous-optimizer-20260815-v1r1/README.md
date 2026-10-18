# Continuous optimizer for the real HFW W560 checkpoint

Status: v1r1 exact-candidate fix implemented locally, not frozen, not staged,
and not launched. The exact
step-3052 parent checkpoint is not inside this package.

The held v1 package and first real failed attempt are immutable and remain
preserved. That attempt stopped before any decision, head, delta promotion, or
checkpoint publication because reconstructing an accepted candidate as
`base + float32(candidate - base)` is not always bit-identical to the solver's
exact candidate.

This package is a controller around the frozen real-checkpoint paired spectral
solver. It is not a proxy, a training job, Adam, backpropagation, or a new model
design.

## Invariants

- The immutable root checkpoint is SHA-256
  `0c69fbf81526282d7d68ac258a98867d7de0f0e88e69e5f9f0543fdf1c535fba`.
- The vendored `branch_search.py` and `production_bridge.py` are byte-identical
  to the v1r4r1 proven-solver package. Their solver arithmetic is not edited.
- `config/controller.json` pins a non-circular runtime manifest covering every
  controller Python module and every regular vendored source/protocol file.
  Bootstrap, workers, committer, and every supervisor wave same-FD verify those
  bytes; the token commitment, relocated-parent receipt, ledger contract, and
  generation-zero head bind the manifest file and content identities.
- Every proposal changes exactly one declared group: base affines, left factors,
  right factors, or state codecs.
- Proposal workers run the unchanged 180-evaluation paired spectral cycle. They
  have no commit authority and see only the precommitted SEARCH bank.
- A job-local artifact adapter replaces only provisional full-checkpoint output
  with exact accepted group-flat replacement bytes, the unchanged solver's
  cumulative/incremental group deltas, and lineage descriptors. It
  restores the original function after the call. Seeds, radii, direction math,
  evaluation order, losses, and acceptance decisions are unchanged.
- The exact accepted group-flat bytes are the sole candidate-evaluation and
  promotion source of truth. The native solver cumulative and incremental
  deltas remain preserved provenance and are verified directionally as exact
  float32 subtraction. They are never used to reconstruct the candidate.
- One file-locked committer serializes all promotions. It rejects every stale
  proposal; it never merges or rebases worker outputs.
- The worker exports every provisionally accepted cumulative candidate in exact
  solver order. The committer revalidates the frozen branch chain, evaluates the
  entire permanent COMMIT objective twice for the current head and every such
  candidate, and selects the lowest replicated mean among candidates that pass
  the configured improvement threshold. It materializes one full checkpoint
  only for that verified promotion.
- All jobs, proposals, deltas, decisions, promoted checkpoints, and immutable
  head records are create-only. A crash can leave an orphan artifact, but it
  cannot advance the head partially. Recovery preserves orphans and re-evaluates
  unresolved proposals.
- Supervisor recovery and commit exceptions produce immutable BLOCK receipts,
  return a nonzero CLI exit status, and stop before another wave.
- Disk reservations are formula-bound to the frozen protocol: the largest group
  (10,046,960 float32 values), the 18-accept ceiling, and all three per-accept
  artifacts require 2,170,143,360 bytes before overhead. The worker reserve is
  2,707,014,272 bytes after a 512 MiB metadata/filesystem allowance. Promotion
  separately reserves 1,815,805,367 bytes for the source-checkpoint-size bound,
  selected exact replacement, selected native delta, and 512 MiB overhead.
- A verified promotion content-addresses both the exact group-flat replacement
  and native solver cumulative delta. Proposal, decision, head, full checkpoint,
  loader, and history validation bind both identities; the full child is
  materialized from the exact replacement.
- No FINAL bank exists in this controller.

## Permanent COMMIT objective

The COMMIT objective is the exact previously published fixed512 bank: 32 starts,
513 tokens per sequence, against validation corpus SHA-256
`8517985c4979dbc95c83137ee534e30e789e2275e4ba1ee1fd31e1f0d566e4d3`.
Its reveal/commit provenance begins from the exact recorded digest
`2e7e92507571b8629d628ab201e29eccc4e778aa0cba3d4973dabe33bdd99526`.
The prior one-call authorization is bound in provenance by its full SHA-256
`7d6476830f364ee4ce0dbe5eda2d8d728b301c77378ed7623675bccd8abddd3e`;
it is not reused as runtime authorization.

This bank is deliberately reclassified as the permanent reusable optimization
objective. It is not held out. Accept/reject decisions optimize it over time, so
results on it must never be described as unbiased generalization evidence.
Before any proposal can start, bootstrap must reproduce the unchanged parent CE
`5.413214594125748` within the configured tolerance and duplicate it exactly.

## Relocated-parent gate

The training runtime receives copied bytes, so its inode will differ from the
original. `prepare-parent`:

1. verifies that the frozen vendor protocol still binds the original path, SHA,
   size, review receipt, checkpoint identity, model-state digest, step, and token
   count;
2. same-FD hashes the relocated checkpoint and review receipt;
3. requires the configured hardlink count (currently three);
4. records the relocated device, inode, size, link count, and timestamps; and
5. derives only an in-memory protocol containing the relocated path/stat before
   calling the frozen parent loader.

The copied `production_protocol.json` is never edited and no byte, review, or
model-state check is weakened.

## Lifecycle

These commands describe the gated lifecycle. They have not been executed on the
real checkpoint:

```text
python -m hfw_continuous.cli prepare-parent --config config/controller.json --output PREPARED.json
python -m hfw_continuous.cli prepare-tokens --config config/controller.json --output token-commitment-attempt1
python -m hfw_continuous.cli bootstrap --config config/controller.json --token-manifest token-commitment-attempt1/token_manifest.json --ledger run-attempt1
python -m hfw_continuous.cli supervise --config config/controller.json --token-manifest token-commitment-attempt1/token_manifest.json --ledger run-attempt1
```

`supervise` runs indefinitely when `max_waves` is null. A bounded value is only
an operational stop control; it does not change solver settings. Each failed or
completed job remains preserved and is never reused.

Launch also requires an external frozen continuous-package source manifest and
release receipt that bind these controller bytes. Those artifacts do not exist
in this unfrozen working package, so this directory is not launch-ready by
itself.

## Tests

`tests/run_integrity_tests.py` contains only non-scientific integrity checks. It
checks frozen source hashes, token-bank sealing/create-only behavior, stale-head
rejection before COMMIT is opened, toy-objective trace parity with and without
the artifact-only adapter, config/token/manifest/controller tamper rejection,
crash-staged head isolation, singleton supervision, formula-bound disk
fail-close, float32 noninvertibility, exact-replacement/delta tamper rejection,
dual-lineage history/loader validation, and supervisor exception BLOCK receipts.
It makes no claim about W560 quality.
