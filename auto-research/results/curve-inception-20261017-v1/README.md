# Persistent curve coefficients: first controlled experiment

This is a new, small synthetic mechanism study. It is **not** a replay of the
historical Super-1 hierarchy, a language-model benchmark, or a fully
self-referential learner. No existing model or archived evidence is modified.

The immediate question is whether a 16-scalar persistent coefficient state is
more useful when it controls retention and mixing than when it provides ordinary
candidate feedback. `adaptive64` and `context64` have exactly the same 10,144
trainable parameters and 64 FP32 recurrent-state scalars. Other controls are a
48-state fixed cell, a 64-state fixed cell, and a standard 64-state GRU.

## Mechanism

The small core inherits the linear minGRU interpolation equation inspected in
`JRowe47/Super-1` at commit
`4b905f740f87ec037c9c9e1e158703c81cc6aed4`,
`auto-research/results/gru-extended-20261016-v1/models.py`
(Git blob `00ec06cfcc3f29a52f285f2b9ca20b701d728ed2`). The old hierarchy itself is
not imported or relabeled. The October 15 attachment was also inspected for
curve-controller and memory-state conventions.

All nonstandard-GRU arms share a token-conditioned pair mixer:

    v, g, a = projection(embedding(x))
    z = sigmoid(g + gate_offset)
    theta = 0.25 * tanh(a + angle_offset)
    h_new = (1-z) * pair_rotate(h, theta) + z * v

Eight cosine coefficients define each of two smooth fields over 24 hidden-state
pairs. In the adaptive arm, one field shifts update logits (gain 2), and the
other shifts pre-tanh mixing angles. In the matched ordinary-context arm, the
two fields instead add to the candidate's paired coordinates (gain 1). Thus the
comparison isolates *where* persistent state enters the transition, not a
perfectly matched numerical perturbation size.

Every eight input tokens, a shared-form controller uses the current token,
updated hidden state, and previous coefficients to propose bounded coefficients:

    proposal, eta_logits = updater([embedding(x), tanh(h_new), p])
    eta = sigmoid(eta_logits)
    p_new = (1-eta)*p + eta*tanh(proposal)

These coefficients affect subsequent inputs only. They are fixed within each
chunk; gradients still cross chunk boundaries. The updater is an ordinary
learned linear map whose weights remain fixed at inference. Storing the updater's
own weights in writable curves is a later experiment. The present coefficient
state changes gate biases and mixing controls, not every GRU weight matrix.

The curve coordinate indexes hidden pairs; it is neither token time nor model
depth. Fixed basis buffers and learned parameters are counted separately from
persistent sequence state. The latter is 256 FP32 bytes for the four 64-state
arms, plus a shared integer clock; this excludes weights and training memory.

## Task and controls

There are four keys, eight possible values, and visible WRITE, QUERY, LOCK,
UNLOCK, and distractor symbols. Two keys are locked after initial writes.
Conflicting writes to locked keys should be ignored. Reopening a key should
permit a new value to replace the old one. The model receives only token IDs;
simulator values appear only in the query loss. No write/lock rules are enforced
inside any learned model.

Training uses independent episodes with 8, 16, or 32 intervening tokens. One
quarter contain no conflicting writes; the rest contain conflicting writes and
may reopen a key. Full-episode backpropagation, query-only cross-entropy, paired
training streams, and the final checkpoint are shared across arms. The 32-token
delay endpoint is in the trained delay range; 256 and 1,024 are extrapolation.

The 250-update pilot uses seed 991 and separate selection streams. Main runs
use seeds 101701–101703 and 3,000 updates each. The protocol was frozen locally
before main training; its timestamp/hash are in `REGISTRATION.json`. Git
publication occurs after execution and is not an independent preregistration.

Read `PROTOCOL.json` for the primary endpoint, useful-effect rule, resource
limits, and the complete distinction between matched and unmatched budgets.
Every outcome is retained. No official test, external dataset, GPU, paid cloud
resource, or language-quality evaluation is involved.

## Reproduce

Dependencies are listed in `requirements.txt`. To rerun from scratch, copy only
`model.py`, `tasks.py`, `experiment.py`, `launch.py`, `analyze.py`,
`PROTOCOL.json`, and `REGISTRATION.json` into a fresh directory. Do not overwrite
the preserved `main/` directory. From the fresh directory:

```sh
python experiment.py --verify
python launch.py --split main --steps 3000 --workers 3
python analyze.py
```

For evaluation-only replay, obtain the companion checkpoint archive and extract
its contents into this result directory; `main/<arm>_s<seed>/final.pt` must match
`CHECKPOINT_MANIFEST.json`. The raw full-step training traces are in that archive
too. Repository training summaries retain every 100th step and the last step.
`analyze.py` replays all primary-delay endpoints, requires exact aggregate and
per-episode agreement, and runs registered zero-coefficient/gate/mixing
interventions. These interventions change a trained model and are not retrained
ablations.

## Related primary sources

- HyperNetworks: https://research.google/pubs/hypernetworks/
- Self-referential weight matrices: https://proceedings.mlr.press/v162/irie22b.html
- Minimal GRUs: https://arxiv.org/abs/2410.01201

No novelty claim is made. Full self-modification, persistent learning across
episodes, consolidation, variable-depth execution, and language retention are
outside this first study.
