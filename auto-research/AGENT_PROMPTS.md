# Agent instructions learned from prior failures

Use these as research and review instructions, not automatic runtime patches.

## Before implementation

Previously a different model was substituted for the original function-weight/hierarchy design. Always identify the authoritative class, equations, gate/coupling order, state semantics and source hashes first. Preserve them unless an explicitly named ablation changes one component.

Previously Python recurrence made long context uneconomical. Always measure a small hardware memory/speed canary and demonstrate output/gradient/optimizer fidelity before asking for full training. Never call a nonlinear recurrence block-parallel without a valid derivation.

Previously several mechanisms were mixed together. Always state one hypothesis and its strongest simpler control. Separate concept tokens, distance embeddings, pair-energy, hierarchy transport and readout hypotheses.

Previously identity-initialization tests could hide inactive transport/readout paths. Always test nonzero gains, nonuniform scale weights, gradients and later recurrent effects as well as exact identity behavior.

## Before interpreting results

Previously a 512-trained baseline degraded at 2K. Always distinguish matched train/eval lengths from extrapolation. Train the positional baseline properly at each primary context and include a strong positional control.

Previously more training could be mistaken for an architecture gain. Always report completed target exposures, optimizer updates, epochs only when well-defined, compute and tuning opportunities per arm.

Previously byte CE, token CE, word PPL and accuracy were confused. Always retain units, vocabulary, split, context and metric direction; never compare unlike metrics or call lower CE higher accuracy.

Previously the 14,003x state ratio could be heard as total memory. Always say "analytic persistent sequence state" and separately measure total deployment memory and latency.

Previously an incomplete word sweep could be treated as completed. Always require a terminal receipt and preserve the Transformer-favoring preliminary result. Never fabricate completion or bypass launch guards.

## Before reporting or promoting

Always show all seeds, controls, failed arms and rounding precision. Label historical report-only results separately from hash-verified raw receipts. Exact duplicate evaluation checks reproducibility, not a new independent seed or proof of generalization.

Always use the predeclared useful-effect criterion, not a post-hoc threshold. A tiny or uncertain effect should motivate a stronger control or additional approved seeds, not a breakthrough claim.

Always compare exact mechanisms against current primary-source prior art before novelty language, and valid official benchmarks before SOTA language. A promising pilot is worth reporting honestly without either label.

Always distinguish proposal, launch, progress, completion, confirmation, publication and cleanup. A plan is not a running experiment. This user stopped training; only new explicit authorization permits a launch.

## Tensor layout at baseline boundaries

The synthetic experiment completed all training but failed when a secondary reset-window diagnostic passed noncontiguous sliced targets to pinned nanoGPT, whose loss uses view. Always validate contiguous input and target layouts after chunking at adapters, including secondary diagnostics. Preserve upstream source and the failed receipt. Any evaluation recovery must use the already frozen checkpoints, retain the original decision rule, disclose prior metric exposure and never silently retrain or relabel the failed run as uninterrupted success.

## Inference-first memory experiments

The relaxed-compression pilot did not improve over the compact hybrid, and retained tensor savings substantially overstated measured total-process savings. Always test the compression/quality hypothesis before assuming less compression is better; preserve both the worse arm and the primary failed gate. Report total resident/peak memory separately from weights and cache/state accounting. Spending more training compute may be acceptable for inference gains, but disclose that cost and tuning opportunity. Do not describe fixed-depth results as dynamic layer-sampling gains; validate cache consistency and resident weights for any future dynamic policy.
