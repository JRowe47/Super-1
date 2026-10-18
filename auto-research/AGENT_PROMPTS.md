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

## Continuous depth versus adaptive execution

A continuous weight function was previously mistaken for an adaptive execution policy. Always inspect the forward path: fixed construction coordinates and a loop over every block do not learn sampling positions or halting. Account for per-block normalization and routing parameters when changing depth.

For token-conditioned depth experiments, always distinguish uniform positions, learned token-independent positions, and token-conditioned positions. Train and evaluate at the same context, preserve common initialization and samplers for the curve arms, and report basis applications and router overhead. Matching block count does not match FLOPs against a dense Transformer.

Small routing gradients can be hidden by FP32 finite-difference cancellation: the initial canary here returned a zero numerical difference despite a nonzero analytic gradient. Always use a nonconstant curve, nonzero router, and a sufficiently precise finite-difference check before accepting connectivity. The FP64 check, causal-prefix equality, and complete optimizer-step parity must pass before language training. Do not weaken a tolerance to manufacture a pass.

Always freeze all checkpoints before reserved confirmation, retain failures and every seed, and distinguish a small single-corpus byte pilot from broad language quality or an optimized deployment result. Uncached full-prefix decoding must be labeled as such.

## Iterative sampling optimization by workload

The user clarified that “recursively” meant a benchmark/gradient-descent/re-evaluation research loop, not a recursive inference controller. Always preserve that distinction. Optimize sampling on dedicated tuning data, choose checkpoints with selection data, and freeze every workload's choices before reserved confirmation. Report each workload separately and retain the initial policy as an eligible fallback.

A shared backbone can learn text while remaining at chance on structured tasks. Always check each workload's baseline competence before interpreting sampling gains. Nonzero sampler gradients and tiny CE changes do not establish retrieval or arithmetic capability. If the backbone has not learned a task, preserve that failure and address task learning in a separately scoped experiment instead of endlessly tuning routing or weakening the success criterion.

Compare per-workload policies against a shared policy with matched per-workload examples and against token-independent positions. Report extra total controller storage, fixed active steps and every adaptation round. Learning-rate reductions and checkpoint selection must be fixed before confirmation. Never update the underlying model during a sampling-only claim; verify frozen weight fingerprints, not just parameter flags.
