# Hierarchical KV memory with function weights preserved

This experimental backend changes the attention history store and readout. It keeps the original function-weight projections, sampled coordinates, chosen execution depth, normalization, residuals and feed-forward computation. It does not replace attention blocks with gated-delta or RWKV blocks.

```python
import torch
from sampling_library.profiles import load_profile
from sampling_library.inference import CachedSampledCurve, cache_bytes
from sampling_library.hierarchical_kv import HierarchicalKV

model, evidence = load_profile("text-depth-library-20261018-v1/layers-4/73")
runtime = CachedSampledCurve(
    model,
    memory_factory=lambda layer, heads, head_dim: HierarchicalKV(
        heads, head_dim,
        recent_capacity=16,
        level_capacities=(4, 4),
        merge_factor=2,
    ),
).eval()
with torch.inference_mode():
    logits, cache = runtime(prompt_ids)
    logits, cache = runtime(next_token_ids, cache)
print(cache_bytes(cache))
```

These capacities illustrate the API, not optimal settings or a promoted use-case profile. Capacity is configured; training does not automatically add slots or layers. For other devices/dtypes, move the entire runtime with `.to(...)` before prefilling.

Recent KV entries stay exact. As the recent buffer fills, the oldest groups move into increasingly coarse levels. Each merge uses a learned per-head scoring direction to pool keys and values; initialization is count-weighted mean pooling. Attention reads the combined recent and summarized entries, with a log-count correction for the number of tokens represented by each summary. Summarized keys already contain the original RoPE positions. This is an approximation of full attention, not an exact replacement once entries merge.

All updates and reads occur in token order during prefill, so later tokens cannot change summaries used for earlier outputs. Memory is bounded by `recent_capacity + sum(level_capacities)` KV entries per executed layer. Tensor accounting excludes model weights and runtime/transient memory; bounded KV storage alone does not prove a large total-memory reduction.

The pooling directions are ordinary trainable parameters in `runtime.memory`. Gradient descent can optimize them while `runtime.model.requires_grad_(False)` preserves the learned function-weight model. The included synthetic distillation test verifies this connection and checks that the frozen backbone does not change. New [text-trained pooling profiles](../auto-research/reports/function-weight-kv-text-descent-20261018.md) are available as explicit research opt-ins. They are not automatically promoted: the original total-memory gates failed, despite a separate 100K mechanics memory win.

Sampling behavior remains that of PR #7: learned token-conditioned `t` coordinates and a selected whole-request depth. Depth/policy/model-weight changes require a fresh full-prefix prefill. Hierarchy configuration changes likewise invalidate the cache. Arbitrary mid-request depth changes are not supported by reusing stale histories.

The earlier gated-delta hybrid pilot in `auto-research/results/memory-quality-frontier-20261018-v1/` is preserved as an alternative-architecture result. Its quality and memory numbers do not describe this function-weight hierarchy. The interrupted RWKV comparison was also a different model family and provides no completed comparison for this implementation.

Validation:

```sh
python3 -B -m unittest sampling_library.test_inference sampling_library.test_hierarchical_kv
```

Checks cover all nine saved profiles without compression, causal/chunk-invariant compressed execution, bounded state and token-count conservation, exact constant-key attention under count-weighted pooling, gradient-based pooling updates with frozen original weights, and stale-cache rejection. Evidence is in `auto-research/results/function-weight-hierarchical-kv-20261018-v1/VERIFICATION.json`. These tests are mechanics checks. The linked text experiments separately measure language loss and total process memory on the actual function-weight model. Broad language quality, distant-fact retrieval, optimized latency, 100K quality and a matched RWKV comparison remain unverified.

Load a measured model/pooling pair with `sampling_library.hierarchy_profiles.load_hierarchy_profile(entry_id, index)`. See the [loading instructions](../auto-research/results/function-kv-text-descent-20261018-v1/REPRODUCE.md). This verifies the selected measurement record and rejects mismatched backbone/pooling pairs.
