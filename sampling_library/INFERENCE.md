# Function-weight inference and attention memory

`CachedSampledCurve` wraps a loaded sampling profile without copying or replacing its model. Q, K, V, output and feed-forward projections still evaluate the original learned weight bases and coefficient curve at the router's sampled depth coordinate `t`. Normalizations, residual paths and embedding/readout weights are unchanged.

```python
import torch
from sampling_library.profiles import load_profile
from sampling_library.inference import CachedSampledCurve, cache_bytes

model, evidence = load_profile("text-depth-library-20261018-v1/layers-4/73")
runtime = CachedSampledCurve(model).eval()
with torch.inference_mode():
    logits, cache = runtime(prompt_ids)
    logits, cache = runtime(next_token_ids, cache)
print(cache_bytes(cache))  # Logical KV tensor bytes; excludes model/runtime memory.
```

The default `ExactKV` backend stores the full KV history. It provides a cached reference for the same function-weight model, not a replacement architecture. Other backends can replace only the attention storage/readout by implementing `forward(q, k, v, state) -> (attention_output, updated_state)`. The surrounding function-weight computations remain identical.

The executed layer count comes from the loaded profile. Token-conditioned coordinates remain dynamic at each executed step. The existing profiles select two, three or four steps for a whole request; they do not implement arbitrary per-token layer skipping. Changing depth, policy or model parameters after a cache was built requires rebuilding it by prefilling the complete prefix with `cache=None`. Caches are also bound to their runtime and batch size. A different depth changes intermediate activations and cannot safely reuse the old layer histories.

The runtime retains all original backbone weights, even for lower-depth profiles. Neither fewer executed layers nor a smaller KV store implies smaller resident model weights. RoPE can execute beyond the training length, but this change provides no new long-context quality evidence.

Validate with:

```sh
python3 -B -m sampling_library.test_inference
```

The tests use all nine saved trained profiles, check full/cached logits and sampling positions, check curve/router gradient equivalence, preserve exact model weights, verify logical cache sizes, and reject incompatible cache reuse. Original experiment sources and checkpoint receipts are immutable. This is an inference correctness result, not a new quality or speed benchmark.
