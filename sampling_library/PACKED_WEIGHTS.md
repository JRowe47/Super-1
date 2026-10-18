# Experimental packed function weights

`packed_weights.py` implements a CPU storage/quality reference for the original sampled function-weight Transformer and pinned nanoGPT. Shared bases and embeddings can be groupwise int4; coefficients/router/norms remain FP16. No dense FP32 weight masters are loaded. Matrix tiles decode on demand, while activations/attention compute in FP32. The token-conditioned t function and request-level depths2/3/4 remain available. Only depth4 has language-quality evidence.

The new [pilot report](../auto-research/reports/function-weight-int4-20261018.md) measures near-FP16 quality with3.575x smaller weight payload and, after a lossless packaging follow-up,3.145x smaller files. Total process RAM is not reduced dramatically at this tiny scale. No cached incremental API, optimized quantized kernel or long-context quality claim is provided. Existing `CachedSampledCurve` and hierarchy paths are unchanged.

## Load a recorded profile

From the repository root, after verifying `scripts/verify_evidence.py`:

```python
import hashlib
import json
from pathlib import Path
import torch
from sampling_library.packed_weights import PackedModel
from sampling_library.packed_weights_archive import read_archive

root = Path.cwd()
index = json.loads((root / 'auto-research/results/function-weight-int4-20261018-v1/LIBRARY.json').read_text())
entry = next(e for e in index['entries'] if e['id'] == 'curve_be4-seed73')
path = root / entry['file']
assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256']
model = PackedModel(read_archive(path))
tokens = torch.tensor(list(b'The rabbit looked at the clock.'), dtype=torch.long)[None]
with torch.inference_mode():
    logits, sampled_t = model(tokens, positions=True, depth=4)
print(logits.shape, sampled_t.shape, model.parameter_bytes)
```

All15 entries are research profiles with `active:false`; they are not universal use-case optima. The index records exact selected checkpoint hashes, tokenizer/context and quality-tested depth. Choose on validation in any new workload; do not choose a seed from this confirmation table to claim better general quality.

## Formats and checks

- `.packed.pt`: original torch deployment bundle, containing only int4/FP16 weights and metadata. Recovery `.pt` files with masters/optimizer are different artifacts.
- `.s1pw`: lossless compact JSON-header/byte-buffer container; typed views share the same backing byte buffer. Checksum and view bounds are checked. Currently little-endian CPU hosts only. All controls use the same format for fair file-size ratios.
- `max_decoded_values * 4` measures the largest decoded weight tile, not total scratch. Integer indices, attention buffers, activations and interpreter cost require measured peak RSS.
- The original3x serialized-storage gate failed. The post-result packaging follow-up is recorded separately and preserves exact tensors/logits.

Reproduce the original30 packed exports with the experiment's `verify.py`, using a new output path on the SSD. Check archive integrity and existing runtimes with `python3 -B -m unittest sampling_library.test_packed_weights_archive sampling_library.test_inference sampling_library.test_hierarchical_kv sampling_library.test_hierarchy_profiles`. The experiment's `package.py` must not overwrite its immutable compact directory.
