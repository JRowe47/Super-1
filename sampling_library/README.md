# Sampling parameter library

This stores learned functions that choose sampling positions **t** from token representations. It stores the best retained research reference for each explicit use case and exact backbone, alongside candidates that failed promotion. It does not claim ideal parameters or infer a task from keywords.

The initial use case is `text/shakespeare-byte`, with three independently trained backbone seeds. Every entry includes sampler and backbone hashes, tokenizer, context, executed layer count, training settings, validation/confirmation scores, observed sampling positions, and links to immutable evidence. Reference status is not a general language-quality endorsement.

```bash
python3 -B -m sampling_library list
python3 -B -m sampling_library verify
python3 -B -m sampling_library.test_library
```

With a compatible four-step `SampledCurve` model already loaded:

```python
from sampling_library import load_router
entry = load_router(model, use_case="text/shakespeare-byte",
                    tokenizer="utf8-bytes-v1", context=64)
logits = model(token_ids)
```

The exact backbone fingerprint selects its matching seed automatically. Unknown use cases, changed weights, tokenizers, contexts, missing files, and corrupt checkpoints are rejected. This does not run training during inference. This initial API supports the four-step sampler used by the text study; it does not change a model's depth.

To add a workload, bind its data and metric, train only on its tuning split, choose checkpoints on validation, freeze choices, then evaluate untouched confirmation. Register an evidence-backed JSON entry using `python3 -B -m sampling_library register --entry entry.json`. The single-writer index preserves prior entries when a reference changes; candidate entries cannot activate themselves. Promotion is a research decision established by the linked protocol and independent verifier, not inferred from a single low score. Do not compare scores across different evaluation regions.

See [text continuation report](../auto-research/reports/text-sampling-library-20261018-v1.md). The latest candidates are retained even when they fail the useful-effect threshold. A future experiment may investigate them; they are not silently installed as defaults.
