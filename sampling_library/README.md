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
from sampling_library.runtime import load_router
entry = load_router(model, use_case="text/shakespeare-byte",
                    tokenizer="utf8-bytes-v1", context=64)
logits = model(token_ids)
```

The exact backbone fingerprint selects its matching seed automatically. Unknown use cases, changed weights, tokenizers, contexts, missing files, and corrupt checkpoints are rejected. This does not run training during inference. This initial API supports the four-step sampler used by the text study; it does not change a model's depth.

To add a workload, bind its data and metric, train only on its tuning split, choose checkpoints on validation, freeze choices, then evaluate untouched confirmation. Register an evidence-backed JSON entry using `python3 -B -m sampling_library register --entry entry.json`. The single-writer index preserves prior entries when a reference changes; candidate entries cannot activate themselves. Promotion is a research decision established by the linked protocol and independent verifier, not inferred from a single low score. Do not compare scores across different evaluation regions.

See [text continuation report](../auto-research/reports/text-sampling-library-20261018-v1.md). The latest candidates are retained even when they fail the useful-effect threshold. A future experiment may investigate them; they are not silently installed as defaults.

## Learned layer count

The separate [depth study](../auto-research/reports/text-depth-library-20261018-v1.md) learns a categorical preference over two, three and four executed layers, while tuning each depth's interior sampling positions. `depth_profiles.json` preserves the learned probabilities, validation-selected policies, and measured quality/latency for each depth and seed. Two layers have fixed endpoints only; three have one learned interior coordinate; four have two.

The categorical logits are trained by gradient descent on expected text prediction loss. At inference an argmax selects one whole-request depth. Training evaluates all depth branches; inference executes one branch. No quality-versus-cost penalty was invented: this first experiment optimizes prediction quality and reports cheaper profiles separately.

```bash
python3 -B -m sampling_library.profiles
```

Select an experimental profile explicitly from the printed IDs:

```python
from sampling_library.profiles import load_profile
model, evidence = load_profile(profile_id)
logits = model(token_ids)
```

The loader verifies the checkpoint, depth, backbone and transitive source hashes before construction. These profiles remain experimental and do not replace the four-layer research reference unless a separately verified promotion passes. They allocate the original full backbone even when executing fewer layers. Profile selection is explicit; it does not classify arbitrary prompts as a particular use case.
