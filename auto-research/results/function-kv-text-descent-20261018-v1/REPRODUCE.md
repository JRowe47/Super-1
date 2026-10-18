# Reproduce the text and memory experiments

Run from the repository root with Python and PyTorch. All output directories must be on the mounted external SSD under `/Volumes/SP AI 01_16/CodexWorkspaces`. The checked-in corpus is Project Gutenberg ebook 11 with its original license; the extracted byte corpus is hash-bound without newline normalization. The raw corpus therefore retains CRLF and trailing blank lines.

The frozen run sources and protocols must not be edited. Reproduce training into a new output directory with `run.py train --output NEW_SSD_DIRECTORY`. The model architectures, exact splits, updates, seeds, optimizer and promotion criteria are in each run's CONFIG/PROTOCOL files. Each run trains two from-scratch references on identical batches, then tunes only pooling directions for two memory capacities. Training both references is not a match of FLOPs or parameter count.

The analyzer is a one-shot recorder intended for an empty experiment execution directory. Do not run it against the published execution directory: it deliberately refuses to overwrite existing measurement files. For fresh isolated memory measurements use `run.py memory --output NEW_SSD_DIRECTORY --arm exact --seed 73 --length 8192 --destination NEW_JSON_PATH`, substituting a configured hierarchy arm for the compressed measurement. The 100K extension has its own protocol and a single paired seed. Its JSON records are descriptive mechanics measurements, not a statistical speed benchmark; other research processes were running on the host.

`python3 -B -m unittest sampling_library.test_hierarchy_profiles sampling_library.test_hierarchical_kv sampling_library.test_inference` checks runtime mechanics and profile binding. `verify_profiles.py` independently reconstructs the loaded profiles and reproduces their full confirmation logits, recording LIBRARY_REPLAY.json once. Existing replay receipts are immutable.

The measured libraries are explicit opt-ins and do not replace active task samplers:

```python
from pathlib import Path
from sampling_library.hierarchy_profiles import load_hierarchy_profile

index = Path("auto-research/results/function-kv-text-descent-20261018-v1/LOADABLE_LIBRARY.json")
runtime, evidence = load_hierarchy_profile(
    "function-kv-text-descent-20261018-v1/h16/73", index
)
# runtime accepts token IDs from the recorded UTF-8 byte tokenizer.
# For another dtype/device, move the entire runtime before prefill.
# Start with cache=None; changing depth or parameters requires fresh prefill.
```

The model source, its transitive dependencies, backbone checkpoint, pooling checkpoint and selected measurement record are checked before loading. These tiny models and byte-token contexts are not substitutes for a trained 70B model or a standard SOTA language-model evaluation.
