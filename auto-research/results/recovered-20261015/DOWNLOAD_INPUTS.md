# Download all located replication inputs

The private [replication-input release](https://github.com/rohanarun/function-weights-research/releases/tag/replication-inputs-20261015) includes the exact corpus/teacher bytes and located tokenizer/cache assets. Sign into GitHub as a collaborator first.

Use a fresh, approved local directory with sufficient free space. On the original Mac, use a mounted writable external SSD, not iCloud Documents/Desktop. These are download/reconstruction steps, not training commands.

    gh release download replication-inputs-20261015 --repo rohanarun/function-weights-research --dir /your/fresh/ssd/directory

In that directory, reconstruct the training corpus in exactly this order (do not append to an existing output):

    cat train.bin.part-001 train.bin.part-002 train.bin.part-003 > train.bin
    shasum -a 256 train.bin val.bin qwen3-0.6b-base-model.safetensors

Expected SHA-256:
- train.bin: 834c82d2ea76e2f17aa43d15862f5ef444322c37d9b29d07fe803bb21ac163e4
- val.bin: 8517985c4979dbc95c83137ee534e30e789e2275e4ba1ee1fd31e1f0d566e4d3
- teacher weights: cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba

INPUT_MANIFEST.json binds every part/asset and ZIP member. Each upload's GitHub SHA-256 digest was checked against the local bytes. The training file is exactly 4,000,000,000 bytes, validation 8,000,000 bytes, teacher weights 1,192,135,096 bytes. Rename the downloaded teacher asset to model.safetensors within the pinned snapshot directory when constructing a runtime; retain revision da87bfb608c14b7cf20ba1ce41287e8de496c0cd.

The cache ZIP preserves all located files under the four registered dataset revisions. Use the historical package's binding/prefetch layout rather than guessing current library cache paths. A cache's current hash and revision identity do not supply a historical content hash that was never recorded.

Original student checkpoints are in the [checkpoint release](https://github.com/rohanarun/function-weights-research/releases/tag/recovered-120m-checkpoints-20261015), not this input release.

These are third-party data/model artifacts for private replication; their upstream terms remain applicable. This does not relicense them under repository MIT. Missing nanoGPT final checkpoint, three byte-exact deployed hierarchy core files, and original runtime/container remain missing, not fabricated. No training or evaluation was launched.
