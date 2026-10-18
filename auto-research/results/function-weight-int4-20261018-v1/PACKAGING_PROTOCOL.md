# Lossless packaging follow-up

The preregistered recovery run completed: curve_be4 passes quality, reduces tensor payload3.575x, but reduces torch deployment files only2.808x. Its original3x serialized-storage gate FAILED and remains failed. Quality metrics have already been observed.

This follow-up changes serialization only: a JSON header and one typed byte buffer replace many tensor ZIP records. No new training, changed quantization, candidate selection, confirmation region or quality threshold. Apply the identical container to ALL15 selected exports, including FP16 and quantized nanoGPT controls. Keep original .packed.pt exports. Compare compact-to-compact file sizes, not compact quantized versus old FP16. Require exact bitwise tensor round trip and exact confirmation logits. Record new format results separately; do not relabel the original protocol as passed.

The loader owns packed/FP16 typed views into a shared byte buffer, no dense FP32 weights. Hash the data and original source export. Verify magic, truncation and corruption rejection. Independently load each compact file in a fresh process and report peak RSS, separately from weight bytes. Source/code/protocol are committed before converting exports. This is a serialization improvement of an already measured model, not independent replication or new language evidence.
