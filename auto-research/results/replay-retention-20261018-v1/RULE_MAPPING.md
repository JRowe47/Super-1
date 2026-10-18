# What changes in this stage

The learner and complete Fourier representation are unchanged. `baseline_core.py`, `baseline_verify.py`, `feedback_core.py`, `feedback_verify.py` and `launch.py` preserve the preceding registered bytes. The original dense local rules, exact BP control, common BP-forward/local-feedback warmup, spectral cap2 and fixed optimizer remain intact. Magnitude normalization is not active in these six arms; it was the preceding mechanism control.

Only the replay allocation changes. Start with the same ordered512 warmup examples selected by the original generator. FIFO keeps a rolling512-example buffer. Mixed pins the first256 as an old bank and keeps the last256 as a rolling queue. Each step uses16 current examples. The same16 uniform replay RNG draws in[0,512) supply all FIFO samples; for mixed, the first8 modulo256 sample the recent queue and the last8 modulo256 sample the old bank. Sampling is with replacement. Modulo is unbiased because512 is exactly twice256. Append the current16 only after constructing the training batch.

Mixed replay preserves a longer history by allocating half the storage to old examples; it also reduces recent queue coverage from32 to16 batches. This is an explicitly tested memory-allocation policy, not a claim that only sample age changed. Both policies retain exactly512 examples and supply the same128 binary targets per online update.

All replayed labels are the labels originally observed. The old bank is not selectively filtered after drift. Generator metadata can count stale labels for reporting, but policy_step receives only matrices and the actual x/y batch. The fixed bank is selected from training examples before any holdout score is available.

The primary contrast is adaptive versus frozen learning under mixed replay, with the same>=5% CE effect and<=2pp per-seed retention loss gates. The full adaptive/frozen × FIFO/mixed factorial distinguishes a replay intervention from the feedback-learning effect; exact BP under both policies checks whether replay's benefits and costs are general. No configuration is chosen after observing outcomes.
