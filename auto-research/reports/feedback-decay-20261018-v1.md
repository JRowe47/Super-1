# Feedback learning/decay factorial: failed verification

The preregistered v1 stopped before pilot or main training. On verification seed15817 in FP64, decoded B1 had spectral norm2.0000000000000004, exceeding the strict cap2 by4.44e-16 after the inherited projection. The failed launcher receipt and read-only diagnosis are preserved. The source remains byte-identical to its registration and pre-execution commit8d5a000f7572962bc4eb29e072bbe71b1bb5f348.

VERIFICATION.json records only inherited checks and is not an overall pass; VERIFY_RECEIPT.json records the failed overall run. There are no training outcomes or checkpoint states. The original cap, FP32/FP64 tolerances, success thresholds and main seeds have not been changed.

A separately frozen v2 applies the original projection, reads back its decoded norm, and corrects only an actual overshoot using a32-epsilon inward margin, failing if three corrections cannot meet the strict cap. This common operator is shared by every feedback-updated arm and the common warmup. v1 verification and diagnosis CPU are charged to the same900-second stage budget. This is a numerical repair, not a scientific parameter search.
