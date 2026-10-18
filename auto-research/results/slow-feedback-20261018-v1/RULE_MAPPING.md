# Fixed FIFO feedback plasticity

The original local equations, complete real orthonormal Fourier representation and bounded learned feedback remain in byte-preserved baseline_core.py and feedback_core.py. See BASELINE_RULE_MAPPING.md. The stream, FIFO512 memory,16 replay samples, current batch16, shared4096-step warmup and every forward hyperparameter remain fixed.

Let g_W and g_F be the original local drives evaluated at the complete pre-update state. For the slow arm:

- Forward: W_next = W -0.05*(g_W + weight_decay_W*W), unchanged.
- Feedback: F_candidate = F -0.05*0.1*(g_F + weight_decay_F*F).
- Apply the unchanged spectral cap2 to every feedback matrix, using margin1-4eps. No cap or scaling is applied to W.

The existing base feedback-rule factor0.8 is already included in g_F and remains unchanged. Decay is0.001 for B matrices and0.01 for V/U; the entire feedback drive including decay is slowed. Warmup remains original BP forward plus full local feedback updates; the0.1 multiplier applies only online. The multiplier has no phase, query, loss, source or time input.

Multiplier0 is exactly frozen feedback; multiplier1 matches normal adaptive feedback, verified over one and12 steps. At the same state, slow/normal/frozen forward updates are bitwise identical. Future forward trajectories can differ because their evolving feedback differs. The experiment does not isolate feedback-learning magnitude from feedback-decay speed, nor does it assert improved alignment with backprop.

The sole primary comparison is slow versus frozen with the same FIFO schedule. Normal adaptive and backprop are explanatory controls. All source, seeds, thresholds and the single0.1 rate are frozen before experimental checks or optimization; no rate search.
