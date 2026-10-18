# Feedback-family learning ablation

The complete real orthonormal Fourier stores, dense synapse rules, forward optimizer, FIFO generator and common repaired cap are preserved. See BASELINE_RULE_MAPPING.md and component_core.py. This tests which feedback matrices learn, not a new representation or recurrence.

Error transport: e3=(sigmoid(z)-y)/4; e2=e3 B2^T; e1=e2 B1^T. Local forward drives use e1 U0^T and e2 U1^T with ReLU gates. Feedback B drives use V-encoded activations and the next error signal; V/U have their original local drives. All are evaluated at the pre-update state; no BP enters online local updates.

- W_next = W -0.05*(g_W+0.001 W), unchanged.
- Active F_candidate = F -0.05*g_F, with original0.8 factor already in g_F; no online feedback decay.
- B family=(B1,B2),4352 parameters. V/U family=(V1,V2,U0,U1),8192 parameters.
- Frozen=(0,0), B-only=(1,0), V/U-only=(0,1), full learning-only=(1,1). Switches are fixed booleans with no query/phase/source/time input.
- Apply the preserved strict cap2 to updated matrices. Implementation calls the common full-bank operator and then restores each inactive matrix exactly; all computed SVD cost is charged. This prevents numerical recapping from changing a frozen family.
- Common4096-step warmup uses the original BP-forward/full-feedback-learning-and-decay update in every arm. Online decay is then off for all local arms.

Full learning-only and frozen endpoints match their preserved reference steps/trajectories bitwise. Same-state forward steps match across masks; subsequent W trajectories can differ as active feedback changes. Every inactive matrix must stay bitwise equal to warmup. B learning changes error signals that reach both hidden layers; this family intervention does not hold every early-layer influence constant. V affects B learning, while U directly decodes feedback for W; grouping them is a declared ablation, not proof of independent causal shares.

Primary B-only versus frozen tests >=5% mean paired CE improvement, no negative seed, and <=2pp additional old-window/final-old loss in every seed. V/U-only and full learning-only are explanatory controls, not alternative primaries.
