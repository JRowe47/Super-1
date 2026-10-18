# U0/U1 feedback-readout learning ablation

The full orthonormal Fourier stores, dense synapse equations, forward optimizer, FIFO generator and repaired cap are preserved. See BASELINE_RULE_MAPPING.md, component_core.py and family_core.py. This changes which feedback readout learns, not the representation or recurrence.

e3=(sigmoid(z)-y)/4; e2=e3 B2^T; e1=e2 B1^T. W0's drive uses e1 U0^T and its ReLU gate; W1 uses e2 U1^T and its ReLU gate. U0's drive is0.8*a1^T e1/N; U1's drive is0.8*a2^T e2/N. All drives use the pre-update state. BP is absent from online local updates.

- W_next=W-0.05*(g_W+0.001 W), unchanged in every local arm, including W0 under U1-only.
- Active U_next=cap(U-0.05*g_U). Online feedback decay is off. U0 and U1 each have2048 coefficients.
- B1/B2/V1/V2 are fixed bitwise at common warmup. InactiveU is also fixed bitwise.
- Frozen=(0,0), U0-only=(1,0), U1-only=(0,1), both-U=(1,1). These fixed booleans receive no phase/query/source/time input.
- Shared repaired cap2 is applied to the candidate full bank, followed by exact restoration of all inactive matrices. All SVD cost is charged; active outputs match the preserved operator.
- Common4096-step warmup retains original BP-forward/full-feedback-learning-and-decay for every arm.

With B fixed, V influences only the disabled B learning drive. Hence both-U and preserved V/U-only have identical W/U/B dynamics even though the latter learns V. Verification checks this bitwise at steps and12-step trajectories before pilot/main. This check does not count as another independent main seed. U1-only is not a fixed-feature intervention: W0 still learns through fixed U0 and can change as upstream activations/errors change.

Primary U1-only versus frozen must reach>=5% mean paired CE reduction with no negative seed and<=2pp extra old-window/final-old loss in every seed. Secondary U0-only and both-U results cannot replace the primary. All tolerances, data, windows, rates and guards remain unchanged.
