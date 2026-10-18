# Rule mapping and scope

This is an independent SAM-inspired hybrid diagnostic. The reference is the gated 2026 formulation, arXiv 2412.20018v2, Algorithm 2, equations 1–4. The inspected author file is pinned by Git blob SHA in SOURCE_PROVENANCE.json. The source was read, not executed or used as a training dependency.

Our column-vector names: W0,W1,W2 are the three forward matrices; B1,B2 are downstream matrices; V1,V2 map forward activity into downstream coordinates; U0,U1 map downstream activity into forward coordinates. Hidden widths are 32 forward and 64 downstream. U2 is the identity at the linear readout. Input downstream matrices are omitted because they do not contribute to forward learning. Hidden upstream gates are ReLU masks, downstream gates are identity. No biases, normalization, or dropout are used.

With row-vector batches a and positive output derivative e3=(sigmoid(z)-y)/4:

```
e2 = e3 B2^T; e1 = e2 B1^T
gW0 = ((e1 U0^T) * gate1)^T x / batch
gW1 = ((e2 U1^T) * gate2)^T a1 / batch
gW2 = e3^T a2 / batch
gBj = 0.8 (aj Vj^T)^T e(j+1) / batch
gVj = 0.8 ej^T aj / batch
gU(j-1) = 0.8 aj^T ej / batch
```

Every matrix update subtracts 0.05 times its drive plus weight decay. Decay is 0.001 for W/B and 0.01 for V/U. Drives are computed from the same pre-update state. The positive-derivative/minus-update convention equals the paper's negative-derivative/plus-update convention. core.dense_reference independently evaluates column-vector sample outer products to check signs, orientations, gates, and normalization. The paired V/U drives are transposes; their mismatch decays by 1−0.05×0.01 per plastic step, before numerical error. This property alone does not establish useful feedback.

Forward and feedback matrices each occupy a separate complete real orthonormal Fourier coefficient vector. core.enc/dec preserve every degree of freedom. A local synapse update is computed after decoding, encoded, then applied to the coefficient store. Verification checks the corresponding dense step. A deliberately incorrect coordinatewise operation on reshaped coefficients must differ, demonstrating why one cannot simply reuse synapse rules in a different basis.

The local path uses an analytic output derivative and no autograd. It is not forward-only or derivative-free. Exact gradients used in diagnostic counterfactual probes are discarded and never reach that path. The shared warmup uses exact gradients on W and local shadow updates on B/V/U, simultaneously. This is an experimental intervention and does not inherit the paper's convergence claims.

Deviations from the author diagnostic benchmark: synthetic binary relations rather than image classification; two hidden layers of 32 and feedback width 64; BCE rather than categorical CE; shared backprop warmup; no momentum; learning rate 0.05; explicit online shifts and FIFO replay; full Fourier storage. The rates 1:0.8 and decay coefficients follow the inspected diagnostic settings, while dimensions, learning rate, schedule and hybrid warmup are our fixed choices. No general SAM reproduction, language-model result, continual-learning solution, biological validation, or compute advantage is claimed.
