# Magnitude control over the preserved local rule

`baseline_core.py`, `baseline_verify.py`, and both `BASELINE_*.md/json` files are byte-preserved from the hash-verified `adaptive-feedback-20261018-v1` source. See SOURCE_PROVENANCE.json. The source of the intervening bounded study was lost; this is a fresh implementation, not an exact reconstruction of that run.

The baseline stores every matrix in a complete real orthonormal Fourier basis. Dense synapse equations are unchanged; no compression or coefficient-wise substitute is introduced. See BASELINE_RULE_MAPPING.md for the original equation mapping.

New projection: after initialization and every feedback update, multiply each B/V/U coefficient vector by `min(1, 2*(1-4*eps)/spectral_norm(decoded_matrix))`. W is untouched. The full dense SVD is charged; no efficient or strictly local implementation claim follows.

For hidden layer k, let `u_A = local_gradient_A + .001*W_k`, and `u_F = local_gradient_F + .001*W_k`. Both are computed at the SAME current W and training batch. The matched frozen policy takes `W_k - .05*u_F*(norm(u_A)/norm(u_F))`; the reciprocal adaptive policy takes `W_k - .05*u_A*(norm(u_F)/norm(u_A))`. The complete SGD drive includes decay. The readout W2 has the same exact local error gradient at a common W and is not scaled.

For frozen_match_adaptive, the actual feedback stays frozen. A separate shadow feedback bank learns from that arm's own current forward activations and batch, using the original local rules and cap. It supplies two positive norm scalars only. For adaptive_match_frozen, the actual feedback adapts and an immutable warmup bank supplies the targets. Neither policy uses hidden exact gradients, holdout labels, phase or step indices. Exact hidden gradients appear in common BP warmup, the explicit BP arm and diagnostic copies only.

A zero target yields a zero drive. A nonzero target with source norm <=1e-30 is an arm failure; there is no arbitrary scale clipping. Every match audits relative norm error <=2e-6. This matches same-state policy magnitudes, not trajectories or FLOPs. Each matched arm carries an extra 12,544-coefficient feedback bank. All arms have 1,504 forward coefficients; the ordinary local model has 12,544 feedback coefficients.
