# Feedback learning and decay separated

Use unchanged full real orthonormal Fourier representation, local synapse equations and original spectral cap. baseline_core.py and feedback_core.py are byte-preserved. See BASELINE_RULE_MAPPING.md. FIFO512,16 replay+16 current, forward optimizer and4096-step common warmup are fixed.

With pre-update local forward drive g_W and feedback drive g_F:

- W_next = W -0.05*(g_W+0.001*W), unchanged in all local arms.
- F_candidate = F -0.05*(L*g_F + D*wd_F*F).
- L and D are fixed booleans. Frozen=(0,0); decay-only=(0,1); learning-only=(1,0); normal adaptive=(1,1).
- wd_F is0.001 for B and0.01 for V/U. g_F already includes the unchanged rule factor0.8.
- Apply spectral cap2 (margin1-4eps) after enabled feedback updates. Frozen returns original feedback without an extra projection.

Common warmup uses BP forward and full local feedback+decay for every arm. Online switches have no phase, loss, query, source, or time input. The decay-only feedback path has no data-dependent local feedback term; its feedback trajectory must be identical under different input/label streams. From a capped initial state, its unprojected analytic trajectory is F_t=(1-0.05*wd_F)^t*F_0; projection is inactive as norms shrink. Numerical checks compare12 steps at the original tolerances.

Forward formulas match bitwise at a common state. Forward trajectories can subsequently differ as feedback changes. The factorial includes interactions; additive causal shares are not assumed. Primary normal adaptive versus decay-only tests a learned-feedback contribution beyond passive decay. The original normal-versus-frozen retention gap is separately reported; a narrower mechanism gate cannot establish a general solution or language/architecture promotion.
