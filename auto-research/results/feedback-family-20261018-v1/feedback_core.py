"""Fresh magnitude-control implementation over the preserved Fourier reference."""
from baseline_core import *  # Re-export unchanged reference for baseline_verify.py.

EXPERIMENT_ARMS = ('backprop', 'random_fixed', 'frozen', 'adaptive',
                   'frozen_match_adaptive', 'adaptive_match_frozen')
MATCH_ARMS = ('frozen_match_adaptive', 'adaptive_match_frozen')
CAP = 2.0


@torch.no_grad()
def bounded(state):
    """Uniform coefficient scaling; a full dense SVD is charged to compute."""
    out = clone(state)
    for k in FK:
        sigma = torch.linalg.svdvals(dec(state[k]).reshape(SHAPES[k]))[0]
        limit = CAP * (1 - 4 * torch.finfo(state[k].dtype).eps)
        factor = min(1., limit / float(sigma)) if float(sigma) > 0 else 1.
        out[k] = state[k] * factor
        # Preserve the original projection unless decoded readback exceeds CAP.
        # This shared numerical repair also applies during common warmup.
        observed = float(torch.linalg.svdvals(dec(out[k]).reshape(SHAPES[k]))[0])
        for _ in range(3):
            if not math.isfinite(observed):
                raise FloatingPointError('Nonfinite projected feedback norm')
            if observed <= CAP:
                break
            out[k] = out[k] * (CAP * (1 - 32 * torch.finfo(out[k].dtype).eps) / observed)
            observed = float(torch.linalg.svdvals(dec(out[k]).reshape(SHAPES[k]))[0])
        if not math.isfinite(observed) or observed > CAP:
            raise FloatingPointError(('Spectral cap correction failed', k, observed))
    return out


def with_feedback(state, bank):
    return {**{k: state[k] for k in WK}, **{k: bank[k] for k in FK}}


@torch.no_grad()
def matched_drive(source, target):
    ns, nt = float(source.norm()), float(target.norm())
    if not math.isfinite(ns + nt):
        raise FloatingPointError('Nonfinite magnitude')
    if nt == 0:
        result, scale = torch.zeros_like(source), 0.
    elif ns <= 1e-30:
        raise FloatingPointError('Nonzero target with unmatchable source direction')
    else:
        scale = nt / ns
        result = source * scale
    error = abs(float(result.norm()) - nt) / max(nt, 1e-30)
    if error > 2e-6:
        raise AssertionError(('Magnitude mismatch', error))
    return result, {'scale': scale, 'relative_norm_error': error,
                    'source_norm': ns, 'target_norm': nt}


@torch.no_grad()
def policy_step(state, x, y, arm, reference=None, shadow=None):
    """Only training state/batch enter. No step, source ID, or holdout inputs.

    Match the FULL SGD drive (local gradient plus decay) for W0 and W1.
    W2 is identical for all local policies at a common forward state.
    """
    if arm not in EXPERIMENT_ARMS:
        raise ValueError(arm)
    if arm == 'backprop':
        return apply(state, exact_grads(state, x, y), WK), shadow, {}
    adapting = arm in ('adaptive', 'adaptive_match_frozen')
    g = local_grads(state, x, y, feedback=adapting)
    new = bounded(apply(state, g, SHAPES)) if adapting else apply(state, g, WK)
    audit = {}
    if arm == 'frozen_match_adaptive':
        if shadow is None:
            raise ValueError('Missing adaptive shadow')
        target_state = with_feedback(state, shadow)
        target_g = local_grads(target_state, x, y, feedback=True)
        shadow_new = bounded(apply(target_state, target_g, FK))
        shadow = {k: shadow_new[k] for k in FK}
    elif arm == 'adaptive_match_frozen':
        if reference is None:
            raise ValueError('Missing frozen reference')
        target_g = local_grads(with_feedback(state, reference), x, y, feedback=False)
    if arm in MATCH_ARMS:
        assert torch.equal(g['W2'], target_g['W2'])
        for k in ('W0', 'W1'):
            source = g[k] + WD_WB * state[k]
            target = target_g[k] + WD_WB * state[k]
            drive, audit[k] = matched_drive(source, target)
            new[k] = state[k] - LR * drive
    return new, shadow, audit


def warm_step(state, x, y):
    local = local_grads(state, x, y, feedback=True)
    exact = exact_grads(state, x, y)
    return bounded(apply(state, {**local, **exact}, SHAPES))
