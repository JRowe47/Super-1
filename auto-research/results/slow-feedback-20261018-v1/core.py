"""One fixed feedback-plasticity intervention; unchanged forward SGD and FIFO."""
import feedback_core as P
from feedback_core import *

REPLAY_ARMS = ('backprop_fifo', 'frozen_fifo', 'adaptive_fifo', 'slow_fifo')
FEEDBACK_MULTIPLIER = 0.1


@torch.no_grad()
def feedback_step(state, x, y, multiplier):
    """Scale the complete feedback SGD drive, including decay, before the cap.

    Forward updates use the original local drive at the pre-update state.
    This function has no time, phase, source, query or model-selection input.
    """
    if not 0 <= multiplier <= 1:
        raise ValueError(multiplier)
    g = local_grads(state, x, y, feedback=multiplier != 0)
    new = apply(state, g, WK)
    if multiplier == 0:
        return new
    for k in FK:
        wd = WD_WB if k.startswith('B') else WD_VU
        new[k] = state[k] - (LR * multiplier) * (g[k] + wd * state[k])
    return bounded(new)


@torch.no_grad()
def policy_step(state, x, y, arm, reference=None, shadow=None):
    if arm == 'slow':
        return feedback_step(state, x, y, FEEDBACK_MULTIPLIER), shadow, {}
    return P.policy_step(state, x, y, arm, reference, shadow)


def make_replay_data(seed, warm_steps, phase_steps, query_n):
    fifo = make_data(seed, warm_steps, phase_steps, query_n)
    rng = np.random.default_rng(seed + 64000000)
    init_ix = rng.choice(len(fifo['base_x']), 512, replace=False)
    return {'fifo': fifo,
            'memory_audit': {'initial_ids': torch.tensor(-(init_ix + 1)),
                             'stored_examples_per_policy': 512,
                             'current_per_step': 16, 'replay_per_step': 16}}
