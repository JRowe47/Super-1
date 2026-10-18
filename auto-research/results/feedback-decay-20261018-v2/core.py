"""Factorial feedback local-learning/decay control with unchanged FIFO and W."""
import feedback_core as P
from feedback_core import *

REPLAY_ARMS = ('backprop_fifo','frozen_fifo','decay_fifo','learning_fifo','adaptive_fifo')
FEEDBACK_COMPONENTS = {'frozen': (False, False), 'decay': (False, True),
                       'learning': (True, False), 'adaptive': (True, True)}


@torch.no_grad()
def component_step(state, x, y, learning, decay):
    """Independent on/off switches; full original rates, pre-update local drives."""
    if type(learning) is not bool or type(decay) is not bool:
        raise ValueError('Boolean component switches required')
    g = local_grads(state, x, y, feedback=learning)
    new = apply(state, g, WK)
    if not learning and not decay:
        return new
    for k in FK:
        drive = g[k] if learning else torch.zeros_like(state[k])
        if decay:
            wd = WD_WB if k.startswith('B') else WD_VU
            drive = drive + wd * state[k]
        new[k] = state[k] - LR * drive
    return bounded(new)


@torch.no_grad()
def policy_step(state, x, y, arm, reference=None, shadow=None):
    if arm in ('decay','learning'):
        learning, decay = FEEDBACK_COMPONENTS[arm]
        return component_step(state, x, y, learning, decay), shadow, {}
    return P.policy_step(state, x, y, arm, reference, shadow)


def make_replay_data(seed, warm_steps, phase_steps, query_n):
    fifo = make_data(seed, warm_steps, phase_steps, query_n)
    rng = np.random.default_rng(seed + 64000000)
    init_ix = rng.choice(len(fifo['base_x']), 512, replace=False)
    return {'fifo': fifo,
            'memory_audit': {'initial_ids': torch.tensor(-(init_ix + 1)),
                             'stored_examples_per_policy': 512,
                             'current_per_step': 16, 'replay_per_step': 16}}
