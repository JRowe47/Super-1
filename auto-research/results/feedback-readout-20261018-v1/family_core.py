"""Feedback-family learning switches over the preserved local-rule reference."""
import component_core as Q
from component_core import *

B_KEYS = ('B1', 'B2')
VU_KEYS = ('V1', 'V2', 'U0', 'U1')
FAMILY_SWITCHES = {'frozen': (False, False), 'b_only': (True, False),
                   'vu_only': (False, True), 'learning': (True, True)}
FAMILY_KEYS = {name: tuple(k for k in FK if (k in B_KEYS and b) or (k in VU_KEYS and vu))
               for name, (b, vu) in FAMILY_SWITCHES.items()}
REPLAY_ARMS = ('backprop_fifo', 'frozen_fifo', 'b_only_fifo', 'vu_only_fifo', 'learning_fifo')


@torch.no_grad()
def family_step(state, x, y, learn_b, learn_vu):
    """Fixed family masks, full local learning rate, no online feedback decay."""
    if type(learn_b) is not bool or type(learn_vu) is not bool:
        raise ValueError('Boolean family switches required')
    active = tuple(k for k in FK if (k in B_KEYS and learn_b) or (k in VU_KEYS and learn_vu))
    g = local_grads(state, x, y, feedback=bool(active))
    new = apply(state, g, WK)
    if not active:
        return new
    for k in active:
        new[k] = state[k] - LR * g[k]
    projected = bounded(new)
    # The inherited cap operates on a full bank. Restore inactive matrices so
    # roundoff in a redundant projection cannot change a frozen coefficient.
    for k in FK:
        if k not in active:
            projected[k] = new[k]
    return projected


@torch.no_grad()
def policy_step(state, x, y, arm, reference=None, shadow=None):
    if arm in ('b_only', 'vu_only'):
        return family_step(state, x, y, *FAMILY_SWITCHES[arm]), shadow, {}
    return Q.policy_step(state, x, y, arm, reference, shadow)
