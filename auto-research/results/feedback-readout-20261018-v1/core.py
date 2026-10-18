"""U0/U1 learning ablation with B and V fixed and online feedback decay off."""
import family_core as FAM
from family_core import *

READOUT_SWITCHES = {'frozen': (False, False), 'u0_only': (True, False),
                    'u1_only': (False, True), 'both_u': (True, True)}
READOUT_KEYS = {name: tuple(k for k, enabled in zip(('U0', 'U1'), switches) if enabled)
                for name, switches in READOUT_SWITCHES.items()}
REPLAY_ARMS = ('backprop_fifo', 'frozen_fifo', 'u0_only_fifo', 'u1_only_fifo', 'both_u_fifo')


@torch.no_grad()
def readout_step(state, x, y, learn_u0, learn_u1):
    if type(learn_u0) is not bool or type(learn_u1) is not bool:
        raise ValueError('Boolean readout switches required')
    active = tuple(k for k, enabled in zip(('U0', 'U1'), (learn_u0, learn_u1)) if enabled)
    g = local_grads(state, x, y, feedback=bool(active))
    new = apply(state, g, WK)
    if not active:
        return new
    for k in active:
        new[k] = state[k] - LR * g[k]
    projected = bounded(new)
    for k in FK:
        if k not in active:
            projected[k] = new[k]
    return projected


@torch.no_grad()
def policy_step(state, x, y, arm, reference=None, shadow=None):
    if arm in ('u0_only', 'u1_only', 'both_u'):
        return readout_step(state, x, y, *READOUT_SWITCHES[arm]), shadow, {}
    return FAM.policy_step(state, x, y, arm, reference, shadow)
