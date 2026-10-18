"""Fixed-storage replay intervention over the byte-preserved feedback learner."""
from feedback_core import *

REPLAY_ARMS = ('backprop_fifo', 'backprop_mixed', 'frozen_fifo', 'frozen_mixed',
               'adaptive_fifo', 'adaptive_mixed')


def make_replay_data(seed, warm_steps, phase_steps, query_n):
    """Same current examples/queries; both replay memories hold 512 examples.

    FIFO: 16 samples from a 512-example queue.
    Mixed: 8 from a 256-example queue, 8 from a fixed 256-example old bank.
    Both start from the SAME 512 warmup examples, and consume the SAME 16
    replay RNG draws per step. Modulo256 is uniform for draws in [0,512).
    Historical labels are never refreshed. Phase is used only to audit stale
    labels, never to choose a replay source, discard examples or change labels.
    """
    fifo = make_data(seed, warm_steps, phase_steps, query_n)
    mixed = dict(fifo)
    rng = np.random.default_rng(seed + 64000000)
    init_ix = rng.choice(len(fifo['base_x']), 512, replace=False)
    bx, by = fifo['base_x'].numpy(), fifo['base_y'].numpy()
    anchors_x, anchors_y = bx[init_ix[:256]].copy(), by[init_ix[:256]].copy()
    anchor_ids = -(init_ix[:256] + 1)
    recent_x, recent_y = bx[init_ix[256:]].copy(), by[init_ix[256:]].copy()
    recent_ids = -(init_ix[256:] + 1)
    count = 4 * phase_steps
    tx = np.empty((count, 32, D), dtype='float32')
    ty = np.empty((count, 32, R), dtype='float32')
    replay_ids = np.empty((count, 16), dtype='int64')
    stale_recent = np.zeros(count, dtype='int64')
    stale_anchor = np.zeros(count, dtype='int64')
    w = {k: v.numpy() for k, v in fifo['world'].items()}
    for t, (xt, yt) in enumerate(zip(fifo['current_x'], fifo['current_y'])):
        x, y = xt.numpy(), yt.numpy()
        draw = rng.integers(512, size=16)
        ri, ai = draw[:8] % 256, draw[8:] % 256
        tx[t] = np.concatenate([x, recent_x[ri], anchors_x[ai]])
        ty[t] = np.concatenate([y, recent_y[ri], anchors_y[ai]])
        replay_ids[t] = np.concatenate([recent_ids[ri], anchor_ids[ai]])
        if t >= 2 * phase_steps:
            stale_recent[t] = int((labels(w, recent_x[ri], True) != recent_y[ri]).sum())
            stale_anchor[t] = int((labels(w, anchors_x[ai], True) != anchors_y[ai]).sum())
        recent_x = np.concatenate([recent_x[16:], x])
        recent_y = np.concatenate([recent_y[16:], y])
        recent_ids = np.concatenate([recent_ids[16:], np.arange(16*t, 16*t+16)])
    mixed.update(train_x=torch.tensor(tx), train_y=torch.tensor(ty),
                 replay_ids=torch.tensor(replay_ids),
                 stale_replayed_facts=torch.tensor(stale_recent + stale_anchor))
    return {'fifo': fifo, 'mixed': mixed,
            'memory_audit': {'initial_ids': torch.tensor(-(init_ix + 1)),
                             'anchor_ids': torch.tensor(anchor_ids),
                             'mixed_stale_recent': torch.tensor(stale_recent),
                             'mixed_stale_anchor': torch.tensor(stale_anchor),
                             'stored_examples_per_policy': 512,
                             'mixed_recent_capacity': 256, 'mixed_anchor_capacity': 256,
                             'current_per_step': 16, 'replay_per_step': 16}}
