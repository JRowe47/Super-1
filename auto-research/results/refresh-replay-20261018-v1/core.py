"""Causal one-row bank refresh over byte-preserved permanent/FIFO controls."""
import replay_core as P
from replay_core import *

REPLAY_ARMS = tuple(policy + '_' + replay for policy in ('backprop', 'frozen', 'adaptive')
                    for replay in ('fifo', 'mixed', 'refreshed'))


def make_replay_data(seed, warm_steps, phase_steps, query_n):
    schedules = P.make_replay_data(seed, warm_steps, phase_steps, query_n)
    fifo = schedules['fifo']
    refreshed = dict(fifo)
    rng = np.random.default_rng(seed + 64000000)
    refresh_rng = np.random.default_rng(seed + 66000000)
    init_ix = rng.choice(len(fifo['base_x']), 512, replace=False)
    bx, by = fifo['base_x'].numpy(), fifo['base_y'].numpy()
    bank_x, bank_y = bx[init_ix[:256]].copy(), by[init_ix[:256]].copy()
    bank_ids = -(init_ix[:256] + 1)
    recent_x, recent_y = bx[init_ix[256:]].copy(), by[init_ix[256:]].copy()
    recent_ids = -(init_ix[256:] + 1)
    count = 4 * phase_steps
    tx = np.empty((count, 32, D), dtype='float32')
    ty = np.empty((count, 32, R), dtype='float32')
    replay_ids = np.empty((count, 16), dtype='int64')
    replacement_ids = np.empty(count, dtype='int64')
    sampled_bank_age = np.empty((count, 8), dtype='int64')
    stale_recent = np.zeros(count, dtype='int64')
    stale_bank = np.zeros(count, dtype='int64')
    w = {k: v.numpy() for k, v in fifo['world'].items()}
    for t, (xt, yt) in enumerate(zip(fifo['current_x'], fifo['current_y'])):
        x, y = xt.numpy(), yt.numpy()
        draw = rng.integers(512, size=16)
        ri, bi = draw[:8] % 256, draw[8:] % 256
        tx[t] = np.concatenate([x, recent_x[ri], bank_x[bi]])
        ty[t] = np.concatenate([y, recent_y[ri], bank_y[bi]])
        replay_ids[t] = np.concatenate([recent_ids[ri], bank_ids[bi]])
        # Initial examples predate online training; -1 is a bookkeeping birth time.
        birth = np.where(bank_ids[bi] < 0, -1, bank_ids[bi] // 16)
        sampled_bank_age[t] = t - birth
        # Oracle labels are diagnostic only, never a refresh/sampling input.
        if t >= 2 * phase_steps:
            stale_recent[t] = int((labels(w, recent_x[ri], True) != recent_y[ri]).sum())
            stale_bank[t] = int((labels(w, bank_x[bi], True) != bank_y[bi]).sum())
        # Both mutations follow replay sampling. Labels are copied as observed.
        recent_x = np.concatenate([recent_x[16:], x])
        recent_y = np.concatenate([recent_y[16:], y])
        recent_ids = np.concatenate([recent_ids[16:], np.arange(16*t, 16*t+16)])
        selected = int(refresh_rng.integers(16))
        slot = t % 256
        bank_x[slot], bank_y[slot] = x[selected], y[selected]
        bank_ids[slot] = replacement_ids[t] = 16*t + selected
    refreshed.update(train_x=torch.tensor(tx), train_y=torch.tensor(ty),
                     replay_ids=torch.tensor(replay_ids),
                     stale_replayed_facts=torch.tensor(stale_recent + stale_bank))
    schedules['refreshed'] = refreshed
    schedules['memory_audit'].update(
        refreshed_stale_recent=torch.tensor(stale_recent),
        refreshed_stale_bank=torch.tensor(stale_bank),
        refreshed_replacement_ids=torch.tensor(replacement_ids),
        refreshed_sampled_bank_age=torch.tensor(sampled_bank_age),
        refreshed_recent_capacity=256, refreshed_bank_capacity=256,
        replacements_per_step=1, refresh_rng_offset=66000000)
    return schedules
