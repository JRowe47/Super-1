"""Four registers, eight values, explicit lock/unlock, no inference oracle.

W(k,v)=8*k+v; Q(k)=32+k; LOCK(k)=36+k; UNLOCK(k)=40+k;
eight distractors 44..51. All episodes start afresh. Simulator values are only
loss targets, never model inputs except the visible write tokens themselves.
"""
import numpy as np
import torch

def episode(seed, batch=32, delay=32, mode='train'):
    rng = np.random.default_rng(seed)
    values = rng.integers(0, 8, size=(batch, 4))
    locked = np.zeros((batch, 4), bool)
    rows = np.arange(batch)
    tokens, labels = [], []
    initial = values.copy()
    lock_order = np.stack([rng.permutation(4) for _ in range(batch)])

    def append(tok):
        tok = np.asarray(tok, dtype=np.int64)
        y = np.full(batch, -100, dtype=np.int64)
        write = tok < 32
        k = np.clip(tok//8, 0, 3)
        accept = write & ~locked[rows, k]
        values[rows[accept], k[accept]] = tok[accept] % 8
        for offset, setting in ((36, True), (40, False)):
            mask = (tok >= offset) & (tok < offset+4)
            locked[rows[mask], tok[mask]-offset] = setting
        query = (tok >= 32) & (tok < 36)
        y[query] = values[rows[query], tok[query]-32]
        tokens.append(tok.copy()); labels.append(y)

    for k in range(4):
        append(8*k + initial[:, k])
    for j in range(2):
        append(36 + lock_order[:, j])
    for t in range(delay):
        u = rng.random(batch)
        tok = 44 + rng.integers(0, 8, size=batch)
        if mode != 'retention':
            wr = u < .45
            tok[wr] = rng.integers(0, 32, size=wr.sum())
        # Queries are read-only in the simulator, not hard-masked in the model.
        qr = (u >= .45) & (u < .60)
        tok[qr] = 32 + rng.integers(0, 4, size=qr.sum())
        append(tok)
    reopen = np.full(batch, mode == 'reopen')
    if mode == 'train':
        reopen = rng.random(batch) < .5
    key = lock_order[:, 0]
    append(np.where(reopen, 40+key, 44+rng.integers(0, 8, size=batch)))
    new = (values[rows, key]+rng.integers(1, 8, size=batch)) % 8
    append(np.where(reopen, 8*key+new, 44+rng.integers(0, 8, size=batch)))
    for k in range(4):
        append(np.full(batch, 32+k))
    x = torch.from_numpy(np.stack(tokens, 1))
    y = torch.from_numpy(np.stack(labels, 1))
    return x, y, dict(locked=torch.from_numpy(locked.copy()),
                     reopen_key=torch.from_numpy(key), reopen=torch.from_numpy(reopen))

def check_oracle(x, y):
    """Independent scalar replay of token semantics."""
    for b in range(len(x)):
        vals, locks = [None]*4, [False]*4
        for t, token in enumerate(x[b].tolist()):
            label = int(y[b, t])
            if token < 32:
                k, v = divmod(token, 8)
                if not locks[k]: vals[k] = v
            elif token < 36:
                assert label == vals[token-32]
            elif token < 40:
                locks[token-36] = True
            elif token < 44:
                locks[token-40] = False
            if not 32 <= token < 36:
                assert label == -100
    return True
