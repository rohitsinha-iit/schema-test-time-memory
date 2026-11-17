"""Rules-and-Exceptions stream for the Stage 0a memory-only simulation.

Each step evicts one chunk of b key-value pairs from the (implicit) working window.
A pair is either an occurrence of a recurring rule (noisy copy of a fixed key/value)
or a one-off exception whose key has cosine `corr` with a randomly chosen rule key.
At `drift_t` a fraction of rules change their value, which tests updating.
Ground-truth labels (kinds, ids) are returned for evaluation only; the memory
algorithms never read them.
"""
import numpy as np


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def make_stream(seed, d=64, n_rules=32, T=2400, b=4, exc_rate=0.2, corr=0.3,
                drift_frac=0.25, key_noise=0.02, val_noise=0.02, zipf=1.0, late_frac=0.0):
    rng = np.random.default_rng(seed)
    K = unit(rng.normal(size=(n_rules, d)))
    V = unit(rng.normal(size=(n_rules, d)))
    freq = 1.0 / np.arange(1, n_rules + 1) ** zipf
    freq /= freq.sum()
    # E1 (16 Nov 2026): late-onset rules start at t_on; their first occurrence looks like a one-off exception
    late_ids = np.array([], dtype=int); t_on = np.zeros(n_rules, dtype=int)
    if late_frac > 0:
        late_ids = rng.choice(n_rules, int(round(late_frac * n_rules)), replace=False)
        t_on[late_ids] = rng.integers(T // 4, 3 * T // 5, size=len(late_ids))
    drift_t = T // 2
    drift_ids = rng.choice(n_rules, int(round(drift_frac * n_rules)), replace=False)
    V2 = V.copy()
    V2[drift_ids] = unit(rng.normal(size=(len(drift_ids), d)))

    keys = np.zeros((T, b, d)); vals = np.zeros((T, b, d))
    kinds = np.zeros((T, b), dtype=np.int8); ids = np.zeros((T, b), dtype=np.int64)
    exc_k, exc_v, exc_t, exc_rule = [], [], [], []
    rule_counts = np.zeros(n_rules, dtype=np.int64)
    for t in range(T):
        Vt = V if t < drift_t else V2
        for i in range(b):
            if rng.random() < exc_rate:
                r = rng.integers(n_rules)
                z = rng.normal(size=d)
                z -= (z @ K[r]) * K[r]
                z = z / np.linalg.norm(z)
                k = corr * K[r] + np.sqrt(1.0 - corr ** 2) * z
                v = unit(rng.normal(size=d))
                kinds[t, i] = 1; ids[t, i] = len(exc_k)
                exc_k.append(k); exc_v.append(v); exc_t.append(t); exc_rule.append(r)
            else:
                if late_frac > 0:
                    f_t = np.where(t_on <= t, freq, 0.0)
                    r = rng.choice(n_rules, p=f_t / f_t.sum())
                else:
                    r = rng.choice(n_rules, p=freq)
                k = unit(K[r] + key_noise * rng.normal(size=d))
                v = unit(Vt[r] + val_noise * rng.normal(size=d))
                kinds[t, i] = 0; ids[t, i] = r
                rule_counts[r] += 1
            keys[t, i] = k; vals[t, i] = v
    return dict(K=K, V=V, V2=V2, drift_t=drift_t, drift_ids=drift_ids,
                keys=keys, vals=vals, kinds=kinds, ids=ids,
                exc_k=np.array(exc_k).reshape(-1, d), exc_v=np.array(exc_v).reshape(-1, d),
                exc_t=np.array(exc_t), exc_rule=np.array(exc_rule),
                rule_counts=rule_counts, late_ids=late_ids, t_on=t_on, T=T, b=b, d=d)
