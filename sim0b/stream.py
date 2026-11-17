"""Token-level Rules-and-Exceptions stream for Stage 0b (stage0b_spec.md, section 1).

A fact is `k_a k_b v SEP`; its key is the pair (k_a, k_b). Each sequence draws a private rule set
(n_r keys, random values, Zipf frequencies), so nothing can be memorized in the weights.
With probability exc_rate a fact is an exception: a fresh key used once; with probability corr its
k_a is copied from a random rule key. 0 to 3 filler tokens go between facts. At the middle of the
sequence 25% of the rules get new values.

Training streams interleave probes `QRY k_a k_b v` (p = 0.15 after a fact, key drawn with a
log-uniform delay in facts). The loss mask marks probe answers and values of facts whose key
appeared earlier. Evaluation streams have no probes; `probes` lists every recurring rule (seen at
least twice) and every exception, to be asked after the stream with memory writes disabled.

Curriculum options (stage0b_spec.md, section 11): `zipf` (0 = uniform rules), filler range
[filler_min, filler_max], `probe_p`, and `key_len` = 1 (single-token keys, sanity run S1 only; `corr` is then
ignored because a copied first token would be the whole rule key), and `probe_delay` = "loguniform" (probe the
key of a fact drawn with a log-uniform delay in facts) or "uniform_key" (probe a key drawn uniformly over the
distinct keys seen so far; spec section 12). Defaults reproduce the original streams.

For held-out probe accuracy, `probe_pos` / `probe_delay` give each in-stream probe's answer position and the
distance from the key's latest earlier occurrence (fact or probe, its first key token) to the token that
predicts the answer.

Per-token `labels` are for diagnostics only. Model code must never read them.
"""
import numpy as np

PAD, BOS, SEP, QRY = 0, 1, 2, 3
N_K, N_V, N_F = 512, 512, 256
K0 = 4
V0 = K0 + N_K
F0 = V0 + N_V
VOCAB = F0 + N_F

# per-token labels (diagnostics only)
L_PAD, L_SPECIAL, L_FILLER, L_RULE_KEY, L_RULE_VAL, L_EXC_KEY, L_EXC_VAL, L_PROBE = range(8)


def make_sequence(rng, n_tokens, exc_rate=0.2, corr=0.3, n_rules=16, zipf=1.0, drift_frac=0.25,
                  probe_p=0.15, probes=True, filler_min=0, filler_max=3, key_len=2, probe_delay="loguniform"):
    """One sequence of exactly n_tokens tokens. Returns a dict of numpy arrays plus an eval probe list."""
    assert key_len in (1, 2) and probe_delay in ("loguniform", "uniform_key")
    if key_len == 1:
        corr = 0.0
    freq = 1.0 / np.arange(1, n_rules + 1) ** zipf
    freq /= freq.sum()
    rule_keys = set()
    while len(rule_keys) < n_rules:
        rule_keys.add(tuple(int(rng.integers(N_K)) for _ in range(key_len)))
    rule_keys = list(rule_keys)
    rule_val = rng.integers(N_V, size=n_rules)
    drift_ids = set(rng.choice(n_rules, int(round(drift_frac * n_rules)), replace=False).tolist())
    drifted = False
    used_keys = set(rule_keys)

    toks, labels, mask = [BOS], [L_SPECIAL], [False]
    facts = []        # (key, value, pos of k_a, kind, rule id or -1)
    last = {}         # key -> index into facts of its latest complete occurrence
    count = {}        # key -> number of complete occurrences
    seen_at = {}      # key -> position of the first key token of its latest occurrence (fact or probe)
    probe_pos, pdelay = [], []

    def emit(t, lab, m=False):
        toks.append(t); labels.append(lab); mask.append(m)

    while len(toks) < n_tokens:
        if not drifted and len(toks) >= n_tokens // 2:
            for r in drift_ids:
                rule_val[r] = rng.integers(N_V)
            drifted = True
        for _ in range(rng.integers(filler_min, filler_max + 1)):
            emit(F0 + int(rng.integers(N_F)), L_FILLER)
        if rng.random() < exc_rate:
            while True:
                ka = rule_keys[rng.integers(n_rules)][0] if rng.random() < corr else int(rng.integers(N_K))
                key = (ka, int(rng.integers(N_K))) if key_len == 2 else (ka,)
                if key not in used_keys:
                    break
            used_keys.add(key)
            val, kind, rid = int(rng.integers(N_V)), "exc", -1
            lk, lv = L_EXC_KEY, L_EXC_VAL
        else:
            rid = int(rng.choice(n_rules, p=freq))
            key, val, kind = rule_keys[rid], int(rule_val[rid]), "rule"
            lk, lv = L_RULE_KEY, L_RULE_VAL
        pos = len(toks)
        seen = key in count
        for kt in key:
            emit(K0 + kt, lk)
        emit(V0 + val, lv, seen); emit(SEP, L_SPECIAL)
        seen_at[key] = pos
        if len(toks) <= n_tokens:                       # count only facts whose value is inside the sequence
            facts.append((key, val, pos, kind, rid))
            last[key] = len(facts) - 1
            count[key] = count.get(key, 0) + 1
        if probes and facts and rng.random() < probe_p:
            if probe_delay == "uniform_key":
                keys = list(last)                                         # distinct keys seen so far
                key = keys[rng.integers(len(keys))]
            else:
                d = int(np.exp(rng.uniform(0, np.log(len(facts)))))      # log-uniform delay in facts, >= 1
                key = facts[len(facts) - max(1, d)][0]
            v = facts[last[key]][1]                                       # latest observed value of that key
            emit(QRY, L_PROBE)
            qpos = len(toks)
            for kt in key:
                emit(K0 + kt, L_PROBE)
            emit(V0 + v, L_PROBE, True)
            probe_pos.append(len(toks) - 1); pdelay.append(len(toks) - 2 - seen_at[key])
            seen_at[key] = qpos

    toks, labels, mask = toks[:n_tokens], labels[:n_tokens], mask[:n_tokens]
    probe_list = []
    for key, i in last.items():
        k, v, pos, kind, rid = facts[i]
        if kind == "rule" and count[key] < 2:
            continue
        old = None
        if kind == "rule" and rid in drift_ids:
            old = next((f[1] for f in reversed(facts) if f[0] == key and f[2] < n_tokens // 2), None)
        probe_list.append(dict(ka=K0 + k[0], kb=K0 + k[1] if key_len == 2 else None, v=V0 + v, kind=kind,
                               delay=n_tokens - pos, drifted=kind == "rule" and rid in drift_ids,
                               old_v=None if old is None else V0 + old, count=count[key]))
    keep = np.array(probe_pos, np.int64) < n_tokens
    return dict(tokens=np.array(toks, np.int64), labels=np.array(labels, np.int8),
                loss_mask=np.array(mask, bool), probes=probe_list,
                probe_pos=np.array(probe_pos, np.int64)[keep], probe_delay=np.array(pdelay, np.int64)[keep])


def make_batch(rng, batch, n_tokens, exc_rates=(0.05, 0.2, 0.5), corr=0.3, return_seqs=False, **kw):
    """Training batch: exc_rate drawn per sequence. Returns tokens (B, n) and loss_mask (B, n)."""
    seqs = [make_sequence(rng, n_tokens, exc_rate=float(rng.choice(exc_rates)), corr=corr, **kw) for _ in range(batch)]
    out = np.stack([s["tokens"] for s in seqs]), np.stack([s["loss_mask"] for s in seqs])
    return (*out, seqs) if return_seqs else out
