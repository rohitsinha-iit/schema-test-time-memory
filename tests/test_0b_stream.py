"""Stage 0b stream: masks, labels and probe lists are consistent."""
import numpy as np

from sim0b.stream import (BOS, F0, K0, N_F, QRY, SEP, V0, VOCAB, L_EXC_KEY, L_EXC_VAL, L_FILLER, L_PROBE,
                          L_RULE_KEY, L_RULE_VAL, L_SPECIAL, make_batch, make_sequence)


def _facts(s):
    """Parse complete facts (pos of k_a, key, value, label of value) from a sequence by its labels."""
    t, lab = s["tokens"], s["labels"]
    out = []
    for p in range(len(t) - 2):
        if lab[p] in (L_RULE_KEY, L_EXC_KEY) and (p == 0 or lab[p - 1] not in (L_RULE_KEY, L_EXC_KEY)):
            out.append((p, (t[p], t[p + 1]), t[p + 2], lab[p + 2]))
    return out


def test_token_ranges_match_labels():
    for seed, ex in [(0, 0.05), (1, 0.2), (2, 0.5)]:
        s = make_sequence(np.random.default_rng(seed), 4096, exc_rate=ex)
        t, lab = s["tokens"], s["labels"]
        assert len(t) == len(lab) == len(s["loss_mask"]) == 4096 and t[0] == BOS
        assert ((t >= 0) & (t < VOCAB)).all()
        isk, isv = (t >= K0) & (t < V0), (t >= V0) & (t < F0)
        assert (isk[np.isin(lab, [L_RULE_KEY, L_EXC_KEY])]).all()
        assert (isv[np.isin(lab, [L_RULE_VAL, L_EXC_VAL])]).all()
        assert ((t >= F0) & (t < F0 + N_F))[lab == L_FILLER].all()
        assert np.isin(t[lab == L_SPECIAL], [BOS, SEP]).all()


def test_loss_mask_only_on_predictable_values():
    s = make_sequence(np.random.default_rng(3), 4096, exc_rate=0.2)
    t, lab, m = s["tokens"], s["labels"], s["loss_mask"]
    assert np.isin(lab[m], [L_RULE_VAL, L_PROBE]).all()
    assert not m[lab == L_EXC_VAL].any()                       # exceptions are used once: never predictable
    seen = set()
    for p, key, v, lv in _facts(s):
        assert m[p + 2] == (key in seen)                       # masked iff the key appeared earlier
        seen.add(key)
    # every probe: QRY k_a k_b v, answer = latest value of that key before the probe, only v is masked
    for p in np.flatnonzero(t == QRY):
        if p + 3 >= len(t):
            continue
        assert (lab[p:p + 4] == L_PROBE).all() and m[p + 3] and not m[p:p + 3].any()
        key = (t[p + 1], t[p + 2])
        prev = [v for q, k, v, _ in _facts(s) if k == key and q < p]
        assert prev and t[p + 3] == prev[-1]


def test_exceptions_fresh_and_eval_probe_list():
    s = make_sequence(np.random.default_rng(4), 4096, exc_rate=0.5, probes=False)
    assert not (s["labels"] == L_PROBE).any() and not (s["tokens"] == QRY).any()
    facts = _facts(s)
    exc_keys = [k for p, k, v, lv in facts if lv == L_EXC_VAL]
    rule_keys = {k for p, k, v, lv in facts if lv == L_RULE_VAL}
    assert len(exc_keys) == len(set(exc_keys)) and not set(exc_keys) & rule_keys
    for pr in s["probes"]:
        key = (pr["ka"], pr["kb"])
        occ = [(p, v) for p, k, v, _ in facts if k == key]
        assert pr["v"] == occ[-1][1] and pr["delay"] == 4096 - occ[-1][0]
        assert (pr["kind"] == "exc") == (key in exc_keys)
        if pr["kind"] == "rule":
            assert len(occ) >= 2
    assert {(p["ka"], p["kb"]) for p in s["probes"] if p["kind"] == "exc"} == set(exc_keys)


def test_corr_copies_rule_first_token():
    s = make_sequence(np.random.default_rng(5), 4096, exc_rate=0.5, corr=1.0, probes=False)
    facts = _facts(s)
    rule_ka = {k[0] for p, k, v, lv in facts if lv == L_RULE_VAL}
    exc_ka = [k[0] for p, k, v, lv in facts if lv == L_EXC_VAL]
    assert np.mean([ka in rule_ka for ka in exc_ka]) > 0.95


def test_batch_shapes():
    tok, m = make_batch(np.random.default_rng(0), 3, 257)
    assert tok.shape == m.shape == (3, 257)


def test_default_streams_unchanged_by_curriculum_options():
    # fingerprint from the generator at commit a57fc13 (G0 eval streams must stay identical)
    import hashlib
    s = make_sequence(np.random.default_rng([100, 50]), 4096, exc_rate=0.05, probes=False)
    t, m = make_batch(np.random.default_rng([0, 5]), 2, 4097)
    h = hashlib.sha256(s["tokens"].tobytes() + t.tobytes() + m.tobytes()).hexdigest()
    assert h == "6dff666591e232b443c830cad7139cd0ccc111e7d6e08b76174ba288679d52f2"


def test_single_token_keys_no_filler_uniform():
    s = make_sequence(np.random.default_rng(6), 512, exc_rate=0.2, zipf=0.0, filler_max=0, probe_p=0.5, key_len=1)
    t, lab, m = s["tokens"], s["labels"], s["loss_mask"]
    assert not (lab == L_FILLER).any()
    for p in np.flatnonzero(np.isin(lab, [L_RULE_KEY, L_EXC_KEY])):     # fact: k v SEP
        assert lab[p + 1] in (L_RULE_VAL, L_EXC_VAL) if p + 1 < len(t) else True
        assert lab[p - 1] in (L_SPECIAL, L_PROBE)                        # preceded by BOS, SEP or a probe
    for p in np.flatnonzero(t == QRY):                                   # probe: QRY k v
        if p + 2 < len(t):
            assert (lab[p:p + 3] == L_PROBE).all() and m[p + 2] and not m[p:p + 2].any()
    assert all(pr["kb"] is None for pr in s["probes"])


def test_filler_range_and_uniform_rules():
    rng = np.random.default_rng(8)
    s = make_sequence(rng, 4096, exc_rate=0.0, zipf=0.0, filler_min=0, filler_max=1, probes=False)
    lab = s["labels"]
    runs, r = [], 0
    for x in lab[1:]:
        if x == L_FILLER:
            r += 1
        elif r or x in (L_RULE_KEY, L_EXC_KEY):
            runs.append(r); r = 0
    assert max(runs) <= 1 and 0 in runs and 1 in runs
    counts = [p["count"] for p in s["probes"]]                          # 16 rules, all recurring
    assert len(counts) == 16 and max(counts) < 3 * min(counts)


def test_probe_positions_and_delays():
    for key_len in (1, 2):
        s = make_sequence(np.random.default_rng(9), 512, exc_rate=0.2, zipf=0.0, filler_max=1, probe_p=0.5, key_len=key_len)
        t, lab, m = s["tokens"], s["labels"], s["loss_mask"]
        assert len(s["probe_pos"]) == len(s["probe_delay"]) > 0
        assert (lab[s["probe_pos"]] == L_PROBE).all() and m[s["probe_pos"]].all()
        for p, d in zip(s["probe_pos"], s["probe_delay"]):
            key = tuple(t[p - key_len:p])
            prev = p - 1 - d                                             # first key token of the latest occurrence
            assert tuple(t[prev:prev + key_len]) == key
            later = [q for q in range(prev + 1, p - key_len) if tuple(t[q:q + key_len]) == key
                     and lab[q] in (L_RULE_KEY, L_EXC_KEY, L_PROBE) and (q == 0 or lab[q - 1] != lab[q] or lab[q] == L_PROBE and t[q - 1] == QRY)]
            assert not later


def test_uniform_key_probes_cover_distinct_keys_uniformly():
    kw = dict(exc_rate=0.2, zipf=0.0, filler_max=0, probe_p=0.5, key_len=1)
    lo = make_sequence(np.random.default_rng(10), 512, **kw)
    un = make_sequence(np.random.default_rng(10), 512, probe_delay="uniform_key", **kw)
    for s in (lo, un):
        t, p = s["tokens"], s["probe_pos"]
        assert (t[p - 2] == QRY).all()                                   # QRY k v with single-token keys
    # copying the most recent fact value pays off far less with uniform keys
    def copy_last(s):
        t, lab, hit = s["tokens"], s["labels"], []
        for p in s["probe_pos"]:
            prev = [q for q in range(p - 2) if lab[q] in (L_RULE_VAL, L_EXC_VAL)]
            hit.append(bool(prev) and t[prev[-1]] == t[p])
        return np.mean(hit)
    many = [make_sequence(np.random.default_rng([11, i]), 512, probe_delay="uniform_key", **kw) for i in range(16)]
    assert np.mean([copy_last(s) for s in many]) < 0.15
    # probe delays spread over the sequence instead of piling up on the last fact
    few = [make_sequence(np.random.default_rng([11, i]), 512, **kw) for i in range(16)]
    med = lambda ss: np.median(np.concatenate([s["probe_delay"] for s in ss]))
    assert med(many) > 2 * med(few)
