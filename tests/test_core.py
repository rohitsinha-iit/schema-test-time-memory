"""M0 unit tests for Stage 0a. Run: python -m pytest -q (from the repo root)"""
import numpy as np
import pytest
from sim.memories import MemorySystem, Hippocampus, Anchors
from sim.run import arms, run_one, H_SLOTS, R_ANCH
from sim.streams import make_stream, unit


def test_prop1_linear_gain_damage_and_optimal_step():
    rng = np.random.default_rng(0)
    dv, dk, nb, na, rho = 6, 10, 5, 4, 0.7
    W = rng.normal(size=(dv, dk)); KB = rng.normal(size=(dk, nb)); VB = rng.normal(size=(dv, nb))
    KA = rng.normal(size=(dk, na)); VA = W @ KA
    RB = W @ KB - VB; MBB = KB.T @ KB; MBA = KB.T @ KA
    lB = lambda X: 0.5 * np.sum((X @ KB - VB) ** 2)
    lA = lambda X: 0.5 * np.sum((X @ KA - VA) ** 2)
    for eta in [1e-3, 5e-3, 2e-2]:
        W2 = W - eta * RB @ KB.T
        G, D = lB(W) - lB(W2), lA(W2) - lA(W)
        Gf = eta * np.sum(RB * (RB @ MBB)) - 0.5 * eta ** 2 * np.sum((RB @ MBB) ** 2)
        Df = 0.5 * eta ** 2 * np.sum((RB @ MBA) ** 2)
        assert np.isclose(G, Gf, rtol=1e-8, atol=1e-10) and np.isclose(D, Df, rtol=1e-8, atol=1e-10)
    eta_star = np.sum(RB * (RB @ MBB)) / (np.sum((RB @ MBB) ** 2) + rho * np.sum((RB @ MBA) ** 2))
    obj = lambda e: (lB(W) - lB(W - e * RB @ KB.T)) - rho * (lA(W - e * RB @ KB.T) - lA(W))
    es = np.linspace(0, 3 * eta_star, 6001)
    assert abs(es[np.argmax([obj(e) for e in es])] / eta_star - 1) < 1e-3


def test_surprise_damage_formula():
    rng = np.random.default_rng(1)
    d = 12; W = rng.normal(size=(d, d))
    kr, ke = unit(rng.normal(size=d)), unit(rng.normal(size=d)); ve = rng.normal(size=d)
    e = ve - W @ ke; eta = 0.3
    W2 = W + eta * np.outer(e, ke)
    assert np.isclose(np.linalg.norm(W2 @ kr - W @ kr), eta * abs(ke @ kr) * np.linalg.norm(e))


def test_gate_accepts_only_net_positive_steps():
    cfg = arms(4)["schema"]
    S = make_stream(3, T=600, exc_rate=0.3, corr=0.6)
    ms = MemorySystem(S["d"], S["b"], {**cfg, "trace": True}, seed=3)
    for t in range(S["T"]):
        ms.observe(S["keys"][t], S["vals"][t], t)
    assert len(ms.trace) > 0
    assert all(G - rho * D > 0 for G, D, rho in ms.trace)


def test_equal_memory_budget():
    b, d = 4, 64
    pair = 2 * d
    for name, cfg in arms(b).items():
        slots = cfg.get("H_slots", 0) if cfg.get("use_H") else 0
        total = slots * b * pair + cfg.get("R", 0) * pair
        assert total == (H_SLOTS * b + R_ANCH) * pair, name


def test_slot_bookkeeping_and_supersede():
    H = Hippocampus(6, 4)
    ks = unit(np.eye(4)[:2]); vs = ks.copy()
    for j in range(2):
        H.put(j, ks[j], vs[j], 0)
    H.tag_and_supersede(ks[:1], vs[:1], 0.9, 0.8, supersede=True)   # agreeing copy: reuse tag
    assert H.r[0] == 1 and H.used[0]
    H.tag_and_supersede(ks[:1], -vs[:1], 0.9, 0.8, supersede=True)  # conflicting copy: supersede
    assert not H.used[0] and H.used[1] and H.n_superseded == 1


def test_redundancy_eviction_keeps_unpredicted_singletons():
    ms = MemorySystem(4, 1, dict(use_H=True, H_slots=2, H_policy="redundancy", dwell=1), seed=0)
    e = unit(np.eye(4))
    ms.W = np.diag([1.0, 0.0, 0.0, 0.0])          # cortex already predicts pair 0, not pair 1
    ms.H.put(0, e[0], e[0], 0); ms.H.put(1, e[1], e[1], 1)
    assert ms._slot_for_new(2) == 0


def test_young_landmark_is_never_evicted():
    ms = MemorySystem(4, 1, dict(use_H=True, H_slots=4, H_policy="redundancy", dwell=2), seed=0)
    e = unit(np.eye(4))
    ms.W = np.diag([0.0, 0.0, 0.0, 1.0])          # cortex predicts only the young landmark (slot 3)
    for j in range(3):
        ms.H.put(j, e[j], e[j], 0)
    ms.H.put(3, e[3], e[3], 9)
    for t in (9, 10):                              # ages 0 and 1 < dwell
        assert ms._slot_for_new(t) != 3
    assert ms._slot_for_new(11) == 3               # age 2 = dwell: now evictable, lowest keep
    assert ms.stats["forced_evictions"] == 0
    for j in range(4):                             # nothing past dwell: forced, oldest goes
        ms.H.put(j, e[j], e[j], 20 + j)
    assert ms._slot_for_new(21) == 0 and ms.stats["forced_evictions"] == 1


def test_dwell_must_fit_in_H():
    with pytest.raises(AssertionError):
        MemorySystem(4, 2, dict(use_H=True, H_slots=4, H_policy="redundancy", dwell=4), seed=0)


def test_older_unpredicted_singleton_evicted_first():
    ms = MemorySystem(4, 1, dict(use_H=True, H_slots=2, H_policy="redundancy", dwell=1), seed=0)
    e = unit(np.eye(4))                           # W = 0: both unpredicted, r = 0
    ms.H.put(0, e[1], e[1], 5); ms.H.put(1, e[0], e[0], 3)
    assert ms._slot_for_new(6) == 1


def test_predicted_landmark_evicted_before_unpredicted_same_age():
    ms = MemorySystem(4, 1, dict(use_H=True, H_slots=2, H_policy="redundancy", dwell=1), seed=0)
    e = unit(np.eye(4))
    ms.W = np.diag([0.0, 1.0, 0.0, 0.0])          # cortex predicts pair 1
    ms.H.put(0, e[0], e[0], 4); ms.H.put(1, e[1], e[1], 4)
    assert ms._slot_for_new(10) == 1


def test_anchor_invalidation_and_reservoir():
    rng = np.random.default_rng(0)
    A = Anchors(5, 4, rng)
    A.add(np.eye(4), np.eye(4)); A.add(np.eye(4), np.eye(4))
    assert len(A.k) == 5 and A.seen == 8
    assert A.invalidate_conflicts(np.eye(4)[:1], np.eye(4)[:1], 0.9, 0.8) == 0   # agreeing: kept
    n = A.invalidate_conflicts(np.eye(4)[:1], -np.eye(4)[:1], 0.9, 0.8)          # conflicting: dropped
    assert n >= 1 and (A.k @ np.eye(4)[0] <= 0.9).all()


def test_sum_gate_scales_terms_and_undilutes_damage():
    rng = np.random.default_rng(0)
    d, nB, nA = 6, 2, 50
    ms = {n: MemorySystem(d, 1, dict(use_H=True, H_slots=64, H_policy="fifo", R=nA, damage_norm=n), seed=0)
          for n in ("mean", "sum")}
    KA = unit(rng.normal(size=(nA, d))); W = rng.normal(size=(d, d))
    KB = unit(rng.normal(size=(nB, d))); VB = rng.normal(size=(nB, d))
    for m in ms.values():
        m.W = W.copy(); m.A.add(KA, KA @ W.T)
    W2 = W - 0.5 * MemorySystem.grad(W, KB, VB)
    lB0, lA0 = MemorySystem.loss(W, KB, VB), MemorySystem.loss(W, KA, KA @ W.T)
    Gm, Dm = ms["mean"].gate_terms(W2, KB, VB, lB0, lA0)
    Gs, Ds = ms["sum"].gate_terms(W2, KB, VB, lB0, lA0)
    assert np.isclose(Gs, nB * Gm) and np.isclose(Ds, nA * Dm)
    rho = Gm / Dm / 2.0                           # accepted under "mean", rejected under "sum" (nA > 2 nB)
    assert Gm - rho * Dm > 0 and Gs - rho * Ds < 0


def test_new_arms_configs():
    a = arms(4)
    assert a["schema_dsum"] == {**a["schema"], "damage_norm": "sum"}
    assert a["schema_dsum_rho4"] == {**a["schema"], "damage_norm": "sum", "rho": 4.0}
    assert a["schema_surprise_pri_dsum"] == {**a["schema_surprise_pri"], "damage_norm": "sum", "rho": 4.0}
    assert a["schema"]["H_policy"] == "fifo" and "schema_fifo_evict" not in a
    assert a["schema_redundancy_evict"] == {**a["schema"], "H_policy": "redundancy"}
    with pytest.raises(AssertionError):
        MemorySystem(4, 1, dict(damage_norm="max"))


def test_frequency_terciles_and_score():
    stream_kw = dict(T=300, exc_rate=0.2, corr=0.6)
    r = run_one("schema", arms(4)["schema"], stream_kw, seed=0)
    S = make_stream(0, **stream_kw)
    n = (S["rule_counts"] >= 2).sum()
    sizes = [len(x) for x in np.array_split(np.arange(n), 3)]
    parts = [r[f"cortex_rule_freq_{k}"] for k in ("hi", "mid", "lo")]
    assert np.isclose(np.dot(sizes, parts) / n, r["cortex_rule"])
    assert np.isclose(r["score"], (r["rule"] + r["cortex_rule"] + r["exc_mid"]) / 3)
    assert "rule_freq_hi" not in r                # terciles are reported for the cortex read only


def test_run_one_smoke():
    r = run_one("schema", arms(4)["schema"], dict(T=300), seed=0)
    assert np.isfinite(r["rule"]) and r["sleeps"] > 0


def test_late_onset_rules_e1():
    S = make_stream(5, T=800, late_frac=0.25)
    late = S["late_ids"]; t_on = S["t_on"]
    assert len(late) == 8
    for r in late:   # a late rule never occurs before its onset
        occ = np.argwhere((S["kinds"] == 0) & (S["ids"] == r))
        assert len(occ) == 0 or occ[:, 0].min() >= t_on[r]
    S0 = make_stream(5, T=300); S1 = make_stream(5, T=300, late_frac=0.0)
    assert np.array_equal(S0["keys"], S1["keys"])          # default stream unchanged


def test_agem_only_projects_without_lookahead():
    from sim.run import arms
    cfg = {**arms(4)["schema"], "gate": False, "agem_only": True}
    r = run_one("x", cfg, dict(T=300), seed=0)
    assert r["steps_shrunk"] == 0 and r["steps_accepted"] == r["steps_tried"] and np.isfinite(r["cortex_rule"])
