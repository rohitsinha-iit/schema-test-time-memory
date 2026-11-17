"""Stage 0a arms, evaluation and a single-run entry point."""
import numpy as np
from .streams import make_stream, unit
from .memories import MemorySystem

H_SLOTS, R_ANCH = 64, 256  # SCHEMA budget: 64 slots x b pairs + 256 anchor pairs + W


def arms(b):
    big = H_SLOTS + R_ANCH // b  # equal-bytes slot count for arms without anchors
    schema = dict(use_H=True, H_slots=H_SLOTS, H_policy="fifo", supersede=True, sleep="select",
                  priority="schema", route=True, gate=True, R=R_ANCH, interleave=0.5, reconsol=True)
    return {
        # single-store, write at arrival (AHN/AllMem-like), with an exact FIFO window of equal bytes
        "arrival_fifo": dict(use_H=True, H_slots=big, arrival_write=True),
        # Titans-like surprise-scaled writes, same window
        "surprise_fifo": dict(use_H=True, H_slots=big, arrival_write=True, surprise=True),
        # sleep over everything, then clear the buffer (Lee et al. style)
        "sleep_all": dict(use_H=True, H_slots=big, sleep="all"),
        # replay with uniform priority, no routing, no gate, no anchors
        "uniform_replay": dict(use_H=True, H_slots=big, sleep="select", priority="uniform"),
        "schema": schema,
        "schema_surprise_pri": {**schema, "priority": "surprise", "r_min": 0},
        "schema_nogate": {**schema, "gate": False},
        "schema_noreconsol": {**schema, "reconsol": False},
        "schema_nosupersede": {**schema, "supersede": False},
        # redundancy-aware eviction (FIFO is the default since the M1 decision, 3 Nov 2026)
        "schema_redundancy_evict": {**schema, "H_policy": "redundancy"},
        # gate with summed (undiluted) gain and damage, see stage0a_sanity_review.md
        "schema_dsum": {**schema, "damage_norm": "sum"},
        "schema_dsum_rho4": {**schema, "damage_norm": "sum", "rho": 4.0},
        "schema_surprise_pri_dsum": {**schema, "priority": "surprise", "r_min": 0, "damage_norm": "sum", "rho": 4.0},
    }


ETA_GRID = {"arrival_write": [0.1, 0.25, 0.5, 1.0], "sleep": [0.25, 0.5, 1.0, 2.0]}


def evaluate(ms, S):
    T = S["T"]
    Vcur = S["V2"]  # evaluation at end of stream, after drift
    rec = np.flatnonzero(S["rule_counts"] >= 2)
    drift = np.isin(rec, S["drift_ids"])
    # frequency terciles of the recurring rules: most frequent third first (stable on ties)
    order = np.argsort(-S["rule_counts"][rec], kind="stable")
    terc = dict(zip(("hi", "mid", "lo"), np.array_split(order, 3)))
    def cos(A, B):
        return np.sum(unit(A + 1e-12) * B, axis=1)
    out = {}
    for tag, co in [("", False), ("cortex_", True)]:
        R = ms.read(S["K"][rec], cortex_only=co)
        c_r = cos(R, Vcur[rec])
        out[tag + "rule"] = float(c_r.mean())
        out[tag + "rule_stable"] = float(c_r[~drift].mean()) if (~drift).any() else np.nan
        out[tag + "rule_drifted"] = float(c_r[drift].mean()) if drift.any() else np.nan
        out[tag + "rule_drifted_old"] = float(cos(R[drift], S["V"][rec][drift]).mean()) if drift.any() else np.nan
        if co:
            for name, ix in terc.items():
                out[f"cortex_rule_freq_{name}"] = float(c_r[ix].mean()) if len(ix) else np.nan
            late = np.isin(rec, S.get("late_ids", np.array([], dtype=int)))
            if late.any():                                      # E1
                out["cortex_rule_late"] = float(c_r[late].mean())
                out["cortex_rule_early"] = float(c_r[~late].mean())
        if len(S["exc_k"]):
            E = ms.read(S["exc_k"], cortex_only=co)
            c_e = cos(E, S["exc_v"])
            age = T - 1 - S["exc_t"]
            for name, lo, hi in [("exc_recent", 0, 64), ("exc_mid", 64, 256), ("exc_old", 256, 1024)]:
                m = (age >= lo) & (age < hi)
                out[tag + name] = float(c_e[m].mean()) if m.any() else np.nan
            out[tag + "exc_all"] = float(c_e.mean())
    out["score"] = (out["rule"] + out["cortex_rule"] + out.get("exc_mid", 0.0)) / 3.0
    out["anchor_loss"] = ms.anchor_loss()
    out["n_anchors"] = int(len(ms.A.k)) if ms.A is not None else 0
    if ms.H is not None:
        out["superseded"] = ms.H.n_superseded; out["cap_evictions"] = ms.H.n_capacity_evictions
    out.update({k: v for k, v in ms.stats.items()})
    return out


def run_one(arm, cfg, stream_kw, seed, eta=None):
    S = make_stream(seed, **stream_kw)
    cfg = dict(cfg)
    if eta is not None:
        cfg["eta_arr" if cfg.get("arrival_write") else "eta_sleep"] = eta
    ms = MemorySystem(S["d"], S["b"], cfg, seed=seed)
    for t in range(S["T"]):
        ms.observe(S["keys"][t], S["vals"][t], t)
    res = evaluate(ms, S)
    res.update(arm=arm, seed=seed, eta=eta, **stream_kw)
    return res


def eta_grid_for(cfg):
    return ETA_GRID["arrival_write"] if cfg.get("arrival_write") else ETA_GRID["sleep"]
