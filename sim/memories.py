"""Memory-only arms for Stage 0a (linear cortex).

All arms share the same linear cortex W (d_v x d_k), read as W k. Arms differ in
how W is written (at arrival, or only through replay during micro-sleep) and in
whether an episodic store H (hippocampus) and an anchor reservoir A exist.
The algorithms only see keys and values; ground-truth labels are never used.

Design fixes after the first smoke run (3 Nov 2026):
  * eviction under capacity pressure removes *redundant* landmarks (already
    predicted by the cortex, or with many agreeing copies), not one-off ones;
  * a new landmark with the same cue but a conflicting value supersedes older
    copies in H (so stale copies are not replayed after a change);
  * anchors are invalidated only when a replayed value contradicts them.

Eviction fix (3 Nov 2026): plain delta / (1 + r) hoarded old exceptions (their
delta grows), so every new landmark was evicted at once. Now a landmark younger
than `dwell` chunks cannot be evicted, and the keep score decays with age:
keep = delta / (1 + r) * exp(-age / value_T).
"""
import numpy as np

DEFAULTS = dict(
    # episodic store H
    use_H=False, H_slots=64, H_policy="fifo", hit_thresh=0.95, cue_thresh=0.9, agree_thresh=0.8,
    supersede=False, dwell=32, value_T=512.0,   # redundancy eviction: min dwell (chunks), keep-score age scale
    # writes at arrival (AHN / AllMem / Titans behaviour)
    arrival_write=False, eta_arr=0.5, surprise=False, decay=0.0,
    # micro-sleep consolidation
    sleep=None,              # None | "all" (consolidate everything, clear H) | "select"
    sleep_every=16, sleep_steps=8, batch=32, eta_sleep=1.0,
    priority="uniform",      # "uniform" | "schema" (reuse x error) | "surprise" (error only)
    route=False, r_min=1, tau_s=0.1, release_consolidated=True,
    # interference gate and anchors
    gate=False, rho=1.0, backtracks=2, agem=True,
    damage_norm="mean",      # gate terms: "mean" (per item) | "sum" (G over replayed items, D over all anchors)
    R=0, interleave=0.5, reconsol=True,
    trace=False,
)


def _unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


class Anchors:
    def __init__(self, R, d, rng):
        self.R, self.rng = R, rng
        self.k = np.zeros((0, d)); self.v = np.zeros((0, d)); self.seen = 0

    def add(self, ks, vs):
        for k, v in zip(ks, vs):
            self.seen += 1
            if len(self.k) < self.R:
                self.k = np.vstack([self.k, k]); self.v = np.vstack([self.v, v])
            else:
                j = self.rng.integers(self.seen)
                if j < self.R:
                    self.k[j] = k; self.v[j] = v

    def invalidate_conflicts(self, keys, vals, cue_thresh, agree_thresh):
        """Drop anchors whose cue matches a replayed key but whose stored output disagrees."""
        if len(self.k) == 0:
            return 0
        ks = self.k @ keys.T                                   # (R, n)
        agree = _unit(self.v) @ _unit(vals).T                  # (R, n)
        conflict = ((ks > cue_thresh) & (agree < agree_thresh)).any(axis=1)
        n = int(conflict.sum())
        if n:
            self.k, self.v = self.k[~conflict], self.v[~conflict]
        return n

    def sample(self, n):
        if len(self.k) == 0 or n <= 0:
            return self.k[:0], self.v[:0]
        idx = self.rng.choice(len(self.k), size=min(n, len(self.k)), replace=False)
        return self.k[idx], self.v[idx]


class Hippocampus:
    """Episodic store with landmark-level tags (reuse r, insertion time)."""

    def __init__(self, n_pairs, d):
        self.N = n_pairs
        self.k = np.zeros((n_pairs, d)); self.v = np.zeros((n_pairs, d))
        self.used = np.zeros(n_pairs, bool); self.r = np.zeros(n_pairs); self.t_in = np.zeros(n_pairs)
        self.n_capacity_evictions = 0; self.n_superseded = 0

    def idx(self):
        return np.flatnonzero(self.used)

    def tag_and_supersede(self, ks, vs, cue_thresh, agree_thresh, supersede):
        """Reuse tags for agreeing copies; optionally release conflicting (stale) copies."""
        idx = self.idx()
        if len(idx) == 0:
            return
        cue = ks @ self.k[idx].T                               # (b, n)
        agree = _unit(vs) @ _unit(self.v[idx]).T
        same = cue > cue_thresh
        self.r[idx[(same & (agree >= agree_thresh)).any(axis=0)]] += 1
        if supersede:
            stale = idx[(same & (agree < agree_thresh)).any(axis=0)]
            if len(stale):
                self.release(stale); self.n_superseded += len(stale)

    def release(self, idx):
        self.used[idx] = False; self.r[idx] = 0

    def put(self, j, k, v, t):
        self.k[j], self.v[j] = k, v
        self.used[j] = True; self.r[j] = 0; self.t_in[j] = t

    def lookup(self, Q, hit_thresh):
        """Most recent landmark among those whose cue matches the query."""
        idx = self.idx()
        if len(idx) == 0:
            return None, np.zeros(len(Q), bool)
        s = Q @ self.k[idx].T
        hit = s > hit_thresh
        score = np.where(hit, self.t_in[idx][None, :], -np.inf)
        j = score.argmax(axis=1)
        return self.v[idx][j], hit.any(axis=1)


class MemorySystem:
    def __init__(self, d, b, cfg, seed=0):
        self.c = {**DEFAULTS, **cfg}
        self.d, self.b = d, b
        self.rng = np.random.default_rng(seed + 12345)
        self.W = np.zeros((d, d))
        c = self.c
        assert c["damage_norm"] in ("mean", "sum"), c["damage_norm"]
        self.H = Hippocampus(c["H_slots"] * b, d) if c["use_H"] else None
        if self.H is not None and c["H_policy"] != "fifo":
            assert c["dwell"] * b < self.H.N, "dwell * b must be below H capacity in pairs"
        self.A = Anchors(c["R"], d, self.rng) if c["R"] > 0 else None
        self.stats = dict(steps_tried=0, steps_accepted=0, steps_shrunk=0, anchors_invalidated=0,
                          released_consistent=0, released_consolidated=0, sleeps=0, forced_evictions=0)
        self.trace = []

    # ----- cortex helpers -----
    @staticmethod
    def loss(W, K, V):
        if len(K) == 0:
            return 0.0
        E = K @ W.T - V
        return 0.5 * float(np.mean(np.sum(E * E, axis=1)))

    @staticmethod
    def grad(W, K, V):
        E = K @ W.T - V
        return E.T @ K / len(K)

    def gate_terms(self, W2, KB, VB, lB0, lA0):
        """Gain G on the replayed batch and damage D on the anchors for the step W -> W2.

        lB0, lA0 are the per-item mean losses at W. With damage_norm "sum" both terms are
        totals (G times the batch size, D times the number of anchors), so damage is not
        diluted by averaging over a reservoir that is larger than the batch."""
        G = lB0 - self.loss(W2, KB, VB)
        D = self.loss(W2, self.A.k, self.A.v) - lA0
        if self.c["damage_norm"] == "sum":
            G, D = G * len(KB), D * len(self.A.k)
        return G, D

    def delta(self, idx):
        E = self.H.k[idx] @ self.W.T - self.H.v[idx]
        return np.sum(E * E, axis=1)  # values are unit norm, so this is a normalized error

    # ----- capacity -----
    def _slot_for_new(self, t):
        free = np.flatnonzero(~self.H.used)
        if len(free):
            return free[0]
        self.H.n_capacity_evictions += 1
        idx = self.H.idx()
        if self.c["H_policy"] == "fifo":
            return idx[np.argmin(self.H.t_in[idx])]
        # redundancy-aware: keep landmarks the cortex cannot predict and that have few copies,
        # but only once they are past the dwell time, and with a score that decays with age
        age = t - self.H.t_in[idx]
        ok = age >= self.c["dwell"]
        if not ok.any():
            self.stats["forced_evictions"] += 1
            return idx[np.argmin(self.H.t_in[idx])]
        idx, age = idx[ok], age[ok]
        keep = self.delta(idx) / (1.0 + self.H.r[idx]) * np.exp(-age / self.c["value_T"])
        return idx[np.lexsort((self.H.t_in[idx], keep))[0]]

    # ----- stream interface -----
    def observe(self, ks, vs, t):
        c = self.c
        if self.H is not None:
            self.H.tag_and_supersede(ks, vs, c["cue_thresh"], c["agree_thresh"], c["supersede"])
        if c["arrival_write"]:
            for k, v in zip(ks, vs):
                e = self.W @ k - v
                eta = c["eta_arr"] * (np.linalg.norm(e) if c["surprise"] else 1.0)
                self.W -= eta * np.outer(e, k)
                if c["decay"] > 0:
                    self.W *= (1.0 - c["decay"])
        if self.H is not None:
            if c["sleep"] == "all" and (~self.H.used).sum() < self.b:
                self._sleep_all()
            for k, v in zip(ks, vs):
                self.H.put(self._slot_for_new(t), k, v, t)
            if c["sleep"] == "select" and (t + 1) % c["sleep_every"] == 0:
                self._sleep_select()

    def _sleep_all(self):
        c = self.c
        self.stats["sleeps"] += 1
        idx = self.H.idx()
        K, V = self.H.k[idx], self.H.v[idx]
        for _ in range(c["sleep_steps"]):
            self.W -= c["eta_sleep"] * self.grad(self.W, K, V)
        self.H.release(idx)

    def _release(self, idx, kind):
        if len(idx) == 0:
            return
        if self.A is not None:
            K = self.H.k[idx]
            self.A.add(K, K @ self.W.T)
        self.H.release(idx)
        self.stats[kind] += len(idx)

    def _sleep_select(self):
        c = self.c
        self.stats["sleeps"] += 1
        idx = self.H.idx()
        if len(idx) == 0:
            return
        dl = self.delta(idx)
        if c["route"]:
            self._release(idx[dl < c["tau_s"]], "released_consistent")
            cand_mask = (dl >= c["tau_s"]) & (self.H.r[idx] >= c["r_min"])
        else:
            cand_mask = dl >= c["tau_s"]
        cand, dcand = idx[cand_mask], dl[cand_mask]
        if len(cand) == 0:
            return
        if c["priority"] == "uniform":
            w = np.ones(len(cand))
        elif c["priority"] == "schema":
            w = np.log1p(self.H.r[cand]) * dcand + 1e-8
        else:
            w = dcand + 1e-8
        p = w / w.sum()
        n_pick = max(1, min(len(cand), c["batch"]))
        for _ in range(c["sleep_steps"]):
            pick = cand[self.rng.choice(len(cand), size=n_pick, replace=False, p=p)]
            KB, VB = self.H.k[pick], self.H.v[pick]
            if self.A is not None and c["reconsol"]:
                self.stats["anchors_invalidated"] += self.A.invalidate_conflicts(
                    KB, VB, c["cue_thresh"], c["agree_thresh"])
            if self.A is not None and c["interleave"] > 0:
                Ka, Va = self.A.sample(int(round(c["interleave"] * len(KB))))
                Kg, Vg = np.vstack([KB, Ka]), np.vstack([VB, Va])
            else:
                Kg, Vg = KB, VB
            g = self.grad(self.W, Kg, Vg)
            self.stats["steps_tried"] += 1
            if c.get("agem_only") and self.A is not None and len(self.A.k):     # P2b: projection, no look-ahead
                gA = self.grad(self.W, self.A.k, self.A.v)
                dot = float(np.sum(gA * g))
                if dot < 0:
                    g = g - dot / (float(np.sum(gA * gA)) + 1e-12) * gA
            if not c["gate"] or self.A is None or len(self.A.k) == 0:
                self.W -= c["eta_sleep"] * g
                self.stats["steps_accepted"] += 1
                continue
            gA = self.grad(self.W, self.A.k, self.A.v)
            if c["agem"]:
                dot = float(np.sum(gA * g))
                if dot < 0:
                    g = g - dot / (float(np.sum(gA * gA)) + 1e-12) * gA
            lB0, lA0 = self.loss(self.W, KB, VB), self.loss(self.W, self.A.k, self.A.v)
            eta = c["eta_sleep"]
            for attempt in range(c["backtracks"] + 1):
                W2 = self.W - eta * g
                G, D = self.gate_terms(W2, KB, VB, lB0, lA0)
                if G - c["rho"] * D > 0:
                    if c["trace"]:
                        self.trace.append((G, D, c["rho"]))
                    self.W = W2
                    self.stats["steps_accepted"] += 1
                    if attempt > 0:
                        self.stats["steps_shrunk"] += 1
                    break
                eta *= 0.5
        if c["release_consolidated"]:
            still = cand[self.H.used[cand]]
            if len(still):
                done = still[self.delta(still) < c["tau_s"]]
                self._release(done, "released_consolidated")

    # ----- reads -----
    def read(self, Q, cortex_only=False):
        out = Q @ self.W.T
        if self.H is not None and not cortex_only:
            vh, hit = self.H.lookup(Q, self.c["hit_thresh"])
            if vh is not None:
                out[hit] = vh[hit]
        return out

    def anchor_loss(self):
        return self.loss(self.W, self.A.k, self.A.v) if self.A is not None and len(self.A.k) else float("nan")
