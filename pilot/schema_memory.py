"""SCHEMA memory for the training-free pilot: hippocampus H, cortex C, anchors A, micro-sleep (proposal 4.1 to 4.4).

Per layer and KV head (group g = layer * Hkv + head), in pre-RoPE key space:
* H: landmarks of evicted chunks (same self-distilled selector as pilot.memory.Hippocampus), one episode per
  evicted chunk, reuse tag r = attention mass the landmark receives when read. FIFO eviction (M1 decision).
* C: C_cap slots (key, value), changed only by consolidation. C(k) = value of the nearest slot by key cosine if
  the cosine is at least theta, else 0. Consolidating an item (k, v): if its nearest slot matches (cos >= theta)
  the slot moves toward it by eta (delta rule on that slot); otherwise the item takes a free slot, or the least
  used slot (moved toward it by eta). Read in the model like H: top n_read slots join the attention softmax.
  This is a competitive (slot) nonlinear fast-weight memory; a linear cortex cannot be read without trained gates.
* A: reservoir of (k, C(k) at consolidation time); damage D = sum over anchors of the increase in
  ||C(k_a) - v_a||^2 / ||v_a||^2 (damage_norm "sum", M1 decision), gain G = the same sum of the decrease over the
  replay batch. Accept a step if G - rho D > 0, else halve eta, at most twice, else defer (per group).
* Micro-sleep every `sleep_every` chunks while H occupancy exceeds occ * H_cap. Routing per episode (delta and r
  averaged over groups): delta < tau released (schema-consistent, anchors added); delta >= tau and r >= r_min
  replayed by priority r * delta ("delta" for the surprise-priority ablation); the rest stays. After replay,
  episodes whose delta fell below tau are released.

Modes: "schema" (above), "uniform" (uniformly sampled episodes, no routing, no gate, no anchors, no release),
"sleep_all" (replay every episode in H, then release all; no gate, no anchors), "arrival" (no H: every landmark is
consolidated at eviction, no gate), "surprise" (as arrival, eta scaled by the item's error delta).
A-GEM projection is not used: the slot update is not a gradient step.
"""
import numpy as np
import torch
import torch.nn.functional as F

from pilot.memory import Hippocampus


class SchemaMemory:
    def __init__(self, eng, H_cap, C_cap, A_cap, mode="schema", gate=True, priority="rdelta", rho=4.0, eta=0.5,
                 theta=0.8, tau=0.5, r_min=0.0, sleep_every=4, occ=0.75, n_ep=4, n_steps=2, m=16, n_read=128, seed=0,
                 reuse="mass", cue=0.8, aux_pairs=0):
        self.eng, self.mode, self.gate, self.priority = eng, mode, gate, priority
        self.rho, self.eta, self.theta, self.tau, self.r_min = rho, eta, theta, tau, r_min
        self.sleep_every, self.occ, self.n_ep, self.n_steps, self.n_read = sleep_every, occ, n_ep, n_steps, n_read
        self.hsel = Hippocampus(eng, cap=0, m=m)                       # landmark selector only
        self.feat_layer = self.hsel.feat_layer
        self.Hkv, d, dev = eng.Hkv, eng.d, eng.dev
        G = self.G = eng.L * eng.Hkv
        z = lambda *s: torch.zeros(*s, device=dev)
        self.H_cap, self.C_cap, self.A_cap = H_cap, C_cap, A_cap
        self.Hk, self.Hv, self.Hr = z(G, H_cap, d), z(G, H_cap, d), z(G, H_cap)
        self.Hval, self.Hep, self.Hpos = np.zeros(H_cap, bool), np.full(H_cap, -1), np.full(H_cap, -1)
        self.Ck, self.Cv, self.Cuse = z(G, C_cap, d), z(G, C_cap, d), z(G, C_cap)
        self.Cval = torch.zeros(G, C_cap, dtype=torch.bool, device=dev)
        self.Ak, self.Av = z(G, max(A_cap, 1), d), z(G, max(A_cap, 1), d)
        self.Aval, self.A_seen = np.zeros(max(A_cap, 1), bool), 0
        self.rng = np.random.default_rng(seed)
        self.n_episode, self.chunks, self.retest = 0, 0, False
        self.slots, self.last = None, [None] * eng.L
        self.st = dict(sleeps=0, steps_tried=0, steps_accepted=0, steps_shrunk=0, steps_rejected=0, released=0,
                       consolidated_items=0, forced_evictions=0, h_writes=0, cue_matches=0)
        # reuse "mass": attention mass a landmark receives when read (proposal 4.2);
        # reuse "cue": number of later landmarks whose fact-level key kappa (pilot/fit_kappa.py) has cosine >= cue
        # with it, counted when they are written (Stage 0a cue match on eviction). kappa costs aux_pairs of budget.
        self.reuse, self.cue = reuse, cue
        if reuse == "cue":
            from pilot.fit_kappa import kappa_path
            kp = torch.load(kappa_path(eng.model.config._name_or_path))
            self.kW = kp["W"].to(dev).float()
            self.Hkap, self.Hcue = z(H_cap, self.kW.shape[0]), z(H_cap)

    # ---------- cortex function and consolidation ----------
    def _err(self, k, v, Ck=None, Cv=None, Cval=None):
        """||C(k) - v||^2 / ||v||^2 per item; k, v: (G, n, d)."""
        Ck = self.Ck if Ck is None else Ck; Cv = self.Cv if Cv is None else Cv; Cval = self.Cval if Cval is None else Cval
        cos = torch.einsum("gnd,gcd->gnc", F.normalize(k, dim=-1), F.normalize(Ck, dim=-1))
        cos = cos.masked_fill(~Cval[:, None, :], -2.0)
        best, s = cos.max(-1)
        out = torch.gather(Cv, 1, s[..., None].expand(-1, -1, Cv.shape[-1])) * (best >= self.theta)[..., None]
        return ((out - v) ** 2).sum(-1) / (v ** 2).sum(-1).clamp_min(1e-6)

    def _candidate(self, k, v, eta, groups):
        """Candidate (Ck, Cv, Cval) after consolidating items k, v (Gs, b, d) into the given groups."""
        Ck, Cv, Cval, use = self.Ck[groups].clone(), self.Cv[groups].clone(), self.Cval[groups].clone(), self.Cuse[groups]
        cos = torch.einsum("gnd,gcd->gnc", F.normalize(k, dim=-1), F.normalize(Ck, dim=-1)).masked_fill(~Cval[:, None], -2.0)
        best, s = cos.max(-1)
        eta = eta if torch.is_tensor(eta) else torch.full(best.shape, float(eta), device=k.device)
        for gi in range(len(groups)):
            unmatched = torch.nonzero(best[gi] < self.theta)[:, 0]
            if len(unmatched):
                free = torch.nonzero(~Cval[gi])[:, 0]
                order = torch.cat([free, torch.argsort(use[gi].masked_fill(~Cval[gi], -1.0))[: max(0, len(unmatched) - len(free))]])
                s[gi, unmatched] = order[: len(unmatched)]
        g_idx = torch.arange(len(groups), device=k.device)[:, None].expand_as(s)
        empty = ~Cval[g_idx, s]
        e = torch.where(empty, torch.ones_like(eta), eta)[..., None]
        cnt = torch.zeros(Ck.shape[:2], device=k.device).index_put_((g_idx, s), torch.ones_like(best), accumulate=True)
        dk = torch.zeros_like(Ck).index_put_((g_idx, s), e * (k - Ck[g_idx, s]), accumulate=True)
        dv = torch.zeros_like(Cv).index_put_((g_idx, s), e * (v - Cv[g_idx, s]), accumulate=True)
        c = cnt.clamp_min(1)[..., None]
        Ck, Cv = Ck + dk / c, Cv + dv / c
        Cval[g_idx, s] = True
        return Ck, Cv, Cval

    def consolidate(self, k, v, groups, eta, gate):
        """Consolidate items into C for `groups`; with the gate, accept per group if G - rho D > 0, halving eta at
        most twice. Returns the accepted mask (len(groups),)."""
        groups = torch.as_tensor(groups, device=k.device)
        acc = torch.zeros(len(groups), dtype=torch.bool, device=k.device)
        if not gate:
            self.Ck[groups], self.Cv[groups], self.Cval[groups] = self._candidate(k, v, eta, groups)
            return ~acc
        e0 = self._err(k, v, self.Ck[groups], self.Cv[groups], self.Cval[groups])
        av = torch.as_tensor(self.Aval, device=k.device)
        Ak, Av = self.Ak[groups][:, av], self.Av[groups][:, av]
        a0 = self._err(Ak, Av, self.Ck[groups], self.Cv[groups], self.Cval[groups]) if av.any() else None
        for shrink in range(3):
            todo = ~acc
            if not todo.any():
                break
            Ck, Cv, Cval = self._candidate(k, v, eta / 2 ** shrink, groups)
            gain = (e0 - self._err(k, v, Ck, Cv, Cval)).sum(-1)
            dmg = (self._err(Ak, Av, Ck, Cv, Cval) - a0).sum(-1) if a0 is not None else torch.zeros_like(gain)
            ok = todo & (gain - self.rho * dmg > 0)
            sel = groups[ok]
            self.Ck[sel], self.Cv[sel], self.Cval[sel] = Ck[ok], Cv[ok], Cval[ok]
            acc |= ok
            self.st["steps_shrunk"] += int(shrink > 0 and ok.any())
        self.st["steps_rejected"] += int((~acc).sum())
        return acc

    def _add_anchors(self, k, groups_mask, item_idx):
        """Reservoir-sample items into A; v_a = C(k) now. k: (G, b, d) for all groups."""
        if self.A_cap == 0:
            return
        cos = torch.einsum("gnd,gcd->gnc", F.normalize(k, dim=-1), F.normalize(self.Ck, dim=-1)).masked_fill(~self.Cval[:, None], -2.0)
        best, s = cos.max(-1)
        vhat = torch.gather(self.Cv, 1, s[..., None].expand(-1, -1, k.shape[-1])) * (best >= self.theta)[..., None]
        for i in item_idx:
            self.A_seen += 1
            j = self.A_seen - 1 if self.A_seen <= self.A_cap else int(self.rng.integers(self.A_seen))
            if j < self.A_cap:
                gm = groups_mask
                self.Ak[gm, j], self.Av[gm, j] = k[gm, i], vhat[gm, i]
                self.Aval[j] = True

    # ---------- engine interface ----------
    def write(self, l, k_pre, v, pos, nll):
        sel = self.hsel.select(pos, nll)
        if len(sel) == 0:
            return
        idx = torch.as_tensor(np.searchsorted(pos, sel), device=k_pre.device)
        k, vv = k_pre[:, idx].float(), v[:, idx].float()                        # (Hkv, b, d)
        groups = list(range(l * self.Hkv, (l + 1) * self.Hkv))
        if self.mode in ("arrival", "surprise"):
            eta = self.eta
            if self.mode == "surprise":
                eta = (self.eta * self._err(k, vv, self.Ck[groups], self.Cv[groups], self.Cval[groups]).clamp(max=1.0))
            self.consolidate(k, vv, groups, eta, gate=False)
            if l == 0:
                self.st["consolidated_items"] += len(sel)
            return
        if l == 0:
            free = np.flatnonzero(~self.Hval)
            while len(free) < len(sel):                                         # FIFO: drop the oldest episode
                old = self.Hep[self.Hval].min()
                drop = self.Hval & (self.Hep == old)
                self.Hval[drop] = False
                self.st["forced_evictions"] += 1
                free = np.flatnonzero(~self.Hval)
            self.slots = free[: len(sel)]
            if self.reuse == "cue":
                from pilot.engine import _CTX
                f = _CTX["st"]["feat"][torch.as_tensor(sel, device=k.device)].float()
                kap = F.normalize(((f - self.hsel.sel["mu"]) / self.hsel.sel["sd"]) @ self.kW.T, dim=-1)
                old = torch.as_tensor(self.Hval, device=k.device)
                if old.any():
                    hit = (kap @ self.Hkap.T >= self.cue) & old[None]                  # (new, H_cap)
                    self.Hcue += hit.sum(0).float()
                    self.st["cue_matches"] += int(hit.sum())
                    self._on_cue(sel, hit)
                sl0 = torch.as_tensor(self.slots, device=k.device)
                self.Hkap[sl0], self.Hcue[sl0] = kap, 0.0
            self.Hval[self.slots], self.Hep[self.slots], self.Hpos[self.slots] = True, self.n_episode, sel
            self.n_episode += 1; self.st["h_writes"] += len(sel)
        sl = torch.as_tensor(self.slots, device=k.device)
        g = torch.as_tensor(groups, device=k.device)[:, None]
        self.Hk[g, sl], self.Hv[g, sl], self.Hr[g, sl] = k, vv, 0.0

    def _on_cue(self, sel, hit):
        """Hook: new landmarks at stream positions sel re-cued the old H slots in hit (new, H_cap). Default: nothing."""

    def _store_read(self, keys, vals, valid, q, p_mem):
        """Top n_read entries of one store for each batch row; keys (Hkv, n, d), valid (Hkv, n) bool."""
        B, Hq, T, d = q.shape
        n = keys.shape[1]
        k_rot = self.eng.rotate(keys, np.full(n, p_mem))
        s = torch.einsum("bhtd,hnd->bhtn", q.float(), k_rot.repeat_interleave(Hq // self.Hkv, 0))
        s = s.amax(2).view(B, self.Hkv, Hq // self.Hkv, n).amax(2).masked_fill(~valid[None], float("-inf"))
        sc, top = s.topk(min(self.n_read, n), -1)
        gk = torch.gather(k_rot[None].expand(B, -1, -1, -1), 2, top[..., None].expand(-1, -1, -1, d))
        gv = torch.gather(vals[None].expand(B, -1, -1, -1), 2, top[..., None].expand(-1, -1, -1, d))
        return gk.to(q.dtype), gv.to(q.dtype), torch.isfinite(sc), top

    def read(self, l, q, p_mem, probe=False):
        g = slice(l * self.Hkv, (l + 1) * self.Hkv)
        parts, self.last[l] = [], []
        hval = torch.as_tensor(self.Hval, device=q.device)
        if self.H_cap and not self.retest and self.Hval.any():
            parts.append(self._store_read(self.Hk[g], self.Hv[g], hval[None].expand(self.Hkv, -1), q, p_mem))
            self.last[l].append("H")
        if self.Cval[g].any():
            parts.append(self._store_read(self.Ck[g], self.Cv[g], self.Cval[g], q, p_mem))
            self.last[l].append("C")
        if not parts:
            return None
        self.last_top = getattr(self, "last_top", {}); self.last_top[l] = [p[3] for p in parts]
        return tuple(torch.cat([p[i] for p in parts], 2) for i in range(3))

    def feedback(self, l, mass):
        g = slice(l * self.Hkv, (l + 1) * self.Hkv)
        o = 0
        for name, top in zip(self.last[l], self.last_top[l]):
            n = top.shape[-1]
            tgt = self.Hr if name == "H" else self.Cuse
            tgt[g].scatter_add_(1, top[0], mass[0, :, o:o + n].float())
            o += n

    def end_chunk(self, pos):
        self.chunks += 1
        if self.mode in ("arrival", "surprise") or self.chunks % self.sleep_every:
            return
        if self.Hval.sum() > self.occ * self.H_cap:
            self.sleep()

    # ---------- micro-sleep ----------
    def _episode_stats(self):
        idx = np.flatnonzero(self.Hval)
        it = torch.as_tensor(idx, device=self.Hk.device)
        delta = self._err(self.Hk[:, it], self.Hv[:, it]).mean(0).cpu().numpy()            # mean over groups
        r = (self.Hcue[it] if self.reuse == "cue" else self.Hr[:, it].mean(0)).cpu().numpy()
        eps = self.Hep[idx]
        u = np.unique(eps)
        d_ep = np.array([delta[eps == e].mean() for e in u]); r_ep = np.array([r[eps == e].sum() for e in u])
        return idx, eps, u, d_ep, r_ep

    def _replay(self, ep_ids, gate, anchors):
        lm = np.flatnonzero(self.Hval & np.isin(self.Hep, ep_ids))
        it = torch.as_tensor(lm, device=self.Hk.device)
        acc = self.consolidate(self.Hk[:, it], self.Hv[:, it], list(range(self.G)), self.eta, gate)
        self.st["steps_tried"] += 1; self.st["steps_accepted"] += int(acc.any())
        self.st["consolidated_items"] += len(lm)
        if anchors and acc.any():
            self._add_anchors(self.Hk[:, it], acc, range(len(lm)))

    def _release(self, ep_ids, anchors):
        lm = np.flatnonzero(self.Hval & np.isin(self.Hep, ep_ids))
        if anchors and len(lm):
            it = torch.as_tensor(lm, device=self.Hk.device)
            self._add_anchors(self.Hk[:, it], torch.ones(self.G, dtype=torch.bool, device=self.Hk.device), range(len(lm)))
        self.Hval[lm] = False
        self.st["released"] += len(np.unique(self.Hep[lm]))

    def sleep(self):
        self.st["sleeps"] += 1
        idx, eps, u, d_ep, r_ep = self._episode_stats()
        self.r_hist = getattr(self, "r_hist", []) + [r_ep]
        self.d_hist = getattr(self, "d_hist", []) + [d_ep]
        if self.mode == "uniform":
            for _ in range(self.n_steps):
                self._replay(self.rng.choice(u, min(self.n_ep, len(u)), replace=False), gate=False, anchors=False)
            return
        if self.mode == "sleep_all":
            for _ in range(self.n_steps):
                for i in range(0, len(u), self.n_ep):
                    self._replay(u[i:i + self.n_ep], gate=False, anchors=False)
            self._release(u, anchors=False)
            return
        self._release(u[d_ep < self.tau], anchors=True)                         # schema-consistent
        cand = (d_ep >= self.tau) & (r_ep >= self.r_min)
        pri = d_ep * (r_ep if self.priority == "rdelta" else 1.0)
        order = [i for i in np.argsort(-pri) if cand[i]]
        replayed = []
        for step in range(self.n_steps):
            batch = order[step * self.n_ep:(step + 1) * self.n_ep]
            if not batch:
                break
            self._replay(u[batch], gate=self.gate, anchors=True)
            replayed += list(u[batch])
        if replayed:
            idx, eps, u2, d2, _ = self._episode_stats()
            self._release(u2[np.isin(u2, replayed) & (d2 < self.tau)], anchors=False)

    def retest_mode(self):
        self.retest = True

    def stats(self):
        q = {}
        if getattr(self, "r_hist", None):
            r, d = np.concatenate(self.r_hist), np.concatenate(self.d_hist)
            q = {f"r_ep_q{int(p*100)}": float(np.quantile(r, p)) for p in (0.25, 0.5, 0.75, 0.9)}
            q.update({f"delta_ep_q{int(p*100)}": float(np.quantile(d, p)) for p in (0.1, 0.5, 0.9)})
        return dict(H_landmarks=int(self.Hval.sum()), C_slots_used=float(self.Cval.float().sum(1).mean()),
                    anchors=int(self.Aval.sum()), **self.st, **q)


H_CAP, C_CAP, A_CAP = 1024, 768, 256         # schema family: 1024 + 768 + 256 = EXTRA pairs
EXTRA = H_CAP + C_CAP + A_CAP

# arm: (window pairs beyond W, SchemaMemory kwargs); every arm holds W + EXTRA pairs per layer per KV head
SPECS = {
    "schema": (0, dict(H_cap=H_CAP, C_cap=C_CAP, A_cap=A_CAP)),
    "schema_nogate": (0, dict(H_cap=H_CAP, C_cap=C_CAP, A_cap=A_CAP, gate=False)),
    "schema_surprise_pri": (0, dict(H_cap=H_CAP, C_cap=C_CAP, A_cap=A_CAP, priority="delta", r_min=0.0)),
    "uniform_replay": (0, dict(H_cap=H_CAP + A_CAP, C_cap=C_CAP, A_cap=0, mode="uniform")),
    "sleep_all": (0, dict(H_cap=H_CAP + A_CAP, C_cap=C_CAP, A_cap=0, mode="sleep_all")),
    "schema_kappa": (0, dict(H_cap=H_CAP - 16, C_cap=C_CAP, A_cap=A_CAP, reuse="cue", aux_pairs=16)),
    "schema_kappa_nogate": (0, dict(H_cap=H_CAP - 16, C_cap=C_CAP, A_cap=A_CAP, reuse="cue", aux_pairs=16, gate=False)),
    "arrival": (EXTRA - C_CAP, dict(H_cap=0, C_cap=C_CAP, A_cap=0, mode="arrival")),
    "surprise": (EXTRA - C_CAP, dict(H_cap=0, C_cap=C_CAP, A_cap=0, mode="surprise")),
}


def mem_pairs(kw):
    """kappa keys (reuse "cue") are shared by all layers: H_cap x 128 floats / (28 layers x 2 KV heads) is about 9
    pairs of 2 x 128 floats per KV head; counted as aux_pairs = 16."""
    return kw["H_cap"] + kw["C_cap"] + kw["A_cap"] + kw.get("aux_pairs", 0)
