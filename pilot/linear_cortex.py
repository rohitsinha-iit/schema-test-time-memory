"""Linear cortex for the pilot (proposal 4.1 to 4.3 with the Stage 0a rules, sim/memories.py), ported to torch and
batched over groups g = layer * Hkv + KV head. H, routing, reuse (kappa cue matches) and anchors' reservoir come
from pilot.schema_memory.SchemaMemory; only the cortex changes.

* C is W_g (d x d) on unit-norm pre-RoPE keys and unit-norm values; loss 0.5 * mean ||W k - v||^2; delta = ||W k - v||^2.
* Replay step (Stage 0a _sleep_select): batch = replayed landmarks plus interleave * |batch| anchors; gradient
  g = E^T K / n; A-GEM projection against the anchor gradient; gate G - rho D > 0 with damage_norm "sum", eta
  halved at most twice; per group.
* Arrival / surprise (Stage 0a observe): per-item delta rule W -= eta e k^T at eviction, eta scaled by ||e|| for
  surprise. sleep_all: `n_steps` full-batch gradient steps over H, then release all. uniform: replay uniformly
  sampled episodes, no gate, no anchors.
* Read (proposal 4.2): o_h += gamma_{l,h} * s_g * W_g (A_{l,h} q_h), with q_h the unit-norm pre-RoPE query of query
  head h, A_{l,h} the query map fitted by pilot/fit_qmap.py (teacher attention, no labels), s_g the running mean
  norm of written values, and gamma a trained scalar gate per layer and query head, initialised to 0
  (pilot/train_gates.py).
* Budget: W is d^2 = 64 pairs of 2d floats per KV head (cortex_pairs).
"""
import os

import numpy as np
import torch
import torch.nn.functional as F

from pilot.schema_memory import SchemaMemory


def qmap_path(model):
    return os.path.join(os.path.dirname(__file__), "data", f"qmap_{model.split('/')[-1]}.pt")


def gates_path(model, arm):
    return os.path.join(os.path.dirname(__file__), "data", f"gates_{model.split('/')[-1]}_{arm}.pt")


class LinearSchemaMemory(SchemaMemory):
    def __init__(self, eng, H_cap, A_cap, mode="schema", eta=0.3, interleave=0.5, cortex_pairs=64, arm=None,
                 gates=None, **kw):
        kw.pop("C_cap", None)
        super().__init__(eng, H_cap=H_cap, C_cap=0, A_cap=A_cap, mode=mode, eta=eta, **kw)
        d, G = eng.d, self.G
        self.W = torch.zeros(G, d, d, device=eng.dev)
        self.interleave = interleave
        self.vnorm = torch.zeros(G, device=eng.dev); self.vn_count = torch.zeros(G, device=eng.dev); self.vcount = 0
        self.qmap = torch.load(qmap_path(eng.model.config._name_or_path)).to(eng.dev)      # (L * Hq, d, d)
        self.gamma = torch.nn.Parameter(torch.zeros(eng.L, eng.Hq, device=eng.dev))
        if gates is not None:
            self.gamma.data.copy_(gates)
        elif arm is not None and os.path.exists(gates_path(eng.model.config._name_or_path, arm)):
            self.gamma.data.copy_(torch.load(gates_path(eng.model.config._name_or_path, arm)).to(eng.dev))

    # ---------- cortex ----------
    @staticmethod
    def _unit(x):
        return F.normalize(x, dim=-1)

    def _err(self, k, v, *a):
        e = torch.einsum("gde,gne->gnd", self.W, self._unit(k)) - self._unit(v)
        return (e ** 2).sum(-1)

    @staticmethod
    def _loss(W, K, V):
        e = torch.einsum("gde,gne->gnd", W, K) - V
        return 0.5 * (e ** 2).sum(-1).mean(-1)                                           # (G,)

    @staticmethod
    def _grad(W, K, V):
        e = torch.einsum("gde,gne->gnd", W, K) - V
        return torch.einsum("gnd,gne->gde", e, K) / K.shape[1]

    def _track_vnorm(self, v, groups=None):
        """Running mean norm of written values per group (v: (len(groups), n, d))."""
        g = torch.arange(self.G, device=v.device) if groups is None else torch.as_tensor(groups, device=v.device)
        self.vn_count[g] += 1
        self.vnorm[g] += (v.norm(dim=-1).mean(-1) - self.vnorm[g]) / self.vn_count[g]
        self.vcount += 1

    def consolidate(self, k, v, groups, eta, gate):
        """Replay step on items k, v (G, n, d) for all groups (Stage 0a _sleep_select inner loop)."""
        K, V = self._unit(k), self._unit(v)
        self._track_vnorm(v)
        G = self.W.shape[0]
        av = torch.as_tensor(self.Aval, device=K.device)
        use_A = self.A_cap > 0 and bool(av.any())
        Kg, Vg = K, V
        if use_A and self.interleave > 0:
            ia = torch.nonzero(av)[:, 0]
            pick = ia[torch.randint(len(ia), (max(1, int(round(self.interleave * K.shape[1]))),), device=K.device)]
            Kg, Vg = torch.cat([K, self._unit(self.Ak[:, pick])], 1), torch.cat([V, self._unit(self.Av[:, pick])], 1)
        g = self._grad(self.W, Kg, Vg)
        if not gate or not use_A:
            self.W -= eta * g
            return torch.ones(G, dtype=torch.bool, device=K.device)
        KA, VA = self._unit(self.Ak[:, av]), self._unit(self.Av[:, av])
        gA = self._grad(self.W, KA, VA)
        dot = (gA * g).sum((1, 2))
        g = torch.where((dot < 0)[:, None, None], g - (dot / ((gA * gA).sum((1, 2)) + 1e-12))[:, None, None] * gA, g)
        lB0, lA0 = self._loss(self.W, K, V), self._loss(self.W, KA, VA)
        acc = torch.zeros(G, dtype=torch.bool, device=K.device)
        e = eta
        for attempt in range(3):
            W2 = self.W - e * g
            gain = (lB0 - self._loss(W2, K, V)) * K.shape[1]
            dmg = (self._loss(W2, KA, VA) - lA0) * KA.shape[1]
            ok = ~acc & (gain - self.rho * dmg > 0)
            self.W[ok] = W2[ok]
            acc |= ok
            self.st["steps_shrunk"] += int(attempt > 0 and ok.any())
            e *= 0.5
        self.st["steps_rejected"] += int((~acc).sum())
        return acc

    def _arrival(self, k, v, groups):
        K, V = self._unit(k), self._unit(v)
        self._track_vnorm(v, groups)
        g = torch.as_tensor(groups, device=K.device)
        W = self.W[g]
        for i in range(K.shape[1]):
            e = torch.einsum("gde,ge->gd", W, K[:, i]) - V[:, i]
            eta = self.eta * (e.norm(dim=-1, keepdim=True)[..., None] if self.mode == "surprise" else 1.0)
            W = W - eta * e[:, :, None] * K[:, i][:, None, :]
        self.W[g] = W

    def _add_anchors(self, k, groups_mask, item_idx):
        if self.A_cap == 0:
            return
        K = self._unit(k)
        vhat = torch.einsum("gde,gne->gnd", self.W, K)
        for i in item_idx:
            self.A_seen += 1
            j = self.A_seen - 1 if self.A_seen <= self.A_cap else int(self.rng.integers(self.A_seen))
            if j < self.A_cap:
                self.Ak[groups_mask, j], self.Av[groups_mask, j] = K[groups_mask, i], vhat[groups_mask, i]
                self.Aval[j] = True

    def write(self, l, k_pre, v, pos, nll):
        if self.mode == "ls_oracle":                       # diagnostic: W = ridge least squares on every landmark so far
            sel = self.hsel.select(pos, nll)
            if len(sel):
                idx = torch.as_tensor(np.searchsorted(pos, sel), device=k_pre.device)
                g = list(range(l * self.Hkv, (l + 1) * self.Hkv))
                K, V = self._unit(k_pre[:, idx].float()), self._unit(v[:, idx].float())
                self._track_vnorm(v[:, idx].float(), g)
                gi = torch.as_tensor(g, device=K.device)
                if not hasattr(self, "XtX"):
                    self.XtX = torch.zeros_like(self.W); self.XtY = torch.zeros_like(self.W)
                self.XtX[gi] += torch.einsum("gnd,gne->gde", K, K)
                self.XtY[gi] += torch.einsum("gnd,gne->gde", K, V)
                eye = torch.eye(self.W.shape[-1], device=K.device)
                self.W[gi] = torch.linalg.solve(self.XtX[gi] + 1e-2 * eye, self.XtY[gi]).transpose(1, 2)
            return
        if self.mode in ("arrival", "surprise"):
            sel = self.hsel.select(pos, nll)
            if len(sel):
                idx = torch.as_tensor(np.searchsorted(pos, sel), device=k_pre.device)
                self._arrival(k_pre[:, idx].float(), v[:, idx].float(), list(range(l * self.Hkv, (l + 1) * self.Hkv)))
                if l == 0:
                    self.st["consolidated_items"] += len(sel)
            return
        super().write(l, k_pre, v, pos, nll)

    def sleep(self):
        if self.mode == "sleep_all":
            self.st["sleeps"] += 1
            it = torch.as_tensor(np.flatnonzero(self.Hval), device=self.Hk.device)
            K, V = self._unit(self.Hk[:, it]), self._unit(self.Hv[:, it])
            self._track_vnorm(self.Hv[:, it])
            for _ in range(self.n_steps):
                self.W -= self.eta * self._grad(self.W, K, V)
            self.st["consolidated_items"] += len(it)
            self._release(np.unique(self.Hep[self.Hval]), anchors=False)
            return
        super().sleep()

    def cortex_out(self, l, q, pos):
        """gamma * s * W (A q) for every query head, (B, Hq, T, d)."""
        if self.vcount == 0:
            return torch.zeros_like(q, dtype=torch.float32)
        B, Hq, T, d = q.shape
        qp = F.normalize(self.eng.rotate(q.float(), pos, inverse=True), dim=-1)
        A = self.qmap[l * Hq:(l + 1) * Hq]                                               # (Hq, d, d): k ~ q A
        kq = F.normalize(torch.einsum("bhtd,hde->bhte", qp, A), dim=-1)
        g = slice(l * self.Hkv, (l + 1) * self.Hkv)
        W = self.W[g].repeat_interleave(Hq // self.Hkv, 0)                              # (Hq, d, d)
        s = self.vnorm[g].repeat_interleave(Hq // self.Hkv, 0)
        out = torch.einsum("hde,bhte->bhtd", W, kq)
        return self.gamma[l][None, :, None, None] * s[None, :, None, None] * out


CORTEX_PAIRS, A_CAP, KAPPA_PAIRS = 64, 256, 16
EXTRA = 2048
LIN_SPECS = {
    "lin_schema": (0, dict(H_cap=EXTRA - CORTEX_PAIRS - A_CAP - KAPPA_PAIRS, A_cap=A_CAP, reuse="cue",
                           aux_pairs=KAPPA_PAIRS, cortex_pairs=CORTEX_PAIRS)),
    "lin_schema_nogate": (0, dict(H_cap=EXTRA - CORTEX_PAIRS - A_CAP - KAPPA_PAIRS, A_cap=A_CAP, reuse="cue",
                                  aux_pairs=KAPPA_PAIRS, cortex_pairs=CORTEX_PAIRS, gate=False)),
    "lin_schema_surprise_pri": (0, dict(H_cap=EXTRA - CORTEX_PAIRS - A_CAP - KAPPA_PAIRS, A_cap=A_CAP, reuse="cue",
                                        aux_pairs=KAPPA_PAIRS, cortex_pairs=CORTEX_PAIRS, priority="delta", r_min=0.0)),
    "lin_uniform": (0, dict(H_cap=EXTRA - CORTEX_PAIRS, A_cap=0, mode="uniform", cortex_pairs=CORTEX_PAIRS)),
    "lin_sleep_all": (0, dict(H_cap=EXTRA - CORTEX_PAIRS, A_cap=0, mode="sleep_all", cortex_pairs=CORTEX_PAIRS)),
    # diagnostic only (not budget-matched): the best a linear cortex could store, to test the read path
    "diag_ls_oracle": (0, dict(H_cap=0, A_cap=0, mode="ls_oracle", cortex_pairs=CORTEX_PAIRS)),
    "lin_arrival": (EXTRA - CORTEX_PAIRS, dict(H_cap=0, A_cap=0, mode="arrival", cortex_pairs=CORTEX_PAIRS)),
    "lin_surprise": (EXTRA - CORTEX_PAIRS, dict(H_cap=0, A_cap=0, mode="surprise", cortex_pairs=CORTEX_PAIRS)),
}


def lin_pairs(kw):
    return kw["H_cap"] + kw["A_cap"] + kw.get("aux_pairs", 0) + kw["cortex_pairs"]
