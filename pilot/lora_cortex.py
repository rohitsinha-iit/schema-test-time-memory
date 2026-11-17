"""Weight-space cortex: LoRA fast weights on the frozen model, changed only by replay (proposal 4.1 and 4.3 with a
TTT-style cortex), read by the model itself on every token.

* LoRA rank r on every layer's MLP (up, gate, down) and attention o_proj; reset to zero at the start of each stream.
* Episode = one batch of tokens leaving the window, kept in H as landmark (k, v) pairs (attention-insertion read,
  as pilot.schema_memory) plus its token ids (the evicted chunk and `ctx` tokens before it) for replay.
* Cortex error delta_j = the model's NLL on the episode's landmark tokens with no context but the episode itself
  (does the cortex already predict it?): `delta="max"` (default since 4 Nov 2026) uses the hardest landmark token,
  `"mean"` their mean. With the mean, episodes were released once their sentence was memorized while the fact could
  not yet be recalled (validation: 80 of about 112 released, retest exceptions 0.012, H emptied). Reuse r_j = kappa
  cue matches (pilot.fit_kappa).
* Replay step on episodes B: Adam step on mean next-token CE over their tokens + lam * KL(base || LoRA) on a fixed
  plain-text anchor segment (interleaving with old knowledge; every arm that writes the cortex uses it).
* Gate (schema): G = sum over B of the drop in landmark NLL, D = sum over anchor episodes (reservoir of consolidated
  episodes) of the rise in landmark NLL; accept if G - rho D > 0, else restore and halve the step, at most twice.
* Arms: arrival (steps on every evicted episode, TTT / dynamic-evaluation-like), surprise (the same, step
  scaled by the episode's error relative to its running mean, Titans-like), uniform (H + uniform replay),
  sleep_all (sweep all of H, then release it), schema and schema_nogate (routing, priority log1p(r) * delta).
* Equal replay compute (fixed on validation, 4 Nov 2026, after schema took 42 steps per stream against 88 for
  uniform and 144 for sleep_all): every replay arm sleeps every `sleep_every` chunks once H holds episodes and
  takes exactly `n_steps` steps per sleep; arrival and surprise take n_steps / sleep_every steps per chunk.
  Schema samples each step's episodes with probability proportional to log1p(r) * delta among routed candidates,
  fresh for every step, as Stage 0a (sim/memories.py _sleep_select).
"""
import copy

import numpy as np
import torch
import torch.nn.functional as F

from pilot.lora_linear import LoRALinear
from pilot.schema_memory import SchemaMemory


def install_lora(eng, r):
    """Wrap the model's MLP and o_proj linears once; returns the LoRA modules."""
    if getattr(eng, "lora_mods", None) is None:
        mods = []
        for layer in eng.model.model.layers:
            for parent, name in ((layer.mlp, "up_proj"), (layer.mlp, "gate_proj"), (layer.mlp, "down_proj"),
                                 (layer.self_attn, "o_proj")):
                lo = LoRALinear(getattr(parent, name), r)
                lo.enabled = False
                setattr(parent, name, lo)
                mods.append(lo)
        eng.lora_mods = mods
        eng.lora_init = [m.A.detach().clone() for m in mods]
    return eng.lora_mods


def lora_enable(eng, on):
    for m in getattr(eng, "lora_mods", None) or []:
        m.enabled = on


def hrd_targets(t_logp, b_logp, nxt, n_pos=4, min_ctx=8, topk=64, g_min=1.0):
    """Hippocampal recall distillation targets for one recall query.

    t_logp: (L, V) log-probs of the frozen base reading [episode; query], at the query's positions (the teacher: the
    model with the hippocampal episode in context). b_logp: (L, V) the frozen base on the query alone. nxt: (L - 1,)
    the query's own next tokens (what the stream actually said).
    A position is a confirmed recall when the episode raised the log-prob of the token that actually came next:
    gain = log p_teacher(next) - log p_base(next). Keep the n_pos positions with the largest gain >= g_min among
    positions with at least min_ctx query tokens of context (none: the episode did not help here, pos None). Generic
    context effects and recalls the stream contradicts (a drifted value, another person's fact) are not targets.
    Targets are the teacher's top-k log-probs at the kept positions, renormalized.
    (First version, 4 Nov 2026: positions ranked by KL(teacher || base) with no check against the next token; the
    diagnostic pilot/diag_hrd.py showed generic context effects and name completions as targets, retest recall 0.)
    """
    L = t_logp.shape[0]
    if L <= min_ctx + 1:
        return dict(pos=None, c0=0.0)
    i = torch.arange(L - 1, device=t_logp.device)
    gain = t_logp[i, nxt] - b_logp[i, nxt]
    gain[:min_ctx] = float("-inf")
    c, pos = gain.topk(min(n_pos, L - 1 - min_ctx))
    if float(c[0]) < g_min:
        return dict(pos=None, c0=float(c[0]))
    pos = pos[c >= g_min]
    tl, ti = t_logp[pos].topk(min(topk, t_logp.shape[-1]), -1)
    return dict(pos=pos, t_ids=ti, t_logp=tl - tl.logsumexp(-1, keepdim=True), c0=float(c[0]))


def kl_topk(t_ids, t_logp, s_logp):
    """KL(teacher || student) on the teacher's top-k support, per position; s_logp (P, V) student log-probs.
    Zero iff the student matches the teacher on the support with all its mass there."""
    return (t_logp.exp() * (t_logp - s_logp.gather(-1, t_ids))).sum(-1)


def lora_pairs(eng_cfg, r):
    """LoRA floats per layer / KV heads / (2 d) = pairs per KV head."""
    h, f, hq, d, hkv = eng_cfg
    per_layer = r * ((h + f) * 3 + (hq * d + h))
    return per_layer / hkv / (2 * d)


class LoRAMemory(SchemaMemory):
    def __init__(self, eng, H_cap, mode="schema", rank=8, lr=1e-3, lam=1.0, ctx=32, R=32, n_text=512, delta="max",
                 target="lm", n_q=4, q_pre=32, q_post=16, n_pos=16, min_ctx=8, topk=64, g_min=1.0,
                 w_min=0.3, recall_from=None, **kw):
        for k in ("C_cap", "A_cap", "lora_pairs", "anchor_pairs"):
            kw.pop(k, None)
        super().__init__(eng, H_cap=H_cap, C_cap=0, A_cap=0, mode=mode, **kw)
        if self.reuse == "recall":                                              # confirmed-recall counts per H slot
            self.Hcue = torch.zeros(H_cap, device=eng.dev)
        self.w_min, self.recall_from = w_min, eng.L // 3 if recall_from is None else recall_from
        self._rec_pos, self._rec_ep = set(), set()
        # target "hrd": hippocampal recall distillation (results/schema_ideas_20261104.md, candidate A)
        self.target, self.n_q, self.q_pre, self.q_post = target, n_q, q_pre, q_post
        self.n_pos, self.min_ctx, self.topk, self.g_min = n_pos, min_ctx, topk, g_min
        self.ep_q = {}                                                          # episode -> recall queries
        mods = install_lora(eng, rank)
        for m, a0 in zip(mods, eng.lora_init):                                  # fresh cortex per stream
            m.A.data.copy_(a0); m.B.data.zero_()
        lora_enable(eng, True)
        self.params = [p for m in mods for p in (m.A, m.B)]
        self.opt = torch.optim.Adam(self.params, lr=lr)
        self.lr, self.lam, self.ctx, self.R, self.delta_kind = lr, lam, ctx, R, delta
        self.ep_tok, self.ep_lm = {}, {}                                        # episode id -> token ids / landmark offsets
        self.anchors, self.A_seen, self.queue = [], 0, []
        self.err_mean, self.err_n = 0.0, 0
        from pilot.text_stream import load_filler
        filler = load_filler(eng.tok)
        self.text = torch.as_tensor(filler[-6144:-6144 + n_text], device=eng.dev, dtype=torch.long)[None]
        with torch.no_grad(), self._plain():
            lora_enable(eng, False)
            self.text_lp = torch.log_softmax(eng.model(self.text).logits[0].float(), -1)
            lora_enable(eng, True)
        self.st.update(lora_steps=0, lora_shrunk=0, lora_rejected=0)
        if target == "hrd":
            self.st.update(hrd_queries=0, hrd_uninformative=0)

    # ---------- plain forwards (full causal attention, no memory) for replay ----------
    def _plain(self):
        import contextlib
        import pilot.engine as E
        if E.FULL_ATTENTION is None:
            from pilot.ahn_train import full_attention
            E.FULL_ATTENTION = full_attention

        @contextlib.contextmanager
        def cm():
            saved = E._CTX.get("st")
            E._CTX["st"] = dict(mode="teacher", h=[None] * self.eng.L, engine=self.eng)
            try:
                yield
            finally:
                E._CTX["st"] = saved
        return cm()

    def _batch(self, eps):
        L = max(len(self.ep_tok[e]) for e in eps)
        ids = torch.zeros(len(eps), L, dtype=torch.long, device=self.eng.dev)
        m = torch.zeros(len(eps), L, dtype=torch.bool, device=self.eng.dev)
        for i, e in enumerate(eps):
            t = self.ep_tok[e]
            ids[i, :len(t)] = torch.as_tensor(t, device=self.eng.dev); m[i, :len(t)] = True
        return ids, m

    def _items(self, eps_or_items):
        """Episode ids (int, stored in ep_tok) or ready (tokens, landmark offsets) tuples -> list of tuples."""
        return [e if isinstance(e, tuple) else (self.ep_tok[int(e)], self.ep_lm[int(e)]) for e in eps_or_items]

    def _lm_nll(self, eps_or_items, kind="mean"):
        """NLL on landmark tokens per episode (no context but the episode), mean or max over the landmarks."""
        items = self._items(eps_or_items)
        out = []
        with torch.no_grad(), self._plain():
            for i in range(0, len(items), 16):
                part = items[i:i + 16]
                L = max(len(t) for t, _ in part)
                ids = torch.zeros(len(part), L, dtype=torch.long, device=self.eng.dev)
                for b, (t, _) in enumerate(part):
                    ids[b, :len(t)] = torch.as_tensor(t, device=self.eng.dev)
                lp = torch.log_softmax(self.eng.model(ids).logits.float(), -1)
                for b, (t, off) in enumerate(part):
                    off = [o for o in off if 0 < o < len(t)]
                    o = torch.as_tensor(off, device=self.eng.dev)
                    if not off:
                        out.append(0.0); continue
                    nl = -lp[b, o - 1].gather(-1, ids[b, o, None])
                    out.append(float(nl.max() if kind == "max" else nl.mean()))
        return np.array(out)

    # ---------- hippocampal recall distillation (target "hrd") ----------
    def _on_cue(self, sel, hit):
        """A later passage re-cued old episodes: keep it (q_pre tokens before the cueing landmark, q_post after) as a
        recall query of each re-cued episode; one query per episode per written chunk, the last n_q kept."""
        if self.target != "hrd":
            return
        from pilot.engine import _CTX
        toks, hit, seen = _CTX["st"]["tokens"], hit.cpu().numpy(), set()
        for j in np.flatnonzero(hit.any(1)):
            s = int(sel[j])
            for e in np.unique(self.Hep[hit[j]]).tolist():
                if e in seen or e not in self.ep_tok:
                    continue
                seen.add(e)
                q = dict(ids=[int(t) for t in toks[max(0, s - self.q_pre): s + self.q_post + 1]], start=max(0, s - self.q_pre))
                self.ep_q[e] = (self.ep_q.get(e, []) + [q])[-self.n_q:]
                self.st["hrd_queries"] += 1

    def recall_hook(self, l, wm, pos):
        """reuse "recall" (label-free, replaces kappa for HRD): at query position t, the H entry with the largest read
        weight over heads is a confirmed recall when the weight is >= w_min and the stream's next token is the token
        stored at that landmark (the model copied it from the episode, and the copy was right). Each confirmed recall
        adds one to the landmark's reuse count and records a recall query around t (once per episode per chunk)."""
        if self.reuse != "recall" or l < self.recall_from or "H" not in self.last[l]:
            return
        from pilot.engine import _CTX
        toks = _CTX["st"]["tokens"]
        top = self.last_top[l][0][0]                                            # (Hkv, M) H slots read per KV head
        Hq, T, M = wm.shape[1], wm.shape[2], top.shape[-1]
        slots = top.repeat_interleave(Hq // top.shape[0], 0).reshape(-1)       # (Hq * M)
        val, idx = wm[0, :, :, :M].float().permute(1, 0, 2).reshape(T, Hq * M).max(-1)
        slot = slots[idx].cpu().numpy()
        val = val.cpu().numpy()
        for t in np.flatnonzero(val >= self.w_min):
            p, s = int(pos[t]), int(slot[t])
            if p + 1 >= len(toks) or t == T - 1 or not self.Hval[s] or self.Hpos[s] < 0:
                continue
            if toks[p + 1] != toks[self.Hpos[s]] or (p, s) in self._rec_pos:
                continue
            self._rec_pos.add((p, s))                                           # once per position, whatever the layer
            e = int(self.Hep[s])
            self.Hcue[s] += 1
            self.st["recalls"] = self.st.get("recalls", 0) + 1
            if e in self._rec_ep or e not in self.ep_tok:
                continue
            self._rec_ep.add(e)                                                 # one query per episode per chunk
            end = min(p + self.q_post, int(pos[-1]))                            # never past the tokens seen so far
            q = dict(ids=[int(x) for x in toks[max(0, p - self.q_pre): end + 1]], start=max(0, p - self.q_pre))
            self.ep_q[e] = (self.ep_q.get(e, []) + [q])[-self.n_q:]
            self.st["hrd_queries"] += 1

    def _targets(self, e, q):
        """Teacher targets of query q of episode e, computed once (the base is frozen; episode and query are fixed)."""
        if "pos" in q:
            return q
        dev = self.eng.dev
        with torch.no_grad(), self._plain():
            lora_enable(self.eng, False)
            try:
                ep = self.ep_tok[e]
                t = torch.log_softmax(self.eng.model(torch.as_tensor([ep + q["ids"]], device=dev)).logits[0, len(ep):].float(), -1)
                b = torch.log_softmax(self.eng.model(torch.as_tensor([q["ids"]], device=dev)).logits[0].float(), -1)
            finally:
                lora_enable(self.eng, True)
        nxt = torch.as_tensor(q["ids"][1:], device=dev)
        q.update(hrd_targets(t, b, nxt, self.n_pos, self.min_ctx, self.topk, self.g_min))
        if q["pos"] is None:
            self.st["hrd_uninformative"] += 1
        return q

    def _hitems(self, eps_or_items):
        """Episode ids -> lists of informative recall queries with targets; lists (anchor items) pass through."""
        out = []
        for e in eps_or_items:
            if isinstance(e, list):
                out.append(e)
            else:
                e = int(e)
                out.append([q for q in (self._targets(e, q) for q in self.ep_q.get(e, [])) if q["pos"] is not None])
        return out

    def _hrd_kl(self, items):
        """Per item, per query: KL(teacher || cortex) at the query's target positions (student: the model with the
        LoRA and no hippocampus, reading the query alone). Returns a list (per item) of lists of (P,) tensors."""
        flat = [(i, q) for i, qs in enumerate(items) for q in qs]
        out = [[] for _ in items]
        for s in range(0, len(flat), 32):
            part = flat[s:s + 32]
            L = max(len(q["ids"]) for _, q in part)
            ids = torch.zeros(len(part), L, dtype=torch.long, device=self.eng.dev)
            for b, (_, q) in enumerate(part):
                ids[b, :len(q["ids"])] = torch.as_tensor(q["ids"], device=self.eng.dev)
            lp = torch.log_softmax(self.eng.model(ids).logits.float(), -1)
            for b, (i, q) in enumerate(part):
                out[i].append(kl_topk(q["t_ids"], q["t_logp"], lp[b, q["pos"].to(lp.device)]))
        return out

    def _hrd_err(self, eps_or_items):
        """Cortical error per episode: mean over its recall queries of the largest KL over target positions;
        nan for an episode with no informative query (never routed, never released)."""
        items = self._hitems(eps_or_items)
        with torch.no_grad(), self._plain():
            kl = self._hrd_kl(items)
        return np.array([np.mean([float(k.max()) for k in ks]) if ks else np.nan for ks in kl])

    def _cerr(self, eps_or_items):
        """Error the gate measures: recall-distillation error (hrd) or landmark NLL (lm)."""
        return self._hrd_err(eps_or_items) if self.target == "hrd" else self._lm_nll(eps_or_items)

    def _step(self, eps, scale=1.0):
        with torch.enable_grad(), self._plain():
            if self.target == "hrd":
                kl = [ks for ks in self._hrd_kl(self._hitems(eps)) if ks]
                # mean over target positions (the first version summed: with up to 16 positions per query the step
                # was far larger than an LM step at the same lr, and hrd_uniform destroyed the model, text NLL 6.3)
                loss = torch.stack([torch.stack([k.mean() for k in ks]).mean() for ks in kl]).mean()
            else:
                ids, m = self._batch(eps)
                logits = self.eng.model(ids).logits[:, :-1].float()
                loss = F.cross_entropy(logits[m[:, 1:]], ids[:, 1:][m[:, 1:]])
            if self.lam > 0:
                lp = torch.log_softmax(self.eng.model(self.text).logits[0].float(), -1)
                loss = loss + self.lam * (self.text_lp.exp() * (self.text_lp - lp)).sum(-1).mean()
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
        for gp in self.opt.param_groups:
            gp["lr"] = self.lr * scale
        self.opt.step()
        self.st["lora_steps"] += 1

    def consolidate_eps(self, eps, gate):
        """One replay step on episodes eps; with the gate, accept only if G - rho D > 0 (backtracking)."""
        if not gate or not self.anchors:
            self._step(eps)
            return True
        b0, a0 = self._cerr(eps), self._cerr(self.anchors)
        snap = ([p.detach().clone() for p in self.params], copy.deepcopy(self.opt.state_dict()))
        for attempt in range(3):
            self._step(eps, scale=0.5 ** attempt)
            gain = float(np.nansum(b0 - self._cerr(eps)))
            dmg = float(np.nansum(self._cerr(self.anchors) - a0))
            if gain - self.rho * dmg > 0:
                self.st["lora_shrunk"] += int(attempt > 0)
                return True
            for p, s in zip(self.params, snap[0]):
                p.data.copy_(s)
            self.opt.load_state_dict(snap[1])
        self.st["lora_rejected"] += 1
        return False

    def _anchor(self, eps):
        for e in eps:
            item = self._hitems([e])[0] if self.target == "hrd" else (self.ep_tok[e], self.ep_lm[e])
            if self.target == "hrd" and not item:
                continue                                                        # nothing measured: not an anchor
            self.A_seen += 1
            j = self.A_seen - 1 if self.A_seen <= self.R else int(self.rng.integers(self.A_seen))
            if j < len(self.anchors):
                self.anchors[j] = item
            elif j < self.R:
                self.anchors.append(item)

    # ---------- engine interface ----------
    def write(self, l, k_pre, v, pos, nll):
        if l == 0:
            from pilot.engine import _CTX
            toks = _CTX["st"]["tokens"]
            sel = self.hsel.select(pos, nll)
            start = max(0, int(pos[0]) - self.ctx)
            ep = self.n_episode if self.mode not in ("arrival", "surprise") else -1 - int(pos[0])    # queue ids < 0
            self.ep_tok[ep] = toks[start:int(pos[-1]) + 1].tolist()
            self.ep_start = getattr(self, "ep_start", {}); self.ep_start[ep] = start      # diagnostics only
            self.ep_lm[ep] = [int(p) - start for p in sel]
            if self.mode in ("arrival", "surprise"):
                self.queue.append(ep)
        if self.mode in ("arrival", "surprise"):
            return
        n0 = self.st["h_writes"]
        super().write(l, k_pre, v, pos, nll)
        if l == 0 and self.reuse == "recall" and self.st["h_writes"] > n0:      # only when this chunk wrote slots
            self.Hcue[torch.as_tensor(self.slots, device=self.Hcue.device)] = 0.0  # fresh slots: no recalls yet

    def end_chunk(self, pos):
        self._rec_pos, self._rec_ep = set(), set()
        if self.mode in ("arrival", "surprise"):
            per_chunk = max(1, self.n_steps // self.sleep_every)
            for ep in self.queue:
                scale = 1.0
                if self.mode == "surprise":
                    e = float(self._lm_nll([ep])[0])
                    self.err_n += 1; self.err_mean += (e - self.err_mean) / self.err_n
                    scale = min(2.0, e / max(self.err_mean, 1e-6))
                for _ in range(per_chunk):
                    self._step([ep], scale)
                self.st["consolidated_items"] += 1
                self.ep_tok.pop(ep, None); self.ep_lm.pop(ep, None)
            self.queue = []
            self.chunks += 1
            return
        self.chunks += 1
        if self.chunks % self.sleep_every == 0 and self.Hval.any():
            self.sleep()

    def sleep(self):
        """n_steps replay steps per sleep for every replay arm (equal compute); see the module docstring."""
        self.st["sleeps"] += 1
        if self.mode == "uniform" and self.target == "hrd":                    # uniform over re-cued episodes
            u = np.array([e for e in np.unique(self.Hep[self.Hval]).tolist() if self._hitems([e])[0]])
            if not len(u):
                self.st["sleeps_without_candidates"] = self.st.get("sleeps_without_candidates", 0) + 1
                return
            for _ in range(self.n_steps):
                self._replay(self.rng.choice(u, min(self.n_ep, len(u)), replace=False), gate=False, anchors=False)
            return
        idx, eps, u, d_ep, r_ep = self._episode_stats()
        self.r_hist = getattr(self, "r_hist", []) + [r_ep]
        self.d_hist = getattr(self, "d_hist", []) + [d_ep]
        if self.mode == "uniform":
            for _ in range(self.n_steps):
                self._replay(self.rng.choice(u, min(self.n_ep, len(u)), replace=False), gate=False, anchors=False)
            return
        if self.mode == "sleep_all":
            order = self.rng.permutation(u)
            for st in range(self.n_steps):
                self._replay(np.roll(order, -st * self.n_ep)[:self.n_ep], gate=False, anchors=False)
            self._release(u, anchors=False)
            return
        self._release(u[d_ep < self.tau], anchors=True)                         # schema-consistent
        cand = (d_ep >= self.tau) & (r_ep >= self.r_min)
        if not cand.any():
            self.st["sleeps_without_candidates"] = self.st.get("sleeps_without_candidates", 0) + 1
            return
        cu = u[cand]
        if self.priority == "rdelta":                                           # reuse x cortical error
            w = np.log1p(r_ep[cand]) * d_ep[cand]
        elif self.priority == "reuse":                                          # ablation: reuse only
            w = np.log1p(r_ep[cand])
        else:                                                                   # "delta": error only (surprise-like)
            w = d_ep[cand].copy()
        w = w + 1e-8
        replayed = set()
        for _ in range(self.n_steps):
            pick = self.rng.choice(cu, min(self.n_ep, len(cu)), replace=False, p=w / w.sum())
            self._replay(pick, gate=self.gate, anchors=True)
            replayed.update(int(e) for e in pick)
        idx, eps, u2, d2, _ = self._episode_stats()
        self._release(u2[np.isin(u2, list(replayed)) & (d2 < self.tau)], anchors=False)

    def _episode_stats(self):
        idx = np.flatnonzero(self.Hval)
        eps = self.Hep[idx]
        u = np.unique(eps)
        d_ep = self._hrd_err(list(u)) if self.target == "hrd" else self._lm_nll(list(u), self.delta_kind)
        r = (self.Hcue[torch.as_tensor(idx, device=self.Hk.device)] if self.reuse in ("cue", "recall")
             else self.Hr[:, torch.as_tensor(idx, device=self.Hk.device)].mean(0)).cpu().numpy()
        r_ep = np.array([r[eps == e].sum() for e in u])
        return idx, eps, u, d_ep, r_ep

    def _replay(self, ep_ids, gate, anchors):
        eps = [int(e) for e in ep_ids if int(e) in self.ep_tok]
        if self.target == "hrd":
            eps = [e for e in eps if self._hitems([e])[0]]
        if not eps:
            return
        ok = self.consolidate_eps(eps, gate)
        self.st["steps_tried"] += 1; self.st["steps_accepted"] += int(ok)
        self.st["consolidated_items"] += len(eps)
        if anchors and ok:
            self._anchor(eps)

    def _release(self, ep_ids, anchors):
        if anchors:
            self._anchor([int(e) for e in ep_ids if int(e) in self.ep_tok])
        super()._release(ep_ids, anchors=False)

    def heldout_text_nll(self):
        """Plain-text NLL with the stream's final cortex (LM-quality guard); 2,048 held-out wikitext tokens."""
        from pilot.text_stream import load_filler
        ids = torch.as_tensor(load_filler(self.eng.tok)[-2048:], device=self.eng.dev, dtype=torch.long)[None]
        with torch.no_grad(), self._plain():
            lp = torch.log_softmax(self.eng.model(ids).logits[0].float(), -1)
        return float(-lp[:-1].gather(-1, ids[0, 1:, None]).mean())

    def stats(self):
        s = super().stats()
        s.update(anchor_episodes=len(self.anchors), heldout_text_nll=self.heldout_text_nll())
        if self.target == "hrd":
            live = np.unique(self.Hep[self.Hval]).tolist()
            s.update(hrd_episodes_with_query=int(sum(bool(self._hitems([e])[0]) for e in live)),
                     hrd_episodes_in_H=len(live))
        return s


CFG_15B = (1536, 8960, 12, 128, 2)                     # hidden, intermediate, q heads, head dim, KV heads
RANK, KAPPA_PAIRS, ANCHOR_PAIRS = 8, 16, 16
CFG_3B = (2048, 11008, 16, 128, 2)                     # Qwen2.5-3B (16 Nov 2026 replication)
import os as _os
CFG_BASE = CFG_3B if _os.environ.get("SCHEMA_BASE", "1.5B") == "3B" else CFG_15B
LORA_P = int(np.ceil(lora_pairs(CFG_BASE, RANK)))      # about 540 pairs per KV head on 1.5B, 676 on 3B
EXTRA = 2048
LORA_SPECS = {
    "lora_schema": (0, dict(H_cap=EXTRA - LORA_P - KAPPA_PAIRS - ANCHOR_PAIRS, reuse="cue", aux_pairs=KAPPA_PAIRS,
                            anchor_pairs=ANCHOR_PAIRS, lora_pairs=LORA_P, rank=RANK)),
    "lora_schema_nogate": (0, dict(H_cap=EXTRA - LORA_P - KAPPA_PAIRS - ANCHOR_PAIRS, reuse="cue", gate=False,
                                   aux_pairs=KAPPA_PAIRS, anchor_pairs=ANCHOR_PAIRS, lora_pairs=LORA_P, rank=RANK)),
    "lora_uniform": (0, dict(H_cap=EXTRA - LORA_P, mode="uniform", lora_pairs=LORA_P, rank=RANK)),
    "lora_sleep_all": (0, dict(H_cap=EXTRA - LORA_P, mode="sleep_all", lora_pairs=LORA_P, rank=RANK)),
    "lora_arrival": (EXTRA - LORA_P, dict(H_cap=0, mode="arrival", lora_pairs=LORA_P, rank=RANK)),
    "lora_surprise": (EXTRA - LORA_P, dict(H_cap=0, mode="surprise", lora_pairs=LORA_P, rank=RANK)),
    # hippocampal recall distillation (results/schema_ideas_20261104.md). Recall queries and reuse come from confirmed
    # hippocampal recalls (LoRAMemory.recall_hook), so no kappa: its pairs go to H.
    "hrd_schema": (0, dict(H_cap=EXTRA - LORA_P - ANCHOR_PAIRS, reuse="recall", anchor_pairs=ANCHOR_PAIRS,
                           lora_pairs=LORA_P, rank=RANK, target="hrd")),
    "hrd_schema_nogate": (0, dict(H_cap=EXTRA - LORA_P - ANCHOR_PAIRS, reuse="recall", gate=False,
                                  anchor_pairs=ANCHOR_PAIRS, lora_pairs=LORA_P, rank=RANK, target="hrd")),
    "hrd_uniform": (0, dict(H_cap=EXTRA - LORA_P, mode="uniform", reuse="recall", lora_pairs=LORA_P, rank=RANK,
                            target="hrd")),
    # pre-registered confirmatory arm (results/prereg_20261116_confirm.md): no gate, no anchor reservoir, its bytes
    # returned to H; and two priority ablations with the same settings
    "hrd_schema_ng": (0, dict(H_cap=EXTRA - LORA_P, reuse="recall", gate=False, R=0, lora_pairs=LORA_P, rank=RANK,
                              target="hrd")),
    "hrd_delta_pri": (0, dict(H_cap=EXTRA - LORA_P, reuse="recall", gate=False, R=0, priority="delta", r_min=0,
                              lora_pairs=LORA_P, rank=RANK, target="hrd")),
    "hrd_reuse_pri": (0, dict(H_cap=EXTRA - LORA_P, reuse="recall", gate=False, R=0, priority="reuse",
                              lora_pairs=LORA_P, rank=RANK, target="hrd")),
}


def spec_pairs(kw):
    """Anchor episodes are token ids (R x about 96 ints), counted as anchor_pairs; episode token ids kept with H
    episodes are a few ints per landmark and not counted."""
    return kw["H_cap"] + kw.get("aux_pairs", 0) + kw.get("anchor_pairs", 0) + kw["lora_pairs"]
