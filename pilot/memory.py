"""Pilot memories for the training-free engine (pilot/engine.py), per layer and KV head.

Hippocampus H (proposal 4.1): every batch of tokens leaving the window is one episode. It keeps `m` landmark
tokens, chosen by the selector of pilot/fit_selector.py (a linear probe on the frozen model's mid-layer hidden
state, fitted by self-distillation from the full-attention teacher's attention; a one-layer stand-in for the
proposal's learned attention pooling). Model surprise (NLL) was tried first and kept no more fact tokens than
chance (validation streams, 4 Nov). Landmarks are pre-RoPE keys and values.
Read: the n_read landmarks with the highest attention logit (max over the chunk's queries of the KV group),
with keys rotated to p_mem, join the attention softmax. Reuse tag r: attention mass a landmark receives.
Eviction: FIFO by episode when over capacity.

Budget unit: one (key, value) pair per KV head per layer. Every arm gets W + EXTRA pairs (hard rule 2).
"""
import numpy as np
import torch

EXTRA = 2048          # pairs per layer per KV head beyond the base window W


class Hippocampus:
    def __init__(self, eng, cap, m=16, n_read=128):
        self.eng, self.cap, self.m, self.n_read = eng, cap, m, n_read
        from pilot.fit_selector import selector_path
        sel = torch.load(selector_path(eng.model.config._name_or_path))
        self.feat_layer = sel["layer"]
        self.sel = {k: v.to(eng.dev).float() for k, v in sel.items() if torch.is_tensor(v)}
        self._sel_cache = (None, None)
        L, H, d = eng.L, eng.Hkv, eng.d
        self.k = [torch.zeros(H, 0, d, device=eng.dev, dtype=torch.bfloat16) for _ in range(L)]
        self.v = [torch.zeros(H, 0, d, device=eng.dev, dtype=torch.bfloat16) for _ in range(L)]
        self.r = [torch.zeros(H, 0, device=eng.dev) for _ in range(L)]
        self.ep = np.zeros(0, np.int64)      # episode id per landmark (same for all layers)
        self.pos = np.zeros(0, np.int64)     # stream position per landmark
        self.n_ep, self.last_read = 0, [None] * L
        self.writes = 0

    def select(self, pos, nll):
        """Landmark positions among the evicted positions `pos`: the m tokens with the highest selector score."""
        if self._sel_cache[0] == int(pos[0]):
            return self._sel_cache[1]
        from pilot.engine import _CTX
        f = _CTX["st"]["feat"][torch.as_tensor(pos, device=self.eng.dev)].float()
        score = (((f - self.sel["mu"]) / self.sel["sd"]) @ self.sel["w"] + self.sel["b"]).cpu().numpy()
        keep = np.sort(pos[np.argsort(-score, kind="stable")[: self.m]])
        self._sel_cache = (int(pos[0]), keep)
        return keep

    def write(self, l, k_pre, v, pos, nll):
        sel = self.select(pos, nll)
        idx = torch.as_tensor(np.searchsorted(pos, sel), device=self.eng.dev)
        self.k[l] = torch.cat([self.k[l], k_pre[:, idx]], 1)
        self.v[l] = torch.cat([self.v[l], v[:, idx]], 1)
        self.r[l] = torch.cat([self.r[l], torch.zeros(k_pre.shape[0], len(sel), device=self.eng.dev)], 1)
        if l == 0:
            self.ep = np.concatenate([self.ep, np.full(len(sel), self.n_ep)]); self.n_ep += 1
            self.pos = np.concatenate([self.pos, sel]); self.writes += len(sel)
        if l == self.eng.L - 1:
            self._evict()

    def _evict(self):
        over = len(self.ep) - self.cap
        if over <= 0:
            return
        cut = int(np.searchsorted(self.ep, self.ep[over - 1], side="right"))     # whole oldest episodes
        for l in range(self.eng.L):
            self.k[l], self.v[l], self.r[l] = self.k[l][:, cut:], self.v[l][:, cut:], self.r[l][:, cut:]
        self.ep, self.pos = self.ep[cut:], self.pos[cut:]

    def retest_mode(self):
        self.retest = True

    def read(self, l, q, p_mem, probe=False):
        n = self.k[l].shape[1]
        if n == 0 or getattr(self, "retest", False):
            return None
        B, Hq, T, d = q.shape
        H = self.k[l].shape[0]
        k_rot = self.eng.rotate(self.k[l], np.full(n, p_mem))                      # (H, n, d)
        s = torch.einsum("bhtd,hnd->bhtn", q.float(), k_rot.repeat_interleave(Hq // H, 0).float())
        s = s.amax(2).view(B, H, Hq // H, n).amax(2)                                # (B, H, n)
        top = s.topk(min(self.n_read, n), -1).indices                              # (B, H, k)
        self.last_read[l] = top
        kk = torch.gather(k_rot[None].expand(B, -1, -1, -1), 2, top[..., None].expand(-1, -1, -1, d))
        vv = torch.gather(self.v[l][None].expand(B, -1, -1, -1), 2, top[..., None].expand(-1, -1, -1, d))
        return kk, vv, torch.ones(B, H, top.shape[-1], dtype=torch.bool, device=q.device)

    def feedback(self, l, mass):
        top = self.last_read[l][0]                                                  # stream mode: B = 1
        self.r[l].scatter_add_(1, top, mass[0])

    def end_chunk(self, pos):
        pass

    def stats(self):
        return dict(H_landmarks=int(len(self.ep)), H_episodes=int(len(np.unique(self.ep))), H_writes=int(self.writes),
                    H_span=int(self.pos.max() - self.pos.min()) if len(self.pos) else 0)


ARMS = {
    "swa_big": lambda W: (W + EXTRA, None),
    "hippo": lambda W: (W, lambda eng: Hippocampus(eng, cap=EXTRA)),
}
