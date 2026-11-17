"""AHN-GDN (ByteDance-Seed/AHN, src/ahn/rnn/gated_deltanet.py + transformer/qwen2_ahn) ported into the pilot engine.

Per layer and query head (k, v repeated from the KV heads, as AHN's pre_attn_forward does):
* the window is exactly per-token: query t attends to keys in (t - W, t], no sinks (AHN inference default);
* the gated delta-rule state holds every token at position <= t - W; query t reads it after the token at t - W
  is written (o_t = d^-1/2 S_t q_t with unit-norm post-RoPE q, k);
* write gates from the written token's normalized layer input h: beta = sigmoid(b_proj h),
  log alpha = -exp(A_log) softplus(a_proj h + dt_bias);
* output: o = RMSNorm(o) * w * swish(g_proj h_query) per head, then the per-head o_proj, added to the attention
  output before the layer's o_proj.
The state costs 16 heads x 128 x 128 floats per layer = 512 (key, value) pairs per KV head.
"""
import glob
import os

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from pilot.gdn import block_outputs

AHN_REPO = "ByteDance-Seed/AHN-GDN-for-Qwen-2.5-Instruct-3B"
STATE_PAIRS = 512


def load_ahn_weights(repo=AHN_REPO):
    from huggingface_hub import snapshot_download
    path = snapshot_download(repo, local_files_only=True)
    sd = {}
    for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        sd.update(load_file(f))
    return sd


class AHNMemory:
    exact_window, sinks = True, 128                         # AHN inference default: 128 attention sinks

    def __init__(self, eng, weights=None, lag_extra=0, no_gate=False, path=None):
        if path is not None:
            weights = torch.load(path)
        self.eng, self.lag_extra, self.no_gate = eng, lag_extra, no_gate             # perturbations: check only
        sd = weights if weights is not None else load_ahn_weights()
        self.P = []
        for l in range(eng.L):
            pre = [k for k in sd if f"layers.{l}.ahn." in k]
            get = lambda name: next(sd[k] for k in pre if k.endswith(name)).to(eng.dev).float()
            self.P.append(dict(a=get("a_proj.weight"), b=get("b_proj.weight"), A_log=get("A_log"), dt=get("dt_bias"),
                               g=get("g_proj.weight"), norm=get("o_norm.weight"), o=get("o_proj.weight")))
        self.S = [None] * eng.L
        self.beta, self.la = [{} for _ in range(eng.L)], [{} for _ in range(eng.L)]
        self.retest = False

    def gates(self, l, h):
        p = self.P[l]
        h = h.float()
        return torch.sigmoid(h @ p["b"].T), -p["A_log"].exp() * F.softplus(h @ p["a"].T + p["dt"])

    def gdn_out(self, l, q, st):
        """Memory branch output (B, Hq, T, d) for the current chunk or probe batch."""
        eng, W, p = self.eng, st["W"], self.P[l]
        h = st["h"][l]                                                              # (B, T, D) normalized input
        B, Hq, T, d = q.shape
        pos = st["pos"]
        stream = st["mode"] == "stream"
        if stream:
            b, la = self.gates(l, h[0])
            self.beta[l].update(zip(pos.tolist(), b)); self.la[l].update(zip(pos.tolist(), la))
        S0 = self.S[l] if self.S[l] is not None else torch.zeros(1, Hq, d, d, device=q.device)
        wp = pos - W - self.lag_extra
        win = st["win"][l]
        ok = (wp >= 0) & (win is not None)
        if win is not None:
            ok &= np.isin(wp, win[2])
        Kw = torch.zeros(1, Hq, T, d, device=q.device); Vw = torch.zeros(1, Hq, T, d, device=q.device)
        bw = torch.zeros(1, Hq, T, device=q.device); lw = torch.zeros(1, Hq, T, device=q.device)
        if ok.any():
            ti = np.flatnonzero(ok)
            wi = torch.as_tensor(np.searchsorted(win[2], wp[ti]), device=q.device)
            g = Hq // win[0].shape[1]
            Kw[:, :, ti] = win[0][:, :, wi].float().repeat_interleave(g, 1)
            Vw[:, :, ti] = win[1][:, :, wi].float().repeat_interleave(g, 1)
            bw[0, :, ti] = torch.stack([self.beta[l][int(x)] for x in wp[ti]], 1)
            lw[0, :, ti] = torch.stack([self.la[l][int(x)] for x in wp[ti]], 1)
        Kn = F.normalize(Kw, dim=-1, eps=1e-6)
        Qn = F.normalize(q.float(), dim=-1, eps=1e-6)
        O, S_end = block_outputs(S0.expand(B, -1, -1, -1), Kn.expand(B, -1, -1, -1), Vw.expand(B, -1, -1, -1), Qn,
                                 lw.expand(B, -1, -1), bw.expand(B, -1, -1), d ** -0.5)
        if stream:
            self.S[l] = S_end[:1]
            for x in wp[ok]:                                                        # written: gates no longer needed
                self.beta[l].pop(int(x), None); self.la[l].pop(int(x), None)
        gq = h.float() @ p["g"].T                                                   # (B, T, Hq)
        On = O * torch.rsqrt(O.pow(2).mean(-1, keepdim=True) + 1e-5) * p["norm"]
        if not self.no_gate:
            On = On * F.silu(gq.transpose(1, 2))[..., None]
        return torch.einsum("bhti,hio->bhto", On, p["o"])

    # pilot memory interface
    def read(self, l, q, p_mem, probe=False):
        return None

    def feedback(self, l, mass):
        pass

    def write(self, l, k_pre, v, pos, nll):
        pass

    def end_chunk(self, pos):
        pass

    def retest_mode(self):
        self.retest = True                                       # window is cleared by the engine; the state stays

    def stats(self):
        return dict(state_pairs=STATE_PAIRS)
