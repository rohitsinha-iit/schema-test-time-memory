"""Stage 0b backbone (stage0b_spec.md, section 2): decoder-only transformer with chunked sliding-window
attention and an AHN-style gated delta-rule gist per layer and head.

Chunked SWA: the sequence is cut into chunks of c tokens; a query in chunk j attends causally inside
its chunk and to the W tokens before the chunk start. The gist of chunk j is the delta-rule state
after all tokens that left the window before chunk j (see model0b/gist.py), read with
gamma(x) * (S q) and a separate output projection, and added to the attention output.

`dense=True, gist=False` gives plain causal attention without a gist (sanity run S1, spec section 11).

`forward(tokens, W)` returns logits; with `return_state=True` it also returns, per layer, the window
cache and the gist state that a segment following the sequence would see. `probe_logits` evaluates
independent short queries against that state without writing memory.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gist import read_states


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d)); self.eps = eps

    def forward(self, x):
        return self.w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


def rope(x, pos, base=10000.0):
    """x (..., T, d), pos (T,) absolute positions."""
    d = x.shape[-1]
    inv = 1.0 / base ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d)
    ang = pos.to(torch.float32)[:, None] * inv[None, :]
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


def swa_mask(n_chunks, chunk, window, device):
    """(n, c, W + c) bool: key slot m of chunk j is absolute position j c - W + m."""
    j = torch.arange(n_chunks, device=device)[:, None, None]
    r = torch.arange(chunk, device=device)[None, :, None]
    m = torch.arange(window + chunk, device=device)[None, None, :]
    return (j * chunk - window + m >= 0) & (m <= window + r)


def chunked_swa(q, k, v, chunk, window):
    """q, k, v (B, H, T, d) with RoPE applied; T and window multiples of chunk. Returns (B, H, T, d)."""
    B, H, T, d = q.shape
    n = T // chunk
    pad = lambda x: torch.cat([x.new_zeros(B, H, window, d), x], dim=2)
    span = lambda x: pad(x).unfold(2, window + chunk, chunk).permute(0, 2, 1, 4, 3).reshape(B * n, H, window + chunk, d)
    qc = q.view(B, H, n, chunk, d).permute(0, 2, 1, 3, 4).reshape(B * n, H, chunk, d)
    mask = swa_mask(n, chunk, window, q.device)[None].expand(B, n, chunk, window + chunk).reshape(B * n, 1, chunk, window + chunk)
    o = F.scaled_dot_product_attention(qc, span(k), span(v), attn_mask=mask)
    return o.view(B, n, H, chunk, d).permute(0, 2, 1, 3, 4).reshape(B, H, T, d)


def dense_swa_reference(q, k, v, chunk, window):
    """Same attention pattern with a dense (T, T) mask, for tests."""
    T = q.shape[2]
    p = torch.arange(T, device=q.device)
    mask = (p[None, :] <= p[:, None]) & (p[None, :] >= (p[:, None] // chunk) * chunk - window)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


class Block(nn.Module):
    def __init__(self, d_model, n_heads, mlp_mult=4, dense=False, gist=True):
        super().__init__()
        self.H, self.dh, self.dense, self.use_gist = n_heads, d_model // n_heads, dense, gist
        self.n1, self.n2 = RMSNorm(d_model), RMSNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        # gist: per-token decay alpha and write strength beta (from the evicted token), read gate gamma
        self.a_proj, self.b_proj, self.g_proj = (nn.Linear(d_model, n_heads) for _ in range(3))
        self.gist_o = nn.Linear(d_model, d_model, bias=False)
        nn.init.constant_(self.a_proj.bias, -5.0)       # alpha = exp(-softplus(.)) starts near 0.993
        hid = mlp_mult * d_model
        self.w1, self.w3 = nn.Linear(d_model, hid, bias=False), nn.Linear(d_model, hid, bias=False)
        self.w2 = nn.Linear(hid, d_model, bias=False)

    def _heads(self, x):  # (B, T, D) -> (B, H, T, dh)
        return x.view(*x.shape[:-1], self.H, self.dh).transpose(-3, -2)

    def _mlp(self, x):
        h = self.n2(x)
        return x + self.w2(F.silu(self.w1(h)) * self.w3(h))

    def forward(self, x, window, chunk, return_state=False):
        B, T, D = x.shape
        h = self.n1(x)
        q, k, v = (self._heads(t) for t in self.qkv(h).chunk(3, dim=-1))
        pos = torch.arange(T, device=x.device)
        qr, kr = rope(q, pos), rope(k, pos)
        if self.dense:
            att = F.scaled_dot_product_attention(qr, kr, v, is_causal=True)
        else:
            att = chunked_swa(qr, kr, v, chunk, window)
        mix = self.o(att.transpose(1, 2).reshape(B, T, D))
        if not self.use_gist:
            assert not return_state
            return self._mlp(x + mix), None
        # gist, in at least float32
        gd = torch.promote_types(x.dtype, torch.float32)
        with torch.autocast(device_type=x.device.type, enabled=False):
            la = -F.softplus(self.a_proj(h).to(gd)).transpose(1, 2)             # (B, H, T)
            beta = torch.sigmoid(self.b_proj(h).to(gd)).transpose(1, 2)
            kn, qn = F.normalize(k.to(gd), dim=-1), F.normalize(q.to(gd), dim=-1)
            S = read_states(kn, v.to(gd), la, beta, chunk, window)              # (B, H, n + 1, dv, dk)
            n = T // chunk
            g = torch.einsum("bhnvk,bhnck->bhncv", S[:, :, :n], qn.view(B, self.H, n, chunk, self.dh))
            g = g.reshape(B, self.H, T, self.dh) * torch.sigmoid(self.g_proj(h).to(gd)).transpose(1, 2)[..., None]
        mix = mix + self.gist_o(g.to(att.dtype).transpose(1, 2).reshape(B, T, D))
        out = self._mlp(x + mix)
        if not return_state:
            return out, None
        return out, dict(k=kr[:, :, -window:], v=v[:, :, -window:], S=S[:, :, n], T=T)

    def probe(self, x, st):
        """x (B, P, t, D): P independent queries at positions T .. T+t-1, reading the window cache and gist
        state `st` (no writes)."""
        B, P, t, D = x.shape
        h = self.n1(x)
        q, k, v = (self._heads(z) for z in self.qkv(h).chunk(3, dim=-1))         # (B, P, H, t, dh)
        pos = torch.arange(st["T"], st["T"] + t, device=x.device)
        qr, kr = rope(q, pos), rope(k, pos)
        Wn = st["k"].shape[2]
        s_cache = torch.einsum("bphid,bhjd->bphij", qr, st["k"]) / math.sqrt(self.dh)
        valid = torch.arange(Wn, device=x.device) >= Wn - st["T"]               # cache slots before position 0
        s_cache = s_cache.masked_fill(~valid, -float("inf"))
        s_self = torch.einsum("bphid,bphjd->bphij", qr, kr) / math.sqrt(self.dh)
        s_self = s_self.masked_fill(~torch.ones(t, t, dtype=torch.bool, device=x.device).tril(), -float("inf"))
        sd = torch.promote_types(v.dtype, torch.float32)
        a = torch.softmax(torch.cat([s_cache, s_self], dim=-1).to(sd), dim=-1).to(v.dtype)
        att = torch.einsum("bphij,bhjd->bphid", a[..., :Wn], st["v"]) + torch.einsum("bphij,bphjd->bphid", a[..., Wn:], v)
        gd = st["S"].dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            qn = F.normalize(q.to(gd), dim=-1)
            g = torch.einsum("bhvk,bphik->bphiv", st["S"], qn)
            g = g * torch.sigmoid(self.g_proj(h).to(gd)).permute(0, 1, 3, 2)[..., None]
        merge = lambda z: z.transpose(-3, -2).reshape(B, P, t, D)
        mix = self.o(merge(att)) + self.gist_o(merge(g.to(att.dtype)))
        return self._mlp(x + mix)


class Backbone(nn.Module):
    def __init__(self, vocab, d_model=256, n_layers=4, n_heads=4, chunk=64, dense=False, gist=True):
        super().__init__()
        self.chunk = chunk
        self.emb = nn.Embedding(vocab, d_model)
        self.blocks = nn.ModuleList(Block(d_model, n_heads, dense=dense, gist=gist) for _ in range(n_layers))
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, tokens, window, return_state=False):
        assert tokens.shape[1] % self.chunk == 0 and window % self.chunk == 0
        x, states = self.emb(tokens), []
        for blk in self.blocks:
            x, st = blk(x, window, self.chunk, return_state)
            states.append(st)
        logits = self.head(self.norm(x))
        return (logits, states) if return_state else logits

    def probe_logits(self, probe_tokens, states):
        """probe_tokens (B, P, t): independent queries after the sequence; returns logits (B, P, t, V)."""
        x = self.emb(probe_tokens)
        for blk, st in zip(self.blocks, states):
            x = blk.probe(x, st)
        return self.head(self.norm(x))
