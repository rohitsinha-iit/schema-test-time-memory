"""R2 gate: KL fine-tuning of AHN-GDN's parameters on the fact streams (proposal 4.5; AHN's own training recipe:
base frozen, KL to the full-attention teacher, AdamW lr 1e-4, wd 0.1, cosine with 10% warmup).

Full-sequence path (as AHN's mem_forward_train): attention by SDPA with a mask of 128 sinks plus the per-token
window (t - W, t]; the memory branch reads, for query t, the state of tokens [sinks, t - W], computed with
pilot.gdn.block_outputs over 64-token blocks. Gradient checkpointing per decoder layer.
Training streams: seeds 1000 and up (never validation 0-2 or test 100+), mixed with plain wikitext segments.

  python -m pilot.ahn_train --steps 400 --out pilot/data/ahn_ft.pt
"""
import argparse
import json
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from pilot.ahn_port import load_ahn_weights
from pilot.engine import _CTX, Engine
from pilot.gdn import block_outputs
from pilot.run_pilot import STREAM
from pilot.text_stream import load_filler, make_stream

NAMES = {"a": "a_proj.weight", "b": "b_proj.weight", "A_log": "A_log", "dt": "dt_bias", "g": "g_proj.weight",
         "norm": "o_norm.weight", "o": "o_proj.weight"}


class AHNParams(torch.nn.Module):
    def __init__(self, sd, L, dev):
        super().__init__()
        self.L = L
        for l in range(L):
            for short, name in NAMES.items():
                t = next(v for k, v in sd.items() if f"layers.{l}.ahn." in k and k.endswith(name))
                self.register_parameter(f"l{l}_{short}", torch.nn.Parameter(t.to(dev).float().clone()))

    def layer(self, l):
        return {s: getattr(self, f"l{l}_{s}") for s in NAMES}

    def state_dict_ahn(self):
        return {f"model.layers.{l}.ahn.fn.{name}": getattr(self, f"l{l}_{s}").detach().cpu().to(torch.bfloat16)
                for l in range(self.L) for s, name in NAMES.items()}


def gdn_full(p, q, k, v, h, W, S, block=64):
    """Memory branch for every position of a full sequence. q (B, Hq, T, d) and k, v (B, Hkv, T, d) post-RoPE,
    h (B, T, D) normalized layer input. Query t >= S + W reads tokens [S, t - W]."""
    B, Hq, T, d = q.shape
    n = T - W - S
    out = torch.zeros(B, Hq, T, d, device=q.device, dtype=torch.float32)
    if n <= 0:
        return out
    g = Hq // k.shape[1]
    hw = h[:, S:S + n].float()
    beta = torch.sigmoid(hw @ p["b"].T).transpose(1, 2)                                     # (B, Hq, n)
    la = (-p["A_log"].exp() * F.softplus(hw @ p["a"].T + p["dt"])).transpose(1, 2)
    K = F.normalize(k[:, :, S:S + n].float(), dim=-1, eps=1e-6).repeat_interleave(g, 1)
    V = v[:, :, S:S + n].float().repeat_interleave(g, 1)
    Q = F.normalize(q[:, :, S + W:].float(), dim=-1, eps=1e-6)
    Sst = torch.zeros(B, Hq, d, d, device=q.device)
    outs = []
    for i in range(0, n, block):
        sl = slice(i, min(n, i + block))
        o, Sst = block_outputs(Sst, K[:, :, sl], V[:, :, sl], Q[:, :, sl], la[..., sl], beta[..., sl], d ** -0.5)
        outs.append(o)
    O = torch.cat(outs, 2)
    hq = h[:, S + W:].float()
    On = O * torch.rsqrt(O.pow(2).mean(-1, keepdim=True) + 1e-5) * p["norm"]
    On = On * F.silu((hq @ p["g"].T).transpose(1, 2))[..., None]
    out[:, :, S + W:] = torch.einsum("bhti,hio->bhto", On, p["o"])
    return out


def full_attention(module, q, k, v, st):
    """mode "teacher": full causal attention; mode "train": sinks + per-token window, plus the memory branch."""
    l, T = module.layer_idx, q.shape[2]
    g = q.shape[1] // k.shape[1]
    if st["mode"] == "teacher":
        o = F.scaled_dot_product_attention(q, k.repeat_interleave(g, 1), v.repeat_interleave(g, 1), is_causal=True)
        return o.transpose(1, 2).contiguous()
    W, S = st["W"], st["sinks"]
    i = torch.arange(T, device=q.device)
    mask = (i[None, :] <= i[:, None]) & ((i[:, None] - i[None, :] < W) | (i[None, :] < S))
    o = F.scaled_dot_product_attention(q, k.repeat_interleave(g, 1), v.repeat_interleave(g, 1), attn_mask=mask)
    o = o.float() + gdn_full(st["ahn"].layer(l), q, k, v, st["h"][l], W, S)
    return o.to(q.dtype).transpose(1, 2).contiguous()


def hidden(eng, ids, mode, **kw):
    _CTX["st"] = dict(mode=mode, h=[None] * eng.L, engine=eng, **kw)
    return eng.model.model(ids, use_cache=False).last_hidden_state[0]


def kl_beyond(eng, h_s, h_t, start, chunk=512):
    """Mean KL(teacher || student) over positions >= start, computed in chunks of positions."""
    lm = eng.model.lm_head
    tot, n = 0.0, 0
    for i in range(start, h_s.shape[0] - 1, chunk):
        sl = slice(i, min(h_s.shape[0] - 1, i + chunk))
        lt = torch.log_softmax(lm(h_t[sl]).float(), -1)
        ls = torch.log_softmax(lm(h_s[sl]).float(), -1)
        tot = tot + (lt.exp() * (lt - ls)).sum(); n += ls.shape[0]
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--n_tokens", type=int, default=6144)
    ap.add_argument("--windows", default="512,1024")
    ap.add_argument("--sinks", type=int, default=128)
    ap.add_argument("--text_frac", type=float, default=0.33)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", action="store_true", help="only compare the full-sequence path with streaming")
    a = ap.parse_args()
    eng = Engine(a.model)
    for p in eng.model.parameters():
        p.requires_grad_(False)
    ahn = AHNParams(load_ahn_weights(), eng.L, eng.dev)
    filler = load_filler(eng.tok)
    import pilot.engine as E
    E.FULL_ATTENTION = full_attention

    if a.check:
        from pilot.ahn_port import AHNMemory
        toks = np.load("/tmp/ahn_toks.npy")                      # the tokens AHN's own code was run on (R0)
        ids = torch.as_tensor(toks, device=eng.dev)[None]
        with torch.no_grad():
            hs = hidden(eng, ids, "train", W=1024, sinks=a.sinks, ahn=ahn)
            lp = torch.log_softmax(eng.model.lm_head(hs).float(), -1)
            full = (-lp[:-1].gather(-1, ids[0, 1:, None])[:, 0]).cpu().numpy()
        stream = eng.run(toks, 1024, memory=AHNMemory(eng))["nll"][1:]
        d = np.abs(full - stream)
        print(f"full-sequence vs streaming AHN: mean |dNLL| {d.mean():.4f}, beyond window {d[1152:].mean():.4f}, max {d.max():.3f}")
        import os
        if os.path.exists("/tmp/ahn_ref.npy"):                       # AHN's own code on the same tokens
            r = np.abs(full - np.load("/tmp/ahn_ref.npy"))
            print(f"full-sequence vs AHN reference: mean |dNLL| {r.mean():.4f}, beyond window {r[1152:].mean():.4f}")
        return

    eng.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    eng.model.train()
    opt = torch.optim.AdamW(ahn.parameters(), lr=a.lr, weight_decay=0.1)
    warm = max(1, int(0.1 * a.steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, a.steps - warm))))
    windows = [int(w) for w in a.windows.split(",")]
    rng = np.random.default_rng(1000)
    log, t0 = [], time.time()
    for step in range(a.steps):
        W = int(rng.choice(windows))
        if rng.random() < a.text_frac:
            o = int(rng.integers(len(filler) - a.n_tokens)); toks, kind = filler[o:o + a.n_tokens], "text"
        else:
            s = make_stream(np.random.default_rng([1000, step]), eng.tok, filler,
                            **dict(STREAM, n_tokens=a.n_tokens, exc_rate=float(rng.choice((0.05, 0.2, 0.5)))))
            toks, kind = s["tokens"], "facts"
        ids = torch.as_tensor(np.asarray(toks), device=eng.dev, dtype=torch.long)[None]
        with torch.no_grad():
            h_t = hidden(eng, ids, "teacher")
        h_s = hidden(eng, ids, "train", W=W, sinks=a.sinks, ahn=ahn)
        loss = kl_beyond(eng, h_s, h_t, a.sinks + W)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ahn.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        log.append(dict(step=step, kind=kind, W=W, kl=float(loss), s=round(time.time() - t0, 1)))
        if (step + 1) % 10 == 0:
            r = log[-10:]
            print(f"step {step + 1} kl facts {np.mean([x['kl'] for x in r if x['kind'] == 'facts'] or [np.nan]):.4f} "
                  f"text {np.mean([x['kl'] for x in r if x['kind'] == 'text'] or [np.nan]):.4f} "
                  f"lr {sched.get_last_lr()[0]:.2e} {log[-1]['s']}s mem {torch.cuda.max_memory_allocated() / 2**30:.1f}GB", flush=True)
    torch.save(ahn.state_dict_ahn(), a.out)
    with open(a.out.replace(".pt", "_log.json"), "w") as f:
        json.dump(dict(args=vars(a), log=log), f)


if __name__ == "__main__":
    main()
