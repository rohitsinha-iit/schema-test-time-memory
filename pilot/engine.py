"""Training-free streaming engine for the pilot: a frozen HF causal LM processed in chunks of c tokens with a
sliding window of W tokens (plus attention sinks), and an optional memory that is read at chunk start and
written with the key/value pairs of tokens leaving the window.

Attention runs in a custom attention function (registered as "pilot"), so every layer sees, per KV head:
[sinks | window | memory entries | current chunk (causal)]. Memory entries are stored as pre-RoPE keys and are
rotated to one position p_mem = chunk start - W (just before the window, as InfLLM does). Attention is computed
explicitly so the mass each memory entry receives can be fed back to the memory (reuse tags).

Probes are read-only: every probe sees the end-of-stream window and memory, nothing is written.
"""
import numpy as np
import torch
from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer

_CTX = {}
FULL_ATTENTION = None          # set by pilot.ahn_train: full-sequence "teacher" / "train" modes


def _rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class Engine:
    def __init__(self, name="Qwen/Qwen2.5-0.5B-Instruct", device="cuda", dtype=torch.bfloat16):
        AttentionInterface.register("pilot", _attention)
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, attn_implementation="pilot")
        self.model.to(device).eval()
        c = self.model.config
        self.L, self.Hq, self.Hkv = c.num_hidden_layers, c.num_attention_heads, c.num_key_value_heads
        self.d = c.hidden_size // c.num_attention_heads
        self.dev = device
        self.inv_freq = self.model.model.rotary_emb.inv_freq.float().to(device)
        for layer in self.model.model.layers:                     # normalized layer input, for AHN gates
            layer.self_attn.register_forward_pre_hook(_capture_h, with_kwargs=True)

    def cos_sin(self, pos):
        f = torch.as_tensor(pos, device=self.dev, dtype=torch.float32)[:, None] * self.inv_freq
        e = torch.cat((f, f), -1)
        return e.cos(), e.sin()

    def rotate(self, x, pos, inverse=False):
        """RoPE on (..., n, d) at positions pos (n,), or its inverse."""
        cos, sin = self.cos_sin(pos)
        x32 = x.float()
        out = x32 * cos + _rotate_half(x32) * (-sin if inverse else sin)
        return out.to(x.dtype)

    def rotate_at(self, x, pos, inverse=False):
        """RoPE (or its inverse) on x (..., d) at per-element positions pos (...,) (a tensor)."""
        f = pos.float()[..., None] * self.inv_freq
        e = torch.cat((f, f), -1)
        cos, sin = e.cos(), e.sin()
        return x * cos + _rotate_half(x) * (-sin if inverse else sin)

    @torch.no_grad()
    def run(self, tokens, W, memory=None, chunk=64, sinks=4, probes=None, probe_batch=32, feat_layer=None,
            capture_layers=None, pair_layers=None, min_dist=64, qk=None, exact_window=False):
        """Stream `tokens` (1-D) through the model; return per-token NLL and, if probes are given, probe results.
        feat_layer: keep each token's hidden state after that layer (st["feat"], used by learned landmark selection).
        capture_layers: sum, per stream token, the attention mass that probe tokens give it in those layers.
        pair_layers: for each stream token t, record the earlier token j (t - j > min_dist, sinks excluded) with the
        largest attention weight from t in any head of those layers (res["pair_j"], res["pair_w"])."""
        n = len(tokens)
        if feat_layer is None and memory is not None:
            feat_layer = getattr(memory, "feat_layer", None)
        sinks = getattr(memory, "sinks", sinks)
        st = dict(engine=self, W=W, memory=memory, sinks=sinks, mode="stream",
                  win=[None] * self.L, sink=[None] * self.L, feat=None, h=[None] * self.L,
                  exact_window=exact_window or getattr(memory, "exact_window", False),
                  capture=None if capture_layers is None else dict(layers=set(capture_layers), mass=np.zeros(n)),
                  pairs=None if pair_layers is None else dict(layers=set(pair_layers), j=np.full(n, -1),
                                                               w=np.zeros(n), min_dist=min_dist),
                  qk=qk)
        st["tokens"] = np.asarray(tokens)
        _CTX["st"] = st
        nll = np.zeros(n, np.float32)
        ids = torch.as_tensor(np.asarray(tokens), device=self.dev)[None]
        prev_logit = None
        for s in range(0, n, chunk):
            e = min(n, s + chunk)
            st["pos"] = np.arange(s, e)
            st["nll"] = None
            out = self.model(ids[:, s:e], position_ids=torch.arange(s, e, device=self.dev)[None], use_cache=False,
                             output_hidden_states=feat_layer is not None)
            if feat_layer is not None:
                h = out.hidden_states[feat_layer][0]
                if st["feat"] is None:
                    st["feat"] = torch.zeros(n, h.shape[-1], device=self.dev, dtype=torch.float16)
                st["feat"][s:e] = h.half()
            lp = torch.log_softmax(out.logits[0].float(), -1)
            tgt = ids[0, s + 1:e + 1]
            nll[s + 1:e] = (-lp[: e - s - 1].gather(-1, tgt[: e - s - 1, None])[:, 0]).cpu().numpy()
            if prev_logit is not None:
                nll[s] = float(-prev_logit[ids[0, s]])
            prev_logit = lp[-1]
            st["nll_all"] = nll
            if memory is not None:
                memory.end_chunk(e)
        res = dict(nll=nll)
        if probes is not None:
            res["probes"] = self._probes(st, probes, n, probe_batch)
        if st["capture"] is not None:
            res["probe_mass"] = st["capture"]["mass"]
        res["feat"] = st["feat"]
        if st["pairs"] is not None:
            res["pair_j"], res["pair_w"] = st["pairs"]["j"], st["pairs"]["w"]
        return res

    @torch.no_grad()
    def run_probes_only(self, probes, n, memory=None, probe_batch=32):
        """Test-retest probes after the last run: window cleared (sinks kept); the memory decides what it still
        exposes (pilot memories hide H in retest mode)."""
        st = _CTX["st"]
        st["win"] = [None] * self.L
        return self._probes(st, probes, n, probe_batch)

    def probe_logits(self, probes, n, retest=False):
        """Logits at each probe's target positions, with autograd (for training read gates). Uses the state of the
        last run; retest clears the window (call it after the normal probes)."""
        st = _CTX["st"]
        if retest:
            st["win"] = [None] * self.L
        st["mode"] = "probe"
        seqs = [p["prompt"] + p["target"] for p in probes]
        T = max(len(x) for x in seqs)
        ids = torch.full((len(probes), T), self.tok.pad_token_id or 0, device=self.dev)
        for b, x in enumerate(seqs):
            ids[b, :len(x)] = torch.as_tensor(x, device=self.dev)
        st["pos"] = np.arange(n, n + T)
        logits = self.model(ids, position_ids=torch.arange(n, n + T, device=self.dev)[None].expand(len(probes), T),
                            use_cache=False).logits
        st["mode"] = "stream"
        return [logits[b, len(p["prompt"]) - 1:len(p["prompt"]) - 1 + len(p["target"])].float()
                for b, p in enumerate(probes)]

    def _probes(self, st, probes, n, bs):
        st["mode"] = "probe"
        out = []
        for i in range(0, len(probes), bs):
            part = probes[i:i + bs]
            seqs = [p["prompt"] + p["target"] for p in part]
            T = max(len(x) for x in seqs)
            ids = torch.full((len(part), T), self.tok.pad_token_id or 0, device=self.dev)
            for b, x in enumerate(seqs):
                ids[b, :len(x)] = torch.as_tensor(x, device=self.dev)
            st["pos"] = np.arange(n, n + T)
            logits = self.model(ids, position_ids=torch.arange(n, n + T, device=self.dev)[None].expand(len(part), T),
                                use_cache=False).logits.float()
            for b, p in enumerate(part):
                a, t = len(p["prompt"]), len(p["target"])
                lp = torch.log_softmax(logits[b, a - 1:a - 1 + t], -1)
                tgt = torch.as_tensor(p["target"], device=self.dev)
                correct = bool((lp.argmax(-1) == tgt).all())
                out.append(dict(correct=correct, logprob=float(lp.gather(-1, tgt[:, None]).sum())))
        st["mode"] = "stream"
        return out


def _attention(module, q, k, v, attention_mask, scaling=None, dropout=0.0, **kw):
    st = _CTX["st"]
    if st.get("mode") in ("teacher", "train"):
        return FULL_ATTENTION(module, q, k, v, st), None
    eng, l, W = st["engine"], module.layer_idx, st["W"]
    B, Hq, T, d = q.shape
    Hkv, g = k.shape[1], q.shape[1] // k.shape[1]
    win, sink = st["win"][l], st["sink"][l]
    ks, vs, kinds = [], [], []                    # context blocks; kinds: 0 context, 1 memory
    if sink is not None:
        ks.append(sink[0].expand(B, -1, -1, -1)); vs.append(sink[1].expand(B, -1, -1, -1)); kinds.append(0)
    if win is not None:
        ks.append(win[0].expand(B, -1, -1, -1)); vs.append(win[1].expand(B, -1, -1, -1)); kinds.append(0)
    mem = st["memory"]
    read = None
    if mem is not None:
        p_mem = int(st["pos"][0]) - W
        read = mem.read(l, q, p_mem, probe=st["mode"] == "probe")     # (k_rot, v, valid) with shapes (B, Hkv, M, d)
        if read is not None:
            mk, mv, mvalid = read
            ks.append(mk.expand(B, -1, -1, -1)); vs.append(mv.expand(B, -1, -1, -1)); kinds.append(1)
    ks.append(k); vs.append(v); kinds.append(0)
    K = torch.cat(ks, 2); V = torch.cat(vs, 2)
    N = K.shape[2]
    Kq = K.repeat_interleave(g, 1); Vq = V.repeat_interleave(g, 1)
    logits = torch.einsum("bhtd,bhnd->bhtn", q.float(), Kq.float()) * (scaling or d ** -0.5)
    mask = torch.ones(B, Hq, T, N, dtype=torch.bool, device=q.device)
    mask[..., N - T:] = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()
    if read is not None:
        M = read[0].shape[2]
        m0 = N - T - M
        mask[..., m0:m0 + M] &= read[2].expand(B, -1, -1).repeat_interleave(g, 1)[:, :, None, :]
    if st["exact_window"]:                                        # per-token window (t - W, t], as AHN
        kpos = [x[2] for x in (sink, win) if x is not None]
        kpos = np.concatenate(kpos + [st["pos"]]) if kpos else st["pos"]
        if read is None and len(kpos) == N:
            ok = torch.as_tensor(kpos[None, :] > st["pos"][:, None] - W, device=q.device)
            if sink is not None and sink[0].shape[2]:
                ok[:, :sink[0].shape[2]] = True
            mask &= ok[None, None]
    logits = logits.masked_fill(~mask, float("-inf"))
    w = torch.softmax(logits, -1)
    o = torch.einsum("bhtn,bhnd->bhtd", w, Vq.float())
    if mem is not None and hasattr(mem, "gdn_out"):
        o = o + mem.gdn_out(l, q, st)                               # AHN-GDN memory branch
    if mem is not None and hasattr(mem, "cortex_out"):
        o = o + mem.cortex_out(l, q, st["pos"])                    # additive gated cortex read (proposal 4.2)
    o = o.to(q.dtype)
    qk = st["qk"]
    if qk is not None and st["mode"] == "stream" and win is not None:
        _capture_qk(qk, eng, l, q, w, win, sink, st["pos"], g)
    pr = st["pairs"]
    if pr is not None and st["mode"] == "stream" and l in pr["layers"] and win is not None:
        cols = win[2]
        off = 0 if sink is None else sink[0].shape[2]
        ww = w[0, :, :, off:off + len(cols)].amax(0)                                      # (T, ncols) max over heads
        far = torch.as_tensor(st["pos"][:, None] - cols[None, :] > pr["min_dist"], device=ww.device)
        bw, bj = ww.masked_fill(~far, 0.0).max(-1)
        bw, bj = bw.cpu().numpy(), cols[bj.cpu().numpy()]
        better = bw > pr["w"][st["pos"]]
        pr["w"][st["pos"][better]] = bw[better]; pr["j"][st["pos"][better]] = bj[better]
    cap = st["capture"]
    if cap is not None and st["mode"] == "probe" and l in cap["layers"]:
        cols = np.concatenate([x[2] for x in (sink, win) if x is not None])           # context positions in K order
        cap["mass"][cols] += w[..., :len(cols)].sum((0, 1, 2)).cpu().numpy()
    if read is not None and st["mode"] == "stream":
        M = read[0].shape[2]
        m0 = N - T - M
        mass = w[..., m0:m0 + M].sum(2).view(B, Hkv, g, M).sum(2)        # (B, Hkv, M)
        mem.feedback(l, mass)
        if hasattr(mem, "recall_hook"):                                  # per-position read weights (B, Hq, T, M)
            mem.recall_hook(l, w[..., m0:m0 + M], st["pos"])
    if st["mode"] == "stream":
        _update_window(st, l, k, v)
    return o.transpose(1, 2).contiguous(), None


def _capture_h(module, args, kwargs):
    st = _CTX.get("st")
    if st is not None and "h" in st:
        st["h"][module.layer_idx] = kwargs.get("hidden_states", args[0] if args else None)


def _capture_qk(qk, eng, l, q, w, win, sink, pos, g):
    """Accumulate least-squares statistics for a query map q -> k: for every head and query, the earlier token more
    than min_dist back that gets its largest attention weight (if above min_w); pre-RoPE, unit norm, weighted."""
    cols = win[2]
    off = 0 if sink is None else sink[0].shape[2]
    ww = w[0, :, :, off:off + len(cols)]                                                  # (Hq, T, ncols)
    far = torch.as_tensor(pos[:, None] - cols[None, :] > qk["min_dist"], device=ww.device)
    bw, bj = ww.masked_fill(~far, 0.0).max(-1)
    hh, tt = torch.nonzero(bw > qk["min_w"], as_tuple=True)
    if len(hh) == 0:
        return
    j = bj[hh, tt]
    kk = eng.rotate_at(win[0][0, hh // g, j].float(), torch.as_tensor(cols, device=q.device)[j], inverse=True)
    qq = eng.rotate_at(q[0, hh, tt].float(), torch.as_tensor(pos, device=q.device)[tt], inverse=True)
    qq, kk, wt = torch.nn.functional.normalize(qq, dim=-1), torch.nn.functional.normalize(kk, dim=-1), bw[hh, tt]
    idx = l * q.shape[1] + hh
    qk["XtX"].index_add_(0, idx, wt[:, None, None] * qq[:, :, None] * qq[:, None, :])
    qk["XtY"].index_add_(0, idx, wt[:, None, None] * qq[:, :, None] * kk[:, None, :])
    qk["n"] += len(hh)


def _update_window(st, l, k, v):
    """Append the chunk, keep `sinks` first tokens and the last W tokens; pass evicted tokens to the memory."""
    eng, W, S = st["engine"], st["W"], st["sinks"]
    pos = st["pos"]
    if st["sink"][l] is None:
        st["sink"][l] = (k[:, :, :S], v[:, :, :S], pos[:S])
        k, v, pos = k[:, :, S:], v[:, :, S:], pos[S:]
    win = st["win"][l]
    if win is None:
        wk, wv, wp = k, v, pos
    else:
        wk, wv, wp = torch.cat([win[0], k], 2), torch.cat([win[1], v], 2), np.concatenate([win[2], pos])
    extra = wk.shape[2] - W
    if extra > 0:
        mem = st["memory"]
        if mem is not None:
            k_pre = eng.rotate(wk[0, :, :extra], wp[:extra], inverse=True)
            mem.write(l, k_pre, wv[0, :, :extra], wp[:extra], st["nll_all"] if "nll_all" in st else None)
        wk, wv, wp = wk[:, :, extra:], wv[:, :, extra:], wp[extra:]
    st["win"][l] = (wk, wv, wp)
