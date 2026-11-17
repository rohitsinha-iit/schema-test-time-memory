"""Fit a fact-level episode key kappa (proposal 4.1) by self-distillation from the full-attention teacher; no labels.

When the teacher reads a stream, a token that refers back to earlier content (a recurring fact restated, a person
mentioned again) puts attention on that earlier token far outside any local window. Pairs (t, j) with
t - j > 64 and the strongest such attention (top `frac` per stream, upper-half layers) are positives. A linear map
P of the layer-L/2 hidden state is trained with a symmetric InfoNCE loss so that kappa = normalize(P h) matches
across positive pairs and not across other tokens in the batch.

Reuse under kappa = number of later landmarks (selector-chosen tokens) with cos(kappa_i, kappa_j) >= c, the
Stage 0a cue-match tag. Labels are read only to report the AUC of recurring-fact vs exception landmarks.

  python -m pilot.fit_kappa
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from pilot.diag_reuse import auc
from pilot.engine import Engine
from pilot.fit_selector import selector_path
from pilot.run_pilot import STREAM
from pilot.text_stream import load_filler, make_stream


def kappa_path(model):
    return os.path.join(os.path.dirname(__file__), "data", f"kappa_{model.split('/')[-1]}.pt")


def landmarks(feat, sel, m=16):
    score = (((feat - sel["mu"]) / sel["sd"]) @ sel["w"] + sel["b"]).cpu().numpy()
    return np.concatenate([np.arange(c, min(c + 64, len(score)))[np.argsort(-score[c:c + 64])[:m]]
                           for c in range(0, len(score), 64)])


def reuse_auc(kap, lm, labels, cs=(0.5, 0.7, 0.8, 0.9)):
    """AUC (rule > exception) of later-landmark cue matches, per threshold, and of the max later cosine."""
    k = F.normalize(kap[torch.as_tensor(lm, device=kap.device)], dim=-1)
    cos = k @ k.T
    later = torch.triu(torch.ones_like(cos, dtype=torch.bool), 1)
    lab = labels[lm]; fact = lab != 0; rule = lab[fact] == 1
    out = {f"cue c{c}": auc(((cos >= c) & later).sum(1).cpu().numpy()[fact], rule) for c in cs}
    out["max later cos"] = auc(cos.masked_fill(~later, -1).amax(1).cpu().numpy()[fact], rule)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--frac", type=float, default=0.1)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--steps", type=int, default=1500)
    a = ap.parse_args()
    eng = Engine(a.model)
    filler = load_filler(eng.tok)
    sel = {k: v.to(eng.dev).float() if torch.is_tensor(v) else v for k, v in torch.load(selector_path(a.model)).items()}
    layer = sel["layer"]

    def collect(seed, ex):
        s = make_stream(np.random.default_rng([seed, int(ex * 1000)]), eng.tok, filler, **dict(STREAM, exc_rate=ex))
        out = eng.run(s["tokens"], 10**9, feat_layer=layer, pair_layers=range(eng.L // 2, eng.L))
        f = (out["feat"].float() - sel["mu"]) / sel["sd"]
        return s, f, out["pair_j"], out["pair_w"], landmarks(out["feat"].float(), sel)

    train = [collect(sd, ex) for sd in (0, 1) for ex in (0.2, 0.5)]
    held = [collect(2, ex) for ex in (0.2, 0.5)]
    A, B = [], []
    for s, f, pj, pw, lm in train:
        ok = pj >= 0
        t = np.flatnonzero(ok & (pw >= np.quantile(pw[ok], 1 - a.frac)))
        A.append(f[t]); B.append(f[pj[t]])
    A, B = torch.cat(A), torch.cat(B)
    print(f"positive pairs {len(A)}")
    torch.manual_seed(0)
    P = torch.nn.Linear(A.shape[1], a.dim, bias=False).to(eng.dev)
    opt = torch.optim.Adam(P.parameters(), lr=1e-3, weight_decay=1e-4)
    for step in range(a.steps):
        i = torch.randint(len(A), (512,), device=eng.dev)
        za, zb = F.normalize(P(A[i]), dim=-1), F.normalize(P(B[i]), dim=-1)
        logits = za @ zb.T / 0.1
        y = torch.arange(len(i), device=eng.dev)
        loss = 0.5 * (F.cross_entropy(logits, y) + F.cross_entropy(logits.T, y))
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 500 == 0:
            print(f"step {step + 1} infonce {loss.item():.3f}", flush=True)
    torch.save(dict(layer=layer, W=P.weight.detach().cpu(), frac=a.frac), kappa_path(a.model))

    with torch.no_grad():
        for name, data in (("fit seeds 0-1", train), ("held-out seed 2", held)):
            res = {}
            for s, f, pj, pw, lm in data:
                for kind, kap in (("raw h", f), ("kappa", P(f))):
                    for k, v in reuse_auc(kap, lm, s["labels"]).items():
                        res.setdefault(f"{kind:6s} {k}", []).append(v)
            for k, v in res.items():
                print(f"{name:16s} {k:22s} AUC rule>exc {np.nanmean(v):.3f}  ({', '.join(f'{x:.2f}' for x in v)})")


if __name__ == "__main__":
    main()
