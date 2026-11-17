"""Fit the hippocampus landmark selector by self-distillation from the full-attention teacher (stand-in for the
proposal's learned attention pooling; no labels are used to fit it).

Target: on calibration streams (validation seeds only), the attention mass that the teacher's question tokens give
each stream token, summed over the upper half of the layers; a token is positive if it is in the top `frac` of
its stream (attention sinks excluded). Features: the token's hidden state after layer L/2 (standardized).
Model: logistic regression. Labels are read only to report coverage on a held-out validation stream.

  python -m pilot.fit_selector --model Qwen/Qwen2.5-1.5B-Instruct
"""
import argparse
import os

import numpy as np
import torch

from pilot.engine import Engine
from pilot.run_pilot import STREAM
from pilot.text_stream import load_filler, make_stream


def selector_path(model):
    return os.path.join(os.path.dirname(__file__), "data", f"selector_{model.split('/')[-1]}.pt")


def collect(eng, filler, seed, ex, layer):
    s = make_stream(np.random.default_rng([seed, int(ex * 1000)]), eng.tok, filler, **dict(STREAM, exc_rate=ex))
    out = eng.run(s["tokens"], 10**9, probes=s["probes"], feat_layer=layer, capture_layers=range(eng.L // 2, eng.L))
    return s, out["feat"].float(), out["probe_mass"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--frac", type=float, default=0.15)
    ap.add_argument("--m", type=int, default=16)
    a = ap.parse_args()
    eng = Engine(a.model)
    filler = load_filler(eng.tok)
    layer = eng.L // 2
    train = [collect(eng, filler, sd, ex, layer) for sd in (0, 1) for ex in (0.2, 0.5)]
    X = torch.cat([f[4:] for _, f, _ in train])
    y = torch.cat([torch.as_tensor(m[4:] >= np.quantile(m[4:], 1 - a.frac), dtype=torch.float32) for _, _, m in train]).cuda()
    mu, sd = X.mean(0), X.std(0) + 1e-4
    Xs = (X - mu) / sd
    w = torch.zeros(X.shape[1], device="cuda", requires_grad=True)
    b = torch.zeros(1, device="cuda", requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=1e-2)
    for it in range(500):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(Xs @ w + b, y) + 1e-3 * (w ** 2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"train loss {loss.item():.4f}  positives {y.mean().item():.3f}")
    sel = dict(layer=layer, mu=mu.cpu(), sd=sd.cpu(), w=w.detach().cpu(), b=b.detach().cpu(), frac=a.frac)
    torch.save(sel, selector_path(a.model))
    for ex in (0.2, 0.5):                                   # held-out validation stream: coverage (labels: report only)
        s, f, m = collect(eng, filler, 2, ex, layer)
        score = (((f - mu) / sd) @ w.detach() + b.detach()).cpu().numpy()
        keep = np.concatenate([np.arange(c, min(c + 64, len(score)))[np.argsort(-score[c:c + 64])[: a.m]]
                               for c in range(0, len(score), 64)])
        fact = s["labels"] != 0
        vals = np.zeros(len(fact), bool)
        for key, val, pos, kind in s["facts"]:
            vals[pos:pos + len(eng.tok(val, add_special_tokens=False)["input_ids"])] = True
        top = m >= np.quantile(m[4:], 1 - a.frac)
        print(f"held-out seed 2 exc {ex}: keep {len(keep)/len(fact):.2f}  fact kept {np.isin(np.flatnonzero(fact), keep).mean():.2f}  "
              f"value kept {np.isin(np.flatnonzero(vals), keep).mean():.2f}  precision {fact[keep].mean():.2f}  "
              f"teacher-top kept {np.isin(np.flatnonzero(top), keep).mean():.2f}")


if __name__ == "__main__":
    main()
