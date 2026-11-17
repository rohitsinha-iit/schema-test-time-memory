"""Fit the cortex query map A_{l,h} (q -> k, pre-RoPE, unit norm) for every layer and query head by weighted ridge
regression on the full-attention teacher's own attention: for each query, the token more than 64 positions back
that it attends to most (weight > min_w). Validation seeds 0 to 1 only; no labels.

  python -m pilot.fit_qmap
"""
import argparse

import numpy as np
import torch

from pilot.engine import Engine
from pilot.linear_cortex import qmap_path
from pilot.run_pilot import STREAM
from pilot.text_stream import load_filler, make_stream


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--min_w", type=float, default=0.05)
    ap.add_argument("--ridge", type=float, default=0.1)
    a = ap.parse_args()
    eng = Engine(a.model)
    filler = load_filler(eng.tok)
    n, d = eng.L * eng.Hq, eng.d
    qk = dict(XtX=torch.zeros(n, d, d, device=eng.dev), XtY=torch.zeros(n, d, d, device=eng.dev), n=0,
              min_dist=64, min_w=a.min_w)
    for seed in (0, 1):
        for ex in (0.2, 0.5):
            s = make_stream(np.random.default_rng([seed, int(ex * 1000)]), eng.tok, filler, **dict(STREAM, exc_rate=ex))
            eng.run(s["tokens"], 10**9, qk=qk)
            print(f"seed {seed} exc {ex}: pairs so far {qk['n']}", flush=True)
    XtX, XtY = qk["XtX"], qk["XtY"]
    lam = a.ridge * XtX.diagonal(dim1=1, dim2=2).mean(-1)[:, None, None] + 1e-6
    A = torch.linalg.solve(XtX + lam * torch.eye(d, device=eng.dev), XtY)                # k ~ q A
    # weighted mean cosine between q A and k on the fitting pairs (from the sufficient statistics)
    num = torch.einsum("nde,nde->n", A, XtY)
    print(f"mean fit trace ratio {float((num / XtX.diagonal(dim1=1, dim2=2).sum(-1).clamp_min(1e-6)).mean()):.3f}")
    torch.save(A.cpu(), qmap_path(a.model))


if __name__ == "__main__":
    main()
