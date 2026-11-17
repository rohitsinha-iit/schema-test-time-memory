"""Diagnostic (validation seed 2): can a linear d x d cortex store the hippocampus's (key, value) pairs at all?
Best-case in-sample error of a ridge least-squares W on n unit-norm items, with raw and with mean-centred keys,
plus the mean pairwise cosine of the keys (anisotropy). No labels used."""
import numpy as np
import torch
import torch.nn.functional as F

from pilot.engine import Engine
from pilot.memory import Hippocampus
from pilot.run_pilot import STREAM, W
from pilot.text_stream import load_filler, make_stream

eng = Engine("Qwen/Qwen2.5-1.5B-Instruct")
s = make_stream(np.random.default_rng([2, 500]), eng.tok, load_filler(eng.tok), **dict(STREAM, exc_rate=0.5))
h = Hippocampus(eng, cap=10**6)
eng.run(s["tokens"], W, memory=h)
for l in (4, 14, 24):
    for hd in range(eng.Hkv):
        K, V = h.k[l][hd].float(), F.normalize(h.v[l][hd].float(), dim=-1)
        Ku = F.normalize(K, dim=-1)
        cosm = (Ku @ Ku.T).mean().item()
        Kc = F.normalize(K - K.mean(0), dim=-1)
        row = []
        for n in (32, 64, 128, 256, 512, 1024):
            for name, KK in (("raw", Ku), ("centred", Kc)):
                X, Y = KK[:n], V[:n]
                Wls = torch.linalg.solve(X.T @ X + 1e-3 * torch.eye(X.shape[1], device=X.device), X.T @ Y)
                row.append(f"{name} n{n} {((X @ Wls - Y) ** 2).sum(-1).mean().item():.2f}")
        print(f"layer {l} head {hd}: mean key cos {cosm:.2f} | " + "  ".join(row), flush=True)
