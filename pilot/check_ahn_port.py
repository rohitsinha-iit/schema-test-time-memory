"""R0 equivalence: our engine + AHNMemory against AHN's own code (pilot/ref_ahn.py) on the same tokens.

  python -m pilot.check_ahn_port --ref ref_nll.npy --tokens toks.npy   (after ref_ahn.py has written ref_nll.npy)
  python -m pilot.check_ahn_port --make_tokens toks.npy                 (writes a test input: filler + facts)
"""
import argparse

import numpy as np

from pilot.ahn_port import AHNMemory
from pilot.engine import Engine
from pilot.run_pilot import STREAM
from pilot.text_stream import load_filler, make_stream

MODEL = "Qwen/Qwen2.5-3B-Instruct"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--make_tokens", default="")
    ap.add_argument("--tokens", default="")
    ap.add_argument("--ref", default="")
    ap.add_argument("--W", type=int, default=1024)
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--fp32", action="store_true")
    a = ap.parse_args()
    import torch
    eng = Engine(MODEL, dtype=torch.float32 if a.fp32 else torch.bfloat16)
    if a.make_tokens:
        s = make_stream(np.random.default_rng(7), eng.tok, load_filler(eng.tok), **dict(STREAM, n_tokens=a.n, exc_rate=0.5))
        np.save(a.make_tokens, s["tokens"])
        print("wrote", a.make_tokens)
        return
    toks = np.load(a.tokens)
    ours = eng.run(toks, a.W, memory=AHNMemory(eng))["nll"][1:]
    swa = eng.run(toks, a.W, sinks=128, exact_window=True)["nll"][1:]
    full = eng.run(toks, 10**9)["nll"][1:]
    ref = np.load(a.ref)
    beyond = np.arange(1, len(toks)) >= a.W + 128
    shift = eng.run(toks, a.W, memory=AHNMemory(eng, lag_extra=1))["nll"][1:]
    nogate = eng.run(toks, a.W, memory=AHNMemory(eng, no_gate=True))["nll"][1:]
    for name, x in (("ours (AHN port)", ours), ("port, lag + 1", shift), ("port, no out gate", nogate),
                    ("window only", swa), ("full attention", full)):
        d = np.abs(x - ref)
        print(f"{name:16s} |nll - ref| mean {d.mean():.4f}  beyond window {d[beyond].mean():.4f}  max {d.max():.3f}  "
              f"corr {np.corrcoef(x, ref)[0, 1]:.5f}  mean nll beyond {x[beyond].mean():.3f} (ref {ref[beyond].mean():.3f})")


if __name__ == "__main__":
    main()
