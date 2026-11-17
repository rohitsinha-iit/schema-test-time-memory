"""LM-quality guard for the retrofit: mean NLL beyond the window on held-out plain text (wikitext-103 validation
split; training draws filler from the train split). 8 segments of 8,192 tokens, W 1024, 128 sinks, per-token window.

  python -m pilot.eval_lm --arms swa,ahn,ahn_ft,teacher --tag r2_lm
"""
import argparse
import datetime
import glob
import json
import os

import numpy as np
import pandas as pd

from pilot.engine import Engine
from pilot.run_pilot import arm_config
from sim.sanity import git_commit

VAL = "~/.cache/huggingface/hub/datasets--Salesforce--wikitext/snapshots/*/wikitext-103-raw-v1/validation-00000-of-00001.parquet"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--n_seg", type=int, default=8)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    path = f"results/pilot_{datetime.date.today():%Y%m%d}_{a.tag}.jsonl"
    if os.path.exists(path):
        raise SystemExit(f"{path} exists")
    eng = Engine(a.model)
    text = "".join(t for t in pd.read_parquet(glob.glob(os.path.expanduser(VAL))[0])["text"] if len(t) > 200)
    ids = np.array(eng.tok(text)["input_ids"])
    segs = [ids[i * 8192:(i + 1) * 8192] for i in range(a.n_seg)]
    with open(path, "w") as f:
        for arm in a.arms.split(","):
            win, fac = arm_config(arm, W=1024)
            nll = [eng.run(s, win, memory=fac(eng) if fac else None, sinks=128, exact_window=True)["nll"][1152:].mean()
                   for s in segs]
            row = dict(arm=arm, model=a.model, n_seg=a.n_seg, nll_beyond=float(np.mean(nll)), per_seg=[float(x) for x in nll],
                       git_commit=git_commit())
            f.write(json.dumps(row) + "\n"); f.flush()
            print(f"{arm:14s} plain-text NLL beyond window {row['nll_beyond']:.4f}", flush=True)


if __name__ == "__main__":
    main()
