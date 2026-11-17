"""Single-seed sanity run of every arm on one stream (default eta, no tuning).

Writes results/<stage>_<YYYYMMDD>_<tag>.jsonl (one full record per run) and a .csv with the summary columns.
Run: python -m sim.sanity --seed 100 --exc_rate 0.2 --corr 0.6
"""
import argparse
import datetime
import inspect
import json
import os
import subprocess
import time

import pandas as pd

from .run import arms, run_one
from .streams import make_stream

COLS = ["arm", "rule", "rule_stable", "rule_drifted", "rule_drifted_old", "exc_recent", "exc_mid", "exc_old",
        "cortex_rule", "cortex_rule_drifted", "anchor_loss", "steps_tried", "steps_shrunk", "superseded",
        "anchors_invalidated", "forced_evictions", "wall_s"]


def git_commit():
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", type=int, default=4)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--exc_rate", type=float, default=0.2)
    ap.add_argument("--corr", type=float, default=0.6)
    ap.add_argument("--stage", default="stage0a")
    ap.add_argument("--tag", default="sanity")
    a = ap.parse_args()
    base = f"results/{a.stage}_{datetime.date.today():%Y%m%d}_{a.tag}"
    for ext in (".jsonl", ".csv"):
        if os.path.exists(base + ext):
            raise SystemExit(f"{base + ext} exists; results files are never overwritten")
    os.makedirs("results", exist_ok=True)
    defaults = {k: p.default for k, p in inspect.signature(make_stream).parameters.items() if k != "seed"}
    stream_kw = {**defaults, "b": a.b, "exc_rate": a.exc_rate, "corr": a.corr}
    commit = git_commit()
    rows = []
    with open(base + ".jsonl", "a") as f:
        for arm, cfg in arms(a.b).items():
            t0 = time.time()
            res = run_one(arm, cfg, stream_kw, a.seed)
            res.update(git_commit=commit, wall_s=round(time.time() - t0, 2))
            f.write(json.dumps(res) + "\n"); f.flush()
            rows.append(res)
            print(f"{arm}: {res['wall_s']}s", flush=True)
    df = pd.DataFrame(rows).reindex(columns=COLS)
    df.to_csv(base + ".csv", index=False)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
