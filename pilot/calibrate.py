"""Hyperparameter calibration on validation seeds 0 to 2 (hard rule 4), one JSON line per (config, stream).
Score per stream (spec section 7, primary): 0.5 * retest rule accuracy (C only) + 0.5 * exception accuracy beyond W.

  python -m pilot.calibrate --arms schema --grid '{"theta": [0.95, 0.98], "r_min": [0, 7, 12], "tau": [0.5, 0.7]}' --tag cal_schema
"""
import argparse
import datetime
import itertools
import json
import os

import numpy as np

from pilot.engine import Engine
from pilot.run_pilot import run_rows
from sim.sanity import git_commit


def score(r):
    return 0.5 * r["retest_rule_acc"] + 0.5 * r["exc_acc_beyond"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True)
    ap.add_argument("--grid", required=True)
    ap.add_argument("--base", default="{}")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    path = f"results/pilot_{datetime.date.today():%Y%m%d}_{a.tag}.jsonl"
    if os.path.exists(path):
        raise SystemExit(f"{path} exists")
    grid = json.loads(a.grid)
    eng, commit = Engine(a.model), git_commit()
    summary = []
    with open(path, "w") as f:
        for vals in itertools.product(*grid.values()):
            hp = {**json.loads(a.base), **dict(zip(grid, vals))}
            for arm in a.arms.split(","):
                rows = list(run_rows(eng, [arm], [0, 1, 2], [0.2, 0.5], hp, True, a.model, commit))
                for r in rows:
                    f.write(json.dumps(r) + "\n")
                f.flush()
                sc = float(np.mean([score(r) for r in rows]))
                summary.append((arm, hp, sc))
                print(f"{arm:20s} {hp} score {sc:.3f}  retest_rule {np.mean([r['retest_rule_acc'] for r in rows]):.3f}  "
                      f"exc_beyond {np.mean([r['exc_acc_beyond'] for r in rows]):.3f}  retest_exc "
                      f"{np.mean([r['retest_exc_acc_beyond'] for r in rows]):.3f}  rule {np.mean([r['rule_acc'] for r in rows]):.3f}", flush=True)
    for arm in a.arms.split(","):
        best = max((s for s in summary if s[0] == arm), key=lambda s: s[2])
        print(f"BEST {arm}: {json.dumps(best[1])} score {best[2]:.3f}")


if __name__ == "__main__":
    main()
