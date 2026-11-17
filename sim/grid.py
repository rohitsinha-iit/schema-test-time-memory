"""Stage 0a grid (P2): eta tuning on validation seeds, then the test grid with the chosen etas.

1. Validation: every arm x every eta in its grid, seeds 0 to 2, exc_rate 0.2, corr 0.3, zipf 1.0.
   The eta with the best mean `score` per arm goes to results/<stage>_<date>_eta.json.
2. Test: exc_rate {0.05, 0.2, 0.5} x corr {0.0, 0.3, 0.6} x zipf {0.5, 1.0, 1.5}, seeds 100 to 104, all arms.
Every run appends one JSON line (validation: _val.jsonl, test: _grid.jsonl); the grid CSV is written at the end.
Resumable: runs already present in a JSONL are skipped, and an existing eta.json is reused.

Run: python -m sim.grid [--date YYYYMMDD] [--workers N]
"""
import argparse
import datetime
import inspect
import itertools
import json
import os
import time
from multiprocessing import Pool

import pandas as pd

from .run import arms, eta_grid_for, run_one
from .sanity import git_commit
from .streams import make_stream

B = 4
VAL_SEEDS, TEST_SEEDS = [0, 1, 2], [100, 101, 102, 103, 104]
VAL_CELL = dict(exc_rate=0.2, corr=0.3, zipf=1.0)
EXC_RATES, CORRS, ZIPFS = [0.05, 0.2, 0.5], [0.0, 0.3, 0.6], [0.5, 1.0, 1.5]
KEY = ("arm", "seed", "eta", "exc_rate", "corr", "zipf")


def stream_kwargs(**cell):
    defaults = {k: p.default for k, p in inspect.signature(make_stream).parameters.items() if k != "seed"}
    return {**defaults, "b": B, **cell}


def _key(r):
    return tuple(r[k] for k in KEY)


def _load(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _work(task):
    arm, cfg, kw, seed, eta = task
    t0 = time.time()
    try:
        res = run_one(arm, cfg, kw, seed, eta=eta)
    except Exception as e:  # recorded, not fatal, so one bad run does not stop the grid
        res = dict(arm=arm, seed=seed, eta=eta, **kw, error=repr(e))
    res["wall_s"] = round(time.time() - t0, 3)
    return res


def run_tasks(tasks, path, workers, commit):
    done = {_key(r) for r in _load(path) if "error" not in r}
    todo = [t for t in tasks if (t[0], t[3], t[4], t[2]["exc_rate"], t[2]["corr"], t[2]["zipf"]) not in done]
    print(f"{path}: {len(tasks)} runs, {len(tasks) - len(todo)} already done, {len(todo)} to run", flush=True)
    t0, n_fail = time.time(), 0
    with Pool(workers) as pool, open(path, "a") as f:
        for i, res in enumerate(pool.imap_unordered(_work, todo, chunksize=1), 1):
            res["git_commit"] = commit
            n_fail += "error" in res
            f.write(json.dumps(res) + "\n"); f.flush()
            if i % 50 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}  {time.time() - t0:.0f}s  failures {n_fail}", flush=True)
    return n_fail, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="stage0a")
    ap.add_argument("--date", default=f"{datetime.date.today():%Y%m%d}")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    a = ap.parse_args()
    base = f"results/{a.stage}_{a.date}"
    os.makedirs("results", exist_ok=True)
    commit = git_commit()
    A = arms(B)

    # 1. validation tuning
    eta_path = base + "_eta.json"
    if os.path.exists(eta_path):
        with open(eta_path) as f:
            etas = json.load(f)
        print(f"reusing {eta_path}")
    else:
        kw = stream_kwargs(**VAL_CELL)
        tasks = [(arm, cfg, kw, s, eta) for arm, cfg in A.items() for eta in eta_grid_for(cfg) for s in VAL_SEEDS]
        n_fail, wall = run_tasks(tasks, base + "_val.jsonl", a.workers, commit)
        print(f"validation: failures {n_fail}, wall {wall:.0f}s")
        val = pd.DataFrame([r for r in _load(base + "_val.jsonl") if "error" not in r])
        mean = val.groupby(["arm", "eta"])["score"].mean().reset_index()
        etas = {arm: float(g.loc[g["score"].idxmax(), "eta"]) for arm, g in mean.groupby("arm")}
        mean.to_csv(base + "_val_scores.csv", index=False)
        with open(eta_path, "w") as f:
            json.dump(etas, f, indent=1)
    print("chosen etas:", json.dumps(etas))

    # 2. test grid
    tasks = [(arm, cfg, stream_kwargs(exc_rate=e, corr=c, zipf=z), s, etas[arm])
             for arm, cfg in A.items() for e, c, z in itertools.product(EXC_RATES, CORRS, ZIPFS) for s in TEST_SEEDS]
    n_fail, wall = run_tasks(tasks, base + "_grid.jsonl", a.workers, commit)
    print(f"test grid: failures {n_fail}, wall {wall:.0f}s")
    rows = _load(base + "_grid.jsonl")
    csv = base + "_grid.csv"
    if os.path.exists(csv):
        print(f"{csv} exists; not overwritten")
    else:
        pd.DataFrame(rows).to_csv(csv, index=False)
    print(f"grid rows {len(rows)}, with error {sum('error' in r for r in rows)}")


if __name__ == "__main__":
    main()
