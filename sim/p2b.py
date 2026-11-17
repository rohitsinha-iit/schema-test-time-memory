"""Stage 0a P2b, E1 and E2 (16 Nov 2026; see stage0a_m1_decision.md and the 16 Nov plan).

P2b: extended eta grids with full re-tuning on validation seeds 0-2; gate attribution arms; anchor budgets.
E1: late-onset rules (a quarter of the rules start between T/4 and 3T/5; their first occurrence is a one-off).
E2: capacity sweep of the cortex (key/value dimension d in 32, 64, 128).
Run: python -m sim.p2b --workers 30
"""
import argparse, json, os, itertools
import pandas as pd
from .grid import run_tasks, _load, stream_kwargs, VAL_SEEDS, TEST_SEEDS, VAL_CELL, EXC_RATES, CORRS, ZIPFS, B
from .run import arms
from .sanity import git_commit

GRID = {"arrival_write": [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0], "sleep": [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]}
FIXED = {"schema_eta05": 0.5, "schema_nogate_eta05": 0.5}
CORE = ["arrival_fifo", "surprise_fifo", "sleep_all", "uniform_replay", "schema", "schema_surprise_pri", "schema_nogate"]


def p2b_arms():
    A = arms(B)
    s = A["schema"]
    A.update({
        "schema_agem_only": {**s, "gate": False, "agem_only": True},
        "schema_lookahead_only": {**s, "damage_norm": "sum", "rho": 4.0, "agem": False},
        "schema_R64": {**s, "R": 64, "H_slots": 64 + (256 - 64) // B},
        "schema_R128": {**s, "R": 128, "H_slots": 64 + (256 - 128) // B},
        "schema_eta05": dict(s), "schema_nogate_eta05": {**s, "gate": False},
    })
    for name, cfg in A.items():                       # equal bytes: slots x b + anchors = 64 x 4 + 256
        assert cfg.get("H_slots", 0) * B + cfg.get("R", 0) == 64 * B + 256, name
    return A


def grid_for(cfg):
    return GRID["arrival_write"] if cfg.get("arrival_write") else GRID["sleep"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="20261116")
    ap.add_argument("--workers", type=int, default=30)
    a = ap.parse_args()
    base = f"results/stage0a_{a.date}_p2b"
    commit = git_commit(); A = p2b_arms()
    eta_path = base + "_eta.json"
    if os.path.exists(eta_path):
        etas = json.load(open(eta_path))
    else:
        kw = stream_kwargs(**VAL_CELL)
        tasks = [(arm, cfg, kw, s, eta) for arm, cfg in A.items() if arm not in FIXED for eta in grid_for(cfg) for s in VAL_SEEDS]
        run_tasks(tasks, base + "_val.jsonl", a.workers, commit)
        val = pd.DataFrame([r for r in _load(base + "_val.jsonl") if "error" not in r])
        mean = val.groupby(["arm", "eta"])["score"].mean().reset_index()
        mean.to_csv(base + "_val_scores.csv", index=False)
        etas = {arm: float(g.loc[g["score"].idxmax(), "eta"]) for arm, g in mean.groupby("arm")}
        etas.update(FIXED)
        json.dump(etas, open(eta_path, "w"), indent=1)
    print("chosen etas:", json.dumps(etas), flush=True)
    tasks = [(arm, cfg, stream_kwargs(exc_rate=e, corr=c, zipf=z), s, etas[arm])
             for arm, cfg in A.items() for e, c, z in itertools.product(EXC_RATES, CORRS, ZIPFS) for s in TEST_SEEDS]
    run_tasks(tasks, base + "_grid.jsonl", a.workers, commit)
    seeds10 = list(range(100, 110))
    tasks = [(arm, A[arm], stream_kwargs(exc_rate=e, corr=0.3, zipf=1.0, late_frac=0.25, drift_frac=0.0), s, etas[arm])
             for arm in CORE for e in EXC_RATES for s in seeds10]
    run_tasks(tasks, f"results/stage0a_{a.date}_e1.jsonl", a.workers, commit)
    for d in (32, 64, 128):
        tasks = [(arm, A[arm], stream_kwargs(exc_rate=e, corr=0.3, zipf=1.0, d=d), s, etas[arm])
                 for arm in CORE for e in (0.2, 0.5) for s in seeds10]
        run_tasks(tasks, f"results/stage0a_{a.date}_e2_d{d}.jsonl", a.workers, commit)
    print("done", flush=True)


if __name__ == "__main__":
    main()
