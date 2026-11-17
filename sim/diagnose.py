"""Diagnostic trace of the hippocampus H for one arm on one stream.

Every `every` chunks it logs H occupancy, exception vs rule landmark counts, their
median age, reuse r and cortical error delta, and the stats counters.
Ground-truth labels are read ONLY here, to classify landmarks; the memory never sees them.

Run: python -m sim.diagnose --arm schema --seed 100 --exc_rate 0.2 --corr 0.6 --out results/x.csv
"""
import argparse
import os

import numpy as np
import pandas as pd

from .memories import MemorySystem
from .run import arms
from .streams import make_stream


def _labels(ms, S):
    """Kind (0 rule, 1 exception) of each used H slot, found by matching its key to the stream."""
    idx = ms.H.idx()
    t_in = ms.H.t_in[idx].astype(int)
    match = np.einsum("nd,nbd->nb", ms.H.k[idx], S["keys"][t_in])   # (n, b) cosines within the source chunk
    return idx, S["kinds"][t_in, match.argmax(axis=1)]


def _med(x):
    return float(np.median(x)) if len(x) else np.nan


def diagnose(arm, cfg, stream_kw, seed, every=300):
    S = make_stream(seed, **stream_kw)
    ms = MemorySystem(S["d"], S["b"], cfg, seed=seed)
    rows = []
    for t in range(S["T"]):
        ms.observe(S["keys"][t], S["vals"][t], t)
        if (t + 1) % every:
            continue
        idx, kind = _labels(ms, S)
        age, r = t - ms.H.t_in[idx], ms.H.r[idx]
        dl = ms.delta(idx) if len(idx) else np.zeros(0)
        row = dict(arm=arm, seed=seed, t=t, occupancy=len(idx) / ms.H.N)
        for name, m in [("exc", kind == 1), ("rule", kind == 0)]:
            row[f"n_{name}"] = int(m.sum())
            row[f"med_age_{name}"] = _med(age[m])
            row[f"med_r_{name}"] = _med(r[m])
            row[f"med_delta_{name}"] = _med(dl[m])
        row.update(cap_evictions=ms.H.n_capacity_evictions, superseded=ms.H.n_superseded, **ms.stats)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="schema")
    ap.add_argument("--b", type=int, default=4)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--exc_rate", type=float, default=0.2)
    ap.add_argument("--corr", type=float, default=0.6)
    ap.add_argument("--every", type=int, default=300)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists; results files are never overwritten")
    df = diagnose(a.arm, arms(a.b)[a.arm], dict(b=a.b, exc_rate=a.exc_rate, corr=a.corr), a.seed, a.every)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df.to_csv(a.out, index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
