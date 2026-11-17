"""Per (arm, exc_rate, corr, zipf) mean and std over seeds of the Stage 0a grid.

Run: python -m sim.summarize results/stage0a_<date>_grid.jsonl
Writes results/stage0a_<date>_summary.csv next to the input.
"""
import argparse
import json
import os

import pandas as pd

GROUP = ["arm", "exc_rate", "corr", "zipf"]
METRICS = ["rule", "cortex_rule", "rule_drifted", "rule_drifted_old", "exc_recent", "exc_mid", "exc_old",
           "anchor_loss",
           # pre-registered breakdowns (stage0a_sanity_review.md, section 3)
           "cortex_rule_drifted", "cortex_rule_freq_hi", "cortex_rule_freq_mid", "cortex_rule_freq_lo"]


def summarize(df):
    g = df.groupby(GROUP)
    out = g[METRICS].agg(["mean", "std"])
    out.columns = [f"{m}_{s}" for m, s in out.columns]
    out.insert(0, "n_seeds", g.size())
    return out.reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("grid")
    a = ap.parse_args()
    with open(a.grid) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    df = pd.DataFrame([r for r in rows if "error" not in r])
    out_path = a.grid.replace("_grid.jsonl", "_summary.csv")
    if os.path.exists(out_path):
        raise SystemExit(f"{out_path} exists; results files are never overwritten")
    summarize(df).to_csv(out_path, index=False)
    print(f"wrote {out_path}: {len(df)} runs")


if __name__ == "__main__":
    main()
