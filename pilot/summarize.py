"""Summarize pilot test runs: per-arm means over streams, and per-stream paired differences of schema vs each arm.
Primary score per stream (spec section 7): 0.5 * retest rule accuracy (C only) + 0.5 * exception accuracy beyond W.

  python -m pilot.summarize results/pilot_20261104_test_a.jsonl results/pilot_20261104_test_b.jsonl --out results/pilot_20261104_test_summary.csv
  python -m pilot.summarize <files> --out <csv> --ref lora_schema    (paired differences against another arm)
"""
import argparse
import json

import numpy as np
import pandas as pd

COLS = ["heldout_text_nll", "rule_acc", "rule_drifted_acc", "exc_acc_beyond", "exc_acc_gt4W", "retest_rule_acc", "retest_exc_acc_beyond"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", default="schema", help="reference arm for the paired differences")
    a = ap.parse_args()
    df = pd.DataFrame([json.loads(l) for f in a.files for l in open(f)])
    df["primary"] = 0.5 * df["retest_rule_acc"] + 0.5 * df["exc_acc_beyond"]
    order = list(dict.fromkeys(df["arm"]))
    rows = []
    for ex in sorted(df["exc_rate"].unique()) + ["all"]:
        d = df if ex == "all" else df[df["exc_rate"] == ex]
        for arm in order:
            x = d[d["arm"] == arm]
            row = dict(exc_rate=ex, arm=arm, n_streams=len(x), **{c: x[c].mean() for c in COLS + ["primary"] if c in x})
            s = d[d["arm"] == a.ref].set_index(["seed", "exc_rate"])["primary"]
            o = x.set_index(["seed", "exc_rate"])["primary"]
            if arm != a.ref and o.notna().any():
                diff = (s - o).dropna()
                row.update(schema_minus_arm=diff.mean(), schema_better_frac=(diff > 0).mean(), schema_tie_frac=(diff == 0).mean())
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
