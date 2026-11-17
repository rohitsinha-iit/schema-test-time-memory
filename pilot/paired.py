"""Paired comparison of a reference arm (--ref, default lora_schema) against every other arm (95% bootstrap CI over streams) plus replay and H internals.

  python pilot/paired.py [--ref ARM] results/<test files>.jsonl > results/<tag>_paired.txt
"""
import json, sys
import numpy as np
import pandas as pd

args = sys.argv[1:]
REF = "lora_schema"
if args[:1] == ["--ref"]:
    REF, args = args[1], args[2:]
df = pd.DataFrame([json.loads(l) for f in args for l in open(f)])
dup = df.duplicated(["arm", "seed", "exc_rate"]).sum()
print("rows", len(df), "duplicate (arm,seed,exc) rows", dup)
print(df.groupby("arm").size().to_string())
df["primary"] = 0.5 * df["retest_rule_acc"] + 0.5 * df["exc_acc_beyond"]
ref = df[df.arm == REF].set_index(["seed", "exc_rate"])["primary"]
rng = np.random.default_rng(0)
print(f"\npaired {REF} - arm (primary), 95% bootstrap CI over streams, wins/ties/losses")
for arm in df.arm.unique():
    if arm == REF:
        continue
    o = df[df.arm == arm].set_index(["seed", "exc_rate"])["primary"]
    d = (ref - o).dropna().values
    if len(d) == 0:
        continue
    bs = [rng.choice(d, len(d)).mean() for _ in range(10000)]
    print(f"{arm:20s} n {len(d):3d} diff {d.mean():+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]  "
          f"W/T/L {(d > 0).sum()}/{(d == 0).sum()}/{(d < 0).sum()}")
cols = ["lora_steps", "steps_tried", "steps_accepted", "steps_rejected", "lora_rejected", "released", "H_landmarks",
        "sleeps", "sleeps_without_candidates", "anchor_episodes", "heldout_text_nll", "wall_s"]
print("\nmeans over streams")
with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.2f}".format):
    print(df[df.arm.str.startswith(("lora", "hrd")) | (df.arm == "hippo")].groupby("arm")[[c for c in cols if c in df]].mean().to_string())
