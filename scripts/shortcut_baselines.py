"""Key-free and key-based shortcut strategies on Stage 0b training loss positions (stage0b_spec.md, section 11).

For every loss position (a repeated fact value or a probe answer), the target is predicted from the tokens before
it only:
  most_frequent_value  most frequent value token so far in the sequence (ties: the most recent of them)
  most_recent_value    the last value token before the position
  same_key_last_128    value of the same key's latest earlier occurrence, if that value is within 128 tokens
  same_key_anywhere    value of the same key's latest earlier occurrence, anywhere
Only facts count as occurrences (probe answers are not counted as seen values or key occurrences); this
convention reproduces the section 11 table exactly. Uses the unchanged training generator.

Run: python scripts/shortcut_baselines.py [--n 48 --seed 7]
"""
import argparse
import datetime
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim0b.stream import F0, K0, QRY, V0, L_PROBE, make_sequence  # noqa: E402


def is_key(t):
    return K0 <= t < V0


def is_val(t):
    return V0 <= t < F0


def strategies(tokens, loss_mask, window=128):
    """Yield {strategy: correct} per loss position."""
    counts, last_seen, recent, key_val = Counter(), {}, None, {}
    for p, t in enumerate(tokens):
        if loss_mask[p]:
            key = (tokens[p - 2], tokens[p - 1])
            mf = max(counts.items(), key=lambda kv: (kv[1], last_seen[kv[0]]))[0] if counts else None
            kv = key_val.get(key)
            yield dict(most_frequent_value=mf == t, most_recent_value=recent == t,
                       same_key_last_128=kv is not None and p - kv[1] <= window and kv[0] == t,
                       same_key_anywhere=kv is not None and kv[0] == t)
        if is_val(t) and p >= 3 and is_key(tokens[p - 2]) and is_key(tokens[p - 1]) and tokens[p - 3] != QRY:
            counts[t] += 1; last_seen[t] = p; recent = t                     # a fact value
            key_val[(tokens[p - 2], tokens[p - 1])] = (t, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n_tokens", type=int, default=4096)
    ap.add_argument("--date", default=f"{datetime.date.today():%Y%m%d}")
    a = ap.parse_args()
    out = f"results/stage0b_{a.date}_shortcuts.csv"
    if os.path.exists(out):
        raise SystemExit(f"{out} exists; results files are never overwritten")
    # same draws as make_batch: exc_rate per sequence, then the sequence
    rng = np.random.default_rng(a.seed)
    rows, n_probe = [], 0
    for _ in range(a.n):
        s = make_sequence(rng, a.n_tokens, exc_rate=float(rng.choice((0.05, 0.2, 0.5))), corr=0.3)
        rows += list(strategies(s["tokens"], s["loss_mask"]))
        n_probe += int((s["loss_mask"] & (s["labels"] == L_PROBE)).sum())   # labels: diagnostic count only
    df = pd.DataFrame(rows)
    res = pd.DataFrame(dict(strategy=df.columns, accuracy=df.mean().values, n_positions=len(df),
                            probe_share=n_probe / len(df), n_streams=a.n, seed=a.seed, n_tokens=a.n_tokens))
    os.makedirs("results", exist_ok=True)
    res.to_csv(out, index=False)
    print(res.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
