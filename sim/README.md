# `sim/`: Stage 0a controlled study (linear cortex)

A CPU-only simulation that isolates *which* episodes a slow memory should consolidate, with keys and values given
directly (no language model).

| file | content |
|---|---|
| `streams.py` | "Rules and Exceptions" streams: 32 Zipf-distributed recurring rules, one-off exceptions at a set rate and key correlation, mid-stream rule drift, optional late-onset rules (`late_frac`) |
| `memories.py` | All memory systems: SCHEMA (H + reuse x error priority + anchors + gate + reconsolidation), surprise priority, arrival/surprise writes, uniform replay, sleep-all, and the gate variants (mean or sum damage, rho, A-GEM only, look-ahead only) |
| `run.py` | Arm definitions (`arms(b)`), evaluation metrics, `run_one()` |
| `grid.py` | M1 grid (3 Nov): eta tuning on validation seeds 0 to 2, then 27 cells x 5 test seeds |
| `p2b.py` | P2b retune with wide eta grids, gate attribution arms, anchor budgets R64/R128, E1 (late-onset rules) and E2 (d = 32, 64, 128) |
| `analyze_p2b.py` | Paired bootstrap contrasts and the report `results/stage0a_20261116_p2b_analysis.md` |
| `sanity.py`, `diagnose.py`, `summarize.py` | Sanity checks and helpers |

**Stream.** T = 2,400 chunks of 4 key/value pairs, d = 64; exception rate {5, 20, 50}% x key correlation
{0, 0.3, 0.6} x Zipf skew {0.5, 1.0, 1.5} = 27 cells; 25% of rules change value mid-stream.
**Equal bytes.** Every arm satisfies `H_slots x b + R = 64 x 4 + 256` (checked by an assert in `p2b.py`).
**Main metric.** `cortex_rule`, cosine recall of recurring rules from the cortex alone.

```bash
PYTHONPATH=. python -m pytest -q tests/test_core.py   # includes the Proposition 1 closed-form gate check
PYTHONPATH=. python -m sim.p2b --workers 30           # about 1 h on 30 cores
PYTHONPATH=. python sim/analyze_p2b.py
```
