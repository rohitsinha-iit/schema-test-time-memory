# Stage 0a, milestone M1: decision (3 Nov 2026)

Source: D:\NeuriAI\project\schema_mem\results\stage0a_20261103_grid.csv (commit 9981db4). 13 arms x 27 cells
(exception rate {5, 20, 50}% x key correlation {0, 0.3, 0.6} x rule-frequency skew {0.5, 1.0, 1.5}) x 5 test seeds
= 1,755 runs, 0 failures. Learning rates tuned per arm on validation seeds 0 to 2. All contrasts below are paired
over the same 135 (cell, seed) streams. Linear cortex only.

![Stage 0a key results]({{artifact:art_2fbb7478-829f-4434-81ad-5ddb37148448}})

## Verdict
**M1: conditional pass.** P1 is strongly supported. P3 passes the pre-registered threshold, but the gain cannot yet
be attributed to the gate itself. P2 holds only while exceptions are a minority. Two learning-rate grids hit their
edge, so one short follow-up (P2b) is needed before Stage 0b.

## P1: recurrence routing vs surprise routing (supported)
| Contrast (cortex-only rule recall) | Mean gain | Streams where positive |
|---|---|---|
| schema vs schema_surprise_pri (same H, same gate) | +0.246 | 135 / 135 (minimum +0.170) |
| schema vs surprise_fifo (Titans-like) | +0.103 | 135 / 135 |
| arrival_fifo vs surprise_fifo (plain vs surprise-scaled writes) | +0.030 | 135 / 135 |

* The gap over surprise priority grows with the exception rate: +0.20 at 5%, +0.23 at 20%, +0.30 at 50%.
* It barely depends on key correlation (at 5%: +0.199, +0.203, +0.214 for correlation 0, 0.3, 0.6). The damage
  is therefore not mainly targeted interference from exceptions near a rule key. It is aggregate crosstalk from
  many exceptions in a cortex of limited capacity, plus rules starved of replay. Evidence for starvation: the gap
  is +0.14 for the most frequent third of rules and +0.30 to +0.31 for the middle and rarest thirds.
* The proposal's single-exception formula is still correct, but the paper should state the effect as a capacity
  and replay-budget argument, not as a correlation argument.
* The predicted crossover is weak. Surprise priority recalls recent exceptions better only at the 50% exception
  rate (+0.10), and is worse at mid ages at 20%.

## P2: exceptions beyond the window (supported only when exceptions are a minority)
| Exception rate | schema minus arrival_fifo, ages 64 to 256 | ages 256 to 1,024 |
|---|---|---|
| 5% | +0.639 (all streams) | +0.878 (all streams) |
| 20% | +0.444 (all streams) | 0.000 |
| 50% | -0.103 (no stream positive) | -0.027 |
At equal bytes, SCHEMA's 256 anchors take half of the episodic budget, so when exceptions flood the stream it holds
fewer of them. Uniform replay, which has twice the episodic store, beats SCHEMA on mid-age exceptions at 20% and
50% (-0.057 and -0.235), while SCHEMA's cortex beats uniform replay at every rate (+0.029, +0.072, +0.191).
This is a real budget trade-off between protecting the cortex and storing more episodes.

## P3: interference gate (threshold met, attribution open)
| Arm vs schema_nogate | Cortex gain | Positive | Median anchor-loss ratio | Steps shrunk |
|---|---|---|---|---|
| schema (mean gate) | +0.013 | 135 / 135 | 0.95 | 0.0% |
| schema_dsum (sum gate, rho 1) | +0.016 | 135 / 135 | 1.02 | 38.8% |
| schema_dsum_rho4 (sum gate, rho 4) | +0.010 | 131 / 135 | 0.42 | 27.3% |
* The mean-normalized gate never shrank or rejected a step, yet "schema" still beats "schema_nogate" by 0.013. That
  gain must come from the A-GEM projection (active whenever the gate is on) or from the learning rate (nogate chose
  0.25, the lowest value in its grid; schema chose 0.5). This needs a clean ablation.
* The sum-normalized gate does act and cuts anchor loss by 58% at rho 4, with a small cortex gain.
* No gate setting rescues surprise priority: schema_surprise_pri and schema_surprise_pri_dsum are identical to the
  last digit, and the gate never fired there. Routing, not the gate, protects the cortex.

## Component decisions (rules fixed in stage0a_sanity_review.md)
* **Eviction: FIFO becomes the default.** Redundancy eviction wins old exceptions by only +0.013 overall
  (threshold 0.05) and loses on cortex recall (FIFO +0.004, +0.012, +0.035 by exception rate). It is better on
  mid-age exceptions only at 50% exceptions (+0.066).
* **Reconsolidation stays.** Without it, drifted-rule recall drops by 0.039 and the old value lingers (+0.108).
* **Superseding stale copies has no measurable effect** (+0.003 cortex). Keep it as an ablation, not a selling point.

## Fairness caveat (must fix before claims)
Learning rates chosen at a grid edge: arrival_fifo and surprise_fifo at 0.1 (lowest), schema_nogate at 0.25
(lowest), uniform_replay, sleep_all and schema_noreconsol at 2.0 (highest). Uniform replay's validation score was
still rising steeply (0.834, 0.894, 0.954 for 0.5, 1, 2), so its cortex gap may shrink with a larger step.

## Next: P2b (short, CPU), then Stage 0b
See the prompt in the chat or HANDOFF file; summary: extend edge grids, isolate gate vs A-GEM vs learning rate,
make FIFO eviction the default, and test smaller anchor budgets.
