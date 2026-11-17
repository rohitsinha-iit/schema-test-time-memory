# One-day pilot: training-free SCHEMA on a frozen Qwen2.5-1.5B-Instruct (4 Nov 2026)

## Setup
* **Model:** Qwen2.5-1.5B-Instruct, frozen.
* **Processing:** 64-token chunks with a 1,024-token sliding window plus 4 sinks; no training of any kind.
* **Stream** (`pilot/text_stream.py`):
  * 8,192 tokens of invented-person facts between wikitext-103 passages; 74% of tokens are filler;
  * 16 recurring facts (Zipf 1.0) and one-off exceptions at rate 0.2 or 0.5, with corr 0.5 (an exception shares a person with a recurring fact);
  * values change for 25% of the recurring facts at the midpoint.
* **Probes:** read-only questions after the stream, scored by exact match of the value.
* **Memory:** every arm holds W + 2,048 key/value pairs per layer per KV head (`tests/test_pilot_budget.py`).
* **Tuning:**
  * hyperparameters tuned on validation seeds 0–2 (`pilot_20261104_cal_*.jsonl`);
  * test seeds 100–119 × exc_rate {0.2, 0.5} = 40 streams per arm.

**Training-free stand-ins** (deviations from the proposal):
* **Hippocampus landmarks:** chosen by a linear probe on the layer-14 hidden state, fitted to the full-attention teacher's question attention on validation streams (self-distillation, no labels). Model surprise (NLL) was tried first and selected at chance.
* **Reads:** H and C entries join the frozen attention softmax, as in InfLLM. Pre-RoPE keys are rotated to just before the window.
* **Cortex:** slot memory (nearest slot by key cosine, delta step) read like H. A linear cortex cannot be read without trained gates.
* **Gate and routing:** as Stage 0a (sum-normalized damage, rho 4, backtracking 2); no A-GEM, because the update is not a gradient step.

## Gates
* **G0′** (validation): full attention 0.88 overall, 0.87 on exceptions beyond W; window-only 0.00 beyond W. This misses the 0.90 bar by 2 points; I went ahead.
* **Go 2** (validation): oracle H 0.73–0.82, the NLL selector 0.03–0.08, the distilled selector 0.38–0.46, the enlarged window 0.29.

## Test results (40 streams, means over streams)
"Retest" means window and H cleared, so only C answers.

| arm | rule (full) | rule drifted | exc beyond W | exc > 4W | retest rule (C only) | retest exc (C only) | primary | schema − arm | schema better |
|---|---|---|---|---|---|---|---|---|---|
| teacher (full attention) | 0.871 | 0.558 | 0.870 | 0.835 | | | | | |
| swa (W only) | 0.480 | 0.508 | 0.000 | 0.000 | | | | | |
| swa_big (equal bytes) | 0.799 | 0.846 | 0.275 | 0.000 | | | | | |
| hippo (H only) | 0.762 | 0.525 | **0.408** | **0.344** | 0.000 | 0.000 | 0.204 | +0.114 | 0.93 |
| schema | 0.764 | 0.660 | 0.359 | 0.265 | 0.277 | 0.149 | 0.318 | | |
| schema_nogate | 0.744 | 0.642 | 0.346 | 0.256 | 0.262 | 0.151 | 0.304 | +0.014 | 0.58 |
| schema_surprise_pri | 0.782 | 0.662 | 0.371 | 0.312 | 0.264 | 0.174 | 0.318 | 0.000 | 0.53 |
| uniform_replay | 0.728 | 0.635 | 0.339 | 0.233 | 0.269 | 0.161 | 0.304 | +0.014 | 0.60 |
| sleep_all | 0.790 | 0.700 | 0.357 | 0.279 | 0.300 | 0.171 | 0.328 | −0.010 | 0.43 |
| arrival (AllMem-like) | 0.782 | 0.704 | 0.397 | 0.368 | 0.239 | 0.192 | 0.318 | −0.000 | 0.43 |
| surprise (Titans-like) | 0.808 | 0.729 | 0.404 | 0.346 | **0.301** | 0.195 | **0.353** | −0.035 | 0.30 |

Primary score = 0.5 × retest rule + 0.5 × exc beyond W. Per-exc_rate tables: `pilot_20261104_test_summary.csv`.

## Pilot criteria (fixed before the test run)
| Prediction | Bar | Result |
|---|---|---|
| P2: H keeps exceptions beyond the window | H ≥ enlarged window + 10 points | **Pass**: 0.408 vs 0.275 (+13.3); beyond 4W 0.344 vs 0.000; H better on 37 of 40 streams |
| P4: recurring facts survive in C alone | ≥ 10 points above no memory (0) | **Pass**: schema retest rule 0.277, against 0.149 for exceptions |
| P1: recurrence routing beats surprise and arrival on retest | ≥ 5 points, ≥ 15/20 streams | **Fail**: schema 0.277 vs surprise 0.301 and arrival 0.239; primary 0.318 vs 0.353 / 0.318 |
| P3: the gate helps | reported | +1.4 points primary, 23 of 40 streams; within noise |

## Diagnosis (validation streams, labels used only here; `pilot/diag_reuse.py`)
SCHEMA's routing depends on a reuse tag that marks recurring facts. In this setting no reuse signal separates recurring-fact landmarks from exception landmarks:
* read attention mass (proposal 4.2): AUC 0.52;
* token-level key cue matches: 0.53–0.64 at most layers and thresholds; 0.76 at layer 14 with cosine ≥ 0.98, but 0.62–0.90 across streams.

The reasons:
* In Stage 0a a key was the fact's identity. Here a key is one token's attention key.
* Recurring facts are paraphrased, and exceptions share people's names with rules.

With r uninformative, priority r·δ behaves like δ, and the cortex arms differ by only a few points.

## What this implies
1. The hippocampal half of the proposal works training-free on a real LLM at equal bytes (P2).
2. The cortex can hold facts after the window and H are cleared (P4). It is not selective, because nothing tells the system which episodes are the same fact recurring.
3. The missing piece is a **fact-level episode key**, the proposal's kappa_j from learned pooling, under which recurrences match and one-offs don't. In this training-free pilot neither attention keys nor read mass provide one.

**Next experiment:** learn kappa (and the pooling) with the proposal's KL self-distillation, or a self-distilled probe like the selector. Then re-check the reuse AUC before rerunning the arms.

## Follow-up: a self-distilled fact-level key kappa (same day)
**Kappa.** `pilot/fit_kappa.py` builds it without labels:
* Pairs: on validation seeds 0–1, the full-attention teacher's strongest attention from a token back to a token more than 64 positions earlier (top 10% per stream, upper layers) gives positive pairs.
* Map: a linear map of the layer-14 hidden state to 128 dimensions, trained with InfoNCE so the two ends of each pair match.
* Reuse: the number of later landmarks whose kappa has cosine ≥ c with a stored landmark (Stage 0a cue match at eviction). Kappa storage is charged as 16 pairs of the hippocampus budget.

**Reuse AUC, recurring-fact vs exception landmarks** (held-out validation seed 2):

| Signal | AUC |
|---|---|
| Read mass | 0.52 |
| Raw hidden state, cue | 0.59–0.77 |
| **Kappa, cue (c 0.7–0.9)** | **0.78** (0.65 at exc 0.2, 0.85–0.90 at exc 0.5) |

**Calibration.** Validation seeds 0–2 chose cue threshold 0.7, r_min 2, theta 0.98, tau 0.5.

**Test** (seeds 100–119, 40 streams; other arms as above):

| arm | exc beyond W | retest rule (C only) | retest exc (C only) | selectivity (rule − exc in C) | primary |
|---|---|---|---|---|---|
| schema (read-mass reuse) | 0.359 | 0.277 | 0.149 | +0.128 | 0.318 |
| **schema_kappa** | 0.356 | **0.329** | 0.143 | **+0.187** | 0.343 |
| schema_kappa_nogate | 0.348 | 0.323 | 0.145 | +0.178 | 0.335 |
| sleep_all | 0.357 | 0.300 | 0.171 | +0.129 | 0.328 |
| arrival | 0.397 | 0.239 | 0.192 | +0.047 | 0.318 |
| surprise | 0.404 | 0.301 | 0.195 | +0.106 | **0.353** |

**schema_kappa minus arm, per stream:**

| Comparison | Primary | Retest rule | Selectivity |
|---|---|---|---|
| vs surprise | −0.010 (better on 42%) | +0.028 (18/40 streams ≥ +5 pts) | +0.080 (65%) |
| vs arrival | +0.025 (65%) | +0.090 (25/40) | +0.140 (82%) |
| vs schema (read mass) | +0.025 (60%) | +0.052 | +0.059 |

**Against the fixed P1 bar** (≥ 5 points on retest on ≥ 75% of streams vs surprise and arrival): still **fail**.
* vs surprise: +2.8 points, 18/40 streams.
* vs arrival: +9.0 points, 25/40 streams.

The direction is now the one the proposal predicts. With kappa, the cortex holds the most recurring facts of any arm and keeps out the most exceptions. Surprise and arrival write more exceptions into C (0.195 and 0.192 against 0.143).

**Still not shown:**
* **Interference.** Surprise does not damage recurring facts much (0.301), so the surprise paradox's damage mechanism is not visible. A slot cortex with a high merge threshold mostly allocates new slots instead of overwriting, so correlated writes rarely interfere.
* **The gate.** For the same reason, the gate changes nothing (kappa vs kappa_nogate +0.7 points, 50% of streams).
* **Primary score.** Surprise still wins it, through exceptions beyond W: its enlarged window plus C beats H at 0.404 vs 0.356.

**Next:**
1. A cortex in which correlated writes interfere (a distributed or linear cortex with trained read gates), so P1 and P3 can be tested as the proposal states them.
2. A sharper kappa: train it with KL, or use pairs from repeated mentions of the same fact.

## Follow-up 2: the proposal's linear cortex (same day). Blocked by the read path.
**Built** (`pilot/linear_cortex.py`, `fit_qmap.py`, `train_gates.py`):
* **Cortex:** W per layer and KV head on unit-norm pre-RoPE keys and values.
* **Replay:** Stage 0a rules ported to torch: batch-gradient replay with anchors mixed in, A-GEM, and the sum-normalized gate with backtracking. The computed gain and damage equal Proposition 1's closed form (`test_linear_cortex_gate_matches_proposition_1`).
* **Read:** o_h += gamma_{l,h} · s · W(A_{l,h} q_h).
  * A is a per-head query→key map, fitted by ridge regression on 918k teacher far-attention pairs (no labels).
  * gamma is 336 scalar gates, initialised to 0 and trained by soft cross-entropy to the full-attention teacher's top-32 answer distribution (normal and retest probes, validation seeds 0–1, memory updates not differentiated).
* **Budget:** W is 64 pairs per KV head, so the other stores get the rest of the 2,048.

**Results** (validation seed 2, held out from gate training; `pilot_20261104_lin_*.jsonl`):

| Run | rule (full) | exc beyond W | retest rule (C only) | cortex fit to its own replayed items (delta q10) |
|---|---|---|---|---|
| lin_schema, eta 0.3 | 0.46–0.62 | 0.19–0.23 | 0.00 | 0.77 (not stored) |
| lin_schema, eta 1 | 0.46 | 0.10 | 0.00 | 0.74 |
| lin_schema, eta 3–30 | 0.00–0.10 | 0.00 | 0.00 | diverges (delta up to 103) |
| **least-squares oracle cortex** (diagnostic, every landmark, not budget-matched) | 0.00–0.31 | 0.00 | **0.00** | best case |

**Diagnosis** (`pilot/diag_linear.py`):
1. **Capacity is not the problem.** Ridge least squares on the real keys stores 64 items with error about 0.00, 128 items with about 0.02, and 1,024 items with 0.1–0.4. Centring the keys does not help.
2. **Storage by replay fails.** Keys are highly correlated (mean cosine 0.4–0.75), so batch-gradient replay is either too slow (eta ≤ 1: delta stays about 0.75 within a stream) or diverges (eta ≥ 3). A per-item delta rule (Kaczmarz sweeps) or preconditioned replay would be needed.
3. **The read fails even with a perfect store.** With the least-squares oracle, cortex-only recall is 0.00, and the trained gates make normal answers worse. A fitted query map plus 336 scalar gates cannot turn a linear associative read into something the frozen model uses. AHN and AllMem train read projections (millions of parameters, KL on large data) for this reason.

**Conclusion.** Testing P1 and P3 with an interfering (linear or SwiGLU) cortex needs the proposal's trained retrofit: read and write projections trained by KL self-distillation through micro-sleep (proposal 4.5, Stage 1). That cannot be done training-free or in minutes.

**Next:**
* Start that retrofit from the released AHN-GDN-3B, whose GDN read path is already trained to be readable.
* Use its GDN state as the cortex: write it only through SCHEMA's selected, gated replay instead of at eviction.
* Fine-tune its projections on the fact streams by KL.

## Correction (found during retrofit R2, same day)
`pilot/text_stream.py` listed the invented names from a Python set of strings. String hashing is randomized per process, so **the same seed gave a different stream in each process**. Fixed by sorting; regression test `test_stream_identical_across_processes`.

**What it affects:**
* **Within one run, nothing.** Every arm in a single run used the same generated streams, so per-stream paired comparisons inside a results file are valid. This covers the main test table: all 11 arms in `test_a` / `test_b`.
* **Across runs.** The same seed gave a different stream draw. `schema_kappa` (follow-up 1, `test_kappa_*`) and the linear-cortex runs were therefore compared with the main arms on different streams of the same distribution. Their **means are valid as unpaired comparisons**, but the "schema better on x% of streams" fractions in follow-up 1 are **not paired** and should be disregarded.
* **No leakage.** Validation seeds (0–2) and test seeds (100+) remained disjoint.
* **Reproducing.** Rerunning any old configuration gives different (now reproducible) streams.
