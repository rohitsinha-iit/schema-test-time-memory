# Hippocampal recall distillation (HRD): held-out test (5 Nov 2026)

Idea, literature and the pre-registered test: `results/schema_ideas_20261104.md` (written before any HRD run).
Code: commit 3c0ee0e (HRD v2). Base model frozen Qwen2.5-1.5B-Instruct, W 1024, LoRA rank 8, equal memory bytes.

## Protocol
* **Tuning.** Validation seeds 0–2 × exc {0.2, 0.5} (`cal_hrd_{a,b,c}`).
  * hrd_schema: lr {3e-4, 1e-3} × τ {0.5, 1.5}. Best: lr 3e-4, τ 0.5 (validation 0.246).
  * hrd_uniform: lr {1e-4, 3e-4, 1e-3}. Best: lr 1e-4 (validation 0.260).
  * Fixed: n_steps 8, r_min 2, w_min 0.3.
* **Test.** Seeds 100–119 × exc {0.2, 0.5}, `--retest`, 40 streams per arm (`pilot_20261105_test_hrd_{schema,uniform}.jsonl`),
  all at commit 3c0ee0e.
  * Earlier arms come from the 4 Nov LoRA test files, on the same streams.
  * The no-gate HRD ablation was run afterwards (see the addendum).
* **Score.** primary = 0.5 · retest rule recall (window and H cleared, cortex only) + 0.5 · exception recall beyond W.

## Results (means over 40 streams)

| arm | primary | retest rule | exc beyond W | rule (all stores) | held-out text NLL |
|---|---|---|---|---|---|
| **hrd_uniform** | **0.285** | 0.259 | 0.311 | 0.700 | 2.39 |
| **hrd_schema** | **0.275** | 0.274 | 0.277 | 0.699 | 2.40 |
| lora_uniform (previous best) | 0.242 | 0.372 | 0.112 | 0.330 | 3.84 |
| hippo (H only, no cortex) | 0.205 | 0.000 | 0.410 | 0.761 | — |
| lora_schema (old objective) | 0.099 | 0.156 | 0.042 | 0.351 | 2.87 |

Base-model text NLL is 2.34.

**Paired differences** (95% bootstrap CI over streams, W/T/L; `hrd_20261105_test_paired*.txt`):

| comparison | primary diff | W/T/L |
|---|---|---|
| hrd_schema − hrd_uniform | −0.010 [−0.032, +0.014] | 16/0/24 |
| hrd_schema − lora_uniform | +0.033 [−0.008, +0.077] | 22/0/18 |
| hrd_schema − hippo | +0.070 [+0.047, +0.094] | 33/0/7 |
| hrd_schema − lora_schema | +0.176 [+0.156, +0.199] | 40/0/0 |
| hrd_uniform − lora_uniform (not pre-registered) | +0.043 [+0.004, +0.084] | 22/0/18 |

**Per-metric differences, hrd_schema minus each baseline:**

| metric | vs hrd_uniform | vs lora_uniform |
|---|---|---|
| retest rule | +0.015 [−0.023, +0.055] | −0.098 [−0.163, −0.031] |
| exceptions beyond W | −0.034 [−0.050, −0.016] | +0.165 [+0.126, +0.203] |
| rule, all stores | −0.001 | +0.369 |

**Internals:**

| internal | hrd_schema | hrd_uniform |
|---|---|---|
| replay steps accepted | 139 of 221 (gate rejected 82) | 223 of 223 |
| episodes released | 0.4 | 0 |
| H landmarks | 1488 | 1504 |
| retest exceptions | 0.00 | 0.00 |
| wall time per stream | 265 s | 91 s |

Neither arm consolidates one-off exceptions (retest exceptions 0.00), as the design intends.

## Pre-registered verdict: FAIL
hrd_schema had to beat both hrd_uniform and lora_uniform with CIs excluding 0. It beats neither:
* vs hrd_uniform: a tie leaning negative;
* vs lora_uniform: positive, but the CI includes 0.

## What works
1. **The HRD write rule.** Distilling the episode-in-context teacher into the cortex at confirmed-recall moments is
   the best cortex objective so far.
   * Both HRD arms beat every earlier arm.
   * hrd_schema beats the old next-token schema by +0.176 (40/0/0).
   * hrd_uniform beats lora_uniform by +0.043 [+0.004, +0.084]. This comparison was not pre-registered.
2. **No forgetting.** Held-out text NLL is 2.39–2.40 vs 3.84 for lora_uniform (base 2.34).
3. **H stays readable.** Exceptions beyond W are 0.28–0.31 vs 0.11 for lora_uniform. Rule accuracy with all stores is
   0.70 vs 0.33.
4. **The cortex adds to H.** hrd_schema beats H alone by +0.070 (33/0/7).
5. **The label-free reuse detector.** A confirmed copy (the model reads an H landmark and the stream then produces that
   landmark's token) is a usable reuse signal that needs no labels.
6. **The exception/rule split holds.** One-offs are never recalled, so they stay in H.

## What fails
1. **SCHEMA's selection machinery adds nothing on top of HRD:** priority = reuse × error, release, and the gate.
   * Uniform sampling over recalled episodes does as well, at about a third of the compute.
   * The recall detector already does the selecting: only recalled episodes produce training queries.
2. **Release does not trigger.** 0.4 episodes are released per stream at τ 0.5. At τ 1.5 more are released (33 on
   validation), but the score falls (0.202).
   * The distillation error rarely drops below τ, because the cortex learns only about a quarter of the rules
     (retest rule 0.27).
3. **The gate costs bytes and steps for no gain.**
   * It rejects 37% of steps.
   * The anchor reservoir takes H bytes, which is the likely reason exceptions beyond W are 0.034 lower than
     hrd_uniform.
   * The no-gate ablation (addendum) confirms the gate is the bottleneck.
4. **Consolidation is weak.** Cortex-only rule recall is 0.27, below lora_uniform's 0.37.
   * lora_uniform learns more, but forgets and damages H reads.
   * Most distillation targets land on name tokens: only about 8% are value tokens (validation diagnostic).

## Addendum: no-gate ablation (run after the verdict above)
**Setup.**
* hrd_schema_nogate uses hrd_schema's validation-chosen settings (lr 3e-4, τ 0.5); it was not tuned separately.
* Same 40 test streams, commit 3c0ee0e.
* Files: `pilot_20261105_test_hrd_nogate_{a,b}.jsonl` and `hrd_20261105_test_paired_{with_nogate,ref_nogate}.txt`.

| arm | primary | retest rule | exc beyond W | rule | text NLL | wall per stream |
|---|---|---|---|---|---|---|
| **hrd_schema_nogate** | **0.310** | **0.365** | 0.254 | 0.698 | 2.41 | 93 s |
| hrd_uniform | 0.285 | 0.259 | 0.311 | 0.700 | 2.39 | 91 s |
| hrd_schema | 0.275 | 0.274 | 0.277 | 0.699 | 2.40 | 265 s |
| lora_uniform | 0.242 | 0.372 | 0.112 | 0.330 | 3.84 | 84 s |

**Paired primary differences, hrd_schema_nogate minus each arm:**

| vs | primary diff | W/T/L |
|---|---|---|
| hrd_uniform | +0.025 [+0.008, +0.042] | 28/0/12 |
| lora_uniform | +0.068 [+0.026, +0.111] | 28/1/11 |
| hrd_schema | +0.034 [+0.015, +0.053] | 29/1/10 |
| hippo | +0.105 [+0.083, +0.127] | 36/0/4 |

**Per metric vs hrd_uniform:**
* retest rule +0.106 [+0.075, +0.136];
* exceptions beyond W −0.056 [−0.073, −0.039];
* rule −0.002 [−0.042, +0.039].

**Reading.**
* **The gate is what held SCHEMA back.** It throws away 37% of the replay steps that the priority chose, and costs
  about 3× the compute.
* **With the gate removed, SCHEMA's priority works.** reuse × cortical error on HRD queries consolidates rules into the
  cortex much better than uniform replay over the same recalled episodes: cortex-only rule recall is 0.365 vs 0.259.
  This is the first time SCHEMA's selection beats uniform replay in any of our tests.
* **The cost is a little exception recall** (−0.056 vs hrd_uniform). Possible causes:
  * the no-gate arm still reserves anchor bytes that it never uses (H 1487 vs 1504 landmarks);
  * cortex training on the prioritized episodes may disturb H reads more.
* **Release is still nearly inactive** (0.6 episodes per stream), so the gain comes from priority, not release.

**Status of the claim.**
* The pre-registered arm (hrd_schema) **fails**, and that verdict stands.
* The no-gate variant passes the same bar (beats hrd_uniform and lora_uniform, CIs exclude 0), but it was chosen after
  seeing test results.
* So this is a **hypothesis for a confirmatory test, not a confirmed result.** Confirming it needs:
  1. hrd_schema_nogate written down as the pre-registered arm;
  2. its anchor bytes given back to H;
  3. a run on fresh test seeds (for example 120–139), compared against hrd_uniform.

## Not verified
* A confirmatory run of hrd_schema_nogate on fresh test seeds.
* Separate validation tuning for hrd_schema_nogate.
* Tuning beyond 2–3 learning rates and 2 τ values.
* Other models and other stream types.
* hrd_uniform vs lora_uniform was not a pre-registered comparison.
