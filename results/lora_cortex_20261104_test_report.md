# LoRA (weight-space) cortex: held-out test, seeds 100–119 (4 Nov 2026)

## Setup
* Frozen Qwen2.5-1.5B-Instruct; W 1024; 4 sinks; text stream as in the pilot (16 rules, exc_rate 0.2 and 0.5).
* 20 seeds × 2 exception rates = 40 streams per arm; 9 arms; 360 rows; no duplicate (arm, seed, exc_rate).
* Equal memory bytes for every arm (tests/test_pilot_budget.py).
* Equal replay compute: every replay arm sleeps every 4 chunks and takes 8 steps per sleep. Arrival and surprise take
  2 steps per chunk.
* Hyperparameters were chosen on validation seeds 0–2 only (lora_cortex_20261104_calibration.md and the
  `cal_lora_max_*` logs):

  | arm | hp |
  |---|---|
  | lora_schema | lr 1e-3, delta = max landmark-token NLL, tau 1.0, r_min 2, cue 0.7, theta 0.98 |
  | lora_schema_nogate | the same, tau 2.0 |
  | lora_uniform, lora_sleep_all | lr 1e-3 |
  | lora_arrival, lora_surprise | lr 3e-4 |

* **Primary score per stream:** 0.5 × retest rule accuracy (cortex only; window and H cleared) + 0.5 × exception
  accuracy beyond W. This is the score used for the LoRA calibration. No LoRA-specific pass bar was pre-registered.

**Validation scores for tau** (lora_schema / lora_schema_nogate): 1.0: 0.143 / 0.085; 2.0: 0.073 / 0.115;
3.0: 0.086 / 0.081. With delta = mean, lora_schema had scored 0.049.

## Results (means over 40 streams)

| arm | primary | retest rule (C only) | exc beyond W | exc > 4W | rule (full) | held-out text NLL |
|---|---|---|---|---|---|---|
| lora_uniform | **0.242** | **0.372** | 0.112 | 0.104 | 0.330 | 3.84 |
| hippo (H only, no cortex) | 0.205 | 0.000 | **0.410** | **0.358** | **0.761** | (base) |
| lora_arrival | 0.101 | 0.094 | 0.107 | 0.000 | 0.468 | 2.40 |
| **lora_schema** | 0.099 | 0.156 | 0.042 | 0.025 | 0.351 | 2.87 |
| lora_surprise | 0.097 | 0.070 | 0.124 | 0.000 | 0.528 | 2.36 |
| lora_sleep_all | 0.067 | 0.113 | 0.022 | 0.000 | 0.102 | 4.97 |
| lora_schema_nogate | 0.059 | 0.101 | 0.017 | 0.000 | 0.080 | 4.63 |
| swa_big (window gets the bytes) | n/a | n/a | 0.275 | 0.000 | 0.799 | (base) |
| swa | n/a | n/a | 0.000 | 0.000 | 0.475 | (base) |

The base model's plain-text NLL is 2.34. Per-exception-rate tables are in pilot_20261104_test_lora_summary_full.csv.

### Paired: lora_schema minus arm (primary; 95% bootstrap CI over the 40 streams; wins/ties/losses)

| arm | diff | 95% CI | W/T/L |
|---|---|---|---|
| lora_uniform | −0.143 | [−0.179, −0.105] | 5/2/33 |
| hippo | −0.106 | [−0.125, −0.087] | 0/1/39 |
| lora_arrival | −0.002 | [−0.026, +0.023] | 19/0/21 |
| lora_surprise | +0.002 | [−0.021, +0.024] | 24/1/15 |
| lora_sleep_all | +0.032 | [+0.010, +0.053] | 23/2/15 |
| lora_schema_nogate | +0.040 | [+0.018, +0.061] | 25/6/9 |

### Internals (means per stream)

| arm | replay steps tried | accepted | rejected by gate | gradient steps incl. retries | episodes released | H landmarks at end | wall s |
|---|---|---|---|---|---|---|---|
| lora_schema | 196 | 80 | 116 | 532 | 59 | 838 | 199 |
| lora_schema_nogate | 191 | 191 | 0 | 191 | 77 | 554 | 72 |
| lora_uniform | 224 | 224 | 0 | 224 | 0 | 1504 | 84 |
| lora_sleep_all | 224 | 224 | 0 | 224 | 112 | 0 | 64 |
| lora_arrival / surprise | n/a | n/a | n/a | 178 | n/a | 0 | 47 / 48 |

## Findings
1. **SCHEMA routing with a LoRA cortex does not beat simple replay.** Compared with uniform replay over the same H at
   equal replay compute, it is worse on the primary score (−0.143, CI excludes 0, 33 of 40 streams). It ties the
   online TTT-style arms (arrival, surprise). Validation predicted this ordering (0.143 vs 0.272).
2. **The interference gate does its job.**
   * Schema with the gate beats schema without it (+0.040, CI excludes 0).
   * It keeps plain-text NLL at 2.87, against 4.63 without the gate and 2.34 for the base.
   * The cost: the gate rejects 59% of the steps it is offered (116 of 196). Schema therefore gets about 80 effective
     cortex updates against 224 for uniform, and its cortex is weaker (retest rule 0.156 vs 0.372).
3. **Release still costs exceptions.**
   * With delta = max, H keeps 838 landmarks (on validation, delta = mean left 515), but uniform keeps 1504.
   * Exceptions beyond W: 0.042 for schema, 0.112 for uniform, 0.410 for hippo.
   * About 59 episodes per stream are still released as "consolidated" before the cortex can answer for them.
4. **Every LoRA arm reads H far worse than hippo.**
   * lora_uniform holds 84% of hippo's landmarks (1504 vs 1792 at equal bytes), yet answers exceptions beyond W at
     0.112 against 0.410. Its online rule accuracy is also lower: 0.330 against 0.761.
   * Two explanations fit and neither is tested:
     * (a) H's keys and values were written by the model before the LoRA changed it, so they go stale as the cortex
       learns. The proposal's anchors protect cortex outputs, not H's stored keys.
     * (b) The LoRA damages the model's general ability to use its context (text NLL 3.84 vs 2.34).
   * Either way, this caps every weight-space-cortex arm's exception score in this setup, schema included.
5. **Score and LM quality trade off.**
   * The arms that keep text NLL near the base (arrival 2.40, surprise 2.36, schema 2.87) all score about 0.10.
   * Uniform's 0.242 comes with text NLL 3.84.
   * No arm was compared at matched LM damage. The ranking could change at a matched text-NLL budget; this is not
     tested.

## Provenance and run notes
* **Code.** Every row records `git_commit a208103…-dirty`. That field is stale: code reaches the server by scp, and
  its git checkout was never updated. With line endings normalized, the server's pilot/*.py files match commit
  12fa5e8 exactly, except pilot/summarize.py (4e4f664), which runs only after the arms.
* **GPU0 interruption.** The first GPU0 job (`test_lora_gpu0`: lora_schema, lora_sleep_all, swa, swa_big, hippo)
  stopped at 16:21 after 9 streams per arm. There was no traceback and no OOM, and the server journal shows nothing.
  Other users were active on the server; the cause is unknown. The missing streams were rerun under new tags with
  identical hp:
  * `test_lora_gpu0_resume_a`: lora_schema, seeds 105–119;
  * `test_lora_gpu0_resume_b`: all 5 arms, seed 104 at exc 0.5;
  * `test_lora_gpu0_resume_c`: the other 4 arms, seeds 105–119.
* **Empty file.** `pilot_20261104_test_lora_b.jsonl` is empty, left by a launch that failed on how the seed list was
  written. It was kept, not overwritten.
* **Files.**
  * Rows: pilot_20261104_test_lora_gpu0.jsonl, pilot_20261104_test_lora_gpu0_resume_{a,b,c}.jsonl and
    pilot_20261104_test_lora_gpu1.jsonl.
  * Summaries: pilot_20261104_test_lora_summary_full.csv (all rows; pilot_20261104_test_lora_summary.csv is an
    earlier partial summary) and pilot_20261104_test_lora_paired.txt (`python pilot/paired.py <files>`).
