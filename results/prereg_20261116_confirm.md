# Pre-registration: confirmatory test of SCHEMA with hippocampal recall distillation (16 Nov 2026)

Written before any tuning or test run of the arms below. Code: pilot/ on the A6000 copy of the repo, with the
16 Nov patch (arms hrd_schema_ng, hrd_delta_pri, hrd_reuse_pri; priority option "reuse").

## Why
On 5 Nov the pre-registered arm hrd_schema (with the interference gate) failed. An unplanned no-gate variant beat
hrd_uniform on the same test seeds (+0.025 [+0.008, +0.042]). That is a hypothesis, not a result. This test checks it
on fresh streams.

## Setup (fixed)
* Frozen Qwen2.5-1.5B-Instruct, W 1024, 4 sinks, 8,192-token text streams (pilot/run_pilot.py STREAM), --retest.
* Filler and landmark selector rebuilt on 16 Nov (filler from wikitext-103 train shard 0; selector fitted by
  self-distillation on validation seeds only). All arms are rerun on the same new streams, so every comparison is paired.
* Equal memory bytes (tests/test_pilot_budget.py). Equal replay compute: 8 steps per sleep, every 4 chunks.
* Primary score per stream = 0.5 x retest rule accuracy (window and H cleared) + 0.5 x exception accuracy beyond W.

## Tuning (validation seeds 0, 1, 2 x exc 0.2, 0.5 only)
* hrd_schema_ng: lr in {1e-4, 3e-4, 1e-3} x tau in {0.5, 1.5}; n_steps 8, r_min 2, w_min 0.3. Pick the best mean
  validation primary score. hrd_delta_pri and hrd_reuse_pri use the same values.
* All other arms keep the settings validated on 4 and 5 Nov (recorded in their result rows).

## Test (seeds 120 to 139 x exc 0.2, 0.5 = 40 streams; never used before)
Arms: hrd_schema_ng, hrd_uniform, hrd_schema, hrd_delta_pri, hrd_reuse_pri, lora_uniform, lora_arrival,
lora_surprise, hippo, swa_big, swa, teacher.

## Decisions (95% paired bootstrap CI over the 40 streams, 10,000 resamples)
* C1 (primary): hrd_schema_ng minus hrd_uniform on the primary score > 0 with the CI excluding 0. Pass means
  "SCHEMA's priority adds to recall distillation"; fail is reported as such.
* C2: hrd_schema_ng minus lora_uniform on the primary score > 0, CI excluding 0.
* C3 (the paper's thesis): hrd_schema_ng minus hrd_delta_pri on retest rule accuracy > 0, CI excluding 0
  (reuse-weighted priority consolidates recurring facts better than error-only, surprise-like priority).
* Reported, no decision: hrd_reuse_pri; hrd_schema (gate) minus hrd_schema_ng; held-out text NLL of every
  cortex arm (guard: within 0.15 nats of the base model); per exception rate breakdowns.
