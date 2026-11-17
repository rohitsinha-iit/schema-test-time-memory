# LoRA (weight-space) cortex: calibration on validation seeds 0–2 (4 Nov 2026)

## Setup
* **Cortex:** rank-8 LoRA (MLP + o_proj, 540 pairs per KV head) on frozen Qwen2.5-1.5B, written only by replay, read by the model itself.
* **H, reuse:** the pilot's H; kappa reuse.
* **delta:** no-context NLL of the episode's landmark tokens.
* **Interleaving:** KL to the base on plain text in every step.
* **Gate:** G − 4D > 0 on a reservoir of anchor episodes.
* **Score:** 0.5 × retest rule (cortex only) + 0.5 × exceptions beyond W.

## Protocol 1: occupancy-triggered sleeps (`cal_lora_schema`, `cal_lora_arr`, `cal_lora_rep`)
Best per arm:

| arm | best | retest rule | retest exc | exc beyond W | score | replay steps |
|---|---|---|---|---|---|---|
| lora_uniform | lr 1e-3, 8 steps | 0.405 | 0.120 | 0.102 | 0.253 | 88 |
| lora_sleep_all | lr 1e-3, 8 steps | 0.355 | 0.148 | 0.123 | 0.239 | 144 |
| lora_schema | lr 1e-3, 8 steps | 0.200 | 0.015 | 0.105 | 0.152 | 42 |
| lora_arrival | lr 1e-3 | 0.053 | 0.000 | 0.103 | 0.078 | 89 |

lr 3e-3 destroys every arm. Schema got far fewer replay steps: releases keep H below the occupancy trigger, and its step loop ran out of candidates. The comparison was confounded by compute.

## Protocol 2: equal replay compute (`cal_lora_eq_*`; every 4 chunks, 8 steps; Stage 0a priority sampling)

| arm | best lr | retest rule | retest exc | exc beyond W | rule (full) | score | text NLL | H at end |
|---|---|---|---|---|---|---|---|---|
| lora_uniform | 1e-3 | 0.390 | 0.228 | 0.154 | 0.305 | **0.272** | 3.51 | 1504 |
| lora_surprise | 3e-4 | 0.055 | 0.003 | 0.116 | 0.442 | 0.085 | | |
| lora_arrival | 3e-4 | 0.042 | 0.013 | 0.116 | 0.476 | 0.079 | 2.41 | |
| lora_schema_nogate | 1e-3 | 0.114 | 0.022 | 0.000 | 0.116 | 0.057 | 4.58 | 475 |
| lora_sleep_all | 1e-3 | 0.092 | 0.015 | 0.021 | 0.079 | 0.056 | 5.04 | 0 |
| lora_schema | 1e-3 | 0.092 | 0.012 | 0.006 | 0.146 | 0.049 | **2.81** | 515 |

Plain-text NLL of the base model is 2.34.

## Findings
1. **The cortex works.** Uniform replay gives 0.39–0.41 retest recall on recurring facts from the weights alone.
2. **The gate protects the model.** At equal compute, schema keeps plain-text NLL at 2.81 against 4.58 without the gate (129 steps rejected, 55 shrunk per stream).
3. **Schema routing fails here through premature release.** delta (landmark NLL reading the episode's own sentence) falls once the LoRA memorizes the sentence, so 80 of about 112 episodes are released as "consolidated". Yet the model cannot answer questions about them (retest exc 0.012). H empties (515 landmarks against 1,504 for uniform) and exceptions beyond W collapse (0.006). In a weight-space cortex, "predicts the episode" does not mean "has consolidated the fact".
4. **Not run on test seeds.**
