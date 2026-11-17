# SCHEMA implementation plan

*3 Nov 2026. Companion to the SCHEMA proposal (schema_proposal.md). Memory and compute numbers marked "estimate" are back-of-envelope figures to be replaced by measurements in the first smoke test.*

## 1. Strategy in one paragraph

Build bottom up and kill the idea cheaply if it is wrong. Stage 0a tests the routing rule and the interference gate on the memory modules alone, with synthetic key-value streams and no language model, so it runs on CPU in minutes. Stage 0b puts the same modules into a tiny sliding-window language model trained from scratch on synthetic token streams (one small GPU). Only if both pass do we retrofit a real pretrained model. For that we start from the released AHN-GDN checkpoint for Qwen2.5-3B-Instruct, so the gist path is already trained and only the new SCHEMA modules are learned.

## 2. Code layout (new repo, working name `schema-mem`)

```
schema/
  memory/
    linear_cortex.py      # delta-rule cortex W (exact Prop. 1 gate available)
    swiglu_cortex.py      # AllMem-style SwiGLU fast weights, closed-form grads, momentum
    hippocampus.py        # K slots: landmark pooling, top-s sparse read, tags r, delta, age
    anchors.py            # reservoir of (k_a, C(k_a)) functional anchors
    gate.py               # look-ahead gain/damage, A-GEM projection, closed-form eta*, null-space option
    router.py             # schema-consistent / replay / exception rules; learned priority net (later)
    sleep.py              # micro-sleep loop: trigger, sampling, interleaving, release
  streams/
    rules_exceptions.py   # Stage 0 stream: recurring rules, one-off exceptions, rule drift
    fact_stream.py        # Stage 2 fictional-entity fact stream with recurrence and revisions
  sim/run_memory_only.py  # Stage 0a driver (no LM)
  models/tiny_lm.py       # Stage 0b: SWA + gist + H + C, 2 to 4 layers
  retrofit/               # Stage 1: patch AHN's Qwen2.5 modeling code with H and C
  eval/                   # test-retest probes, LongBench-6, LV-Eval, InfiniteBench, CLINC/Banking/DBpedia
  baselines/              # surprise-gated (Titans rule), uniform replay, sleep-all (Lee et al.), EWC, metaplastic
tests/
  test_prop1.py           # gain/damage formulas exact for the linear cortex
  test_gate.py            # accepted steps never raise G - rho*D below 0
  test_causality.py       # reads never see a chunk before it is evicted; read-before-update ordering
  test_budget.py          # byte accounting per arm (so matched-budget claims are enforced in code)
  test_slots.py           # insert, tag, release, capacity eviction bookkeeping
```

Stage 0 code is pure PyTorch, so it runs on Windows and on CPU. The retrofit stage depends on AHN's forks of flash-linear-attention and LLaMA-Factory (Triton kernels), so it has to run on one of the Linux GPU servers.

## 3. Milestones with pre-set pass criteria

| ID | When | What | Hardware | Pass criterion (decided now, before results) |
|---|---|---|---|---|
| M0 | Days 1 to 2 | Memory modules, gate, router, unit tests | CPU (local) | All tests pass; Prop. 1 matches to 1e-8 |
| M1 | Days 3 to 5 | Stage 0a memory-only simulation, 5 seeds | CPU (local) | P1: recurrence routing beats surprise routing on rule retention, with the gap growing in exception rate and key correlation. P3: removing the gate raises anchor loss by at least 20% with acquisition within 2 points |
| M2 | Week 2 | Stage 0b tiny LM (about 10M to 30M parameters) trained end to end, 3 seeds | 1 GPU, under 6 GB | SCHEMA beats both uniform replay and surprise gating on combined rule plus exception recall at equal memory bytes; gates g_H, g_C are clearly nonzero |
| M3 | Weeks 3 to 4 | Stage 1 retrofit of Qwen2.5-3B + released AHN-GDN; self-distillation on about 50M tokens | 1 GPU, 24 to 48 GB | No more than 0.5 average points lost on the LongBench six-task subset vs AHN-GDN-3B; nonzero gate use in most layers |
| M4 | Weeks 5 to 6 | Stage 2 continual streams and test-retest probes, 3 seeds | 1 to 2 GPUs | Facts that recurred remain answerable after window and H are cleared (P4); exceptions recallable beyond AHN's horizon (P2) |

Fail rules. M1 failing on P1 means the routing idea is wrong as stated: rework the routing signal before writing any LM code. M2 failing means the outer loop cannot learn to use the new stores: try the stream curriculum and gate warm-up before giving up. M3 failing on regression means the read path hurts base behaviour: lower gate initialization or restrict SCHEMA to upper layers.

## 4. Stage details

### Stage 0a: memory-only simulation (start here)

* Stream generator: n_r rules (keys on the unit sphere, values random), each recurring with frequency f; exceptions whose keys have cosine c to a rule key, appearing once; at fixed times a fraction of rules change value.
* Arms: write-at-arrival (AHN/AllMem behaviour), Titans-style surprise gating, uniform replay, sleep over all episodes (Lee et al. style), SCHEMA, and SCHEMA without gate, without anchors, with surprise priority.
* Cortex: linear first (exact gate), then SwiGLU.
* Grid: f in {2, 4, 8}, c in {0, 0.3, 0.6}, exception rate in {5%, 20%, 50%}, 5 seeds. Thousands of runs, minutes each on CPU.
* Output: rule recall, exception recall at delays, update accuracy after drift, anchor loss, accepted step fraction.

### Stage 0b: tiny language model

* 2 to 4 layers, width 256 to 512, SWA window 64 to 256 tokens, vocabulary from the synthetic task. Rules and exceptions are now encoded as token sequences, so the model must learn keys and values itself.
* Train the whole tiny model (no frozen base at this scale), then freeze the backbone and train only SCHEMA modules, to mirror the retrofit regime.
* Estimate: 1 to 3 GPU-hours per run on a 24 GB card, less than 6 GB of memory.

### Stage 1: retrofit Qwen2.5-3B-Instruct

* Starting point: `ByteDance-Seed/AHN-GDN-for-Qwen-2.5-Instruct-3B` (13.0M AHN parameters) on `Qwen/Qwen2.5-3B-Instruct`. AHN has released checkpoints only for 3B, 7B and 14B, so 3B is the smallest model we can start from without training our own AHN.
* Trainable: landmark pooling queries, H and C read gates (zero-initialized), cortex projections and its learning-rate and momentum networks. Base and gist frozen at first; unfreeze the gist in an ablation.
* Data: a ChatQA2 subset plus the stream curriculum (concatenated documents with recurring and one-off facts). Teacher is the same frozen base with full attention.
* Memory gotcha (estimate): with a vocabulary of about 152k, a single fp32 logit tensor for 16k tokens is close to 10 GB, so the KL must be computed in chunks of about 1k tokens, never on full-sequence logits. With that and gradient checkpointing, sequences of about 16k should fit on a 24 GB card and 24k on a 48 GB card.
* Compute (estimate): AHN reports about 320 A100-hours for 7B on 1B tokens. Training only the new modules on about 50M tokens at 3B, with 2 to 4 times overhead for replay, comes to roughly 15 to 30 GPU-hours per run on an A6000 or 4090.

### Stage 2: continual-learning evaluation

* Fact stream with recurrence counts {1, 2, 4, 8}, distractors, revisions, probes at increasing delays, and test-retest after clearing window and H.
* CLINC, Banking, DBpedia class-incremental protocol (as used for Hope) and one multi-session memory benchmark (LongMemEval or LoCoMo).
* Every arm gets the same memory bytes, enforced by `test_budget.py`.

## 5. Which GPUs can be used

| Machine | What it can run | Constraint |
|---|---|---|
| Local laptop (RTX 4060, 8 GB) | Stage 0b at the smallest sizes | The sandbox reports no GPU this session, so it cannot be used from here right now. Other programs often hold most of its memory, and the Triton-based AHN kernels are not a Windows target, so it cannot run Stage 1 |
| Local CPU (16 cores, 32 GB) | All of Stage 0a and the unit tests | None; no GPU needed |
| 4090 + A5000 server (24 GB each) | Stage 0b; Stage 1 at 3B with about 16k sequences | Usually occupied by your em_sweep jobs (about 21 of 24 GB each) |
| A6000 server (4 x 48 GB, shared) | Stage 1 at 24k sequences; Stage 2 | Shared with other users; also carries CVPR work |
| RTX 6000 Ada server (4 x 48 GB) | Same as the A6000 | Used in another of your projects; only with your say-so |

Timing. The CVPR 2027 submission deadline is 16 Nov, with supplementary material due 23 Nov, so GPU time on the servers is probably best kept for that until then. Stages 0a and M0 need no GPU at all and can start now. Stage 0b jobs are small enough (under 6 GB) to share a card, and Stage 1 fits naturally after the deadline.

## 6. First three days, concretely

1. Day 1: write `linear_cortex.py`, `gate.py`, `anchors.py`, `test_prop1.py`, `test_gate.py`; generate the Rules and Exceptions stream.
2. Day 2: `hippocampus.py`, `router.py`, `sleep.py`, budget and slot tests; the arrival-write and surprise baselines.
3. Day 3: run the Stage 0a grid on CPU, produce the crossover figure (rule retention vs exception rate for surprise and recurrence routing), and decide M1.
