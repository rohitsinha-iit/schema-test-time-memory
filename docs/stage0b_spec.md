# Stage 0b specification: SCHEMA inside a tiny sliding-window language model

*Written 3 Nov 2026 by the analysis session. Implementation is done by Claude Code following the prompts in
section 10. Numbers marked "estimate" must be replaced by measurements from the smoke runs.*

## 0. Question and what carries over from Stage 0a
Stage 0a showed, for a linear cortex fed with given keys and values, that replaying what recurs and is not yet
predicted protects the cortex far better than replaying what is surprising (+0.246 cortex rule recall, 135/135
paired streams). Stage 0b asks whether that survives three things Stage 0a did not have:
1. keys and values come from a model's own attention projections, not from the stream generator;
2. most evicted tokens are filler, not facts, so the hippocampus must not fill up with junk;
3. the memory branches are trained by an outer loop and must be used through learned gates.

Decisions carried over from Stage 0a (stage0a_m1_decision.md): FIFO eviction is the default; reconsolidation of
anchors on value conflict stays; superseding stale copies stays on but is not a claimed component; the gate is
the sum-normalized one with rho 4 and A-GEM projection; gate attribution is still open, so schema_nogate is a
required arm.

## 1. Token stream (sim0b/stream.py)
Vocabulary: specials {PAD, BOS, SEP, QRY}; key tokens K (512); value tokens V (512); filler tokens F (256).
* A **fact** is the 4 tokens `k_a k_b v SEP`. A key is the token pair (k_a, k_b).
* Each sequence draws its own private rule set: n_r = 16 keys with random values and Zipf(1.0) frequencies.
  Because rules are re-drawn per sequence, nothing can be memorized in the backbone weights during training; all
  knowledge must be acquired at test time.
* With probability exc_rate the next fact is an **exception**: a fresh key used once, with a random value.
  With probability corr its first token k_a is copied from a random rule key (the analogue of key correlation).
* Between facts insert 0 to 3 filler tokens, uniform.
* **Drift**: at the middle of the sequence, 25% of the rules get new values.
* **Training probes**: after a fact, with probability 0.15 insert `QRY k_a k_b v` for a key seen earlier,
  sampled with a log-uniform delay (in facts). Loss is cross-entropy on (i) probe answers and (ii) value tokens of
  facts whose key appeared earlier in the sequence (the generator emits this mask). First-occurrence values are
  unpredictable and are masked out.
* **Evaluation streams** have no interleaved probes. After the stream, a read-only probe segment asks every
  recurring rule (current value) and every exception, with memory writes disabled. Delay is measured from the last
  occurrence of the key to the end of the stream.
* exc_rate is sampled per training sequence from {0.05, 0.2, 0.5}; corr fixed at 0.3. Evaluation runs each
  exc_rate separately.
* The generator also returns per-token labels (filler, rule key, rule value, exception key, exception value, probe)
  for diagnostics only. Model code must never read them (same rule as Stage 0a).

## 2. Backbone (model0b/backbone.py)
* Decoder-only transformer: 4 layers, d_model 256, 4 heads (d_h 64), SwiGLU MLP 4x, RMSNorm, RoPE in attention.
  About 4M parameters plus embeddings (estimate).
* **Chunked sliding-window attention**: the sequence is processed in chunks of c = 64 tokens. Each chunk attends
  causally to itself and to the previous W tokens (W = 128 by default). Memory reads inside a chunk use the memory
  state as of the start of the chunk (chunk-level causality, as in LaCT and AllMem). This keeps the loop at
  L / c steps per layer.
* **Gist (AHN-style)**: a gated delta-rule state per head, written with the attention K and V of every token that
  leaves the window (AHN Eq. 5: alpha, beta per token from x; read gamma(x) q h W_o, scalar gate per head, grouped
  output projection). Pure PyTorch; a sequential loop over the evicted chunk is acceptable if the smoke run shows
  acceptable speed, otherwise use the chunkwise (WY / UT-transform) delta rule. Do not use Triton or torch.compile
  (Windows).
* Train the backbone once per seed with **randomized window W in {64, 128, 256}** per batch, as AHN does, so that
  evaluating with a larger window later is fair.

## 3. SCHEMA branches (model0b/schema_layer.py), added to every layer
All branches reuse the layer's attention keys and values (L2-normalized keys), like AHN, so they can be bolted onto
a frozen backbone. Per head:
* **Hippocampus H**: N = 256 token-level landmarks (k, v) with tags r (reuse) and t_in. Written when a chunk leaves
  the window: every evicted token becomes one landmark (no pooling in Stage 0b). Read: top-s (s = 32) by q . k,
  then softmax attention over those, output projection, per-channel gate g_H initialized to 0.
* **Cortex C**: linear fast weight W_C (d_h x d_h) per head, read q W_C^T, per-channel gate g_C initialized to 0.
  Updated only in micro-sleep (except in the arrival and surprise arms). Inner learning rate
  eta = base_eta * softplus(w_eta . x + b) per landmark, with base_eta a config value.
* **Anchors A**: R = 128 pairs (k, C(k) at release time), reservoir sampling, per head.
* **Tags and routing** exactly as Stage 0a, but computed on model keys and values: on eviction, cue match
  (cos > cue_thresh) with agreeing value (cos >= agree_thresh) increments r; conflicting value supersedes. Routing
  at micro-sleep: delta < tau_s is released (schema-consistent); delta >= tau_s and r >= r_min is replayed with
  priority log1p(r) * delta; the rest stays.
* **Thresholds are not transferable from Stage 0a** (keys here are learned). Make cue_thresh, agree_thresh and
  tau_s config values, and log, per evaluation, the distribution of max cue similarity for true recurrences vs
  first occurrences (labels used only in this diagnostic). Calibrate on validation streams before the main runs.
* **Micro-sleep** every S = 2 evicted chunks (128 tokens), N = 2 replay steps, replay batch 64 landmarks, gate with
  damage_norm "sum", rho 4, A-GEM on, backtracking 2.
* **Differentiability**: C's updates are differentiable functions of K, V and the learned eta, so the outer loop
  backpropagates through them (as in TTT). Selection, tagging and routing are discrete and get stop-gradient.

## 4. Arms and equal memory budget
Memory is counted per layer per head in floats. Shared by all arms: the gist state (d_h^2). The rest of the budget
is B = (W + N + R) * 2 d_h + d_h^2, with W = 128, N = 256, R = 128.
| Arm | What it has | How the budget is spent |
|---|---|---|
| swa_big | backbone + gist only | window enlarged to use all of B |
| arrival | C written with every evicted chunk (chunkwise TTT, AllMem-like) | C + enlarged window |
| surprise | as arrival, per-token eta scaled by error norm (Titans-like) | C + enlarged window |
| uniform_replay | H + C, uniform priority, no routing, no gate, no anchors | H gets N + R landmarks |
| schema | H + C + A, routing, gate, reconsolidation, FIFO eviction | as specified |
| schema_nogate | schema without the gate (A-GEM off too) | as schema |
| schema_surprise_pri | schema with priority delta only, r_min 0 | as schema |
| full_attn (reference) | separately trained full-attention model | no budget; upper bound, not a peer |
A unit test must recompute the floats per arm and assert equality (except full_attn).

## 5. Training regime (mirrors the Stage 1 retrofit)
1. **Backbone** (per seed): train backbone + gist from scratch on training streams, CE loss as in section 1,
   AdamW lr 1e-3 cosine, batch 16 x 4096 tokens, steps chosen by the smoke run (target: within-window probe
   accuracy >= 95%).
2. **Arms** (per seed, per arm): freeze backbone and gist; train only the new parameters (gates, output projections,
   eta networks) with the same loss, same data order, same number of steps for every arm. swa_big has no new
   parameters and is evaluated directly with its enlarged window.
3. Tune only base_eta per arm, over {0.1, 0.3, 1.0, 3.0}, on validation seeds with short runs. If the best value
   is at an edge, extend the grid once before the main runs (lesson from Stage 0a).

## 6. Evaluation
Evaluation streams: 4096 tokens (training length) and 8192 tokens (2x), exc_rate in {0.05, 0.2, 0.5}, 64 streams
per condition, seeds 100 to 102 for the backbone and arms. Reported per arm:
* **rule_acc_full**: probe accuracy on recurring rules with everything readable;
* **rule_acc_retest**: same after clearing window and H (only C and gist remain), the test-retest probe;
* **rule_acc_drifted** and **rule_old_value_rate** (fraction answering the pre-drift value);
* **exc_acc** by delay bucket in tokens: within window, 1 to 4 x W beyond, more than 4 x W beyond;
* **branch use**: mean |g_H| and |g_C|, and the share of the token-mixer output norm from H and C on probe tokens;
* **ablations at test**: zero C, zero H (no retraining), report the drop in rule_acc_retest and exc_acc;
* **H composition** at the end of each stream (diagnostic): fraction of landmarks by token type, median r and
  delta per type; gate statistics (steps tried, shrunk, rejected), anchor count.

## 7. Pass criteria for M2 (fixed now)
Primary score per stream: 0.5 * rule_acc_retest + 0.5 * exc_acc (beyond-window buckets pooled), at 4096 tokens.
1. schema beats uniform_replay and schema_surprise_pri on the primary score by at least 2 points, positive in
   every seed (3 of 3), averaged over the three exc_rates;
2. the C and H branches are used: zeroing C drops rule_acc_retest by at least 10 points, and zeroing H drops
   beyond-window exc_acc by at least 10 points;
3. reported but not deciding: whether the Stage 0a ordering on the cortex reproduces, i.e. rule_acc_retest of
   schema above schema_surprise_pri and above surprise.
Gate check (G0, before any SCHEMA work): the backbone reaches >= 95% within-window probe accuracy, and
beyond-window exception accuracy of swa_big is clearly below that (otherwise the task does not need memory and
must be made harder).

## 8. Compute (estimates)
Model about 4M parameters; 4096-token sequences; chunk loop of 64 steps per layer. On the laptop RTX 4060
(about 6.9 GB free) a backbone run of a few thousand steps should take on the order of an hour, and each arm
similar. 7 arms x 3 seeds is roughly a day of laptop GPU time, or a few hours on the 4090. Measure throughput in
the smoke run first and report tokens per second.

## 9. Files and outputs
`sim0b/stream.py`, `model0b/{backbone.py, gist.py, schema_layer.py, memory_torch.py}`, `train0b.py`,
`eval0b.py`, `tests/test_0b_*.py`. Results: `results/stage0b_<date>_<tag>.jsonl` (one line per arm x seed x
condition, with config, git commit, wall time) and checkpoints under `ckpt0b/` (gitignored).

## 10. Prompts for Claude Code (run in order; bring results back after P3a and after P3c)

### P3-pre (5 minutes)
```
Read CLAUDE.md and stage0a_m1_decision.md. Make FIFO eviction the default for the "schema" arm in sim/run.py
(rename the redundancy variant to schema_redundancy_evict), keep test_equal_memory_budget passing, commit.
Do nothing else from P2b yet.
```

### P3a: stream, backbone, learnability gate G0
```
Read CLAUDE.md and stage0b_spec.md (sections 1, 2, 5.1, 7 G0, 8, 9). Implement sim0b/stream.py and the backbone
with chunked sliding-window attention and the AHN-style gated delta-rule gist, in pure PyTorch (no Triton, no
torch.compile). Unit tests: stream masks and labels are consistent; chunked SWA equals a dense masked reference
on a short sequence; the gist update equals a naive per-token loop; no position reads memory written by a later
chunk. Then a smoke training run on the laptop GPU (check free memory with nvidia-smi first; stop if under 5 GB
free and tell me): report tokens per second, peak memory, the training curve, and G0 metrics (within-window probe
accuracy, beyond-window exception accuracy for windows 128 and 512). Save results/stage0b_<date>_g0.jsonl. Commit.
Do not tune the task to pass G0; if it fails, report and stop.
```

### P3b: memory modules in PyTorch
```
Read stage0b_spec.md section 3. Port Hippocampus, Anchors, the linear cortex, routing, micro-sleep and the gate
from sim/memories.py to batched PyTorch modules (model0b/memory_torch.py), per layer and head, batch-first.
Equivalence test: driven with the Stage 0a stream (given keys and values, one head, batch 1, float64) the torch
version must reproduce sim/memories.py metrics for arms schema, schema_nogate, schema_surprise_pri and
uniform_replay on seeds 100 and 101 within 1e-6. Also test differentiability: gradients flow from the read output
to base_eta and the eta network through two micro-sleep steps. Commit.
```

### P3c: integrate, budgets, smoke for every arm
```
Read stage0b_spec.md sections 3 to 6. Add the SCHEMA branches to the backbone (all arms in section 4), the
equal-budget test, train0b.py (frozen backbone regime) and eval0b.py. Calibrate cue_thresh, agree_thresh and tau_s
on validation streams using the similarity diagnostic (report the histograms as CSV). Run a short smoke for every
arm on seed 0 (few hundred steps) and save results/stage0b_<date>_smoke.jsonl with all section 6 metrics and the
H composition diagnostic. Commit. Do not run the full experiment yet.
```

### P3d: main runs (after the smoke is reviewed in the analysis session)
Written after P3c is reviewed.

## 11. Revision 1 (3 Nov 2026): G0 failed; diagnosis and fix

**Result (commit a57fc13).** After 2,000 steps (65 min, 33.9k tokens/s, 5.4 GB peak), within-window probe accuracy
was 0.02 to 0.10 and beyond-window exception accuracy about 0.002 (chance is 1/512). Training masked accuracy
plateaued at about 0.25 from step 650 on.

**Diagnosis: a frequency shortcut, not a code bug (as far as the numbers show).** Measured on 48 training streams
with the unchanged generator (28,588 loss positions, 17% of them probes):
| Strategy on training loss positions | Accuracy |
|---|---|
| predict the most frequent value seen so far in the sequence | 0.269 |
| predict the most recent value | 0.132 |
| recall the value of the same key, if seen within the last 128 tokens | 0.753 |
| recall the value of the same key from anywhere earlier | 0.993 |
The plateau (0.25) matches the frequency shortcut. Evaluation probes ask each of the 16 rules once, where the same
shortcut scores 1/16 = 0.0625, and the measured eval rule accuracy is 0.0635 to 0.0645. With Zipf-skewed rules and
83% of the loss on repeated rule values, the model found a strategy that never looks at the key.

**Fix (training curriculum only; evaluation streams and the G0 criterion stay exactly as before).**
1. Backbone training uses uniform rule frequencies (zipf 0), so the frequency shortcut is worth only 1/16.
2. Curriculum: phase A on 512-token sequences, 0 to 1 filler, probe_p 0.5, until held-out within-window probe
   accuracy is at least 0.90 (max 3,000 steps); phase B on the full 4,096-token stream (zipf 0, probe_p 0.3),
   2,000 steps, randomized window as before.
3. Two sanity runs come first, to separate "cannot learn recall" from "learns the shortcut":
   S1, dense causal attention, no gist, 2 layers, 512 tokens, single-token keys, uniform rules, no filler,
   probe_p 0.5, at most 1,500 steps; S2, the real chunked-SWA backbone with 2-token keys on the phase A stream.
   Both should exceed 0.95 probe accuracy. If S1 fails, suspect the attention or position code, not the task.
4. Eval adds: rule probe accuracy by rule-frequency rank, and the frequency-shortcut accuracy on the same probes,
   so a shortcut can never again pass silently.
SCHEMA branches will be trained and evaluated on the original Zipf 1.0 streams, as section 5 says.

## 12. Revision 2 (4 Nov 2026): S1 failed; a second shortcut, plus a short training budget

**Result (commit aa4ae36).** S1 (dense attention, no gist, 2 layers, 512 tokens, single-token keys, uniform rules,
no filler, probe_p 0.5) reached probe accuracy 0.193 after 1,500 steps. Training took 61 s at 405k tokens/s, and
the cosine schedule took the learning rate to zero. Accuracy was flat from about step 200.

**Diagnosis.** Scored on 64 S1 streams (3,654 probes) with the unchanged generator:
| Strategy on S1 probes | Accuracy |
|---|---|
| copy the value of the most recent fact | 0.244 |
| most frequent value so far | 0.113 |
| value of the same key (what we want) | 1.000 |
The log-uniform probe delay puts a lot of probes on the fact just before, so "copy the last value" pays off and
the model settled there. This is a second shortcut, not yet evidence of a bug. A bug is still possible and must be
ruled out with a reference model.

**Fix.**
1. Curriculum and sanity streams draw the probed key **uniformly over distinct keys seen so far** (option
   probe_delay="uniform_key"). The log-uniform option stays for later stages.
2. No decay to zero in sanity runs: linear warmup, then constant learning rate, stopping at the accuracy target or
   at max_steps (S1 and S2: 10,000 steps, about 7 minutes for S1 at the measured speed).
3. Reference model R1: a minimal standard GPT (learned absolute positions, F.scaled_dot_product_attention with
   is_causal, same width, depth, optimizer and data as S1). It separates "the data cannot be learned" from "our
   model code is broken".
4. Unit tests that must pass before training: (a) causality, changing tokens after position i leaves outputs at
   positions up to i unchanged, for both dense and chunked SWA; (b) RoPE gives the same attention scores when all
   positions are shifted by a constant; (c) a 1-layer model learns "predict the previous token" above 99% within
   500 steps.
5. Every evaluation prints, next to model accuracy, the accuracy of the two shortcuts on the same probes.

**Decision table for the next result.**
| R1 | S1 (ours, dense) | Reading | Next step |
|---|---|---|---|
| pass | pass | the problem was data shortcuts and training length | S2, then curriculum and G0 v2 |
| pass | fail | bug in our attention, position or loss code | find it with the unit tests; do not continue |
| fail | fail | the task or loss pipeline is wrong (masks, targets) | stop and report |

## 13. Revision 3 (4 Nov 2026): the masked-only loss was the blocker

**Result (commit a5376cd).** With uniform-key probes and constant LR for 10,000 steps, both the reference GPT R1
(0.070) and S1 (0.066) stayed at the level of the two shortcuts (0.068 and 0.074) while the loss kept falling.

**Independent check in the analysis session.** A separately written minimal GPT (2 layers, d 256, learned
positions, scaled_dot_product_attention with is_causal, AdamW lr 1e-3, batch 32 x 512) on the same S1 stream
(pool of 4,000 training sequences, 128 fresh held-out sequences):
| Training loss | Held-out probe accuracy at steps 500 / 1,000 / 1,500 / 2,000 / 3,000 |
|---|---|
| masked targets only (as in train0b.py) | 0.052 / 0.052 / 0.048 / 0.040 / 0.036 (memorizes the pool; train loss 0.89) |
| next-token loss on every token | 0.018 / 0.681 / 0.915 / 0.933 / 0.919 |
So there is no bug in train0b.py or the models: the stream is learnable, but only about 90 of 512 positions carry
loss, and that sparse signal does not produce the attention circuit for recall by key (an induction head).
With loss on every token the circuit forms between steps 500 and 1,000.

**Fix.** Training loss for the backbone and every arm = mean next-token cross-entropy over all positions, plus the
masked-target loss with weight 1 (so probe answers and repeated values still get extra weight). Evaluation, the
shortcut reporting and the G0 criterion are unchanged. Use a fresh batch every step (the Pool already does this);
do not train on a fixed pool, which the check above shows can be memorized.
