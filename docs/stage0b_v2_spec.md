# Stage 0b v2 specification (DRAFT): SCHEMA on a pretrained, frozen base

*Drafted 4 Nov 2026 by Claude Code at the user's request, for review in the analysis session before any code is
written. Replaces the from-scratch backbone of stage0b_spec.md (sections 2, 5.1, 11 to 13). Items marked
"verified" were checked on 4 Nov 2026; items marked "estimate" or "open" must be settled before P4a runs.*

## 0. Why the change

Stage 0b v1 put a precondition on the critical path that the proposal never needed: a tiny transformer trained
from scratch had to learn in-context key-to-value recall (an induction circuit) before any SCHEMA component could
be tested. Four attempts failed (G0, revisions 1 to 3). The last checks (4 Nov) rule out the training loop, the loss
weighting, bf16, and fresh-versus-pooled data: a bare AdamW loop with next-token loss on all positions, with our
stream and our reference GPT, stays at 0.06 to 0.07 probe accuracy (the shortcut level) after 1,500 to 3,000 steps,
and does not even learn to predict which in-context key comes next (5.1 nats; copying the 16 keys gives about 2.8).
The analysis session's 0.92 with its own script is not reproduced; that difference is unexplained.

The proposal itself (sections 4.1, 4.2, 4.5, 6.3) assumes a **frozen pretrained base whose attention already
retrieves**, with SCHEMA's stores built on that base's keys and values, starting from a **trained AHN**, and trained
by **self-distillation (KL to a full-attention teacher)**. This spec does exactly that, at small scale.

What carries over unchanged: Stage 0a (sim/), the decisions in stage0a_m1_decision.md (FIFO eviction, sum gate,
rho 4, A-GEM, reconsolidation on, supersede on but not claimed), the P3b plan (torch port of sim/memories.py with
an equivalence test), the arms and the equal-budget rule, the M2 pass criteria (stage0b_spec.md section 7, primary
score and branch-use thresholds), and seed discipline (tune on 0 to 2, report on 100 and up).

## 1. Base, teacher, gist

* **Base**: Qwen2.5-3B-Instruct, frozen, bf16 (verified config: 36 layers, hidden 2048, 16 query heads,
  2 KV heads, head dim 128, RoPE theta 1e6, 32k context).
* **Teacher**: the same weights with full causal attention over the whole stream. No second model in memory.
* **Gist**: the released AHN-GDN-for-Qwen-2.5-Instruct-3B (13.0M parameters; verified: trained with KL, base
  frozen, sliding window 256 randomized, sequences up to 24k). This is the "trained AHN as the starting point" of
  proposal section 4.2, and it is also the AHN baseline arm.
* Development option (open): Qwen2.5-0.5B-Instruct with a gist we train ourselves with the AHN code (no released
  checkpoint exists below 3B), for faster iteration before moving to 3B.

## 2. Stream: fictional-entity facts in natural text (proposal sections 4.5.1 and 6.4.2)

The Rules-and-Exceptions structure of Stage 0a, written as text the base can already read.
* **Entities**: invented names (random syllables, rejected if they appear in a reference word list), so the base
  cannot know any fact in advance.
* **Facts**: an entity, an attribute and a value, rendered by one of 3 paraphrase templates per attribute, for
  example "The vault code of Marlow Tesk is 4817." Values are unpredictable by construction (4-digit numbers,
  invented words), so only memory can supply them.
* **Rules**: n_r facts per stream that recur, Zipf frequencies (as Stage 0a), each recurrence a random paraphrase.
* **Exceptions**: one-off facts with probability exc_rate. **Correlation** (the analogue of key cosine): with
  probability corr an exception reuses a rule's entity with a different attribute, or a near-duplicate name
  (one syllable changed). Which of the two is "open"; both can be knobs.
* **Drift**: at T/2, 25% of rules are restated with new values ("Marlow Tesk's vault code was changed to 2209.").
* **Filler**: distractor documents from a public corpus (for example a wikitext-103 or FineWeb sample) between
  facts, which sets the gap in tokens and makes most evicted tokens non-facts (spec v1 section 0, point 2).
* **Probes** (evaluation, read-only, memory writes disabled): "Question: What is the vault code of Marlow Tesk?
  Answer:" appended after the stream, one probe per rule and per exception, all sharing the end-of-stream state.
  Score: exact match of the greedy answer, plus answer log-probability. Delay = tokens from the fact's last
  occurrence to the end, bucketed as in v1 (within W, 1 to 4 W beyond, more than 4 W beyond).
* **Training streams** also contain interleaved probes with answers (so KL covers the answering behaviour).
* Lengths (estimate): student window W = 1024 plus sinks; streams of 8,192 tokens for development and 16,384 for
  the main runs.
* The generator returns labels (fact kind, rule or exception, drift) for evaluation and diagnostics only; memory
  code never reads them (hard rule 1).
* Shortcut reporting stays: most frequent value of the attribute in the stream, and most recent value, scored on
  the same probes.

## 3. Student: SCHEMA branches on the frozen base

Per layer, per KV head (2 per layer; each serves a group of 8 query heads, read with the group's queries):
* **Working memory**: frozen base attention over the sliding window W plus sinks.
* **Gist**: AHN-GDN, unchanged.
* **Hippocampus H** (proposal 4.1): K episode slots. When a chunk of c = 64 tokens leaves the window, attention
  pooling with m learned queries turns its keys and values into m landmarks; kappa_j = normalized mean landmark key.
  Defaults (estimate): K = 128, m = 8, so 1,024 landmarks per head. Tags r_j (reuse), delta_j (cortical error),
  a_j (age).
* **Read** (proposal 4.2): top-s slots by q . kappa_j (s = 8), softmax over their s * m landmarks, added to the head
  output before o_proj with a per-channel gate g_H initialised to 0. The **reuse tag comes from read attention
  mass**: r_j <- lambda_r r_j + sum of attention mass slot j received during the chunk (the "awake reactivation"
  tag; Stage 0a's cue match at eviction becomes an ablation).
* **Cortex C**: linear fast weight (d_h x d_h = 128 x 128) first, AllMem SwiGLU with momentum second (proposal 4.1).
  Read q -> C(q) with per-channel gate g_C initialised to 0. Changed only by micro-sleep.
* **Anchors A**: R = 128 pairs per head, reservoir sampling.
* **Micro-sleep** (proposal 4.3): triggered when H occupancy exceeds theta * K (theta = 0.75) or at a document
  boundary. Routing (schema-consistent / recurring and unpredicted / one-off), priority r * delta, **interleaved
  replay batches** (episode landmarks mixed with anchors at a fixed ratio, default 3:1), interference gate with
  damage_norm "sum", rho 4, A-GEM, backtracking 2, release with anchors.
* **Keys for memory (open, needs a diagnostic in P4a)**: stored keys and memory queries should be
  position-independent, because an episode must be addressable from any later position. Proposal: use pre-RoPE keys
  and queries, L2-normalized. P4a measures, on the teacher, how often pre-RoPE q . k ranks the correct fact token
  first, compared with RoPE scores at the true positions.

## 4. Training (proposal 4.5)

* Loss: KL(teacher || student) on every position after the first W tokens (earlier positions are identical in both).
  No CE term.
* Trainable: H pooling queries, read gates g_H and g_C, memory read projections if any, the cortex learning-rate
  (and later momentum) networks. Base frozen. Gist frozen in the first runs (open: also fine-tune it on the streams
  for every arm equally, so that SCHEMA's gain is not just in-domain training).
* Differentiate through one or two unrolled micro-sleep steps; routing, tagging and selection are stop-gradient.
* Window W, chunk c, K and m randomized per example during training, as AHN and AllMem do (estimate of ranges: W in
  {512, 1024, 2048}).
* The same streams, the same order and the same number of steps for every arm.

## 5. Arms and equal memory budget

As stage0b_spec.md section 4, with the AHN arms made explicit:
| Arm | Description |
|---|---|
| swa_big | window enlarged to use the whole budget (the comparison reviewers check first) |
| ahn_gdn | released gist, no new branches (also stream-fine-tuned if section 4's open item is adopted) |
| arrival | C written with every evicted chunk (AllMem-like) |
| surprise | as arrival, per-token step scaled by error (Titans-like) |
| sleep_all | micro-sleep over all of H without selection (Lee et al.-like) |
| uniform_replay | H + C, uniform priority, no routing, no gate, no anchors |
| schema | H + C + A, routing, gate, interleaving, reconsolidation, FIFO eviction |
| schema_nogate | schema without gate and A-GEM |
| schema_surprise_pri | priority delta only, r_min 0 |
| teacher | full attention (upper bound, not a peer) |
| knn_all | kNN over all evicted keys (unbounded reference, not budget-matched) |
Budget per layer per KV head in floats: window W * 2 d_h, plus H: K * m * 2 d_h + K * (d_h + 3), plus C: d_h^2 (linear)
or the SwiGLU parameter count, plus A: R * 2 d_h. The gist state is shared by all arms. A unit test recomputes it
for every arm (hard rule 2).

## 6. Evaluation

Section 6 of stage0b_spec.md, on the new streams: rule_acc_full, rule_acc_retest (window and H cleared, only C and
gist remain), rule_acc_drifted and rule_old_value_rate, exception accuracy by delay bucket, branch use (gate
magnitudes and output-norm shares), zero-C and zero-H ablations at test, H composition (fraction of slots that hold
facts versus filler; median r and delta per type), gate statistics, and the shortcut accuracies. Added:
* **LM-quality guard** (cheap stand-in for P5): KL to the teacher and perplexity on held-out natural long text
  (for example a PG-19 subset) for ahn_gdn and schema; LongBench is Stage 1.
* Consolidation efficiency G / (G + D) over a stream, and the delta-versus-r scatter coloured by fate (proposal 6.4).

## 7. Gates and pass criteria

**G0' (inference only, before any training; thresholds open, to be fixed by the analysis session before the run):**
1. The teacher answers the probes: suggested at least 0.90 overall and at least 0.85 on exceptions beyond 4 W.
2. The task needs memory: SWA-only (window W, no gist) exception accuracy beyond W is suggested at most 0.20.
3. Reported, not deciding: the released ahn_gdn at the same W, by delay bucket. If it already answers beyond-window
   exceptions at 0.80 or more, make the stream harder (more facts per stream, longer gaps) before SCHEMA work.
4. Reported: the key-addressing diagnostic of section 3.

**M2**: stage0b_spec.md section 7, unchanged (hard rule 3).

## 8. Environment and compute

* Server 10.0.117.206 (verified 4 Nov: Linux 5.15, 32 cores, 125 GB RAM, 407 GB free disk, RTX 4090 and RTX
  A5000 with 24 GB each, both idle; system Python 3.8). Needs its own Python 3.11 environment with AHN's stack
  (PyTorch 2.5.1 with CUDA 12.4, the forked flash-linear-attention, transformers per AHN's pyproject). One run per GPU.
* Laptop (Windows, no Triton): stream generator, unit tests, analysis. AHN's GDN kernels need Triton, so model runs
  go to the server.
* Cost (estimate): about 1e15 FLOPs per 16k-token training stream (teacher forward, student forward and backward
  through the frozen base); at an effective 60 TFLOP/s on the 4090 that is about 15 s per stream, so 2,000
  streams take about 8 hours per arm and seed. 9 arms x 3 seeds is about 9 days on two GPUs. Levers: cache the
  teacher's top-k logits once per stream (removes about a third), develop on 8k streams, tune only base_eta on
  short runs, and use the 0.5B development option.

## 9. Milestones (prompts to be written by the analysis session)

* **P4a**: server environment; clone AHN, reproduce inference of the released AHN-GDN-3B on one long-context
  example (proposal section 9, step 1); stream generator v2 with unit tests (labels consistent, values unpredictable
  for the base, paraphrases, drift, shortcut tables); G0' evaluation (teacher, SWA-only, ahn_gdn) and the
  key-addressing diagnostic. Stop and report.
* **P4b**: H branch (pooling, top-s read, reuse tags from read mass) on the ahn_gdn student, trained with KL; arms
  swa_big, ahn_gdn, ahn_gdn + H; smoke run and H composition.
* **P4c**: torch memory modules (the P3b equivalence test against sim/memories.py stays), linear cortex, anchors,
  routing, gate, interleaving, micro-sleep trigger; every arm; equal-budget test; smoke for every arm.
* **P4d**: SwiGLU cortex; base_eta tuning on validation seeds; main runs on test streams.

## 10. Open questions for the analysis session

1. Pre-RoPE keys for memory, or keys rotated to a fixed reference position?
2. 3B with the released gist from the start, or the 0.5B development path first?
3. G0' thresholds, stream length and W.
4. Fine-tune the gist on the streams for all arms, or keep it frozen?
5. Correlation knob: shared entity, near-duplicate name, or both?
6. Should the from-scratch v1 path be explained (the analysis session's 0.92 script compared with ours) for the
   record, or closed as superseded?
