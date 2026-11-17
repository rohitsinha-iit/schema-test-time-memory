# Making SCHEMA's main idea work: literature, candidates, choice (4 Nov 2026)

Written before any experiment on the chosen idea.

## 1. What failed, and why (our evidence)
SCHEMA's main idea: replay priority = reuse × cortical error, consolidate recurring items, keep one-offs in H, release
what the cortex has learned, and gate each step against damage. In the LoRA-cortex test (seeds 100–119, equal bytes and
equal replay compute), schema lost to uniform replay over H: −0.143 [−0.179, −0.105].

1. **The cortical error measured the wrong thing.** delta was the NLL of the episode's own sentence, read with the
   sentence in context. It falls when the LoRA memorizes the surface text, not when the model can recall the fact from a
   cue. This caused premature release (H 838 vs 1504 landmarks; exceptions beyond W 0.042 vs 0.112) and a wrong
   priority.
2. **The training target was also wrong.** Next-token loss on the episode text stores the sentence but does not make the
   fact extractable from a question.
3. **The gate starved learning.** It rejected 59% of steps (80 accepted vs 224 for uniform).
4. **H reads degrade as the LoRA learns.** lora_uniform holds 84% of hippo's landmarks but scores 0.112 vs 0.410 on
   exceptions beyond W.

## 2. Literature
* **Knowledge stored ≠ knowledge extractable.** Allen-Zhu & Li, Physics of LMs 3.1 (ICML 2024): facts trained in one
  surface form can be memorized yet not extracted (0% QA) unless augmented. This matches failures 1 and 2.
* **Context distillation beats fine-tuning for knowledge injection.**
  * Snell et al. 2022 (Learning by distilling context).
  * Padmanabhan et al., NeurIPS 2023: a teacher conditioned on the entity definition, distilled into the student,
    propagates knowledge better than fine-tuning or editing, without hurting other contexts.
  * Caccia et al., COLM 2025 (Knowledge Modules, Deep Context Distillation): LoRA modules trained to match a teacher
    that sees the document. Next-token prediction "performs poorly" as the objective.
  * SDFT (arXiv 2601.19897) and iSDFT (arXiv 2609.24646) use a demonstration-conditioned teacher for continual
    learning, with a KL anchor to a frozen base.
* **Sparse, local updates forget less.**
  * Lin et al. 2025, Sparse Memory Finetuning (arXiv 2510.15103): updating only the memory-layer slots that the new
    data accesses more than background data (TF-IDF) gives an 11% drop on held-out NQ, vs 71% for LoRA and 89% for
    full finetuning.
  * Biderman et al., TMLR 2024: LoRA learns less and forgets less.
* **Dual memory with routing.** WISE (NeurIPS 2024): main memory plus an edited side memory, with a router deciding
  which to use.
* **Episodic memory for long-context LMs, no consolidation.** EM-LLM (ICLR 2025), Larimar, AHN, Titans.
* **Sleep and consolidation (2026 preprints).**
  * Behrouz et al., Language Models Need Sleep (arXiv 2606.03979): distillation of fast memories into slow ones, plus
    RL "dreaming".
  * EVAF (arXiv 2606.29916): gated LoRA writes with a test–retest protocol.
  * SLEEP (prediction-error tagging plus LoRA consolidation).
* **Gap.** None of these selects what to consolidate from where the episodic memory was actually needed, or uses a
  label-free "the hippocampus no longer adds anything" test to release episodes. That is the gap SCHEMA's main idea
  targets.

## 3. Candidates

| # | idea | fixes | cost | risk |
|---|---|---|---|---|
| A | **Hippocampal recall distillation (HRD).** When a later passage re-cues an H episode (a kappa match, the reuse signal we already compute), store that passage as a recall query. The cortex is trained to make the model *without* H predict, at the query, what the model *with the episode in context* predicts (context distillation, teacher = hippocampus-assisted model). Cortical error delta = KL(teacher ‖ cortex) at the episode's recall queries; priority = log(1 + reuse) × delta; release when delta < tau; gate G − ρD on the same quantity over anchor queries. One-offs are never re-cued, so they are never consolidated and stay in H. | 1, 2; maybe 3 (targeted steps damage less) | two short plain forwards per query; teacher cached once | the teacher may not copy the value; few queries per episode |
| B | Self-generated QA (SEAL / Active-Reading style): the base model writes questions about each episode; train on them; release when the cortex answers them without context. | 1, 2 | generation per episode | 1.5B writes poor questions; the question format may drift toward the probe format |
| C | Sparse memory-layer cortex (Lin et al. 2025): a key–value slot layer with TF-IDF slot selection instead of LoRA. | 3, 4 | new layer trained from scratch online | large build; no pretrained memory layer for Qwen |
| D | Gate by projection (A-GEM): project the step off the anchor gradient instead of rejecting it. | 3 | small | does not touch the main failure (1) |
| E | H-safe cortex placement: LoRA only in the upper layers, H read only below them. | 4 | small | caps cortex capacity; not about the main idea |

## 4. Choice: A (HRD)
A is the only candidate that makes SCHEMA's own quantities right without labels:
* **reuse** = the episode was needed by later text;
* **cortical error** = how much the hippocampus still adds at those moments;
* **release** = the hippocampus no longer adds anything;
* **gate** measured on the same thing.

The literature says distillation from an in-context teacher is the better write rule (Padmanabhan 2023, Caccia 2025).
B is close to A but needs generation and risks probe-format leakage. C, D and E fix secondary failures and can be added
to A later.

**Pre-registered test (written before running):**
* **Primary comparison.** At equal bytes and equal replay compute, hrd_schema vs:
  * hrd_uniform (same distillation objective, uniform sampling over re-cued episodes, no release, no gate). This
    isolates SCHEMA's routing, priority, release and gate.
  * lora_uniform (the previous best arm).
* **Score.** The same primary score: 0.5 × retest rule + 0.5 × exceptions beyond W.
* **Seeds.** Tune tau and lr on validation seeds 0–2; report on test seeds 100–119.
* **Claim.** "The main idea works" only if hrd_schema beats both, with the paired 95% CI excluding 0.
