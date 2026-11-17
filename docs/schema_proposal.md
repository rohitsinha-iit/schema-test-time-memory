# SCHEMA: selective, interference-gated consolidation from an AHN hippocampus into a test-time-trained cortex

*Research proposal, 12 Oct 2026. "SCHEMA" is a working name (Selective Consolidation of Hippocampal Episodes into Memory under Anti-interference control); check for name collisions before submission.*

## 0. One-paragraph pitch

AHN and AllMem both pair a lossless sliding window with a compressed memory, and both write every token into that memory at the moment it leaves the window, exactly once, in arrival order, with no check on what the write does to older content. In the language of Complementary Learning Systems (CLS) theory, that collapses the hippocampus and the neocortex into a single store and skips the three things CLS says make lifelong learning work: delayed consolidation, selective replay, and interleaving that protects old knowledge. SCHEMA splits the store in two. An AHN-derived **hippocampus** encodes each evicted chunk into a small, separately addressable episode the moment it leaves the window (fast acquisition). An AllMem-style nonlinear **TTT cortex** changes only when an episode is replayed and an **interference gate** confirms that the update helps the new episode more than it damages anchors of previously consolidated knowledge (slow, protected integration). Which episodes get replayed is decided by how often they are reused and whether the cortex already predicts them. Surprising one-off events are deliberately *not* pushed into the cortex; they stay retrievable as exact episodes. This reverses the surprise rule used by Titans and gives a falsifiable claim. The base LLM is frozen throughout, so the model keeps learning at deployment without retraining its foundation weights.

![SCHEMA architecture: three stores plus a gate. Episodes enter the hippocampus at once; only reused, not-yet-predicted episodes are replayed into the cortex, and only through the interference gate.]({{artifact:art_5b64a86c-6a66-49a7-ba92-75145031c97b}})

---

## 1. What the two source papers do, read through CLS

| Property | AHN (Fang et al., ICML 2026) | AllMem (Wang et al., arXiv 2602.13680) | What CLS asks for |
|---|---|---|---|
| Short-term store | Sliding-window KV (default 32k at inference), lossless | Sliding-window attention (2k to 8k) with sinks, lossless | Same: limited, exact working memory |
| Long-term store | One recurrent state per head (Mamba2, DeltaNet or GatedDeltaNet), linear | One nonlinear SwiGLU fast-weight memory per head, chunk-wise TTT with momentum | Two stores: a fast, sparse, pattern-separated episodic store and a slow, distributed one |
| When a write happens | When a token leaves the window | Every chunk, during the forward pass | Immediately for the episodic store; later, during offline replay, for the slow store |
| What gets written | Everything (learned α, β gates control strength) | Everything (learned token-wise learning rate, chunk-wise momentum) | Everything to the episodic store; a selected subset to the slow store |
| How many times | Once | Once | Many times, interleaved with old material |
| Interference control | Decay gate only | Weight normalization placed between read and update; frozen base | Explicit: new learning should not overwrite old structure |
| Training | Self-distillation (KL to full-attention teacher), base frozen, about 0.4% extra parameters | Self-distillation (KL only worked best), base frozen, 66M extra parameters on 0.6B | (not a CLS property) |

Two facts from these papers matter for the design:

1. AHN's gradient probe (its Fig. 6) shows the compressor already prefers informative tokens (numbers, math symbols) over pronouns and special tokens. Selection is learnable from self-distillation, but in AHN it is a *write-strength* decision made once, at eviction, with no second chance.
2. AllMem's own design principle 2 asks for "multiscale" learning with long-term and short-term memory "treated differently", yet both its memory and AHN's operate at a single timescale beyond the window. AllMem's conclusion explicitly points to building a multi-level memory hierarchy as future work. SCHEMA is one concrete answer to that open item.

## 2. Where the prior work you listed leaves a gap

| Work | What it contributes | What it leaves open (for this question) |
|---|---|---|
| Titans (Behrouz et al., 2025) | Deep neural memory written at test time; write strength driven by "surprise" (gradient magnitude with momentum); forgetting gate | Surprise is used as a reason to *consolidate*. CLS suggests surprise is a reason to store an *episode* and consolidate only if the pattern recurs. One store, so no episodic route |
| MIRAS (Behrouz et al., 2025) | Unifies memories by four choices: architecture, attentional bias (inner objective), retention gate, inner optimizer | Retention is regularization inside one store. Nothing decides *which* items deserve the slow store |
| Nested Learning / Hope (Behrouz et al., 2025) | Continuum memory system: modules updated at different frequencies, knowledge flows from fast to slow levels online | Every token still flows through every level. No episodic buffer, no replay, no explicit test of interference |
| Language Models Need Sleep (Lee, McLeish, Goldstein, Fanti, arXiv 2605.26099; later retitled "Do Language Models Need Sleep? Offline Recurrence for Improved Online Inference") | Periodically turns recent context into SSM fast weights with N offline recurrent passes before clearing the KV cache; more sleep helps reasoning | Consolidates the whole context, without selection, without keeping episodes, and without measuring damage to earlier consolidated content |
| Language Models Need Sleep: Learning to Self-Modify and Consolidate Memories (Behrouz, Hashemi, Javanmard, Mirrokni, arXiv 2606.03979) | Sleep phase with knowledge seeding (on-policy distillation plus RL imitation) and dreaming (RL-generated rehearsal curriculum) | Consolidation is expensive (RL, synthetic data) and targets slower parameters. Not a per-chunk, inference-time mechanism |
| Palimpsa (arXiv 2602.09075) | Bayesian metaplasticity: each memory state gets its own plasticity; prevents both catastrophic forgetting and "catastrophic remembering"; Mamba2 is a special case | Within one fixed-size store. Complementary to SCHEMA: it could be the optimizer of SCHEMA's cortex |

**The gap.** None of these combines (a) a separately addressable episodic store that keeps exceptions exact, (b) consolidation decided per episode by reuse and by what the slow store already predicts, and (c) an explicit, measured trade-off between gain on the new episode and loss on old consolidated knowledge, inside an inference-time memory with a frozen base. SCHEMA is built to fill exactly that gap.

## 3. Central hypothesis and the "surprise paradox"

**H0.** A test-time memory continually acquires and preserves knowledge better when it (i) writes every experience into an episodic store at once, (ii) consolidates into slow fast-weights only what recurs and is not yet predicted, and (iii) accepts each consolidation step only if its measured gain on recent episodes exceeds its measured damage on anchors of old knowledge.

**The surprise paradox (a sharp, testable sub-claim).** Consider a delta-rule memory $W$ with unit-norm keys. A one-off exception $(k_e, v_e)$ with error $e_e = v_e - W k_e$ is written with step $\eta_e$:
$$W' = W + \eta_e\, e_e k_e^\top .$$
The recall of a stored regularity $(k_r, v_r)$ changes by
$$W' k_r - W k_r = \eta_e\, (k_e^\top k_r)\, e_e, \qquad \lVert \Delta \rVert = \eta_e\, |c|\, \lVert e_e \rVert,$$
where $c = k_e^\top k_r$. The plain delta rule already writes in proportion to the error. Any surprise gate that makes $\eta_e$ grow with $\lVert e_e\rVert$ makes the damage to correlated regularities grow faster than linearly in surprise. So the most surprising single events do the most harm to what the memory already knows. Under SCHEMA, a one-off exception is not written into the cortex until it recurs, so its first-order damage to the cortex is zero, and it stays exactly retrievable in the hippocampus. This is the CLS story of McClelland, McNaughton and O'Reilly (1995) in two lines, and it predicts a crossover that can be measured: surprise gating should win on single-exposure recall shortly after exposure and lose on regularity retention as the exception rate and key correlation grow.

Two pieces of biology motivate the routing signals. Schema-consistent information can be consolidated quickly (Tse et al., 2007), which motivates the "already predicted, consolidate cheaply" route. Awake hippocampal reactivation tags experiences that are later replayed during sleep, and Yang et al. (Science, 2024) showed that hippocampal sharp-wave ripples select which experiences are kept for memory. This motivates using later reuse of an episode as its consolidation tag.

## 4. Method

### 4.1 Stores (per layer, per KV head)

* **Working memory $\mathcal{W}$.** Frozen base attention over a sliding window of $W$ tokens plus sinks, exactly as in AHN and AllMem.
* **Gist state $h$.** The original AHN module (AHN-GDN recommended), updated with every token that leaves the window, exactly as in AHN Eq. (5). This keeps a lossy summary of everything and makes SCHEMA a strict superset of AHN.
* **Hippocampus $\mathcal{H}$.** $K$ episode slots. When a chunk of $c$ tokens leaves the window, an AHN-style encoder turns its KV pairs into an episode
  $$e_j = \big(\kappa_j,\; \{(\tilde k_{j,i}, \tilde v_{j,i})\}_{i=1}^{m},\; r_j,\; \delta_j,\; a_j\big),$$
  where the $m$ landmark pairs come from attention pooling with $m$ learned queries over the chunk's keys and values, $\kappa_j$ is the normalized mean landmark key used for addressing, $r_j$ is the reuse tag, $\delta_j$ is the cortical error, and $a_j$ is age. The write is immediate and needs no gradient step. This is principle 1, fast acquisition.
* **Cortex $\mathcal{C}$.** A nonlinear fast-weight memory $C_\theta$ (AllMem's residual SwiGLU, with its chunk-wise momentum optimizer and its read, normalize, update ordering). Unlike AllMem, $\mathcal{C}$ is **not** updated by the forward pass. It changes only through consolidation (Section 4.3) and it persists across documents within a stream.
* **Anchor reservoir $\mathcal{A}$.** $R$ pairs $(k_a, \hat v_a)$, where $k_a$ is a landmark key of an episode that was consolidated and $\hat v_a = C_\theta(k_a)$ is the cortex's output *at the time of consolidation*. This is a functional anchor in the sense of learning without forgetting, maintained by reservoir sampling, so it costs a fixed $R(d_k + d_v)$ numbers.

### 4.2 Read path

$$y_t = \mathrm{SWA}(q_t) + \gamma_G(x_t)\, q_t h\, W_o \;+\; g_H(x_t)\, \mathrm{Read}_{\mathcal{H}}(q_t) \;+\; g_C(x_t)\, C_\theta(q_t)$$

$\mathrm{Read}_{\mathcal{H}}$ selects the top $s$ slots by $q_t^\top \kappa_j$ and runs softmax attention over their $s \cdot m$ landmarks, so recall of a stored exception is sparse and nearly exact (pattern separation). Following AllMem, $g_H$ and $g_C$ are per-channel gates initialized to zero, so a trained AHN is the starting point and the new branches must earn their weight. Every read also updates the reuse tag of the selected slots:
$$r_j \leftarrow \lambda_r\, r_j + \textstyle\sum_{t \in \text{chunk}} \alpha_{t,j},$$
where $\alpha_{t,j}$ is the attention mass slot $j$ receives. This is the "awake reactivation" tag.

### 4.3 Consolidation ("micro-sleep")

Consolidation runs when hippocampal occupancy exceeds $\theta K$ or at a document boundary, so wake-time latency is unchanged (as in Lee et al.). Each micro-sleep has $N$ steps.

**Routing (principle 2, selective replay).** For each episode, compute the cortical error on its landmarks with a scale-free loss,
$$\delta_j = \frac{1}{m}\sum_i \frac{\lVert C_\theta(\tilde k_{j,i}) - \tilde v_{j,i}\rVert^2}{\lVert \tilde v_{j,i}\rVert^2}.$$
Then:

| Case | Condition | Action |
|---|---|---|
| Schema-consistent | $\delta_j < \tau_s$ | One small step (or none), add anchors, free the slot |
| Recurring and not yet predicted | $\delta_j \ge \tau_s$ and $r_j \ge r_{\min}$ | Replay with priority $\pi_j \propto r_j^{a}\, \delta_j^{b}$ (default $a = b = 1$) |
| One-off exception | $\delta_j \ge \tau_s$ and $r_j < r_{\min}$ | Keep as an episode. Release only under capacity pressure, lowest $r_j e^{-a_j/T}$ first. Its content still lives in the gist state $h$ |

A learned variant replaces the table by a small priority network $\pi_\phi(r_j, \delta_j, a_j, \text{novelty}_j)$ trained by the outer loop (Section 4.5).

**Interleaving.** Each replay batch $\mathcal{B}$ mixes landmarks of sampled episodes with a sample of anchors from $\mathcal{A}$ at a fixed ratio, so the cortex never trains on new material alone. This is the classic CLS remedy.

**Interference gate (principle 3).** For a candidate update $\Delta\theta$ computed on $\mathcal{B}$, measure
$$G = \ell_{\mathcal{B}}(\theta) - \ell_{\mathcal{B}}(\theta + \Delta\theta), \qquad D = \ell_{\mathcal{A}}(\theta + \Delta\theta) - \ell_{\mathcal{A}}(\theta),$$
with $\ell_{\mathcal{A}}(\theta) = \frac{1}{|\mathcal{A}_b|}\sum_a \lVert C_\theta(k_a) - \hat v_a\rVert^2$. Accept if $G - \rho D > 0$; otherwise halve the step, at most twice, and if it still fails, defer the episode (it stays in $\mathcal{H}$). Before the look-ahead, project out conflict in the A-GEM manner: if $g_{\mathcal{A}}^\top g_{\mathcal{B}} < 0$, use $\tilde g = g_{\mathcal{B}} - \frac{g_{\mathcal{A}}^\top g_{\mathcal{B}}}{\lVert g_{\mathcal{A}}\rVert^2} g_{\mathcal{A}}$. AllMem gives these gradients in closed form for the SwiGLU memory, so one extra gradient and two extra small forward passes per step are the whole cost, and they are paid only during micro-sleep. The ratio $G/(G+D)$ is logged as an interpretable "consolidation efficiency" diagnostic.

**Proposition 1 (interference in a linear cortex).** Let the cortex be linear, $C_W(k) = Wk$, with batch keys $K_{\mathcal{B}}$, values $V_{\mathcal{B}}$, residual $R_{\mathcal{B}} = W K_{\mathcal{B}} - V_{\mathcal{B}}$, and anchors fitted at consolidation so that $R_{\mathcal{A}} = 0$. For the gradient step $W' = W - \eta R_{\mathcal{B}} K_{\mathcal{B}}^\top$:
$$G(\eta) = \eta \langle R_{\mathcal{B}}, R_{\mathcal{B}} M_{\mathcal{BB}}\rangle - \tfrac{\eta^2}{2}\lVert R_{\mathcal{B}} M_{\mathcal{BB}}\rVert^2, \qquad D(\eta) = \tfrac{\eta^2}{2}\lVert R_{\mathcal{B}} M_{\mathcal{BA}}\rVert^2,$$
with $M_{\mathcal{BB}} = K_{\mathcal{B}}^\top K_{\mathcal{B}}$ and $M_{\mathcal{BA}} = K_{\mathcal{B}}^\top K_{\mathcal{A}}$ (losses with the factor $\tfrac12$). The step that maximizes $G - \rho D$ is
$$\eta^\star = \frac{\langle R_{\mathcal{B}}, R_{\mathcal{B}} M_{\mathcal{BB}}\rangle}{\lVert R_{\mathcal{B}} M_{\mathcal{BB}}\rVert^2 + \rho\, \lVert R_{\mathcal{B}} M_{\mathcal{BA}}\rVert^2}.$$
So in a linear memory the interference-aware step size shrinks exactly with the key overlap between the new episode and the anchors. This matches a recent empirical finding on pretrained GLA and GDN models that interference from later writes depends on their overlap with what is already stored (arXiv 2609.33093). A hard-protection variant projects the update onto the null space of the anchor keys, $\Delta W \leftarrow \Delta W (I - K_{\mathcal{A}}(K_{\mathcal{A}}^\top K_{\mathcal{A}} + \epsilon I)^{-1} K_{\mathcal{A}}^\top)$, which gives zero first-order damage at the price of capacity. (Both the gain and damage expressions, the optimal step, and the exception-damage formula in Section 3 were checked numerically against random matrices; the full derivation belongs in the appendix.)

**Release.** After a micro-sleep, any replayed episode whose $\delta_j$ falls below $\tau_s$ is considered consolidated: its slot is freed and a few of its landmark keys, together with the new cortical outputs, enter $\mathcal{A}$. This is the computational analogue of an episode becoming hippocampus-independent.

### 4.4 Algorithm (one layer)

```
on chunk j leaving the window:
    h <- AHN_update(h, KV_j)                    # gist, unchanged from AHN
    e_j <- encode(KV_j); H.insert(e_j)          # fast acquisition, no gradient
on every read at step t:
    y_t <- SWA + gist + g_H * Read_H(q_t) + g_C * C(q_t)
    update reuse tags r of the slots read
if H.occupancy > theta*K or document boundary:   # micro-sleep
    refresh delta_j for all slots
    free schema-consistent slots after one cheap step, add anchors
    for step in 1..N:
        B <- sample recurring, unpredicted episodes by priority, plus anchors
        g_B, g_A <- grads of l_B and l_A at theta
        if g_A . g_B < 0: g_B <- project(g_B, g_A)
        try eta, eta/2, eta/4:
            G, D <- look-ahead gain and damage
            if G - rho*D > 0: theta <- theta + update(eta); break
        else: defer B (episodes stay in H)
    release episodes with delta < tau_s, add their anchors
if H is full and nothing is consolidable:
    release the slot with the lowest decayed reuse (its tokens remain in the gist)
```

### 4.5 Training the meta-parameters (outer loop)

Following AHN and AllMem, the base LLM is frozen and trained only through self-distillation: KL from a full-attention teacher to the SCHEMA student. AHN's own ablation found that KL beat next-token cross-entropy and that randomized windows beat a fixed window, and AllMem found KL alone worked best, so both choices are kept. Trainable: the gist AHN, the landmark pooling queries, the read gates, the cortex projections and its learning-rate and momentum networks, and optionally $\pi_\phi$. Window, chunk size, slot count $K$ and landmark count $m$ are randomized per example, as in both papers.

Two additions are needed for the continual setting:

1. **Stream curriculum.** Training sequences are concatenations of documents in which some facts recur at controlled gaps and others appear once, so the outer loop sees cases where consolidation pays off and cases where it hurts.
2. **Differentiating through consolidation.** Unroll one or two micro-sleep steps inside each training sequence and backpropagate through the cortex update, treating the discrete routing decisions as stop-gradient (or using soft priorities). First-order truncation is the fallback if memory is tight.

### 4.6 Memory and compute budget

Per layer and KV head, the cache is $W$ window pairs, plus $K m$ landmark pairs, plus the gist state ($H^2$ per head, as in AHN), plus cortex fast weights, plus $R$ anchors. For a fair comparison every baseline gets the same total bytes, and the most important baseline is a plain sliding window enlarged by $K m$ tokens. For example, $K = 256$ slots with $m = 32$ landmarks equals 8,192 extra KV pairs, so the matched control is SWA with an extra 8k window. Wake-time cost per token adds one top-$s$ sparse read; micro-sleep cost is $N$ replay steps on batches of a few hundred keys.

## 5. Relation to the closest recent work (novelty risk)

A targeted search turned up several 2026 papers in the same space. None combines the three ingredients inside a test-time memory with a frozen base, but each is close enough that it must be cited and, where possible, run as a baseline.

| Work | Overlap | Difference from SCHEMA |
|---|---|---|
| EVAF (arXiv 2606.29916) | Surprise and valence gated LoRA consolidation with replay and an L2 anchor | Consolidates into LoRA for agent behaviour; uses surprise as a *positive* gate; not an architectural memory |
| SleepGate (Ying Xie, 2026, "Learning to Forget: Sleep-Inspired Memory Consolidation for Resolving Proactive Interference in Large Language Models"; arXiv 2603.14517) | Sleep-inspired forgetting gate and consolidation over the KV cache | Manages the KV cache; no TTT cortex and no interference test |
| TRC² (arXiv 2602.22479) | Hippocampal replay stores (recent ring buffer plus long-term reservoir) for continual LM learning | Replay at training time with next-token loss; not an inference-time memory |
| CLS for continual episodic memory (bioRxiv 2026.08.24.746712) | Per-episode sparse adapters as hippocampus, batch replay as cortex | Its cortical consolidation fine-tunes the base weights, which is what your question rules out |
| ComMem (CoLLAs 2026) | Fast detailed cache plus slow abstract memory for test-time adaptation of VLMs | Classification prototypes, not sequence-model fast weights |
| Fast-weight Product Key Memory (arXiv 2601.00671) | Fast-weight episodic memory alongside GDN and SWA | No consolidation route; a candidate implementation for $\mathcal{H}$ |

Two further papers shape the evaluation. *TTT with KV binding is secretly linear attention* (arXiv 2602.21204, ICML 2026) argues that TTT layers act as learned linear attention rather than memorizing. SCHEMA therefore needs test-retest probes showing that information survives in $\mathcal{C}$ after the window and $\mathcal{H}$ are cleared (Section 6.3). *How Linear Attention Remembers* (arXiv 2609.33093) supports the overlap-based interference model behind Proposition 1.

This search was targeted, not exhaustive. A one-day systematic check (Semantic Scholar plus OpenReview for ICLR 2027 submissions) is recommended before committing.

## 6. Hypotheses, experiments and kill criteria

### 6.1 Falsifiable predictions

| ID | Prediction | Kill criterion |
|---|---|---|
| P1 | On a stream of recurring rules plus one-off exceptions, recurrence routing beats surprise routing on rule retention, and the gap grows with exception rate and key correlation | No crossover across 3 seeds at matched budget: drop the surprise-paradox claim |
| P2 | Exceptions remain recallable from $\mathcal{H}$ at delays where AHN and AllMem have lost them | Exception recall not above an enlarged-window SWA at equal bytes |
| P3 | The interference gate lowers anchor loss and backward forgetting without lowering acquisition | Gate removal changes neither: the gate is decorative |
| P4 | After clearing window and $\mathcal{H}$, facts that recurred are still answerable from $\mathcal{C}$ (test-retest) | No retained recall from $\mathcal{C}$ alone: reframe as an episodic-memory paper |
| P5 | No regression on standard long-context benchmarks relative to AHN at the same window | More than 0.5 average points lost on the LongBench six-task subset |

### 6.2 Stage 0: mechanism tests on synthetic streams (days, single GPU)

A "Rules and Exceptions" stream built on MQAR-style associative recall: $n_r$ key-value rules recur with frequency $f$; exceptions with keys at controlled cosine $c$ to rule keys appear once; at known times some rules change, to measure both forgetting and "catastrophic remembering" (Palimpsa's term). Small models trained from scratch (about 20M to 125M parameters). Compare AHN-GDN, AllMem-style TTT, a Titans-style surprise-gated cortex, a Palimpsa-style metaplastic cortex, uniform replay, sleep without selection (Lee et al. style), and SCHEMA with its ablations. Three seeds minimum.

### 6.3 Stage 1: retrofit a pretrained model (tens of GPU-hours per run)

Start from the public AHN code and checkpoints (ByteDance-Seed/AHN) on Qwen2.5-0.5B or 1.5B Instruct, or from Qwen3-0.6B to compare with AllMem. Self-distill on ChatQA2 plus the stream curriculum. Check P5 on the same splits both papers report: LV-Eval and InfiniteBench at 128k, and the six LongBench tasks with average length above 8k. Add RULER or BABILong for multi-hop memory.

### 6.4 Stage 2: continual learning without retraining the base

1. **Class-incremental in-context classification** on CLINC, Banking and DBpedia, the protocol Nested Learning used for Hope, so results are directly comparable.
2. **Streaming knowledge incorporation** with fictional entities: facts recur 1, 2, 4 or 8 times across a long stream with distractor documents in between, some are later revised, and probes come at increasing delays. The test-retest probe from EVAF is used: query after the window and $\mathcal{H}$ are cleared, so only $\mathcal{C}$ can answer.
3. **Multi-session dialogue memory** (LongMemEval or LoCoMo), which includes knowledge updates and temporal questions.

Metrics: immediate acquisition, retention as a function of delay, backward transfer, exception recall, update accuracy after revisions, fraction of slots consolidated, wake latency, and micro-sleep FLOPs, all at matched memory bytes.

**Baselines.** Enlarged-window SWA (equal bytes), AHN-GDN, AllMem, Titans-style surprise writes into the same cortex, Hope-style multi-frequency cortex, Palimpsa-style metaplastic cortex, Lee et al.-style sleep over all of $\mathcal{H}$, uniform replay, EWC on the cortex, and kNN retrieval over all evicted KVs (Memorizing Transformer style; an unbounded-memory reference, labeled as not budget-matched). Full attention is reported as the teacher reference. Given your earlier finding that simple replay baselines are hard to beat in large pretrained models, uniform replay and the enlarged window are the comparisons reviewers will look at first.

**Ablations.** No $\mathcal{H}$ (reduces to AllMem plus gist), consolidate at eviction (reduces to AHN or AllMem behaviour), surprise priority instead of reuse priority, no interference gate, gate without A-GEM projection, null-space hard protection, no anchors, no gist, learned versus rule priority, and sleep length $N$.

**Diagnostics worth a figure.** Scatter of $\delta_j$ against $r_j$ coloured by each episode's fate; $G/(G+D)$ over a stream; anchor loss over time; per-layer use of $g_H$ and $g_C$.

### 6.5 Rough compute estimate

AHN reports about 10 hours on 32 A100s to train AHNs for a 7B model on 1B tokens, roughly 320 A100-hours. Scaling by parameters times tokens, a 0.5B model on 200M tokens is about 5 A100-hours for an AHN-only run. SCHEMA adds episodic reads, unrolled replay and a teacher on longer streams; a guess of 2 to 4 times overhead gives about 10 to 20 A100-hours per retrofit run. Stage 0 runs should each take a few GPU-hours or less. A full study with three seeds lands in the low hundreds of GPU-hours. These are order-of-magnitude estimates from AHN's single reported number, not measurements.

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Gates $g_H$, $g_C$ stay near zero because self-distillation on single documents never rewards cross-document memory | Stream curriculum with recurring facts; report gate magnitudes per layer |
| Replay compute makes the method look expensive | Report micro-sleep FLOPs separately; vary $N$; show the accuracy against compute frontier |
| Reviewers argue the cortex just acts as linear attention | Test-retest probes after clearing window and $\mathcal{H}$ (P4) |
| Standard long-context benchmarks do not reward consolidation | Stage 2 streams are the main evidence; Stage 1 only shows no regression |
| Small anchor sets give noisy damage estimates | Ablate $R$; use the closed-form linear gate as a low-variance surrogate |
| Close concurrent work appears | Section 5 table, plus the systematic search before writing |

## 8. Claimed contributions (draft)

1. A diagnosis: sliding-window plus memory hybrids (AHN, AllMem, TTT-E2E-style layers) consolidate at eviction, unconditionally, once and in order, and a controlled stream shows when this fails.
2. SCHEMA: a three-store test-time memory with a frozen base, in which an AHN hippocampus feeds a TTT cortex through selective, interleaved, interference-gated replay.
3. An interference-aware consolidation rule with a closed form for linear memories and a cheap look-ahead for nonlinear ones.
4. The surprise paradox: evidence that surprise should route experiences to episodic storage and recurrence should route them to the cortex, the opposite of surprise-gated writing.
5. An evaluation protocol for test-time continual learning with matched memory budgets and test-retest probes.

## 9. Suggested first week

1. Clone the AHN repository and reproduce one small AHN-GDN result as the reference implementation.
2. Write the Rules and Exceptions stream generator with knobs for $f$, $c$, exception rate and rule drift.
3. Implement $\mathcal{H}$, routing and the interference gate on a small from-scratch model, with the linear cortex first (Proposition 1 gives an exact check) and the SwiGLU cortex second.
4. Run P1 and P3 with three seeds against surprise routing and uniform replay. This is the go or no-go for the whole idea.

## Sources read for this proposal

* AHN: Fang, Yu, Zhong, Ye, Xiong, Wei. *Artificial Hippocampus Networks for Efficient Long-Context Modeling.* ICML 2026 (arXiv 2510.07318). Attached PDF.
* AllMem: Wang, Wang, Peng, Qin, Kostelec, Sourmpis, Laborieux, Guo. *AllMem: A Memory-centric Recipe for Efficient Long-context Modeling.* arXiv 2602.13680. Attached PDF.
* Titans (arXiv 2501.00663), MIRAS ("It's All Connected", Behrouz et al., 2025), Nested Learning (arXiv 2512.24695), Language Models Need Sleep (arXiv 2605.26099 and arXiv 2606.03979), Palimpsa (arXiv 2602.09075), and the related 2026 works listed in Section 5, found by web search on 12 Oct 2026.
* McClelland, McNaughton, O'Reilly (1995), Psychological Review; Tse et al. (2007), Science; Yang et al. (2024), Science.
