# Retrofit R2 gate: KL fine-tuning of AHN-GDN-3B on the fact streams (4 Nov 2026)

## Setup
* **Code:** `pilot/ahn_train.py`.
* **Trainable:** AHN-GDN's 13M parameters only, starting from the released checkpoint; base frozen.
* **Loss:** KL to the full-attention teacher on every position beyond sinks + W. AHN's recipe: AdamW lr 1e-4, wd 0.1, cosine with 10% warmup.
* **Budget:** 400 steps × one 6,144-token sequence (55 minutes on the RTX 4090).
* **Windows:** W random in {512, 1024}, 128 sinks.
* **Data:** 2/3 fact streams (training seeds 1000+, exc ∈ {0.05, 0.2, 0.5}), 1/3 plain wikitext-103 train.
* **Equivalence:** the full-sequence training path matches AHN's own code (mean |ΔNLL| 0.034) and our streaming path (0.052).

## Gate (fixed before training)
Validation seeds 0–2 × exc {0.2, 0.5}; all arms in one run, so streams are paired (`pilot_20261104_r2_gate.jsonl`).
* **Recall bar:** fine-tuned AHN ≥ 0.20 on exceptions beyond W, or ≥ 0.10 retest on recurring facts.
* **Plain-text guard:** NLL ≤ released + 0.05.

| arm | exc beyond W | retest rule | answer log-prob | fact-stream NLL | plain-text NLL |
|---|---|---|---|---|---|
| full attention | 0.865 | | −0.92 | 2.829 | 1.980 |
| window only | 0.005 | | −13.18 | 3.027 | 2.044 |
| window + 512 pairs | 0.068 | | −11.74 | 2.957 | 2.024 |
| released AHN | 0.005 | 0.014 | −10.75 | 3.003 | 2.035 |
| **fine-tuned AHN** | **0.000** | **0.014** | −9.36 | **2.862** | **2.008** |

**Result: the recall bar FAILS, the plain-text guard PASSES.** Per the plan, the write-policy comparison (P1/P3) is not run.

## Reading
* **Better as a gist, not as a store.** Fine-tuning closes 81% of AHN's fact-stream NLL gap to full attention and improves plain text, so the memory learned to carry what helps prediction in general. It did not learn to return an exact 4-digit value or an invented word for a question about a fact beyond the window: answer log-prob is −9.4 against the teacher's −0.9.
* **This is a weak negative.** Two reasons:
  1. The budget is about 0.4% of AHN's own training (400 sequences against about 90k).
  2. The training streams had no in-stream questions. The only KL signal for exact recall comes from a recurring fact restated more than W tokens after its last mention, which is sparse. Stage 0b v2 spec section 2 asks for interleaved probes in training streams; this run did not have them.
