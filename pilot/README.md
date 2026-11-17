# `pilot/`: SCHEMA on frozen Qwen2.5 models

## Pipeline
1. `text_stream.py` builds a stream: facts about invented people (16 recurring, paraphrased; one-off exceptions at
   rate 0.2 or 0.5 that share a person with a recurring fact; 25% of values changed at the midpoint) between
   wikitext-103 passages (about 74% filler). Questions are asked after the stream and scored by exact match.
2. `engine.py` runs the frozen model chunk by chunk (64 tokens) with a sliding window W = 1,024 plus 4 sinks; extra
   key/value entries (H landmarks) join the attention softmax.
3. `fit_selector.py` fits the landmark selector (a linear probe on the layer-14 hidden state, self-distilled from the
   full-attention teacher's question attention on validation streams; no labels). Saved to `data/selector_*.pt`.
4. `lora_cortex.py` implements the rank-8 LoRA cortex (MLP and o_proj), next-token and HRD write rules, priorities,
   release, anchors and the gate, and defines the arm table `LORA_SPECS` with the byte budget of each arm.
5. `run_pilot.py` runs arms on seeds and writes one JSON line per (arm, seed, exception rate) to
   `results/pilot_<date>_<tag>.jsonl`, including all metrics, internals, the commit and the hyperparameters.

## Arms (`--arms`)

| arm | meaning |
|---|---|
| `teacher` | full attention over the whole stream (upper reference) |
| `swa`, `swa_big` | window only; window enlarged by the same extra bytes |
| `hippo` | H only, no cortex |
| `ahn`, `swa_ahnbytes` | released AHN-GDN-3B memory (3B only); window enlarged by AHN's bytes |
| `lora_arrival`, `lora_surprise` | LoRA written at arrival with next-token loss (AllMem-like), or with surprise-scaled steps (Titans-like) |
| `lora_uniform` | uniform replay over H with next-token loss |
| `hrd_uniform` | HRD write rule, uniform replay over recalled episodes |
| `hrd_schema` | HRD + SCHEMA priority + anchors + interference gate |
| `hrd_schema_ng` | **pre-registered SCHEMA arm**: HRD + reuse x error priority, no gate, anchor bytes returned to H |
| `hrd_delta_pri` | HRD, error-only (surprise-like) priority |
| `hrd_reuse_pri` | HRD, reuse-only priority |

Older arms from 4 to 5 Nov (`schema*`, `lora_schema*`, `hrd_schema_nogate`, linear cortex) remain in the code for
provenance.

## Main options of `run_pilot.py`
`--model` (default Qwen/Qwen2.5-1.5B-Instruct), `--arms`, `--seeds`, `--exc_rates` (default 0.2,0.5), `--tag`,
`--retest` (also probe with window and H cleared), `--n_tokens` (default 8192), `--hp` (JSON, either one dict for
all arms or `{"arm": {...}, "*": {}}`). Set `SCHEMA_BASE=3B` for the 3B LoRA budget.

Example (one validation seed, both exception rates):
```bash
PYTHONPATH=. python -m pilot.run_pilot --arms swa_big,hippo,hrd_schema_ng --seeds 0 --tag demo --retest \
  --hp '{"n_steps":8,"r_min":2,"w_min":0.3,"lr":0.0003,"tau":0.5}'
```

## Analysis
* `analyze_confirm.py`: pre-registered decisions C1 to C3 and reported contrasts -> `results/v2test_20261116_*`.
* `analyze_scale.py`: gate (40 streams), 3B and 16k -> `results/schema_final_runs_20261117.md`.
* `make_figs.py`: Figures 1 and 2 of the paper. `paired.py`, `summarize.py`: older summaries.

## Other modules (earlier experiments, see the research record)
`ahn_port.py`, `gdn.py`, `ref_ahn.py`, `check_ahn_port.py`, `ahn_train.py` (AHN-GDN-3B port and the R2 fine-tune);
`linear_cortex.py`, `fit_qmap.py`, `train_gates.py` (linear cortex with trained read gates);
`fit_kappa.py`, `diag_*.py` (reuse-signal diagnostics); `memory.py`, `schema_memory.py` (training-free pilot).
