# SCHEMA: recall, then consolidate

Code, data products and analysis for **SCHEMA**, a test-time memory for frozen language models inspired by
Complementary Learning Systems (CLS) theory, and for the 4-page workshop paper
*"Recall, Then Consolidate: Hippocampal Recall Distillation and Selective Replay for Test-Time Memory"* (`paper/`).

The question behind the project: can a model keep acquiring new experiences, consolidate the useful ones into slow
weights, and keep what it already knew, without retraining its foundation weights?

SCHEMA has three stores on top of a frozen base model:

| store | what it is | when it changes |
|---|---|---|
| working memory | the base model's sliding attention window (plus sinks) | every token |
| hippocampus `H` | exact landmark key/value pairs of evicted tokens | at once, no gradient step |
| cortex `C` | a linear fast-weight matrix (controlled study) or a rank-8 LoRA (LLM study) | only during replay ("micro-sleeps") |

Replay priority combines **reuse** (how often an episode was needed again) with **cortical error** (how badly `C`
alone still handles it). An optional **interference gate** accepts a replay step only if its gain on new episodes
outweighs its damage on anchors of old knowledge. On an LLM the cortex is written with **hippocampal recall
distillation (HRD)**: when the model is seen to recall an `H` landmark, the surrounding passage becomes a recall query,
and the cortex is trained so that the model *without* `H` matches the frozen model that has the episode in context.

The full research record (hypotheses, every experiment including the failed ones, all numbers) is in
[`docs/SCHEMA_research_record.md`](docs/SCHEMA_research_record.md).

## Main results

All contrasts are paired over streams, 95% bootstrap CIs (10,000 resamples), W/T/L = wins/ties/losses.

| experiment | contrast | result |
|---|---|---|
| Controlled study (linear cortex, 135 streams) | recurrence priority vs error-only priority, cortex recall | +0.151 [+0.134, +0.170], 135/135 |
| Controlled study, E1 late-onset rules | same contrast | +0.143 [+0.110, +0.180], 30/30 |
| Controlled study, E2 capacity (50% one-offs) | same contrast at d = 32 / 64 / 128 | +0.332 / +0.238 / +0.140 |
| Qwen2.5-1.5B, pre-registered C1 (40 fresh streams) | SCHEMA vs uniform replay (both HRD), primary | +0.039 [+0.022, +0.056], 33/0/7, **pass** |
| Qwen2.5-1.5B, pre-registered C2 | SCHEMA vs next-token LoRA replay, primary | +0.091 [+0.048, +0.134], 27/0/13, **pass** |
| Qwen2.5-1.5B, pre-registered C3 | reuse-weighted vs error-only priority, cortex recall | +0.022 [-0.025, +0.071], 19/7/14, **fail** |
| Qwen2.5-1.5B, gate (40 streams) | with gate minus without, primary | -0.052 [-0.075, -0.030], 8/2/30 |
| Qwen2.5-3B (40 streams) | SCHEMA vs uniform replay, primary | -0.010 [-0.032, +0.014] (does not replicate) |
| Qwen2.5-3B | SCHEMA vs released AHN-GDN-3B, primary | +0.312 [+0.282, +0.343], 40/0/0 |
| Qwen2.5-1.5B, 16k-token streams (20 streams) | SCHEMA vs uniform replay, primary / cortex recall / one-offs | +0.019 [-0.004, +0.043] / +0.079 [+0.038, +0.122] / -0.041 |

Primary score = 0.5 x cortex-only recall of recurring facts (window and `H` cleared) + 0.5 x recall of one-off facts
more than `W` tokens back.

## Repository layout

```
sim/          Stage 0a controlled study: streams, memories (all arms), runner, grids, analysis   -> sim/README.md
pilot/        Frozen-LLM study: engine, H, LoRA cortex + HRD, AHN port, runner, analysis, figures -> pilot/README.md
tests/        Unit tests (budget equality, Proposition 1 gate, causality, stream determinism, ...)
results/      Every raw result row (.jsonl), summaries (.csv) and analysis reports (.md)          -> results/README.md
paper/        LaTeX source, figure files and the table-filling script for the workshop paper      -> paper/README.md
scripts/      One-command reproduction scripts (environment, data, each experiment, analysis)
docs/         Proposal, implementation plan, Stage 0b specs and the full research record
model0b/, sim0b/, train0b.py, eval0b.py
              Stage 0b (a small model trained from scratch). It never passed its sanity gates and was dropped;
              kept for the record only. Not needed to reproduce any reported result.
```

## Reproducing the results

### 1. Environment
Tested on Linux with an NVIDIA RTX A6000 (48 GB), Python 3.10, torch 2.5.1 (CUDA 12.4), transformers 4.57.1.

```bash
bash scripts/setup_env.sh          # creates .venv, installs requirements, runs the CPU unit tests
```

or manually: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
(install the CUDA build of torch first if you need it). Every command below is run from the repository root with
`PYTHONPATH=.`.

### 2. Data (LLM study only)
```bash
bash scripts/prepare_data.sh       # downloads Qwen2.5-1.5B/3B-Instruct, AHN-GDN-3B and wikitext-103; builds the filler
```
This writes `pilot/data/filler_ids.npy` (about 236 MB, not in git; rebuilt deterministically from wikitext-103 train
shard 0). The fitted landmark selectors `pilot/data/selector_Qwen2.5-{1.5B,3B}-Instruct.pt` **are** in git, so the
exact selectors used for the paper are reused. To refit them (validation seeds only):
`python -m pilot.fit_selector --model Qwen/Qwen2.5-1.5B-Instruct`.

### 3. Experiments

| script | what it reproduces | hardware | time |
|---|---|---|---|
| `scripts/run_stage0a.sh` | Stage 0a P2b grid, E1, E2 (2,565 + 210 + 420 runs) and its analysis | CPU, 30 workers | about 1 h |
| `scripts/run_confirmatory.sh` | Pre-registered test on Qwen2.5-1.5B (tuning on seeds 0 to 2, test on 120 to 139) | 1 to 2 GPUs (24 GB+) | about 12 GPU-h |
| `scripts/run_3b.sh` | Qwen2.5-3B replication (tuning, then 9 arms incl. released AHN-GDN-3B) | 1 GPU (48 GB) | about 10 GPU-h |
| `scripts/run_16k.sh` | 16k-token robustness run on Qwen2.5-1.5B | 1 GPU | about 5 GPU-h |
| `scripts/analyze_all.sh` | All analysis reports, figures and the paper tables from `results/` | CPU | minutes |

The analysis step needs only the files in `results/`, which are all committed, so the tables and figures can be
regenerated without a GPU: `bash scripts/analyze_all.sh`.

### 4. Seeds and settings
* Validation (tuning only): seeds 0, 1, 2. Test: Stage 0a 100 to 104; first LLM tests 100 to 119; confirmatory,
  3B and 16k tests 120 to 139 (16k: 120 to 129). Exception rates 0.2 and 0.5.
* Pre-registration (written before tuning): [`results/prereg_20261116_confirm.md`](results/prereg_20261116_confirm.md).
* Hyperparameters are passed to `pilot.run_pilot --hp` as JSON and recorded in every result row (`hp` field).
* Budget equality across arms is enforced by `tests/test_pilot_budget.py` (W + 2,048 key/value pairs per layer and
  KV head; the LoRA cortex is charged 540 pairs on 1.5B and 676 on 3B via `SCHEMA_BASE=3B`).

## Known caveats
* GPU results are reproducible up to CUDA nondeterminism in bf16; small differences in individual streams are
  expected, means and CIs should match closely.
* `results/v2test_20261116_analysis.md` was produced when the gated arm had 10 of 40 streams; rerunning
  `pilot/analyze_confirm.py` now uses all 40. The 40-stream numbers are in
  `results/schema_final_runs_20261117.md`.
* `stage0a_*` and `pilot_2026110[45]_*` files come from earlier code versions (see the research record); they are
  kept for provenance.
* The 3B and 16k analyses were not pre-registered.

## Citation
Anonymous workshop submission (2026). Please do not distribute before the review period ends.
