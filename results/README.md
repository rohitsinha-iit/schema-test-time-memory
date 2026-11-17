# `results/`

Every raw result row is a JSON line with the arm, seed, exception rate, all metrics, internals, the
hyperparameters (`hp`) and the code commit. File names: `<stage>_<date>_<tag>.jsonl`.

## Files behind the paper

| file(s) | experiment |
|---|---|
| `stage0a_20261116_p2b_{val,grid}.jsonl`, `_p2b_eta.json`, `_p2b_val_scores.csv` | Stage 0a P2b retune |
| `stage0a_20261116_e1.jsonl`, `stage0a_20261116_e2_d{32,64,128}.jsonl` | E1 late-onset rules, E2 capacity |
| `stage0a_20261116_p2b_analysis.md` | Stage 0a report (all contrasts) |
| `prereg_20261116_confirm.md` | Pre-registration of the confirmatory test |
| `pilot_20261116_v2val_sanity.jsonl`, `pilot_*_v2tune_ng_*.jsonl` | Validation sanity and tuning (seeds 0 to 2) |
| `pilot_20261116_v2test_{fixA,fixB,ngA,ngB,gateA,gateB}.jsonl` | Confirmatory test, seeds 120 to 139 |
| `v2test_20261116_{analysis.md,contrasts.csv,arm_means.csv}` | Confirmatory analysis (gate arm at 10 streams when written) |
| `pilot_*_v3b_tune_*.jsonl`, `v3b_tuning_choice.json`, `pilot_20261117_v3b_testA.jsonl` | Qwen2.5-3B tuning and test |
| `pilot_20261117_v2long16k_{A,B}.jsonl` | 16k-token run |
| `schema_final_runs_20261117.md` | Gate (40 streams), 3B and 16k analysis |
| `fig1_stage0a.*`, `fig2_neural.*` | Paper figures |

## Earlier experiments (provenance)
* `stage0a_20261103_*`: first Stage 0a grid (M1).
* `stage0b_*`: Stage 0b from-scratch model (failed sanity gates).
* `pilot_20261104_*`, `*_20261104_report.md`: training-free pilot, linear cortex, AHN retrofit (R0, R2), LoRA cortex
  with next-token loss.
* `pilot_20261105_*`, `hrd_20261105_test_report.md`, `schema_ideas_20261104.md`: first HRD test.

Note: test streams from before the evening of 4 Nov were generated before a stream-determinism fix (see
`pilot_20261104_report.md`, "Correction"); paired comparisons within one file are valid.
