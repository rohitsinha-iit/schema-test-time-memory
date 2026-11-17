# Confirmatory test, Qwen2.5-1.5B-Instruct, fresh seeds 120-139 x exc 0.2, 0.5 (16 Nov 2026)

Pre-registration: results/prereg_20261116_confirm.md; tuning choice: results/prereg_20261116_tuning_choice.json.

## Arm means

| arm           |   primary |   retest_rule |   exc_acc_beyond |   exc_acc_gt4W |   rule_acc |   rule_drifted_acc |   retest_exc_acc_beyond |   heldout_text_nll |   wall_s |   n_streams |
|:--------------|----------:|--------------:|-----------------:|---------------:|-----------:|-------------------:|------------------------:|-------------------:|---------:|------------:|
| teacher       |     0.433 |       nan     |            0.865 |          0.835 |      0.87  |              0.658 |                 nan     |            nan     |   12.242 |          40 |
| swa           |     0     |       nan     |            0     |          0     |      0.421 |              0.44  |                 nan     |            nan     |    7.042 |          40 |
| swa_big       |     0.121 |       nan     |            0.241 |          0     |      0.785 |              0.806 |                 nan     |            nan     |    8.853 |          40 |
| hippo         |     0.194 |         0     |            0.387 |          0.318 |      0.787 |              0.7   |                   0     |            nan     |   18.418 |          40 |
| lora_arrival  |     0.094 |         0.096 |            0.092 |          0     |      0.447 |              0.431 |                   0.003 |              2.407 |   92.383 |          40 |
| lora_surprise |     0.097 |         0.087 |            0.107 |          0     |      0.557 |              0.554 |                   0.003 |              2.365 |   96.977 |          40 |
| lora_uniform  |     0.204 |         0.32  |            0.088 |          0.083 |      0.254 |              0.202 |                   0.142 |              4.078 |  166.408 |          40 |
| hrd_uniform   |     0.256 |         0.257 |            0.255 |          0.185 |      0.688 |              0.648 |                   0     |              2.381 |  189.785 |          40 |
| hrd_schema    |     0.25  |         0.259 |            0.241 |          0.194 |      0.642 |              0.483 |                   0     |              2.403 |  992.49  |          10 |
| hrd_delta_pri |     0.284 |         0.347 |            0.221 |          0.163 |      0.657 |              0.596 |                   0     |              2.42  |  237.328 |          40 |
| hrd_reuse_pri |     0.306 |         0.385 |            0.226 |          0.162 |      0.695 |              0.625 |                   0     |              2.415 |  246.3   |          40 |
| hrd_schema_ng |     0.295 |         0.369 |            0.221 |          0.156 |      0.675 |              0.606 |                   0     |              2.418 |  246.688 |          40 |

## Pre-registered decisions (metric primary; paired 95% bootstrap CI over streams; W/T/L)

| comparison | diff | 95% CI | W/T/L | n | exc 0.2 | exc 0.5 |
|---|---|---|---|---|---|---|
| C1 hrd_schema_ng - hrd_uniform | +0.039 | [+0.022, +0.056] **PASS** | 33/0/7 | 40 | +0.034 | +0.045 |
| C2 hrd_schema_ng - lora_uniform | +0.091 | [+0.048, +0.134] **PASS** | 27/0/13 | 40 | +0.119 | +0.064 |

## Pre-registered thesis test (metric retest_rule; paired 95% bootstrap CI over streams; W/T/L)

| comparison | diff | 95% CI | W/T/L | n | exc 0.2 | exc 0.5 |
|---|---|---|---|---|---|---|
| C3 hrd_schema_ng - hrd_delta_pri | +0.022 | [-0.025, +0.071] **FAIL** | 19/7/14 | 40 | +0.041 | +0.004 |

## Reported: primary score (metric primary; paired 95% bootstrap CI over streams; W/T/L)

| comparison | diff | 95% CI | W/T/L | n | exc 0.2 | exc 0.5 |
|---|---|---|---|---|---|---|
| hrd_schema_ng - hrd_reuse_pri | -0.010 | [-0.030, +0.008] | 15/5/20 | 40 | -0.005 | -0.016 |
| hrd_schema_ng - hrd_delta_pri | +0.011 | [-0.013, +0.037] | 21/3/16 | 40 | +0.023 | -0.000 |
| hrd_schema - hrd_schema_ng | -0.037 | [-0.060, -0.012] | 2/1/7 | 10 | -0.035 | -0.038 |
| hrd_schema_ng - hippo | +0.101 | [+0.076, +0.128] | 35/0/5 | 40 | +0.112 | +0.091 |
| hrd_schema_ng - lora_surprise | +0.198 | [+0.170, +0.228] | 40/0/0 | 40 | +0.197 | +0.199 |
| hrd_schema_ng - lora_arrival | +0.201 | [+0.174, +0.226] | 40/0/0 | 40 | +0.207 | +0.195 |
| hrd_schema_ng - swa_big | +0.174 | [+0.146, +0.204] | 39/0/1 | 40 | +0.201 | +0.148 |

## Reported: cortex-only rule recall (metric retest_rule; paired 95% bootstrap CI over streams; W/T/L)

| comparison | diff | 95% CI | W/T/L | n | exc 0.2 | exc 0.5 |
|---|---|---|---|---|---|---|
| hrd_schema_ng - hrd_uniform | +0.112 | [+0.081, +0.143] | 30/9/1 | 40 | +0.099 | +0.125 |
| hrd_reuse_pri - hrd_delta_pri | +0.038 | [+0.002, +0.076] | 18/12/10 | 40 | +0.052 | +0.025 |
| hrd_schema_ng - hrd_reuse_pri | -0.016 | [-0.051, +0.020] | 11/13/16 | 40 | -0.011 | -0.021 |
| hrd_schema_ng - lora_surprise | +0.282 | [+0.237, +0.330] | 39/1/0 | 40 | +0.268 | +0.296 |
| hrd_schema_ng - lora_arrival | +0.273 | [+0.230, +0.317] | 38/2/0 | 40 | +0.270 | +0.276 |
| hrd_schema_ng - lora_uniform | +0.049 | [-0.013, +0.112] | 23/4/13 | 40 | +0.071 | +0.028 |

## Reported: exceptions beyond W (metric exc_acc_beyond; paired 95% bootstrap CI over streams; W/T/L)

| comparison | diff | 95% CI | W/T/L | n | exc 0.2 | exc 0.5 |
|---|---|---|---|---|---|---|
| hrd_schema_ng - hrd_uniform | -0.034 | [-0.050, -0.016] | 10/4/26 | 40 | -0.032 | -0.035 |
| hrd_schema_ng - hippo | -0.166 | [-0.191, -0.142] | 0/1/39 | 40 | -0.173 | -0.160 |
| hrd_schema_ng - swa_big | -0.020 | [-0.051, +0.011] | 17/3/20 | 40 | +0.005 | -0.046 |
| hrd_schema_ng - lora_surprise | +0.114 | [+0.087, +0.144] | 34/5/1 | 40 | +0.127 | +0.102 |

## Reported: held-out text NLL (lower is better) (metric heldout_text_nll; paired 95% bootstrap CI over streams; W/T/L)

| comparison | diff | 95% CI | W/T/L | n | exc 0.2 | exc 0.5 |
|---|---|---|---|---|---|---|
| hrd_schema_ng - hrd_uniform | +0.037 | [+0.032, +0.041] | 40/0/0 | 40 | +0.035 | +0.038 |
| hrd_schema_ng - lora_uniform | -1.660 | [-2.029, -1.350] | 0/0/40 | 40 | -1.769 | -1.552 |
| hrd_schema - hrd_schema_ng | -0.017 | [-0.026, -0.006] | 1/0/9 | 10 | -0.012 | -0.022 |

## Internals (means per stream)

| arm           |   steps_tried |   steps_accepted |   lora_rejected |   lora_shrunk |   hrd_queries |   hrd_uninformative |   recalls |   hrd_episodes_with_query |   hrd_episodes_in_H |
|:--------------|--------------:|-----------------:|----------------:|--------------:|--------------:|--------------------:|----------:|--------------------------:|--------------------:|
| hrd_delta_pri |         222.4 |            222.4 |             0   |           0   |         436.7 |                   0 |    1006.6 |                      64.2 |                93.9 |
| hrd_reuse_pri |         220.4 |            220.4 |             0   |           0   |         443.6 |                   0 |    1023.7 |                      64.4 |                93.9 |
| hrd_schema    |         220.8 |            135.6 |            85.2 |          64.4 |         450.9 |                   0 |    1019.1 |                      64.9 |                93   |
| hrd_schema_ng |         220.4 |            220.4 |             0   |           0   |         447.1 |                   0 |    1015.4 |                      64.3 |                94   |
| hrd_uniform   |         222.4 |            222.4 |             0   |           0   |         450.8 |                   0 |    1049.9 |                      64   |                94   |
