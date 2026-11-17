# SCHEMA final runs: 3B replication, 16k context, gated arm (17 Nov 2026)
Paired bootstrap, 10,000 resamples, seed 0. Primary = 0.5 x cortex retest rule + 0.5 x exceptions beyond W. Not pre-registered (secondary analyses).

## Qwen2.5-3B, seeds 120 to 139 x exc 0.2/0.5 (tuned on validation: ng lr 1e-4, uniform lr 3e-4)
| arm | n | primary | cortex retest | beyond W | >4W | NLL |
|---|---|---|---|---|---|---|
| teacher | 40 | 0.441 | None | 0.882 | 0.853 | None |
| swa | 40 | 0.0 | None | 0.0 | 0.0 | None |
| swa_big | 40 | 0.124 | None | 0.247 | 0.0 | None |
| hippo | 40 | 0.189 | 0.0 | 0.377 | 0.312 | None |
| ahn | 40 | 0.007 | 0.009 | 0.005 | 0.008 | None |
| swa_ahnbytes | 40 | 0.033 | None | 0.066 | 0.0 | None |
| hrd_uniform | 40 | 0.329 | 0.379 | 0.279 | 0.173 | 2.204 |
| hrd_schema_ng | 40 | 0.319 | 0.349 | 0.289 | 0.159 | 2.186 |
| hrd_delta_pri | 40 | 0.301 | 0.307 | 0.296 | 0.181 | 2.185 |

| contrast | n | mean | 95% CI | win/tie/loss |
|---|---|---|---|---|
| hrd_schema_ng-hrd_uniform|prim | 40 | -0.010 | [-0.032, +0.014] | 17/1/22 |
| hrd_schema_ng-hrd_uniform|rt | 40 | -0.029 | [-0.073, +0.016] | 12/10/18 |
| hrd_schema_ng-hrd_uniform|excB | 40 | +0.010 | [-0.006, +0.026] | 19/10/11 |
| hrd_schema_ng-hrd_delta_pri|prim | 40 | +0.018 | [-0.007, +0.042] | 23/2/15 |
| hrd_schema_ng-hrd_delta_pri|rt | 40 | +0.042 | [-0.008, +0.092] | 22/7/11 |
| hrd_schema_ng-hrd_delta_pri|excB | 40 | -0.006 | [-0.025, +0.012] | 13/11/16 |
| hrd_schema_ng-hippo|prim | 40 | +0.131 | [+0.106, +0.156] | 38/0/2 |
| hrd_schema_ng-hippo|rt | 40 | +0.349 | [+0.304, +0.395] | 39/1/0 |
| hrd_schema_ng-hippo|excB | 40 | -0.088 | [-0.113, -0.064] | 3/2/35 |
| hrd_schema_ng-ahn|prim | 40 | +0.312 | [+0.282, +0.343] | 40/0/0 |
| hrd_schema_ng-ahn|rt | 40 | +0.340 | [+0.294, +0.386] | 39/1/0 |
| hrd_schema_ng-ahn|excB | 40 | +0.285 | [+0.256, +0.315] | 40/0/0 |
| hrd_uniform-hippo|prim | 40 | +0.140 | [+0.115, +0.164] | 39/0/1 |
| hrd_uniform-hippo|rt | 40 | +0.379 | [+0.340, +0.416] | 40/0/0 |
| hrd_uniform-hippo|excB | 40 | -0.098 | [-0.123, -0.075] | 1/2/37 |

## 1.5B, 16k-token streams, seeds 120 to 129 x exc 0.2/0.5
| arm | n | primary | cortex retest | beyond W | >4W | NLL |
|---|---|---|---|---|---|---|
| swa_big | 20 | 0.059 | None | 0.118 | 0.0 | None |
| hippo | 20 | 0.108 | 0.0 | 0.217 | 0.168 | None |
| hrd_uniform | 20 | 0.243 | 0.361 | 0.124 | 0.068 | 2.384 |
| hrd_schema_ng | 20 | 0.262 | 0.441 | 0.084 | 0.051 | 2.433 |
| hrd_delta_pri | 20 | 0.264 | 0.443 | 0.085 | 0.053 | 2.447 |

| contrast | n | mean | 95% CI | win/tie/loss |
|---|---|---|---|---|
| hrd_schema_ng-hrd_uniform|prim | 20 | +0.019 | [-0.004, +0.043] | 13/0/7 |
| hrd_schema_ng-hrd_uniform|rt | 20 | +0.079 | [+0.038, +0.122] | 13/4/3 |
| hrd_schema_ng-hrd_uniform|excB | 20 | -0.041 | [-0.055, -0.027] | 1/2/17 |
| hrd_schema_ng-hrd_delta_pri|prim | 20 | -0.002 | [-0.030, +0.024] | 13/1/6 |
| hrd_schema_ng-hrd_delta_pri|rt | 20 | -0.002 | [-0.045, +0.039] | 6/9/5 |
| hrd_schema_ng-hrd_delta_pri|excB | 20 | -0.001 | [-0.023, +0.020] | 10/2/8 |
| hrd_schema_ng-hippo|prim | 20 | +0.154 | [+0.132, +0.177] | 20/0/0 |
| hrd_schema_ng-hippo|rt | 20 | +0.441 | [+0.394, +0.489] | 20/0/0 |
| hrd_schema_ng-hippo|excB | 20 | -0.133 | [-0.151, -0.115] | 0/0/20 |

## Gated SCHEMA (1.5B, 40 streams) vs no-gate
| arm | n | primary | cortex retest | beyond W | >4W | NLL |
|---|---|---|---|---|---|---|
| hrd_schema | 40 | 0.243 | 0.235 | 0.251 | 0.177 | 2.391 |

| contrast | n | mean | 95% CI | win/tie/loss |
|---|---|---|---|---|
| hrd_schema-hrd_schema_ng|prim | 40 | -0.052 | [-0.075, -0.030] | 8/2/30 |
| hrd_schema-hrd_schema_ng|rt | 40 | -0.135 | [-0.181, -0.090] | 4/6/30 |
| hrd_schema-hrd_schema_ng|excB | 40 | +0.030 | [+0.011, +0.049] | 24/8/8 |
