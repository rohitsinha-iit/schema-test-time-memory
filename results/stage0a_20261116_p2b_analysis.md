# Stage 0a P2b, E1, E2 analysis (16 Nov 2026)

Rows: 2565; arms 19; cells x seeds per arm 135; errors: 0

## Chosen learning rates (validation seeds 0-2) and edge check

| arm | eta | grid min | grid max | at edge |
|---|---|---|---|---|
| arrival_fifo | 0.1 | 0.01 | 1.0 | no |
| schema | 0.5 | 0.25 | 8.0 | no |
| schema_R128 | 0.5 | 0.25 | 8.0 | no |
| schema_R64 | 1.0 | 0.25 | 8.0 | no |
| schema_agem_only | 1.0 | 0.25 | 4.0 | no |
| schema_dsum | 2.0 | 0.25 | 8.0 | no |
| schema_dsum_rho4 | 0.5 | 0.25 | 8.0 | no |
| schema_eta05 | 0.5 | fixed | fixed | n/a |
| schema_lookahead_only | 2.0 | 0.25 | 8.0 | no |
| schema_nogate | 1.0 | 0.25 | 4.0 | no |
| schema_nogate_eta05 | 0.5 | fixed | fixed | n/a |
| schema_noreconsol | 2.0 | 0.25 | 8.0 | no |
| schema_nosupersede | 0.5 | 0.25 | 8.0 | no |
| schema_redundancy_evict | 0.5 | 0.25 | 8.0 | no |
| schema_surprise_pri | 1.0 | 0.25 | 8.0 | no |
| schema_surprise_pri_dsum | 1.0 | 0.25 | 8.0 | no |
| sleep_all | 8.0 | 0.25 | 8.0 | yes |
| surprise_fifo | 0.1 | 0.01 | 1.0 | no |
| uniform_replay | 2.0 | 0.25 | 8.0 | no |

## Arm means over 27 cells x 5 seeds

| arm                      |   rule |   cortex_rule |   cortex_rule_drifted |   cortex_rule_drifted_old |   exc_recent |   exc_mid |   exc_old |   anchor_loss |   n_anchors |   score |
|:-------------------------|-------:|--------------:|----------------------:|--------------------------:|-------------:|----------:|----------:|--------------:|------------:|--------:|
| schema_dsum              |  0.986 |         0.984 |                 0.98  |                     0.055 |        1     |     0.74  |     0.319 |         0.011 |     256     |   0.904 |
| schema_dsum_rho4         |  0.985 |         0.983 |                 0.98  |                     0.093 |        1     |     0.737 |     0.314 |         0.004 |     256     |   0.902 |
| schema_agem_only         |  0.985 |         0.982 |                 0.978 |                     0.052 |        1     |     0.74  |     0.316 |         0.014 |     256     |   0.902 |
| schema_R64               |  0.985 |         0.982 |                 0.979 |                     0.048 |        1     |     0.915 |     0.438 |         0.012 |      64     |   0.961 |
| schema_R128              |  0.985 |         0.982 |                 0.979 |                     0.068 |        1     |     0.858 |     0.404 |         0.008 |     128     |   0.942 |
| schema_nogate            |  0.984 |         0.982 |                 0.978 |                     0.048 |        1     |     0.739 |     0.315 |         0.018 |     256     |   0.901 |
| schema                   |  0.984 |         0.981 |                 0.978 |                     0.072 |        1     |     0.737 |     0.315 |         0.008 |     256     |   0.901 |
| schema_eta05             |  0.984 |         0.981 |                 0.978 |                     0.072 |        1     |     0.737 |     0.315 |         0.008 |     256     |   0.901 |
| schema_nosupersede       |  0.984 |         0.981 |                 0.978 |                     0.072 |        1     |     0.738 |     0.316 |         0.008 |     256     |   0.901 |
| schema_nogate_eta05      |  0.983 |         0.98  |                 0.977 |                     0.06  |        1     |     0.738 |     0.315 |         0.016 |     256     |   0.9   |
| schema_lookahead_only    |  0.98  |         0.974 |                 0.965 |                     0.102 |        1     |     0.732 |     0.306 |         0.008 |     256     |   0.895 |
| schema_noreconsol        |  0.98  |         0.972 |                 0.941 |                     0.183 |        1     |     0.734 |     0.307 |         0.065 |     256     |   0.895 |
| schema_redundancy_evict  |  0.972 |         0.964 |                 0.951 |                     0.082 |        0.962 |     0.717 |     0.328 |         0.009 |     256     |   0.884 |
| arrival_fifo             |  0.964 |         0.891 |                 0.885 |                     0.022 |        1     |     0.39  |     0.044 |       nan     |       0     |   0.748 |
| uniform_replay           |  0.974 |         0.867 |                 0.861 |                     0.006 |        1     |     0.814 |     0.21  |       nan     |       0     |   0.885 |
| surprise_fifo            |  0.963 |         0.861 |                 0.856 |                     0.034 |        1     |     0.419 |     0.064 |       nan     |       0     |   0.748 |
| schema_surprise_pri_dsum |  0.943 |         0.83  |                 0.824 |                     0.012 |        1     |     0.352 |     0.085 |         0.006 |     219.674 |   0.708 |
| schema_surprise_pri      |  0.943 |         0.83  |                 0.823 |                     0.011 |        1     |     0.352 |     0.085 |         0.006 |     219.674 |   0.708 |
| sleep_all                |  0.931 |         0.672 |                 0.682 |                     0.007 |        1     |     0.263 |     0.034 |       nan     |       0     |   0.622 |

## P1: recurrence routing vs alternatives (metric cortex_rule; paired over streams; 95% bootstrap CI)

| comparison | mean diff | 95% CI | positive | exc 0.05 | exc 0.2 | exc 0.5 |
|---|---|---|---|---|---|---|
| schema - schema_surprise_pri | +0.151 | [+0.134, +0.170] | 135/135 | +0.053 | +0.137 | +0.265 |
| schema - surprise_fifo | +0.120 | [+0.104, +0.138] | 135/135 | +0.035 | +0.095 | +0.231 |
| schema - arrival_fifo | +0.091 | [+0.074, +0.108] | 120/135 | +0.016 | +0.063 | +0.194 |
| schema - uniform_replay | +0.114 | [+0.096, +0.133] | 135/135 | +0.033 | +0.084 | +0.226 |
| schema - sleep_all | +0.309 | [+0.244, +0.378] | 120/135 | +0.375 | +0.356 | +0.197 |
| arrival_fifo - surprise_fifo | +0.030 | [+0.028, +0.031] | 135/135 | +0.019 | +0.033 | +0.037 |

## P2: exceptions beyond the window (metric exc_mid; paired over streams; 95% bootstrap CI)

| comparison | mean diff | 95% CI | positive | exc 0.05 | exc 0.2 | exc 0.5 |
|---|---|---|---|---|---|---|
| schema - arrival_fifo | +0.347 | [+0.284, +0.409] | 90/135 | +0.639 | +0.572 | -0.169 |
| schema - uniform_replay | -0.077 | [-0.107, -0.047] | 25/135 | +0.000 | +0.070 | -0.301 |

## P2: old exceptions (metric exc_old; paired over streams; 95% bootstrap CI)

| comparison | mean diff | 95% CI | positive | exc 0.05 | exc 0.2 | exc 0.5 |
|---|---|---|---|---|---|---|
| schema - arrival_fifo | +0.271 | [+0.198, +0.346] | 45/135 | +0.884 | -0.043 | -0.028 |
| schema - uniform_replay | +0.105 | [+0.070, +0.142] | 45/135 | +0.394 | -0.066 | -0.014 |

## Gate attribution (metric cortex_rule; paired over streams; 95% bootstrap CI)

| comparison | mean diff | 95% CI | positive | exc 0.05 | exc 0.2 | exc 0.5 |
|---|---|---|---|---|---|---|
| schema - schema_nogate | -0.000 | [-0.001, +0.000] | 54/135 | -0.000 | -0.000 | +0.000 |
| schema_agem_only - schema_nogate | +0.001 | [+0.000, +0.001] | 90/135 | +0.001 | +0.001 | +0.000 |
| schema_lookahead_only - schema_nogate | -0.008 | [-0.012, -0.004] | 54/135 | +0.001 | -0.001 | -0.023 |
| schema_dsum_rho4 - schema_nogate | +0.001 | [+0.001, +0.002] | 90/135 | +0.001 | +0.001 | +0.002 |
| schema_eta05 - schema_nogate_eta05 | +0.001 | [+0.001, +0.002] | 105/135 | +0.000 | +0.001 | +0.002 |

## Gate attribution, anchor loss (negative = less damage) (metric anchor_loss; paired over streams; 95% bootstrap CI)

| comparison | mean diff | 95% CI | positive | exc 0.05 | exc 0.2 | exc 0.5 |
|---|---|---|---|---|---|---|
| schema - schema_nogate | -0.010 | [-0.011, -0.009] | 0/135 | -0.009 | -0.010 | -0.011 |
| schema_agem_only - schema_nogate | -0.004 | [-0.005, -0.003] | 6/135 | -0.004 | -0.003 | -0.005 |
| schema_lookahead_only - schema_nogate | -0.010 | [-0.011, -0.009] | 0/135 | -0.009 | -0.009 | -0.011 |
| schema_dsum_rho4 - schema_nogate | -0.014 | [-0.015, -0.013] | 0/135 | -0.013 | -0.013 | -0.015 |

## Anchor budget, cortex (metric cortex_rule; paired over streams; 95% bootstrap CI)

| comparison | mean diff | 95% CI | positive | exc 0.05 | exc 0.2 | exc 0.5 |
|---|---|---|---|---|---|---|
| schema_R64 - schema | +0.001 | [+0.000, +0.001] | 66/135 | +0.000 | +0.000 | +0.002 |
| schema_R128 - schema | +0.001 | [+0.000, +0.001] | 69/135 | -0.000 | +0.001 | +0.002 |

## Anchor budget, exceptions (metric exc_mid; paired over streams; 95% bootstrap CI)

| comparison | mean diff | 95% CI | positive | exc 0.05 | exc 0.2 | exc 0.5 |
|---|---|---|---|---|---|---|
| schema_R64 - schema | +0.177 | [+0.139, +0.218] | 75/135 | +0.000 | +0.029 | +0.504 |
| schema_R128 - schema | +0.121 | [+0.096, +0.147] | 75/135 | +0.000 | +0.029 | +0.334 |
| schema_R64 - uniform_replay | +0.101 | [+0.082, +0.120] | 70/135 | +0.000 | +0.099 | +0.203 |

## Reconsolidation (metric cortex_rule_drifted_old; paired over streams; 95% bootstrap CI)

| comparison | mean diff | 95% CI | positive | exc 0.05 | exc 0.2 | exc 0.5 |
|---|---|---|---|---|---|---|
| schema_noreconsol - schema | +0.111 | [+0.100, +0.121] | 132/135 | +0.094 | +0.119 | +0.118 |

## E1: late-onset rules (first occurrence is a one-off; 10 seeds x 3 exception rates)

| arm                 |   ('cortex_rule_late', 0.05) |   ('cortex_rule_late', 0.2) |   ('cortex_rule_late', 0.5) |   ('cortex_rule_early', 0.05) |   ('cortex_rule_early', 0.2) |   ('cortex_rule_early', 0.5) |   ('exc_mid', 0.05) |   ('exc_mid', 0.2) |   ('exc_mid', 0.5) |
|:--------------------|-----------------------------:|----------------------------:|----------------------------:|------------------------------:|-----------------------------:|-----------------------------:|--------------------:|-------------------:|-------------------:|
| arrival_fifo        |                        0.979 |                       0.934 |                       0.801 |                         0.983 |                        0.94  |                        0.817 |               0.413 |              0.402 |              0.403 |
| schema              |                        0.987 |                       0.985 |                       0.982 |                         0.988 |                        0.987 |                        0.985 |               1     |              0.977 |              0.236 |
| schema_nogate       |                        0.987 |                       0.982 |                       0.981 |                         0.987 |                        0.986 |                        0.984 |               1     |              0.98  |              0.234 |
| schema_surprise_pri |                        0.939 |                       0.862 |                       0.724 |                         0.942 |                        0.865 |                        0.752 |               0.539 |              0.292 |              0.183 |
| sleep_all           |                        0.125 |                       0.836 |                       0.804 |                         0.088 |                        0.842 |                        0.807 |               0.196 |              0.278 |              0.319 |
| surprise_fifo       |                        0.957 |                       0.898 |                       0.765 |                         0.961 |                        0.904 |                        0.782 |               0.45  |              0.432 |              0.42  |
| uniform_replay      |                        0.961 |                       0.924 |                       0.755 |                         0.957 |                        0.925 |                        0.768 |               1     |              0.948 |              0.535 |

| late-rule cortex recall | diff | 95% CI | positive |
|---|---|---|---|
| schema - schema_surprise_pri | +0.143 | [+0.110, +0.180] | 30/30 |
| schema - surprise_fifo | +0.111 | [+0.082, +0.143] | 30/30 |
| schema - arrival_fifo | +0.080 | [+0.052, +0.110] | 28/30 |
| schema - uniform_replay | +0.105 | [+0.072, +0.143] | 30/30 |
| schema - sleep_all | +0.396 | [+0.256, +0.545] | 30/30 |

## E2: cortex capacity (key/value dimension d; 10 seeds x exc 0.2, 0.5)

| d | exc | schema | surprise_pri | schema - surprise_pri [95% CI] | schema - uniform_replay [95% CI] | schema - surprise_fifo |
|---|---|---|---|---|---|---|
| 32 | 0.2 | 0.917 | 0.728 | +0.189 [+0.177, +0.203] | +0.132 [+0.121, +0.146] | +0.161 |
| 32 | 0.5 | 0.901 | 0.569 | +0.332 [+0.321, +0.345] | +0.290 [+0.270, +0.314] | +0.292 |
| 64 | 0.2 | 0.986 | 0.866 | +0.120 [+0.116, +0.125] | +0.062 [+0.058, +0.066] | +0.082 |
| 64 | 0.5 | 0.984 | 0.746 | +0.238 [+0.226, +0.251] | +0.219 [+0.197, +0.241] | +0.208 |
| 128 | 0.2 | 0.991 | 0.930 | +0.062 [+0.059, +0.065] | +0.025 [+0.022, +0.029] | +0.027 |
| 128 | 0.5 | 0.990 | 0.850 | +0.140 [+0.134, +0.148] | +0.105 [+0.096, +0.112] | +0.103 |
