#!/usr/bin/env bash
# Pre-registered confirmatory test on Qwen2.5-1.5B-Instruct (results/prereg_20261116_confirm.md).
# Step 1 tunes the pre-registered arm on validation seeds 0-2; step 2 runs every arm on test seeds 120-139.
# Set GPU_A / GPU_B to two GPUs to run halves in parallel (default: both on GPU 0, sequentially).
source "$(dirname "$0")/common.sh"
GPU_A=${GPU_A:-0}; GPU_B=${GPU_B:-$GPU_A}

# 1. validation tuning of hrd_schema_ng: lr {1e-4, 3e-4, 1e-3} x tau {0.5, 1.5}
for lr in 0.0001 0.0003 0.001; do for tau in 0.5 1.5; do
  CUDA_VISIBLE_DEVICES=$GPU_A run --arms hrd_schema_ng --seeds 0,1,2 --tag v2tune_ng_lr${lr}_tau${tau} --retest \
    --hp "{\"n_steps\":8,\"r_min\":2,\"w_min\":0.3,\"lr\":$lr,\"tau\":$tau}"
done; done
# The paper's choice was lr 3e-4, tau 0.5 (best mean validation primary score, 0.277).

# 2. test arms (settings fixed before the test run)
HP_FIXED='{"hrd_uniform":{"n_steps":8,"r_min":2,"w_min":0.3,"lr":0.0001},"hrd_schema":{"n_steps":8,"r_min":2,"w_min":0.3,"lr":0.0003,"tau":0.5},"lora_uniform":{"n_steps":8,"lr":0.001},"lora_arrival":{"n_steps":8,"lr":0.0003},"lora_surprise":{"n_steps":8,"lr":0.0003},"*":{}}'
HP_NG='{"n_steps":8,"r_min":2,"w_min":0.3,"lr":0.0003,"tau":0.5}'
ALL=$SEEDS_TEST_A,$SEEDS_TEST_B
CUDA_VISIBLE_DEVICES=$GPU_A run --arms teacher,swa,swa_big,hippo,hrd_uniform --seeds $ALL --tag v2test_fixA --retest --hp "$HP_FIXED" &
CUDA_VISIBLE_DEVICES=$GPU_B run --arms lora_uniform,lora_arrival,lora_surprise --seeds $ALL --tag v2test_fixB --retest --hp "$HP_FIXED" &
wait
CUDA_VISIBLE_DEVICES=$GPU_A run --arms hrd_schema_ng,hrd_delta_pri,hrd_reuse_pri --seeds $SEEDS_TEST_A --tag v2test_ngA --retest --hp "$HP_NG" &
CUDA_VISIBLE_DEVICES=$GPU_B run --arms hrd_schema_ng,hrd_delta_pri,hrd_reuse_pri --seeds $SEEDS_TEST_B --tag v2test_ngB --retest --hp "$HP_NG" &
wait
# gated SCHEMA (reported arm; about 4x slower)
CUDA_VISIBLE_DEVICES=$GPU_A run --arms hrd_schema --seeds $SEEDS_TEST_A --tag v2test_gateA --retest --hp "$HP_NG" &
CUDA_VISIBLE_DEVICES=$GPU_B run --arms hrd_schema --seeds $SEEDS_TEST_B --tag v2test_gateB --retest --hp "$HP_NG" &
wait
python pilot/analyze_confirm.py    # -> results/v2test_<date>_analysis.md, _contrasts.csv, _arm_means.csv
