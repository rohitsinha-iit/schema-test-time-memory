#!/usr/bin/env bash
# 16,384-token streams on Qwen2.5-1.5B-Instruct (robustness run), seeds 120-129.
source "$(dirname "$0")/common.sh"
export CUDA_VISIBLE_DEVICES=${GPU:-0}
HP16='{"hrd_uniform":{"n_steps":8,"r_min":2,"w_min":0.3,"lr":0.0001},"hrd_schema_ng":{"n_steps":8,"r_min":2,"w_min":0.3,"lr":0.0003,"tau":0.5},"hrd_delta_pri":{"n_steps":8,"r_min":2,"w_min":0.3,"lr":0.0003,"tau":0.5},"*":{}}'
ARMS=swa_big,hippo,hrd_uniform,hrd_schema_ng,hrd_delta_pri
run --arms $ARMS --seeds 120,121,122,123,124 --n_tokens 16384 --tag v2long16k_A --retest --hp "$HP16"
run --arms $ARMS --seeds 125,126,127,128,129 --n_tokens 16384 --tag v2long16k_B --retest --hp "$HP16"
