#!/usr/bin/env bash
# Qwen2.5-3B-Instruct replication: lr tuning on validation seeds 0,1, then 9 arms on test seeds 120-139.
source "$(dirname "$0")/common.sh"
export SCHEMA_BASE=3B CUDA_VISIBLE_DEVICES=${GPU:-0}
M=Qwen/Qwen2.5-3B-Instruct
[ -f pilot/data/selector_Qwen2.5-3B-Instruct.pt ] || python -m pilot.fit_selector --model $M
for arm in hrd_schema_ng hrd_uniform; do for lr in 0.0003 0.0001; do
  run --model $M --arms $arm --seeds 0,1 --tag v3b_tune_${arm}_lr$lr --retest \
    --hp "{\"n_steps\":8,\"r_min\":2,\"w_min\":0.3,\"lr\":$lr,\"tau\":0.5}"
done; done
HP=$(python - <<'EOF'
import json, glob, numpy as np
best = {}
for arm in ("hrd_schema_ng", "hrd_uniform"):
    sc = {}
    for f in glob.glob(f"results/pilot_*_v3b_tune_{arm}_lr*.jsonl"):
        rows = [json.loads(l) for l in open(f) if l.strip()]
        sc[float(f.split("_lr")[1][:-6])] = np.mean([0.5 * x["retest_rule_acc"] + 0.5 * x["exc_acc_beyond"] for x in rows])
    best[arm] = dict(n_steps=8, r_min=2, w_min=0.3, lr=max(sc, key=sc.get), tau=0.5); best[arm + "_scores"] = sc
json.dump(best, open("results/v3b_tuning_choice.json", "w"), indent=1)
hp = {k: v for k, v in best.items() if not k.endswith("_scores")}; hp["hrd_delta_pri"] = hp["hrd_schema_ng"]; hp["*"] = {}
print(json.dumps(hp))
EOF
)
echo "chosen: $HP"     # paper run: hrd_schema_ng lr 1e-4, hrd_uniform lr 3e-4
run --model $M --arms teacher,swa,swa_big,hippo,ahn,swa_ahnbytes,hrd_uniform,hrd_schema_ng,hrd_delta_pri \
    --seeds $SEEDS_TEST_A,$SEEDS_TEST_B --tag v3b_testA --retest --hp "$HP"
