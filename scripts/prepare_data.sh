#!/usr/bin/env bash
# Download models and wikitext-103, then build pilot/data/filler_ids.npy (the filler text between facts).
source "$(dirname "$0")/common.sh"
python - <<'EOF'
from huggingface_hub import snapshot_download, hf_hub_download
for m in ["Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct"]:
    print(m, snapshot_download(m, allow_patterns=["*.json", "*.safetensors", "*.txt", "merges.txt", "vocab.json"]))
print(snapshot_download("ByteDance-Seed/AHN-GDN-for-Qwen-2.5-Instruct-3B"))
for f in ["wikitext-103-raw-v1/train-00000-of-00002.parquet", "wikitext-103-raw-v1/validation-00000-of-00001.parquet"]:
    print(hf_hub_download("Salesforce/wikitext", f, repo_type="dataset"))
EOF
python - <<'EOF'
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from pilot.text_stream import load_filler
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
p = hf_hub_download("Salesforce/wikitext", "wikitext-103-raw-v1/train-00000-of-00002.parquet", repo_type="dataset")
print("filler tokens", len(load_filler(tok, p)))
EOF
ls -la pilot/data
