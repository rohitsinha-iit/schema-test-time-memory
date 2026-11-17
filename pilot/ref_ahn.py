"""Reference for the AHN port (runs in AHN's own environment, ~/AHN/.venv): per-token NLL of AHN's merged
Qwen2.5-3B + AHN-GDN model on given tokens, in AHN's inference (prefill) path.

  python pilot/ref_ahn.py <merged model dir> <tokens.npy> <window> <sinks> <out.npy> [bfloat16|float32]
"""
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from ahn.transformer.qwen2_ahn import register_customized_qwen2

register_customized_qwen2()
merged, toks, W, S, out = sys.argv[1], np.load(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
dtype = getattr(torch, sys.argv[6]) if len(sys.argv) > 6 else torch.bfloat16
model = AutoModelForCausalLM.from_pretrained(merged, torch_dtype=dtype).cuda().eval()
model.config.sliding_window, model.config.num_attn_sinks = W, S
ids = torch.as_tensor(toks, device="cuda")[None]
with torch.inference_mode():
    lp = torch.log_softmax(model(ids, use_cache=True).logits[0].float(), -1)
np.save(out, (-lp[:-1].gather(-1, ids[0, 1:, None])[:, 0]).cpu().numpy())
print("saved", out, len(toks))
