"""Diagnostic (validation seeds; reads labels, never an arm): does a reuse signal separate recurring-rule landmarks
from exception landmarks? AUC (rule > exception) of
  (a) read mass r: attention mass a landmark receives while in H (proposal 4.2),
  (b) cue matches: number of later landmarks whose pre-RoPE key has cosine >= c with it, per layer (Stage 0a style).
H keeps every landmark (no eviction) so each signal sees the whole stream.
"""
import sys

import numpy as np
import torch
import torch.nn.functional as F

from pilot.engine import Engine
from pilot.memory import Hippocampus
from pilot.run_pilot import STREAM, W
from pilot.text_stream import load_filler, make_stream


def auc(score, pos):
    s1, s0 = score[pos], score[~pos]
    if len(s1) == 0 or len(s0) == 0:
        return float("nan")
    allv = np.concatenate([s1, s0]); r = allv.argsort().argsort().astype(float) + 1
    return (r[: len(s1)].sum() - len(s1) * (len(s1) + 1) / 2) / (len(s1) * len(s0))


def main(model):
    eng = Engine(model)
    filler = load_filler(eng.tok)
    res = {}
    for seed in (0, 1):
        for ex in (0.2, 0.5):
            s = make_stream(np.random.default_rng([seed, int(ex * 1000)]), eng.tok, filler, **dict(STREAM, exc_rate=ex))
            h = Hippocampus(eng, cap=10**6)
            eng.run(s["tokens"], W, memory=h)
            lab = s["labels"][h.pos]
            fact = lab != 0
            is_rule = lab[fact] == 1
            r = torch.stack([x for x in h.r]).sum((0, 1)).cpu().numpy()          # read mass summed over layers, heads
            res.setdefault("read_mass", []).append(auc(r[fact], is_rule))
            for l in (2, eng.L // 4, eng.L // 2, 3 * eng.L // 4, eng.L - 1):
                k = F.normalize(h.k[l].float(), dim=-1)                          # (Hkv, n, d)
                cos = torch.einsum("hnd,hmd->hnm", k, k).amax(0)                  # max over KV heads
                later = torch.triu(torch.ones_like(cos, dtype=torch.bool), 1)
                for c in (0.9, 0.95, 0.98):
                    cnt = ((cos >= c) & later).sum(1).cpu().numpy()
                    res.setdefault(f"cue L{l} c{c}", []).append(auc(cnt[fact], is_rule))
            print(f"seed {seed} exc {ex}: landmarks {len(h.pos)}, fact {fact.sum()}, rule {is_rule.sum()}", flush=True)
    for k, v in res.items():
        print(f"{k:22s} AUC rule>exc {np.nanmean(v):.3f}  ({', '.join(f'{x:.2f}' for x in v)})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-1.5B-Instruct")
