"""Train the cortex read gates gamma (one scalar per layer and query head, initialised to 0) of a linear-cortex arm
by distillation from the full-attention teacher (proposal 4.5): soft cross-entropy to the teacher's top-32 answer
distribution on the probe answer tokens, with the arm's own memory state, in normal and in retest mode (window and H
cleared). Memory updates are not differentiated (first-order truncation). Validation seeds 0 to 1 only.

  python -m pilot.train_gates --arm lin_schema --hp '{"theta": 0.98}'
"""
import argparse
import json

import numpy as np
import torch

from pilot.engine import Engine
from pilot.linear_cortex import LIN_SPECS, LinearSchemaMemory, gates_path
from pilot.run_pilot import STREAM, W
from pilot.text_stream import load_filler, make_stream


def teacher_targets(eng, s, k=32, bs=16):
    eng.run(s["tokens"], 10**9)
    out = []
    with torch.no_grad():
        for i in range(0, len(s["probes"]), bs):
            for lg in eng.probe_logits(s["probes"][i:i + bs], len(s["tokens"])):
                p, ix = torch.softmax(lg, -1).topk(k, -1)
                out.append((ix, p / p.sum(-1, keepdim=True)))
    return out


def soft_ce(logits, targets):
    loss = 0.0
    for lg, (ix, p) in zip(logits, targets):
        loss = loss - (p * torch.log_softmax(lg, -1).gather(-1, ix)).sum(-1).mean()
    return loss / len(logits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--hp", default="{}")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--bs", type=int, default=16)
    a = ap.parse_args()
    eng = Engine(a.model)
    for p in eng.model.parameters():
        p.requires_grad_(False)
    filler = load_filler(eng.tok)
    extra_w, kw = LIN_SPECS[a.arm]
    hp = json.loads(a.hp)
    streams = [make_stream(np.random.default_rng([sd, int(ex * 1000)]), eng.tok, filler, **dict(STREAM, exc_rate=ex))
               for sd in (0, 1) for ex in (0.2, 0.5)]
    targets = [teacher_targets(eng, s) for s in streams]
    gamma = torch.nn.Parameter(torch.zeros(eng.L, eng.Hq, device=eng.dev))
    opt = torch.optim.Adam([gamma], lr=a.lr)
    for ep in range(a.epochs):
        for s, tg in zip(streams, targets):
            mem = LinearSchemaMemory(eng, gates=gamma.detach(), **{**hp, **kw})
            mem.gamma = gamma
            eng.run(s["tokens"], W + extra_w, memory=mem)
            n, P = len(s["tokens"]), s["probes"]
            losses = []
            for retest in (False, True):
                if retest:
                    mem.retest_mode()
                for i in range(0, len(P), a.bs):
                    with torch.enable_grad():
                        lg = eng.probe_logits(P[i:i + a.bs], n, retest=retest)
                        loss = soft_ce(lg, tg[i:i + a.bs]) * len(lg) / len(P)
                        loss.backward()
                    losses.append(loss.item() * len(P) / len(lg))
            opt.step(); opt.zero_grad()
            print(f"epoch {ep} stream loss normal {np.mean(losses[:len(losses)//2]):.3f} retest {np.mean(losses[len(losses)//2:]):.3f} "
                  f"|gamma| {gamma.abs().mean().item():.4f} max {gamma.abs().max().item():.3f}", flush=True)
    torch.save(gamma.detach().cpu(), gates_path(a.model, a.arm))


if __name__ == "__main__":
    main()
