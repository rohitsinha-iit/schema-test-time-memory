"""HRD diagnostic (validation seed; labels used only to describe what the arm did, never by the arm).

Runs one stream with an HRD arm, then for a sample of re-cued episodes prints the recall query with its target
positions, the teacher's top token there, the student's probability of it, and whether the target token is a fact
token (stream label 1 rule, 2 exception) and whether the episode and the query state the same fact.

  python -m pilot.diag_hrd --arm hrd_uniform --seed 0 --exc 0.5
"""
import argparse
import json

import numpy as np
import torch

from pilot.engine import Engine
from pilot.run_pilot import STREAM, W, arm_config
from pilot.text_stream import load_filler, make_stream


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="hrd_uniform")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exc", type=float, default=0.5)
    ap.add_argument("--hp", default='{"r_min":2,"cue":0.7,"n_steps":8,"lr":0.001,"tau":0.25}')
    ap.add_argument("--n", type=int, default=12)
    a = ap.parse_args()
    eng = Engine("Qwen/Qwen2.5-1.5B-Instruct")
    s = make_stream(np.random.default_rng([a.seed, int(a.exc * 1000)]), eng.tok, load_filler(eng.tok), **dict(STREAM, exc_rate=a.exc))
    w, make = arm_config(a.arm, W, json.loads(a.hp))
    mem = make(eng)
    eng.run(s["tokens"], w, memory=mem)
    lab, dec = s["labels"], lambda ids: eng.tok.decode(ids).replace("\n", " ")
    rows, rng = [], np.random.default_rng(0)
    eps = [e for e in mem.ep_q if mem._hitems([e])[0]]
    is_val = np.zeros(len(lab), bool)                                           # value tokens of every fact
    for (_, v, vpos, _k) in s["facts"]:
        is_val[vpos:vpos + len(eng.tok(v, add_special_tokens=False)["input_ids"])] = True
    tg = []
    for e in eps:
        for q in mem._hitems([e])[0]:
            for p in q["pos"].tolist():
                t = q["start"] + int(p) + 1                                     # stream position of the predicted token
                tg.append((lab[t] > 0, is_val[t]))
    tg = np.array(tg)
    n_q = sum(len(mem.ep_q[e]) for e in mem.ep_q)
    print(f"re-cued episodes with confirmed recalls: {len(eps)} of {len(mem.ep_q)}; queries recorded {n_q}, "
          f"uninformative {mem.st['hrd_uninformative']}; targets {len(tg)}: on a fact sentence {tg[:, 0].mean():.2f}, "
          f"on a value token {tg[:, 1].mean():.2f}")
    # kappa precision: does a recall query state the same (person, attribute) as the re-cued episode?
    fpos = [(k, vpos) for (k, v, vpos, _k) in s["facts"]]
    keys_in = lambda a, b: {k for k, vp in fpos if a <= vp < b}
    same, strict = [], []
    for e, qs in mem.ep_q.items():
        ek = keys_in(mem.ep_start[e], mem.ep_start[e] + len(mem.ep_tok[e]))
        for q in qs:
            same.append(bool(ek & keys_in(q["start"], q["start"] + len(q["ids"]))))
            if "pos" not in q or q["pos"] is None:
                continue
            # stricter target rule: the teacher gives the actual next token >= 0.5 and the base <= 0.05
            ep = mem.ep_tok[e]
            with torch.no_grad(), mem._plain():
                from pilot.lora_cortex import lora_enable
                lora_enable(eng, False)
                t = torch.softmax(eng.model(torch.as_tensor([ep + q["ids"]], device=eng.dev)).logits[0, len(ep):].float(), -1)
                b = torch.softmax(eng.model(torch.as_tensor([q["ids"]], device=eng.dev)).logits[0].float(), -1)
                lora_enable(eng, True)
            for p in range(8, len(q["ids"]) - 1):
                nx = q["ids"][p + 1]
                if t[p, nx] >= 0.5 and b[p, nx] <= 0.05:
                    strict.append((same[-1], is_val[q["start"] + p + 1], lab[q["start"] + p + 1] > 0))
    strict = np.array(strict)
    print(f"recall queries that restate the episode's own fact (re-cue precision): {np.mean(same):.2f} of {len(same)}")
    if len(strict):
        print(f"strict targets: {len(strict)}; on a value token {strict[:, 1].mean():.2f}; in a same-fact query "
              f"{strict[:, 0].mean():.2f}; value AND same fact {(strict[:, 0] & strict[:, 1]).mean():.2f}; "
              f"on any fact sentence {strict[:, 2].mean():.2f}")
    ep_lab = {e: np.bincount(lab[np.asarray(mem.Hpos[mem.Hep == e])].clip(0), minlength=3) for e in eps}
    print("episode composition (filler, rule, exception landmarks) summed over re-cued episodes:",
          np.sum([v for v in ep_lab.values()], 0))
    for e in rng.choice(eps, min(a.n, len(eps)), replace=False).tolist():
        q = mem._hitems([e])[0][-1]
        with torch.no_grad(), mem._plain():
            lp = torch.log_softmax(eng.model(torch.as_tensor([q["ids"]], device=eng.dev)).logits[0].float(), -1)
        print(f"\n=== episode {e}  landmark labels {ep_lab[e].tolist()}  gain0 {q['c0']:.2f}")
        print("EPISODE:", dec(mem.ep_tok[e])[-160:])
        print("QUERY  :", dec(q["ids"]))
        for p, ti, tl in zip(q["pos"].tolist(), q["t_ids"][:, 0].tolist(), q["t_logp"][:, 0].tolist()):
            nxt = q["ids"][p + 1] if p + 1 < len(q["ids"]) else None
            print(f"  pos {p:2d} after '{dec(q['ids'][max(0, p - 4):p + 1])}': teacher top '{dec([ti])}' p={np.exp(tl):.2f}"
                  f"  student p={float(lp[p, ti].exp()):.2f}  actual next '{dec([nxt]) if nxt is not None else '-'}'"
                  f"  label {lab[q['start'] + p + 1] if q['start'] + p + 1 < len(lab) else '-'}")


if __name__ == "__main__":
    main()
