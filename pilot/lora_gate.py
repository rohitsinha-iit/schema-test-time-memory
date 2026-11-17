"""Feasibility gate for a weight-space cortex: LoRA fast weights on the frozen model, written only by gradient
steps on replayed episodes (next-token loss), read by the model itself.

Episodes are built like stream chunks: 40 wikitext tokens, one fact sentence (invented person, attribute,
value), 10 more filler tokens. Per trial:
  1. learn set A (8 facts): full-batch Adam steps on the episodes' LM loss; after steps 1, 2, 5, 10, 20 probe A
     and a never-trained control set with no context at all (question only);
  2. interference: continue from that state and train set B (8 facts) for the same steps, either correlated
     (same persons as A, other attributes) or uncorrelated (new persons); probe A again.
Plain-text NLL (held-out wikitext) is measured before and after. Validation-style seeds only (0 to 2).
Gate (fixed 4 Nov 2026): recall >= 0.5 within <= 10 steps, and >= 10 points of A lost when B is trained.

  python -m pilot.lora_gate --out results/pilot_20261104_lora_gate.jsonl
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from pilot.lora_linear import LoRALinear
from pilot.text_stream import ATTRS, PROBE, TEMPLATES, _value, _word, load_filler
from sim.sanity import git_commit


def add_lora(model, r):
    params = []
    for layer in model.model.layers:
        for parent, name in ((layer.mlp, "up_proj"), (layer.mlp, "down_proj"), (layer.mlp, "gate_proj"),
                             (layer.self_attn, "o_proj")):
            lo = LoRALinear(getattr(parent, name), r)
            setattr(parent, name, lo)
            params += [lo.A, lo.B]
    return params


def make_facts(rng, n, persons=None):
    out = []
    for i in range(n):
        p = persons[i] if persons is not None else f"{_word(rng, 2)} {_word(rng, 2)}"
        a = int(rng.integers(len(ATTRS)))
        out.append(dict(person=p, attr=a, value=_value(rng, ATTRS[a][1])))
    return out


def episode(tok, filler, rng, f):
    o = int(rng.integers(len(filler) - 100))
    s = TEMPLATES[rng.integers(len(TEMPLATES))].format(a=ATTRS[f["attr"]][0], p=f["person"], v=f["value"])
    return filler[o:o + 40].tolist() + tok("\n" + s + "\n", add_special_tokens=False)["input_ids"] + filler[o + 40:o + 50].tolist()


@torch.no_grad()
def recall(model, tok, facts, dev):
    hits = []
    for f in facts:
        a = ATTRS[f["attr"]][0]
        pr = tok(PROBE.format(a=a, p=f["person"]), add_special_tokens=False)["input_ids"]
        tg = tok(" " + f["value"], add_special_tokens=False)["input_ids"]
        ids = torch.as_tensor(pr + tg, device=dev)[None]
        pred = model(ids).logits[0, len(pr) - 1:len(pr) - 1 + len(tg)].argmax(-1)
        hits.append(bool((pred == torch.as_tensor(tg, device=dev)).all()))
    return float(np.mean(hits))


@torch.no_grad()
def text_nll(model, ids):
    lp = torch.log_softmax(model(ids).logits[0].float(), -1)
    return float(-lp[:-1].gather(-1, ids[0, 1:, None]).mean())


def lora_scale(model, on):
    for mod in model.modules():
        if isinstance(mod, LoRALinear):
            mod.enabled = on


def train(model, params, eps, steps, lr, dev, ckpts, probe_fn, anchor=None, lam=0.0):
    """anchor: plain-text ids (1, n); adds lam * KL(base || LoRA model) on them to every step (interleaving)."""
    opt = torch.optim.Adam(params, lr=lr)
    if anchor is not None and lam > 0:
        with torch.no_grad():
            lora_scale(model, False)
            base_lp = torch.log_softmax(model(anchor).logits[0].float(), -1)
            lora_scale(model, True)
    L = max(len(e) for e in eps)
    ids = torch.full((len(eps), L), 0, device=dev); m = torch.zeros(len(eps), L, dtype=torch.bool, device=dev)
    for i, e in enumerate(eps):
        ids[i, :len(e)] = torch.as_tensor(e, device=dev); m[i, :len(e)] = True
    res = {}
    for s in range(1, steps + 1):
        logits = model(ids).logits[:, :-1].float()
        loss = F.cross_entropy(logits[m[:, 1:]], ids[:, 1:][m[:, 1:]])
        if anchor is not None and lam > 0:
            lp = torch.log_softmax(model(anchor).logits[0].float(), -1)
            loss = loss + lam * (base_lp.exp() * (base_lp - lp)).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if s in ckpts:
            res[s] = probe_fn()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--lrs", default="1e-4,3e-4,1e-3")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--out", required=True)
    ap.add_argument("--anchor_lam", default="0", help="comma list of KL-to-base weights on plain-text anchors")
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists")
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(a.model)
    filler = load_filler(tok)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev)
    for p in model.parameters():
        p.requires_grad_(False)
    params = add_lora(model, a.rank)
    init = [p.detach().clone() for p in params]
    text_ids = torch.as_tensor(filler[-2048:], device=dev, dtype=torch.long)[None]          # held-out text (NLL)
    anchor = torch.as_tensor(filler[-6144:-4096], device=dev, dtype=torch.long)[None]        # anchor text (KL)
    ckpts = (1, 2, 5, 10, 20)
    commit = git_commit()
    with open(a.out, "w") as fo:
        for seed in (0, 1, 2):
            for lr, lam in [(float(x), float(y)) for x in a.lrs.split(",") for y in a.anchor_lam.split(",")]:
                for cond in ("correlated", "uncorrelated"):
                    for p, p0 in zip(params, init):
                        p.data.copy_(p0)
                    rng = np.random.default_rng([seed, 77])
                    A = make_facts(rng, 8)
                    ctrl = make_facts(rng, 8)
                    if cond == "correlated":
                        Bf = []
                        for f in A:
                            at = (f["attr"] + 1 + int(rng.integers(len(ATTRS) - 1))) % len(ATTRS)
                            Bf.append(dict(person=f["person"], attr=at, value=_value(rng, ATTRS[at][1])))
                    else:
                        Bf = make_facts(rng, 8)
                    epsA = [episode(tok, filler, rng, f) for f in A]
                    epsB = [episode(tok, filler, rng, f) for f in Bf]
                    nll0 = text_nll(model, text_ids)
                    base = dict(A=recall(model, tok, A, dev), ctrl=recall(model, tok, ctrl, dev))
                    model.train()
                    learn = train(model, params, epsA, 20, lr, dev, ckpts,
                                  lambda: dict(A=recall(model, tok, A, dev), ctrl=recall(model, tok, ctrl, dev)), anchor, lam)
                    nll1 = text_nll(model, text_ids)
                    interf = train(model, params, epsB, 20, lr, dev, ckpts,
                                   lambda: dict(A=recall(model, tok, A, dev), B=recall(model, tok, Bf, dev)), anchor, lam)
                    nll2 = text_nll(model, text_ids)
                    row = dict(seed=seed, lr=lr, anchor_lam=lam, rank=a.rank, cond=cond, base=base, learn=learn, interf=interf,
                               text_nll=[nll0, nll1, nll2], model=a.model, git_commit=commit)
                    fo.write(json.dumps(row) + "\n"); fo.flush()
                    print(f"seed {seed} lr {lr:g} lam {lam:g} {cond:12s} base A {base['A']:.2f} | learn A "
                          + " ".join(f"s{s}:{v['A']:.2f}" for s, v in learn.items())
                          + f" ctrl {learn[20]['ctrl']:.2f} | after B: A " + " ".join(f"s{s}:{v['A']:.2f}" for s, v in interf.items())
                          + f" B s20 {interf[20]['B']:.2f} | text nll {nll0:.3f}->{nll1:.3f}->{nll2:.3f}", flush=True)


if __name__ == "__main__":
    main()
