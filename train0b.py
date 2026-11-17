"""Stage 0b backbone training (stage0b_spec.md, sections 5.1 and 11) and the G0 learnability check.

Backbone + gist from scratch (or from --init), AdamW, cosine schedule, randomized window per batch. Loss
(--loss, spec section 13): "full_plus_masked" (default) = mean next-token CE over all positions + mean CE over
the masked targets (probe answers and predictable fact values); "masked" = the masked term only (revisions 1, 2). Batches are generated in worker processes from (seed, step), so every
run with the same seed and stream options sees the same data in the same order.

Every --eval_every steps, held-out probe accuracy is measured on 64 sequences with the training stream options
(seed [seed, 99]): over all in-stream probes, and over probes whose key last occurred within --eval_window
tokens. With --stop_acc > 0 training stops once the within-window accuracy reaches it.

Runs:
  original G0 smoke:  python train0b.py --steps 2000 --tag g0 --g0
  sanity S1:          python train0b.py --tag s1 --seq 512 --batch 32 --micro 32 --layers 2 --dense --no_gist
                        --key_len 1 --zipf 0 --filler_max 0 --probe_p 0.5 --steps 1500 --stop_acc 0.95
  sanity S2 / phase A: --seq 512 --batch 32 --micro 32 --zipf 0 --filler_max 1 --probe_p 0.5 --steps 3000
                        --stop_acc 0.95 (S2) / 0.90 (phase A)
  phase B:            --init <phase A ckpt> --zipf 0 --probe_p 0.3 --steps 2000
"""
import argparse
import datetime
import json
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from eval0b import g0_eval, shortcut_hits
from model0b.backbone import Backbone
from model0b.reference import GPT
from sim.sanity import git_commit
from sim0b.stream import VOCAB, make_batch, make_sequence

STREAM_KEYS = ("zipf", "filler_min", "filler_max", "probe_p", "key_len", "probe_delay")


def _batch(args):
    seed, step, batch, n_tokens, stream = args
    return make_batch(np.random.default_rng([seed, step]), batch, n_tokens, **stream)


def loss_terms(logits, y, m, n_masked, n_all, kind="full_plus_masked"):
    """Contribution of one micro-batch: (loss, masked CE). Sums are divided by the whole batch's counts, so
    summing over micro-batches gives mean CE over all positions (if kind is "full_plus_masked") + mean CE over
    masked targets."""
    ce = F.cross_entropy(logits.float().flatten(0, 1), y.flatten(), reduction="none").view_as(y)
    masked = (ce * m).sum() / n_masked
    if kind == "masked":
        return masked, masked
    return ce.sum() / n_all + masked, masked


def lr_at(step, steps, lr, warmup, schedule="cosine"):
    """Linear warmup, then cosine decay to zero at `steps`, or constant (sanity runs, spec section 12)."""
    if step < warmup:
        return lr * (step + 1) / warmup
    if schedule == "constant":
        return lr
    return 0.5 * lr * (1 + np.cos(np.pi * (step - warmup) / max(1, steps - warmup)))


@torch.no_grad()
def heldout_probe_acc(model, seqs, window, micro, dev="cuda"):
    """Accuracy on in-stream probe answers: all probes, and probes whose key last occurred within `window`,
    next to the accuracy of the two key-free shortcuts on the same probes."""
    model.eval()
    hit, within, rec, frq = [], [], [], []
    for i in range(0, len(seqs), micro):
        part = seqs[i:i + micro]
        tok = torch.as_tensor(np.stack([s["tokens"] for s in part]), device=dev)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = model(tok[:, :-1], window).argmax(-1).cpu().numpy()
        for b, s in enumerate(part):
            p = s["probe_pos"]
            hit.append(pred[b, p - 1] == s["tokens"][p]); within.append(s["probe_delay"] <= window)
            r, f = shortcut_hits(s["tokens"], p); rec.append(r); frq.append(f)
    model.train()
    hit, within, rec, frq = (np.concatenate(x) for x in (hit, within, rec, frq))
    acc = lambda x, m=None: float(x.mean()) if m is None else (float(x[m].mean()) if m.any() else float("nan"))
    return dict(probe_acc_all=acc(hit), probe_acc_within=acc(hit, within),
                shortcut_recent_all=acc(rec), shortcut_freq_all=acc(frq),
                shortcut_recent_within=acc(rec, within), shortcut_freq_within=acc(frq, within),
                n_probes=int(len(hit)), n_within=int(within.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--micro", type=int, default=8, help="micro-batch size (gradient accumulation to --batch)")
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--schedule", choices=("cosine", "constant"), default="cosine")
    ap.add_argument("--windows", default="64,128,256")
    ap.add_argument("--loss", choices=("masked", "full_plus_masked"), default="full_plus_masked")
    # model
    ap.add_argument("--model", choices=("backbone", "gpt"), default="backbone", help="gpt = reference model R1")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--dense", action="store_true", help="dense causal attention (S1)")
    ap.add_argument("--no_gist", action="store_true")
    ap.add_argument("--init", default="", help="checkpoint to start from (phase B)")
    # stream options (defaults = original training streams)
    ap.add_argument("--zipf", type=float, default=1.0)
    ap.add_argument("--filler_min", type=int, default=0)
    ap.add_argument("--filler_max", type=int, default=3)
    ap.add_argument("--probe_p", type=float, default=0.15)
    ap.add_argument("--key_len", type=int, default=2)
    ap.add_argument("--probe_delay", choices=("loguniform", "uniform_key"), default="loguniform")
    # held-out probe accuracy and early stop
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--eval_window", type=int, default=128)
    ap.add_argument("--stop_acc", type=float, default=0.0)
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tag", default="g0")
    ap.add_argument("--jsonl", default="", help="append a summary line here (default results/stage0b_<date>_<tag>.jsonl)")
    ap.add_argument("--date", default=f"{datetime.date.today():%Y%m%d}")
    ap.add_argument("--g0", action="store_true", help="run the G0 evaluation after training")
    ap.add_argument("--g0_windows", default="128,512")
    ap.add_argument("--max_minutes", type=float, default=0, help="stop early after this wall time (0 = off)")
    a = ap.parse_args()

    base = f"results/stage0b_{a.date}_{a.tag}"
    jsonl = a.jsonl or base + ".jsonl"
    for path in (base + "_curve.csv",) + (() if a.jsonl else (jsonl,)):
        if os.path.exists(path):
            raise SystemExit(f"{path} exists; results files are never overwritten")
    os.makedirs("results", exist_ok=True); os.makedirs("ckpt0b", exist_ok=True)
    commit = git_commit()
    dev = "cuda"
    torch.manual_seed(a.seed)
    if a.model == "gpt":
        model = GPT(VOCAB, n_layers=a.layers, max_len=a.seq).to(dev)
    else:
        model = Backbone(VOCAB, n_layers=a.layers, dense=a.dense, gist=not a.no_gist).to(dev)
    if a.init:
        model.load_state_dict(torch.load(a.init)["model"])
    n_params = sum(p.numel() for p in model.parameters())
    emb = {id(m.weight) for m in model.modules() if isinstance(m, torch.nn.Embedding)}
    decay = [p for p in model.parameters() if p.dim() >= 2 and id(p) not in emb]       # no decay on embeddings
    other = [p for p in model.parameters() if not (p.dim() >= 2 and id(p) not in emb)]
    opt = torch.optim.AdamW([dict(params=decay, weight_decay=a.wd), dict(params=other, weight_decay=0.0)],
                            lr=a.lr, betas=(0.9, 0.95))
    windows = [int(w) for w in a.windows.split(",")]
    stream = {k: getattr(a, k) for k in STREAM_KEYS}
    wrng = np.random.default_rng([a.seed, 7])
    hrng = np.random.default_rng([a.seed, 99])
    heldout = [make_sequence(hrng, a.seq + 1, exc_rate=float(hrng.choice((0.05, 0.2, 0.5))), corr=0.3, **stream)
               for _ in range(64)]
    print(f"params {n_params:,}  vocab {VOCAB}  stream {stream}", flush=True)

    curve, t0, tokens = [], time.time(), 0
    torch.cuda.reset_peak_memory_stats()
    acc_l = loss_l = mloss_l = n_l = 0.0
    last_step, stop_reason, ho = a.steps, "steps", {}
    with Pool(a.workers) as pool:
        it = pool.imap(_batch, ((a.seed, s, a.batch, a.seq + 1, stream) for s in range(a.steps)))
        for step in range(a.steps):
            tok, mask = next(it)
            tok, mask = torch.as_tensor(tok, device=dev), torch.as_tensor(mask, device=dev)
            W = int(wrng.choice(windows))
            for g in opt.param_groups:
                g["lr"] = lr_at(step, a.steps, a.lr, a.warmup, a.schedule)
            opt.zero_grad(set_to_none=True)
            n_tot, n_all = mask[:, 1:].sum(), mask[:, 1:].numel()
            loss_s = mloss_s = acc_s = 0.0
            for s in range(0, a.batch, a.micro):                     # loss over the whole batch (see loss_terms)
                x, y, m = tok[s:s + a.micro, :-1], tok[s:s + a.micro, 1:], mask[s:s + a.micro, 1:]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(x, W)
                loss, mloss = loss_terms(logits, y, m, n_tot, n_all, a.loss)
                loss.backward()
                loss_s += loss.item(); mloss_s += mloss.item()
                acc_s += (logits[m].argmax(-1) == y[m]).sum().item() / n_tot.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tokens += a.batch * a.seq
            loss_l += loss_s; mloss_l += mloss_s; acc_l += acc_s; n_l += 1
            do_eval = (step + 1) % a.eval_every == 0 or step + 1 == a.steps
            if (step + 1) % a.log_every == 0 or do_eval:
                el = time.time() - t0
                row = dict(step=step + 1, loss=loss_l / n_l, masked_loss=mloss_l / n_l, masked_acc=acc_l / n_l, lr=lr_at(step, a.steps, a.lr, a.warmup, a.schedule),
                           tokens_per_s=tokens / el, elapsed_s=el, peak_mem_gb=torch.cuda.max_memory_allocated() / 2**30)
                if do_eval:
                    ho = heldout_probe_acc(model, heldout, a.eval_window, a.micro)
                    row.update(ho)
                curve.append(row)
                print(" ".join(f"{k} {v:.4g}" for k, v in row.items()), flush=True)
                acc_l = loss_l = mloss_l = n_l = 0.0
                if do_eval and a.stop_acc and ho["probe_acc_within"] >= a.stop_acc:
                    last_step, stop_reason = step + 1, f"probe_acc_within >= {a.stop_acc}"
                    print(f"stopping at step {last_step}: {stop_reason}", flush=True)
                    break
                if a.max_minutes and el > 60 * a.max_minutes:
                    last_step, stop_reason = step + 1, f"wall limit {a.max_minutes} min"
                    print(f"stopping at step {last_step}: {stop_reason}", flush=True)
                    break
    train_wall = time.time() - t0
    pd.DataFrame(curve).to_csv(base + "_curve.csv", index=False)
    ckpt = f"ckpt0b/backbone_{a.tag}_s{a.seed}.pt"
    torch.save(dict(model=model.state_dict(), args=vars(a), step=last_step), ckpt)
    summary = dict(train_steps=last_step, stop_reason=stop_reason, train_wall_s=round(train_wall, 1),
                   tokens_per_s=round(tokens / train_wall), peak_mem_gb=round(torch.cuda.max_memory_allocated() / 2**30, 3),
                   n_params=n_params, final_loss=curve[-1]["loss"], final_masked_loss=curve[-1]["masked_loss"], final_masked_acc=curve[-1]["masked_acc"],
                   **{f"final_{k}": v for k, v in ho.items()})
    print(json.dumps(summary), flush=True)
    cfg = dict(model=a.model, d_model=256, n_layers=a.layers, n_heads=4, chunk=64, dense=a.dense, gist=not a.no_gist, vocab=VOCAB,
               train_windows=windows, batch=a.batch, micro=a.micro, seq=a.seq, lr=a.lr, warmup=a.warmup, wd=a.wd, schedule=a.schedule, loss=a.loss,
               steps_planned=a.steps, init=a.init, stop_acc=a.stop_acc, eval_window=a.eval_window, stream=stream)
    if not a.g0:
        with open(jsonl, "a") as f:
            f.write(json.dumps(dict(stage="0b", tag=a.tag, seed=a.seed, **summary, config=cfg, ckpt=ckpt,
                                    curve=base + "_curve.csv", git_commit=commit)) + "\n")
        return
    with open(jsonl, "a") as f:
        for W in [int(w) for w in a.g0_windows.split(",")]:
            for ex in (0.05, 0.2, 0.5):
                te = time.time()
                res = g0_eval(model, W, ex)
                row = dict(stage="0b", tag=a.tag, arm="backbone", seed=a.seed, eval_seed=100, window=W, exc_rate=ex,
                           corr=0.3, eval_tokens=4096, n_streams=64, **res, **summary, config=cfg, ckpt=ckpt,
                           git_commit=commit, eval_wall_s=round(time.time() - te, 1))
                f.write(json.dumps(row) + "\n"); f.flush()
                print(f"W {W} exc {ex}: " + " ".join(f"{k} {v:.3f}" for k, v in res.items() if k.endswith("acc")), flush=True)


if __name__ == "__main__":
    main()
