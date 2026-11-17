"""Run pilot arms on text streams and write one JSON line per (arm, stream seed, exc_rate).

  python -m pilot.run_pilot --arms teacher,swa --seeds 0,1,2 --tag g0
Validation seeds 0 to 2 (calibration only), test seeds 100 and up (hard rule 4).
"""
import argparse
import datetime
import json
import os
import time

import numpy as np

from pilot.engine import Engine
from pilot.text_stream import load_filler, make_stream
from sim.sanity import git_commit

STREAM = dict(n_tokens=8192, n_rules=16, zipf=1.0, corr=0.5, drift_frac=0.25, filler_min=20, filler_max=80)
W = 1024


def arm_config(name, W=W, hp=None):
    """(window, memory factory or None)."""
    if name == "teacher":
        return 10**9, None
    if name == "swa":
        return W, None
    if name == "ahn":                                            # released AHN-GDN (Qwen2.5-3B only)
        from pilot.ahn_port import AHNMemory
        return W, AHNMemory
    if name.startswith("ahn_ft"):                               # fine-tuned AHN-GDN: pilot/data/<name>.pt
        import os
        from pilot.ahn_port import AHNMemory
        path = os.path.join(os.path.dirname(__file__), "data", f"{name}.pt")
        return W, (lambda eng: AHNMemory(eng, path=path))
    if name == "swa_ahnbytes":                                   # window enlarged by the AHN state's bytes
        from pilot.ahn_port import STATE_PAIRS
        return W + STATE_PAIRS, None
    from pilot.memory import ARMS
    if name in ARMS:
        return ARMS[name](W)
    from pilot.lora_cortex import LORA_SPECS, LoRAMemory
    if name in LORA_SPECS:
        extra_w, kw = LORA_SPECS[name]
        return W + extra_w, (lambda eng: LoRAMemory(eng, **{**(hp or {}), **kw}))
    from pilot.linear_cortex import LIN_SPECS, LinearSchemaMemory
    if name in LIN_SPECS:
        extra_w, kw = LIN_SPECS[name]
        return W + extra_w, (lambda eng: LinearSchemaMemory(eng, arm=name, **{**(hp or {}), **kw}))
    from pilot.schema_memory import SPECS, SchemaMemory
    extra_w, kw = SPECS[name]
    return W + extra_w, (lambda eng: SchemaMemory(eng, **{**(hp or {}), **kw}))


def metrics(probes, res, W=W):
    c = np.array([r["correct"] for r in res]); kind = np.array([p["kind"] for p in probes])
    d = np.array([p["delay"] for p in probes]); dr = np.array([p["drifted"] for p in probes])
    acc = lambda m: float(c[m].mean()) if m.any() else float("nan")
    exc, rule = kind == "exc", kind == "rule"
    return dict(rule_acc=acc(rule), rule_drifted_acc=acc(rule & dr), rule_stable_acc=acc(rule & ~dr),
                exc_acc_within=acc(exc & (d <= W)), exc_acc_beyond=acc(exc & (d > W)),
                exc_acc_1to4W=acc(exc & (d > W) & (d <= 4 * W)), exc_acc_gt4W=acc(exc & (d > 4 * W)),
                n_rule=int(rule.sum()), n_exc_within=int((exc & (d <= W)).sum()), n_exc_beyond=int((exc & (d > W)).sum()),
                n_exc_gt4W=int((exc & (d > 4 * W)).sum()), mean_logprob=float(np.mean([r["logprob"] for r in res])))


def run_rows(eng, arms, seeds, exc_rates, hp, retest, model, commit, sinks=4, exact_window=False):
    """Yield one result row per (seed, exc_rate, arm)."""
    filler = load_filler(eng.tok)
    for seed in seeds:
        for ex in exc_rates:
            kw = dict(STREAM, exc_rate=ex)
            s = make_stream(np.random.default_rng([seed, int(ex * 1000)]), eng.tok, filler, **kw)
            for arm in arms:
                t0 = time.time()
                arm_hp = hp.get(arm, hp.get("*", {})) if hp and isinstance(next(iter(hp.values())), dict) else hp
                win, mem_factory = arm_config(arm, hp=arm_hp)
                from pilot.lora_cortex import lora_enable
                lora_enable(eng, False)                                   # LoRA cortex only in lora_* arms
                mem = mem_factory(eng) if mem_factory else None
                out = eng.run(s["tokens"], win, memory=mem, probes=s["probes"], sinks=sinks, exact_window=exact_window)
                row = dict(stage="pilot", arm=arm, seed=seed, exc_rate=ex, stream=kw, window=win, model=model, hp=arm_hp,
                           sinks=getattr(mem, "sinks", sinks), exact_window=exact_window or getattr(mem, "exact_window", False),
                           **metrics(s["probes"], out["probes"]), stream_nll=float(out["nll"][1:].mean()))
                if mem is not None:
                    row.update(mem.stats())
                    if retest:
                        mem.retest_mode()
                        rt = eng.run_probes_only(s["probes"], len(s["tokens"]), mem)
                        row.update({f"retest_{k}": v for k, v in metrics(s["probes"], rt).items()})
                row.update(git_commit=commit, wall_s=round(time.time() - t0, 1))
                yield row


def show(row):
    rt = f" | retest rule {row['retest_rule_acc']:.3f} exc {row['retest_exc_acc_beyond']:.3f}" if "retest_rule_acc" in row else ""
    print(f"seed {row['seed']} exc {row['exc_rate']} {row['arm']:22s} rule {row['rule_acc']:.3f} exc_in {row['exc_acc_within']:.3f} "
          f"exc_beyond {row['exc_acc_beyond']:.3f} (>4W {row['exc_acc_gt4W']:.3f}) drift {row['rule_drifted_acc']:.3f}{rt} "
          f"{row['wall_s']}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="teacher,swa")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--exc_rates", default="0.2,0.5")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n_tokens", type=int, default=8192, help="stream length (16 Nov 2026: 16384 robustness run)")
    ap.add_argument("--date", default=f"{datetime.date.today():%Y%m%d}")
    ap.add_argument("--retest", action="store_true", help="also probe with window and H cleared (only C remains)")
    ap.add_argument("--sinks", type=int, default=4)
    ap.add_argument("--exact_window", action="store_true", help="per-token sliding window (t - W, t], as AHN")
    ap.add_argument("--hp", default="{}", help="SchemaMemory hyperparameters as JSON (tuned on validation seeds): "
                    "one dict for all arms, or {arm: dict, '*': default}")
    a = ap.parse_args()
    STREAM["n_tokens"] = a.n_tokens
    path = f"results/pilot_{a.date}_{a.tag}.jsonl"
    if os.path.exists(path):
        raise SystemExit(f"{path} exists; results files are never overwritten")
    eng = Engine(a.model)
    with open(path, "w") as f:
        for row in run_rows(eng, a.arms.split(","), [int(x) for x in a.seeds.split(",")],
                            [float(x) for x in a.exc_rates.split(",")], json.loads(a.hp), a.retest, a.model, git_commit(),
                            sinks=a.sinks, exact_window=a.exact_window):
            f.write(json.dumps(row) + "\n"); f.flush()
            show(row)


if __name__ == "__main__":
    main()
