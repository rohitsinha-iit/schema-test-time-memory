"""Secondary analyses (not pre-registered): gated SCHEMA on all 40 confirmatory streams, the Qwen2.5-3B replication
and the 16k-token run. Paired bootstrap over streams (10,000 resamples, random.Random(0)).
Writes results/schema_final_runs_20261117.md.  Run from the repo root: python pilot/analyze_scale.py
"""
import glob, json, math, random, statistics as st

R = "results"


def load(pats):
    rows = []
    for p in pats:
        for f in sorted(glob.glob(f"{R}/{p}")):
            rows += [json.loads(l) for l in open(f) if l.strip()]
    return rows


def isnan(v): return v is None or (isinstance(v, float) and math.isnan(v))
def prim(r): return 0.5 * (0.0 if isnan(r.get("retest_rule_acc")) else r["retest_rule_acc"]) + 0.5 * (r["exc_acc_beyond"] or 0)
def rt(r): return r["retest_rule_acc"] or 0
def eb(r): return r["exc_acc_beyond"]


def table(rows):
    by = {}
    for r in rows: by.setdefault(r["arm"], {})[(r["seed"], r["exc_rate"])] = r
    out = {}
    for a, dd in by.items():
        v = list(dd.values())
        g = lambda k: [x[k] for x in v if not isnan(x.get(k))]
        m = lambda L: round(st.mean(L), 3) if L else None
        out[a] = dict(n=len(v), prim=m([prim(x) for x in v]), rt=m(g("retest_rule_acc")), excB=m(g("exc_acc_beyond")),
                      gt4W=m(g("exc_acc_gt4W")), nll=m(g("heldout_text_nll")))
    return by, out


def contrast(by, a, b, key, B=10000):
    ks = sorted(set(by[a]) & set(by[b])); d = [key(by[a][k]) - key(by[b][k]) for k in ks]
    rng = random.Random(0); bs = sorted(st.mean(rng.choices(d, k=len(d))) for _ in range(B))
    return dict(n=len(d), mean=round(st.mean(d), 3), lo=round(bs[int(.025 * B)], 3), hi=round(bs[int(.975 * B) - 1], 3),
                w=sum(x > 1e-9 for x in d), t=sum(abs(x) <= 1e-9 for x in d), l=sum(x < -1e-9 for x in d))


def fm(o):
    s = "| arm | n | primary | cortex retest | beyond W | >4W | NLL |\n|---|---|---|---|---|---|---|\n"
    for a, x in o.items(): s += f"| {a} | {x['n']} | {x['prim']} | {x['rt']} | {x['excB']} | {x['gt4W']} | {x['nll']} |\n"
    return s


def fc(c):
    s = "| contrast | n | mean | 95% CI | win/tie/loss |\n|---|---|---|---|---|\n"
    for k, v in c.items(): s += f"| {k} | {v['n']} | {v['mean']:+.3f} | [{v['lo']:+.3f}, {v['hi']:+.3f}] | {v['w']}/{v['t']}/{v['l']} |\n"
    return s


K = [("prim", prim), ("rt", rt), ("excB", eb)]
by3, o3 = table(load(["pilot_*v3b_test*.jsonl"]))
c3 = {f"{a}-{b}|{kn}": contrast(by3, a, b, kf) for a, b in [("hrd_schema_ng", "hrd_uniform"), ("hrd_schema_ng", "hrd_delta_pri"),
      ("hrd_schema_ng", "hippo"), ("hrd_schema_ng", "ahn"), ("hrd_uniform", "hippo")] for kn, kf in K}
by16, o16 = table(load(["pilot_*v2long16k*.jsonl"]))
c16 = {f"{a}-{b}|{kn}": contrast(by16, a, b, kf) for a, b in [("hrd_schema_ng", "hrd_uniform"), ("hrd_schema_ng", "hrd_delta_pri"),
       ("hrd_schema_ng", "hippo")] for kn, kf in K}
byg, og = table(load(["pilot_*v2test_gate*.jsonl"]))
byn, _ = table(load(["pilot_*v2test_ng*.jsonl"])); byg["hrd_schema_ng"] = byn["hrd_schema_ng"]
cg = {f"hrd_schema-hrd_schema_ng|{kn}": contrast(byg, "hrd_schema", "hrd_schema_ng", kf) for kn, kf in K}
md = f"""# SCHEMA final runs: 3B replication, 16k context, gated arm (17 Nov 2026)
Paired bootstrap, 10,000 resamples, seed 0. Primary = 0.5 x cortex retest rule + 0.5 x exceptions beyond W. Not pre-registered (secondary analyses).

## Qwen2.5-3B, seeds 120 to 139 x exc 0.2/0.5 (tuned on validation: ng lr 1e-4, uniform lr 3e-4)
{fm(o3)}
{fc(c3)}
## 1.5B, 16k-token streams, seeds 120 to 129 x exc 0.2/0.5
{fm(o16)}
{fc(c16)}
## Gated SCHEMA (1.5B, 40 streams) vs no-gate
{fm(og)}
{fc(cg)}"""
open(f"{R}/schema_final_runs_20261117.md", "w").write(md)
print("wrote", f"{R}/schema_final_runs_20261117.md")
