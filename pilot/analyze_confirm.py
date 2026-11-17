
import json, glob, numpy as np, pandas as pd, os, sys
R = "results"
def load(pat):
    rows = []
    for f in sorted(glob.glob(f"{R}/{pat}")):
        rows += [json.loads(l) for l in open(f) if l.strip()]
    return pd.DataFrame(rows)
df = load("pilot_20261116_v2test_[fgn]*.jsonl")
df = df.drop_duplicates(["arm", "seed", "exc_rate"], keep="first")
df["retest_rule"] = df.get("retest_rule_acc", np.nan)
df["primary"] = 0.5 * df["retest_rule"].fillna(0.0) + 0.5 * df["exc_acc_beyond"]
key = ["seed", "exc_rate"]
rng = np.random.default_rng(0)
def boot(x, n=10000):
    x = np.asarray(x, float); i = rng.integers(0, len(x), (n, len(x))); m = x[i].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), int((x > 0).sum()), int((x == 0).sum()), int((x < 0).sum()), len(x)
def pair(d, a, b, m):
    A = d[d.arm == a].set_index(key)[m]; B = d[d.arm == b].set_index(key)[m]
    ix = A.index.intersection(B.index); return (A.loc[ix] - B.loc[ix]).dropna()
L = []; P = L.append
order = ["teacher", "swa", "swa_big", "hippo", "lora_arrival", "lora_surprise", "lora_uniform", "hrd_uniform", "hrd_schema", "hrd_delta_pri", "hrd_reuse_pri", "hrd_schema_ng"]
cols = ["primary", "retest_rule", "exc_acc_beyond", "exc_acc_gt4W", "rule_acc", "rule_drifted_acc", "retest_exc_acc_beyond", "heldout_text_nll", "wall_s"]
cols = [c for c in cols if c in df]
tab = df.groupby("arm")[cols].mean().reindex([a for a in order if a in set(df.arm)])
tab["n_streams"] = df.groupby("arm").size()
P("# Confirmatory test, Qwen2.5-1.5B-Instruct, fresh seeds 120-139 x exc 0.2, 0.5 (16 Nov 2026)\n")
P("Pre-registration: results/prereg_20261116_confirm.md; tuning choice: results/prereg_20261116_tuning_choice.json.\n")
P("## Arm means\n"); P(tab.round(3).to_markdown())
tab.round(4).to_csv(f"{R}/v2test_20261116_arm_means.csv")
def block(title, comps, m):
    P(f"\n## {title} (metric {m}; paired 95% bootstrap CI over streams; W/T/L)\n")
    P("| comparison | diff | 95% CI | W/T/L | n | exc 0.2 | exc 0.5 |\n|---|---|---|---|---|---|---|")
    out = []
    for a, b, tag in comps:
        if a not in set(df.arm) or b not in set(df.arm): continue
        d = pair(df, a, b, m); mu, lo, hi, w, t, l, n = boot(d.values)
        dd = d.reset_index(); e2 = dd[dd.exc_rate == 0.2][m].mean(); e5 = dd[dd.exc_rate == 0.5][m].mean()
        verdict = ""
        if tag: verdict = " **PASS**" if lo > 0 else " **FAIL**"
        P(f"| {tag}{' ' if tag else ''}{a} - {b} | {mu:+.3f} | [{lo:+.3f}, {hi:+.3f}]{verdict} | {w}/{t}/{l} | {n} | {e2:+.3f} | {e5:+.3f} |")
        out.append(dict(a=a, b=b, metric=m, tag=tag, diff=mu, lo=lo, hi=hi, w=w, t=t, l=l, n=n))
    return out
rows = []
rows += block("Pre-registered decisions", [("hrd_schema_ng", "hrd_uniform", "C1"), ("hrd_schema_ng", "lora_uniform", "C2")], "primary")
rows += block("Pre-registered thesis test", [("hrd_schema_ng", "hrd_delta_pri", "C3")], "retest_rule")
rows += block("Reported: primary score", [("hrd_schema_ng", "hrd_reuse_pri", ""), ("hrd_schema_ng", "hrd_delta_pri", ""), ("hrd_schema", "hrd_schema_ng", ""), ("hrd_schema_ng", "hippo", ""), ("hrd_schema_ng", "lora_surprise", ""), ("hrd_schema_ng", "lora_arrival", ""), ("hrd_schema_ng", "swa_big", "")], "primary")
rows += block("Reported: cortex-only rule recall", [("hrd_schema_ng", "hrd_uniform", ""), ("hrd_reuse_pri", "hrd_delta_pri", ""), ("hrd_schema_ng", "hrd_reuse_pri", ""), ("hrd_schema_ng", "lora_surprise", ""), ("hrd_schema_ng", "lora_arrival", ""), ("hrd_schema_ng", "lora_uniform", "")], "retest_rule")
rows += block("Reported: exceptions beyond W", [("hrd_schema_ng", "hrd_uniform", ""), ("hrd_schema_ng", "hippo", ""), ("hrd_schema_ng", "swa_big", ""), ("hrd_schema_ng", "lora_surprise", "")], "exc_acc_beyond")
rows += block("Reported: held-out text NLL (lower is better)", [("hrd_schema_ng", "hrd_uniform", ""), ("hrd_schema_ng", "lora_uniform", ""), ("hrd_schema", "hrd_schema_ng", "")], "heldout_text_nll")
pd.DataFrame(rows).to_csv(f"{R}/v2test_20261116_contrasts.csv", index=False)
icols = [c for c in ["steps_tried", "steps_accepted", "lora_rejected", "lora_shrunk", "hrd_queries", "hrd_uninformative", "recalls", "released_consistent", "released_consolidated", "hrd_episodes_with_query", "hrd_episodes_in_H"] if c in df]
P("\n## Internals (means per stream)\n"); P(df[df.arm.str.startswith("hrd")].groupby("arm")[icols].mean().round(1).to_markdown())
# 3B
d3 = load("pilot_20261116_v2test_3b*.jsonl")
if len(d3):
    d3 = d3.drop_duplicates(["arm", "seed", "exc_rate"])
    d3["retest_rule"] = d3.get("retest_rule_acc", np.nan); d3["primary"] = 0.5 * d3["retest_rule"].fillna(0) + 0.5 * d3["exc_acc_beyond"]
    t3 = d3.groupby("arm")[[c for c in cols if c in d3]].mean(); t3["n_streams"] = d3.groupby("arm").size()
    P("\n## Qwen2.5-3B replication (seeds 120-129 x exc 0.2, 0.5; settings carried over from 1.5B, not re-tuned)\n"); P(t3.round(3).to_markdown())
    t3.round(4).to_csv(f"{R}/v2test_3b_20261116_arm_means.csv")
    df3 = df; df = d3
    rows3 = block("3B: primary", [("hrd_schema_ng", "hrd_uniform", ""), ("hrd_schema_ng", "hippo", ""), ("hrd_schema_ng", "ahn", ""), ("hrd_schema_ng", "swa_big", "")], "primary")
    rows3 += block("3B: cortex-only rule recall", [("hrd_schema_ng", "hrd_uniform", "")], "retest_rule")
    rows3 += block("3B: exceptions beyond W", [("hippo", "ahn", ""), ("hippo", "swa_ahnbytes", ""), ("hrd_schema_ng", "ahn", "")], "exc_acc_beyond")
    pd.DataFrame(rows3).to_csv(f"{R}/v2test_3b_20261116_contrasts.csv", index=False)
    df = df3
open(f"{R}/v2test_20261116_analysis.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
