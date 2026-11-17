
import json, numpy as np, pandas as pd, glob, os
R = "results"; D = "20261116"
def load(p): return pd.DataFrame([json.loads(l) for l in open(p) if l.strip()])
g = load(f"{R}/stage0a_{D}_p2b_grid.jsonl"); g = g[~g.get("error", pd.Series(index=g.index, dtype=object)).notna()] if "error" in g else g
etas = json.load(open(f"{R}/stage0a_{D}_p2b_eta.json"))
key = ["exc_rate", "corr", "zipf", "seed"]
rng = np.random.default_rng(0)
def boot(x, n=10000):
    x = np.asarray(x); i = rng.integers(0, len(x), (n, len(x))); m = x[i].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), int((x > 0).sum()), len(x)
def pair(df, a, b, m, keys=key):
    A = df[df.arm == a].set_index(keys)[m]; B = df[df.arm == b].set_index(keys)[m]
    return (A - B.loc[A.index]).dropna()
out = []
P = out.append
P("# Stage 0a P2b, E1, E2 analysis (16 Nov 2026)\n")
P(f"Rows: {len(g)}; arms {g.arm.nunique()}; cells x seeds per arm {len(g)//g.arm.nunique()}; errors: {int(g['error'].notna().sum()) if 'error' in g else 0}\n")
val = pd.read_csv(f"{R}/stage0a_{D}_p2b_val_scores.csv")
P("## Chosen learning rates (validation seeds 0-2) and edge check\n")
P("| arm | eta | grid min | grid max | at edge |\n|---|---|---|---|---|")
for arm, e in sorted(etas.items()):
    v = val[val.arm == arm]
    if len(v):
        lo, hi = v.eta.min(), v.eta.max(); P(f"| {arm} | {e} | {lo} | {hi} | {'yes' if e in (lo, hi) else 'no'} |")
    else: P(f"| {arm} | {e} | fixed | fixed | n/a |")
mets = ["rule", "cortex_rule", "cortex_rule_drifted", "cortex_rule_drifted_old", "exc_recent", "exc_mid", "exc_old", "anchor_loss", "n_anchors", "score"]
mets = [m for m in mets if m in g]
mean = g.groupby("arm")[mets].mean().sort_values("cortex_rule", ascending=False)
mean.round(4).to_csv(f"{R}/stage0a_{D}_p2b_arm_means.csv")
P("\n## Arm means over 27 cells x 5 seeds\n"); P(mean.round(3).to_markdown())
def table(title, pairs, m, by_exc=True):
    P(f"\n## {title} (metric {m}; paired over streams; 95% bootstrap CI)\n")
    P("| comparison | mean diff | 95% CI | positive | " + (" | ".join(f"exc {e}" for e in (0.05, 0.2, 0.5)) if by_exc else "") + " |")
    P("|---|---|---|---|" + ("---|" * 3 if by_exc else ""))
    rows = []
    for a, b in pairs:
        if a not in set(g.arm) or b not in set(g.arm): continue
        d = pair(g, a, b, m); mu, lo, hi, npos, n = boot(d.values)
        ex = ""
        if by_exc:
            dd = d.reset_index()
            ex = " | ".join(f"{dd[dd.exc_rate == e][m].mean():+.3f}" for e in (0.05, 0.2, 0.5))
        P(f"| {a} - {b} | {mu:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {npos}/{n} | {ex} |")
        rows.append(dict(a=a, b=b, metric=m, diff=mu, lo=lo, hi=hi, pos=npos, n=n))
    return rows
allrows = []
allrows += table("P1: recurrence routing vs alternatives", [("schema", "schema_surprise_pri"), ("schema", "surprise_fifo"), ("schema", "arrival_fifo"), ("schema", "uniform_replay"), ("schema", "sleep_all"), ("arrival_fifo", "surprise_fifo")], "cortex_rule")
allrows += table("P2: exceptions beyond the window", [("schema", "arrival_fifo"), ("schema", "uniform_replay")], "exc_mid")
allrows += table("P2: old exceptions", [("schema", "arrival_fifo"), ("schema", "uniform_replay")], "exc_old")
allrows += table("Gate attribution", [("schema", "schema_nogate"), ("schema_agem_only", "schema_nogate"), ("schema_lookahead_only", "schema_nogate"), ("schema_dsum_rho4", "schema_nogate"), ("schema_eta05", "schema_nogate_eta05")], "cortex_rule")
allrows += table("Gate attribution, anchor loss (negative = less damage)", [("schema", "schema_nogate"), ("schema_agem_only", "schema_nogate"), ("schema_lookahead_only", "schema_nogate"), ("schema_dsum_rho4", "schema_nogate")], "anchor_loss")
allrows += table("Anchor budget, cortex", [("schema_R64", "schema"), ("schema_R128", "schema")], "cortex_rule")
allrows += table("Anchor budget, exceptions", [("schema_R64", "schema"), ("schema_R128", "schema"), ("schema_R64", "uniform_replay")], "exc_mid")
allrows += table("Reconsolidation", [("schema_noreconsol", "schema")], "cortex_rule_drifted_old")
pd.DataFrame(allrows).to_csv(f"{R}/stage0a_{D}_p2b_contrasts.csv", index=False)
# E1
e1 = load(f"{R}/stage0a_{D}_e1.jsonl")
P("\n## E1: late-onset rules (first occurrence is a one-off; 10 seeds x 3 exception rates)\n")
t = e1.groupby(["arm", "exc_rate"])[["cortex_rule_late", "cortex_rule_early", "exc_mid"]].mean().unstack("exc_rate")
P(t.round(3).to_markdown())
k1 = ["exc_rate", "seed"]
e1rows = []
for b in ["schema_surprise_pri", "surprise_fifo", "arrival_fifo", "uniform_replay", "sleep_all"]:
    d = pair(e1, "schema", b, "cortex_rule_late", k1); mu, lo, hi, npos, n = boot(d.values)
    e1rows.append(f"| schema - {b} | {mu:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {npos}/{n} |")
P("\n| late-rule cortex recall | diff | 95% CI | positive |\n|---|---|---|---|"); [P(x) for x in e1rows]
e1.groupby(["arm", "exc_rate"])[["cortex_rule_late", "cortex_rule_early"]].agg(["mean", "std", "count"]).to_csv(f"{R}/stage0a_{D}_e1_summary.csv")
# E2
e2 = pd.concat([load(f"{R}/stage0a_{D}_e2_d{d}.jsonl") for d in (32, 64, 128)])
P("\n## E2: cortex capacity (key/value dimension d; 10 seeds x exc 0.2, 0.5)\n")
P("| d | exc | schema | surprise_pri | schema - surprise_pri [95% CI] | schema - uniform_replay [95% CI] | schema - surprise_fifo |\n|---|---|---|---|---|---|---|")
e2rows = []
for d in (32, 64, 128):
    for e in (0.2, 0.5):
        s = e2[(e2.d == d) & (e2.exc_rate == e)]
        r = []
        for b in ("schema_surprise_pri", "uniform_replay", "surprise_fifo"):
            x = pair(s, "schema", b, "cortex_rule", ["seed"]); mu, lo, hi, npos, n = boot(x.values); r.append((mu, lo, hi))
            e2rows.append(dict(d=d, exc_rate=e, b=b, diff=mu, lo=lo, hi=hi))
        m = s.groupby("arm").cortex_rule.mean()
        P(f"| {d} | {e} | {m['schema']:.3f} | {m['schema_surprise_pri']:.3f} | {r[0][0]:+.3f} [{r[0][1]:+.3f}, {r[0][2]:+.3f}] | {r[1][0]:+.3f} [{r[1][1]:+.3f}, {r[1][2]:+.3f}] | {r[2][0]:+.3f} |")
pd.DataFrame(e2rows).to_csv(f"{R}/stage0a_{D}_e2_contrasts.csv", index=False)
open(f"{R}/stage0a_{D}_p2b_analysis.md", "w").write("\n".join(out) + "\n")
# figures
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300})
C = {"schema": ("SCHEMA (reuse x error priority)", "#C44E52", 2.0), "schema_surprise_pri": ("SCHEMA, error-only priority", "#8172B3", 1.3),
     "uniform_replay": ("uniform replay", "#55A868", 1.3), "arrival_fifo": ("write at arrival (AHN/AllMem-like)", "#4C72B0", 1.3),
     "surprise_fifo": ("surprise-scaled writes (Titans-like)", "#937860", 1.3)}
def line(ax, df, x, m, keys):
    for arm, (lab, col, lw) in C.items():
        s = df[df.arm == arm].groupby([x, "seed"])[m].mean().reset_index()
        mu = s.groupby(x)[m].mean(); sd = s.groupby(x)[m].std(); n = s.groupby(x)[m].count()
        from scipy import stats
        ci = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
        ax.fill_between(mu.index, mu - ci, mu + ci, color=col, alpha=0.15, lw=0)
        ax.plot(mu.index, mu.values, marker="o", ms=3, color=col, lw=lw, label=lab, zorder=3 if arm == "schema" else 2)
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.6))
line(ax[0], g, "exc_rate", "cortex_rule", None); ax[0].set_title("Cortex-only rule recall", loc="left"); ax[0].set_ylabel("cosine to true value")
line(ax[1], g, "exc_rate", "exc_mid", None); ax[1].set_title("Exceptions 64 to 256 chunks old", loc="left")
for a in ax[:2]:
    a.set_xticks([0.05, 0.2, 0.5]); a.set_xticklabels(["5%", "20%", "50%"]); a.set_xlabel("exceptions in stream")
line(ax[2], e2[e2.exc_rate == 0.5], "d", "cortex_rule", None); ax[2].set_xscale("log", base=2); ax[2].set_xticks([32, 64, 128]); ax[2].set_xticklabels(["32", "64", "128"])
ax[2].set_title("Capacity sweep (50% exceptions)", loc="left"); ax[2].set_xlabel("cortex dimension d")
ax[0].legend(frameon=False, loc="lower left")
for a, l in zip(ax, "abc"): a.text(-0.18, 1.06, l, transform=a.transAxes, fontweight="bold", fontsize=10)
fig.tight_layout(); fig.savefig(f"{R}/fig_stage0a_{D}.png"); fig.savefig(f"{R}/fig_stage0a_{D}.pdf")
print("\n".join(out))
