
import json, glob, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats
R = "results"; D = "20261116"
plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5, "legend.fontsize": 6.5, "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300, "font.family": "serif",
    "xtick.direction": "out", "ytick.direction": "out", "pdf.fonttype": 42})
def load(p): return pd.DataFrame([json.loads(l) for l in open(p) if l.strip()])
C = {"schema": ("SCHEMA (reuse x error)", "#C44E52", 2.0, "o"), "schema_surprise_pri": ("SCHEMA, error-only priority", "#8172B3", 1.2, "s"),
     "uniform_replay": ("uniform replay", "#55A868", 1.2, "^"), "arrival_fifo": ("write at arrival (AHN/AllMem-like)", "#4C72B0", 1.2, "D"),
     "surprise_fifo": ("surprise-scaled writes (Titans-like)", "#937860", 1.2, "v")}
def line(ax, df, x, m):
    for arm, (lab, col, lw, mk) in C.items():
        s = df[df.arm == arm].groupby([x, "seed"])[m].mean().reset_index()
        mu = s.groupby(x)[m].mean(); sd = s.groupby(x)[m].std(); n = s.groupby(x)[m].count()
        ci = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
        ax.fill_between(mu.index, mu - ci, mu + ci, color=col, alpha=0.15, lw=0)
        ax.plot(mu.index, mu.values, marker=mk, ms=3, color=col, lw=lw, label=lab, zorder=3 if arm == "schema" else 2)
g = load(f"{R}/stage0a_{D}_p2b_grid.jsonl"); e1 = load(f"{R}/stage0a_{D}_e1.jsonl")
e2 = pd.concat([load(f"{R}/stage0a_{D}_e2_d{d}.jsonl") for d in (32, 64, 128)])
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.35))
line(ax[0], g, "exc_rate", "cortex_rule"); ax[0].set_title("Recurring rules, cortex only", loc="left")
line(ax[1], e1, "exc_rate", "cortex_rule_late"); ax[1].set_title("Late-onset rules (E1)", loc="left")
for a in ax[:2]:
    a.set_xticks([0.05, 0.2, 0.5]); a.set_xticklabels(["5%", "20%", "50%"]); a.set_xlabel("one-off exceptions in stream")
ax[0].set_ylabel("recall (cosine to true value)")
line(ax[2], e2[e2.exc_rate == 0.5], "d", "cortex_rule"); ax[2].set_xscale("log", base=2); ax[2].set_xticks([32, 64, 128]); ax[2].set_xticklabels(["32", "64", "128"])
ax[2].minorticks_off(); ax[2].set_title("Capacity (E2), 50% one-offs", loc="left"); ax[2].set_xlabel("cortex dimension d")
for a, l in zip(ax, "abc"):
    a.text(-0.3, 1.08, l, transform=a.transAxes, fontweight="bold", fontsize=9); a.margins(x=0.06)
h, lab = ax[0].get_legend_handles_labels()
fig.legend(h, lab, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.13, 1, 1)); fig.savefig(f"{R}/fig1_stage0a.pdf"); fig.savefig(f"{R}/fig1_stage0a.png")
# neural figure
def loadn(pat):
    rows = []
    for f in sorted(glob.glob(f"{R}/{pat}")): rows += [json.loads(l) for l in open(f) if l.strip()]
    d = pd.DataFrame(rows).drop_duplicates(["arm", "seed", "exc_rate"])
    d["retest_rule"] = d.get("retest_rule_acc", np.nan); d["primary"] = 0.5 * d["retest_rule"].fillna(0) + 0.5 * d["exc_acc_beyond"]; return d
n = loadn("pilot_20261116_v2test_[fgn]*.jsonl")
arms = [("swa_big", "enlarged window"), ("hippo", "hippocampus only"), ("lora_arrival", "write at arrival"), ("lora_surprise", "surprise-scaled writes"),
        ("lora_uniform", "uniform replay, LM loss"), ("hrd_uniform", "uniform replay, HRD"), ("hrd_delta_pri", "error-only priority, HRD"),
        ("hrd_reuse_pri", "reuse-only priority, HRD"), ("hrd_schema", "SCHEMA with gate, HRD"), ("hrd_schema_ng", "SCHEMA, HRD")]
arms = [a for a in arms if a[0] in set(n.arm)]

fig = plt.figure(figsize=(7.0, 2.5))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.1], wspace=0.12, left=0.2, right=0.985, bottom=0.17, top=0.86)
ax = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), None]
for k, (m, title) in enumerate([("retest_rule", "Recurring facts, cortex only"), ("exc_acc_beyond", "One-off facts beyond window")]):
    a = ax[k]
    for i, (arm, lab) in enumerate(arms):
        x = n[n.arm == arm][m].dropna().values
        if not len(x):
            a.text(0.01, i, "n/a", va="center", fontsize=6, color="0.5"); continue
        mu = x.mean(); ci = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
        col = "#C44E52" if arm == "hrd_schema_ng" else ("#8172B3" if arm == "hrd_delta_pri" else "#555555")
        a.errorbar(mu, i, xerr=ci, fmt="o", ms=3.5, color=col, lw=1, capsize=1.5)
    a.set_yticks(range(len(arms))); a.set_yticklabels([l for _, l in arms] if k == 0 else [])
    a.set_ylim(len(arms) - 0.5, -0.5); a.set_title(title, loc="left"); a.set_xlabel("exact-match accuracy")
gs2 = fig.add_gridspec(1, 1, left=0.775, right=0.985, bottom=0.17, top=0.86)
ax[1].set_position([0.2 + (0.555 * 0.5) * 1.02, 0.17, 0.555 * 0.47, 0.69]); ax[0].set_position([0.2, 0.17, 0.555 * 0.47, 0.69])
ax[2] = fig.add_axes([0.86, 0.17, 0.125, 0.69])
comps = [("hrd_schema_ng", "hrd_uniform", "primary", "C1"), ("hrd_schema_ng", "lora_uniform", "primary", "C2"),
         ("hrd_schema_ng", "hrd_delta_pri", "retest_rule", "C3"), ("hrd_schema", "hrd_schema_ng", "primary", "gate")]
rng = np.random.default_rng(0)
for i, (a1, b1, m, lab) in enumerate(comps):
    A = n[n.arm == a1].set_index(["seed", "exc_rate"])[m]; B = n[n.arm == b1].set_index(["seed", "exc_rate"])[m]
    ix = A.index.intersection(B.index)
    if not len(ix): continue
    d = (A.loc[ix] - B.loc[ix]).values; bs = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    ax[2].errorbar(d.mean(), i, xerr=[[d.mean() - lo], [hi - d.mean()]], fmt="o", ms=3.5, color="#C44E52" if i < 3 else "#555555", lw=1, capsize=1.5)
ax[2].axvline(0, color="0.6", lw=0.6, ls="--"); ax[2].set_yticks(range(len(comps))); ax[2].set_yticklabels([c[3] for c in comps])
ax[2].set_ylim(len(comps) - 0.5, -0.5); ax[2].set_title("Paired diff.", loc="left"); ax[2].set_xlabel("difference")
for a, l in zip(ax, "abc"): a.text(-0.04 if l != "a" else -0.05, 1.09, l, transform=a.transAxes, fontweight="bold", fontsize=9, ha="right")
fig.savefig(f"{R}/fig2_neural.pdf"); fig.savefig(f"{R}/fig2_neural.png")
print("figs ok")
