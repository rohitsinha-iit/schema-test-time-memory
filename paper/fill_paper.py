
import json, glob, math, os, pandas as pd
R = "results"; J = os.environ["JD"]
tex = open("paper/main_template.tex", encoding="utf-8").read()
def load(pat):
    rows = []
    for f in sorted(glob.glob(pat)):
        rows += [json.loads(l) for l in open(f) if l.strip()]
    return rows
def nz(v): return None if v is None or (isinstance(v, float) and math.isnan(v)) else v
def means(rows):
    out = {}
    for a in sorted({r["arm"] for r in rows}):
        v = {(r["seed"], r["exc_rate"]): r for r in rows if r["arm"] == a}.values()
        g = lambda k: [nz(x.get(k)) for x in v if nz(x.get(k)) is not None]
        mm = lambda L: sum(L)/len(L) if L else float("nan")
        out[a] = dict(n=len(v), primary=mm([0.5*(nz(x.get("retest_rule_acc")) or 0) + 0.5*x["exc_acc_beyond"] for x in v]),
                      retest_rule=mm(g("retest_rule_acc")), exc_acc_beyond=mm(g("exc_acc_beyond")), heldout_text_nll=mm(g("heldout_text_nll")))
    return out
m = pd.read_csv(f"{R}/v2test_20261116_arm_means.csv", index_col=0)
gate = means(load(f"{J}/pilot_*v2test_gate*.jsonl"))["hrd_schema"]
for k in ("primary", "retest_rule", "exc_acc_beyond", "heldout_text_nll"): m.loc["hrd_schema", k] = gate[k]
m.loc["hrd_schema", "n_streams"] = gate["n"]
f = lambda v: "n/a" if pd.isna(v) else f"{v:.3f}"
names = [("swa_big", "Enlarged window"), ("hippo", "Episodic store"), ("lora_arrival", "LoRA, at arrival"),
         ("lora_surprise", "LoRA, surprise-scaled"), ("lora_uniform", "LoRA, uniform replay"), ("hrd_uniform", "HRD, uniform replay"),
         ("hrd_delta_pri", "HRD, error-only pri."), ("hrd_reuse_pri", "HRD, reuse-only pri."), ("hrd_schema", "HRD, SCHEMA + gate"),
         ("hrd_schema_ng", "HRD, SCHEMA")]
rows = []
for a, lab in names:
    r = m.loc[a]; lab2 = r"\textbf{" + lab + "}" if a == "hrd_schema_ng" else lab
    rt = "n/a" if a == "swa_big" else f(r["retest_rule"])
    nll = "base" if a in ("swa_big", "hippo") else f(r["heldout_text_nll"])
    rows.append(f"{lab2} & {f(r['primary'])} & {rt} & {f(r['exc_acc_beyond'])} & {nll}\\\\")
tex = tex.replace("TABLEROWS", "\n".join(rows))
m3 = means(load(f"{J}/pilot_*v3b_test*.jsonl")); m16 = means(load(f"{J}/pilot_*v2long16k*.jsonl"))
assert all(v["n"] == 40 for v in m3.values()) and all(v["n"] == 20 for v in m16.values()), (m3, m16)
sc = [("hippo", "Episodic store"), ("ahn", "AHN-GDN-3B"), ("hrd_uniform", "HRD, uniform"), ("hrd_delta_pri", "HRD, error-only"), ("hrd_schema_ng", r"\textbf{HRD, SCHEMA}")]
srows = []
for a, lab in sc:
    cells = []
    for M in (m3, m16):
        if a in M: cells += [f(M[a]["primary"]), f(M[a]["retest_rule"]), f(M[a]["exc_acc_beyond"])]
        else: cells += ["", "", ""]
    srows.append(lab + " & " + " & ".join(cells) + r"\\")
tex = tex.replace("SCALEROWS", "\n".join(srows))
assert "\u2013" not in tex and "\u2014" not in tex and "--" not in tex
open("paper/main.tex", "w", encoding="utf-8").write(tex)
print("gate n", gate["n"], "3B arms", sorted(m3), "16k arms", sorted(m16))
