#!/usr/bin/env bash
# Regenerate every analysis report, figure and paper table from results/ (CPU only).
source "$(dirname "$0")/common.sh"
python sim/analyze_p2b.py             # Stage 0a P2b/E1/E2 -> results/stage0a_20261116_p2b_analysis.md
python pilot/analyze_confirm.py       # confirmatory test  -> results/v2test_20261116_analysis.md
python pilot/analyze_scale.py         # gate (40 streams), 3B, 16k -> results/schema_final_runs_20261117.md
python pilot/make_figs.py             # -> results/fig1_stage0a.{pdf,png}, results/fig2_neural.{pdf,png}
cp results/fig1_stage0a.pdf results/fig2_neural.pdf paper/
JD=results python paper/fill_paper.py # -> paper/main.tex
echo "compile with: cd paper && tectonic main.tex   (or pdflatex main.tex)"
