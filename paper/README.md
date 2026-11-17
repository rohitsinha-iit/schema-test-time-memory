# `paper/`

* `main_template.tex`: paper text with two placeholders, `TABLEROWS` and `SCALEROWS`.
* `fill_paper.py`: fills the tables from `results/` (`JD=results python paper/fill_paper.py`, run from the repo
  root) and writes `main.tex`.
* `main.tex`, `main.pdf`, `fig1_stage0a.*`, `fig2_neural.*`: the compiled 4-page draft and its figures.

The layout approximates the AAAI two-column style with the `article` class; move the body into the official AAAI
author kit before submission. Compile with `tectonic main.tex` or `pdflatex main.tex`.
