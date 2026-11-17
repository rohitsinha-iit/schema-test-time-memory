#!/usr/bin/env bash
# Stage 0a (controlled study, CPU): P2b retune grid, E1 late-onset rules, E2 capacity sweep, then the analysis.
# Writes results/stage0a_<date>_p2b_*.jsonl/csv/json, results/stage0a_<date>_e1.jsonl, results/stage0a_<date>_e2_d*.jsonl
source "$(dirname "$0")/common.sh"
WORKERS=${WORKERS:-30}
python -m pytest -q tests/test_core.py
python -m sim.p2b --workers "$WORKERS" --date 20261116
python sim/analyze_p2b.py          # -> results/stage0a_20261116_p2b_analysis.md and the contrast CSVs
# The earlier M1 grid (3 Nov, 13 arms) is reproduced with: python -m sim.grid --date 20261103
