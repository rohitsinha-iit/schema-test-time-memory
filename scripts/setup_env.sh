#!/usr/bin/env bash
# Create a virtual environment and run the CPU unit tests.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'transformers', transformers.__version__)"
PYTHONPATH=. python -m pytest -q tests/test_core.py tests/test_pilot_budget.py
