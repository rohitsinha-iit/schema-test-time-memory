# Sourced by the other scripts. Run everything from the repository root.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
SEEDS_TEST_A=120,121,122,123,124,125,126,127,128,129
SEEDS_TEST_B=130,131,132,133,134,135,136,137,138,139
run () { python -m pilot.run_pilot "$@" 2>&1 | grep -v Loading; }
