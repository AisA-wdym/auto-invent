#!/usr/bin/env bash
# Verify a deployment without starting it: every static gate plus all three composition checks.
#
# Distinct from `make gates`, which also runs the test suite. This is the subset an *operator* wants
# before starting a validator — it touches no network, no chain and no credential, and it answers
# "will this configuration run" rather than "is this code correct".
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"

echo "── static gates"
"$PY" tools/check_purity.py
"$PY" tools/gen_models.py --check
"$PY" tools/reachability.py
.venv/bin/ruff check .

echo
echo "── composition"
"$PY" -m gateway --check
"$PY" -m validator --check --season "${AI_SEASON:-config/season.example.json}"
"$PY" -m portal --check

echo
echo "all checks passed."
