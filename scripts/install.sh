#!/usr/bin/env bash
# One-shot setup. What `make venv` does, for people who arrive expecting a script.
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null; then
  echo "installing uv…"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev,gateway,validator]"

echo
echo "installed. next:"
echo "  make gates                    # every check CI runs"
echo "  .venv/bin/python -m validator --check"
echo "  .venv/bin/python -m gateway --check"
echo "  .venv/bin/python -m portal --check"
