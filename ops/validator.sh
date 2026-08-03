#!/usr/bin/env bash
# Bare-metal validator launcher, for operators who prefer pm2 to compose.
#
#   pm2 start ops/validator.sh --name auto-invent-validator
#
# `--check` runs first and a failure aborts. Starting a validator whose config does not validate
# would mean discovering the problem after a pack hash was already committed on chain.
set -euo pipefail
cd "$(dirname "$0")/.."

: "${AI_NETUID:?set AI_NETUID}"
: "${AI_VALIDATOR_OPENROUTER_KEY_FILE:?point this at your OpenRouter key file}"

PY="${PY:-.venv/bin/python}"

"$PY" -m validator --check

exec "$PY" -m validator \
  --netuid "$AI_NETUID" \
  --network "${AI_NETWORK:-finney}" \
  --wallet "${AI_WALLET:-default}" \
  --hotkey "${AI_HOTKEY:-default}" \
  --redis-url "${AI_REDIS_URL:-redis://127.0.0.1:6379/0}" \
  --rcg-endpoint "${AI_RCG_ENDPOINT:-http://127.0.0.1:8081}" \
  --log-level "${AI_LOG_LEVEL:-INFO}"
