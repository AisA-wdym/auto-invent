.DEFAULT_GOAL := help
PY := .venv/bin/python

.PHONY: help venv test test-determinism purity lint reach models schema gates clean

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

venv: ## Create the virtualenv and install dev dependencies
	uv venv --python 3.12 .venv
	uv pip install --python $(PY) -e ".[dev,gateway,validator]"

test: ## Run the full suite under a randomised hash seed
	PYTHONHASHSEED=random $(PY) -m pytest -q -m "not live"

live: ## The vertical slice against real infrastructure. Spends money; needs Docker and both keys.
	# Excluded from `make test` because it costs a few dollars and takes about ten minutes. It is
	# also the only thing in the suite that has ever caught a defect in the container, the image,
	# the scaffold or the price table — so it is worth running before anything is called finished.
	$(PY) -m pytest -q -m live tests/integration/test_live_round.py

test-determinism: ## Run the determinism suite five times under different hash seeds
	@for seed in 0 1 7 42 20260803; do \
		echo "--- PYTHONHASHSEED=$$seed"; \
		PYTHONHASHSEED=$$seed $(PY) -m pytest -q -m determinism || exit 1; \
	done

measure: ## Run the section 27 mainnet measurement gates
	$(PY) -m pytest -q -m measurement

purity: ## Static check: no clock, network or global RNG in the scoring path
	$(PY) tools/check_purity.py

reach: ## Static check: every enforcement point has a production call path
	$(PY) tools/reachability.py

models: ## Regenerate protocol/models/ from the JSON Schemas
	$(PY) tools/gen_models.py

schema: ## Static check: generated models have not drifted from the schemas
	$(PY) tools/gen_models.py --check

lint: ## Ruff
	.venv/bin/ruff check .

check: ## Compose every process and validate the config, without a network, chain or credential
	$(PY) -m gateway --check
	# The testnet season, deliberately. `season.example.json` is mainnet-shaped and sets
	# allow_unprobed_packs to false, so it *correctly* refuses until 7.4 step 5's probe is wired —
	# a red check there is the build telling the truth. The testnet config declares that
	# degradation, which is what makes it checkable today.
	$(PY) -m validator --check --season config/season.testnet.json

gates: purity reach schema lint check test-determinism test ## Everything CI runs

clean:
	rm -rf .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
