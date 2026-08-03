# auto-invent

A Bittensor subnet in which miners submit complete **autonomous invention laboratories**,
validators independently generate fresh research problems every day, execute every
laboratory under equal budgets, and score the resulting idea portfolios through
deterministic integrity checks and calibrated multi-model LLM judging.

The subnet's digital commodity is not a research answer. It is a *lab*: a versioned,
executable bundle that receives an unfamiliar research problem and returns valuable,
non-obvious, technically coherent ideas a human researcher could seriously consider.

- **Architecture:** [`architecture.md`](architecture.md) — v3.0, the specification this
  repository implements.
- **Migration:** [`MIGRATION.md`](MIGRATION.md) — what carries over from the earlier
  simulated-user design, what changes, and what is deleted.

## Status

Early implementation. See `MIGRATION.md` for what exists and what does not.

Nothing here is ready for mainnet, and the criterion for that is deliberately not
"the code runs" — it is `architecture.md` §27: the validator must reliably rank
deliberately strong, weak, copied, impossible and superficially novel portfolios in the
correct order, and a competing lab must repeatedly beat direct frontier-model use.

## Gates

`make gates` runs everything CI runs. Each exists because it caught a real defect:

| | |
|---|---|
| `make lint` | Style and dead code. |
| `make purity` | The reward path may not read a clock, a global RNG, or the network. A single `time.time()` in scoring desynchronises every validator and raises nothing. |
| `make schema` | Generated models have not drifted from `protocol/schemas/`. |
| `make reach` | Every enforcement point has a **production** call path. A guard reachable only from a test is indistinguishable from an absent guard at runtime. |
| `make test` | The suite, under a randomised hash seed. |
| `make test-determinism` | The determinism subset, under five fixed hash seeds. |
