# auto-invent

**A Bittensor subnet where miners submit autonomous invention laboratories, and validators pay for
research architecture rather than for research answers.**

Each round every validator generates a fresh pack of research problems, runs every miner's
laboratory against the same pack under an identical budget, and scores the resulting portfolios. What
a miner sells is not an idea — it is a *machine that produces ideas*, and it has to keep producing
them on problems nobody has seen.

```
        ┌──────────────────────────────────────────────────────────┐
        │            Bittensor · commit-reveal · weights           │
        └───────────────────────────┬──────────────────────────────┘
                                    │
     miner                     validator                      OpenRouter
       │                            │                              │
   seal bundle ──commitment──▶ read metagraph                      │
       │                            │                              │
       │                     commit salt (T-450)                   │
       │                     draw randomness (T-300)               │
       │                     generate the pack ────────────────────┤ validator's key
       │                     commit pack hash (T-100)              │
       │                            │                              │
       └────── reveal (T+0) ──▶ run laboratory ─── RCG ────────────┤ miner's key
                                    │                              │
                             13 hard gates                         │
                             canonicalise                          │
                             prior art ────────────────────────────┤ validator's key
                             judge panels ─────────────────────────┤ validator's key
                                    │
                             daily → rolling score
                                    │
                             weights (T+6900)
```

---

## What makes this subnet different

**The validator is not in the content path.** It does not write the questions by hand and does not
grade by hand. It commits to a salt *before* the randomness that seeds generation exists, and commits
the pack's hash *before* any miner's bundle is opened. So it cannot see a submission and then choose a
problem to suit it — the ordering is enforced in code ([`validator/cycle.py`](validator/cycle.py))
and a config that breaks it is refused at load rather than discovered on a day it matters.

**Two accounts, one provider.** Every model call goes through OpenRouter. Miners spend their own key;
validators spend theirs. Under one provider surface a swapped key *succeeds* — it authenticates,
returns a completion, and silently bills the wrong party — so the separation is structural: two
distinct types, and purpose selects the credential rather than a parameter
([`gateway/credentials.py`](gateway/credentials.py)).

**Nobody is paid for beating nobody.** If no laboratory exceeds the reference template's own score,
the emission burns. Being best in a weak field is not qualification
([`validator/weights.py`](validator/weights.py)).

**Presentation is stripped before judging.** A judge sees a neutral fact sheet: no identity, no
branding, no markdown, and any unsupported percentage explicitly marked unverified. Selling well
earns nothing ([`validator/canonicalizer/`](validator/canonicalizer/)).

---

## Documentation

| Read this | If you are |
|---|---|
| [docs/miner.md](docs/miner.md) | building a laboratory and submitting it |
| [docs/validator.md](docs/validator.md) | running a validator |
| [docs/incentive.md](docs/incentive.md) | working out what gets rewarded, and why |
| [architecture.md](architecture.md) | implementing against the protocol, or reviewing it |

---

## Quick start

```bash
git clone https://github.com/AisA-wdym/auto-invent && cd auto-invent
make venv
make gates          # lint, purity, schema drift, reachability, determinism, tests
```

`make gates` is what CI runs. All six must pass before anything merges.

### Miner

```bash
ail-miner init my-lab              # a laboratory that runs on the first invocation
cd my-lab && docker build -t my-lab .
ail-miner validate .               # every offline check a validator makes, before you spend a day
ail-miner seal . --out ../sealed --spend-cap 25
ail-miner submit ../sealed --round 2026-08-03 --url https://… --netuid N
```

Full walkthrough: [docs/miner.md](docs/miner.md).

### Validator

```bash
export AI_VALIDATOR_OPENROUTER_KEY_FILE=/run/secrets/openrouter
python -m validator --check                       # validate the deployment; no chain, no credential
docker compose -f ops/docker-compose.yml up -d    # gateway, Redis, internal sandbox network
python -m validator --netuid N --wallet my --hotkey val --redis-url redis://127.0.0.1:6379/0
```

`--check` builds the whole object graph, validates the season config, the cycle ordering, the judge
panels and every model pin, then exits. It needs no chain, no network and no credential — so a
deployment can be verified before it is trusted with one.

Full walkthrough: [docs/validator.md](docs/validator.md).

---

## Repository layout

```
protocol/            Everything both sides must agree on, byte for byte
  canonical.py         the one deterministic encoder; refuses floats in hashed objects
  fixedpoint.py        parts-per-million integer arithmetic
  seeds.py             round seed, salt commitment, seeded slot assignment
  receipts.py          the hash-chained record of every external call
  commitments.py       the on-chain wire format (submission, salt, pack)
  schemas/             seven JSON Schemas; protocol/models/ is generated from them

chain/               The only package that imports bittensor
  client.py            ChainClient (eight methods), BittensorChain, FakeChain

gateway/             The Research Compute Gateway
  credentials.py       two typed resolvers; purpose selects the payer
  tokens.py            HMAC session tokens: ceilings in, credential never
  metering.py          RCC pricing and a reserve-then-settle ledger
  adapters/            OpenRouter, for both accounts
  api.py               the only address a sandbox can reach
  __main__.py          the gateway process

validator/
  cycle.py             round boundaries and identity, orderings enforced at load
  scheduler.py         which step is due, decided from a block height. Pure
  driver.py            the loop: poll, ask the scheduler, run, record
  roundstate.py        the recovery record and the gated public document
  challenge_factory/   plan → generate → lint → safety → dedup → critic → probe → commit
  sandbox/             hardened container and the single-request runner
  canonicalizer/       the neutral fact sheet
  prior_art/           what the search found, never a novelty claim
  judge/               panels, anchored screening, Swiss pairwise, Bradley-Terry
  scoring/             the hard gates and the scoring criterion, daily and rolling
  weights.py           softmax, cap, and the burn
  __main__.py          the validator process

miner/
  cli/                 ail-miner: init, validate, run, seal, submit
                       `run` rehearses in the validator's own sandbox, with the same gates
  reference/           the one template — which is also the qualification floor

ops/                 Dockerfiles, compose, runbook
tools/               the three static gates
tests/               unit, integration, adversarial, measurement, localnet
```

---

## The six gates

Each catches a class of defect the others cannot.

| Gate | What it catches | Why a test cannot |
|---|---|---|
| `make lint` | style, dead code, unsorted imports | — |
| `make purity` | a clock, RNG or network import in a scoring module | a process compared against itself agrees perfectly |
| `make schema` | generated models drifting from the JSON Schemas | drift stays silent until a field is read |
| `make reach` | a rule no production call path reaches | a test calling a guard directly passes either way |
| `make test-determinism` | dict-order dependence, under five hash seeds | one seed cannot show it |
| `make test` | everything else | — |

The reachability gate is the unusual one. It pins each protocol rule to the symbol that enforces it
and walks from the three process entry points, because a guard nothing reaches is indistinguishable
at runtime from a guard nobody wrote. It holds **30 enforcement points** on a live call path, and it
is strict: adding a rule without wiring it fails the build.

Both `--check` flags exist because the reachability gate proves a call path *exists* and cannot
prove it *works*. They compose the real object graph and validate it, so a deployment can be checked
before it is trusted with a credential.

---

## Status

| Layer | State |
|---|---|
| Protocol, schemas, fixed-point, seeds, receipts, commitments | complete, tested |
| Gateway (RCG) | complete, tested, verified live against OpenRouter |
| Chain layer | complete against bittensor 11; localnet tests pending |
| Challenge factory | complete, tested, verified live |
| Sandbox and hard gates | complete, tested |
| Canonicalizer, prior art | complete, tested |
| Judge panels, screening, Swiss tournament | complete, tested |
| Scoring and weights | complete, tested |
| Block-driven round loop | complete, running on testnet 542 |
| Measurement gates | **pending** |

885 tests, six gates green. Not ready for mainnet, and the criterion for that is deliberately not
"the code runs": the validator must reliably rank deliberately strong, weak, copied, impossible and
superficially novel portfolios in the right order, and a competing laboratory must repeatedly beat
direct frontier-model use.

---

## Licence

Apache 2.0.
