# Contributing

## The one thing to know first

Six gates run on every push, and each of them exists because it caught a defect the others could not:

```bash
make gates
```

| Gate | What it catches | Why a test cannot |
|---|---|---|
| `lint` | style, dead code, unsorted imports | — |
| `purity` | a clock, RNG or network import in a scoring module | a process compared against itself agrees perfectly |
| `schema` | generated models drifting from the JSON Schemas | drift stays silent until a field is read |
| `reach` | a rule no production call path reaches | a test calling a guard directly passes either way |
| `test-determinism` | dict-order dependence, under five hash seeds | one seed cannot show it |
| `test` | everything else | — |

## Adding a rule means adding it to the reachability gate

`tools/reachability.py` pins each protocol rule to the symbol that enforces it, and walks from the
three process entry points. If you implement a rule from `architecture.md`, add the pair:

```python
"7.4.4 a candidate duplicating the last 90 days is rejected":
    "validator.challenge_factory.dedup:is_duplicate",
```

A symbol that does not exist is a hard failure, so the table cannot drift ahead of the code. A symbol
that exists but nothing reaches is also a failure, because a guard nothing reaches is indistinguishable
at runtime from a guard nobody wrote.

## What a good change looks like here

**Say why, not what.** The code says what it does. A comment earns its place by explaining the
alternative that was rejected and what it would have cost. Several modules in this repository carry a
paragraph about a defect that was measured and fixed; those paragraphs are the most valuable text in
the file, because they are the only defence against the fix being undone by someone who finds the
simpler version cleaner.

**No silent fallbacks.** This is the rule most often broken with good intentions. A default that makes
a check pass is worse than a crash:

```python
# No. A challenge with no resource_limits now passes the budget gate.
maximum = int(limits.get("maximum_rcc", 0))
if maximum and measured > maximum: ...

# Yes. Unverifiable is not satisfied.
maximum = _ceiling(limits, "maximum_rcc")
if maximum is None:
    return GateResult(Gate.BUDGET, False, "the budget cannot be verified")
```

That exact defect shipped and was found by an audit rather than by a test. Ask of every default:
*if this fires, what stops being checked?*

**Measure incentives, do not reason about them.** Two real defects here were found by computing what a
strategy would score, not by reading the code:

- redistributing the rank weight of missing ideas made a one-idea portfolio score 900,000 ppm against
  a genuinely diverse portfolio's 777,000 — padding beat diversity;
- a 17.5% weight cap flattened every field below six qualifiers to an equal split, removing any
  incentive to be the better of two laboratories.

If you touch `validator/scoring/`, `validator/weights.py` or the rank weights, write the test that
computes what the cheapest strategy earns.

**Errors name the consequence.** `"invalid config"` sends someone to read source. Say what the value
would have broken:

```python
raise CycleError(
    f"salt_commit_offset ({self.salt_commit_offset}) is not before randomness_offset "
    f"({self.randomness_offset}). A validator that commits its salt after seeing the randomness "
    "can grind the salt until the derived seed produces a pack it likes."
)
```

## Integers, not floats, in anything hashed

Every ratio in the protocol is a parts-per-million integer. `protocol/canonical.py` refuses a float in
a hashed object and names the path where it found one, because a value read as `0.1` and a value
computed as `0.05 + 0.05` are different doubles — and two hosts would then disagree about the bytes.

Floats are permitted in exactly two places, both at a boundary: the Bradley-Terry fit, and an
embedding vector a provider returned. Both convert to ppm before anything downstream sees them.

## Layout

Where a change belongs:

| If you are changing | Go to |
|---|---|
| something both sides must agree on byte-for-byte | `protocol/` |
| anything that talks to the chain | `chain/client.py` — the only module that imports `bittensor` |
| how a laboratory reaches a model | `gateway/` |
| how problems are generated or filtered | `validator/challenge_factory/` |
| what invalidates a response | `validator/scoring/gates.py` |
| how a portfolio is scored | `validator/scoring/`, `validator/judge/` |
| how emission is allocated | `validator/weights.py` |
| what a miner runs or submits | `miner/` |
| what the public sees | `portal/` — read-only by construction; it has no write path |

## Tests

`tests/unit` for a module in isolation, `tests/integration` for a composed surface over HTTP,
`tests/adversarial` for an attack (assert both that it fails *and* that evidence was recorded),
`tests/measurement` for §27's mainnet gates, `tests/localnet` for anything needing a chain.

Mark a test `@pytest.mark.determinism` if it must reproduce under a different hash seed. That is most
of the scoring path.

## Commits

Say what changed and why it matters. If a change fixes a defect, say what the defect would have cost —
that sentence is what stops the fix being reverted later as an unnecessary complication.

Use the repository's git identity for commits; do not add co-author trailers.
