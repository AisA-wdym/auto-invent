# Running a validator

A validator generates its own problems, runs every miner's laboratory under identical conditions,
and submits weights. It does not receive problems from the owner and it does not grade by hand.

You will need: a machine that can run Docker, a Bittensor wallet registered as a validator on the
subnet, an OpenRouter account, and Redis.

---

## Sizing

| | |
|---|---|
| CPU | 8 vCPU minimum. One laboratory container at a time gets 2048 CPU shares; the validator's own threads share the rest. |
| RAM | 16 GB. A laboratory is capped at 8 GB and swap is disabled (a leaking laboratory must be killed, not swapped — swapping consumes host I/O that every *later* laboratory's measured wall time shares). |
| Disk | 100 GB. Container images dominate; each miner ships one. |
| Network | outbound HTTPS to OpenRouter and your subtensor endpoint. **No inbound.** |

**Your OpenRouter spend is the real cost.** Per day: twenty problems × four candidates × (generation
+ critique), plus the discrimination probe, plus eight judge panels × three families across the
tournament. Budget for it deliberately and watch it — the validator has its own RCC ledger, capped
per round, so a generation loop cannot spend your balance overnight.

---

## Setup

### 1. Credentials

```bash
# Preferred: a file, so it can be permission-restricted.
install -m 0400 /dev/stdin /run/secrets/openrouter <<< "sk-or-v1-…"
export AI_VALIDATOR_OPENROUTER_KEY_FILE=/run/secrets/openrouter
```

An environment variable also works (`AI_VALIDATOR_OPENROUTER_KEY`) and is what most deployments
actually do, but anything that can list your process can read it.

There is **no fallback**. Challenge generation and judging are validator costs; if the key is absent
the validator refuses to start rather than using a miner's key, because that would make the
equal-budget guarantee a fiction.

### 2. Verify the deployment before trusting it with anything

```bash
python -m validator --check
```

This builds the entire object graph and validates it: the season config parses, the cycle's orderings
hold, every judge panel satisfies §16.1's family cap, every model is snapshot-pinned, the criterion
weights sum to exactly one whole, the slot plan deals the declared counts. Then it exits.

No chain, no network, no credential. Run it in CI and run it after every config change.

```bash
python -m gateway --check      # the same, for the gateway
```

If either fails, it names each problem and what it would have cost. Do not proceed past a failure.

### 3. Bring up the gateway, Redis and the sandbox network

```bash
docker compose -f ops/docker-compose.yml up -d
```

That creates:

- the **gateway** (RCG) — the only address a laboratory can reach;
- **Redis** — bound to loopback, holding challenge packs and dedup fingerprints;
- `auto-invent-sandbox` — an **internal** Docker bridge with the gateway attached and nothing else.

`--internal` is the control that matters. It removes the NAT rule that would otherwise give a
container on the bridge a route to the internet, so a laboratory has no outbound path except to the
gateway. The validator checks this itself (`assert_egress_confined`) before running anything.

Redis **must not be reachable from the sandbox.** A laboratory that could reach it could read the
whole pack — every problem, including the ones it has not been given yet, and other rounds' packs.
The validator refuses to start against a Redis bound to `0.0.0.0` and warns loudly about any
non-loopback address.

### 4. Run

```bash
python -m validator \
  --netuid N \
  --network finney \
  --wallet my-wallet --hotkey my-validator \
  --redis-url redis://127.0.0.1:6379/0 \
  --rcg-endpoint http://rcg:8081
```

Or with pm2, which is what most operators use:

```bash
pm2 start ops/validator.sh --name auto-invent-validator
pm2 logs auto-invent-validator
```

---

## The daily cycle

Seven timed interactions, all measured in **blocks** rather than wall clock — a block height is the
same fact for everyone reading the same chain, and wall clock would make "before the randomness" a
question about NTP.

| Offset | Phase | What happens |
|---|---|---|
| T-600 | submission close | miners can no longer commit |
| T-450 | salt commit | you commit a salt, **before** the randomness that will be mixed with it exists |
| T-300 | randomness | the post-deadline block hash is drawn |
| T-100 | pack commit | the pack's hash goes on chain, **before** the pack is written to Redis |
| T+0 | reveal | sealed bundles open; execution begins |
| T+4200 | execution close | containers are terminated |
| T+6900 | weights | the vector is submitted |

Three of those orderings are the whole security argument, and each is checked at load:

- **salt before randomness** — otherwise you could grind the salt until the derived seed produced a
  pack you liked;
- **randomness before generation** — the seed needs it;
- **pack hash before reveal** — otherwise you could read a submission and regenerate your challenges
  to suit it.

A config that breaks any of them is refused with a message saying which guarantee it broke.

### What the pack commitment does and does not say

It states the pack hash, the challenge count, and how many challenges each generator family wrote. It
does **not** say which slot came from which family. Publishing the per-slot split before execution
would tell a laboratory which half of the pack to expect from whom, and the point of two generator
families is that it cannot know. The attribution becomes public at §6.3 disclosure, which is when it
should.

---

## What generation costs, and what it buys

Twenty problems per day. Each slot generates four candidates and keeps the first that survives:

```
linter (free)  →  safety (free)  →  dedup (free)  →  critic (1 call)  →  probe (expensive)
```

Cheapest-first, so a candidate the linter rejects never reaches the probe. The discrimination probe is
the single largest cost — it runs the reference template in several configurations plus a degraded
answer — and it runs last.

Watch `rejections_by_step()` in the logs. Its distribution is your earliest health signal:

| Rising count | What it means |
|---|---|
| `dedup` | the generator is repeating itself; problem supply is narrowing |
| `discrimination` | problems are getting easier, or the judge panel is getting worse |
| `critic` | usually a provider issue rather than a generation issue |
| `linter` | the generation prompt has drifted from the schema |

A pack that cannot be filled **raises** rather than shipping short. Nineteen challenges scored against
a twenty-challenge commitment cannot be verified by anyone, and a validator quietly shipping short
packs would diverge from its peers in a way that looks like a scoring disagreement rather than a
generation failure.

---

## Divergence between validators is expected

§27 draws a distinction worth internalising:

| Measurement | Requirement | Why |
|---|---|---|
| **cross-validator** rank correlation | ≥ 0.60 | you generate your own problems, so you *should* disagree with your peers. Perfect agreement would mean the problems are not doing any work. |
| **same-bundle rerun** correlation | ≥ 0.80 | rerunning the same bundle on the same problems must reproduce. This is the property a clock read or a global RNG in the scoring path silently destroys. |

`make purity` enforces the second by forbidding the scoring modules from even importing a clock, an
RNG or the network. No test can catch that on its own — a process compared against itself agrees
perfectly.

---

## Failure modes and what to do

| Symptom | Cause | Action |
|---|---|---|
| `validator cannot start — N config problem(s)` | the season config | fix what it names; each message says what the check buys |
| `no --redis-url: using an in-memory challenge store` | Redis not configured | **configure Redis before mainnet.** A restart mid-round loses a pack whose hash is already committed, and it cannot be regenerated because the seed's randomness has passed |
| `the network … is not internal` | the sandbox bridge has a route out | `docker network rm auto-invent-sandbox` and let the validator recreate it, after checking what else is attached |
| `pack for … is not the committed pack` | the stored pack was edited, or two packs share a date | do not score against it; investigate |
| `order-swap inconsistency … exceeds the ceiling` | a judge is measuring presentation position, not content | its verdicts should be discounted and the judge removed from the panel |
| `weight submission failed after N attempts` | chain unreachable | the cycle must **not** proceed as if it succeeded; an unsubmitted vector leaves the previous one in force, paying yesterday's ranking for today's work |
| receipt reconciliation mismatch | spend outside the receipted path | an **incident**, not a rounding difference. Investigate before the next round |

---

## What you must never do

**Never use a miner's credential for your own work.** Challenge generation, critique, judging and
prior art are your costs. A validator that billed judging to a miner's key could exhaust a
rival-sponsored laboratory at will, and the equal-budget guarantee would be fiction.

Under one provider surface this would *succeed* — the API authenticates, returns a completion, and
bills the wrong party. The code refuses it structurally (two distinct credential types, purpose
selects), and every receipt records `credential_owner` so per-account totals can be reconciled. That
reconciliation is the only check that catches what the API will not.

**Never expose Redis to the sandbox network.** See above.

**Never run a laboratory image by tag.** Only by `sha256:` digest. A tag can be repointed after the
deadline, which is exactly what §6.1 exists to prevent. The runner refuses a tag.

---

## Operational checklist

Daily:

- [ ] `python -m validator --check` passes
- [ ] the sandbox network is still internal
- [ ] your OpenRouter spend is within expectation
- [ ] `rejections_by_step()` has not shifted
- [ ] the weight vector was submitted, and included

Per season:

- [ ] every model in the season config is snapshot-pinned to an immutable route
- [ ] the reference template's digest matches what you are running
- [ ] `reference_labs` has at least two configurations, or the discrimination probe cannot measure
      spread at all
- [ ] receipt totals reconcile against provider-reported usage
