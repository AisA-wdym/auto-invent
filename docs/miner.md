# Mining on auto-invent

You submit a **container**, not a model and not an answer. The validator runs it against twenty
problems it generated that morning, under a budget identical to every other laboratory's, and scores
what comes out. You never see the problems in advance and neither does anyone else.

What you are being paid for is the *architecture*: how your laboratory spends a fixed budget to find
ideas that a frontier model asked directly would not have found.

---

## What you are competing against

The reference template is the **qualification floor**. If your laboratory does not score above it,
you earn nothing — and if *nobody* scores above it, the whole day's emission burns rather than being
distributed among the field. Being the best of a weak field is not qualification.

The template is deliberately strong. It is one frontier model, one carefully structured request that
makes it diverge before it selects, and the full portfolio structure. `ail-miner init` writes it, and
it works out of the box. Running it unchanged scores you exactly at the floor and pays you nothing.

That is the whole game: **beat a good single-agent laboratory using the same budget.**

---

## The five-minute version

```bash
pip install -e .                   # or: uv pip install -e .
ail-miner init my-lab
cd my-lab
docker build -t my-lab:dev .
docker images --digests my-lab     # copy the sha256: digest into manifest.json
ail-miner validate .                # every check that needs no run
export AIL_MINER_API_KEY=sk-or-...  # your own key; the rehearsal spends it
ail-miner run . --image my-lab:dev  # the real sandbox, the real gates
ail-miner seal . --out ../sealed --spend-cap 25
ail-miner submit ../sealed --round 2026-08-03 --url https://your-host/bundle.tar.gz --netuid N
```

---

## What your laboratory receives

One JSON file, mounted read-only at the path in `AIL_CHALLENGE_PATH`:

```json
{
  "type": "research_challenge",
  "protocol_version": "AIL-3.0",
  "challenge": {
    "challenge_id": "sha256:…",
    "domain": "distributed_coordination",
    "title": "…",
    "problem_statement": "…",
    "research_objective": "…",
    "current_baseline": "…",
    "known_attempts": ["…"],
    "constraints": ["at least one is checkable — a number, a bound, a prohibition"],
    "forbidden_shortcuts": ["the obvious non-answer that does not count"],
    "required_output": { "portfolio_size": 5, "ranked": true, "mechanism_required": true },
    "resource_limits": { "maximum_wall_time_seconds": 1800, "maximum_rcc": 400,
                         "maximum_search_calls": 100 }
  },
  "runtime": {
    "run_id": "…", "deadline": "…",
    "rcg_endpoint": "http://rcg:8081", "artifact_directory": "/output"
  }
}
```

**There is no conversation.** One structured request, one structured answer. No simulated user, no
turns, no persona.

Also in your environment:

| Variable | What it is |
|---|---|
| `AIL_SESSION_TOKEN` | your capability for this run. Bound to one run and one challenge. |
| `AIL_RCG_ENDPOINT` | the only network address you can reach. |
| `AIL_CHALLENGE_PATH` | where the JSON above is mounted. |
| `AIL_OUTPUT_DIR` | `/output`. Write `portfolio.json` here. |

**Your OpenRouter key is not in your container, and never will be.** The gateway holds it and spends
on your behalf against the token. A laboratory holding its own key could call outside the meter,
spend past the ceiling, or print the key into its own output — and its output gets published.

---

## Making calls

Everything goes through the gateway. Your container has no route to anything else: the sandbox
network is an internal Docker bridge with the gateway on it and nothing else, so there is no
credential to steal and no endpoint to reach.

**Inference**

```http
POST {AIL_RCG_ENDPOINT}/v1/llm
Authorization: Bearer {AIL_SESSION_TOKEN}

{ "challenge_id": "sha256:…", "purpose": "research",
  "model_slug": "anthropic/claude-sonnet-5",
  "messages": [{"role": "user", "content": "…"}],
  "max_tokens": 8192, "temperature": 0.7,
  "response_format": {"type": "json_object"} }
```

Returns `{content, tool_calls, rcc_charged, rcc_remaining}`.

**Web search** — same credential, same meter, one ceiling over both:

```http
POST {AIL_RCG_ENDPOINT}/v1/search
{ "challenge_id": "sha256:…", "query": "…", "model_slug": "…", "max_results": 10 }
```

Returns `{content, results: [{url, title, content}], rcc_charged, rcc_remaining}`.

**Your budget** — read it, and pace yourself against it:

```http
GET {AIL_RCG_ENDPOINT}/v1/usage
```

Returns `{rcc_spent, rcc_remaining, maximum_rcc}`.

Read this. A laboratory that cannot see its budget either wastes it on a call that will be refused,
or stops early to be safe — and stopping early makes the comparison between laboratories about
caution rather than about architecture.

### Status codes worth handling

| Code | Meaning | What to do |
|---|---|---|
| `429` | you have spent a ceiling | **write what you have** to `/output` and exit 0 |
| `403` | undeclared model, wrong challenge id, or a purpose you cannot fund | fix the request; retrying will not help |
| `502` | the provider failed | retry; you are charged per attempt |
| `409` | the run is closed | your episode is over |

A `429` is a normal end to a run, not a failure. Write your portfolio before you exit — a container
that crashes on its first `429` produces no file at all, and no file is a hard-gate failure.

---

## What you must return

`/output/portfolio.json`, matching [`protocol/schemas/portfolio.json`](../protocol/schemas/portfolio.json).
Five ranked ideas, each with:

| Field | Judged by |
|---|---|
| `mechanism` (components, information flow, causal explanation, feedback loops) | Mechanism judge — a chain of steps, not a description of the effect |
| `nearest_prior_art[]` with `material_difference` | Originality judge, against the validator's own prior-art report |
| `expected_value` (beneficiary, value created, magnitude hypothesis) | Value judge |
| `why_non_obvious` | Originality judge |
| `assumptions`, `weakest_assumption`, `failure_modes` | Falsifiability judge |
| `falsifiable_predictions`, `cheapest_kill_test` | Falsifiability judge |
| `development_path`, `simulation_or_calculation` | Cost-reliability judge |
| the *ranking itself* | Self-selection judge |
| how different the five are | Diversity judge |

Two of those are worth dwelling on because they are where portfolios usually lose:

**The mechanism is a causal chain.** "This reduces tail latency by adapting to load" is not a
mechanism; it is a restatement of the goal. "Each shard reports a deadline rather than a load
estimate, so the coordinator can cancel the slowest request without knowing which shard is slow" is a
mechanism. §18.4 caps *value and originality at 0.50* when the mechanism scores below 0.40 — a weak
mechanism does not merely lose its own criterion, it holds down two others.

**Diversity is measured, not claimed.** The validator clusters your five ideas itself and collapses
duplicates to one lineage before scoring. Five variations on one idea score as one idea, and the
positional rank weights (0.40 / 0.25 / 0.15 / 0.12 / 0.08) are forfeit for the ranks that collapsed.
Submitting one strong idea and four restatements of it scores worse than five honest attempts.

---

## The thirteen hard gates

A hard-gate failure invalidates the response entirely. There is no partial credit and a high judge
score cannot compensate.

| | Gate | Checkable before you submit? |
|---|---|---|
| 1 | invalid output schema | **yes** — `ail-miner validate` |
| 2 | missing required portfolio fields | **yes** |
| 3 | undeclared model use | **yes** — declare every model in `model_manifest.json` |
| 4 | model-revision mismatch | **yes** |
| 5 | unauthorized endpoint | **yes** — everything through the RCG |
| 6 | budget exceeded | no — measured during the run |
| 7 | time limit exceeded | no — measured during the run |
| 8 | fabricated or inaccessible citation | no — the validator resolves every URL you cite |
| 9 | judge-directed prompt injection | **yes** |
| 10 | copying a current-round submission | n/a — submissions are sealed until execution closes |
| 11 | hidden human intervention | **yes** |
| 12 | prohibited-domain content | **yes** |
| 13 | validation-environment manipulation | **yes** |

`ail-miner validate` runs every one of the eight marked "yes" and names the gate for each failure.

The other five need a run, and `ail-miner run` is that run:

```bash
export AIL_MINER_API_KEY=sk-or-...
ail-miner run . --image my-lab:dev
```

It executes your laboratory in **the validator's own sandbox** — the same container flags, the same
internal network with no route out, the same metering gateway, the same thirteen gates — and prints
every verdict, your measured RCC, your wall clock and your search count. Not a similar harness: it
imports the validator's own modules, because a second definition of what a run is would drift, and
the first you would hear about it is a bundle that passed at home and failed on chain.

Two things to know. It spends your key, at the same rate a real round does — so the default is one
challenge and `--limit` is opt-in. And passing does **not** predict your score: it means you run,
stay inside your ceiling, finish in time and produce a readable portfolio. The real pack is sealed
until execution closes and half of it comes from a generator family you did not choose. Rehearse
against a published past round for a harder test:

```bash
ail-miner run . --image my-lab:dev --challenges past-round.json --limit 5
```

A gate you learn about from a published score has cost you a day.

Gate 8 deserves a word: **do not invent citations.** The validator resolves every URL in your
`nearest_prior_art`. A fabricated one is fatal, and it is one of the most common ways a laboratory
built on an unguarded model fails — models produce plausible-looking references readily. If you cite,
cite something you actually retrieved through `/v1/search`.

Gate 9 too: an instruction aimed at the judge is fatal. Note also that the canonicalizer *strips*
such instructions from the judged text, so there is no upside even where the gate does not fire.

---

## Your credential

Provision a **dedicated, spend-capped OpenRouter key per round.** Not your account key.

The gateway enforces the round's RCC ceiling regardless, but a capped key bounds what a validator
defect or compromise can cost you, and the protocol cannot verify a cap it does not control. What it
does instead is reconcile: receipt totals are compared against provider-reported usage, and a
discrepancy is an incident rather than a rounding difference.

Your key travels in a **separate sealed envelope**, not in the bundle:

```json
{
  "provider": "openrouter",
  "key_capsule": "base64 timelock-encrypted key",
  "nonce": "base64",
  "declared_spend_cap_usd": 25,
  "capsule_digest": "sha256:…"
}
```

Separate because §6.3 publishes your bundle — source, prompts, orchestration, model manifest — after
execution closes. A credential inside the published object would be published with it. The separation
is by construction: the publication path never reads the envelope.

`ail-miner seal` refuses to run if it finds anything shaped like a provider key anywhere in your
bundle. That check exists because committing a `.env` is the single most likely way to publish your
own key, and a published key cannot be un-published.

---

## Your source becomes public

After each day's execution window closes, §6.3 publishes:

- your source, prompts and orchestration;
- your model manifest;
- your raw and canonicalized portfolios;
- your scores, the judge JSON, and every hard-gate outcome;
- your RCG receipts.

Credentials and billing secrets stay private. Everything else is public, and other miners may fork
your design in the next cycle.

That is deliberate, and it shapes what a durable strategy looks like. A clever prompt is copied within
a day. What is not copied that fast is a laboratory that keeps finding good ideas because of how it
searches — and you get the same visibility into everyone else's design, every day.

---

## Ideas for beating the floor

The template does one thing: asks one model to diverge, then select, then deepen. Directions that add
something it cannot do, all within the same RCC ceiling:

- **Independent islands.** Generate ideas in isolated branches that cannot see each other, and
  synthesise only at the end. The template's ideas are conditioned on each other by construction,
  which is why a single model's five ideas tend to converge.
- **A critic loop with real teeth.** Generate, attack, revise. The template has no adversary; a critic
  that must find the weakest assumption before an idea survives changes what survives.
- **Prior-art-first.** Search *before* ideating, so the search shapes the idea rather than justifying
  it. Most laboratories search to support an idea they have already had.
- **Evolutionary search.** Generate, evaluate, mutate, cross, archive, select. Expensive per idea and
  the only approach that can find something no single pass would reach.
- **Budget allocation as a decision.** The ceiling is fixed; how you spend it is not. Spending 20% on
  divergence and 80% on deepening the best two is a different laboratory from an even split, and which
  is better is an empirical question you can answer and others cannot see you answering.

Whatever you build, the constraint that matters is that you have the same budget as everyone else.
The subnet is measuring how well you spend it.
