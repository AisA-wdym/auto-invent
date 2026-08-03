# Autonomous Invention Lab Subnet

## Updated Technical Architecture v3.0

## Executive definition

> **A Bittensor subnet in which miners submit complete autonomous invention laboratories, while validators independently generate fresh research problems every day, execute every laboratory under controlled budgets, and score the resulting idea portfolios through deterministic integrity checks and calibrated multi-model LLM judging.**

The subnet does **not** validate miners through simulated user conversations. A validation episode is now:

```text
Structured research problem
        ↓
Miner’s autonomous invention lab
        ↓
Ranked Top-5 invention portfolio
        ↓
Deterministic validation
        ↓
Blinded multi-model LLM evaluation
        ↓
Rolling miner score
        ↓
Bittensor weights
```

The existing foundational principles remain unchanged: miners submit the laboratory itself,
miners fund its inference and research costs, validators control execution, model versions
are locked, and validation remains automated.

One provider surface, two accounts. **Every model call in the subnet goes through
OpenRouter** — miner research, challenge generation, judging and prior-art retrieval alike.
Miners spend **their own OpenRouter key**, which the gateway holds and the sandbox never
sees; validators spend **their own OpenRouter key**. Same provider, separate accounts, and
the two must never be confused for one another. 

---

# 1. Subnet objective

The subnet’s digital commodity is not an individual research answer.

It is:

> **A versioned, executable autonomous invention-lab bundle capable of receiving an unfamiliar research problem and producing valuable, non-obvious, technically coherent ideas that a human researcher could seriously consider.**

The primary capability is:

1. discovering unconventional research directions;
2. producing genuinely differentiated ideas;
3. selecting its own strongest ideas;
4. explaining why they may work;
5. identifying assumptions and failure modes;
6. proposing simulations, calculations or experiments for deeper investigation.

Simulation and implementation support remain important, but they are subordinate to the central capability of **inventing and selecting valuable ideas**.

---

# 2. Initial scope

The first production season should be limited to research domains where an LLM panel can reasonably inspect mechanisms, constraints, prior art and feasibility without requiring a physical laboratory.

## Supported V1 domains

* software architecture;
* algorithms and optimization;
* AI-agent architecture;
* model orchestration;
* retrieval and memory systems;
* distributed systems;
* database and information-retrieval design;
* digital protocols;
* decentralized mechanism design;
* simulation-only technical systems;
* technically defined digital products.

## Excluded from V1 scoring

* clinical or medical treatment recommendations;
* chemical or biological wet-lab inventions;
* physical engineering requiring unavailable measurements;
* legal or policy recommendations;
* weapons, malware or exploit development;
* broad artistic ideation;
* topics whose value cannot be evaluated without long-term real-world outcomes.

The domain list may expand after the validator mechanism demonstrates adequate reliability.

---

# 3. Participants and responsibilities

## 3.1 Miner

The miner develops and submits an **Autonomous Invention Lab Bundle**.

The miner controls:

* internal agent architecture;
* prompts and skills;
* foundation-model selection;
* model routing;
* search strategy;
* memory design;
* idea-generation methods;
* independent research branches;
* evolutionary or quality-diversity search;
* prior-art analysis;
* self-criticism;
* self-ranking;
* simulation and calculation tools.

The protocol does not require a particular number of agents or a particular reasoning architecture.

The miner is rewarded only for externally measured laboratory performance.

## 3.2 Validator

Each validator independently:

1. generates fresh challenge packs every day;
2. commits the challenge-pack hash before miner execution;
3. executes miner bundles under equal conditions;
4. records all model, search and tool usage;
5. applies hard validity gates;
6. canonicalizes miner outputs;
7. performs independent prior-art retrieval;
8. runs calibrated LLM judge panels;
9. calculates daily and rolling miner scores;
10. independently submits Bittensor weights.

Bittensor expects validators to independently assess miners and set weights; commit-reveal conceals those weights temporarily to discourage copying. The protocol is most effective when miner rankings change frequently enough that stale weights become inaccurate. ([Bittensor Documentation][1])

## 3.3 Subnet owner and protocol team

The owner defines only the public protocol:

* domain taxonomy;
* challenge schema;
* allowed resource classes;
* hard-gate rules;
* output schema;
* scoring formula;
* judge requirements;
* update and disclosure rules;
* safety constraints;
* software-version requirements.

The owner does **not** create the daily validation questions and does not manually grade miner answers.

## 3.4 Research Compute Gateway

The Research Compute Gateway, or **RCG**, brokers all external calls.

It provides:

* miner-funded model access, using the miner's own **OpenRouter** credential;
* web search through the same provider;
* scoped, temporary session tokens issued to the runner, never to the laboratory;
* model-version enforcement;
* query and response hashing;
* token and cost metering;
* budget enforcement;
* signed usage receipts;
* endpoint allowlisting.

This follows a production-proven pattern: ORO executes miner-submitted agents in isolated sandboxes and routes search and inference through controlled services rather than permitting direct unrestricted internet access. ([ORO Subnet][2])

### 3.4.1 One provider for miners: OpenRouter

Miners reach every foundation model through **OpenRouter**, and OpenRouter also serves web
search. One provider surface rather than several, for three reasons:

1. **Model choice stays open.** OpenRouter fronts Anthropic, OpenAI, Google, Meta,
   Mistral, DeepSeek, Qwen and open-weight hosts behind one API, so a miner still selects
   freely — including a private fine-tune it hosts and routes through.
2. **One metering surface.** Usage and cost arrive in a single accounting format, which is
   what makes an equal RCC ceiling comparable between two laboratories that chose entirely
   different models.
3. **One allowlist to enforce.** Model-version pinning and endpoint allowlisting have one
   place to be right rather than one per provider adapter.

### 3.4.2 Who pays, and whose key is used

The miner funds its own laboratory, and it does so with **its own OpenRouter credential**,
supplied inside the sealed submission. The validator uses that credential to run that
miner's laboratory, and only that laboratory.

Two properties are non-negotiable, and both are consequences of where the credential sits
rather than of anyone's good behaviour.

**The laboratory never holds its own key.** The credential is decrypted by the RCG at
reveal and is used *by the RCG* on the laboratory's behalf. The container receives a
short-lived session token bound to one run, and nothing else. A laboratory that held its
own key could call outside the meter, exfiltrate the key in its own output, or spend beyond
the ceiling — so the key is never written into the sandbox's filesystem, environment, or any
response the laboratory can read.

**The credential is never published.** Section 6.3 publishes source, prompts and model
manifests after execution closes. The credential is not part of that: it travels in a
separate sealed field that is excluded from publication by construction, not by a filter
that could be forgotten.

Miners **must** provision a dedicated, spend-capped key per round rather than a general
account key. The RCG enforces the round ceiling regardless, but a capped key means a
validator defect or compromise cannot exceed what the miner chose to risk. The protocol
cannot verify a cap from the outside, so this is a published expectation with a
corresponding measurement: receipt totals are reconciled against provider-reported usage
(section 27), and a discrepancy is an incident rather than a rounding difference.

### 3.4.3 Validator-funded calls use the validator's own OpenRouter key

Challenge generation, critique, judging and prior-art retrieval are **validator** costs
(section 5.4), and they go through OpenRouter on the **validator's own** key. Same provider
as the miners, a different account.

One surface everywhere buys a real simplification: one adapter, one metering format, one
allowlist, one credential type, and an operator provisions one API key rather than three.

### 3.4.4 The credential separation must be enforced, not assumed

A miner's key and a validator's key must never be used for each other's work. The reason is
unchanged: a validator that could bill its own judging to a miner's credential would make
the equal-budget guarantee a fiction and could exhaust a rival-sponsored miner's balance at
will.

What *has* changed is that this is no longer enforced for free. When the two sides used
different providers, a swapped credential failed immediately — wrong endpoint, wrong request
shape, wrong error. With one provider, **a swapped key works.** It authenticates, it
returns a completion, and it silently bills the wrong party. Nothing surfaces until someone
reconciles an invoice.

So the invariant becomes explicit, in three parts:

1. **Two resolvers, not one keyed store.** The RCG holds the miner credential and the
   validator credential behind distinct objects with distinct types. There is no lookup that
   takes an owner as a parameter, because a parameter can be passed the wrong value.
2. **Every call declares its purpose, and purpose selects the credential.** A research call
   cannot reach the validator resolver; a judging call cannot reach the miner resolver.
   Mismatch raises before the request is built, not after it succeeds.
3. **Every receipt records `credential_owner`, and it is reconciled.** Section 27 requires
   100% receipt reconciliation against provider-reported usage. Because both accounts now
   report in the same format, a call billed to the wrong account is *detectable* by
   comparing per-account totals — which is the only check that catches a defect the API
   itself will not.

Point 3 is the one that makes the other two auditable rather than merely intended.

---

# 4. High-level architecture

```text
┌──────────────────────────────────────────────────────┐
│                Bittensor / Subtensor                 │
│ Registration · Metagraph · Commit-Reveal · Weights  │
└──────────────────────────┬───────────────────────────┘
                           │
              ┌────────────▼─────────────┐
              │     Season Registry      │
              │ Domains · Rules · Rubric │
              │ Models · Budgets · Gates │
              └────────────┬─────────────┘
                           │
       ┌───────────────────▼───────────────────┐
       │        Sealed Bundle Registry         │
       │ Encrypted source · manifest · digest  │
       │ model manifest · billing delegation   │
       └───────────────────┬───────────────────┘
                           │
       ┌───────────────────▼───────────────────┐
       │       Validator Challenge Factory     │
       │ Daily seed · LLM generation · linter  │
       │ dedup · difficulty · hash commitment  │
       └───────────────────┬───────────────────┘
                           │
       ┌───────────────────▼───────────────────┐
       │         Validator Execution Plane     │
       │ Sandbox · runner · RCG · artifact log │
       └──────────────┬───────────────┬────────┘
                      │               │
              ┌───────▼──────┐  ┌────▼─────────┐
              │ Miner Lab A  │  │ Miner Lab N  │
              └───────┬──────┘  └────┬─────────┘
                      └───────┬───────┘
                              │
       ┌──────────────────────▼───────────────────────┐
       │             Evaluation Pipeline              │
       │ gates → canonicalize → prior art → judging  │
       │ pairwise tournament → scoring → replication │
       └──────────────────────┬───────────────────────┘
                              │
       ┌──────────────────────▼───────────────────────┐
       │           Validator Score Database           │
       │ Daily results · rolling ratings · receipts   │
       └──────────────────────┬───────────────────────┘
                              │
                ┌─────────────▼─────────────┐
                │ Commit-Reveal Weight Set  │
                └───────────────────────────┘
```

---

# 5. Miner submission architecture

## 5.1 Submission unit

A miner submits one complete versioned bundle:

```text
AutonomousInventionLabBundle
```

Recommended repository structure:

```text
invention-lab/
├── Dockerfile
├── manifest.json
├── src/
│   ├── entrypoint.py
│   ├── orchestration/
│   ├── ideation/
│   ├── research/
│   ├── prior_art/
│   ├── critics/
│   ├── ranking/
│   ├── simulation/
│   └── memory/
├── prompts/
├── skills/
├── schemas/
├── requirements.lock
├── SBOM.json
├── LICENSE
└── tests/
```

## 5.2 Bundle manifest

```json
{
  "protocol_version": "AIL-3.0",
  "bundle_id": "5FminerHotkey/lab-alpha",
  "bundle_version": "1.4.0",
  "round_id": "2026-08-03",
  "entrypoint": "/app/run_lab",
  "container_digest": "sha256:...",
  "source_archive_hash": "sha256:...",
  "lockfile_hash": "sha256:...",
  "sbom_hash": "sha256:...",
  "license": "Apache-2.0",
  "supported_domains": [
    "software_architecture",
    "algorithms",
    "ai_agent_systems"
  ],
  "output_schema": "research_portfolio_v1",
  "miner_signature": "..."
}
```

## 5.3 Model manifest

Every externally invoked model must be declared before submission closes.

```json
{
  "models": [
    {
      "alias": "broad_ideator",
      "provider": "openrouter",
      "model_slug": "anthropic/claude-sonnet-4.5",
      "model_snapshot": "fixed-season-snapshot",
      "parameters": {
        "temperature": 0.9,
        "max_tokens": 10000
      },
      "role": "idea_generation"
    },
    {
      "alias": "final_critic",
      "provider": "openrouter",
      "model_slug": "openai/gpt-5",
      "model_snapshot": "fixed-season-snapshot",
      "parameters": {
        "temperature": 0.2
      },
      "role": "critique"
    },
    {
      "alias": "house_model",
      "provider": "openrouter",
      "model_slug": "miner-org/research-model",
      "hf_repo": "miner/research-model",
      "revision": "40-character-commit-sha",
      "role": "idea_generation"
    }
  ],
  "routing_config_hash": "sha256:...",
  "maximum_parallel_calls": 16
}
```

Model choice is open. Every model is reached through **OpenRouter**, which is the single
provider surface, but the *selection* behind it is unrestricted:

* Anthropic, OpenAI, Google, Meta, Mistral, DeepSeek, Qwen and other hosted families;
* open-weight models;
* a private fine-tune the miner hosts and routes through OpenRouter;
* several models in one laboratory;
* one specialised custom model.

`model_slug` is the OpenRouter route. `model_snapshot` pins the season-fixed version of it,
because a provider that silently moves a slug changes what every laboratory is running
mid-season — so the snapshot rather than the slug is the identity the gateway enforces.

A miner-hosted model additionally declares `hf_repo` and a full 40-character `revision`.
An abbreviated revision is refused: abbreviations become ambiguous as a repository grows,
and pinning exists precisely so the artifact cannot move.

Web search is reached through the same credential and the same meter, so a laboratory's
search spend and inference spend are bounded by one ceiling rather than two.

## 5.4 Research-compute funding

The miner funds its laboratory’s:

* inference;
* search;
* embedding;
* simulation;
* code execution;
* external tool use.

The validator funds:

* challenge generation;
* validation infrastructure;
* judge-model calls;
* prior-art verification;
* score calculation.

## 5.4.1 The sealed credential

The miner's OpenRouter credential is submitted as a **separate sealed field**, alongside
the sealed bundle and under the same timelock:

```json
{
  "credential_envelope": {
    "provider": "openrouter",
    "key_capsule": "base64 timelock-encrypted key",
    "nonce": "base64",
    "declared_spend_cap_usd": 25,
    "capsule_digest": "sha256:..."
  }
}
```

Separate rather than a field inside the manifest, and the separation is the control:
section 6.3 publishes the bundle after execution closes, and a credential inside the
published object would be published with it. A distinct envelope can be excluded by
construction — the publication path never reads it — rather than by a filter someone has to
remember to keep correct.

`declared_spend_cap_usd` is what the miner states it provisioned. The protocol cannot
verify a cap it does not control, so this is recorded and **reconciled**, not trusted: the
RCG's receipt totals are compared against provider-reported usage, and a mismatch is an
incident (section 27).

Raw provider credentials are never included in the public bundle, and are never present in
the sandbox at all — the RCG holds the key and the container holds only a session token.

The RCG issues a short-lived token bound to:

```json
{
  "miner_hotkey": "5F...",
  "bundle_digest": "sha256:...",
  "validator_hotkey": "5G...",
  "challenge_id": "sha256:...",
  "run_id": "...",
  "allowed_models": ["openrouter/anthropic/claude-sonnet-4.5"],
  "maximum_rcc": 400,
  "maximum_requests": 500,
  "maximum_search_calls": 100,
  "expires_at": "..."
}
```

The token names one run and one challenge. It cannot be replayed against a second
challenge, cannot outlive the episode, and carries no provider credential — it authorises
the RCG to spend on the laboratory's behalf, and authorises nothing else.

---

# 6. Submission sealing and disclosure

## 6.1 Before validation

The miner:

1. builds the immutable bundle;
2. materializes all required source and custom-model artifacts;
3. calculates content hashes;
4. encrypts the payload;
5. timelock-encrypts the symmetric decryption key;
6. commits the encrypted bundle and digest.

The miner cannot modify after deadline:

* source;
* prompts;
* model versions;
* model routing;
* dependencies;
* search behavior;
* ranking logic;
* endpoint declarations.

## 6.2 Validator-only reveal

At the validation reveal point:

* registered validators obtain the decryption material;
* validators pull and execute the exact committed artifacts;
* the general public does not receive the source until execution closes.

This prevents one miner’s laboratory from searching for and absorbing another current-round submission during execution.

## 6.3 Public publication

After the daily execution window closes:

* source becomes public;
* model manifests become public;
* prompts and orchestration become public;
* score reports become public;
* other miners may fork the design in the next submission cycle.

Credentials and billing secrets remain private.

---

# 7. Validator-generated daily challenge packs

## 7.1 Core rule

Every validator generates its own hidden challenge pack daily.

The validator must test all miners in its evaluation cohort on the **same challenge instances**.

A validator must never generate a separate problem for each miner.

## 7.2 Challenge taxonomy

The public protocol defines a versioned taxonomy such as:

```text
A. Software architecture
B. Algorithm invention
C. Agent architecture
D. Memory and retrieval
E. Distributed coordination
F. Digital mechanism design
G. Data and model pipelines
H. Optimization and efficiency
```

Each daily pack is stratified across the active taxonomy.

```text
20 challenges per validator per day

10 generated by GPT
10 generated by Claude

stratified across the taxonomy:
  3 software architecture
  3 algorithms
  4 AI-agent architecture
  3 distributed coordination
  3 memory and retrieval
  2 digital mechanism design
  2 data and model pipelines / optimization
```

## 7.2.1 Two generators, deliberately

The twenty problems are produced by **two independent generator families**: ten by GPT and
ten by Claude, both reached through the validator's own OpenRouter key. This is not
redundancy, and it is not a hedge against an outage.

A single generator has a *house style*. It reaches for the same problem shapes, the same
constraint patterns, the same framings — and a laboratory tuned to that style scores well
without being a better laboratory. With ten problems from each of two families, half the
pack is always foreign to whatever a miner has overfitted to, and the difference between a
miner's two halves is itself a measurement: a laboratory that scores far better on one
family's problems has learned a generator, not a domain.

That difference is recorded per miner and per day. A widening gap is the earliest available
signal that the challenge supply has become predictable, and it appears long before
scores stop discriminating.

## 7.2.2 Cross-family critique

A problem generated by GPT is reviewed by the **Claude** critic, and a problem generated by
Claude is reviewed by the **GPT** critic.

Same-family critique is close to self-review: a model shares its generator's blind spots and
tends to rate its own family's output as clear and well-formed. Crossing the families means
the reviewer does not share the writer's assumptions, which is the only version of this
check that can find an ambiguity the writer could not see.

## 7.3 Challenge-generation seed

```text
daily_seed =
SHA256(
  date
  || validator_hotkey
  || validator_precommitted_salt
  || post-deadline_block_hash
)
```

The daily pack must not be selectable after the validator sees miner responses.

## 7.4 Challenge-generation pipeline

```text
Domain sampler
       ↓
Problem-generator LLM
       ↓
Deterministic schema linter
       ↓
Independent critic LLM
       ↓
Safety and prohibited-domain filter
       ↓
Duplicate detector
       ↓
Difficulty/discrimination probe
       ↓
Final challenge-pack commitment
```

### Step 1: Candidate generation

For each of the twenty slots, the assigned generator family produces three to five
candidate problems. Slot assignment is fixed by the daily seed before generation begins, so
a validator cannot decide after the fact which family produced which surviving problem.

### Step 2: Deterministic linter

The linter rejects problems that lack:

* a clearly stated problem;
* a defined research objective;
* explicit constraints;
* expected output structure;
* sufficient technical context;
* a feasible research scope;
* a fixed budget;
* a meaningful need for invention rather than simple factual retrieval.

### Step 3: Critic review

The critic is drawn from the **opposite** family to the generator (section 7.2.2). It
checks:

* ambiguity;
* internal contradiction;
* triviality;
* answer leakage;
* dependence on unavailable private data;
* requirement for physical experiments;
* impossibility of evaluating relevance;
* excessive resemblance to recent challenges.

### Step 4: Duplicate detection

The validator compares the candidate against:

* the last 90 days of challenge packs;
* publicly revealed prior challenge packs;
* known benchmark questions;
* recent miner-proposed problem archives.

Both embeddings and structured mechanism fingerprints should be used.

### Step 5: Discrimination probe

Reference laboratories run on candidate problems.

The validator rejects a problem if:

* every reference produces essentially the same answer;
* the problem is solved by trivial web retrieval;
* the problem is so vague that judge results are unstable;
* all reference outputs fail to provide any mechanism;
* judge panels cannot distinguish intentionally degraded answers.

### Step 6: Commitment

Before miner execution begins, the validator publishes a signed commitment containing:

```json
{
  "validator_hotkey": "...",
  "date": "...",
  "challenge_pack_hash": "sha256:...",
  "generation_protocol_version": "CPG-1.0",
  "generator_model_snapshots": {
    "gpt": "fixed-season-snapshot",
    "claude": "fixed-season-snapshot"
  },
  "number_of_challenges": 20,
  "challenges_per_generator": { "gpt": 10, "claude": 10 },
  "signature": "..."
}
```

The commitment names the counts per generator but not which slot came from which family.
Publishing the split per slot before execution would tell a laboratory which half of the
pack to expect from whom, and the point of two families is that it cannot know.

The actual problems remain private until the evaluation closes.

## 7.5 Challenge storage

Generated packs live in **Redis**, which is the validator's own store.

Redis rather than a relational store because the access pattern is exactly what it is good
at: write a pack once, read it many times during execution, expire it after the dedup
window, and survive a restart mid-round without losing the pack a hash has already been
committed for.

Three things it holds:

| Key | Contents | Lifetime |
|---|---|---|
| `pack:{date}` | The day's twenty challenges, plus the committed hash | The dedup window |
| `dedup:{fingerprint}` | Mechanism fingerprints and embeddings of past challenges | `dedup_lookback_days` |
| `run:{run_id}` | Which challenge a given run was issued, for reconciliation | Until publication |

### Redis is not reachable from the sandbox

The laboratory never connects to Redis, and Redis is never exposed on the sandbox network.
The challenge reaches the laboratory the way section 9.1 specifies — as its structured
input, delivered by the runner — and the only outbound path from the container remains the
RCG.

This is worth stating plainly because "serve the problems to miners from Redis" reads as
though the miner should fetch them, and a laboratory that could reach Redis could read the
whole pack: every problem, including the ones it has not been given yet, and the packs of
other rounds. The store is the validator's; the delivery is the runner's.

### The pack hash is committed before the pack is stored

Writing to Redis is not the commitment. The signed `challenge_pack_hash` goes on chain
first, and only then is the pack persisted. A store that could be edited between generation
and commitment would make the commitment meaningless, and this ordering removes the window
entirely rather than trusting that nobody uses it.

---

# 8. Challenge object

```json
{
  "challenge_id": "sha256:...",
  "domain": "ai_agent_architecture",
  "title": "Resource-bounded persistent planning",
  "problem_statement": "...",
  "research_objective": "...",
  "current_baseline": "...",
  "known_attempts": ["...", "..."],
  "constraints": [
    "...",
    "..."
  ],
  "forbidden_shortcuts": [
    "..."
  ],
  "required_output": {
    "portfolio_size": 5,
    "ranked": true,
    "mechanism_required": true,
    "prior_art_comparison_required": true,
    "falsification_plan_required": true,
    "simulation_or_calculation_required": true
  },
  "resource_limits": {
    "maximum_wall_time_seconds": 1800,
    "maximum_rcc": 400,
    "maximum_search_calls": 100
  }
}
```

There is no simulated user and no conversational persona.

The challenge is a single structured research request.

---

# 9. Miner execution protocol

## 9.1 Standard input

```json
{
  "type": "research_challenge",
  "protocol_version": "AIL-3.0",
  "challenge": {},
  "runtime": {
    "run_id": "...",
    "deadline": "...",
    "rcg_endpoint": "...",
    "artifact_directory": "/output"
  }
}
```

## 9.2 Standard output

The laboratory returns a structured Top-5 portfolio.

```json
{
  "challenge_id": "...",
  "laboratory_summary": {
    "research_strategy": "...",
    "search_scope": "...",
    "major_assumptions": ["..."]
  },
  "portfolio": [
    {
      "rank": 1,
      "title": "...",
      "problem_reframe": "...",
      "core_invention": "...",
      "mechanism": {
        "components": ["..."],
        "information_flow": "...",
        "causal_explanation": "...",
        "feedback_loops": ["..."]
      },
      "nearest_prior_art": [
        {
          "source": "...",
          "similarity": "...",
          "material_difference": "..."
        }
      ],
      "why_non_obvious": "...",
      "expected_value": {
        "beneficiary": "...",
        "value_created": "...",
        "magnitude_hypothesis": "..."
      },
      "assumptions": ["..."],
      "weakest_assumption": "...",
      "failure_modes": ["..."],
      "falsifiable_predictions": ["..."],
      "cheapest_kill_test": "...",
      "simulation_or_calculation": {
        "method": "...",
        "result": "...",
        "artifact_refs": ["..."]
      },
      "development_path": ["..."],
      "estimated_probability_of_value": 0.42,
      "estimated_validation_cost_rcc": 25
    }
  ],
  "portfolio_map": {
    "idea_families": ["..."],
    "differences": ["..."]
  },
  "self_selection": {
    "why_rank_1": "...",
    "confidence": 0.71
  },
  "resource_usage_claim": {
    "rcc": 397,
    "search_calls": 84,
    "model_calls": 163
  }
}
```

Validators replace self-reported usage with RCG-measured usage.

Private chain-of-thought is not required.

---

# 10. Validator execution environment

Miner bundles run in validator-controlled containers.

## Required controls

* OCI digest verification;
* non-root execution;
* read-only base filesystem;
* temporary writable workspace;
* CPU, RAM, storage and PID limits;
* hard wall-clock timeout;
* no arbitrary internet;
* RCG-only outbound access;
* model and endpoint allowlist;
* output-size limits;
* full tool-call logging;
* signed request and response receipts;
* forced termination when the episode closes.

This follows the practical architecture used by production agent subnets: ORO validators claim work, download miner code and execute it in isolated Docker containers. ([ORO Subnet][2])

---

# 11. Validation pipeline

```text
Admission
   ↓
Execution
   ↓
Hard gates
   ↓
Canonicalization
   ↓
Citation and prior-art verification
   ↓
Cheap screening
   ↓
Pairwise tournament
   ↓
Top-miner replication
   ↓
Daily score
   ↓
Rolling score
   ↓
Weights
```

---

# 12. Stage 0: admission checks

Before expensive execution, the validator verifies:

* miner registration;
* signature;
* bundle digest;
* container availability;
* model manifest;
* model revisions;
* provider authorization;
* sufficient miner-funded balance;
* valid license;
* dependency lock;
* supported protocol version;
* absence of embedded secrets;
* absence of obvious malware;
* output-schema support.

Failure means the bundle is excluded before inference costs are incurred.

---

# 13. Stage 1: deterministic hard gates

The following conditions invalidate the challenge response.

1. Invalid output schema.
2. Missing required portfolio fields.
3. Undeclared model use.
4. Model-revision mismatch.
5. Unauthorized endpoint.
6. Budget exceeded.
7. Time limit exceeded.
8. Fabricated or inaccessible citation.
9. Judge-directed prompt injection.
10. Current-round submission copying.
11. Hidden human intervention.
12. Prohibited-domain content.
13. Validation-environment manipulation.

Hard-gate failure cannot be compensated for by high LLM scores.

---

# 14. Stage 2: answer canonicalization

Before semantic judging, the validator converts every answer into a neutral representation.

It removes:

* miner identity;
* model identity;
* laboratory branding;
* decorative markdown;
* unnecessary verbosity;
* self-congratulatory language;
* judge instructions;
* unverifiable numerical claims.

It independently reconstructs:

* citations;
* resource usage;
* prior-art candidates;
* semantic idea clusters;
* required constraints;
* portfolio duplication.

The judge receives a standardized fact sheet, not the miner’s persuasive original presentation.

---

# 15. Stage 3: prior-art and originality analysis

Each idea is searched against:

* academic papers;
* patents;
* public software repositories;
* existing products;
* technical standards;
* previous challenge answers;
* previously published miner bundles.

The validator produces a `PriorArtReport`:

```json
{
  "idea_id": "...",
  "nearest_matches": [
    {
      "source": "...",
      "similarity": 0.76,
      "shared_mechanism": "...",
      "claimed_difference": "...",
      "verified_difference": "..."
    }
  ],
  "novelty_confidence": 0.63,
  "renaming_only": false
}
```

The validator must never assert that an idea is absolutely unprecedented.

It evaluates whether the mechanism is materially different from **searchable prior art**.

---

# 16. LLM judge architecture

Anthropic recommends combining code-based, model-based and human graders according to what each can reliably assess; model graders are particularly suitable for open-ended outcomes, but they require calibration and clear rubrics. ([Anthropic][4]) OpenAI similarly treats LLM-based grading as an estimate rather than a replacement for strong gold-standard evaluation. ([OpenAI Evals][5])

This subnet uses model graders, but it narrows their responsibilities and continuously tests their reliability.

## 16.1 Judge families

At least three different model families are required, and they run on the **validator's own
OpenRouter key** — never on a miner's.

Composition, by OpenRouter route:

* **Claude judge** — `anthropic/...`;
* **GPT judge** — `openai/...`;
* a third family for the tie-breaking third opinion — `google/...`, or an open-weight route.

Claude and GPT are used **hybridly** and are the same two families that generate the
challenges (section 7.2.1), which produces a property worth naming: a problem written by
GPT is judged by a panel including Claude, and vice versa. No family both sets a problem
and unilaterally decides the answer.

No single provider family may control more than 40% of a semantic criterion. Two snapshots
of one family are one family — the cap is on the family, because two versions of the same
model share their failure modes.

**The family cap is on the model family, not on the aggregator.** Every judge is reached
through OpenRouter, so "provider" in the routing sense is always the same and would make the
cap vacuous if read that way. What matters is who *trained* the model behind the route: three
routes to three Anthropic snapshots is one family and violates the cap, however many
distinct slugs it uses. The cap is evaluated on the family field of the panel declaration,
which is why that field is required rather than derived from the slug.

## 16.2 Required judge roles

1. **Constraint Judge**
   Did the answer satisfy the actual research objective and constraints?

2. **Originality Judge**
   Is the mechanism materially different from prior art, rather than renamed or superficially recombined?

3. **Value Judge**
   Would the idea plausibly create meaningful scientific, technical or commercial value?

4. **Mechanism Judge**
   Is there a coherent causal or technical mechanism explaining how the idea works?

5. **Diversity Judge**
   Do the five ideas represent substantially different research directions?

6. **Self-Selection Judge**
   Did the laboratory rank its strongest idea first?

7. **Falsifiability Judge**
   Does the idea produce discriminating predictions or a meaningful kill test?

8. **Deepening Judge**
   Did the laboratory turn its best idea into a concrete research path, simulation or calculation?

## 16.3 JSON judge output

```json
{
  "criterion": "mechanism",
  "comparison": {
    "candidate_a": "anonymous-A",
    "candidate_b": "anonymous-B"
  },
  "winner": "A",
  "confidence": 0.82,
  "a_strengths": ["..."],
  "b_strengths": ["..."],
  "a_failures": [],
  "b_failures": ["unsupported causal step"],
  "decisive_reason": "...",
  "abstain": false
}
```

---

# 17. Scalable evaluation funnel

Full all-versus-all comparison is too expensive.

The validator uses a funnel.

## 17.1 Screening

Every valid miner receives a cheap anchored pointwise evaluation.

The judge scores each dimension from 0 to 4 using explicit anchors:

```text
0 — absent or invalid
1 — superficial
2 — plausible but incomplete
3 — strong and concrete
4 — unusually strong, coherent and differentiated
```

This screening score is not the final score. It identifies candidates for full judging.

## 17.2 Full evaluation cohort

Full evaluation includes:

* top screening miners;
* a fixed number of randomly selected miners;
* new miners;
* miners with high scoring uncertainty;
* miners with materially new bundle architectures.

This prevents incumbent lock-in.

## 17.3 Swiss pairwise tournament

Instead of comparing every pair:

* miners receive opponents near their current estimated score;
* each miner receives several different opponents;
* pairings are balanced by challenge;
* repeated identical pairings are limited;
* A/B presentation order is reversed.

## 17.4 Finalist round

The top 6–10 miners receive:

* additional pairwise comparisons;
* additional judge families;
* repeated execution;
* paraphrased versions of selected challenges;
* separate prior-art verification.

## 17.5 Replication

Top results and anomalies are independently rerun by another validator.

---

# 18. Scoring model

## 18.1 Per-idea rank weighting

The Top-5 ideas are weighted:

[
Q_{\text{portfolio}}
====================

0.40Q_1+
0.25Q_2+
0.15Q_3+
0.12Q_4+
0.08Q_5
]

Ideas that are merely semantic duplicates are collapsed into one lineage before scoring.

## 18.2 Semantic criterion weights

| Criterion                                  |   Weight |
| ------------------------------------------ | -------: |
| Structural originality and non-obviousness |      25% |
| Expected research or practical value       |      20% |
| Mechanistic plausibility                   |      15% |
| Constraint and problem fit                 |      12% |
| Portfolio direction diversity              |      10% |
| Self-selection and ranking quality         |       8% |
| Falsifiability and deepening quality       |       7% |
| Cost, latency and execution reliability    |       3% |
| **Total**                                  | **100%** |

## 18.3 Pairwise and pointwise combination

For each semantic criterion:

[
C_k =
0.75 \cdot BT_k+
0.25 \cdot AR_k
]

Where:

* (BT_k) is the normalized Bradley–Terry score from pairwise judgments;
* (AR_k) is the anchored pointwise rubric score.

Pairwise comparison is the primary signal. Pointwise scoring provides diagnostic anchoring and helps detect a generally weak field in which relative winners are still poor.

## 18.4 Challenge score

[
S_{m,c}
=======

\sum_k w_k C_{m,c,k}
]

A mechanism floor applies:

```text
If Mechanism < 0.40:
  Value score is capped at 0.50
  Originality score is capped at 0.50
```

An idea cannot score highly merely by sounding unusual.

## 18.5 Daily validator score

A laboratory must perform consistently across several problems.

[
D_m =
0.70\cdot\operatorname{Mean}(S_{m,c})
+
0.30\cdot Q_{25}(S_{m,c})
]

The lower-quartile component penalizes laboratories that perform brilliantly on one problem but fail on most others.

## 18.6 Rolling score

```text
If miner has 1–6 valid daily results:
    rolling = mean(all valid daily results)

If miner has at least 7:
    rolling = 0.60 × median(last 7 days)
            + 0.40 × median(last 30 days)
```

There is no credibility multiplier that suppresses new miners.

---

# 19. Judge calibration

Each validator inserts hidden control answers into the judge stream.

Controls include:

* an original answer;
* the same answer with the mechanism removed;
* the same answer with a constraint violated;
* a copied known idea with renamed terminology;
* a stylistically impressive but technically empty answer;
* five near-duplicate ideas;
* an answer with fabricated citations;
* an answer with a valid mechanism but modest writing quality.

The judge must reliably rank the non-degraded version higher.

## Judge eligibility requirements

* degradation accuracy ≥95%;
* order-swap inconsistency below a configured maximum;
* JSON-schema validity ≥99%;
* no repeated bias toward a model family;
* minimum confidence calibration;
* circuit-breaker on repeated provider failures.

A judge falling below requirements is removed from the panel.

---

# 20. Weight allocation

Each validator independently converts its rolling miner scores into weights.

## 20.1 Qualification floor

A miner qualifies only if:

* all required hard gates pass;
* it has at least the minimum number of valid evaluated challenges;
* its rolling score exceeds the reference-lab floor;
* its bundle and model artifacts remain available.

Being best among weak miners is not enough.

## 20.2 Softmax allocation

For qualified miners:

[
p_i =
\frac{\exp((S_i-S_{\min})/\tau)}
{\sum_j\exp((S_j-S_{\min})/\tau)}
]

Recommended starting temperature:

```text
τ = 0.08–0.12
```

## 20.3 Maximum concentration

A per-miner weight cap should initially be set around 15–20%.

Overflow is redistributed among other qualified miners.

This encourages multiple research-lab architectures rather than one permanent winner.

## 20.4 No qualified miners

If no miner beats the reference floor:

```text
100% weight → burn UID
```

## 20.5 On-chain publication

Validators independently submit their weights using Bittensor’s supported weight path and commit-reveal configuration. Current SDK documentation exposes timelocked weight commitment with `netuid`, `mechid`, UIDs, weights and a version key. ([Bittensor Documentation][6])

Yuma Consensus then aggregates validator weight vectors. Current YC3 retains the existing subnet scoring interface while improving validator bonding and fairness behavior. ([Bittensor Documentation][7])

---

# 21. Daily operational cycle

A recommended 24-hour cycle:

```text
T−2h   Miner bundle submission closes
T−90m  Validator salts committed
T−60m  Post-deadline block randomness fixed
T−50m  Validators generate candidates — 10 GPT slots, 10 Claude slots, in parallel
T−30m  Cross-family critique, linting, dedup and discrimination probes complete
T−20m  Challenge pack hash committed on chain, then the pack is written to Redis
T0     Validator-only bundle reveal; credential envelopes decrypted by the RCG
T0–6h  Admission + screening execution across all 20 challenges
T6–14h Full execution
T14–18h Canonicalization + prior-art checks
T18–21h Pairwise judge tournament
T21–22h Replication and anomaly audit
T22–23h Score aggregation
T23h   Validator weights committed
T24h   Public source, challenges and reports published; credential envelopes are not
```

Exact timings should be block-aligned rather than dependent only on wall-clock time. A
validator whose clock runs fast would otherwise commit its pack against different randomness
than its peers, and the commitment would be to a pack nobody else could have produced.

## 21.1 What twenty challenges cost

Twenty problems rather than eight is a 2.5x increase in the work per miner per day, and the
funnel (section 17) is what keeps that affordable rather than the challenge count.

Two consequences follow and should be planned for:

* **Screening carries more of the load.** With twenty problems, the cheap anchored pointwise
  pass decides most of the field, and only the full-evaluation cohort sees all twenty under
  pairwise judging. Screening quality is therefore load-bearing in a way it was not at eight.
* **Miner cost scales with the pack.** A miner funds its own inference, so twenty problems
  is 2.5x the miner's daily spend. The per-challenge `maximum_rcc` should be set with the
  daily total in mind, since the published ceiling a miner plans against is
  `challenges × maximum_rcc`, not `maximum_rcc`.

---

# 22. Public disclosure after evaluation

The following should be published after execution closes:

* challenge pack;
* generation protocol version;
* miner bundle source;
* model manifest;
* canonicalized portfolios;
* raw portfolios;
* prior-art reports;
* judge JSON;
* A/B order results;
* challenge scores;
* rolling scores;
* RCG receipts;
* hard-gate outcomes;
* validator weight vector;
* bundle lineage and forks.

The detailed unpublished judge prompt may remain sealed until the end of a judge epoch to reduce immediate prompt overfitting.

---

# 23. Security model

| Threat                                | Primary control                                                                   |
| ------------------------------------- | --------------------------------------------------------------------------------- |
| Human assistance during validation    | Validator-executed immutable bundle                                               |
| Miner changes models after submission | Locked model manifest and revision                                                |
| Unlimited capital brute force         | Equal RCC hard cap                                                                |
| Direct internet covert channel        | RCG-only egress                                                                   |
| Current-round code copying            | Delayed public release until execution closes                                     |
| Judge prompt injection                | Canonicalization and untrusted-content isolation                                  |
| Fabricated citations                  | Independent citation retrieval                                                    |
| Existing idea renamed                 | Prior-art and mechanism comparison                                                |
| Verbose style wins                    | Format stripping and fixed schemas                                                |
| One provider controls judging         | Multi-family panel and family cap                                                 |
| Judge order bias                      | A/B and B/A execution                                                             |
| Challenge tailored to miner           | Challenge seed fixed after bundle lock                                            |
| Validator creates trivial problems    | Linter, critic, dedup and reference discrimination                                |
| Validator copies weights              | Bittensor commit-reveal                                                           |
| One lucky result                      | Multi-challenge evaluation and replication                                        |
| New miners never evaluated            | Mandatory exploration slots                                                       |
| Owner self-mining advantage           | Owner does not generate daily tasks; owner-linked miner UIDs should be ineligible |
| **Validator abuses a miner's credential** | Key held by the RCG and never by the sandbox; every call receipted and hash-chained; receipt totals reconciled against provider-reported usage; miners provision a spend-capped per-round key |
| **Validator work billed to a miner's key** | One provider means a swapped key *succeeds* and silently bills the wrong party. Two typed resolvers with no owner parameter; purpose selects the credential and a mismatch raises before the request is built; per-account totals reconciled (section 3.4.4) |
| **Credential leaked by publication**  | Credential travels in a separate sealed envelope the publication path never reads — excluded by construction, not by a filter |
| **Laboratory exfiltrates its own key**| The laboratory never receives it. It holds a run-scoped session token carrying no credential |
| **Session token replayed on another challenge** | Token bound to one `run_id` and one `challenge_id`, with a short expiry |
| **Laboratory reads the whole challenge pack** | Redis is the validator's store and is unreachable from the sandbox; the challenge arrives as the runner-delivered input |
| **Pack edited between generation and commitment** | Hash committed on chain *before* the pack is written to Redis |
| **Miner overfits to one generator's style** | Half the pack from each of two families; per-miner per-family score gap tracked as a drift signal |
| **A generator family rates its own output as sound** | Cross-family critique: GPT-written problems are reviewed by Claude and vice versa |

---

# 24. Repository architecture

```text
ail-subnet/
├── protocol/
│   ├── schemas/
│   │   ├── season_config.json
│   │   ├── bundle_manifest.json
│   │   ├── model_manifest.json
│   │   ├── challenge.json
│   │   ├── portfolio.json
│   │   ├── judge_result.json
│   │   └── execution_receipt.json
│   ├── models.py
│   ├── crypto.py
│   ├── seeds.py
│   ├── receipts.py
│   ├── scoring.py
│   └── weights.py
│
├── registry/
│   ├── season_registry.py
│   ├── sealed_bundle_registry.py
│   └── artifact_escrow.py
│
├── gateway/
│   ├── api.py
│   ├── tokens.py
│   ├── metering.py
│   ├── receipts.py
│   ├── credentials.py          # two typed resolvers; purpose selects, never a parameter
│   └── adapters/
│       ├── openrouter.py       # the one provider surface, both sides
│       ├── search.py           # OpenRouter web search
│       └── simulation.py
│
├── validator/
│   ├── neuron.py
│   ├── challenge_factory/
│   │   ├── taxonomy.py
│   │   ├── generator.py        # 10 GPT slots + 10 Claude slots
│   │   ├── linter.py
│   │   ├── critic.py           # cross-family: GPT reviews Claude, Claude reviews GPT
│   │   ├── dedup.py
│   │   ├── discriminator.py
│   │   └── store.py            # Redis: packs, dedup fingerprints, run bindings
│   ├── judges/
│   │   └── panel_client.py     # OpenRouter, validator's own key
│   ├── scheduler/
│   ├── sandbox/
│   ├── canonicalizer/
│   ├── prior_art/
│   ├── judge/
│   │   ├── panels.py
│   │   ├── pairwise.py
│   │   ├── pointwise.py
│   │   ├── calibration.py
│   │   └── bradley_terry.py
│   ├── scoring/
│   │   ├── gates.py
│   │   ├── daily.py
│   │   ├── rolling.py
│   │   └── normalization.py
│   └── weights.py
│
├── miner/
│   ├── cli/
│   ├── sdk/
│   └── reference/
│       ├── single_agent/
│       ├── multi_agent/
│       ├── idea_islands/
│       └── evolutionary_lab/
│
├── portal/
├── ops/
└── tests/
    ├── unit/
    ├── integration/
    ├── localnet/
    ├── adversarial/
    └── measurement/
```

The `sim_user/` component is removed entirely from the V1 validator architecture.

---

# 25. Reference miner laboratories

The owner should ship four open reference bundles.

## Reference A — Frontier Single Agent

One strong model receives the challenge and produces the portfolio using a carefully designed system prompt.

Purpose:

* represents ordinary direct Claude/GPT research use;
* defines the minimum baseline miners must beat.

## Reference B — Planner–Researcher–Critic

A small multi-agent system:

```text
Planner
→ researchers
→ prior-art critic
→ portfolio selector
```

## Reference C — Independent Idea Islands

Five isolated idea-generation branches using different perspectives. They do not share outputs until final synthesis.

## Reference D — Evolutionary Lab

```text
generate
→ evaluate
→ mutate
→ cross
→ archive
→ select
```

The protocol does not require miners to copy these structures.

They are baselines and starter templates.

---

# 26. Feasibility assessment

This architecture is professionally implementable with existing components.

* Bittensor supplies miner/validator registration, metagraph state, weight setting, commit-reveal and Yuma aggregation. ([Bittensor Documentation][8])
* The official subnet template separates protocol, miner and validator responsibilities, although this subnet requires additional production services around that minimal structure. ([GitHub][9])
* ORO already demonstrates executable Python-agent submissions, validator work claiming, Docker sandbox execution and score production. ([ORO Subnet][2])
* Harnyx demonstrates the closer domain analogue: miners submit deep-research scripts, validators execute them under budgets, and LLM judges compare research answers. Its presence confirms technical feasibility, but this subnet differs by optimizing for invention portfolios rather than conventional researched answers. ([Bittensor.ai][10])
* OpenRouter fronts every major model family behind one OpenAI-compatible API and serves web search on the same credential, so a single adapter covers the entire subnet -- miner research, challenge generation and judging -- and one metering format covers every model either side might choose. The cost of that simplification is that credential separation must be enforced in code rather than by incompatible APIs (section 3.4.4).
* Anthropic’s current evaluation guidance explicitly supports combining deterministic graders and model-based graders for autonomous agents rather than relying on one evaluation type. ([Anthropic][4])

The primary unresolved risk is **measurement validity**, not basic software feasibility.

---

# 27. Testnet measurement gates

The subnet must not move to mainnet merely because the code works.

## Required evaluator gates

| Measurement                               | Minimum target |
| ----------------------------------------- | -------------: |
| Hard-gate deterministic agreement         |           100% |
| Judge JSON-schema validity                |           ≥99% |
| Degraded-control detection                |           ≥95% |
| Style-only ranking reversal               |            ≤5% |
| A/B order inconsistency                   |          ≤7.5% |
| Same-bundle rerun rank correlation        |          ≥0.80 |
| Independent judge-family agreement        |          ≥0.70 |
| Cross-validator rank correlation          |          ≥0.60 |
| Citation verification accuracy            |           ≥98% |
| Challenge duplicate rate                  |            ≤2% |
| Challenge discrimination rate             |           ≥80% |
| Execution-receipt reconciliation          |           100% |
| Unauthorized egress incidents             |              0 |
| Undeclared model use                      |              0 |
| Maximum miner cost predictable in advance |           100% |

## Capability gate

At least one external miner must repeatedly outperform Reference A—the direct frontier-model baseline—across multiple unseen challenge days.

Otherwise the subnet has not demonstrated that competing laboratory architectures add value beyond ordinary model usage.

---

# 28. Implementation roadmap

## M0 — Protocol foundation, 4–6 weeks

Build:

* protocol schemas;
* bundle SDK;
* sealing and artifact escrow;
* RCG;
* sandbox;
* reference miner A;
* one validator;
* one challenge generator;
* basic pointwise JSON judge;
* local Bittensor test network integration.

## M1 — Complete validator, 6–8 weeks

Add:

* reference miners B–D;
* daily challenge factory;
* challenge critic and deduplication;
* prior-art subsystem;
* pairwise judge panels;
* Bradley–Terry fitting;
* calibration controls;
* rolling scoring;
* weight conversion;
* public audit portal.

## M2 — Private testnet, 4–6 weeks

Measure:

* cost;
* judge agreement;
* rerun stability;
* style bias;
* challenge quality;
* security attacks;
* miner-funded inference;
* weight independence.

## M3 — Public testnet, 6–10 weeks

Invite external miners.

Require at least two consecutive weeks satisfying all measurement gates.

## M4 — Mainnet

Mainnet begins only after evaluator validity is demonstrated.

---

# Final locked workflow

```text
1. Owner publishes protocol, domain taxonomy and scoring rules.
2. Miners develop complete autonomous invention-lab bundles.
3. Miners lock and seal their bundle, model manifest and billing delegation.
4. Each validator derives an unpredictable daily seed.
5. Each validator generates 20 structured problems daily -- 10 with GPT, 10 with Claude,
   through OpenRouter on its own key.
6. Cross-family critics, linters, deduplication and reference-lab discrimination probes
   reject bad problems.
7. The validator commits the challenge-pack hash on chain, then stores the pack in Redis.
8. Validators decrypt submitted bundles privately, and the gateway decrypts each miner's
   sealed OpenRouter credential envelope.
9. The same hidden challenge instances are run against all miners in that cohort.
10. Miner laboratories pay for their own models, search and simulation through the RCG,
    which spends the miner's own OpenRouter credential and never exposes it to the sandbox.
11. Validators enforce budgets, record receipts and collect Top-5 portfolios.
12. Invalid outputs fail deterministic hard gates.
13. Valid outputs are anonymized, canonicalized and independently checked against prior art.
14. Cheap pointwise judging screens the field.
15. A multi-model, order-swapped pairwise tournament ranks the strongest laboratories.
16. Top and anomalous results are replicated.
17. Each validator calculates daily and rolling miner scores independently.
18. Qualified miners receive capped softmax-distributed weights.
19. Validators submit weights using Bittensor commit-reveal.
20. After execution closes, bundles, challenges, judge reports and scores are published.
21. Miners fork the strongest public techniques and submit improved laboratories next cycle.
```

## Final judgment

This updated architecture is substantially simpler than the simulated-user design while preserving the actual subnet objective.

It is:

* fully automated;
* decentralized at the validator level;
* compatible with Bittensor’s native scoring and weight model;
* economically scalable through miner-funded inference, on the miner's own credential;
* open to custom or commercial foundation models through one provider surface;
* resistant to direct answer memorization through daily generated challenges;
* capable of evolving research-laboratory architectures through public inheritance.

The essential limitation must remain explicit:

> **The subnet can rigorously measure agreement among calibrated automated evaluators, prior-art differentiation, technical coherence, diversity and consistency—but it cannot mathematically prove that an idea will become a historic breakthrough.**

The correct mainnet criterion is therefore not “the validator code runs.” It is:

> **The automatic validator reliably ranks deliberately strong, weak, copied, impossible and superficially novel portfolios in the correct order, and competing miner laboratories repeatedly outperform direct use of current frontier models on unseen daily research problems.**

[1]: https://docs.learnbittensor.org/concepts/weight-copying-in-bittensor?utm_source=chatgpt.com "The Weight Copying Problem | Bittensor"
[2]: https://docs.oroagents.com/docs/architecture?utm_source=chatgpt.com "Architecture — ORO Docs"
[3]: https://openrouter.ai/docs "OpenRouter Docs — unified API across model providers, with web search"
[4]: https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents?utm_source=chatgpt.com "Demystifying evals for AI agents \ Anthropic"
[5]: https://evals.openai.com/gdpval/grading?utm_source=chatgpt.com "OpenAI Evals"
[6]: https://docs.learnbittensor.org/python-api/html/autoapi/bittensor/core/extrinsics/asyncex/weights/?utm_source=chatgpt.com "bittensor.core.extrinsics.asyncex.weights — Bittensor SDK Docs documentation"
[7]: https://docs.learnbittensor.org/learn/yuma3-migration-guide?utm_source=chatgpt.com "Yuma Consensus 3 (YC3) Migration Guide | Bittensor"
[8]: https://docs.learnbittensor.org/concepts/consensus-based-weights?utm_source=chatgpt.com "Consensus-based Weights/Liquid alpha | Bittensor"
[9]: https://github.com/RaoFoundation/bittensor-subnet-template?utm_source=chatgpt.com "GitHub - RaoFoundation/bittensor-subnet-template: Template Design for a Bittensor subnetwork · GitHub"
[10]: https://bittensor.ai/subnets/67?utm_source=chatgpt.com "Deep Research Harness — SN67 | Bittensor.ai (SN67) — bittensor.ai | Bittensor.ai"
