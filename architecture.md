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

The existing foundational principles remain unchanged: miners submit the laboratory itself, miners fund its inference and research costs, validators control execution, model versions are locked, and validation remains automated. 

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

* miner-funded model access;
* scoped, temporary credentials;
* model-version enforcement;
* query and response hashing;
* token and cost metering;
* search-provider access;
* budget enforcement;
* signed usage receipts;
* endpoint allowlisting.

This follows a production-proven pattern: ORO executes miner-submitted agents in isolated sandboxes and routes search and inference through controlled services rather than permitting direct unrestricted internet access. ([ORO Subnet][2])

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
      "provider": "chutes",
      "endpoint_id": "chute-...",
      "hf_repo": "miner/research-model",
      "revision": "40-character-commit-sha",
      "parameters": {
        "temperature": 0.9,
        "max_tokens": 10000
      },
      "role": "idea_generation"
    },
    {
      "alias": "final_critic",
      "provider": "anthropic",
      "model_snapshot": "fixed-season-snapshot",
      "parameters": {
        "temperature": 0.2
      },
      "role": "critique"
    }
  ],
  "routing_config_hash": "sha256:...",
  "maximum_parallel_calls": 16
}
```

Model choice is open.

A miner may use:

* OpenAI;
* Anthropic;
* Gemini;
* Chutes-hosted models;
* private fine-tuned models;
* open-weight models;
* several models in one laboratory;
* one specialized custom model.

For miner-owned models, Chutes can deploy a Hugging Face model behind an OpenAI-compatible vLLM API while locking deployment to a specific Hugging Face revision. ([Chutes][3])

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

Raw provider credentials are never included in the public bundle.

The RCG issues a short-lived token bound to:

```json
{
  "miner_hotkey": "5F...",
  "bundle_digest": "sha256:...",
  "validator_hotkey": "5G...",
  "challenge_pack_hash": "sha256:...",
  "allowed_endpoints": ["..."],
  "maximum_rcc": 400,
  "maximum_requests": 500,
  "expires_at": "..."
}
```

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

Example:

```text
8 challenges per validator per day

2 software/algorithm
2 AI-agent architecture
1 distributed system
1 retrieval/memory
1 mechanism design
1 wildcard from the active domain pool
```

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

The generator produces three to five candidate problems for each required challenge slot.

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

An independent critic model checks:

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
  "generator_model_snapshots": ["..."],
  "number_of_challenges": 8,
  "signature": "..."
}
```

The actual problems remain private until the evaluation closes.

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

At least three different model families are required.

Example composition:

* Anthropic judge;
* OpenAI judge;
* Gemini judge;
* optional Chutes-hosted open-weight judge.

No single provider family may control more than 40% of a semantic criterion.

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
T−50m  Validators generate candidate challenges
T−30m  Challenge quality gates complete
T−20m  Challenge pack hash committed
T0     Validator-only bundle reveal
T0–6h  Admission + screening execution
T6–14h Full execution
T14–18h Canonicalization + prior-art checks
T18–21h Pairwise judge tournament
T21–22h Replication and anomaly audit
T22–23h Score aggregation
T23h   Validator weights committed
T24h   Public source, challenges and reports published
```

Exact timings should be block-aligned rather than dependent only on wall-clock time.

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
│   └── adapters/
│       ├── openai.py
│       ├── anthropic.py
│       ├── gemini.py
│       ├── chutes.py
│       ├── search.py
│       └── simulation.py
│
├── validator/
│   ├── neuron.py
│   ├── challenge_factory/
│   │   ├── taxonomy.py
│   │   ├── generator.py
│   │   ├── linter.py
│   │   ├── critic.py
│   │   ├── dedup.py
│   │   └── discriminator.py
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
* Chutes supports OpenAI-compatible serving and pinned Hugging Face model revisions, making miner-owned model deployment technically straightforward. ([Chutes][3])
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
5. Each validator uses LLMs to generate a structured challenge pack.
6. Automated linters, critics, deduplication and reference labs reject bad problems.
7. The validator commits the final challenge-pack hash.
8. Validators decrypt submitted bundles privately.
9. The same hidden challenge instances are run against all miners in that cohort.
10. Miner laboratories pay for their own models, search and simulation through RCG.
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
* economically scalable through miner-funded inference;
* open to custom or commercial foundation models;
* resistant to direct answer memorization through daily generated challenges;
* capable of evolving research-laboratory architectures through public inheritance.

The essential limitation must remain explicit:

> **The subnet can rigorously measure agreement among calibrated automated evaluators, prior-art differentiation, technical coherence, diversity and consistency—but it cannot mathematically prove that an idea will become a historic breakthrough.**

The correct mainnet criterion is therefore not “the validator code runs.” It is:

> **The automatic validator reliably ranks deliberately strong, weak, copied, impossible and superficially novel portfolios in the correct order, and competing miner laboratories repeatedly outperform direct use of current frontier models on unseen daily research problems.**

[1]: https://docs.learnbittensor.org/concepts/weight-copying-in-bittensor?utm_source=chatgpt.com "The Weight Copying Problem | Bittensor"
[2]: https://docs.oroagents.com/docs/architecture?utm_source=chatgpt.com "Architecture — ORO Docs"
[3]: https://chutes.ai/docs/templates/vllm?utm_source=chatgpt.com "VLLM Template - Docs - Chutes"
[4]: https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents?utm_source=chatgpt.com "Demystifying evals for AI agents \ Anthropic"
[5]: https://evals.openai.com/gdpval/grading?utm_source=chatgpt.com "OpenAI Evals"
[6]: https://docs.learnbittensor.org/python-api/html/autoapi/bittensor/core/extrinsics/asyncex/weights/?utm_source=chatgpt.com "bittensor.core.extrinsics.asyncex.weights — Bittensor SDK Docs documentation"
[7]: https://docs.learnbittensor.org/learn/yuma3-migration-guide?utm_source=chatgpt.com "Yuma Consensus 3 (YC3) Migration Guide | Bittensor"
[8]: https://docs.learnbittensor.org/concepts/consensus-based-weights?utm_source=chatgpt.com "Consensus-based Weights/Liquid alpha | Bittensor"
[9]: https://github.com/RaoFoundation/bittensor-subnet-template?utm_source=chatgpt.com "GitHub - RaoFoundation/bittensor-subnet-template: Template Design for a Bittensor subnetwork · GitHub"
[10]: https://bittensor.ai/subnets/67?utm_source=chatgpt.com "Deep Research Harness — SN67 | Bittensor.ai (SN67) — bittensor.ai | Bittensor.ai"
