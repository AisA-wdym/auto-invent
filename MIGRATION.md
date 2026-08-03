# Migration to v3.0

`architecture.md` v3.0 replaces a design built around a **simulated-user conversation**
with one built around **validator-generated structured research problems**. This file
records what carries over from that earlier work, what changes, and what is deleted —
so that nothing is ported by habit and nothing is rebuilt that already works.

The prior implementation lives at `/root/dev/auto/ail/`. It reached six green gates and
1,956 tests, and roughly 60% of it is orthogonal to the change: sealing, metering,
sandboxing, chain I/O and the audit trail do not care where a problem came from.

---

## The change in one line

> The problem used to be a **conversation with a hidden persona**, authored by the owner.
> It is now a **structured single-shot research request**, generated daily by each
> validator and committed by hash before any miner runs.

That removes the owner from the content path entirely — which was the single largest
open risk in the previous design, because it required expert-authored scenarios every
season and made the owner a trusted party.

---

## Ports essentially unchanged

These are correct under v3.0 for the same reasons they were correct before. Each arrives
with its tests.

| Area | Why it is unaffected |
|---|---|
| `protocol/canonical.py`, `fixedpoint.py` | Deterministic encoding and the no-float rule. Independent of problem source. |
| `protocol/seeds.py` | Commit-then-reveal seed derivation. v3.0 §7.3 uses the same construction with a per-validator salt. |
| `protocol/receipts.py`, `sealing.py`, `chain_io.py` | Receipt chains, timelock sealing, weight extrinsics. |
| `protocol/rcc.py` | Credit metering and the price table. |
| `gateway/**` (the RCG) | v3.0 §5.4 and §3.4 describe the gateway this already is: scoped tokens, model-version enforcement, metering, signed receipts, endpoint allowlist. |
| `runtime/**` | Container launcher, HTTPS transports with a fixed destination table, key store, durable store, chain and beacon clients. |
| `registry/**` | Sealed bundle custody, validator-only reveal, publication gate. §6 is unchanged. |
| `validator/sandbox/container.py`, `egress.py` | §10's required controls, already enforced as data and checked. |
| `validator/canonicalizer` ← `judge/canonicalize.py` | §14 is the same stage: strip identity, strip injections, bound length. |
| `validator/signer/**` | §20.5 wallet isolation. |
| `miner/bundle.py`, `secrets.py`, `compliance.py` | Manifest verification, secret scanning, smoke checks. |
| `tools/**` — the five gates | **The most important thing to bring.** Every real defect in the prior build was caught by these or by executing the code, never by review. |

## Changes materially

| Area | v2 | v3.0 |
|---|---|---|
| `validator/sandbox/runner.py` | Multi-turn `lab-io/1` conversation, turn limits, idle timeouts | One request in, one portfolio out (§9). Substantially simpler; the transcript machinery stays for evidence. |
| `validator/judge/panel.py` | Pointwise only, reliability-weighted median | Keeps the family cap, order swap and circuit breaker. **Adds** pairwise + Bradley–Terry, combined `0.75·BT + 0.25·anchored` (§18.3). |
| `validator/scoring/` | Per-scenario percentile, EMA blend, credibility multiplier | Daily `0.70·mean + 0.30·Q25` (§18.5), rolling median 7d/30d (§18.6). **The credibility multiplier is removed** — §18.6 states explicitly that nothing may suppress new miners. |
| `validator/weights.py` | Time-decay proportional / winner-take-all | Softmax at τ=0.08–0.12, qualification floor against the reference labs, 15–20% cap, 100% burn when nobody qualifies (§20). |
| `validator/state/projection.py` | Ten season states S0–S9 | The §21 **daily** cycle. Forward-only projection and the signed history stay; the state set changes. |
| `validator/timeline.py` | Season deadlines → block heights | Same conversion, daily cadence. §21: "block-aligned rather than dependent only on wall-clock time" — which is the D-025 finding, now in the spec. |
| `protocol/season.py` | Season config as the only rules artifact | Still the rules artifact, but domains/taxonomy/rubric replace the scenario corpus. |

## Deleted

| | Why |
|---|---|
| `validator/sim_user/**` | §24: "The `sim_user/` component is removed entirely from the V1 validator architecture." |
| `validator/challenge/dsl.py` — the persona corpus | Replaced by generated challenges. This removes the recurring expert-authoring cost. |
| `protocol/labio.py`'s conversation half | No turns, no user messages. A single structured request/response replaces it. |
| The credibility multiplier | §18.6, explicitly. |

## New, and this is where the risk now lives

| | |
|---|---|
| `validator/challenge_factory/` | §7.4's seven-stage pipeline: generate → lint → critic → safety → dedup → **discrimination probe** → commit. The discrimination probe is the load-bearing one: it runs the reference labs on a candidate and rejects the problem if they all answer alike, if trivial retrieval solves it, or if judges cannot separate a deliberately degraded answer. That is what stops a validator producing worthless problems. |
| `validator/prior_art/` | §15. Retrieval against papers, patents, repositories, products, standards, and prior challenge answers. §15's constraint matters: *never assert absolute novelty* — only difference from **searchable** prior art. |
| `validator/judge/{pairwise,bradley_terry,pointwise,calibration}.py` | §16–§17, §19. |
| `protocol/schemas/{challenge,portfolio,model_manifest}.json` | §8, §9.2, §5.3. |
| `miner/reference/template.py` | §25, as amended by the owner: **one** miner template rather than four reference architectures. It is the **qualification floor** (§20.1), so it is not a demo — it is part of the reward mechanism, and it has to be genuinely good or the floor is easy. The discrimination probe (§7.4 step 5) runs it in several declared configurations, because condition 1 measures spread and one configuration has none. |
| `tests/measurement/` | §27's fifteen gates. These decide mainnet, so they are tests, not a report. |

---

## What is carried forward from the prior build's mistakes

Recorded because they cost real time and the same shapes will recur here.

1. **A test that asserts a shape rather than a substance is worse than no test.** Four were
   found in the prior build. The last one — "a full round completes" — would have passed
   with every real component deleted.
2. **Reachability is not correctness.** A gate proving every enforcement point is on a call
   path did not prove the calls *succeed*: the composed gateway refused every request while
   that gate was green. Both entry points now need a `--check` that composes and executes.
3. **Before writing an adapter for an interface, grep for existing implementers.** One
   module here was written twice because I did not.
4. **Two sources of truth for one value is always a defect**, even when both are correct
   today. Three separate findings reduced to this.
5. **A refusal must name the input it lacks.** Silent no-ops produce signed evidence for
   work nobody did.
