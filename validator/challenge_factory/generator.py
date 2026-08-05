"""Step 1 of 7.4: candidate generation. Ten slots to GPT, ten to Claude.

Each slot's owning family produces `candidates_per_slot` candidates (3–5 per 7.4 step 1), and the
pipeline keeps the first that survives the linter, the critic and the dedup check. Several
candidates per slot rather than one because the downstream filters reject a substantial fraction,
and a slot with one candidate that fails leaves a hole in a pack whose size is committed.

## The prompt states what will reject the answer

The generator is told the linter's eight requirements explicitly. That is not a courtesy: a
generator that does not know `forbidden_shortcuts` is required omits it about half the time, and
every omission costs a full candidate. Telling it up front converts token spend into accepted
candidates.

It is also told what it must *not* produce — the excluded domains of 2 — for the same reason and
one more: a candidate that reaches the safety filter has already been paid for.

## Slot assignment is an input, never a decision

`generate` takes its slots. It cannot choose which family writes what, because 7.4 step 1 fixes
that from the daily seed before generation begins — otherwise a validator could generate both
halves of the pack, keep whichever half suited a submission it had seen, and attribute the
survivors freely.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from protocol.receipts import Purpose
from validator.challenge_factory.taxonomy import EXCLUDED_DOMAINS, Slot
from validator.model_client import ModelClient, ModelReply

__all__ = ["Candidate", "GeneratorConfig", "generate_for_slot"]

_log = logging.getLogger(__name__)

_SYSTEM = """\
You design research problems for an autonomous invention benchmark. Competing laboratories \
receive one problem and return a ranked portfolio of five inventions. Your problem decides what \
the comparison between those laboratories measures.

A good problem here is one where a strong laboratory and a weak one produce visibly different \
answers. That rules out two failure modes equally: a problem so easy that every laboratory \
answers it the same way measures nothing, and a problem so vague that judges disagree at random \
measures noise.

Return one JSON object and nothing else.
"""

_USER = """\
Design one research problem.

Domain: {domain}
Slot: {index} of {total}

## Required JSON shape

{{
  "title": "under 100 characters",
  "domain": "{domain}",
  "problem_statement": "at least 200 characters: the situation, why it is hard, what is at \
stake, and what a solution would change",
  "research_objective": "at least 60 characters, and not a restatement of the problem: what the \
laboratory is being asked to produce",
  "current_baseline": "what practitioners do today, and where it falls short",
  "known_attempts": ["approaches already tried, and why each is insufficient"],
  "constraints": ["at least two; at least one must be checkable — a number, a bound, or a \
prohibition"],
  "forbidden_shortcuts": ["at least one: the obvious non-answer that must not count"],
  "required_output": {{
    "portfolio_size": 5,
    "ranked": true,
    "mechanism_required": true,
    "prior_art_comparison_required": true,
    "falsification_plan_required": true,
    "simulation_or_calculation_required": true
  }},
  "resource_limits": {{
    "maximum_wall_time_seconds": {wall_time},
    "maximum_rcc": {rcc},
    "maximum_search_calls": {search_calls}
  }}
}}

## What will reject your problem

- No checkable constraint. "Should be efficient" cannot be scored for fit; "must answer within \
200ms at the 99th percentile" can.
- No forbidden_shortcuts. Without them, restating the baseline is a valid submission.
- Answerable by looking something up. This benchmark rewards invention; a problem a search \
engine answers rewards search.
- Requiring a physical measurement, a private dataset, or an outcome that takes months to \
observe. None of those can be evaluated here.
- Any of these domains, which are excluded from scoring: {excluded}.
- A problem where every competent laboratory would give essentially the same answer. If you can \
name the answer while writing the problem, the problem is too easy.

## What makes this slot different

{variation}
"""

#: One nudge per candidate within a slot, so the three-to-five candidates differ by more than
#: sampling temperature. Without this a generator asked the same question five times returns five
#: paraphrases of one problem, and the slot effectively had one candidate.
_VARIATIONS = (
    "Pick a problem where the hard part is a resource bound that forces a genuine design "
    "trade-off, not an implementation detail.",
    "Pick a problem where two reasonable approaches exist and the interesting question is which "
    "one survives contact with scale.",
    "Pick a problem arising from a failure mode practitioners have learned to live with rather "
    "than solve.",
    "Pick a problem where the obvious approach works until one assumption breaks, and name the "
    "assumption in the constraints without naming the fix.",
    "Pick a problem at the boundary between this domain and an adjacent one, where the standard "
    "techniques of neither quite apply.",
)


@dataclass(frozen=True, slots=True)
class GeneratorConfig:
    """`challenge_generation` from the season config."""

    protocol_version: str
    candidates_per_slot: int
    dedup_lookback_days: int
    duplicate_threshold_ppm: int
    minimum_reference_spread_ppm: int
    #: 7.4 step 5 conditions 5 and 3. Config-declared rather than code-defaulted — see `assess`.
    minimum_degradation_gap_ppm: int
    maximum_judge_instability_ppm: int
    #: Whether a pack that skipped the discrimination probe may be committed. False everywhere it
    #: matters; see the schema for why it exists at all.
    allow_unprobed_packs: bool
    #: family -> its config block.
    generators: Mapping[str, Mapping[str, Any]]
    #: The budget written into every generated challenge. Taken from the season's miner pricing,
    #: never defaulted: this is what the laboratory is told it may spend and what the gateway
    #: enforces, so a placeholder here is a round in which nothing can afford to run.
    maximum_wall_time_seconds: int
    maximum_rcc: int
    maximum_search_calls: int

    @classmethod
    def from_season(cls, season: Mapping[str, Any]) -> GeneratorConfig:
        block = season["challenge_generation"]
        if not 1 <= int(block["candidates_per_slot"]) <= len(_VARIATIONS):
            raise ValueError(
                f"candidates_per_slot is {block['candidates_per_slot']}; this module has "
                f"{len(_VARIATIONS)} distinct variation prompts, and asking one generator the "
                "same question more often than that yields paraphrases rather than candidates"
            )
        return cls(
            protocol_version=str(block["protocol_version"]),
            candidates_per_slot=int(block["candidates_per_slot"]),
            dedup_lookback_days=int(block["dedup_lookback_days"]),
            duplicate_threshold_ppm=int(block["duplicate_threshold_ppm"]),
            minimum_reference_spread_ppm=int(block["minimum_reference_spread_ppm"]),
            minimum_degradation_gap_ppm=int(block["minimum_degradation_gap_ppm"]),
            maximum_judge_instability_ppm=int(block["maximum_judge_instability_ppm"]),
            allow_unprobed_packs=bool(block["allow_unprobed_packs"]),
            generators={
                str(entry["family"]): dict(entry) for entry in block["generators"]
            },
            **_budget(season),
        )


#: What one call to a frontier model costs, in RCC, at the season's own pricing. A budget below
#: this cannot buy a single request, so a laboratory given one has nothing to do but fail — and it
#: fails at gate 13.1 with no portfolio, which reads as a broken laboratory rather than as a budget
#: that was never usable.
_ONE_CALL_RCC = 200_000


def _budget(season: Mapping[str, Any]) -> dict[str, int]:
    """The per-challenge budget, from the season's miner pricing.

    Read here rather than defaulted on the dataclass, because a default is a value that looks
    deliberate and is not: the challenge carries it to the laboratory, the gateway enforces it, and
    nothing downstream compares it against what the season actually allows.
    """
    pricing = season["providers"]["miner_pricing"]
    budget = {
        "maximum_rcc": int(pricing["maximum_rcc"]),
        "maximum_search_calls": int(pricing["maximum_search_calls"]),
        "maximum_wall_time_seconds": int(pricing["maximum_wall_time_seconds"]),
    }
    if budget["maximum_rcc"] < _ONE_CALL_RCC:
        raise ValueError(
            f"miner_pricing.maximum_rcc is {budget['maximum_rcc']}, which does not cover one "
            f"frontier-model call at roughly {_ONE_CALL_RCC} RCC. Every laboratory would refuse to "
            "start work it cannot finish and fail gate 13.1 with no portfolio, which looks like a "
            "field of broken laboratories rather than a budget nobody could use."
        )
    return budget


@dataclass(frozen=True, slots=True)
class Candidate:
    """One generated problem, before any filter has looked at it."""

    slot: Slot
    body: Mapping[str, Any]
    #: Which of the slot's candidates this was, for the audit trail.
    attempt: int
    rcc: int
    generator_model: str

    def with_body(self, body: Mapping[str, Any]) -> Candidate:
        return Candidate(
            slot=self.slot,
            body=body,
            attempt=self.attempt,
            rcc=self.rcc,
            generator_model=self.generator_model,
        )


async def generate_for_slot(
    client: ModelClient,
    *,
    slot: Slot,
    config: GeneratorConfig,
    excluded: frozenset[str] = EXCLUDED_DOMAINS,
) -> list[Candidate]:
    """Candidates for one slot, all from that slot's owning family.

    Requested concurrently. They are independent — no candidate depends on another — so
    sequencing them would multiply a slot's latency by `candidates_per_slot` for nothing, and
    twenty slots at four candidates is eighty calls whose serial cost is a large part of the
    day's window.
    """
    requests = [
        {
            "family": slot.generator_family,
            "purpose": Purpose.CHALLENGE_GENERATION,
            "system": _SYSTEM,
            "user": _USER.format(
                domain=slot.domain,
                index=slot.index + 1,
                total=config.candidates_per_slot,
                wall_time=config.maximum_wall_time_seconds,
                rcc=config.maximum_rcc,
                search_calls=config.maximum_search_calls,
                excluded=", ".join(sorted(excluded)),
                variation=_VARIATIONS[attempt],
            ),
            # Warmer than the 0.0 default. Candidate diversity within a slot is the point, and at
            # zero the variation prompts are the only source of difference.
            "temperature": 0.8,
            # The same reasoning-model allowance as the critic and the judges. A candidate is a
            # large object — statement, objective, baseline, constraints, forbidden shortcuts —
            # and at 4,096 the thinking left too little for it: the adapter was measured retrying
            # on `Expecting value: line 145 column 1`, a JSON body that stopped mid-document.
            "max_tokens": 12_288,
        }
        for attempt in range(config.candidates_per_slot)
    ]

    candidates: list[Candidate] = []
    for attempt, result in enumerate(await client.ask_many(requests)):
        if isinstance(result, Exception):
            # One failed candidate is expected and survivable; the slot has others. Logged at
            # warning because a *pattern* of these means the pack will come out short.
            _log.warning("slot %d candidate %d failed: %s", slot.index, attempt, result)
            continue
        body = _as_body(result, slot)
        if body is None:
            continue
        candidates.append(
            Candidate(
                slot=slot,
                body=body,
                attempt=attempt,
                rcc=result.rcc,
                generator_model=result.model,
            )
        )
    return candidates


def _as_body(result: ModelReply, slot: Slot) -> Mapping[str, Any] | None:
    """The reply as a challenge body, or None if it is not one.

    The domain is overwritten with the slot's domain rather than trusted from the reply. A
    generator that drifted to a neighbouring domain would otherwise unbalance the stratification
    7.2 declares, and the drift would be invisible because the resulting challenge is perfectly
    valid — just not the one the plan called for.
    """
    if not isinstance(result.parsed, Mapping):
        _log.warning(
            "slot %d: %s returned %s rather than an object",
            slot.index,
            result.family,
            type(result.parsed).__name__,
        )
        return None
    body = dict(result.parsed)
    declared = body.get("domain")
    if declared != slot.domain:
        _log.info(
            "slot %d: %s wrote domain %r; overwritten with the planned %r",
            slot.index,
            result.family,
            declared,
            slot.domain,
        )
    body["domain"] = slot.domain
    body["generator_family"] = slot.generator_family
    body["critic_family"] = slot.critic_family
    return body

