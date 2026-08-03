"""Scoring a round: §17's funnel over what the executions produced.

Every piece this calls is built and unit-tested somewhere else. What was missing was the composition
— the thing that turns 20N portfolios into one standings table — and composition is where the
expensive mistakes live, because each piece is individually correct and the errors are all in how
they are joined.

## The funnel, and why it is a funnel

Judging is the validator's largest cost, and pairwise judging is quadratic in the field. §17 spends
it in three tiers:

1. **Screening** (17.1) — every valid response gets a cheap anchored pointwise score from every
   panel. This is the pass that has to be affordable at N × 20.
2. **The cohort** (17.2) — the top screeners, plus a random draw, plus every laboratory with no
   history. The last two are not fairness decoration: without them the ranking is a function of last
   week's ranking, a new entrant can never be compared against an incumbent, and incumbency becomes
   self-perpetuating on a subnet whose entire premise is that today's winner can be forked tomorrow.
3. **The tournament** (17.3) — Swiss pairings within the cohort, both presentation orders, per
   challenge.

## Unmeasured is not zero, at every level

This is the rule the whole file is arranged around, and it is easy to violate by accident.

A criterion that no judge could score is `None`, and `challenge_score` redistributes its weight. A
criterion scored *zero* costs the laboratory its full weight. With originality at 25%, one judge
outage silently written as zero is a quarter of a day's score removed for something that was never
the miner's fault — and it looks exactly like a laboratory that produced nothing original.

The same rule applies upward: a laboratory outside the cohort has no pairwise measurement, so its
`pairwise_ppm` is `None` and its screen carries the score. Not zero.

## Determinism

Every random choice comes from the day's seed. Two validators on the same pack and the same
executions must select the same cohort and the same pairings, or 17.5's replication compares two
different tournaments and concludes the validators disagree.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from validator.canonicalizer.neutral import canonicalize
from validator.execution import Execution, RoundExecution
from validator.judge.bradley_terry import fit, strengths_to_ppm
from validator.judge.pairwise import PairVerdict, combine_orders, compare_pair, swiss_pairings
from validator.judge.panels import Panel
from validator.judge.pointwise import aggregate, screen_portfolio
from validator.model_client import ModelClient
from validator.scoring.criteria import (
    ChallengeScore,
    CriterionInputs,
    ScoringConfig,
    challenge_score,
)
from validator.scoring.daily import (
    DailyConfig,
    DailyScore,
    RollingScore,
    ScoreHistory,
    daily_score,
    rolling_score,
)

__all__ = ["FunnelConfig", "LabScore", "RoundScores", "score_round"]

_log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FunnelConfig:
    """§17's tiers, from the season config."""

    cohort_top: int
    cohort_random: int
    cohort_new: int
    swiss_rounds: int
    prior_art_ideas_per_portfolio: int

    @classmethod
    def from_season(cls, season: Mapping[str, Any]) -> FunnelConfig:
        funnel = season["funnel"]
        return cls(
            cohort_top=int(funnel["cohort_top"]),
            cohort_random=int(funnel["cohort_random"]),
            cohort_new=int(funnel["cohort_new"]),
            swiss_rounds=int(funnel["swiss_rounds"]),
            prior_art_ideas_per_portfolio=int(funnel["prior_art_ideas_per_portfolio"]),
        )


@dataclass(frozen=True, slots=True)
class LabScore:
    """One laboratory's result for a round."""

    uid: int
    hotkey: str
    #: challenge_id -> the challenge score in ppm. Only valid executions appear.
    per_challenge_ppm: Mapping[str, int]
    daily: DailyScore | None
    rolling: RollingScore | None
    valid_challenges: int
    #: 7.2.1's overfit signal: the score gap between the two generator families.
    family_gap_ppm: int
    in_cohort: bool
    failed_gates: tuple[str, ...]
    rcc_spent: int

    @property
    def rolling_ppm(self) -> int:
        return self.rolling.score_ppm if self.rolling else 0

    @property
    def daily_ppm(self) -> int:
        return self.daily.score_ppm if self.daily else 0


@dataclass(frozen=True, slots=True)
class RoundScores:
    """A round's standings, plus what it cost to produce them."""

    labs: tuple[LabScore, ...] = ()
    cohort: tuple[int, ...] = ()
    #: Criteria that could not be scored at all, per challenge. An operator health signal: a
    #: criterion missing across every challenge is a judge outage, not a field of bad portfolios.
    unscored_criteria: Mapping[str, int] = field(default_factory=dict)
    judging_rcc: int = 0


def _seeded_choice(seed: bytes, pool: Sequence[int], count: int) -> list[int]:
    """A deterministic draw of `count` uids from `pool`.

    Deterministic from the day's seed rather than from `random`, because two validators must draw
    the same cohort — 17.5's replication compares tournaments, and two different cohorts are two
    different tournaments, which would read as the validators disagreeing about the miners.
    """
    if count <= 0 or not pool:
        return []
    ranked = sorted(
        pool,
        key=lambda uid: hashlib.blake2b(
            seed + uid.to_bytes(4, "big"), digest_size=8
        ).digest(),
    )
    return ranked[:count]


async def score_round(
    *,
    execution: RoundExecution,
    challenges: Sequence[Mapping[str, Any]],
    client: ModelClient,
    panels: Mapping[str, Panel],
    funnel: FunnelConfig,
    scoring: ScoringConfig,
    daily: DailyConfig,
    seed: bytes,
    round_id: str,
    history: Mapping[int, ScoreHistory],
) -> RoundScores:
    """Screen, select, judge, and reduce to standings."""
    valid = [item for item in execution.executions if item.valid]
    if not valid:
        _log.warning(
            "round %s: no execution passed its gates, so there is nothing to judge", round_id
        )
        return RoundScores(labs=_empty_labs(execution))

    canonical = _canonicalise(valid)
    _log.info("round %s: screening %d valid responses", round_id, len(canonical))

    screens, unscored = await _screen_all(canonical, client=client, panels=panels)
    cohort = _select_cohort(
        screens=screens, funnel=funnel, seed=seed, history=history, execution=execution
    )
    _log.info("round %s: cohort is %s", round_id, sorted(cohort))

    pairwise = await _tournament(
        canonical=canonical,
        cohort=cohort,
        challenges=challenges,
        client=client,
        panels=panels,
        funnel=funnel,
        seed=seed,
    )

    labs = _reduce(
        execution=execution,
        challenges=challenges,
        screens=screens,
        pairwise=pairwise,
        cohort=cohort,
        scoring=scoring,
        daily=daily,
        history=history,
        round_id=round_id,
    )
    return RoundScores(
        labs=labs,
        cohort=tuple(sorted(cohort)),
        unscored_criteria=unscored,
        judging_rcc=client.spent() if hasattr(client, "spent") else 0,
    )


# --------------------------------------------------------------------------
# 14: neutralise before anything reads it
# --------------------------------------------------------------------------


def _canonicalise(valid: Sequence[Execution]) -> dict[tuple[int, str], dict[str, Any]]:
    """Strip presentation and substitute measured usage, before a judge sees anything.

    Before screening rather than before the tournament. Screening is a judge too, and a screen
    run on raw portfolios would let formatting decide the cohort — after which the tournament would
    be perfectly neutral about a field that presentation had already chosen.
    """
    canonical: dict[tuple[int, str], dict[str, Any]] = {}
    for item in valid:
        if item.result is None or item.result.portfolio is None:
            continue
        neutral = canonicalize(item.result.portfolio, measured_usage=item.result.measured_usage)
        canonical[(item.uid, item.challenge_id)] = dict(neutral.body)
    return canonical


# --------------------------------------------------------------------------
# 17.1: the cheap pass
# --------------------------------------------------------------------------


async def _screen_all(
    canonical: Mapping[tuple[int, str], Mapping[str, Any]],
    *,
    client: ModelClient,
    panels: Mapping[str, Panel],
) -> tuple[dict[tuple[int, str], dict[str, int]], dict[str, int]]:
    """Anchored pointwise scores for every response on every criterion.

    Returns `(uid, challenge) -> {criterion: ppm}` with unscored criteria **absent** rather than
    zero, and a count of how often each criterion could not be scored at all.
    """
    keys = list(canonical)
    tasks = [
        _screen_one(client, panels=panels, portfolio=canonical[key]) for key in keys
    ]
    collected = await asyncio.gather(*tasks)

    screens: dict[tuple[int, str], dict[str, int]] = {}
    unscored: dict[str, int] = {}
    for key, scored in zip(keys, collected, strict=True):
        screens[key] = scored
        for criterion in panels:
            if criterion not in scored:
                unscored[criterion] = unscored.get(criterion, 0) + 1
    if unscored:
        _log.warning(
            "criteria that could not be screened at all: %s. Their weight is redistributed rather "
            "than counted as zero, but a criterion missing across the whole field is a judge "
            "outage rather than a field of bad portfolios.",
            unscored,
        )
    return screens, unscored


async def _screen_one(
    client: ModelClient, *, panels: Mapping[str, Panel], portfolio: Mapping[str, Any]
) -> dict[str, int]:
    scored: dict[str, int] = {}
    for criterion, panel in panels.items():
        try:
            votes = await screen_portfolio(client, panel=panel, portfolio=portfolio)
        except Exception as error:  # noqa: BLE001 - one criterion must not lose a whole portfolio
            _log.warning("screening %s failed: %s", criterion, error)
            continue
        value, voters = aggregate(votes)
        if voters == 0:
            # Absent, not zero. `aggregate` returns (0, 0) for no voters, and writing that 0 into
            # the mapping would be indistinguishable from a panel that agreed the answer was worth
            # nothing.
            continue
        scored[criterion] = value
    return scored


# --------------------------------------------------------------------------
# 17.2: who gets the expensive pass
# --------------------------------------------------------------------------


def _select_cohort(
    *,
    screens: Mapping[tuple[int, str], Mapping[str, int]],
    funnel: FunnelConfig,
    seed: bytes,
    history: Mapping[int, ScoreHistory],
    execution: RoundExecution,
) -> set[int]:
    """Top screeners, a seeded random draw, and every laboratory with no history.

    The last two are the anti-lock-in provisions of 17.2, and they are not decoration. Without the
    random draw the cohort is a function of the screen, which correlates with last week's cohort;
    without the new-miner admission a first-day laboratory is judged only by the cheap pass and can
    never be compared against an incumbent it has never been paired with.
    """
    mean_screen: dict[int, int] = {}
    counts: dict[int, int] = {}
    for (uid, _challenge), scored in screens.items():
        if not scored:
            continue
        mean_screen[uid] = mean_screen.get(uid, 0) + sum(scored.values()) // len(scored)
        counts[uid] = counts.get(uid, 0) + 1
    averaged = {uid: total // counts[uid] for uid, total in mean_screen.items() if counts[uid]}

    ranked = sorted(averaged, key=lambda uid: (-averaged[uid], uid))
    cohort = set(ranked[: funnel.cohort_top])

    newcomers = [
        uid
        for uid in sorted(execution.by_uid)
        if uid not in cohort and not history.get(uid, ScoreHistory((), ())).scores_ppm
    ]
    cohort.update(newcomers[: funnel.cohort_new])

    remaining = [uid for uid in ranked if uid not in cohort]
    cohort.update(_seeded_choice(seed, remaining, funnel.cohort_random))
    return cohort


# --------------------------------------------------------------------------
# 17.3: the tournament
# --------------------------------------------------------------------------


async def _tournament(
    *,
    canonical: Mapping[tuple[int, str], Mapping[str, Any]],
    cohort: set[int],
    challenges: Sequence[Mapping[str, Any]],
    client: ModelClient,
    panels: Mapping[str, Panel],
    funnel: FunnelConfig,
    seed: bytes,
) -> dict[tuple[int, str], dict[str, int]]:
    """Swiss pairings per challenge, both orders, reduced to Bradley-Terry strengths.

    Returns `(uid, challenge) -> {criterion: ppm}`. A challenge with fewer than two cohort members
    produces nothing, which is correct: a pairwise measurement needs a pair, and inventing one from
    a single portfolio is how a field of one gets a perfect score.
    """
    pairwise: dict[tuple[int, str], dict[str, int]] = {}

    for challenge in challenges:
        challenge_id = str(challenge.get("challenge_id", ""))
        present = sorted(
            uid for uid in cohort if (uid, challenge_id) in canonical
        )
        if len(present) < 2:
            continue

        verdicts: list[PairVerdict] = []
        faced: list[frozenset[int]] = []
        standings = [(uid, 500_000) for uid in present]

        for round_number in range(funnel.swiss_rounds):
            pairings = swiss_pairings(
                standings, seed=seed, round_number=round_number, already_paired=faced
            )
            comparisons = []
            for left, right in pairings:
                faced.append(frozenset((left, right)))
                for panel in panels.values():
                    comparisons.append(
                        compare_pair(
                            client,
                            panel=panel,
                            uid_a=left,
                            uid_b=right,
                            portfolio_a=canonical[(left, challenge_id)],
                            portfolio_b=canonical[(right, challenge_id)],
                        )
                    )
            for batch in await asyncio.gather(*comparisons, return_exceptions=True):
                if isinstance(batch, BaseException):
                    # One pair's judging failing is one comparison lost, not a tournament lost.
                    _log.warning("a comparison failed: %s", batch)
                    continue
                verdicts.extend(batch)

        for criterion in panels:
            per_criterion = [verdict for verdict in verdicts if verdict.criterion == criterion]
            if not per_criterion:
                continue
            pairings_, inconsistency = combine_orders(per_criterion)
            if not pairings_:
                continue
            strengths = strengths_to_ppm(fit(pairings_))
            if inconsistency:
                _log.debug(
                    "%s on %s: %d ppm order-swap inconsistency",
                    criterion,
                    challenge_id,
                    inconsistency,
                )
            for uid, value in strengths.items():
                pairwise.setdefault((int(uid), challenge_id), {})[criterion] = value

    return pairwise


# --------------------------------------------------------------------------
# 18: down to one number per laboratory
# --------------------------------------------------------------------------


def _reduce(
    *,
    execution: RoundExecution,
    challenges: Sequence[Mapping[str, Any]],
    screens: Mapping[tuple[int, str], Mapping[str, int]],
    pairwise: Mapping[tuple[int, str], Mapping[str, int]],
    cohort: set[int],
    scoring: ScoringConfig,
    daily: DailyConfig,
    history: Mapping[int, ScoreHistory],
    round_id: str,
) -> tuple[LabScore, ...]:
    family_of = {
        str(challenge.get("challenge_id", "")): str(challenge.get("generator_family", ""))
        for challenge in challenges
    }
    labs: list[LabScore] = []

    for uid in sorted(execution.by_uid):
        items = execution.for_uid(uid)
        hotkey = items[0].hotkey if items else ""
        per_challenge: dict[str, int] = {}
        by_family: dict[str, list[int]] = {}

        for item in items:
            if not item.valid:
                continue
            key = (uid, item.challenge_id)
            inputs = _criterion_inputs(screens.get(key, {}), pairwise.get(key, {}))
            if not inputs:
                continue
            scored: ChallengeScore = challenge_score(inputs, scoring)
            per_challenge[item.challenge_id] = scored.total_ppm
            by_family.setdefault(family_of.get(item.challenge_id, ""), []).append(
                scored.total_ppm
            )

        day = None
        roll = None
        if per_challenge:
            day = daily_score(list(per_challenge.values()), daily)
            past = history.get(uid, ScoreHistory((), ()))
            roll = rolling_score(
                ScoreHistory(
                    dates=[*past.dates, round_id],
                    scores_ppm=[*past.scores_ppm, day.score_ppm],
                ),
                daily,
            )

        labs.append(
            LabScore(
                uid=uid,
                hotkey=hotkey,
                per_challenge_ppm=per_challenge,
                daily=day,
                rolling=roll,
                valid_challenges=len(per_challenge),
                family_gap_ppm=_family_gap(by_family),
                in_cohort=uid in cohort,
                failed_gates=tuple(
                    sorted({gate for item in items for gate in item.failed_gates})
                ),
                rcc_spent=sum(item.measured_rcc for item in items),
            )
        )
    return tuple(labs)


def _criterion_inputs(
    screen: Mapping[str, int], pair: Mapping[str, int]
) -> dict[str, CriterionInputs]:
    """Join the two measurements per criterion, keeping absence as absence.

    A criterion measured by neither is omitted from the mapping entirely, so `challenge_score`
    redistributes its weight. A criterion measured by one is carried with `None` for the other,
    which is what `CriterionInputs` is `None`-able for: a laboratory outside the cohort has no
    pairwise measurement, and scoring that as zero would make exclusion from the cohort a penalty
    rather than a sampling decision.
    """
    inputs: dict[str, CriterionInputs] = {}
    for criterion in {*screen, *pair}:
        entry = CriterionInputs(
            pairwise_ppm=pair.get(criterion),
            pointwise_ppm=screen.get(criterion),
        )
        if entry.scoreable:
            inputs[criterion] = entry
    return inputs


def _family_gap(by_family: Mapping[str, Sequence[int]]) -> int:
    """7.2.1's drift signal: the gap between the two generator families' mean scores.

    Published rather than kept internal. A miner who can see it can fix it; one who cannot is being
    penalised for something invisible. Zero when only one family produced valid results, because a
    gap needs two sides — and reporting the single side's mean as a gap would flag every laboratory
    that failed half a pack.
    """
    means = [
        sum(scores) // len(scores)
        for family, scores in sorted(by_family.items())
        if family and scores
    ]
    if len(means) < 2:
        return 0
    return max(means) - min(means)


def _empty_labs(execution: RoundExecution) -> tuple[LabScore, ...]:
    """Standings for a round where nothing passed its gates.

    Every laboratory still appears, with its failed gates. A round that published an empty table
    would be indistinguishable from a round with no entrants, and the miners who ran and failed
    would have nothing to fix.
    """
    return tuple(
        LabScore(
            uid=uid,
            hotkey=items[0].hotkey if items else "",
            per_challenge_ppm={},
            daily=None,
            rolling=None,
            valid_challenges=0,
            family_gap_ppm=0,
            in_cohort=False,
            failed_gates=tuple(sorted({gate for item in items for gate in item.failed_gates})),
            rcc_spent=sum(item.measured_rcc for item in items),
        )
        for uid, items in sorted(execution.by_uid.items())
    )
