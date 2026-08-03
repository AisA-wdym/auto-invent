"""Step 5 of 7.4: the discrimination probe. Does this problem separate laboratories at all?

Every earlier step asks whether a problem is *well-formed*. This one asks whether it is *useful*,
which is a different question with a different answer: a perfectly clear, perfectly constrained
problem that every laboratory answers identically contributes nothing to a ranking and costs a
full slot to ask.

Five rejection conditions from 7.4 step 5:

1. every reference produces essentially the same answer;
2. the problem is solved by trivial web retrieval;
3. the problem is so vague that judge results are unstable;
4. all reference outputs fail to provide any mechanism;
5. judge panels cannot distinguish intentionally degraded answers.

Conditions 1 and 5 are the two that matter most and they fail in opposite directions. **1** catches
a problem that is too easy: no spread, so no information. **5** catches a problem that is
unscoreable: the judges cannot tell a deliberately damaged answer from an intact one, which means
their scores on the real answers were not measuring anything either.

Condition 5 is the sharper instrument, and the reason is worth stating. Spread between references
can be produced by noise — four laboratories will differ somewhat on anything. But if a panel
cannot distinguish an answer with its mechanism section removed from one with it intact, no amount
of spread among real answers is evidence. It is the only one of the five that tests the *judge*
rather than the problem, and a problem is only as good as the panel's ability to score it.

## This step is expensive, and that is the design

Four reference laboratories plus a degraded-answer probe, per candidate. That is the single
largest cost in generation, and 21.1 budgets for it. The alternative — committing a pack and
discovering after execution that six of twenty problems did not discriminate — costs the whole
cohort's run against those six, which is far more.

Because it is expensive, it runs **last**: after the linter, the critic, the safety filter and
dedup have all had their cheap chance to reject.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from protocol.fixedpoint import PPM

__all__ = [
    "DiscriminationVerdict",
    "ProbeOutcome",
    "ReferenceProbe",
    "assess",
]

_log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ProbeOutcome:
    """What the reference laboratories and the degradation probe produced for one candidate.

    Scores are ppm integers. `reference_scores` maps a reference laboratory's name to the score a
    judge panel gave its portfolio; `degraded_scores` maps the same names to the score given to a
    *deliberately damaged* version of that same portfolio.
    """

    #: reference name -> panel score in ppm.
    reference_scores: Mapping[str, int]
    #: reference name -> panel score for the degraded variant, in ppm.
    degraded_scores: Mapping[str, int]
    #: How many references produced a usable mechanism section at all.
    with_mechanism: int
    #: Whether a plain web search answered the problem.
    answered_by_retrieval: bool
    #: Panel score variance across repeated judgings of the *same* portfolio, in ppm. High means
    #: the judges disagree with themselves, which is condition 3.
    judge_instability_ppm: int


class ReferenceProbe(Protocol):
    """Runs reference laboratories and a degradation check against a candidate.

    An interface rather than an implementation because this is where the probe becomes a real
    laboratory run — containers, the RCG, a judge panel — and the pipeline must be testable without
    any of that. The real implementation lives in `validator/sandbox/` and the judge, which is why
    it is not here: this module owns the *decision*, not the machinery.
    """

    async def probe(self, candidate: Mapping[str, Any]) -> ProbeOutcome: ...


@dataclass(frozen=True, slots=True)
class DiscriminationVerdict:
    """Whether a candidate discriminates, and which condition failed."""

    discriminates: bool
    #: The measured spread between the best and worst reference, in ppm.
    spread_ppm: int
    #: The measured gap between intact and degraded answers, in ppm.
    degradation_gap_ppm: int
    failures: tuple[str, ...] = ()

    def reason(self) -> str:
        return "; ".join(self.failures)


def assess(
    outcome: ProbeOutcome,
    *,
    minimum_spread_ppm: int,
    minimum_degradation_gap_ppm: int = 100_000,
    maximum_instability_ppm: int = 150_000,
) -> DiscriminationVerdict:
    """Apply 7.4 step 5's five conditions to a probe's measurements.

    Pure: takes measurements, returns a decision. The probe does the expensive work and this makes
    the call, so the threshold logic is testable against constructed measurements — including the
    boundary cases, which are the ones that decide whether a marginal problem enters a pack.
    """
    failures: list[str] = []

    scores = dict(outcome.reference_scores)
    if len(scores) < 2:
        # With one reference there is no spread to measure. Rejecting is the only honest option:
        # accepting would mean condition 1 was never checked, and reporting a spread of zero would
        # misattribute a missing measurement to an easy problem.
        return DiscriminationVerdict(
            discriminates=False,
            spread_ppm=0,
            degradation_gap_ppm=0,
            failures=(
                f"only {len(scores)} reference laboratory produced a result; condition 1 needs at "
                "least two to measure spread, and a missing measurement is not a passing one",
            ),
        )

    # Condition 1: every reference produces essentially the same answer.
    spread = max(scores.values()) - min(scores.values())
    if spread < minimum_spread_ppm:
        failures.append(
            f"references span only {spread} ppm (floor {minimum_spread_ppm}): every reference "
            "produced essentially the same answer, so this problem cannot separate laboratories "
            "and a slot spent on it measures nothing"
        )

    # Condition 2: solved by trivial web retrieval.
    if outcome.answered_by_retrieval:
        failures.append(
            "a plain web search answered the problem, so it rewards retrieval rather than "
            "invention"
        )

    # Condition 3: so vague that judge results are unstable.
    if outcome.judge_instability_ppm > maximum_instability_ppm:
        failures.append(
            f"judges scored the same portfolio {outcome.judge_instability_ppm} ppm apart across "
            f"repeats (ceiling {maximum_instability_ppm}): the problem is vague enough that the "
            "panel disagrees with itself, so its scores on real answers are noise"
        )

    # Condition 4: all reference outputs fail to provide any mechanism.
    if outcome.with_mechanism == 0:
        failures.append(
            "no reference produced a mechanism at all. 18.4 caps value and originality on a weak "
            "mechanism, so a problem where nobody can state one scores every laboratory at the cap"
        )

    # Condition 5: judge panels cannot distinguish intentionally degraded answers.
    gap = _degradation_gap(outcome)
    if gap < minimum_degradation_gap_ppm:
        failures.append(
            f"degraded answers scored within {gap} ppm of intact ones (floor "
            f"{minimum_degradation_gap_ppm}): the panel cannot tell a damaged portfolio from a "
            "whole one, so its scores on this problem are not measuring quality. This is the "
            "condition that matters most — spread among real answers can be noise, but a panel "
            "that fails this was never measuring anything to begin with."
        )

    return DiscriminationVerdict(
        discriminates=not failures,
        spread_ppm=spread,
        degradation_gap_ppm=gap,
        failures=tuple(failures),
    )


def _degradation_gap(outcome: ProbeOutcome) -> int:
    """Mean drop from intact to degraded, over references measured both ways.

    The *mean* rather than the maximum. A maximum would let one reference whose degraded variant
    happened to score badly carry the whole check — and the check is about whether the panel can
    reliably tell the difference, which is a property of the panel across answers rather than of
    one lucky pair.

    A reference present in only one of the two maps is skipped rather than treated as a zero. A
    missing degraded score means that probe failed, and scoring it as a total collapse would
    manufacture a large gap out of an outage.
    """
    paired = [
        (outcome.reference_scores[name], outcome.degraded_scores[name])
        for name in sorted(outcome.reference_scores)
        if name in outcome.degraded_scores
    ]
    if not paired:
        return 0
    drops = [max(0, intact - degraded) for intact, degraded in paired]
    return sum(drops) // len(drops)


def instability_ppm(repeats: Sequence[int]) -> int:
    """Spread across repeated judgings of one portfolio, for condition 3.

    Range rather than variance: the question is "how far apart can two judgings of the same thing
    land", and that is what a range answers directly. Variance would need a scale to interpret and
    would understate a single wild outlier, which is exactly the case that matters — one judging in
    ten landing 400,000 ppm away means a miner's score depends on which judging they got.
    """
    if len(repeats) < 2:
        return 0
    return max(0, min(PPM, max(repeats) - min(repeats)))
