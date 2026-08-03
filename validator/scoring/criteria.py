"""One challenge's score: rank weighting, pairwise/pointwise combination, mechanism floor.

architecture.md 18.1 through 18.4. The order of operations is the substance of this module,
because several of the steps are only correct in one sequence and each wrong ordering still
produces a plausible number.

```
per-idea scores  --18.1-->  portfolio score per criterion
                            (rank-weighted 0.40/0.25/0.15/0.12/0.08,
                             duplicate ideas collapsed to one lineage first)
                                     |
pairwise (BT) + pointwise (anchored) --18.3--> C_k = 0.75 BT + 0.25 AR
                                     |
                            --18.4--> mechanism floor: if mechanism < 0.40,
                                      value and originality capped at 0.50
                                     |
                            --18.4--> S = sum over k of w_k C_k
```

## Duplicates are collapsed before rank weighting, not after

18.1: "ideas that are merely semantic duplicates are collapsed into one lineage before
scoring." Collapsing *after* would let a laboratory submit its best idea five times: each copy
would score well, the rank weights would all land on the same idea, and a one-idea portfolio
would earn a five-idea score. Collapsing first means the duplicates occupy one rank and the
remaining weight redistributes over what is actually distinct — so padding a portfolio with
restatements lowers it.

## The mechanism floor is applied to the combined score, not to either input

18.4 caps value and originality when mechanism is below the floor. Applied before the
pairwise/pointwise combination it would cap the wrong quantity: a laboratory could score badly
on pointwise mechanism, be capped, and then have a strong pairwise mechanism result lift it
back over the floor — so the cap would depend on which input was consulted first. The floor is
a statement about the criterion's final value, so it is applied there.

## Both weight sets are validated on every call

`assert_sums_to_one` runs each time rather than once at load. It is cheap, and the failure it
catches is a weight set that nearly sums to one — which silently rescales every score computed
with it, changing rankings without changing anything visible.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from protocol.fixedpoint import (
    PPM,
    apply_weights,
    assert_sums_to_one,
    clamp_ppm,
    mul_ppm,
)

__all__ = [
    "CriterionInputs",
    "ScoringConfig",
    "ScoringError",
    "challenge_score",
    "collapse_duplicates",
    "combine_pairwise_pointwise",
    "rank_weighted",
]

_log = logging.getLogger(__name__)

#: Criteria the mechanism floor caps. Named rather than "all except mechanism", because the
#: floor is a specific claim: an idea cannot be *valuable* or *original* without a mechanism.
#: It says nothing about whether the constraints were met or the portfolio was diverse, and
#: capping those would punish work the floor makes no claim about.
_FLOOR_CAPS = ("value", "originality")


class ScoringError(ValueError):
    """A score that cannot be computed from the inputs given."""


@dataclass(frozen=True, slots=True)
class ScoringConfig:
    """The scoring parameters, from the signed season."""

    criterion_weights_ppm: Mapping[str, int]
    rank_weights_ppm: Sequence[int]
    pairwise_weight_ppm: int
    pointwise_weight_ppm: int
    mechanism_floor_ppm: int
    capped_on_weak_mechanism_ppm: int

    def __post_init__(self) -> None:
        assert_sums_to_one(dict(self.criterion_weights_ppm), label="criterion_weights_ppm")
        assert_sums_to_one(
            {str(i): w for i, w in enumerate(self.rank_weights_ppm)}, label="rank_weights_ppm"
        )
        combined = self.pairwise_weight_ppm + self.pointwise_weight_ppm
        if combined != PPM:
            raise ScoringError(
                f"pairwise + pointwise weights sum to {combined}, not {PPM}. A split that does "
                "not sum to one whole rescales every criterion score."
            )
        if self.pairwise_weight_ppm < self.pointwise_weight_ppm:
            # 18.3 makes pairwise the primary signal. A configuration that inverted it would
            # be legal arithmetic and the wrong mechanism: pointwise rewards fluency, which is
            # the failure pairwise exists to avoid.
            raise ScoringError(
                f"pointwise weight ({self.pointwise_weight_ppm}) exceeds pairwise "
                f"({self.pairwise_weight_ppm}). Section 18.3 makes pairwise the primary signal; "
                "pointwise is a diagnostic anchor, and inverting them would reward fluency."
            )

    @classmethod
    def from_season(cls, season: Mapping[str, object]) -> ScoringConfig:
        scoring = season["scoring"]  # type: ignore[index]
        return cls(
            criterion_weights_ppm=dict(season["criterion_weights_ppm"]),  # type: ignore[arg-type]
            rank_weights_ppm=list(season["rank_weights_ppm"]),  # type: ignore[arg-type]
            pairwise_weight_ppm=int(scoring["pairwise_weight_ppm"]),  # type: ignore[index]
            pointwise_weight_ppm=int(scoring["pointwise_weight_ppm"]),  # type: ignore[index]
            mechanism_floor_ppm=int(scoring["mechanism_floor_ppm"]),  # type: ignore[index]
            capped_on_weak_mechanism_ppm=int(
                scoring["capped_on_weak_mechanism_ppm"]  # type: ignore[index]
            ),
        )


@dataclass(frozen=True, slots=True)
class CriterionInputs:
    """One criterion's two measurements.

    `pairwise_ppm` may be `None` when a criterion produced no comparisons — a panel below the
    family minimum, or every judge abstaining. That is not zero, and the distinction is the
    whole of why this is `None`-able: a criterion nobody could compare has its weight
    redistributed, while a criterion scored zero docks the miner its full weight.
    """

    pairwise_ppm: int | None
    pointwise_ppm: int | None

    @property
    def scoreable(self) -> bool:
        return self.pairwise_ppm is not None or self.pointwise_ppm is not None


def collapse_duplicates(
    per_idea_ppm: Sequence[int], lineages: Sequence[int]
) -> list[int]:
    """Reduce per-idea scores to one score per distinct lineage (18.1).

    `lineages[i]` is the lineage id of idea `i`; ideas the canonicalizer found semantically
    duplicate share one. The surviving score for a lineage is its **best** member, and the
    duplicates vanish rather than averaging in.

    Best rather than mean, deliberately: a laboratory that produced one strong idea and
    restated it weakly four times has produced one strong idea, and averaging would punish it
    below a laboratory that produced the same idea once. What the collapse removes is the
    *credit for repetition*, not the quality of the work.

    Returned in descending order, so the rank weights apply to the strongest surviving
    lineages. Order in the input is the laboratory's own ranking, which 18.1's weights are
    about — but a collapse can leave the input order no longer descending, and rank weights
    applied to an unsorted list would credit position rather than quality.
    """
    if len(per_idea_ppm) != len(lineages):
        raise ScoringError(
            f"{len(per_idea_ppm)} idea scores against {len(lineages)} lineage labels"
        )
    best: dict[int, int] = {}
    for score, lineage in zip(per_idea_ppm, lineages, strict=True):
        best[lineage] = max(best.get(lineage, 0), score)
    return sorted(best.values(), reverse=True)


def rank_weighted(per_idea_ppm: Sequence[int], rank_weights_ppm: Sequence[int]) -> int:
    """Roll per-idea scores up to one portfolio score (18.1).

    `Q = 0.40 Q1 + 0.25 Q2 + 0.15 Q3 + 0.12 Q4 + 0.08 Q5`, positionally, with **fixed** weights.
    A portfolio with fewer surviving lineages than declared ranks scores zero for the missing
    positions.

    ## Why the missing positions are not redistributed

    An earlier version of this function redistributed the unused weight, reasoning that the
    duplicate collapse had already removed the credit for repetition and scoring the empty slots
    as zero would charge for it twice. That reasoning is wrong, and a test caught it by measuring
    the outcome: a portfolio of five distinct ideas scoring 900k down to 500k earned **777,000**,
    while five copies of one 900k idea collapsed to a single lineage and earned **900,000**.

    Padding beat genuine diversity. Redistribution made the optimal strategy "submit one
    excellent idea and leave the rest empty", which defeats both the Top-5 requirement and the
    diversity criterion at once.

    The distinction the earlier version missed is *whose* failure a missing measurement is:

    * A criterion no judge could score is the **validator's** gap. Its weight redistributes —
      a laboratory must not pay for a judge outage.
    * A rank with no distinct idea behind it is the **miner's** gap. The challenge asked for
      `portfolio_size` distinct ideas; fewer were delivered. Its weight is forfeit.

    So redistribution is correct in `challenge_score` and wrong here, and the two look identical
    until the incentive is measured.
    """
    if not per_idea_ppm:
        raise ScoringError("a portfolio with no ideas has no score")
    if len(per_idea_ppm) > len(rank_weights_ppm):
        # More surviving lineages than the season ranks. Extra ideas beyond the requested
        # portfolio size earn nothing rather than diluting the weighted positions, so a
        # laboratory cannot improve its score by attaching a long tail of extras.
        _log.info(
            "%d lineages against %d declared ranks; scoring the top %d",
            len(per_idea_ppm),
            len(rank_weights_ppm),
            len(rank_weights_ppm),
        )
    total = 0
    for position, weight in enumerate(rank_weights_ppm):
        score = per_idea_ppm[position] if position < len(per_idea_ppm) else 0
        total += score * weight
    return clamp_ppm(total // PPM)


def combine_pairwise_pointwise(inputs: CriterionInputs, config: ScoringConfig) -> int | None:
    """`C_k = 0.75 BT_k + 0.25 AR_k` (18.3), or `None` if neither was measured.

    When only one of the two exists it carries the whole weight rather than being scaled by its
    own share. Scaling would make a criterion with only a pointwise result score at most 0.25 of
    what it earned — indistinguishable from having scored badly, when in fact one measurement
    was simply unavailable.
    """
    pairwise, pointwise = inputs.pairwise_ppm, inputs.pointwise_ppm
    if pairwise is None and pointwise is None:
        return None
    if pointwise is None:
        return clamp_ppm(pairwise)  # type: ignore[arg-type]
    if pairwise is None:
        _log.info("criterion scored from the pointwise anchor alone; no comparisons were made")
        return clamp_ppm(pointwise)
    return clamp_ppm(
        mul_ppm(pairwise, config.pairwise_weight_ppm)
        + mul_ppm(pointwise, config.pointwise_weight_ppm)
    )


def _apply_mechanism_floor(
    criteria: dict[str, int], config: ScoringConfig
) -> tuple[dict[str, int], bool]:
    """Cap value and originality when mechanism is below the floor (18.4).

    Applied to the combined criterion scores, not to either input: applied earlier, a weak
    pointwise mechanism could be lifted back over the floor by a strong pairwise result, and
    the cap would depend on which input was consulted first.

    A portfolio with no mechanism score is *not* capped. The floor is a claim about a measured
    mechanism, and an unmeasured one is not a weak one — capping on absence would punish a
    laboratory for a judge outage.
    """
    mechanism = criteria.get("mechanism")
    if mechanism is None or mechanism >= config.mechanism_floor_ppm:
        return criteria, False

    capped = dict(criteria)
    for name in _FLOOR_CAPS:
        if name in capped:
            capped[name] = min(capped[name], config.capped_on_weak_mechanism_ppm)
    _log.info(
        "mechanism %d is below the floor %d: value and originality capped at %d. An idea "
        "cannot score highly merely by sounding unusual.",
        mechanism,
        config.mechanism_floor_ppm,
        config.capped_on_weak_mechanism_ppm,
    )
    return capped, True


@dataclass(frozen=True, slots=True)
class ChallengeScore:
    """One challenge's outcome for one laboratory, with the working shown.

    The intermediate values are kept rather than discarded because architecture.md 22 publishes
    challenge scores, and a published score nobody can decompose is a number to be trusted
    rather than checked.
    """

    total_ppm: int
    criteria_ppm: Mapping[str, int]
    mechanism_floor_applied: bool
    omitted_criteria: tuple[str, ...]


def challenge_score(
    inputs: Mapping[str, CriterionInputs], config: ScoringConfig
) -> ChallengeScore:
    """`S = sum over k of w_k C_k` (18.4), with the floor applied first.

    Criteria that could not be measured are **omitted**, so their weight redistributes over
    the rest. That is the one behaviour in this module most likely to be got wrong by
    simplification: `criteria.get(name, 0)` reads as a harmless default and costs a miner the
    criterion's full weight for a judge outage that was never its fault. With originality at
    25%, one such default is a quarter of the score.
    """
    combined: dict[str, int] = {}
    omitted: list[str] = []
    for name, measurement in sorted(inputs.items()):
        if name not in config.criterion_weights_ppm:
            _log.warning("criterion %r is not weighted by this season; ignoring", name)
            continue
        value = combine_pairwise_pointwise(measurement, config)
        if value is None:
            omitted.append(name)
            continue
        combined[name] = value

    # Criteria the season weights but the caller never mentioned are omissions too.
    omitted.extend(name for name in config.criterion_weights_ppm if name not in combined)

    if not combined:
        raise ScoringError(
            "no criterion could be scored for this challenge. Returning zero would be "
            "indistinguishable from a portfolio that was scored and found worthless."
        )

    capped, floor_applied = _apply_mechanism_floor(combined, config)
    total = apply_weights(capped, dict(config.criterion_weights_ppm))

    return ChallengeScore(
        total_ppm=clamp_ppm(total),
        criteria_ppm=capped,
        mechanism_floor_applied=floor_applied,
        omitted_criteria=tuple(sorted(set(omitted))),
    )
