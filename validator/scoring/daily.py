"""The daily score, and the rolling score built from daily ones.

architecture.md 18.5 and 18.6.

```
D_m = 0.70 * mean(S_m,c) + 0.30 * Q25(S_m,c)          -- one day, over challenges
rolling = mean(all valid days)                         -- fewer than 7 valid days
        = 0.60 * median(last 7) + 0.40 * median(last 30)   -- 7 or more
```

## Why the daily score is not just a mean

The lower-quartile term at 30% is what 18.5 calls out: it "penalizes laboratories that perform
brilliantly on one problem but fail on most others". Twenty challenges a day makes that
measurable — a laboratory that spikes on the two problems matching its house style and
collapses on the rest has the same mean as a consistent one, and a different quartile.

## Why the rolling score switches estimator rather than scaling

18.6 states plainly: *"There is no credibility multiplier that suppresses new miners."*

The predecessor to this design had one, and it made a new coldkey worth close to nothing for
roughly three seasons. So the estimator switch here must not become one by accident. It
**selects how a score is computed**; it never multiplies the result. A laboratory with three
excellent days scores exactly what those three days earned — not a fraction of it pending
seniority.

Median rather than mean for the established case, and the reason is the opposite of
suppression: a median is robust to one bad day, so a validator outage or a single unlucky pack
cannot erase a week of good work.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from protocol.fixedpoint import PPM, clamp_ppm, mean_ppm, mul_ppm, quantile_ppm

__all__ = [
    "DailyConfig",
    "DailyScore",
    "ScoreHistory",
    "daily_score",
    "rolling_score",
]

_log = logging.getLogger(__name__)

#: The quantile 18.5 weights at 30%.
_LOWER_QUARTILE_PPM = 250_000


class DailyScoreError(ValueError):
    """A daily or rolling score that cannot be computed as given."""


@dataclass(frozen=True, slots=True)
class DailyConfig:
    """Scoring cadence parameters, from the signed season."""

    daily_mean_weight_ppm: int
    daily_q25_weight_ppm: int
    rolling_short_days: int
    rolling_long_days: int
    rolling_short_weight_ppm: int
    rolling_long_weight_ppm: int
    minimum_days_for_median: int
    minimum_valid_challenges: int

    def __post_init__(self) -> None:
        for label, first, second in (
            ("daily", self.daily_mean_weight_ppm, self.daily_q25_weight_ppm),
            ("rolling", self.rolling_short_weight_ppm, self.rolling_long_weight_ppm),
        ):
            if first + second != PPM:
                raise DailyScoreError(
                    f"{label} weights sum to {first + second}, not {PPM}: a split that does not "
                    "sum to one whole rescales every score computed with it"
                )
        if self.rolling_short_days >= self.rolling_long_days:
            raise DailyScoreError(
                f"the short window ({self.rolling_short_days}d) must be shorter than the long "
                f"one ({self.rolling_long_days}d), or the two medians measure the same thing"
            )

    @classmethod
    def from_season(cls, season: Mapping[str, object]) -> DailyConfig:
        scoring = season["scoring"]  # type: ignore[index]
        allocation = season["weight_allocation"]  # type: ignore[index]
        return cls(
            daily_mean_weight_ppm=int(scoring["daily_mean_weight_ppm"]),  # type: ignore[index]
            daily_q25_weight_ppm=int(scoring["daily_q25_weight_ppm"]),  # type: ignore[index]
            rolling_short_days=int(scoring["rolling_short_days"]),  # type: ignore[index]
            rolling_long_days=int(scoring["rolling_long_days"]),  # type: ignore[index]
            rolling_short_weight_ppm=int(
                scoring["rolling_short_weight_ppm"]  # type: ignore[index]
            ),
            rolling_long_weight_ppm=int(scoring["rolling_long_weight_ppm"]),  # type: ignore[index]
            minimum_days_for_median=int(scoring["minimum_days_for_median"]),  # type: ignore[index]
            minimum_valid_challenges=int(
                allocation["minimum_valid_challenges"]  # type: ignore[index]
            ),
        )


@dataclass(frozen=True, slots=True)
class DailyScore:
    """One laboratory's day, with the two components kept separately.

    Both are retained because 22 publishes daily scores, and the mean and the quartile say
    different things: a large gap between them *is* the diagnosis — it means inconsistency
    rather than weakness, and the two are addressed differently.
    """

    score_ppm: int
    mean_ppm: int
    lower_quartile_ppm: int
    valid_challenges: int
    qualifies: bool


def daily_score(
    challenge_scores_ppm: Sequence[int], config: DailyConfig
) -> DailyScore:
    """`0.70 * mean + 0.30 * Q25` over one day's challenges (18.5).

    `qualifies` reports whether the day met `minimum_valid_challenges` (20.1). Reported rather
    than enforced here: a day below the minimum still has a real score, and whether it counts
    toward weight is a decision for the allocator, which is the only place that can see the
    whole field. Deciding it here would make a partial day silently vanish rather than being
    excluded on the record.
    """
    if not challenge_scores_ppm:
        raise DailyScoreError(
            "a day with no valid challenges has no score. Zero would be indistinguishable from "
            "a laboratory that ran and failed everything."
        )

    mean = mean_ppm(list(challenge_scores_ppm))
    quartile = quantile_ppm(list(challenge_scores_ppm), ppm=_LOWER_QUARTILE_PPM)
    combined = mul_ppm(mean, config.daily_mean_weight_ppm) + mul_ppm(
        quartile, config.daily_q25_weight_ppm
    )
    return DailyScore(
        score_ppm=clamp_ppm(combined),
        mean_ppm=mean,
        lower_quartile_ppm=quartile,
        valid_challenges=len(challenge_scores_ppm),
        qualifies=len(challenge_scores_ppm) >= config.minimum_valid_challenges,
    )


@dataclass(frozen=True, slots=True)
class ScoreHistory:
    """A laboratory's daily scores, most recent last.

    Dates are carried alongside the scores so a window means "the last N *days*" rather than
    "the last N *results*". A laboratory that missed three days must not have a stale week
    counted as current, and a list of bare scores cannot tell the difference.
    """

    dates: Sequence[str]
    scores_ppm: Sequence[int]

    def __post_init__(self) -> None:
        if len(self.dates) != len(self.scores_ppm):
            raise DailyScoreError(
                f"{len(self.dates)} dates against {len(self.scores_ppm)} scores"
            )
        if list(self.dates) != sorted(self.dates):
            raise DailyScoreError(
                "history must be in ascending date order: a window taken from an unsorted "
                "history would select an arbitrary set of days rather than the most recent"
            )

    def window(self, days: int) -> list[int]:
        """The most recent `days` results."""
        return list(self.scores_ppm[-days:]) if days > 0 else []


def _median_ppm(values: Sequence[int]) -> int:
    """Median, floored on an even count.

    Floored rather than rounded, matching every other rounding decision in the subnet, so a
    score can never be inflated by a rounding artefact.
    """
    if not values:
        raise DailyScoreError("median of an empty window")
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) // 2


@dataclass(frozen=True, slots=True)
class RollingScore:
    """The score weights are allocated from."""

    score_ppm: int
    valid_days: int
    estimator: str


def rolling_score(history: ScoreHistory, config: DailyConfig) -> RollingScore:
    """The rolling score (18.6). Selects an estimator; never scales the result.

    Below `minimum_days_for_median`, the plain mean of every valid day. At or above it,
    `0.60 * median(short window) + 0.40 * median(long window)`.

    The switch must not become a credibility multiplier. 18.6 forbids one explicitly, and the
    predecessor design's version made a new coldkey worth almost nothing for three seasons. So a
    laboratory with three excellent days scores exactly what those days earned — the estimator
    changes, the magnitude is not discounted for youth.

    Asserted by a test: a newcomer whose every day is perfect scores `PPM`, not a fraction of it.
    """
    if not history.scores_ppm:
        raise DailyScoreError(
            "no valid days. A laboratory with no results has no score, which is different from "
            "a score of zero — the allocator excludes it rather than ranking it last."
        )

    valid_days = len(history.scores_ppm)

    if valid_days < config.minimum_days_for_median:
        # Every valid day, weighted equally. No discount for being new.
        return RollingScore(
            score_ppm=clamp_ppm(mean_ppm(list(history.scores_ppm))),
            valid_days=valid_days,
            estimator=f"mean of {valid_days} day(s)",
        )

    short = _median_ppm(history.window(config.rolling_short_days))
    long = _median_ppm(history.window(config.rolling_long_days))
    blended = mul_ppm(short, config.rolling_short_weight_ppm) + mul_ppm(
        long, config.rolling_long_weight_ppm
    )
    return RollingScore(
        score_ppm=clamp_ppm(blended),
        valid_days=valid_days,
        estimator=(
            f"{config.rolling_short_weight_ppm}ppm x median({config.rolling_short_days}d) + "
            f"{config.rolling_long_weight_ppm}ppm x median({config.rolling_long_days}d)"
        ),
    )
