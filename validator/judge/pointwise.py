"""17.1: anchored pointwise screening. Every valid miner, cheaply.

Five anchors, 0 to 4, stated explicitly in the prompt:

    0 — absent or invalid
    1 — superficial
    2 — plausible but incomplete
    3 — strong and concrete
    4 — unusually strong, coherent and differentiated

## Why anchors rather than "score 0 to 100"

An unanchored scale drifts. The same judge, given the same portfolio a week apart, returns 72 and
then 65 — not because it changed its mind but because 72 was never a measurement of anything. Worse,
two judges on an unanchored scale disagree systematically rather than randomly: one centres on 70
and the other on 50, so their average is a function of panel composition.

Five anchors with stated meanings turn the judgement into a classification, which models do far more
reproducibly than they estimate magnitudes. 27's requirement of same-bundle rerun correlation at
0.80 is a requirement about exactly this.

## Screening is not the score, and the funnel depends on that

"This screening score is not the final score. It identifies candidates for full judging." The
pointwise result feeds 18.3's *anchored* component at 0.25 weight, and it selects the cohort for the
Swiss tournament. Both uses tolerate a noisier estimate than the final ranking needs, which is what
makes one cheap call per miner acceptable here and not in 17.3.

## The 0-to-4 scale becomes ppm by anchor position, not by division

A raw score of 3 maps to 750,000 ppm. That is `3/4`, but it is written as a lookup rather than a
division because the anchors are ordinal categories, not measurements on a ratio scale — the
distance from "superficial" to "plausible but incomplete" is not obviously the same as from "strong"
to "unusually strong". Expressing the mapping as a table makes it a stated decision that can be
revised, rather than an arithmetic accident.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from protocol.receipts import Purpose
from validator.judge.panels import JUDGE_ROLES, Panel
from validator.model_client import ModelClient, ModelReply

__all__ = [
    "ANCHORS",
    "ANCHOR_PPM",
    "PointwiseScore",
    "screen_portfolio",
    "score_to_ppm",
]

_log = logging.getLogger(__name__)

#: 17.1's anchors, verbatim. In the prompt, so the judge classifies against stated meanings.
ANCHORS: Mapping[int, str] = {
    0: "absent or invalid",
    1: "superficial",
    2: "plausible but incomplete",
    3: "strong and concrete",
    4: "unusually strong, coherent and differentiated",
}

#: Anchor -> ppm. A table rather than `score * PPM // 4`, because the anchors are ordinal
#: categories: the gap between "superficial" and "plausible but incomplete" need not equal the gap
#: between "strong" and "unusually strong". Written out so the spacing is a decision on record.
ANCHOR_PPM: Mapping[int, int] = {
    0: 0,
    1: 250_000,
    2: 500_000,
    3: 750_000,
    4: 1_000_000,
}

_SYSTEM = """\
You score one research portfolio on one criterion, against fixed anchors.

You are scoring a neutralised fact sheet: identity, branding and presentation have been removed, \
and any unsupported quantity is marked [unverified]. Judge what remains. Do not reward \
confident phrasing, and do not penalise plainness.

Return one JSON object and nothing else.
"""

_USER = """\
## Criterion: {criterion}

{question}

## Anchors — choose exactly one

{anchors}

## The portfolio

{portfolio}

{extra}

## Required JSON shape

{{
  "score": 0,
  "anchor": "the anchor text you chose, copied exactly",
  "reasoning": "one or two sentences naming the specific evidence that decided it",
  "abstain": false
}}

Set `abstain` to true only if the portfolio contains nothing that bears on this criterion at all — \
not because the answer is difficult. An abstention removes your vote and redistributes its weight, \
so abstaining on a hard case shifts the decision to the other judges rather than registering doubt.
"""


@dataclass(frozen=True, slots=True)
class PointwiseScore:
    """One judge's anchored score on one criterion for one portfolio."""

    criterion: str
    family: str
    #: 0-4, or None when the judge abstained or could not be read.
    raw: int | None
    score_ppm: int
    reasoning: str
    abstained: bool
    rcc: int

    @property
    def voted(self) -> bool:
        """Whether this counts toward the criterion.

        An abstention and an unreadable reply are both "did not vote". They are recorded distinctly
        for the audit trail (22) but treated identically by the aggregation, because in both cases
        no
        judgement exists and substituting one would be inventing it.
        """
        return not self.abstained and self.raw is not None


def score_to_ppm(raw: int) -> int:
    """Anchor to ppm, refusing anything off the scale.

    A judge returning 7 on a 0-4 scale has not understood the rubric, and clamping to 4 would give
    it the top score for that misunderstanding. Refused instead, and the vote is dropped.
    """
    if raw not in ANCHOR_PPM:
        raise ValueError(
            f"{raw} is not one of 17.1's anchors {sorted(ANCHOR_PPM)}. Clamping to the nearest "
            "would award a judge that misread the rubric the score its misreading implies."
        )
    return ANCHOR_PPM[raw]


async def screen_portfolio(
    client: ModelClient,
    *,
    panel: Panel,
    portfolio: Mapping[str, Any],
    prior_art: Sequence[Mapping[str, Any]] = (),
    duplicate_clusters: Sequence[Sequence[int]] = (),
) -> list[PointwiseScore]:
    """Score one portfolio on one criterion, with every judge on the panel.

    All judges concurrently, and failures returned as values — so a panel of three that loses one
    scores with two. `asyncio.gather` propagating the first exception would discard the two verdicts
    that did arrive and turn one model's rate limit into a criterion nobody scored.
    """
    import json

    extra = ""
    if panel.criterion == "originality" and prior_art:
        # The Originality judge is explicitly told to judge against the report rather than its own
        # impression of what exists — 15 forbids the validator from asserting absolute novelty, and
        # a judge left to its own recall would assert it implicitly.
        extra = (
            "## Prior art the validator found\n\n"
            f"{json.dumps([dict(entry) for entry in prior_art], indent=2, sort_keys=True)}\n\n"
            "Judge originality against this. Where the report found nothing, that means nothing "
            "was found — not that nothing exists."
        )
    elif panel.criterion == "diversity" and duplicate_clusters:
        extra = (
            "## Ideas the validator has already clustered as one lineage\n\n"
            f"{[list(cluster) for cluster in duplicate_clusters]}\n\n"
            "These indices are the same idea restated. Do not count them as distinct directions."
        )

    requests = [
        {
            "family": judge.family,
            "purpose": Purpose.JUDGING,
            "system": _SYSTEM,
            "user": _USER.format(
                criterion=panel.criterion,
                question=JUDGE_ROLES[panel.criterion],
                anchors="\n".join(f"- **{value}** — {text}" for value, text in ANCHORS.items()),
                portfolio=json.dumps(dict(portfolio), indent=2, sort_keys=True),
                extra=extra,
            ),
            "max_tokens": 1_536,
        }
        for judge in panel.judges
    ]

    replies = await client.ask_many(requests)
    return [
        _read(reply, criterion=panel.criterion, family=judge.family)
        for judge, reply in zip(panel.judges, replies, strict=True)
    ]


def _read(
    reply: ModelReply | Exception, *, criterion: str, family: str
) -> PointwiseScore:
    """Turn one judge's reply into a score, or into a recorded non-vote."""
    if isinstance(reply, Exception):
        _log.info("%s judge for %s did not answer: %s", family, criterion, reply)
        return PointwiseScore(
            criterion=criterion,
            family=family,
            raw=None,
            score_ppm=0,
            reasoning=f"the judge could not be reached or read ({reply})",
            abstained=False,
            rcc=0,
        )

    parsed = reply.parsed
    if not isinstance(parsed, Mapping):
        return PointwiseScore(
            criterion=criterion,
            family=family,
            raw=None,
            score_ppm=0,
            reasoning=f"returned {type(parsed).__name__} rather than an object",
            abstained=False,
            rcc=reply.rcc,
        )

    reasoning = str(parsed.get("reasoning", ""))
    if bool(parsed.get("abstain")):
        return PointwiseScore(
            criterion=criterion,
            family=family,
            raw=None,
            score_ppm=0,
            reasoning=reasoning,
            abstained=True,
            rcc=reply.rcc,
        )

    raw = parsed.get("score")
    if not isinstance(raw, int) or isinstance(raw, bool):
        # `isinstance(True, int)` is True in Python, and a judge returning `"score": true` would
        # otherwise be read as 1 — a superficial score awarded for a type error.
        return PointwiseScore(
            criterion=criterion,
            family=family,
            raw=None,
            score_ppm=0,
            reasoning=f"score was {raw!r}, not one of {sorted(ANCHOR_PPM)}",
            abstained=False,
            rcc=reply.rcc,
        )

    try:
        score_ppm = score_to_ppm(raw)
    except ValueError as error:
        return PointwiseScore(
            criterion=criterion,
            family=family,
            raw=None,
            score_ppm=0,
            reasoning=str(error),
            abstained=False,
            rcc=reply.rcc,
        )

    return PointwiseScore(
        criterion=criterion,
        family=family,
        raw=raw,
        score_ppm=score_ppm,
        reasoning=reasoning,
        abstained=False,
        rcc=reply.rcc,
    )


def aggregate(scores: Sequence[PointwiseScore]) -> tuple[int, int]:
    """Mean of the votes, and how many there were. Returns (ppm, voters).

    The **mean** rather than the median, because a panel is three or four judges and a median over
    three discards two thirds of the evidence to guard against an outlier that three votes cannot
    identify anyway. The guard against an outlier judge is 19's calibration, which removes it from
    the panel — not a robust statistic that quietly tolerates it forever.

    Zero voters returns `(0, 0)`, and the caller must treat a zero-voter criterion as *unscored*
    rather than as scored zero. `protocol.fixedpoint.apply_weights` redistributes over the criteria
    present, which is why it must be told the criterion is absent rather than handed a zero.
    """
    votes = [score.score_ppm for score in scores if score.voted]
    if not votes:
        return 0, 0
    return sum(votes) // len(votes), len(votes)
