"""17.3's Swiss tournament and 16.3's pairwise verdict.

Full all-versus-all is too expensive: 40 miners on 20 challenges is 15,600 comparisons per
criterion.
Swiss pairing gets a ranking from a few rounds by pairing near-equal opponents, which is where the
information is — a comparison between the best and worst laboratory has a foregone conclusion and
costs the same as an informative one.

## Order swapping is not a refinement, it is the measurement

"A/B presentation order is reversed." Language models have a position bias: presented with two
answers, they favour one slot at a rate well above chance regardless of content. A tournament that
presented each pair once would measure that bias and the answers together, with no way to separate
them.

So every pair is judged **twice**, once in each order, and the two verdicts are combined:

* both orders agree → a win for the agreed winner;
* the orders disagree → a **tie**, because the judge preferred a *position* rather than an answer.

Recording a disagreement as a tie rather than discarding it matters. Discarding would silently
delete the comparisons a judge found hardest, and those are exactly the near-equal pairs that Swiss
pairing deliberately produces — so the tournament would lose precision where it needs most.

The disagreement rate is also the panel's bias measurement, which 19 uses as a calibration signal
(`order_swap_inconsistency_ceiling_ppm` in the season config). It is not a nuisance to be minimised
away; it is the number that tells you whether the panel can be trusted.

## Pairings are deterministic given a seed

Swiss pairing needs a tie-break, and a validator that chose one freely could pair a favoured
laboratory against weak opponents. So pairings derive from the daily seed via the same hash-chain
stream as everything else, and two validators with the same seed and the same standings pair
identically.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from protocol.fixedpoint import PPM
from protocol.receipts import Purpose
from protocol.seeds import _seeded_stream
from validator.judge.bradley_terry import Outcome, Pairing
from validator.judge.panels import JUDGE_ROLES, Panel
from validator.model_client import ModelClient, ModelReply

__all__ = [
    "PairVerdict",
    "combine_orders",
    "compare_pair",
    "swiss_pairings",
]

_log = logging.getLogger(__name__)

#: Output ceiling for one comparison. Larger than the pointwise ceiling because 16.3's verdict
#: carries strengths, failures and a decisive reason for *both* candidates — and because a reasoning
#: model spends the same budget thinking first. See the note in `pointwise.py`.
_JUDGE_OUTPUT_TOKENS = 12_288

_SYSTEM = """\
You compare two research portfolios on one criterion and choose the better one.

Both are neutralised fact sheets: identity, branding and presentation have been removed, and any \
unsupported quantity is marked [unverified]. They are labelled A and B in a random order that \
carries no information — the same pair is shown to you and to another judge in the opposite order, \
and a preference that follows the label rather than the content is detected and discarded.

Choose on substance. If they are genuinely equal on this criterion, say so.

Return one JSON object and nothing else.
"""

_USER = """\
## Criterion: {criterion}

{question}

## Candidate A

{candidate_a}

## Candidate B

{candidate_b}

## Required JSON shape

{{
  "winner": "A",
  "confidence": 0.8,
  "a_strengths": ["..."],
  "b_strengths": ["..."],
  "a_failures": [],
  "b_failures": ["..."],
  "decisive_reason": "the one difference that decided it",
  "abstain": false
}}

`winner` is "A", "B", or "tie". Use "tie" when they are genuinely equal on this criterion — not \
when the comparison is difficult. Set `abstain` to true only if neither portfolio contains anything
\
bearing on this criterion.
"""


@dataclass(frozen=True, slots=True)
class PairVerdict:
    """16.3's judge output for one pair in one presentation order."""

    criterion: str
    family: str
    #: The uid shown in slot A, and in slot B. Recorded so a verdict can be re-attributed after the
    #: order is swapped — without this, combining two orders would be guesswork.
    slot_a: int
    slot_b: int
    #: "A", "B", or "tie", as the judge said it — in *slot* terms, not uid terms.
    winner: str
    confidence_ppm: int
    decisive_reason: str
    abstained: bool
    rcc: int

    def winning_uid(self) -> int | None:
        """The uid the judge preferred, or None for a tie or abstention."""
        if self.abstained or self.winner == "tie":
            return None
        return self.slot_a if self.winner == "A" else self.slot_b


def swiss_pairings(
    standings: Sequence[tuple[int, int]],
    *,
    seed: bytes,
    round_number: int,
    already_paired: Sequence[frozenset[int]] = (),
) -> list[tuple[int, int]]:
    """Pair near-equal opponents, avoiding repeats. Deterministic given the seed.

    `standings` is (uid, score_ppm), and pairing walks it in score order taking adjacent pairs —
    which is what puts the comparisons where the information is.

    Two departures from textbook Swiss, both for reasons this tournament has and chess does not:

    **Repeat avoidance looks ahead rather than swapping.** "Repeated identical pairings are
    limited", and when the natural partner has already been faced, the next unfaced laboratory
    down the standings is taken instead. A textbook Swiss would backtrack to preserve pairing
    quality; here the cost of a slightly worse pairing is small, and the cost of a complicated
    deterministic backtrack is that two validators must implement it identically.

    **An odd field gives a bye to the *middle*, not the leader.** A bye is a free non-comparison, so
    giving it to the leader would let the leader hold its position without being tested. Given to
    the
    middle, it costs the tournament the least information — the middle's rank is the least
    determined by any single comparison.
    """
    ordered = sorted(standings, key=lambda entry: (-entry[1], entry[0]))
    if len(ordered) < 2:
        return []

    faced = {frozenset(pair) for pair in already_paired}
    stream = _seeded_stream(seed, f"swiss-round-{round_number}".encode())

    remaining = [uid for uid, _ in ordered]
    if len(remaining) % 2 == 1:
        # Bye to the middle. Deterministic index, so every validator gives the same bye.
        bye_index = len(remaining) // 2
        bye = remaining.pop(bye_index)
        _log.info("round %d: uid %d receives a bye", round_number, bye)

    pairs: list[tuple[int, int]] = []
    while len(remaining) >= 2:
        first = remaining.pop(0)
        partner_index = 0
        for index, candidate in enumerate(remaining):
            if frozenset({first, candidate}) not in faced:
                partner_index = index
                break
        else:
            # Everyone left has been faced. Pair the nearest anyway rather than leaving laboratories
            # unpaired: a second comparison of the same pair still carries information about the
            # *judge*, and an unpaired laboratory contributes nothing to the ranking at all.
            _log.info(
                "round %d: uid %d has faced every remaining opponent; repeating the nearest",
                round_number,
                first,
            )
        second = remaining.pop(partner_index)
        # The seeded stream decides which of the two goes in slot A for the *first* presentation.
        # It is swapped for the second presentation regardless, so this only decides which order
        # comes first — but it must still be seeded, so two validators agree on the record.
        if next(stream) % 2:
            first, second = second, first
        pairs.append((first, second))
        faced.add(frozenset({first, second}))

    return pairs


async def compare_pair(
    client: ModelClient,
    *,
    panel: Panel,
    uid_a: int,
    uid_b: int,
    portfolio_a: Mapping[str, Any],
    portfolio_b: Mapping[str, Any],
) -> list[PairVerdict]:
    """Judge one pair in both orders, with every judge on the panel.

    Both orders for every judge, so the bias measurement is per-judge rather than per-panel. A panel
    where one family has a strong position bias and another has none averages to a moderate bias,
    and 19's calibration needs to remove the one judge rather than distrust the panel.
    """
    import json

    forward = [
        _request(panel, judge.family, uid_a, uid_b, portfolio_a, portfolio_b, json)
        for judge in panel.judges
    ]
    reversed_order = [
        _request(panel, judge.family, uid_b, uid_a, portfolio_b, portfolio_a, json)
        for judge in panel.judges
    ]

    replies = await client.ask_many([*forward, *reversed_order])
    verdicts: list[PairVerdict] = []
    for index, reply in enumerate(replies):
        judge = panel.judges[index % len(panel.judges)]
        if index < len(panel.judges):
            slot_a, slot_b = uid_a, uid_b
        else:
            slot_a, slot_b = uid_b, uid_a
        verdicts.append(
            _read(reply, criterion=panel.criterion, family=judge.family, slot_a=slot_a,
            slot_b=slot_b)
        )
    return verdicts


def _request(
    panel: Panel,
    family: str,
    slot_a: int,
    slot_b: int,
    body_a: Mapping[str, Any],
    body_b: Mapping[str, Any],
    json_module: Any,
) -> dict[str, Any]:
    return {
        "family": family,
        "purpose": Purpose.JUDGING,
        "system": _SYSTEM,
        "user": _USER.format(
            criterion=panel.criterion,
            question=JUDGE_ROLES[panel.criterion],
            candidate_a=json_module.dumps(dict(body_a), indent=2, sort_keys=True),
            candidate_b=json_module.dumps(dict(body_b), indent=2, sort_keys=True),
        ),
        "max_tokens": _JUDGE_OUTPUT_TOKENS,
    }


def _read(
    reply: ModelReply | Exception, *, criterion: str, family: str, slot_a: int, slot_b: int
) -> PairVerdict:
    if isinstance(reply, Exception):
        return PairVerdict(
            criterion=criterion,
            family=family,
            slot_a=slot_a,
            slot_b=slot_b,
            winner="tie",
            confidence_ppm=0,
            decisive_reason=f"the judge could not be reached or read ({reply})",
            abstained=True,
            rcc=0,
        )

    parsed = reply.parsed
    if not isinstance(parsed, Mapping):
        return PairVerdict(
            criterion=criterion,
            family=family,
            slot_a=slot_a,
            slot_b=slot_b,
            winner="tie",
            confidence_ppm=0,
            decisive_reason=f"returned {type(parsed).__name__} rather than an object",
            abstained=True,
            rcc=reply.rcc,
        )

    winner = str(parsed.get("winner", "")).strip().upper()
    if winner not in {"A", "B", "TIE"}:
        # An unreadable winner is an abstention, not a coin flip. Guessing would put a fabricated
        # comparison into the Bradley-Terry fit, and the fit cannot distinguish a guess from a
        # judgement.
        return PairVerdict(
            criterion=criterion,
            family=family,
            slot_a=slot_a,
            slot_b=slot_b,
            winner="tie",
            confidence_ppm=0,
            decisive_reason=f"winner was {parsed.get('winner')!r}, not A, B or tie",
            abstained=True,
            rcc=reply.rcc,
        )

    confidence = parsed.get("confidence")
    confidence_ppm = (
        max(0, min(PPM, int(float(confidence) * PPM)))
        if isinstance(confidence, int | float) and not isinstance(confidence, bool)
        else 0
    )

    return PairVerdict(
        criterion=criterion,
        family=family,
        slot_a=slot_a,
        slot_b=slot_b,
        winner="tie" if winner == "TIE" else winner,
        confidence_ppm=confidence_ppm,
        decisive_reason=str(parsed.get("decisive_reason", "")),
        abstained=bool(parsed.get("abstain")),
        rcc=reply.rcc,
    )


def combine_orders(verdicts: Sequence[PairVerdict]) -> tuple[list[Pairing], int]:
    """Fold both presentation orders into pairings, and report the disagreement rate.

    Returns `(pairings, inconsistency_ppm)`.

    A judge that preferred laboratory X in one order and laboratory Y in the other expressed a
    preference for a *position*. That becomes a tie: the comparison happened and produced no
    information about the answers, which is different from not happening.

    The inconsistency rate is the panel's position-bias measurement, and 19 compares it against
    `order_swap_inconsistency_ceiling_ppm`. It is reported rather than corrected because a biased
    judge should be removed from the panel, not compensated for — compensation would leave it
    voting.
    """
    by_judge: dict[tuple[str, frozenset[int]], list[PairVerdict]] = {}
    for verdict in verdicts:
        key = (verdict.family, frozenset({verdict.slot_a, verdict.slot_b}))
        by_judge.setdefault(key, []).append(verdict)

    pairings: list[Pairing] = []
    disagreements = 0
    comparable = 0

    for (_family, participants), pair in sorted(by_judge.items(), key=lambda item:
    sorted(item[0][1])):
        voting = [verdict for verdict in pair if not verdict.abstained]
        if not voting:
            continue
        left, right = sorted(participants)

        winners = {verdict.winning_uid() for verdict in voting}
        if len(voting) < 2:
            # Only one order came back. Counted, because discarding it would throw away a real
            # verdict, but it contributes nothing to the bias measurement — one presentation cannot
            # show a position preference.
            single = next(iter(winners))
            pairings.append(_pairing(left, right, single))
            continue

        comparable += 1
        if len(winners) == 1:
            pairings.append(_pairing(left, right, next(iter(winners))))
        else:
            disagreements += 1
            pairings.append(_pairing(left, right, None))

    inconsistency = disagreements * PPM // comparable if comparable else 0
    if comparable and inconsistency > 250_000:
        _log.warning(
            "%d of %d order-swapped comparisons disagreed (%.1f%%). At this rate the panel is "
            "largely measuring presentation position rather than content.",
            disagreements,
            comparable,
            inconsistency / 10_000,
        )
    return pairings, inconsistency


def _pairing(left: int, right: int, winner: int | None) -> Pairing:
    """One comparison in `bradley_terry`'s shape.

    uids become strings because `Pairing` keys competitors by name — the fit is over an arbitrary
    label set, and 17.5's cross-validator replication compares rankings of hotkeys rather than of
    uids, which are recycled on deregistration.
    """
    if winner is None:
        # A tie, which the fit scores as half a win each rather than discarding. Discarding would
        # throw away the information that two laboratories are close, which is what a ranking most
        # needs in the middle of the field.
        return Pairing(a=str(left), b=str(right), winner=Outcome.TIE)
    return Pairing(
        a=str(left),
        b=str(right),
        winner=Outcome.A if winner == left else Outcome.B,
    )
