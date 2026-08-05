"""Bradley-Terry: turning pairwise verdicts into a ranking.

architecture.md 18.3 makes pairwise the primary signal at 0.75 weight. The reason is in 16.3:
a judge asked "is this good, 1-10?" rewards confident prose, while a judge asked "which of
these two is better on mechanism?" has to find a discriminating difference. But pairwise
verdicts are not a score — they are a tournament, and Bradley-Terry is what converts one into
the other.

The model: each laboratory has a latent strength, and the probability that A beats B is
`s_A / (s_A + s_B)`. Fitting means finding the strengths that best explain the verdicts
observed. A laboratory that beat strong opponents scores above one that beat weak ones, which
is exactly what a Swiss tournament needs — since under Swiss pairing nobody faces the
same field.

## Why the fit is iterative and why that is fine here

There is no closed form. The standard fit is Zermelo's majorisation-minimisation update,
which is monotone: each iteration cannot decrease the likelihood. That makes it safe to stop
after a fixed number of iterations rather than on a convergence test — and stopping on a
fixed count is what makes the result reproducible.

A convergence test would compare a float against a threshold, so two hosts whose arithmetic
differs in the last bit could take a different number of iterations and land on different
strengths. architecture.md 27 requires same-bundle rerun rank correlation at 0.80 or above,
and a variable iteration count is a direct threat to it. So: fixed iterations, inputs sorted,
and the output converted to ppm integers at the boundary so nothing downstream ever sees a
float.

## Cross-validator agreement is not required; self-agreement is

27 asks for cross-validator rank correlation of 0.60 and same-bundle rerun correlation of
0.80. Validators are *expected* to differ — they run different challenge packs — so this fit
does not need to be bit-identical between hosts. It needs to be identical between two runs on
one host, which fixed iterations and sorted inputs deliver.

## An undefeated laboratory has unbounded strength

If a competitor never loses, the likelihood increases without limit as its strength grows, and
the fit diverges. Every real implementation regularises; this one adds a symmetric prior — a
half-win and a half-loss against a virtual average opponent — which is the standard fix and
has a useful reading: an undefeated laboratory with three wins is ranked above a laboratory
with two, but not infinitely above, because three wins is not yet proof.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from protocol.fixedpoint import PPM, clamp_ppm

__all__ = ["BradleyTerryError", "Outcome", "Pairing", "fit", "strengths_to_ppm"]

_log = logging.getLogger(__name__)

#: Iterations of the MM update. Fixed rather than convergence-tested, so the result is
#: reproducible: a threshold comparison on floats can take a different number of steps on two
#: hosts and land on different strengths.
#:
#: 200 is far past where the update stops moving for tournaments of this size (tens of
#: competitors, a few hundred comparisons). Verified by a test that the last fifty iterations
#: change no ranking.
_ITERATIONS = 200

#: The symmetric prior, in half-games against a virtual average opponent. Without it an
#: undefeated competitor's strength diverges. With it, undefeated still ranks first — just not
#: infinitely far first, which is the honest reading of a small unbeaten record.
_PRIOR_WEIGHT = 0.5


class BradleyTerryError(ValueError):
    """A tournament that cannot be fitted as given."""


class Outcome:
    """Who won one comparison. Strings rather than an enum for a wire-shaped value."""

    A = "A"
    B = "B"
    TIE = "tie"


@dataclass(frozen=True, slots=True)
class Pairing:
    """One judged comparison.

    `weight` lets a comparison count for less than one game. Used for a tie, which is scored
    as half a win each rather than discarded: discarding ties would throw away the
    information that two laboratories are close, which is precisely the information a ranking
    needs most in the middle of the field.
    """

    a: str
    b: str
    winner: str
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.a == self.b:
            raise BradleyTerryError(f"{self.a!r} cannot be compared with itself")
        if self.winner not in (Outcome.A, Outcome.B, Outcome.TIE):
            raise BradleyTerryError(f"unknown outcome {self.winner!r}")
        if self.weight <= 0:
            raise BradleyTerryError("a comparison must carry positive weight")


def fit(
    pairings: Sequence[Pairing],
    *,
    iterations: int = _ITERATIONS,
    prior_weight: float = _PRIOR_WEIGHT,
) -> dict[str, float]:
    """Latent strengths from pairwise verdicts. Deterministic for a given input.

    Returns strengths normalised so they sum to the number of competitors, which keeps them
    interpretable — 1.0 is exactly average — and keeps the later ppm conversion stable as the
    field grows.

    Competitors are sorted before fitting. Iterating a dict would make the arithmetic depend
    on insertion order, and floating-point addition is not associative: the same tournament
    processed in two orders can produce strengths that differ in the last bits, which is
    enough to swap two adjacent ranks.
    """
    if not pairings:
        raise BradleyTerryError(
            "no comparisons to fit. An empty tournament has no ranking, and returning a flat "
            "one would claim every laboratory is equal rather than that nothing is known."
        )

    competitors = sorted({name for pairing in pairings for name in (pairing.a, pairing.b)})
    index = {name: position for position, name in enumerate(competitors)}
    count = len(competitors)

    # wins[i] is i's total win credit; games[(i, j)] is how often i and j met.
    wins = [0.0] * count
    met: dict[tuple[int, int], float] = {}
    for pairing in sorted(pairings, key=lambda p: (p.a, p.b, p.winner)):
        i, j = index[pairing.a], index[pairing.b]
        if pairing.winner == Outcome.A:
            wins[i] += pairing.weight
        elif pairing.winner == Outcome.B:
            wins[j] += pairing.weight
        else:
            # A tie is half a win each. Discarding it would lose the information that two
            # laboratories are close, which is what a ranking most needs mid-field.
            wins[i] += pairing.weight / 2
            wins[j] += pairing.weight / 2
        key = (i, j) if i < j else (j, i)
        met[key] = met.get(key, 0.0) + pairing.weight

    strengths = [1.0] * count

    for _ in range(iterations):
        updated = [0.0] * count
        for position in range(count):
            # The MM update: new strength is total wins divided by the sum, over opponents,
            # of games / (own strength + opponent strength).
            denominator = 0.0
            for opponent in range(count):
                if opponent == position:
                    continue
                key = (position, opponent) if position < opponent else (opponent, position)
                games = met.get(key)
                if games:
                    denominator += games / (strengths[position] + strengths[opponent])
            # The symmetric prior: a half-win and a half-loss against an average opponent.
            # This is what stops an undefeated competitor diverging.
            numerator = wins[position] + prior_weight
            denominator += 2 * prior_weight / (strengths[position] + 1.0)
            updated[position] = numerator / denominator if denominator > 0 else strengths[position]

        # Renormalise to mean 1.0. The model is scale-invariant, so without this the whole
        # vector drifts and the ppm conversion below would depend on how many iterations ran.
        total = sum(updated)
        if total <= 0:
            raise BradleyTerryError("the fit collapsed to zero strength across the field")
        strengths = [value * count / total for value in updated]

    return dict(zip(competitors, strengths, strict=True))


def strengths_to_ppm(strengths: Mapping[str, float]) -> dict[str, int]:
    """Strengths to ppm, by rank position rather than by magnitude.

    The conversion is the last place a float exists. Everything downstream — the criterion
    combination, the daily score, the weight vector — is integer.

    Rank position rather than a linear rescale of the strengths themselves, and the reason is
    substantive. Bradley-Terry strengths are on a multiplicative scale with no upper bound: one
    laboratory that dominates a weak field can hold a strength ten times the next, and a linear
    rescale would hand it a near-perfect score while compressing everyone else to nearly zero.
    That would make the pairwise component behave like winner-take-all, which architecture.md
    20.2 deliberately does not want — it uses a softmax over scores with a temperature and a
    cap, precisely so the *distribution* is a policy decision made once, at weight allocation,
    rather than an accident of how the fit happened to spread.

    So this reports where a laboratory placed, evenly spaced. Ties in strength share a
    position, so two genuinely equal laboratories receive equal scores rather than being
    separated by a floating-point artefact.
    """
    if not strengths:
        raise BradleyTerryError("no strengths to convert")
    if len(strengths) == 1:
        # A single competitor placed first out of one. Reporting PPM would claim a perfect
        # result from no comparison; reporting the midpoint says "ranked, but against nobody".
        return {name: PPM // 2 for name in strengths}

    # Sorted by strength descending, then by name, so equal strengths order reproducibly.
    ordered = sorted(strengths.items(), key=lambda item: (-item[1], item[0]))
    positions: dict[str, int] = {}
    rank = 0
    previous_strength: float | None = None
    for offset, (name, strength) in enumerate(ordered):
        # Equal strengths share a rank, so a floating-point artefact cannot separate two
        # laboratories the tournament found indistinguishable.
        if previous_strength is None or strength != previous_strength:
            rank = offset
            previous_strength = strength
        positions[name] = rank

    span = len(ordered) - 1
    return {
        name: clamp_ppm(PPM - (position * PPM // span)) for name, position in positions.items()
    }
