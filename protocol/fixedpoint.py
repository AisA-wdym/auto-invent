"""Integer arithmetic in parts-per-million, for everything a score depends on.

`season_config.json` declares every ratio as a ppm integer, so no float enters the subnet
through configuration. This module is what keeps floats out of the *arithmetic* as well: a
weighted sum of ppm values done in floating point would reintroduce exactly the divergence
the schema removed.

## Why the rounding rule is stated rather than inherited

Python's `//` floors toward negative infinity, and `round()` uses banker's rounding. Both are
defensible; what matters is that one is chosen and used everywhere, because two validators
using different ones produce different scores from identical inputs.

Floor is chosen, and the reason is directional: every value here is a share of something
finite. Flooring means a set of shares can never sum to more than the whole, so a rounding
artefact can only ever under-allocate — which is recoverable by redistribution — and never
over-allocate, which is not.

Where that matters most is `apply_weights`. It sums the numerators before dividing once,
rather than dividing per term and summing the results. Dividing per term floors each
contribution and loses up to one unit per criterion; with eight criteria that is a
systematic downward bias on every score in the subnet, identical in direction for everyone
and therefore invisible in a ranking — right up to the point it decides a qualification
floor.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

__all__ = [
    "PPM",
    "FixedPointError",
    "apply_weights",
    "assert_sums_to_one",
    "clamp_ppm",
    "mean_ppm",
    "mul_ppm",
    "quantile_ppm",
    "to_ppm",
]

#: One whole, in parts per million. Every ratio in the subnet is expressed against this.
PPM = 1_000_000


class FixedPointError(ValueError):
    """A fixed-point value or weight set that cannot be used as declared."""


def to_ppm(numerator: int, denominator: int) -> int:
    """`numerator / denominator` as ppm, floored.

    Integer inputs only. A float here would defeat the point of the module, and accepting
    one "for convenience" is how the first float enters a codebase that had none.
    """
    if isinstance(numerator, bool) or isinstance(denominator, bool):
        raise FixedPointError("booleans are not ratios")
    if not isinstance(numerator, int) or not isinstance(denominator, int):
        raise FixedPointError(
            f"to_ppm takes integers, got {type(numerator).__name__}/"
            f"{type(denominator).__name__}. A float would reintroduce the divergence ppm "
            "exists to remove."
        )
    if denominator == 0:
        raise FixedPointError("denominator is zero")
    return numerator * PPM // denominator


def mul_ppm(value: int, ratio_ppm: int) -> int:
    """`value * ratio`, where `ratio` is ppm. Floored, one division."""
    return value * ratio_ppm // PPM


def clamp_ppm(value: int) -> int:
    """Bound a ppm value to `[0, PPM]`.

    Used where a computation can legitimately overshoot — a Bradley-Terry score normalised
    against a shifting field, for instance — but a share above one whole is meaningless
    downstream. Clamps rather than raising, because the overshoot is a property of the
    estimator rather than an error, and refusing would strand a round on a rounding artefact.
    """
    return max(0, min(PPM, value))


def assert_sums_to_one(weights: Mapping[str, int], *, label: str) -> None:
    """Raise unless a weight set sums to exactly `PPM`.

    Exactly, with no tolerance. A tolerance would let a weight set that nearly sums to one
    pass, and "nearly" is a silent rescaling of every score computed with it — the kind of
    defect that changes rankings without changing anything visible.
    """
    total = sum(weights.values())
    if total != PPM:
        difference = total - PPM
        raise FixedPointError(
            f"{label} sums to {total}, not {PPM} (off by {difference:+d}). A weight set that "
            "does not sum to one whole silently rescales every score computed with it."
        )
    negative = sorted(name for name, weight in weights.items() if weight < 0)
    if negative:
        raise FixedPointError(f"{label} has negative weights: {negative}")


def apply_weights(values: Mapping[str, int], weights: Mapping[str, int]) -> int:
    """Weighted sum of ppm values with ppm weights, in one division.

    Only the criteria present in `values` contribute, and the weights are **renormalised
    over exactly those**. That is how an unscoreable criterion is expressed: omit it, and its
    weight is redistributed across the rest.

    Omission and zero must stay distinct. A criterion no judge could score is not a criterion
    scored zero — writing zero would dock the full weight, so a single judge outage on a 25%
    criterion would cost a quarter of the score for a reason that was never the miner's.
    Callers therefore omit rather than default, and this function's contract is what makes
    that safe.
    """
    present = {name: weights[name] for name in values if name in weights}
    denominator = sum(present.values())
    if denominator <= 0:
        raise FixedPointError(
            f"no weighted criterion is present: values={sorted(values)}, "
            f"weights={sorted(weights)}. A score cannot be computed from nothing, and "
            "returning zero would be indistinguishable from a genuinely zero score."
        )
    # One division, at the end. Dividing per term would floor each contribution and lose up
    # to one unit per criterion — a systematic downward bias, identical for everyone and so
    # invisible in a ranking, until it decides a qualification floor.
    numerator = sum(values[name] * present[name] for name in present)
    return numerator // denominator


def mean_ppm(values: Sequence[int]) -> int:
    """Arithmetic mean of ppm values, floored."""
    if not values:
        raise FixedPointError("mean of an empty sequence")
    return sum(values) // len(values)


def quantile_ppm(values: Sequence[int], *, ppm: int) -> int:
    """Lower-interpolated quantile of ppm values.

    `quantile_ppm(scores, ppm=250_000)` is the lower quartile that architecture.md 18.5
    weights at 30% — the component that penalises a laboratory brilliant on one problem and
    failing the rest.

    Lower interpolation between the two bracketing samples, computed in integers. Nearest-rank
    would be simpler but jumps discontinuously as the sample count changes, which would make
    a miner's daily score depend on how many challenges happened to be valid that day rather
    than on how it performed.
    """
    if not values:
        raise FixedPointError("quantile of an empty sequence")
    if not 0 <= ppm <= PPM:
        raise FixedPointError(f"quantile position {ppm} is outside [0, {PPM}]")

    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    # Position in units of (PPM * index) so the interpolation stays integral.
    scaled = ppm * (len(ordered) - 1)
    lower_index, remainder = divmod(scaled, PPM)
    if remainder == 0:
        return ordered[lower_index]
    lower, upper = ordered[lower_index], ordered[lower_index + 1]
    return lower + (upper - lower) * remainder // PPM
