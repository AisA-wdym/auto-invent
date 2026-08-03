"""Parts-per-million arithmetic.

The schema keeps floats out of configuration; this module keeps them out of the arithmetic.
A weighted sum of ppm values done in floating point would reintroduce exactly the divergence
the schema removed.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from protocol.fixedpoint import (
    PPM,
    FixedPointError,
    apply_weights,
    assert_sums_to_one,
    clamp_ppm,
    mean_ppm,
    mul_ppm,
    quantile_ppm,
    to_ppm,
)

pytestmark = pytest.mark.determinism

# architecture.md 18.2
CRITERIA = {
    "originality": 250_000,
    "value": 200_000,
    "mechanism": 150_000,
    "constraint_fit": 120_000,
    "diversity": 100_000,
    "self_selection": 80_000,
    "falsifiability": 70_000,
    "cost_reliability": 30_000,
}


# --------------------------------------------------------------------------
# The spec's own numbers
# --------------------------------------------------------------------------


def test_the_criterion_weights_sum_to_one_whole():
    assert_sums_to_one(CRITERIA, label="criterion weights")


def test_the_rank_weights_sum_to_one_whole():
    """architecture.md 18.1: 0.40, 0.25, 0.15, 0.12, 0.08."""
    ranks = {"1": 400_000, "2": 250_000, "3": 150_000, "4": 120_000, "5": 80_000}
    assert_sums_to_one(ranks, label="rank weights")


def test_a_weight_set_that_nearly_sums_to_one_is_refused():
    """No tolerance, deliberately.

    'Nearly' is a silent rescaling of every score computed with the set — a defect that
    changes rankings without changing anything visible.
    """
    nearly = dict(CRITERIA)
    nearly["originality"] -= 1
    with pytest.raises(FixedPointError, match=r"off by -1"):
        assert_sums_to_one(nearly, label="criterion weights")


def test_a_negative_weight_is_refused():
    with pytest.raises(FixedPointError, match="negative weights"):
        assert_sums_to_one({"a": 1_200_000, "b": -200_000}, label="w")


# --------------------------------------------------------------------------
# Rounding direction: floor, and why
# --------------------------------------------------------------------------


def test_conversion_floors_rather_than_rounds():
    """Every value here is a share of something finite.

    Flooring means shares can never sum to more than the whole, so a rounding artefact can
    only under-allocate — recoverable by redistribution — and never over-allocate.
    """
    assert to_ppm(1, 3) == 333_333
    assert to_ppm(2, 3) == 666_666
    assert to_ppm(1, 3) + to_ppm(1, 3) + to_ppm(1, 3) < PPM


@given(st.integers(min_value=1, max_value=10_000), st.integers(min_value=1, max_value=10_000))
def test_a_share_never_exceeds_the_whole(numerator, denominator):
    if numerator <= denominator:
        assert to_ppm(numerator, denominator) <= PPM


def test_a_float_input_is_refused_rather_than_accepted_for_convenience():
    """Accepting one 'for convenience' is how the first float enters a codebase that had none."""
    with pytest.raises(FixedPointError, match="A float would reintroduce"):
        to_ppm(1.0, 3)  # type: ignore[arg-type]


def test_a_boolean_is_not_a_ratio():
    with pytest.raises(FixedPointError, match="booleans are not ratios"):
        to_ppm(True, 2)  # type: ignore[arg-type]


def test_a_zero_denominator_is_refused():
    with pytest.raises(FixedPointError, match="denominator is zero"):
        to_ppm(1, 0)


# --------------------------------------------------------------------------
# The weighted sum: one division, and omission is not zero
# --------------------------------------------------------------------------


def test_a_full_criterion_set_weights_correctly():
    values = dict.fromkeys(CRITERIA, 800_000)
    assert apply_weights(values, CRITERIA) == 800_000


def test_dividing_once_rather_than_per_term_avoids_a_systematic_bias():
    """The reason `apply_weights` sums numerators before dividing.

    Per-term division floors each contribution and loses up to one unit per criterion. With
    eight criteria that is a downward bias identical for everyone — invisible in a ranking,
    right up to the point it decides a qualification floor.
    """
    values = {name: 333_333 for name in CRITERIA}
    once = apply_weights(values, CRITERIA)
    per_term = sum(values[n] * CRITERIA[n] // PPM for n in CRITERIA)
    assert once >= per_term
    assert once == 333_333


def test_an_omitted_criterion_has_its_weight_redistributed():
    """Omission is how 'no judge could score this' is expressed."""
    full = dict.fromkeys(CRITERIA, 600_000)
    without = {name: 600_000 for name in CRITERIA if name != "originality"}
    assert apply_weights(without, CRITERIA) == apply_weights(full, CRITERIA) == 600_000


def test_omission_and_zero_are_not_the_same_thing():
    """The distinction that protects a miner from a judge outage.

    Writing zero for an unscoreable criterion docks its full weight: losing originality at
    25% would cost a quarter of the score for a reason that was never the miner's.
    """
    omitted = {name: 800_000 for name in CRITERIA if name != "originality"}
    zeroed = {**{name: 800_000 for name in CRITERIA}, "originality": 0}

    assert apply_weights(omitted, CRITERIA) == 800_000
    assert apply_weights(zeroed, CRITERIA) == 600_000  # a quarter of the score gone
    assert apply_weights(omitted, CRITERIA) > apply_weights(zeroed, CRITERIA)


def test_an_empty_value_set_raises_rather_than_returning_zero():
    """Returning zero would be indistinguishable from a genuinely zero score."""
    with pytest.raises(FixedPointError, match="cannot be computed from nothing"):
        apply_weights({}, CRITERIA)


def test_a_value_with_no_declared_weight_is_ignored():
    """A criterion the season does not weight cannot contribute to a score."""
    values = {**dict.fromkeys(CRITERIA, 500_000), "invented_criterion": 1_000_000}
    assert apply_weights(values, CRITERIA) == 500_000


# --------------------------------------------------------------------------
# The lower quartile: architecture.md 18.5's 30% component
# --------------------------------------------------------------------------


def test_the_lower_quartile_penalises_one_brilliant_result():
    """Exactly what 18.5 weights at 30%.

    Two laboratories with the same mean, one consistent and one spiky. The quartile separates
    them; the mean cannot.
    """
    consistent = [600_000] * 8
    spiky = [1_000_000, 1_000_000, 400_000, 400_000, 400_000, 400_000, 400_000, 800_000]
    assert mean_ppm(consistent) == mean_ppm(spiky) == 600_000
    assert quantile_ppm(consistent, ppm=250_000) > quantile_ppm(spiky, ppm=250_000)


def test_the_quantile_interpolates_rather_than_jumping_by_rank():
    """Nearest-rank would make a daily score depend on how many challenges were valid.

    Interpolation means adding one result shifts the quartile smoothly rather than stepping.
    """
    assert quantile_ppm([0, 100_000], ppm=500_000) == 50_000
    assert quantile_ppm([0, 100_000, 200_000, 300_000], ppm=250_000) == 75_000


def test_the_quantile_of_one_value_is_that_value():
    assert quantile_ppm([420_000], ppm=250_000) == 420_000


def test_the_quantile_is_independent_of_input_order():
    unordered = [800_000, 100_000, 500_000, 300_000]
    assert quantile_ppm(unordered, ppm=250_000) == quantile_ppm(sorted(unordered), ppm=250_000)


@given(st.lists(st.integers(min_value=0, max_value=PPM), min_size=1, max_size=30))
def test_the_quantile_always_lies_within_the_sample_range(values):
    result = quantile_ppm(values, ppm=250_000)
    assert min(values) <= result <= max(values)


def test_an_empty_sequence_raises():
    with pytest.raises(FixedPointError, match="empty sequence"):
        quantile_ppm([], ppm=250_000)
    with pytest.raises(FixedPointError, match="empty sequence"):
        mean_ppm([])


def test_a_quantile_position_outside_the_range_is_refused():
    with pytest.raises(FixedPointError, match="outside"):
        quantile_ppm([1, 2], ppm=PPM + 1)


# --------------------------------------------------------------------------
# Clamping
# --------------------------------------------------------------------------


def test_clamping_bounds_an_overshooting_estimator():
    """A Bradley-Terry score normalised against a shifting field can overshoot.

    Clamped rather than refused: the overshoot is a property of the estimator, and raising
    would strand a round on a rounding artefact.
    """
    assert clamp_ppm(1_200_000) == PPM
    assert clamp_ppm(-50) == 0
    assert clamp_ppm(500_000) == 500_000


def test_multiplication_by_a_ratio_floors():
    assert mul_ppm(1000, 333_333) == 333
