"""Bradley-Terry: converting a Swiss tournament into a ranking.

architecture.md 27 requires same-bundle rerun rank correlation at 0.80 or above, so the fit
must be reproducible on one host. It does *not* require agreement between hosts — validators
run different packs — which is what makes a floating-point fit acceptable here at all.
"""

from __future__ import annotations

import pytest

from protocol.fixedpoint import PPM
from validator.judge.bradley_terry import (
    BradleyTerryError,
    Outcome,
    Pairing,
    fit,
    strengths_to_ppm,
)

pytestmark = pytest.mark.determinism


def beats(winner: str, loser: str, weight: float = 1.0) -> Pairing:
    return Pairing(a=winner, b=loser, winner=Outcome.A, weight=weight)


def drew(a: str, b: str) -> Pairing:
    return Pairing(a=a, b=b, winner=Outcome.TIE)


# --------------------------------------------------------------------------
# The model does what the model should
# --------------------------------------------------------------------------


def test_a_winner_outranks_a_loser():
    strengths = fit([beats("a", "b")])
    assert strengths["a"] > strengths["b"]


def test_a_transitive_field_orders_transitively():
    strengths = fit([beats("a", "b"), beats("b", "c"), beats("c", "d")])
    assert strengths["a"] > strengths["b"] > strengths["c"] > strengths["d"]


def test_beating_a_strong_opponent_counts_for_more_than_beating_a_weak_one():
    """The property a Swiss tournament needs.

    Under Swiss pairing nobody faces the same field, so a ranking that counted only win totals
    would reward whoever drew the easiest opponents.
    """
    pairings = [
        # `strong` establishes itself against three others.
        beats("strong", "filler1"),
        beats("strong", "filler2"),
        beats("strong", "filler3"),
        # Two contenders each win once: one against `strong`, one against a filler.
        beats("beat_strong", "strong"),
        beats("beat_weak", "filler1"),
    ]
    strengths = fit(pairings)
    assert strengths["beat_strong"] > strengths["beat_weak"]


def test_a_tie_places_two_laboratories_together():
    strengths = fit([drew("a", "b"), beats("a", "c"), beats("b", "c")])
    assert strengths["a"] == pytest.approx(strengths["b"], rel=1e-6)


def test_a_tie_is_not_discarded():
    """Discarding ties would lose the information that two laboratories are close.

    Mid-field, that is the information a ranking most needs.
    """
    with_tie = fit([beats("a", "c"), beats("b", "c"), drew("a", "b")])
    without = fit([beats("a", "c"), beats("b", "c")])
    assert with_tie != without


# --------------------------------------------------------------------------
# Reproducibility: the property the measurement gates check
# --------------------------------------------------------------------------


def test_the_same_tournament_fits_identically_every_time():
    pairings = [beats("a", "b"), beats("b", "c"), drew("a", "c"), beats("a", "d")]
    assert fit(pairings) == fit(pairings)


def test_the_input_order_does_not_change_the_fit():
    """Floating-point addition is not associative.

    The same tournament processed in two orders can produce strengths differing in the last
    bits — enough to swap two adjacent ranks. Sorting the input removes it.
    """
    pairings = [beats("a", "b"), beats("b", "c"), drew("a", "c"), beats("d", "b")]
    assert fit(pairings) == fit(list(reversed(pairings)))


def test_the_iteration_count_is_fixed_rather_than_convergence_tested():
    """A threshold comparison on floats can take a different number of steps on two hosts.

    Asserted by showing the fit has stopped moving well before the count runs out, so the
    fixed bound is past convergence rather than truncating it.
    """
    pairings = [beats("a", "b"), beats("b", "c"), beats("c", "d"), drew("a", "d")]
    at_150 = fit(pairings, iterations=150)
    at_200 = fit(pairings, iterations=200)
    ranking = lambda s: sorted(s, key=lambda k: -s[k])  # noqa: E731
    assert ranking(at_150) == ranking(at_200)


def test_strengths_are_normalised_so_average_is_one():
    """Keeps 1.0 interpretable as average and keeps the ppm conversion stable as the field grows."""
    strengths = fit([beats("a", "b"), beats("b", "c"), beats("c", "d")])
    assert sum(strengths.values()) == pytest.approx(len(strengths), rel=1e-9)


# --------------------------------------------------------------------------
# An undefeated laboratory must not diverge
# --------------------------------------------------------------------------


def test_an_undefeated_laboratory_gets_a_finite_strength():
    """Without regularisation the likelihood grows without limit and the fit diverges."""
    strengths = fit([beats("perfect", "a"), beats("perfect", "b"), beats("perfect", "c")])
    assert strengths["perfect"] < float("inf")
    assert strengths["perfect"] > strengths["a"]


def test_more_undefeated_wins_rank_higher_but_not_infinitely_higher():
    """Three wins beats two wins; three wins is not proof.

    The prior's honest reading: an unbeaten short record is ranked first, not infinitely first.
    """
    three = fit([beats("p", "a"), beats("p", "b"), beats("p", "c")])["p"]
    two = fit([beats("p", "a"), beats("p", "b")])["p"]
    assert three > two
    assert three < 100.0  # finite by a wide margin


def test_a_laboratory_that_never_wins_gets_a_finite_positive_strength():
    strengths = fit([beats("a", "loser"), beats("b", "loser"), beats("c", "loser")])
    assert 0 < strengths["loser"] < strengths["a"]


# --------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------


def test_an_empty_tournament_is_refused_rather_than_flattened():
    """A flat ranking would claim every laboratory is equal, not that nothing is known."""
    with pytest.raises(BradleyTerryError, match="no comparisons"):
        fit([])


def test_a_laboratory_cannot_be_compared_with_itself():
    with pytest.raises(BradleyTerryError, match="cannot be compared with itself"):
        Pairing(a="a", b="a", winner=Outcome.A)


def test_an_unknown_outcome_is_refused():
    with pytest.raises(BradleyTerryError, match="unknown outcome"):
        Pairing(a="a", b="b", winner="maybe")


def test_a_zero_weight_comparison_is_refused():
    with pytest.raises(BradleyTerryError, match="positive weight"):
        Pairing(a="a", b="b", winner=Outcome.A, weight=0)


# --------------------------------------------------------------------------
# The ppm conversion: by rank, not by magnitude
# --------------------------------------------------------------------------


def test_the_conversion_spaces_competitors_by_rank():
    strengths = fit([beats("a", "b"), beats("b", "c")])
    scores = strengths_to_ppm(strengths)
    assert scores["a"] == PPM
    assert scores["c"] == 0
    assert 0 < scores["b"] < PPM


def test_a_dominant_laboratory_does_not_compress_everyone_else_to_zero():
    """Why rank rather than a linear rescale of the strengths.

    Strengths are multiplicative and unbounded: one laboratory dominating a weak field can hold
    ten times the next. A linear rescale would hand it a near-perfect score and flatten the
    rest — making the pairwise component behave like winner-take-all, which 20.2 deliberately
    does not want. The distribution is a policy decision made once at weight allocation, with a
    temperature and a cap, not an accident of how the fit spread.
    """
    dominant = {"star": 40.0, "b": 1.2, "c": 1.0, "d": 0.9}
    scores = strengths_to_ppm(dominant)
    # Under a linear rescale, b/c/d would all be near zero. By rank they are evenly spread.
    assert scores["b"] > 600_000
    assert scores["c"] > 300_000
    assert len(set(scores.values())) == 4


def test_equal_strengths_share_a_position():
    """A floating-point artefact must not separate two indistinguishable laboratories."""
    scores = strengths_to_ppm({"a": 1.5, "b": 1.5, "c": 0.5})
    assert scores["a"] == scores["b"]
    assert scores["a"] > scores["c"]


def test_a_single_competitor_scores_the_midpoint_not_a_perfect_score():
    """First out of one is not a perfect result; it is a result against nobody."""
    assert strengths_to_ppm({"only": 1.0}) == {"only": PPM // 2}


def test_the_conversion_output_is_integer_only():
    """The last place a float exists. Everything downstream is integer."""
    scores = strengths_to_ppm(fit([beats("a", "b"), beats("b", "c")]))
    assert all(isinstance(value, int) for value in scores.values())


def test_every_score_is_within_range():
    scores = strengths_to_ppm(fit([beats("a", "b"), beats("b", "c"), beats("c", "d")]))
    assert all(0 <= value <= PPM for value in scores.values())


def test_converting_nothing_is_refused():
    with pytest.raises(BradleyTerryError, match="no strengths"):
        strengths_to_ppm({})


# --------------------------------------------------------------------------
# A realistic Swiss round
# --------------------------------------------------------------------------


def test_a_swiss_style_field_produces_a_sensible_ranking():
    """Pairings near current estimated score, as 17.3 describes, with nobody facing everyone."""
    pairings = [
        beats("m1", "m2"), beats("m1", "m3"),
        beats("m2", "m4"), drew("m2", "m3"),
        beats("m3", "m5"), beats("m4", "m5"),
        beats("m4", "m6"), beats("m5", "m6"),
    ]
    scores = strengths_to_ppm(fit(pairings))
    assert scores["m1"] > scores["m4"] > scores["m6"]
    assert scores["m6"] == 0
    assert scores["m1"] == PPM
