"""The scoring model: architecture.md 18.

Most of these tests measure a *difference* rather than asserting a value, because the mistakes
that matter here all produce a plausible number. A wrong ordering, a `.get(name, 0)`, a per-term
division — each yields a score that looks like a score.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from protocol.fixedpoint import PPM
from validator.scoring.criteria import (
    CriterionInputs,
    ScoringConfig,
    ScoringError,
    challenge_score,
    collapse_duplicates,
    combine_pairwise_pointwise,
    rank_weighted,
)
from validator.scoring.daily import (
    DailyConfig,
    DailyScoreError,
    ScoreHistory,
    daily_score,
    rolling_score,
)

pytestmark = pytest.mark.determinism

SEASON = json.loads(pathlib.Path("config/season.example.json").read_text())
CONFIG = ScoringConfig.from_season(SEASON)
CADENCE = DailyConfig.from_season(SEASON)

ALL_CRITERIA = tuple(SEASON["criterion_weights_ppm"])


def measured(**over: int | None) -> dict[str, CriterionInputs]:
    """Every criterion measured at 700000 unless overridden. `None` means unmeasured."""
    inputs = {name: CriterionInputs(700_000, 700_000) for name in ALL_CRITERIA}
    for name, value in over.items():
        inputs[name] = (
            CriterionInputs(None, None) if value is None else CriterionInputs(value, value)
        )
    return inputs


# --------------------------------------------------------------------------
# The shipped config is a valid scoring configuration
# --------------------------------------------------------------------------


def test_the_reference_season_produces_a_valid_scoring_config():
    assert CONFIG.pairwise_weight_ppm == 750_000
    assert CONFIG.pointwise_weight_ppm == 250_000
    assert CONFIG.mechanism_floor_ppm == 400_000


def test_inverting_the_pairwise_split_is_refused():
    """18.3 makes pairwise primary. Inverting it is legal arithmetic and the wrong mechanism.

    Pointwise rewards fluency, which is the failure pairwise exists to avoid.
    """
    with pytest.raises(ScoringError, match="would reward fluency"):
        ScoringConfig(
            criterion_weights_ppm=SEASON["criterion_weights_ppm"],
            rank_weights_ppm=SEASON["rank_weights_ppm"],
            pairwise_weight_ppm=250_000,
            pointwise_weight_ppm=750_000,
            mechanism_floor_ppm=400_000,
            capped_on_weak_mechanism_ppm=500_000,
        )


def test_a_weight_set_that_does_not_sum_to_one_is_refused_at_construction():
    broken = dict(SEASON["criterion_weights_ppm"])
    broken["value"] += 1
    with pytest.raises(Exception, match="sums to"):
        ScoringConfig(
            criterion_weights_ppm=broken,
            rank_weights_ppm=SEASON["rank_weights_ppm"],
            pairwise_weight_ppm=750_000,
            pointwise_weight_ppm=250_000,
            mechanism_floor_ppm=400_000,
            capped_on_weak_mechanism_ppm=500_000,
        )


# --------------------------------------------------------------------------
# 18.1: duplicates collapse before rank weighting
# --------------------------------------------------------------------------


def test_padding_a_portfolio_with_restatements_lowers_it():
    """The property collapsing first delivers, measured.

    Collapsing after would let a laboratory submit its best idea five times: each copy scores
    well, the rank weights all land on the same idea, and a one-idea portfolio earns a
    five-idea score.
    """
    genuine = collapse_duplicates(
        [900_000, 800_000, 700_000, 600_000, 500_000], lineages=[1, 2, 3, 4, 5]
    )
    padded = collapse_duplicates(
        [900_000, 900_000, 900_000, 900_000, 900_000], lineages=[1, 1, 1, 1, 1]
    )
    assert rank_weighted(genuine, CONFIG.rank_weights_ppm) > rank_weighted(
        padded, CONFIG.rank_weights_ppm
    )


def test_a_collapsed_lineage_keeps_its_best_member_not_its_mean():
    """One strong idea restated weakly is still one strong idea.

    Averaging would rank it below a laboratory that produced the same idea once — punishing the
    quality rather than removing the credit for repetition.
    """
    assert collapse_duplicates([900_000, 100_000], lineages=[1, 1]) == [900_000]


def test_the_collapse_returns_descending_order():
    """Rank weights applied to an unsorted list would credit position rather than quality.

    A collapse can leave the laboratory's own order no longer descending.
    """
    assert collapse_duplicates([100_000, 900_000, 500_000], lineages=[1, 2, 3]) == [
        900_000, 500_000, 100_000
    ]


def test_mismatched_lineage_labels_are_refused():
    with pytest.raises(ScoringError, match="lineage labels"):
        collapse_duplicates([1, 2, 3], lineages=[1, 2])


def test_the_rank_weights_favour_the_first_idea():
    """0.40 on rank one, per 18.1."""
    strong_first = rank_weighted([900_000, 100_000], CONFIG.rank_weights_ppm)
    strong_second = rank_weighted([100_000, 900_000], CONFIG.rank_weights_ppm)
    assert strong_first > strong_second


def test_a_missing_rank_forfeits_its_weight_rather_than_redistributing():
    """The miner's gap, not the validator's — so the weight is forfeit.

    Redistributing here — on the reasoning that the duplicate collapse already removed the credit
    for repetition — inverts the incentive: see the padding test above. A rank with no distinct idea
    behind it means the challenge's `portfolio_size` was not met.
    """
    one_idea = rank_weighted([800_000], CONFIG.rank_weights_ppm)
    assert one_idea == 320_000  # 0.40 x 800000, the remaining 60% forfeit


def test_redistribution_applies_to_an_unmeasured_criterion_but_not_a_missing_rank():
    """The distinction the two look identical without.

    A criterion no judge could score is the validator's gap and redistributes. A rank with no
    idea behind it is the miner's gap and is forfeit.
    """
    # Validator's gap: full score preserved.
    assert challenge_score(measured(originality=None), CONFIG).total_ppm == 700_000
    # Miner's gap: weight forfeit.
    assert rank_weighted([700_000], CONFIG.rank_weights_ppm) < 700_000


def test_extra_ideas_beyond_the_requested_size_earn_nothing():
    """So a laboratory cannot improve its score with a long tail of extras."""
    five = rank_weighted([900_000] * 5, CONFIG.rank_weights_ppm)
    ten = rank_weighted([900_000] * 10, CONFIG.rank_weights_ppm)
    assert five == ten == 900_000


def test_an_empty_portfolio_is_refused():
    with pytest.raises(ScoringError, match="no ideas has no score"):
        rank_weighted([], CONFIG.rank_weights_ppm)


# --------------------------------------------------------------------------
# 18.3: the pairwise/pointwise combination
# --------------------------------------------------------------------------


def test_the_combination_weights_pairwise_at_three_quarters():
    combined = combine_pairwise_pointwise(CriterionInputs(1_000_000, 0), CONFIG)
    assert combined == 750_000


def test_a_criterion_with_only_a_pointwise_result_carries_full_weight():
    """Scaling it by its own 0.25 share would cap it at a quarter of what it earned.

    That is indistinguishable from having scored badly, when one measurement was simply
    unavailable.
    """
    assert combine_pairwise_pointwise(CriterionInputs(None, 800_000), CONFIG) == 800_000


def test_a_criterion_with_only_a_pairwise_result_carries_full_weight():
    assert combine_pairwise_pointwise(CriterionInputs(800_000, None), CONFIG) == 800_000


def test_a_criterion_with_neither_measurement_is_none_not_zero():
    """`None` is what makes the weight redistribute instead of docking the miner."""
    assert combine_pairwise_pointwise(CriterionInputs(None, None), CONFIG) is None


# --------------------------------------------------------------------------
# 18.4: the mechanism floor
# --------------------------------------------------------------------------


def test_a_weak_mechanism_caps_value_and_originality():
    """"An idea cannot score highly merely by sounding unusual.\""""
    inputs = measured(mechanism=300_000, value=900_000, originality=900_000)
    result = challenge_score(inputs, CONFIG)
    assert result.mechanism_floor_applied is True
    assert result.criteria_ppm["value"] == 500_000
    assert result.criteria_ppm["originality"] == 500_000


def test_the_floor_leaves_other_criteria_alone():
    """The floor claims a mechanism is needed for *value and originality*.

    It says nothing about whether constraints were met or the portfolio was diverse, and
    capping those would punish work it makes no claim about.
    """
    result = challenge_score(measured(mechanism=300_000, constraint_fit=900_000), CONFIG)
    assert result.criteria_ppm["constraint_fit"] == 900_000


def test_a_strong_mechanism_does_not_trigger_the_floor():
    result = challenge_score(measured(mechanism=500_000, value=900_000), CONFIG)
    assert result.mechanism_floor_applied is False
    assert result.criteria_ppm["value"] == 900_000


def test_an_unmeasured_mechanism_does_not_trigger_the_floor():
    """An unmeasured mechanism is not a weak one.

    Capping on absence would punish a laboratory for a judge outage.
    """
    result = challenge_score(measured(mechanism=None, value=900_000), CONFIG)
    assert result.mechanism_floor_applied is False
    assert result.criteria_ppm["value"] == 900_000


def test_the_floor_is_applied_after_the_combination_not_before():
    """Applied earlier, the cap would depend on which input was consulted first.

    Here pointwise mechanism is weak and pairwise is strong. Combined, mechanism clears the
    floor, so no cap applies — which is the correct reading of a criterion whose primary
    signal was strong.
    """
    inputs = measured()
    inputs["mechanism"] = CriterionInputs(pairwise_ppm=600_000, pointwise_ppm=100_000)
    combined = combine_pairwise_pointwise(inputs["mechanism"], CONFIG)
    assert combined >= CONFIG.mechanism_floor_ppm
    assert challenge_score(inputs, CONFIG).mechanism_floor_applied is False


# --------------------------------------------------------------------------
# Omission versus zero
# --------------------------------------------------------------------------


def test_an_unmeasured_criterion_has_its_weight_redistributed():
    full = challenge_score(measured(), CONFIG)
    without = challenge_score(measured(originality=None), CONFIG)
    assert full.total_ppm == without.total_ppm == 700_000
    assert "originality" in without.omitted_criteria


def test_omitting_originality_costs_nothing_where_zeroing_it_costs_a_quarter():
    """`criteria.get(name, 0)` reads as a harmless default and is worth 25% of a score.

    Measured rather than asserted, because the two code paths look identical.
    """
    omitted = challenge_score(measured(originality=None), CONFIG).total_ppm
    zeroed = challenge_score(measured(originality=0), CONFIG).total_ppm
    assert omitted == 700_000
    assert zeroed == 525_000
    assert omitted - zeroed == 175_000


def test_a_challenge_with_no_scoreable_criterion_is_refused():
    """Zero would be indistinguishable from a portfolio scored and found worthless."""
    nothing = {name: CriterionInputs(None, None) for name in ALL_CRITERIA}
    with pytest.raises(ScoringError, match="indistinguishable"):
        challenge_score(nothing, CONFIG)


def test_a_criterion_the_season_does_not_weight_cannot_contribute():
    inputs = {**measured(), "invented": CriterionInputs(1_000_000, 1_000_000)}
    assert challenge_score(inputs, CONFIG).total_ppm == 700_000


def test_the_working_is_kept_so_a_published_score_can_be_checked():
    """22 publishes challenge scores. One nobody can decompose is trusted, not checked."""
    result = challenge_score(measured(mechanism=300_000), CONFIG)
    assert set(result.criteria_ppm) == set(ALL_CRITERIA)
    assert result.mechanism_floor_applied is True


# --------------------------------------------------------------------------
# 18.5: the daily score and its lower quartile
# --------------------------------------------------------------------------


def test_the_quartile_separates_a_spiky_laboratory_from_a_consistent_one():
    """Exactly what 18.5 weights at 30%.

    Twenty challenges a day makes this measurable: a laboratory that spikes on the two problems
    matching its house style has the same mean as a consistent one.
    """
    consistent = [600_000] * 20
    spiky = [1_000_000] * 5 + [466_666] * 15

    steady = daily_score(consistent, CADENCE)
    volatile = daily_score(spiky, CADENCE)
    assert steady.mean_ppm == pytest.approx(volatile.mean_ppm, abs=1_000)
    assert steady.score_ppm > volatile.score_ppm


def test_both_components_are_reported_separately():
    """The gap between them *is* the diagnosis: inconsistency, not weakness."""
    result = daily_score([1_000_000, 200_000, 200_000, 200_000], CADENCE)
    assert result.mean_ppm > result.lower_quartile_ppm


def test_a_day_below_the_minimum_reports_rather_than_vanishing():
    """A partial day has a real score; whether it counts is the allocator's decision.

    Deciding it here would make the day silently disappear rather than being excluded on the
    record.
    """
    partial = daily_score([800_000, 800_000], CADENCE)
    assert partial.qualifies is False
    assert partial.score_ppm > 0
    assert partial.valid_challenges == 2


def test_a_full_day_qualifies():
    assert daily_score([700_000] * 20, CADENCE).qualifies is True


def test_a_day_with_no_valid_challenges_is_refused():
    with pytest.raises(DailyScoreError, match="indistinguishable"):
        daily_score([], CADENCE)


# --------------------------------------------------------------------------
# 18.6: the rolling score must not suppress newcomers
# --------------------------------------------------------------------------


def test_a_newcomer_with_perfect_days_scores_perfectly():
    """The protocol forbids a credibility multiplier, and the estimator switch must not become one.

    Such a multiplier makes a new coldkey worth almost nothing for several seasons. A laboratory
    with three excellent days scores what those days earned.
    """
    history = ScoreHistory(dates=["2026-08-01", "2026-08-02", "2026-08-03"], scores_ppm=[PPM] * 3)
    assert rolling_score(history, CADENCE).score_ppm == PPM


def test_a_newcomer_and_a_veteran_with_identical_scores_rank_identically():
    """The estimator changes; the magnitude is never discounted for youth."""
    newcomer = ScoreHistory(
        dates=[f"2026-08-0{d}" for d in range(1, 4)], scores_ppm=[800_000] * 3
    )
    veteran = ScoreHistory(
        dates=[f"2026-08-{d:02d}" for d in range(1, 15)], scores_ppm=[800_000] * 14
    )
    assert rolling_score(newcomer, CADENCE).score_ppm == rolling_score(veteran, CADENCE).score_ppm


def test_below_the_threshold_the_estimator_is_the_plain_mean():
    history = ScoreHistory(dates=["2026-08-01", "2026-08-02"], scores_ppm=[600_000, 800_000])
    result = rolling_score(history, CADENCE)
    assert result.score_ppm == 700_000
    assert result.estimator == "mean of 2 day(s)"


def test_at_the_threshold_the_estimator_becomes_the_blended_median():
    history = ScoreHistory(
        dates=[f"2026-08-{d:02d}" for d in range(1, 8)], scores_ppm=[700_000] * 7
    )
    result = rolling_score(history, CADENCE)
    assert "median" in result.estimator
    assert result.score_ppm == 700_000


def test_the_median_absorbs_one_catastrophic_day():
    """Robust rather than suppressive: a validator outage cannot erase a good week."""
    good = [800_000] * 7
    with_outage = [800_000] * 6 + [0]
    dates = [f"2026-08-{d:02d}" for d in range(1, 8)]

    clean = rolling_score(ScoreHistory(dates=dates, scores_ppm=good), CADENCE)
    damaged = rolling_score(ScoreHistory(dates=dates, scores_ppm=with_outage), CADENCE)
    assert damaged.score_ppm == clean.score_ppm


def test_recent_performance_outweighs_older_performance():
    """0.60 on the 7-day median against 0.40 on the 30-day."""
    dates = [f"2026-08-{d:02d}" for d in range(1, 15)]
    improving = ScoreHistory(dates=dates, scores_ppm=[200_000] * 7 + [900_000] * 7)
    declining = ScoreHistory(dates=dates, scores_ppm=[900_000] * 7 + [200_000] * 7)
    assert rolling_score(improving, CADENCE).score_ppm > rolling_score(declining, CADENCE).score_ppm


def test_an_unsorted_history_is_refused():
    """A window from an unsorted history selects an arbitrary set of days."""
    with pytest.raises(DailyScoreError, match="ascending date order"):
        ScoreHistory(dates=["2026-08-03", "2026-08-01"], scores_ppm=[1, 2])


def test_a_history_with_no_days_is_refused():
    """No results is different from a score of zero: the allocator excludes rather than ranks."""
    with pytest.raises(DailyScoreError, match="different from"):
        rolling_score(ScoreHistory(dates=[], scores_ppm=[]), CADENCE)


def test_a_short_window_longer_than_the_long_one_is_refused():
    with pytest.raises(DailyScoreError, match="measure the same thing"):
        DailyConfig(
            daily_mean_weight_ppm=700_000,
            daily_q25_weight_ppm=300_000,
            rolling_short_days=30,
            rolling_long_days=7,
            rolling_short_weight_ppm=600_000,
            rolling_long_weight_ppm=400_000,
            minimum_days_for_median=7,
            minimum_valid_challenges=6,
        )
