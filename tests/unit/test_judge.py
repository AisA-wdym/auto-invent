"""Judge panels, screening and the Swiss tournament: architecture.md 16-17.

Two properties get most of the attention here, because both fail silently:

* the **family cap** counted on families rather than routes — a slug-derived reading passes three
  Anthropic snapshots as three families and the cap becomes decorative;
* the **order swap**, which is not a refinement but the measurement that separates a preference for
  an answer from a preference for a slot.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from gateway.adapters.openrouter import ModelPin
from protocol.fixedpoint import PPM
from validator.judge.bradley_terry import fit
from validator.judge.pairwise import PairVerdict, combine_orders, swiss_pairings
from validator.judge.panels import (
    JUDGE_ROLES,
    Panel,
    PanelError,
    PanelJudge,
    assert_family_cap,
    panels_from_season,
    pins_for,
)
from validator.judge.pointwise import (
    ANCHOR_PPM,
    ANCHORS,
    PointwiseScore,
    aggregate,
    score_to_ppm,
)

pytestmark = pytest.mark.determinism

SEASON = json.loads(pathlib.Path("config/season.example.json").read_text())
SEED = bytes(range(32))


def judge(family: str, slug: str = "") -> PanelJudge:
    return PanelJudge(family=family, pin=ModelPin(slug=slug or f"{family}/model", snapshot="s"))


# --------------------------------------------------------------------------
# 16.1: the family cap, and the reading that would make it vacuous
# --------------------------------------------------------------------------


def test_the_example_season_declares_a_valid_panel_for_every_criterion():
    panels = panels_from_season(SEASON)
    assert set(panels) == set(JUDGE_ROLES)


def test_three_snapshots_of_one_family_breach_the_cap():
    """The reading that matters. Every judge is reached through OpenRouter, so a cap on the
    *routing* provider is a cap on nothing — three distinct Anthropic slugs is one family, because
    two versions of one model share their failure modes."""
    with pytest.raises(PanelError, match="Two snapshots of one family are one family"):
        assert_family_cap(
            "mechanism",
            [
                judge("anthropic", "anthropic/claude-sonnet-5"),
                judge("anthropic", "anthropic/claude-opus-5"),
                judge("anthropic", "anthropic/claude-haiku-4.5"),
            ],
            family_cap_ppm=400_000,
        )


def test_three_distinct_families_satisfy_a_forty_percent_cap():
    assert_family_cap(
        "mechanism",
        [judge("anthropic"), judge("openai"), judge("google")],
        family_cap_ppm=400_000,
    )


def test_two_of_four_votes_is_within_a_fifty_percent_cap_and_over_a_forty():
    """The boundary, measured in both directions."""
    panel = [judge("anthropic"), judge("anthropic"), judge("openai"), judge("google")]
    assert_family_cap("mechanism", panel, family_cap_ppm=500_000)
    with pytest.raises(PanelError):
        assert_family_cap("mechanism", panel, family_cap_ppm=400_000)


def test_a_panel_with_too_few_families_is_refused():
    """16.1 requires at least three; fewer means the criterion's failure modes are shared."""
    with pytest.raises(PanelError, match="requires at least"):
        Panel(
            criterion="mechanism",
            judges=(judge("anthropic"), judge("openai")),
            reliability_floor_ppm=900_000,
        ).assert_valid(minimum_families=3, family_cap_ppm=400_000)


def test_a_panel_with_no_judges_is_refused():
    with pytest.raises(PanelError, match="no judges"):
        assert_family_cap("mechanism", [], family_cap_ppm=400_000)


def test_a_judge_with_no_declared_family_is_refused():
    """The family field is required rather than derived: a miner-hosted fine-tune routed through
    OpenRouter has a slug that says nothing about what trained it."""
    with pytest.raises(PanelError, match="cannot be counted against"):
        PanelJudge(family="", pin=ModelPin("x/y", "s"))


def test_a_criterion_outside_16_2_is_refused():
    """A criterion with no declared role has no rubric, so each judge would invent one."""
    with pytest.raises(PanelError, match="not one of"):
        Panel(
            criterion="vibes",
            judges=(judge("anthropic"), judge("openai"), judge("google")),
            reliability_floor_ppm=900_000,
        ).assert_valid(minimum_families=3, family_cap_ppm=400_000)


def test_every_role_in_16_2_has_a_stated_question():
    """A judge told only which criterion to score invents its own rubric, and two judges then
    score different things."""
    assert len(JUDGE_ROLES) == 8
    for criterion, question in JUDGE_ROLES.items():
        assert len(question) > 60, f"{criterion} has no substantive question"


def test_a_family_pinned_to_two_different_routes_is_refused():
    """The cap treats them as one family, so two routes under one name would make the cap and the
    routing disagree about what a family is."""
    panels = {
        "mechanism": Panel(
            criterion="mechanism",
            judges=(judge("anthropic", "anthropic/a"), judge("openai"), judge("google")),
            reliability_floor_ppm=900_000,
        ),
        "value": Panel(
            criterion="value",
            judges=(judge("anthropic", "anthropic/b"), judge("openai"), judge("google")),
            reliability_floor_ppm=850_000,
        ),
    }
    with pytest.raises(PanelError, match="one family"):
        pins_for(panels)


def test_the_example_season_yields_one_pin_per_family():
    pins = pins_for(panels_from_season(SEASON))
    assert set(pins) == {"anthropic", "openai", "google"}


# --------------------------------------------------------------------------
# 17.1: anchored scoring
# --------------------------------------------------------------------------


def test_the_anchors_are_17_1s_five():
    assert list(ANCHORS) == [0, 1, 2, 3, 4]
    assert ANCHORS[0] == "absent or invalid"
    assert ANCHORS[4] == "unusually strong, coherent and differentiated"


def test_the_anchor_mapping_spans_the_full_range():
    assert ANCHOR_PPM[0] == 0
    assert ANCHOR_PPM[4] == PPM


def test_the_anchor_mapping_is_monotone():
    values = [ANCHOR_PPM[raw] for raw in sorted(ANCHOR_PPM)]
    assert values == sorted(values)


def test_a_score_off_the_scale_is_refused_rather_than_clamped():
    """Clamping 7 to 4 would award the top score to a judge that misread the rubric."""
    with pytest.raises(ValueError, match="not one of"):
        score_to_ppm(7)


def test_a_negative_score_is_refused():
    with pytest.raises(ValueError):
        score_to_ppm(-1)


def test_aggregation_averages_only_the_votes():
    scores = [
        PointwiseScore("mechanism", "anthropic", 4, PPM, "", False, 1),
        PointwiseScore("mechanism", "openai", 2, 500_000, "", False, 1),
        PointwiseScore("mechanism", "google", None, 0, "", True, 1),
    ]
    value, voters = aggregate(scores)
    assert voters == 2
    assert value == (PPM + 500_000) // 2


def test_an_abstention_does_not_pull_the_mean_toward_zero():
    """An abstention removes a vote; counting it as zero would be a fabricated score in the
    direction that looks harmless."""
    with_abstention = aggregate(
        [
            PointwiseScore("m", "a", 4, PPM, "", False, 1),
            PointwiseScore("m", "b", None, 0, "", True, 1),
        ]
    )
    without = aggregate([PointwiseScore("m", "a", 4, PPM, "", False, 1)])
    assert with_abstention == without


def test_no_voters_reports_zero_voters_rather_than_a_zero_score():
    """The caller must treat this as *unscored* so `apply_weights` redistributes; a zero score
    would cost the miner the criterion's full weight for a judge outage."""
    assert aggregate([]) == (0, 0)
    assert aggregate([PointwiseScore("m", "a", None, 0, "", True, 1)]) == (0, 0)


def test_an_unreadable_reply_and_an_abstention_are_both_non_votes():
    unreadable = PointwiseScore("m", "a", None, 0, "could not parse", False, 1)
    abstained = PointwiseScore("m", "b", None, 0, "nothing bears on this", True, 1)
    assert not unreadable.voted
    assert not abstained.voted


# --------------------------------------------------------------------------
# 17.3: Swiss pairing
# --------------------------------------------------------------------------


def standings(count: int) -> list[tuple[int, int]]:
    return [(uid, PPM - uid * 50_000) for uid in range(1, count + 1)]


def test_pairing_puts_near_equal_laboratories_together():
    """Where the information is: a comparison between the best and worst has a foregone
    conclusion and costs the same as an informative one."""
    pairs = swiss_pairings(standings(8), seed=SEED, round_number=1)
    for left, right in pairs:
        assert abs(left - right) <= 2, f"{left} vs {right} is not a near pairing"


def test_every_laboratory_in_an_even_field_is_paired():
    pairs = swiss_pairings(standings(8), seed=SEED, round_number=1)
    paired = {uid for pair in pairs for uid in pair}
    assert paired == set(range(1, 9))


def test_an_odd_field_gives_the_bye_to_the_middle_not_the_leader():
    """A bye is a free non-comparison. Given to the leader it would let the leader hold its
    position without being tested."""
    pairs = swiss_pairings(standings(9), seed=SEED, round_number=1)
    paired = {uid for pair in pairs for uid in pair}
    missing = set(range(1, 10)) - paired
    assert len(missing) == 1
    assert missing != {1}, "the leader must not receive the bye"


def test_pairing_is_deterministic_for_a_seed():
    first = swiss_pairings(standings(8), seed=SEED, round_number=1)
    second = swiss_pairings(standings(8), seed=SEED, round_number=1)
    assert first == second


def test_the_round_number_changes_the_seeded_stream():
    """Each round draws from a differently-labelled stream, so two rounds are not forced to pair
    identically. The *guarantee* that they differ is the repeat check below — this only asserts the
    round number reaches the stream, which a shared label would silently break."""
    from protocol.seeds import _seeded_stream

    first = next(_seeded_stream(SEED, b"swiss-round-1"))
    second = next(_seeded_stream(SEED, b"swiss-round-2"))
    assert first != second


def test_a_second_round_avoids_the_first_rounds_pairings():
    """"Repeated identical pairings are limited"."""
    first = swiss_pairings(standings(8), seed=SEED, round_number=1)
    second = swiss_pairings(
        standings(8), seed=SEED, round_number=2, already_paired=[frozenset(p) for p in first]
    )
    assert not ({frozenset(p) for p in first} & {frozenset(p) for p in second})


def test_a_field_of_one_produces_no_pairings():
    assert swiss_pairings([(1, PPM)], seed=SEED, round_number=1) == []


def test_an_empty_field_produces_no_pairings():
    assert swiss_pairings([], seed=SEED, round_number=1) == []


def test_pairing_does_not_depend_on_the_input_order():
    """Standings arrive from a dict in production, so iteration order must not change pairing."""
    forward = swiss_pairings(standings(8), seed=SEED, round_number=1)
    backward = swiss_pairings(list(reversed(standings(8))), seed=SEED, round_number=1)
    assert {frozenset(p) for p in forward} == {frozenset(p) for p in backward}


# --------------------------------------------------------------------------
# The order swap: the measurement, not a refinement
# --------------------------------------------------------------------------


def verdict(slot_a: int, slot_b: int, winner: str, **over) -> PairVerdict:
    fields = dict(
        criterion="mechanism",
        family="anthropic",
        slot_a=slot_a,
        slot_b=slot_b,
        winner=winner,
        confidence_ppm=800_000,
        decisive_reason="the mechanism is explicit",
        abstained=False,
        rcc=10,
    )
    fields.update(over)
    return PairVerdict(**fields)


def test_two_orders_agreeing_produce_a_win():
    """Judge picks uid 1 whichever slot it is in: a preference for the answer."""
    pairings, inconsistency = combine_orders([verdict(1, 2, "A"), verdict(2, 1, "B")])
    assert len(pairings) == 1
    assert pairings[0].winner == "A"
    assert pairings[0].a == "1"
    assert inconsistency == 0


def test_two_orders_disagreeing_produce_a_tie():
    """Judge picks whatever is in slot A: a preference for the *position*. Recorded as a tie
    rather than discarded, because discarding would delete the near-equal pairs Swiss pairing
    exists to produce."""
    pairings, inconsistency = combine_orders([verdict(1, 2, "A"), verdict(2, 1, "A")])
    assert pairings[0].winner == "tie"
    assert inconsistency == PPM


def test_a_declared_tie_in_both_orders_is_a_tie():
    pairings, inconsistency = combine_orders([verdict(1, 2, "tie"), verdict(2, 1, "tie")])
    assert pairings[0].winner == "tie"
    assert inconsistency == 0, "agreeing on a tie is not a position preference"


def test_the_inconsistency_rate_is_measured_across_comparisons():
    """19 compares this against `order_swap_inconsistency_ceiling_ppm`. It is a measurement of the
    panel, not a nuisance to minimise away."""
    consistent = [verdict(1, 2, "A"), verdict(2, 1, "B")]
    biased = [verdict(3, 4, "A"), verdict(4, 3, "A")]
    _pairings, inconsistency = combine_orders([*consistent, *biased])
    assert inconsistency == PPM // 2


def test_an_abstention_in_one_order_does_not_count_as_a_disagreement():
    """One presentation cannot show a position preference, so it contributes no bias measurement."""
    pairings, inconsistency = combine_orders(
        [verdict(1, 2, "A"), verdict(2, 1, "tie", abstained=True)]
    )
    assert len(pairings) == 1
    assert pairings[0].winner == "A"
    assert inconsistency == 0


def test_a_pair_where_both_orders_abstained_yields_no_pairing():
    """No comparison happened, so nothing enters the fit."""
    pairings, _ = combine_orders(
        [
            verdict(1, 2, "tie", abstained=True),
            verdict(2, 1, "tie", abstained=True),
        ]
    )
    assert pairings == []


def test_verdicts_from_different_judges_are_combined_separately():
    """A panel where one family has a position bias and another has none must not average to a
    moderate bias: 19 removes the one judge rather than distrusting the panel."""
    pairings, inconsistency = combine_orders(
        [
            verdict(1, 2, "A", family="anthropic"),
            verdict(2, 1, "B", family="anthropic"),
            verdict(1, 2, "A", family="openai"),
            verdict(2, 1, "A", family="openai"),
        ]
    )
    assert len(pairings) == 2
    assert inconsistency == PPM // 2


def test_combined_pairings_feed_the_bradley_terry_fit():
    """The end of the pipeline: verdicts become a ranking."""
    pairings, _ = combine_orders(
        [verdict(1, 2, "A"), verdict(2, 1, "B"), verdict(1, 2, "A", family="openai"),
         verdict(2, 1, "B", family="openai")]
    )
    strengths = fit(pairings)
    assert strengths["1"] > strengths["2"]


def test_the_winning_uid_is_read_through_the_slot_mapping():
    """Without recording which uid sat in which slot, combining two orders would be guesswork."""
    assert verdict(7, 9, "A").winning_uid() == 7
    assert verdict(7, 9, "B").winning_uid() == 9
    assert verdict(7, 9, "tie").winning_uid() is None
    assert verdict(7, 9, "A", abstained=True).winning_uid() is None
