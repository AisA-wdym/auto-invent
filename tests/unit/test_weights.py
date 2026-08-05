"""Weight allocation: architecture.md 20.

The last stage that can change emissions, so the tests are mostly about the two outcomes that
must remain possible — paying nobody, and refusing to let one laboratory take everything.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from protocol.fixedpoint import PPM
from validator.weights import Candidate, WeightsConfig, WeightsError, allocate

pytestmark = pytest.mark.determinism

SEASON = json.loads(pathlib.Path("config/season.example.json").read_text())
CONFIG = WeightsConfig.from_season(SEASON)
FLOOR = 500_000  # the reference laboratory's own rolling score


def candidate(uid: int, score: int, **over) -> Candidate:
    kwargs = dict(
        uid=uid,
        rolling_score_ppm=score,
        valid_challenges=20,
        hard_gates_passed=True,
        artifacts_available=True,
    )
    kwargs.update(over)
    return Candidate(**kwargs)


# --------------------------------------------------------------------------
# 20.1: the floor is absolute, so paying nobody must be possible
# --------------------------------------------------------------------------


def test_a_field_entirely_below_the_floor_burns_everything():
    """"Being best among weak miners is not enough."

    A rank-based floor would always pay someone, which is exactly what 20.4 refuses: if no
    laboratory beats direct frontier-model use, the subnet has not shown that competing
    architectures add value, and emitting anyway would pay for that.
    """
    result = allocate(
        [candidate(1, 400_000), candidate(2, 300_000), candidate(3, 200_000)],
        reference_floor_ppm=FLOOR,
        config=CONFIG,
    )
    assert result.burned is True
    assert result.uids == [CONFIG.burn_uid]
    assert result.weights == [65_535]


def test_the_best_of_a_weak_field_still_does_not_qualify():
    """The distinction between an absolute floor and a relative one, measured."""
    result = allocate([candidate(1, 499_999)], reference_floor_ppm=FLOOR, config=CONFIG)
    assert result.burned is True


def test_a_score_exactly_at_the_floor_does_not_qualify():
    """20.1 says the score must *exceed* the floor, not match it."""
    assert allocate([candidate(1, FLOOR)], reference_floor_ppm=FLOOR, config=CONFIG).burned


def test_a_score_one_unit_above_the_floor_qualifies():
    assert not allocate([candidate(1, FLOOR + 1)], reference_floor_ppm=FLOOR, config=CONFIG).burned


# --------------------------------------------------------------------------
# Every exclusion states its reason
# --------------------------------------------------------------------------


def test_a_failed_hard_gate_excludes_and_says_so():
    """22 publishes hard-gate outcomes, and the reasons need different fixes."""
    result = allocate(
        [candidate(1, 900_000, hard_gates_passed=False), candidate(2, 700_000)],
        reference_floor_ppm=FLOOR,
        config=CONFIG,
    )
    assert result.excluded[1] == "a hard gate failed"
    assert 1 not in result.uids


def test_an_unavailable_bundle_excludes():
    result = allocate(
        [candidate(1, 900_000, artifacts_available=False), candidate(2, 700_000)],
        reference_floor_ppm=FLOOR,
        config=CONFIG,
    )
    assert "no longer available" in result.excluded[1]


def test_too_few_valid_challenges_excludes_with_the_count():
    result = allocate(
        [candidate(1, 900_000, valid_challenges=2), candidate(2, 700_000)],
        reference_floor_ppm=FLOOR,
        config=CONFIG,
    )
    assert "2 valid challenges" in result.excluded[1]


def test_being_below_the_floor_is_reported_distinctly_from_a_gate_failure():
    result = allocate(
        [candidate(1, 100_000), candidate(2, 700_000)],
        reference_floor_ppm=FLOOR,
        config=CONFIG,
    )
    assert "does not exceed the reference floor" in result.excluded[1]


# --------------------------------------------------------------------------
# 20.2: softmax on the gap, not the ratio
# --------------------------------------------------------------------------


def test_a_higher_score_receives_more_weight():
    result = allocate(
        [candidate(1, 900_000), candidate(2, 700_000), candidate(3, 600_000)],
        reference_floor_ppm=FLOOR,
        config=CONFIG,
    )
    shares = result.weights_ppm
    assert shares[1] > shares[2] > shares[3]


def test_the_gap_decides_the_split_rather_than_the_ratio():
    """The reason softmax rather than proportional.

    Two fields with the same score *ratio* but different *gaps* must allocate differently — a
    bounded score needs the gap to matter, since 0.60 against 0.30 means something quite
    different in a tight field than in a spread one.
    """
    tight = allocate(
        [candidate(1, 700_000), candidate(2, 690_000)], reference_floor_ppm=FLOOR, config=CONFIG
    ).weights_ppm
    spread = allocate(
        [candidate(1, 1_000_000), candidate(2, 510_000)], reference_floor_ppm=FLOOR, config=CONFIG
    ).weights_ppm
    assert tight[1] / max(tight[2], 1) < spread[1] / max(spread[2], 1)


def test_equal_scores_receive_equal_weight():
    result = allocate(
        [candidate(1, 800_000), candidate(2, 800_000), candidate(3, 800_000)],
        reference_floor_ppm=FLOOR,
        config=CONFIG,
    )
    shares = result.weights_ppm
    assert shares[1] == shares[2] == shares[3]


def test_a_zero_temperature_is_refused():
    """At zero the softmax is winner-take-all, which the cap exists to prevent."""
    with pytest.raises(WeightsError, match="winner-take-all"):
        WeightsConfig(
            temperature_ppm=0,
            maximum_weight_ppm=175_000,
            minimum_valid_challenges=6,
            burn_uid=0,
            reference_floor_lab="reference_a",
        )


def test_an_extreme_score_spread_does_not_overflow_the_exponential():
    """Shifting by the minimum before exponentiating is what keeps `exp` in range.

    Without it, a ppm score over a ppm temperature is an exponent in the hundreds, which
    overflows to infinity and yields `nan` weights.
    """
    result = allocate(
        [candidate(1, PPM), candidate(2, 500_001)], reference_floor_ppm=FLOOR, config=CONFIG
    )
    assert all(weight >= 0 for weight in result.weights)
    assert sum(result.weights) > 0


# --------------------------------------------------------------------------
# 20.3: the cap, and redistribution that has to iterate
# --------------------------------------------------------------------------


def test_a_cap_the_field_cannot_absorb_does_not_flatten_the_ranking():
    """Below six qualifiers a 17.5% cap is unsatisfiable.

    Everyone lands exactly on it, and because Bittensor renormalises the vector it receives,
    `[cap, cap]` becomes 50/50 — two laboratories with very different scores receiving identical
    emission, and no incentive to be the better of the two.

    So the applied cap relaxes to `PPM / N`, the tightest the field can absorb.
    """
    result = allocate(
        [candidate(1, 950_000), candidate(2, 600_000)], reference_floor_ppm=FLOOR, config=CONFIG
    )
    shares = result.weights_ppm
    assert shares[1] > shares[2], "a smaller field must still rank"


def test_the_declared_cap_binds_once_the_field_can_absorb_it():
    """Six qualifiers at 17.5% is exactly satisfiable, so nothing relaxes."""
    result = allocate(
        [candidate(1, PPM)] + [candidate(uid, 520_000) for uid in range(2, 8)],
        reference_floor_ppm=FLOOR,
        config=CONFIG,
    )
    assert all(share <= CONFIG.maximum_weight_ppm + 1 for share in result.weights_ppm.values())
    assert 1 in result.capped


def test_no_laboratory_exceeds_the_declared_cap():
    """"This encourages multiple research-lab architectures rather than one permanent winner.\""""
    result = allocate(
        [candidate(1, PPM)] + [candidate(uid, 510_000) for uid in range(2, 9)],
        reference_floor_ppm=FLOOR,
        config=CONFIG,
    )
    assert all(share <= CONFIG.maximum_weight_ppm + 1 for share in result.weights_ppm.values())


def test_a_dominant_laboratory_is_capped_and_recorded():
    result = allocate(
        [candidate(1, PPM)] + [candidate(uid, 520_000) for uid in range(2, 10)],
        reference_floor_ppm=FLOOR,
        config=CONFIG,
    )
    assert 1 in result.capped


def test_redistribution_iterates_so_a_second_laboratory_cannot_be_left_over_the_cap():
    """One pass would hand the overflow out and leave a second qualifier above the limit."""
    result = allocate(
        [candidate(1, PPM), candidate(2, 990_000), candidate(3, 980_000)]
        + [candidate(uid, 505_000) for uid in range(4, 12)],
        reference_floor_ppm=FLOOR,
        config=CONFIG,
    )
    over = {uid: s for uid, s in result.weights_ppm.items() if s > CONFIG.maximum_weight_ppm + 1}
    assert not over, f"left above the cap: {over}"


def test_redistribution_preserves_the_ranking_among_the_uncapped():
    """Distributed by existing share, so it does not flatten the field it hands weight to."""
    result = allocate(
        [candidate(1, PPM), candidate(2, 800_000), candidate(3, 700_000), candidate(4, 600_000)]
        + [candidate(uid, 510_000) for uid in range(5, 10)],
        reference_floor_ppm=FLOOR,
        config=CONFIG,
    )
    shares = result.weights_ppm
    assert shares[2] >= shares[3] >= shares[4]


# --------------------------------------------------------------------------
# One qualifier: capped, with the remainder burned
# --------------------------------------------------------------------------


def test_a_single_qualifier_is_capped_and_the_rest_burns():
    """One laboratory above the floor has earned its capped share; the rest nobody earned.

    Lifting it to 100% would breach the cap the season declared, and the cap is what stops a
    permanent winner.
    """
    result = allocate(
        [candidate(1, 900_000), candidate(2, 100_000)],
        reference_floor_ppm=FLOOR,
        config=CONFIG,
    )
    shares = result.weights_ppm
    assert shares[1] <= CONFIG.maximum_weight_ppm + 1
    assert shares[CONFIG.burn_uid] > 0
    assert result.burned is False  # a qualifier exists; this is not the 20.4 burn


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------


def test_the_same_field_allocates_identically_every_time():
    field = [candidate(uid, 500_001 + uid * 40_000) for uid in range(1, 9)]
    assert allocate(field, reference_floor_ppm=FLOOR, config=CONFIG).weights == allocate(
        field, reference_floor_ppm=FLOOR, config=CONFIG
    ).weights


def test_the_candidate_order_does_not_change_the_vector():
    """Floating-point addition is not associative.

    Summing the softmax denominator in dict order would make the vector depend on insertion
    order, and same-bundle rerun correlation is measured.
    """
    field = [candidate(uid, 500_001 + uid * 40_000) for uid in range(1, 9)]
    forward = allocate(field, reference_floor_ppm=FLOOR, config=CONFIG)
    backward = allocate(list(reversed(field)), reference_floor_ppm=FLOOR, config=CONFIG)
    assert forward.uids == backward.uids
    assert forward.weights == backward.weights


def test_the_vector_is_emitted_in_the_u16_range():
    result = allocate(
        [candidate(uid, 500_001 + uid * 50_000) for uid in range(1, 7)],
        reference_floor_ppm=FLOOR,
        config=CONFIG,
    )
    assert all(0 <= weight <= 65_535 for weight in result.weights)
    assert result.uids == sorted(result.uids)
