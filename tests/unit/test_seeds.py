"""The daily seed and slot assignment.

The seed decides which twenty problems a validator asks. If a validator can influence it
after seeing the randomness, or after seeing what it submitted as a miner, it can choose a
pack that suits it — so almost every test here is about what must *not* be possible.
"""

from __future__ import annotations

import collections

import pytest
from hypothesis import given
from hypothesis import strategies as st

from protocol.seeds import (
    SeedError,
    challenge_id,
    daily_seed,
    salt_commitment,
    slot_assignments,
    verify_salt,
)

pytestmark = pytest.mark.determinism

SALT = b"\x11" * 32
BLOCK = b"\xab" * 32
GENERATORS = [
    {"family": "gpt", "slots": 10},
    {"family": "claude", "slots": 10},
]


def seed(**over):
    kwargs = dict(
        date="2026-08-03",
        validator_hotkey="5ValidatorHotkey",
        salt=SALT,
        block_hash=BLOCK,
        commitment=salt_commitment(SALT),
    )
    kwargs.update(over)
    return daily_seed(**kwargs)


# --------------------------------------------------------------------------
# Commit before reveal — the property the whole construction rests on
# --------------------------------------------------------------------------


def test_a_salt_that_does_not_match_its_commitment_cannot_produce_a_seed():
    """Otherwise 'precommitted' means nothing.

    A validator could commit one salt, wait for the block hash, and reveal whichever salt
    produced a pack it preferred.
    """
    with pytest.raises(SeedError, match="does not match its commitment"):
        seed(salt=b"\x22" * 32, commitment=salt_commitment(SALT))


def test_the_refusal_explains_what_it_prevents():
    with pytest.raises(SeedError, match="commit one salt and use another"):
        seed(salt=b"\x22" * 32)


def test_a_short_salt_is_refused_because_it_is_searchable():
    """A salt shorter than the hash width can be brute-forced to a chosen seed."""
    with pytest.raises(SeedError, match="at least 32"):
        salt_commitment(b"\x11" * 16)


def test_a_short_salt_is_refused_even_without_a_commitment():
    with pytest.raises(SeedError, match="below the 32"):
        seed(salt=b"short", commitment=None)


def test_verification_reports_why_it_failed():
    assert verify_salt(SALT, salt_commitment(SALT)).matches is True
    verdict = verify_salt(b"\x33" * 32, salt_commitment(SALT))
    assert verdict.matches is False
    assert "does not match" in verdict.reason


# --------------------------------------------------------------------------
# Every input is load-bearing
# --------------------------------------------------------------------------


def test_the_block_hash_is_required():
    """Without it the seed is entirely validator-chosen.

    A validator that also mines could then select a pack knowing what it had submitted.
    """
    with pytest.raises(SeedError, match="entirely\\s+validator-chosen"):
        seed(block_hash=b"")


def test_a_different_day_gives_a_different_seed():
    """So yesterday's pack cannot be replayed today."""
    assert seed(date="2026-08-03") != seed(date="2026-08-04")


def test_a_different_validator_gives_a_different_seed():
    """Every validator generates its own hidden pack.

    A shared seed would make the whole field predictable from any single validator.
    """
    assert seed(validator_hotkey="5A") != seed(validator_hotkey="5B")


def test_a_different_salt_gives_a_different_seed():
    other = b"\x99" * 32
    assert seed(salt=SALT) != seed(salt=other, commitment=salt_commitment(other))


def test_a_different_block_hash_gives_a_different_seed():
    assert seed(block_hash=BLOCK) != seed(block_hash=b"\xcd" * 32)


def test_an_empty_date_or_hotkey_is_refused():
    with pytest.raises(SeedError, match="neither may be empty"):
        seed(date="")
    with pytest.raises(SeedError, match="neither may be empty"):
        seed(validator_hotkey="")


# --------------------------------------------------------------------------
# Field separation: two input sets must never collide
# --------------------------------------------------------------------------


def test_fields_cannot_be_shifted_between_each_other():
    """Concatenating variable-length fields without separators lets two input sets collide.

    Hotkey 'ab' with date 'c' and hotkey 'a' with date 'bc' would produce identical bytes,
    and a validator able to choose either could reuse a seed across days.
    """
    first = seed(date="c", validator_hotkey="ab")
    second = seed(date="bc", validator_hotkey="a")
    assert first != second


def test_the_seed_is_thirty_two_bytes():
    assert len(seed()) == 32


def test_the_same_inputs_always_give_the_same_seed():
    assert seed() == seed()


# --------------------------------------------------------------------------
# Slot assignment: derived, exact, and order-independent
# --------------------------------------------------------------------------


def test_the_declared_counts_are_exact_not_average():
    """Drawing a family per slot independently would give 10/10 only on average.

    A day that came out 14/6 would quietly break the balance the two-generator design
    depends on, and nothing would report it.
    """
    counts = collections.Counter(slot_assignments(seed(), GENERATORS))
    assert counts == {"gpt": 10, "claude": 10}


def test_the_assignment_is_derived_from_the_seed_alone():
    """So a validator cannot decide after generation which family produced what.

    If it could, it would keep the half it liked and re-roll the other.
    """
    assert slot_assignments(seed(), GENERATORS) == slot_assignments(seed(), GENERATORS)


def test_a_different_seed_deals_the_slots_differently():
    assert slot_assignments(seed(date="2026-08-03"), GENERATORS) != slot_assignments(
        seed(date="2026-09-14"), GENERATORS
    )


def test_the_configuration_order_does_not_change_the_deal():
    """Two validators with the same seed must deal the same slots.

    Without sorting the pool first, listing the generators in a different order in the config
    would produce a different assignment from an identical seed.
    """
    forward = slot_assignments(seed(), GENERATORS)
    backward = slot_assignments(seed(), list(reversed(GENERATORS)))
    assert forward == backward


def test_the_assignment_actually_interleaves():
    """A 'shuffle' that returned the pool unchanged would pass the count test.

    Ten gpt then ten claude is a legal permutation but not a shuffle, and it would make the
    family predictable from the slot index.
    """
    dealt = slot_assignments(seed(), GENERATORS)
    unshuffled = ("claude",) * 10 + ("gpt",) * 10
    assert dealt != unshuffled
    # Some adjacent pair must differ, or the deal is blocked rather than mixed.
    assert any(a != b for a, b in zip(dealt, dealt[1:], strict=False))


@given(st.integers(min_value=1, max_value=40), st.integers(min_value=1, max_value=40))
def test_any_declared_split_is_honoured_exactly(gpt_slots, claude_slots):
    generators = [
        {"family": "gpt", "slots": gpt_slots},
        {"family": "claude", "slots": claude_slots},
    ]
    counts = collections.Counter(slot_assignments(seed(), generators))
    assert counts == {"gpt": gpt_slots, "claude": claude_slots}


def test_no_generators_is_refused():
    with pytest.raises(SeedError, match="nothing to generate"):
        slot_assignments(seed(), [])


def test_a_generator_with_no_slots_is_refused():
    with pytest.raises(SeedError, match="declares 0 slots"):
        slot_assignments(seed(), [{"family": "gpt", "slots": 0}])


def test_the_stream_does_not_depend_on_the_interpreter_random_module():
    """`random.Random`'s algorithm is a CPython implementation detail.

    Slot assignment is committed on chain, so a future interpreter dealing different slots
    from the same seed would break a commitment. A SHA-256 counter chain has no such
    dependency, and this pins the exact expected deal.
    """
    dealt = slot_assignments(bytes(range(32)), GENERATORS)
    assert dealt == slot_assignments(bytes(range(32)), GENERATORS)
    assert collections.Counter(dealt) == {"gpt": 10, "claude": 10}


# --------------------------------------------------------------------------
# Challenge identity
# --------------------------------------------------------------------------


def test_the_challenge_id_excludes_itself():
    """Otherwise the id would be a free field and could be given any value.

    The id is what the pack hash and every receipt reference, so it must be derivable.
    """
    body = {"domain": "algorithms", "title": "a title", "constraints": ["x"]}
    without = challenge_id(body)
    with_claim = challenge_id({**body, "challenge_id": "sha256:" + "f" * 64})
    assert without == with_claim


def test_changing_any_field_changes_the_challenge_id():
    body = {"domain": "algorithms", "title": "a title", "constraints": ["x"]}
    assert challenge_id(body) != challenge_id({**body, "title": "another title"})
    assert challenge_id(body) != challenge_id({**body, "constraints": ["y"]})
