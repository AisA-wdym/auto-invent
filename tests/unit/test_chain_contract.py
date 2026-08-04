"""What the real chain requires, recorded from the calls that failed against it.

`FakeChain` is honest about *our* shape. Every defect below was in the SDK's shape, so no test
against the fake could have caught one — they were found by publishing a commitment and reading a
metagraph on testnet subnet 542, and each one stopped the protocol dead:

* a commitment that could not be built (`calls.CommitmentInfo` is `typing.Any`),
* a commitment field that could not be encoded (`Raw`, hex, 240 bytes),
* a commitment the pallet refused (signed with the coldkey),
* a metagraph that could not be read (`Balance.tao` on subnet alpha),
* a capsule that could not be sealed (`block_time` is a method).

These tests pin the *contract*, not the SDK: they assert the shapes this module builds, so a future
SDK that changes one fails here rather than at the first extrinsic of a live round.
"""

from __future__ import annotations

import pytest

from chain.client import _COMMITMENT_FIELD_BYTES, Neuron, SubnetView, _alpha_of

pytestmark = pytest.mark.determinism


# --------------------------------------------------------------------------
# Commitment field encoding
# --------------------------------------------------------------------------


def test_a_commitment_field_holds_at_most_128_bytes():
    """The pallet's `Data` enum runs `Raw0`…`Raw128`. `Raw129` does not exist, so a payload longer
    than this has to be split rather than sent."""
    assert _COMMITMENT_FIELD_BYTES == 128


def _chunks(raw: bytes) -> list[bytes]:
    size = _COMMITMENT_FIELD_BYTES
    return [raw[index : index + size] for index in range(0, len(raw), size)]


def test_a_full_length_commitment_splits_across_two_fields():
    """A submission commitment is about 240 bytes — a round id, two digests and an artifact URL. It
    was sent as one field and the chain answered `no variant named "Raw" in type 276`."""
    raw = b"x" * 240
    parts = _chunks(raw)
    assert [len(part) for part in parts] == [128, 112]
    assert b"".join(parts) == raw


def test_the_variant_name_states_the_byte_count():
    """`Raw` alone is not a variant, and a hex-encoded value doubles the length — a 32-byte field
    sent as hex was rejected as 64 bytes. Both were wrong at once."""
    parts = _chunks(b"y" * 200)
    names = [f"Raw{len(part)}" for part in parts]
    assert names == ["Raw128", "Raw72"]
    assert all(int(name.removeprefix("Raw")) <= 128 for name in names)


def test_a_short_commitment_stays_in_one_field():
    parts = _chunks(b"z" * 90)
    assert [len(part) for part in parts] == [90]


# --------------------------------------------------------------------------
# Stake is denominated in the subnet's own alpha
# --------------------------------------------------------------------------


class _AlphaBalance:
    """A dTAO subnet balance: `.alpha` works, `.tao` raises rather than converting."""

    alpha = 12.5

    @property
    def tao(self):
        raise RuntimeError("This balance is subnet-542 alpha, not TAO.")


class _TaoBalance:
    tao = 3.0

    @property
    def alpha(self):
        raise RuntimeError("not an alpha balance")


def test_subnet_stake_is_read_as_alpha():
    """`float(stake.tao)` raised `UnitMismatchError` on every neuron of a real dTAO subnet, so every
    metagraph read failed — which is every submission, every score and every weight vector."""
    assert _alpha_of(_AlphaBalance()) == 12.5


def test_a_tao_balance_still_reads():
    """Root and pre-dTAO subnets denominate in TAO. Both are tried, in that order."""
    assert _alpha_of(_TaoBalance()) == 3.0


def test_an_absent_or_unreadable_stake_is_zero_rather_than_an_error():
    """This figure decides nothing — it is a diagnostic — so a metagraph read must not fail over it.
    What did fail over it was insisting on one unit."""
    assert _alpha_of(None) == 0.0
    assert _alpha_of(object()) == 0.0


# --------------------------------------------------------------------------
# 23: the owner-eligibility rule reads one snapshot
# --------------------------------------------------------------------------


def _view(owner: str) -> SubnetView:
    return SubnetView(
        netuid=542,
        mechid=0,
        block=100,
        neurons=(
            Neuron(0, "hkOwner", "ckOwner", 1.0, False, True),
            Neuron(1, "hkMiner", "ckMiner", 1.0, False, True),
            Neuron(2, "hkOwnerAlt", "ckOwner", 1.0, False, True),
        ),
        commitments=(),
        owner_coldkey=owner,
    )


def test_owner_linked_hotkeys_are_found_by_coldkey():
    assert _view("ckOwner").owner_linked_hotkeys() == frozenset({"hkOwner", "hkOwnerAlt"})


def test_an_unknown_owner_excludes_nobody():
    """The permissive direction, so `prepare_all` logs that the rule is not in force. A silently
    empty set looks exactly like "the owner did not submit"."""
    assert _view("").owner_linked_hotkeys() == frozenset()


def test_every_hotkey_on_one_coldkey_is_excluded_together():
    """Measured on subnet 542, where all five hotkeys share the owner's coldkey: the rule refuses
    the whole field. That is the rule working, and it means a test subnet needs miner hotkeys on a
    coldkey that does not own the subnet."""
    view = SubnetView(
        netuid=542,
        mechid=0,
        block=100,
        neurons=tuple(Neuron(uid, f"hk{uid}", "ckOwner", 1.0, False, True) for uid in range(5)),
        commitments=(),
        owner_coldkey="ckOwner",
    )
    assert len(view.owner_linked_hotkeys()) == 5
