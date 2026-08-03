"""The deterministic encoder.

Every hash the subnet compares passes through here, so a defect in this module surfaces
everywhere else as two validators disagreeing about content when they disagree about bytes.
"""

from __future__ import annotations

import cbor2
import pytest
from hypothesis import given
from hypothesis import strategies as st

from protocol.canonical import (
    FloatInHashedObject,
    assert_no_floats,
    canonical_bytes,
    digest_bytes,
    digest_object,
    is_digest,
)

pytestmark = pytest.mark.determinism


# --------------------------------------------------------------------------
# Determinism: the property the whole module exists for
# --------------------------------------------------------------------------


def test_key_order_does_not_change_the_encoding():
    """Two validators building the same pack can insert keys in different orders.

    Without canonical encoding the digests would differ and the disagreement would look like
    a disagreement about the pack.
    """
    forward = {"a": 1, "b": 2, "c": 3}
    backward = {"c": 3, "b": 2, "a": 1}
    assert canonical_bytes(forward) == canonical_bytes(backward)
    assert digest_object(forward) == digest_object(backward)


def test_nested_key_order_does_not_change_the_encoding():
    """Sorting only the top level would leave every nested object a divergence source."""
    first = {"outer": {"x": 1, "y": [{"p": 1, "q": 2}]}}
    second = {"outer": {"y": [{"q": 2, "p": 1}], "x": 1}}
    assert canonical_bytes(first) == canonical_bytes(second)


def test_the_encoding_is_stable_across_calls():
    document = {"challenge_id": "sha256:" + "a" * 64, "constraints": ["x", "y"]}
    assert canonical_bytes(document) == canonical_bytes(document)


@given(
    st.dictionaries(
        st.text(min_size=1, max_size=8),
        st.one_of(st.integers(), st.text(max_size=20), st.booleans(), st.none()),
        max_size=8,
    )
)
def test_any_float_free_mapping_round_trips(document):
    """Encode then decode must return what went in, or a digest attests to something else."""
    assert cbor2.loads(canonical_bytes(document)) == document


@given(st.dictionaries(st.text(min_size=1, max_size=6), st.integers(), min_size=1, max_size=6))
def test_reordering_any_mapping_leaves_the_digest_unchanged(document):
    shuffled = dict(reversed(list(document.items())))
    assert digest_object(document) == digest_object(shuffled)


# --------------------------------------------------------------------------
# Floats are refused, and the refusal names the path
# --------------------------------------------------------------------------


def test_a_top_level_float_is_refused():
    with pytest.raises(FloatInHashedObject, match=r"float at \$\.ratio"):
        canonical_bytes({"ratio": 0.25})


def test_a_deeply_nested_float_is_found():
    """Checking only the fields a caller remembers is how a float reaches a hashed object."""
    document = {"scoring": {"weights": {"criteria": [{"originality": 0.25}]}}}
    with pytest.raises(FloatInHashedObject) as raised:
        assert_no_floats(document)
    assert "$.scoring.weights.criteria[0].originality" in str(raised.value)


def test_the_message_says_to_change_the_declaration_not_the_value():
    """The fix is always a schema change, so the error says so rather than 'invalid value'."""
    with pytest.raises(FloatInHashedObject, match="parts-per-million"):
        assert_no_floats({"x": 0.1})


def test_two_ways_of_writing_the_same_ratio_are_different_doubles():
    """The reason floats are refused rather than rounded, asserted rather than asserted-to.

    If these were equal, refusing floats would be excessive caution. They are not.
    """
    assert 0.1 + 0.2 != 0.3
    assert 0.05 + 0.05 == 0.1  # this one happens to agree, which is the trap


def test_booleans_are_not_floats():
    """`bool` is an `int` subclass and encodes exactly; refusing it would be wrong."""
    assert canonical_bytes({"ranked": True, "mechanism_required": False})


def test_arbitrarily_large_integers_are_allowed():
    """CBOR encodes big integers exactly, so the float rule does not extend to them."""
    assert canonical_bytes({"memory_bytes": 2**70})


def test_a_string_is_not_walked_as_a_sequence():
    """Recursing into a string would iterate characters without terminating."""
    assert_no_floats({"title": "a long title with many characters"})


def test_bytes_are_not_walked_as_a_sequence():
    assert_no_floats({"salt": b"\x00" * 32})


# --------------------------------------------------------------------------
# Two hashing modes, and why the distinction matters
# --------------------------------------------------------------------------


def test_source_bytes_hashing_is_of_exactly_the_bytes_given():
    """The right mode for anything from outside: the producer can reproduce it."""
    raw = b'{"portfolio": [], "challenge_id": "x"}'
    assert digest_bytes(raw) == digest_bytes(raw)
    assert digest_bytes(raw) != digest_bytes(raw + b" ")


def test_re_encoding_a_foreign_artifact_changes_its_digest():
    """Why `digest_bytes` exists rather than always encoding the object.

    A miner sends bytes. Parsing and re-encoding them yields a digest of *our* serialisation
    of *their* object — which the miner cannot reproduce from what they sent, so a
    disagreement about it would be unresolvable.
    """
    import json

    sent = b'{"b": 2, "a": 1}'
    parsed = json.loads(sent)
    assert digest_bytes(sent) != digest_object(parsed)


def test_a_digest_is_self_describing():
    assert digest_bytes(b"x").startswith("sha256:")
    assert len(digest_bytes(b"x")) == len("sha256:") + 64


@pytest.mark.parametrize(
    "value",
    [
        "a" * 64,  # no prefix
        "sha256:" + "a" * 63,  # too short
        "sha256:" + "a" * 65,  # too long
        "sha256:" + "A" * 64,  # uppercase
        "sha256:" + "g" * 64,  # not hex
        "md5:" + "a" * 32,
        "",
        None,
        123,
    ],
)
def test_a_malformed_digest_is_rejected(value):
    assert is_digest(value) is False


def test_a_well_formed_digest_is_accepted():
    assert is_digest(digest_bytes(b"anything")) is True


# --------------------------------------------------------------------------
# Two representations of one digest
# --------------------------------------------------------------------------


def test_the_two_digest_forms_compare_equal():
    """`digest_object` writes `sha256:<hex>`; `protocol.commitments` stores bare hex because an
    on-chain commitment pays for every byte. Both are deliberate, and `==` across them is wrong."""
    from protocol.canonical import same_digest

    prefixed = "sha256:" + "ab" * 32
    assert same_digest(prefixed, "ab" * 32)
    assert same_digest("ab" * 32, prefixed)
    assert same_digest(prefixed, prefixed)


def test_case_does_not_change_a_digest():
    from protocol.canonical import same_digest

    assert same_digest("sha256:" + "AB" * 32, "ab" * 32)


def test_different_digests_are_still_different():
    """The comparison is permissive about *form*, not about value."""
    from protocol.canonical import same_digest

    assert not same_digest("sha256:" + "ab" * 32, "cd" * 32)
    assert not same_digest("ab" * 32, "ab" * 31 + "cc")
