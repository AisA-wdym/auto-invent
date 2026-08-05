"""The one deterministic encoder, and the rules that keep it deterministic.

Every hash the subnet compares — a challenge pack hash committed on chain, a receipt
chain link, a bundle digest — passes through here. If two validators encode the same
object differently, every downstream comparison fails for a reason that looks like
disagreement about content rather than about bytes.

## Two hashing modes, and choosing wrongly is the classic error

**Hash the object** when the subnet constructs it. A challenge pack is assembled from
generated problems, so there is no authoritative byte sequence until we make one, and
canonical CBOR is what makes that choice reproducible.

**Hash the source bytes** when something else constructed it. A miner's portfolio arrives
as bytes over stdout; re-encoding it before hashing would produce a digest of *our*
serialisation of *their* object. Any difference — key order, an integer that round-trips as
a float, a field our model does not know — changes the digest, and the miner could not
reproduce it from what they sent. `digest_bytes` exists for exactly that case and is the
right call whenever the artifact came from outside.

## Why floats are refused rather than rounded

CBOR encodes a float as an IEEE-754 double, which is deterministic. The problem is upstream:
a value read from JSON as `0.1` and a value computed as `0.05 + 0.05` are different doubles,
and both are plausible ways for the same declared ratio to arrive. Rounding at the encoder
would hide that; refusing surfaces it where it can be fixed — by declaring the field as an
integer in parts-per-million, which is what `season_config.json` does throughout.

`assert_no_floats` therefore names the path it found rather than the value, because the fix
is always to change how that field is declared, never to change its value.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import cbor2

__all__ = [
    "FloatInHashedObject",
    "assert_no_floats",
    "canonical_bytes",
    "digest_bytes",
    "digest_object",
]

#: Prefix on every digest the subnet emits. Present so a digest is self-describing: a bare
#: 64-hex string could be any algorithm, and the schemas pin this form precisely so a reader
#: never has to guess which.
_PREFIX = "sha256:"


class FloatInHashedObject(TypeError):
    """A float reached an object that will be hashed.

    A `TypeError` rather than a `ValueError`: the value is not out of range, the *type* is
    wrong for a hashed position, and the fix is a schema change rather than a different
    number.
    """


def assert_no_floats(value: Any, *, path: str = "$") -> None:
    """Raise if any float appears anywhere in `value`.

    Walks the whole structure rather than checking known fields. Checking only the fields
    a caller remembers is how a float reaches a hashed object through the one branch nobody
    listed — and the failure then surfaces as two validators disagreeing about a digest,
    with nothing pointing at the cause.

    `bool` is not a float and is allowed. `int` of any size is allowed: CBOR encodes
    arbitrary-precision integers exactly, so a large integer is safe where a float is not.
    """
    if isinstance(value, float):
        raise FloatInHashedObject(
            f"float at {path}: a value read as 0.1 and a value computed as 0.05 + 0.05 are "
            "different doubles, so two hosts can disagree on the bytes. Declare this field "
            "as an integer in parts-per-million instead."
        )
    if isinstance(value, Mapping):
        for key, item in value.items():
            assert_no_floats(item, path=f"{path}.{key}")
    elif isinstance(value, str | bytes | bytearray):
        # Strings and byte strings are sequences, and recursing into a string would walk
        # its characters forever. Checked before the Sequence branch for that reason.
        return
    elif isinstance(value, Sequence):
        for index, item in enumerate(value):
            assert_no_floats(item, path=f"{path}[{index}]")


def canonical_bytes(value: Any) -> bytes:
    """Deterministic CBOR for an object the subnet constructed.

    `canonical=True` gives RFC 8949 deterministic encoding: map keys sorted by their encoded
    form, shortest-form integers, no indefinite-length items. That removes dict insertion
    order — which is otherwise a live source of divergence, because two validators building
    the same pack from the same problems can legitimately insert keys in different orders.

    Refuses floats first. Encoding a float would succeed and produce a digest that a peer
    computing from the same declared values might not reproduce.
    """
    assert_no_floats(value)
    return cbor2.dumps(value, canonical=True)


def digest_object(value: Any) -> str:
    """`sha256:...` over the canonical encoding of an object we constructed."""
    return digest_bytes(canonical_bytes(value))


def same_digest(left: str, right: str) -> bool:
    """Whether two digests name the same bytes, whichever form each is written in.

    This exists because the codebase has two representations of one value and both are deliberate.
    `digest_object` and the 5.2 manifests write `sha256:<hex>`; `protocol.commitments` strips the
    prefix, because an on-chain commitment pays for every byte and the algorithm is fixed by the
    protocol rather than carried per commitment.

    Neither choice is wrong, and comparing across them with `==` is wrong everywhere: a validator
    would refuse *every* submission, each with a message saying the digest was malformed, which is
    the one explanation that is not true.
    """
    return left.removeprefix("sha256:").lower() == right.removeprefix("sha256:").lower()


def digest_bytes(raw: bytes) -> str:
    """`sha256:...` over bytes exactly as given.

    The right call for anything that arrived from outside — a miner's portfolio, a source
    archive, a container layer. Hashing the bytes received means the producer can reproduce
    the digest from what they sent, which is what makes a disagreement about it meaningful.
    """
    return _PREFIX + hashlib.sha256(raw).hexdigest()


def is_digest(value: object) -> bool:
    """Whether `value` is a well-formed digest in the form the schemas require."""
    if not isinstance(value, str) or not value.startswith(_PREFIX):
        return False
    body = value[len(_PREFIX) :]
    return len(body) == 64 and all(character in "0123456789abcdef" for character in body)
