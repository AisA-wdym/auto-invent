"""The daily seed, and the commitments that make it unpredictable.

architecture.md 7.3:

```
daily_seed = SHA256(
    date || validator_hotkey || validator_precommitted_salt || post-deadline_block_hash
)
```

Each of the four inputs is doing something specific, and the construction only works if all
four are present.

**`date`** separates days, so yesterday's seed cannot be replayed today.

**`validator_hotkey`** makes the seed per-validator. Section 7.1 requires every validator to
generate its *own* hidden pack, and a shared seed would give every validator the same
twenty problems — which would make the whole field predictable from any one validator.

**`validator_precommitted_salt`** is what the validator commits *before* it can see the
block hash. Without it a validator could wait for the randomness, observe what seed it would
produce, and regenerate until it liked the pack. With it, the salt is fixed first and the
randomness arrives afterwards, so neither side can be chosen given the other.

**`post-deadline_block_hash`** is the part the validator cannot influence, and it is taken
*after* the submission deadline. Before the deadline, a validator that also mines could pick
a pack knowing what it had submitted; after it, the submission set is already frozen.

## Commit before reveal, verified rather than trusted

`salt_commitment` and `verify_salt` exist because "precommitted" has to mean something. A
salt revealed at generation time is checked against the commitment made before the
randomness was known, and a reveal that does not match is excluded — not warned about. The
whole ordering property collapses if an unmatched reveal is accepted, since a validator could
then commit one salt and use another.

## Slot assignment is derived, never chosen

`slot_assignments` allocates the day's twenty slots between the two generator families
(section 7.2.1) from the seed alone. Derived rather than chosen so a validator cannot decide
after generation which family produced which surviving problem — which would let it keep
whichever half suited it and re-roll the other.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .canonical import digest_bytes

__all__ = [
    "SaltVerdict",
    "SeedError",
    "challenge_id",
    "daily_seed",
    "salt_commitment",
    "slot_assignments",
    "verify_salt",
]

_log = logging.getLogger(__name__)

#: A salt shorter than this is not a commitment. 32 bytes is the hash's own width; anything
#: less is searchable, and a searchable salt is a salt the committer can choose after the
#: fact by brute force.
_MINIMUM_SALT_BYTES = 32


class SeedError(ValueError):
    """A seed input that cannot be used as given."""


def salt_commitment(salt: bytes) -> str:
    """`sha256:...` over a salt, for publishing before the randomness is known."""
    if len(salt) < _MINIMUM_SALT_BYTES:
        raise SeedError(
            f"salt is {len(salt)} bytes; at least {_MINIMUM_SALT_BYTES} are required. A "
            "shorter salt is searchable, which would let it be chosen after the block hash "
            "is known — the exact thing committing it first prevents."
        )
    return digest_bytes(salt)


@dataclass(frozen=True, slots=True)
class SaltVerdict:
    """Whether a revealed salt matches what was committed, and why not."""

    matches: bool
    reason: str = ""


def verify_salt(salt: bytes, commitment: str) -> SaltVerdict:
    """Check a revealed salt against its commitment.

    Constant-time comparison via `compare_digest`. The commitment is public, so a timing
    side channel is not obviously exploitable here — but the cost of being careful is one
    function call, and the cost of being wrong is a validator learning it can grind a salt
    against the check.
    """
    if len(salt) < _MINIMUM_SALT_BYTES:
        return SaltVerdict(
            False, f"salt is {len(salt)} bytes, below the {_MINIMUM_SALT_BYTES} minimum"
        )
    if not hmac.compare_digest(salt_commitment(salt), commitment):
        return SaltVerdict(False, "revealed a salt that does not match its commitment")
    return SaltVerdict(True)


def daily_seed(
    *,
    date: str,
    validator_hotkey: str,
    salt: bytes,
    block_hash: bytes,
    commitment: str | None = None,
) -> bytes:
    """The 32-byte seed for one validator's day (architecture.md 7.3).

    `commitment` is optional in the signature and required in practice. Passing it verifies
    the salt against what was committed before the block hash was known, which is the only
    thing that makes the ordering property real. It is a parameter rather than an unconditional
    argument solely so that a test can construct a seed without a commitment ceremony — every
    production path supplies it.

    Domain-separated with a fixed label and length-prefixed fields. Concatenating variable-length
    strings without separators means two different input sets can produce identical bytes:
    hotkey `"ab"` with date `"c"` and hotkey `"a"` with date `"bc"` would collide, and a
    validator able to choose either could reuse a seed across days.
    """
    if commitment is not None:
        verdict = verify_salt(salt, commitment)
        if not verdict.matches:
            raise SeedError(
                f"cannot derive a seed for {date}: {verdict.reason}. Accepting an unmatched "
                "reveal would let a validator commit one salt and use another, which is the "
                "whole of what committing first prevents."
            )
    elif len(salt) < _MINIMUM_SALT_BYTES:
        raise SeedError(f"salt is {len(salt)} bytes, below the {_MINIMUM_SALT_BYTES} minimum")

    if not date or not validator_hotkey:
        raise SeedError("date and validator_hotkey are both required and neither may be empty")
    if not block_hash:
        raise SeedError(
            "block_hash is empty: without the post-deadline randomness the seed is entirely "
            "validator-chosen, and the pack could be selected to suit a submission"
        )

    digest = hashlib.sha256()
    digest.update(b"auto-invent/daily-seed/1")
    for field in (date.encode(), validator_hotkey.encode(), salt, block_hash):
        # Length-prefixed, so no two distinct field sets can concatenate to the same bytes.
        digest.update(len(field).to_bytes(4, "big"))
        digest.update(field)
    return digest.digest()


def slot_assignments(seed: bytes, generators: Sequence[Mapping[str, object]]) -> tuple[str, ...]:
    """Which generator family owns each of the day's slots.

    Returns one family name per slot, in slot order. Derived from the seed alone, so the
    assignment is fixed before generation begins and a validator cannot decide afterwards
    which family produced a surviving problem — which would let it keep the half it liked
    and re-roll the other.

    The deal is a seeded Fisher-Yates over a multiset built from the declared slot counts.
    Building the multiset first and shuffling it guarantees the exact declared counts:
    drawing a family per slot independently would give ten and ten only on average, and a
    day that came out fourteen-six would quietly break the balance the two-generator design
    depends on.
    """
    pool: list[str] = []
    for generator in sorted(generators, key=lambda g: str(g["family"])):
        family = str(generator["family"])
        count = int(generator["slots"])  # type: ignore[arg-type]
        if count < 1:
            raise SeedError(f"generator {family!r} declares {count} slots")
        pool.extend([family] * count)

    if not pool:
        raise SeedError("no generator slots declared; there would be nothing to generate")

    # Sorted by family above, so the pool's starting order is independent of how the caller
    # ordered its configuration. Without that, two validators with the same seed and the same
    # generators listed in different orders would deal different slots.
    stream = _seeded_stream(seed, b"slot-assignment")
    for index in range(len(pool) - 1, 0, -1):
        swap = next(stream) % (index + 1)
        pool[index], pool[swap] = pool[swap], pool[index]
    return tuple(pool)


def challenge_id(body: Mapping[str, object]) -> str:
    """Content address of a challenge, over its body minus the id itself.

    Excludes `challenge_id` so the value is derivable rather than declared: a challenge whose
    id was a free field could be given any id at all, and the id is what the pack hash and
    every receipt reference.
    """
    from .canonical import digest_object

    return digest_object({key: value for key, value in body.items() if key != "challenge_id"})


def _seeded_stream(seed: bytes, label: bytes):
    """An unbounded deterministic integer stream from a seed.

    A counter-mode hash chain rather than `random.Random`. `Random`'s Mersenne Twister is
    seeded reproducibly, but its algorithm is an implementation detail of CPython, so a
    future interpreter could deal different slots from the same seed — and slot assignment is
    committed on chain. A hash chain has no such dependency: SHA-256 of a counter is the same
    everywhere, forever.
    """
    counter = 0
    while True:
        block = hashlib.sha256(seed + label + counter.to_bytes(8, "big")).digest()
        # Four 8-byte draws per hash, so the stream is cheap without weakening it.
        for offset in range(0, 32, 8):
            yield int.from_bytes(block[offset : offset + 8], "big")
        counter += 1
