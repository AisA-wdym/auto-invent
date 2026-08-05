"""The on-chain commitment wire format. Pure encode/decode, no network.

architecture.md 6.1 and 7.4 step 6. Three different things get committed to the chain, all
through the one Commitments channel a hotkey has:

* a **miner submission** — the sealed bundle's digest and where to fetch it;
* a **validator salt** — a commitment to the salt that will seed the day's challenge pack,
  published *before* the randomness it will be mixed with exists;
* a **validator pack** — the hash of the generated pack, published *before* the pack is stored.

Encoding lives here, apart from the chain client, because the format is the interface between a
miner's CLI and every validator that reads it. A format defined inside the client would be
defined twice — once for writing and once for reading — and the two would drift.

## One hotkey, one commitment slot: the chaining problem

The Commitments pallet stores one commitment per (netuid, hotkey). Writing a second overwrites
the first. That collides with the cycle in 21, where a validator commits a salt at T-450 and a
pack hash at T-100 — the pack commitment would destroy the salt commitment, and a salt nobody
can verify is a salt the validator could have chosen after seeing the randomness.

So the pack commitment **carries the salt commitment forward**. After T-100 the single live
commitment states both, and the two can be checked against each other. What the earlier write
still supplies is *timing*: it existed at a block before the randomness block, which is the
property that matters, and which any peer reading the chain at that height can confirm.

That leaves one honest gap, worth naming: a validator that never wrote the salt commitment at
T-450 and instead composed both fields at T-100 produces a commitment that *looks* correct.
Detecting that needs the chain at the earlier height — archive access, or a peer that observed
it live. `verify_salt_timing` is where that check goes, and it takes the observed block as an
argument rather than pretending it can be derived from the commitment alone.

## Why a compact text format rather than CBOR

The pallet charges for space and caps it. A text format with a tag, pipe separators and hex
digests fits a submission in about 190 bytes and stays readable in a block explorer — which
matters because this is the channel a miner uses to prove it submitted on time, and a miner
needs to be able to look at it. Everything security-relevant here is a digest computed
elsewhere by `protocol.canonical`, so the encoding carries no consensus weight of its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

__all__ = [
    "COMMITMENT_MAX_BYTES",
    "CommitmentError",
    "Kind",
    "PackCommitment",
    "SaltCommitment",
    "SubmissionCommitment",
    "decode",
    "verify_salt_timing",
]

#: Protocol tag. Version-prefixed so a format change is a new tag rather than a silent
#: reinterpretation of old bytes — a decoder that meets an unknown version refuses.
PREFIX = "ail1"

#: The pallet's per-commitment budget is larger than this, but staying under 256 bytes keeps a
#: commitment inside one storage item on every runtime we have seen, and keeps the fee flat.
COMMITMENT_MAX_BYTES = 256

_SEPARATOR = "|"


class CommitmentError(ValueError):
    """A commitment that cannot be encoded within the budget, or cannot be parsed."""


class Kind(str, Enum):
    SUBMISSION = "sub"
    SALT = "salt"
    PACK = "pack"


def _check(field: str, value: str) -> str:
    """Refuse a separator inside a field.

    A pipe in a URL would shift every field after it, so a decoder would read a bundle digest
    as a capsule digest and compare the wrong things. Refused at encode, where the miner can
    still fix it, rather than at decode, where it is someone else's problem.
    """
    if _SEPARATOR in value:
        raise CommitmentError(
            f"{field} contains {_SEPARATOR!r}, the field separator: {value!r}. It would shift "
            "every following field, so a reader would compare the wrong digests."
        )
    if "\n" in value or "\r" in value:
        raise CommitmentError(f"{field} contains a newline: {value!r}")
    return value


def _hex64(field: str, value: str) -> str:
    """Normalise a digest to bare lowercase hex.

    `sha256:` prefixes are accepted on input and stripped, because the manifests in 5.2 write
    them that way and a miner copying from a manifest should not have to know that the chain
    format does not. Length is checked because a truncated digest is a digest that collides.
    """
    raw = value.removeprefix("sha256:").lower()
    if len(raw) != 64 or any(character not in "0123456789abcdef" for character in raw):
        raise CommitmentError(
            f"{field} is not a 64-character hex SHA-256 digest: {value!r}. An abbreviated digest "
            "becomes ambiguous as the set of artifacts grows, and pinning exists so the artifact "
            "cannot move."
        )
    return raw


@dataclass(frozen=True, slots=True)
class SubmissionCommitment:
    """A miner's sealed submission. Written once per round, before the deadline."""

    round_id: str
    bundle_digest: str
    capsule_digest: str
    artifact_url: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "bundle_digest", _hex64("bundle_digest", self.bundle_digest))
        object.__setattr__(self, "capsule_digest", _hex64("capsule_digest", self.capsule_digest))
        _check("round_id", self.round_id)
        _check("artifact_url", self.artifact_url)
        if not self.artifact_url.startswith(("https://", "ipfs://")):
            # Plain HTTP would let anyone on the path substitute the bundle. The digest would
            # catch it, but only after a download the validator paid for — and a validator that
            # downloads attacker-chosen bytes is a validator running attacker-chosen input
            # through its unarchiver.
            raise CommitmentError(
                f"artifact_url must be https:// or ipfs://, not {self.artifact_url!r}: a plain "
                "HTTP fetch can be substituted in transit"
            )

    def encode(self) -> str:
        return _join(
            Kind.SUBMISSION,
            self.round_id,
            self.bundle_digest,
            self.capsule_digest,
            self.artifact_url,
        )


@dataclass(frozen=True, slots=True)
class SaltCommitment:
    """A validator's precommitted salt, written before the randomness block."""

    round_id: str
    salt_commitment: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "salt_commitment", _hex64("salt_commitment", self.salt_commitment)
        )
        _check("round_id", self.round_id)

    def encode(self) -> str:
        return _join(Kind.SALT, self.round_id, self.salt_commitment)


@dataclass(frozen=True, slots=True)
class PackCommitment:
    """A validator's challenge pack hash (7.4 step 6), written before the pack is stored.

    Carries `salt_commitment` forward from the earlier write, because one hotkey has one
    commitment slot and this write overwrites that one. See the module docstring.
    """

    round_id: str
    pack_hash: str
    salt_commitment: str
    challenge_count: int
    generation_protocol_version: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "pack_hash", _hex64("pack_hash", self.pack_hash))
        object.__setattr__(
            self, "salt_commitment", _hex64("salt_commitment", self.salt_commitment)
        )
        _check("round_id", self.round_id)
        _check("generation_protocol_version", self.generation_protocol_version)
        if self.challenge_count <= 0:
            raise CommitmentError(
                f"a pack commitment declares {self.challenge_count} challenges; an empty pack "
                "would score every laboratory on nothing and leave them all equal"
            )

    def encode(self) -> str:
        return _join(
            Kind.PACK,
            self.round_id,
            self.pack_hash,
            self.salt_commitment,
            str(self.challenge_count),
            self.generation_protocol_version,
        )


def _join(kind: Kind, *fields: str) -> str:
    encoded = _SEPARATOR.join((f"{PREFIX}:{kind.value}", *fields))
    size = len(encoded.encode())
    if size > COMMITMENT_MAX_BYTES:
        raise CommitmentError(
            f"commitment is {size} bytes, over the {COMMITMENT_MAX_BYTES}-byte budget. The "
            "usual cause is a long artifact URL; publish behind a shorter one rather than "
            "raising the budget, because a larger commitment costs more and may not fit."
        )
    return encoded


def decode(raw: str) -> SubmissionCommitment | SaltCommitment | PackCommitment:
    """Parse a commitment read from the chain.

    Every failure raises rather than returning `None`. A caller iterating over every hotkey's
    commitment will meet commitments from other subnets and other protocol versions, and it is
    that caller's job to skip them — but it should skip them *knowingly*, because a silent
    `None` for a malformed commitment from a registered miner would look identical to a miner
    that never submitted.
    """
    text = raw.strip()
    header, separator, remainder = text.partition(_SEPARATOR)
    if not separator:
        raise CommitmentError(f"commitment has no fields: {text[:64]!r}")
    prefix, _, tag = header.partition(":")
    if prefix != PREFIX:
        raise CommitmentError(
            f"commitment prefix {prefix!r} is not {PREFIX!r}: not this protocol, or a version "
            "this decoder does not understand"
        )
    fields = remainder.split(_SEPARATOR)

    try:
        kind = Kind(tag)
    except ValueError as error:
        raise CommitmentError(f"unknown commitment kind {tag!r}") from error

    if kind is Kind.SUBMISSION:
        if len(fields) != 4:
            raise CommitmentError(f"a submission commitment has 4 fields, got {len(fields)}")
        return SubmissionCommitment(
            round_id=fields[0],
            bundle_digest=fields[1],
            capsule_digest=fields[2],
            artifact_url=fields[3],
        )
    if kind is Kind.SALT:
        if len(fields) != 2:
            raise CommitmentError(f"a salt commitment has 2 fields, got {len(fields)}")
        return SaltCommitment(round_id=fields[0], salt_commitment=fields[1])

    if len(fields) != 5:
        raise CommitmentError(f"a pack commitment has 5 fields, got {len(fields)}")
    try:
        count = int(fields[3])
    except ValueError as error:
        raise CommitmentError(f"challenge_count is not an integer: {fields[3]!r}") from error
    return PackCommitment(
        round_id=fields[0],
        pack_hash=fields[1],
        salt_commitment=fields[2],
        challenge_count=count,
        generation_protocol_version=fields[4],
    )


def verify_salt_timing(
    *,
    pack: PackCommitment,
    observed_salt: SaltCommitment,
    salt_block: int,
    randomness_block: int,
) -> None:
    """Check that the salt was committed before the randomness it was mixed with existed.

    The property 7.3 depends on: a validator that could choose its salt after seeing the
    post-deadline block hash could steer the day's challenges. Committing first removes that.

    `salt_block` is a parameter rather than something read from the pack commitment, because the
    pack commitment cannot prove when the salt was written — only that the writer knew it. The
    block comes from a peer that observed the commitment live, or from archive state at that
    height. A caller with neither cannot perform this check, and should say so rather than
    calling this with a guess.
    """
    if observed_salt.round_id != pack.round_id:
        raise CommitmentError(
            f"salt commitment is for round {observed_salt.round_id} but the pack is for "
            f"{pack.round_id}: a salt reused across rounds makes both rounds predictable from one"
        )
    if observed_salt.salt_commitment != pack.salt_commitment:
        raise CommitmentError(
            f"round {pack.round_id}: the pack commitment carries salt commitment "
            f"{pack.salt_commitment} but the chain recorded {observed_salt.salt_commitment}. The "
            "validator committed one salt and generated with another."
        )
    if salt_block >= randomness_block:
        raise CommitmentError(
            f"round {pack.round_id}: the salt was committed at block {salt_block}, at or after "
            f"the randomness block {randomness_block}. A salt chosen with the randomness in hand "
            "lets the validator steer which challenges it generates."
        )
