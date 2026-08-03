"""The hash-chained receipt of every external call a laboratory made.

architecture.md 13 makes "budget exceeded" and "unauthorized endpoint" **hard gates**, and
27 requires **100% execution-receipt reconciliation**. Neither is possible from a log: a log
can be edited, truncated, or reordered after the fact and nothing in it objects.

A chain can. Each entry commits to the previous one, so removing a call breaks every link
after it, and reordering two calls breaks both. Verification is a single pass that recomputes
the chain and compares — which is what turns "the miner spent 400 RCC" from an assertion into
something a third party can check against bytes.

## What a receipt records, and what it deliberately does not

Recorded: sequence, tool, provider, model and revision, request and response **hashes**, RCC,
token counts, attempt count, and whose credential paid.

Not recorded: request or response **content**. The gateway must be able to prove what was
asked without becoming an archive of every prompt any miner ever wrote — an archive that
would be a standing liability and, once published at section 22, a corpus of every
competitor's technique.

## Attempts are counted, so a retry storm is not free

`attempts` is on the call rather than derived. A provider failure that consumed budget across
three attempts costs three attempts' worth, and a receipt that recorded only the successful
one would let a laboratory retry without limit at no charge — and would make the receipt
disagree with the provider's invoice, which is precisely what reconciliation exists to catch.

## Whose credential paid is part of the evidence

`credential_owner` and `purpose` are recorded on every call because a single provider surface
means a swapped credential *succeeds* (architecture.md 3.4.4). The API will not object; only
per-account reconciliation will. So the receipt carries the claim, and reconciliation checks
it.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .canonical import canonical_bytes, digest_bytes, digest_object

__all__ = [
    "GENESIS",
    "Call",
    "CredentialOwner",
    "Purpose",
    "Receipt",
    "ReceiptBroken",
    "Tool",
    "chain_head",
    "reconcile",
    "verify_chain",
]

_log = logging.getLogger(__name__)

#: The first link's predecessor. A literal rather than an empty string so an entry whose
#: `previous` was never populated is distinguishable from a genuine chain start.
GENESIS = "genesis"


class Tool(str, Enum):
    LLM = "llm"
    SEARCH = "search"
    SIM = "sim"
    EMBEDDING = "embedding"
    CODE = "code"


class CredentialOwner(str, Enum):
    """Which account paid. See architecture.md 3.4.4."""

    MINER = "miner"
    VALIDATOR = "validator"


class Purpose(str, Enum):
    """What a call was for. Purpose selects the credential, so it is evidence."""

    RESEARCH = "research"
    SEARCH = "search"
    SIMULATION = "simulation"
    CHALLENGE_GENERATION = "challenge_generation"
    CRITIQUE = "critique"
    JUDGING = "judging"
    PRIOR_ART = "prior_art"


#: Which owner each purpose must be billed to. The mapping is the invariant from
#: architecture.md 3.4.4 expressed as data, so it can be asserted rather than remembered —
#: and so adding a purpose forces a decision about who pays for it.
_PURPOSE_OWNER: dict[Purpose, CredentialOwner] = {
    Purpose.RESEARCH: CredentialOwner.MINER,
    Purpose.SEARCH: CredentialOwner.MINER,
    Purpose.SIMULATION: CredentialOwner.MINER,
    Purpose.CHALLENGE_GENERATION: CredentialOwner.VALIDATOR,
    Purpose.CRITIQUE: CredentialOwner.VALIDATOR,
    Purpose.JUDGING: CredentialOwner.VALIDATOR,
    Purpose.PRIOR_ART: CredentialOwner.VALIDATOR,
}


class ReceiptBroken(ValueError):
    """The chain does not verify, or a call is billed to the wrong account."""


@dataclass(frozen=True, slots=True)
class Call:
    """One external call, as evidence."""

    seq: int
    tool: Tool
    provider: str
    request_hash: str
    response_hash: str
    rcc: int
    purpose: Purpose
    credential_owner: CredentialOwner
    previous: str = GENESIS
    model: str = ""
    revision: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    attempts: int = 1

    def __post_init__(self) -> None:
        if self.attempts < 1:
            raise ReceiptBroken(
                f"call {self.seq} records {self.attempts} attempts; a call that happened had "
                "at least one, and zero would make a retry free"
            )
        if self.rcc < 0:
            raise ReceiptBroken(f"call {self.seq} records negative RCC")
        expected = _PURPOSE_OWNER[self.purpose]
        if self.credential_owner is not expected:
            # The check the API will not do for us. One provider surface means a swapped key
            # authenticates and returns a completion; this is where it stops.
            raise ReceiptBroken(
                f"call {self.seq}: purpose {self.purpose.value!r} must be billed to "
                f"{expected.value!r}, not {self.credential_owner.value!r}. With one provider "
                "surface a swapped credential succeeds and silently bills the wrong party, so "
                "this is refused here rather than discovered on an invoice."
            )

    def link_body(self) -> dict[str, Any]:
        """Exactly the fields the chain link commits to.

        Named explicitly rather than derived from the dataclass, because a field added later
        would then silently change every link — and a chain whose definition moves cannot be
        verified against history.
        """
        return {
            "seq": self.seq,
            "tool": self.tool.value,
            "provider": self.provider,
            "model": self.model,
            "revision": self.revision,
            "request_hash": self.request_hash,
            "response_hash": self.response_hash,
            "rcc": self.rcc,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "attempts": self.attempts,
            "purpose": self.purpose.value,
            "credential_owner": self.credential_owner.value,
            "previous": self.previous,
        }

    def link(self) -> str:
        return digest_object(self.link_body())


def chain_head(calls: Sequence[Call]) -> str:
    """The head link, and therefore a commitment to every call before it."""
    return calls[-1].link() if calls else GENESIS


def verify_chain(calls: Sequence[Call]) -> None:
    """Recompute the chain and raise on the first break.

    Checks sequence numbers as well as links. A chain whose links verify but whose sequence
    skips a number is a chain missing a call that was never linked — the links alone cannot
    see that, because what was removed was removed before it was ever committed to.
    """
    expected_previous = GENESIS
    for index, call in enumerate(calls):
        if call.seq != index:
            raise ReceiptBroken(
                f"call at position {index} declares seq {call.seq}: a gap means a call was "
                "dropped before it was linked, which the links themselves cannot detect"
            )
        if call.previous != expected_previous:
            raise ReceiptBroken(
                f"call {call.seq} chains to {call.previous!r} but the previous link is "
                f"{expected_previous!r}: the chain has been reordered or an entry removed"
            )
        expected_previous = call.link()


@dataclass(slots=True)
class Receipt:
    """One run's complete evidence, built as calls happen."""

    run_id: str
    miner_hotkey: str
    bundle_digest: str
    challenge_id: str
    validator_hotkey: str
    calls: list[Call] = field(default_factory=list)

    def record(
        self,
        *,
        tool: Tool,
        provider: str,
        request: bytes,
        response: bytes,
        rcc: int,
        purpose: Purpose,
        credential_owner: CredentialOwner,
        model: str = "",
        revision: str = "",
        tokens_in: int = 0,
        tokens_out: int = 0,
        attempts: int = 1,
    ) -> Call:
        """Append one call, hashing the bodies rather than keeping them.

        Bodies are hashed here and discarded. The caller cannot opt out by passing a hash
        instead: taking bytes and hashing them internally is what guarantees the recorded
        digest is of what was actually sent, rather than of whatever the caller claimed.
        """
        call = Call(
            seq=len(self.calls),
            tool=tool,
            provider=provider,
            request_hash=digest_bytes(request),
            response_hash=digest_bytes(response),
            rcc=rcc,
            purpose=purpose,
            credential_owner=credential_owner,
            previous=chain_head(self.calls),
            model=model,
            revision=revision,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            attempts=attempts,
        )
        self.calls.append(call)
        return call

    @property
    def totals(self) -> dict[str, int]:
        """Measured usage. What replaces a laboratory's self-reported claim."""
        return {
            "rcc": sum(call.rcc for call in self.calls),
            "model_calls": sum(1 for call in self.calls if call.tool is Tool.LLM),
            "search_calls": sum(1 for call in self.calls if call.tool is Tool.SEARCH),
            "tokens_in": sum(call.tokens_in for call in self.calls),
            "tokens_out": sum(call.tokens_out for call in self.calls),
            "attempts": sum(call.attempts for call in self.calls),
        }

    def spend_by_owner(self) -> dict[str, int]:
        """RCC per paying account, for reconciliation (architecture.md 3.4.4 point 3)."""
        by_owner: dict[str, int] = {owner.value: 0 for owner in CredentialOwner}
        for call in self.calls:
            by_owner[call.credential_owner.value] += call.rcc
        return by_owner

    def as_document(self) -> dict[str, Any]:
        """The receipt as `execution_receipt.json` describes it."""
        verify_chain(self.calls)
        return {
            "run_id": self.run_id,
            "miner_hotkey": self.miner_hotkey,
            "bundle_digest": self.bundle_digest,
            "challenge_id": self.challenge_id,
            "validator_hotkey": self.validator_hotkey,
            "credential_owner": CredentialOwner.MINER.value,
            "purpose": Purpose.RESEARCH.value,
            "calls": [call.link_body() for call in self.calls],
            "totals": self.totals,
            "chain_head": chain_head(self.calls),
        }

    def signing_payload(self) -> bytes:
        """Bytes the validator signs. Over the document, so the signature covers the chain."""
        return canonical_bytes(self.as_document())


def reconcile(receipt: Receipt, *, provider_reported_rcc: int, tolerance: int = 0) -> None:
    """Compare measured spend against what the provider itself reports.

    The check architecture.md 27 requires at 100%, and the only one that catches a call made
    outside the receipted path — because such a call appears in the provider's accounting and
    nowhere in ours.

    `tolerance` defaults to zero. A non-zero tolerance is sometimes operationally necessary
    (a provider rounding differently, a call in flight at the boundary), but it must be an
    explicit decision at the call site: a default tolerance is a standing allowance for
    exactly the discrepancy this function exists to find.
    """
    measured = receipt.totals["rcc"]
    difference = provider_reported_rcc - measured
    if abs(difference) > tolerance:
        raise ReceiptBroken(
            f"run {receipt.run_id}: the provider reports {provider_reported_rcc} RCC and the "
            f"receipt chain accounts for {measured} ({difference:+d}). Unaccounted spend means "
            "a call was made outside the receipted path; unaccounted credit means a receipt "
            "records a call the provider never saw. Both are incidents."
        )
