"""Session tokens: one run, one challenge, no credential.

architecture.md 5.4.1. The laboratory never holds a provider key. It holds this instead — an
HMAC-signed capability naming exactly one run against exactly one challenge, with the round
ceilings written into the signed body.

## Why the ceilings are inside the token

A token that said only "this is run R" would leave the budget entirely in the gateway's
mutable state. That works until two gateway processes serve the same run, or a process restarts
mid-episode and rebuilds its ledger from nothing. Putting the ceilings in the signed body means
every process that can verify the token also knows the limit, so a restart cannot raise it.

Spend still has to live in the ledger — a token cannot record what it has already spent. So the
split is: **the token is the authority on the limit, the ledger is the authority on the
consumption.** A lost ledger fails closed, because `metering` refuses a run whose spend it
cannot account for rather than assuming zero.

## Why HMAC rather than a random opaque handle

A random handle requires a lookup to mean anything, which puts the gateway's store on the path
of every single call and makes a store outage an outage of the whole subnet. A signed token
verifies from bytes. The cost is that a token cannot be revoked by deleting a row — so it is
short-lived by construction (`expires_at` is in the signed body, and the episode deadline is
its ceiling), and the run-level kill switch lives in `metering` where the spend is.

## Replay across challenges is the attack this exists to stop

Two laboratories are scored on the same challenge instances (7.1). A token replayable against a
second challenge would let a laboratory spend its whole round ceiling on the one challenge it
found easiest and submit nothing elsewhere — or, worse, spend a *rival's* budget. So
`challenge_id` is in the signed body and checked on every call, and a mismatch is refused
rather than logged.
"""

from __future__ import annotations

import hmac
import json
import logging
from base64 import urlsafe_b64decode, urlsafe_b64encode
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any

__all__ = [
    "SessionToken",
    "TokenError",
    "TokenIssuer",
]

_log = logging.getLogger(__name__)

#: Signed-body version. Present so a change to the body's shape cannot be read as the old
#: shape: a verifier that does not recognise the version refuses rather than misinterprets.
_VERSION = 1


class TokenError(RuntimeError):
    """A token that is absent, malformed, unsigned, expired, or bound elsewhere."""


@dataclass(frozen=True, slots=True)
class SessionToken:
    """The capability a container receives. Carries limits, never a credential."""

    run_id: str
    miner_hotkey: str
    bundle_digest: str
    validator_hotkey: str
    challenge_id: str
    #: OpenRouter routes this run may address, from the miner's declared model manifest.
    #: Enumerated rather than pattern-matched: a pattern that admitted `anthropic/*` would
    #: admit a model the miner never declared, and 13.3 makes undeclared model use a hard gate.
    allowed_models: tuple[str, ...]
    maximum_rcc: int
    maximum_requests: int
    maximum_search_calls: int
    #: Unix seconds. Never later than the episode deadline — see `TokenIssuer.issue`.
    expires_at: int

    def body(self) -> dict[str, Any]:
        """Exactly what the signature covers.

        Enumerated explicitly, like `Call.link_body`, so a field added later cannot silently
        fall outside the signature — which would make it attacker-controlled while looking
        signed.
        """
        return {
            "v": _VERSION,
            "run_id": self.run_id,
            "miner_hotkey": self.miner_hotkey,
            "bundle_digest": self.bundle_digest,
            "validator_hotkey": self.validator_hotkey,
            "challenge_id": self.challenge_id,
            "allowed_models": list(self.allowed_models),
            "maximum_rcc": self.maximum_rcc,
            "maximum_requests": self.maximum_requests,
            "maximum_search_calls": self.maximum_search_calls,
            "expires_at": self.expires_at,
        }

    def permits_model(self, model: str) -> bool:
        return model in self.allowed_models


@dataclass(slots=True)
class TokenIssuer:
    """Signs and verifies session tokens with a per-process secret.

    The secret is generated per gateway process and never persisted. A restart therefore
    invalidates every outstanding token, which is the safe direction: the runner reissues for
    the current episode, and a token from a previous episode cannot survive into this one.
    """

    secret: bytes = field(repr=False)

    def __post_init__(self) -> None:
        if len(self.secret) < 32:
            raise TokenError(
                f"token secret is {len(self.secret)} bytes; HMAC-SHA256 needs at least 32 to "
                "carry a full block of entropy, and this secret authorises spending"
            )

    def issue(self, token: SessionToken, *, episode_deadline: int) -> str:
        """Encode and sign, refusing a token that would outlive its episode.

        10 requires "forced termination when the episode closes". A token valid past the
        deadline would let a container that survived termination keep spending — so the clamp
        is here, at issue, rather than trusted to whoever computed `expires_at`.
        """
        if token.expires_at > episode_deadline:
            raise TokenError(
                f"run {token.run_id}: token expiry {token.expires_at} is past the episode "
                f"deadline {episode_deadline}. A token that outlives its episode is a "
                "container that can spend after it was supposed to be killed."
            )
        if token.maximum_rcc <= 0:
            raise TokenError(
                f"run {token.run_id}: a token with no RCC ceiling authorises unbounded spend "
                "on the miner's key"
            )
        if not token.allowed_models:
            raise TokenError(
                f"run {token.run_id}: no allowed models. An empty allowlist read as "
                "'unrestricted' would defeat gate 13.3, so it is refused at issue."
            )

        payload = _encode(token.body())
        signature = self._sign(payload)
        _log.info(
            "issued session token for run %s (challenge %s, ceiling %d RCC)",
            token.run_id,
            token.challenge_id,
            token.maximum_rcc,
        )
        return f"{payload.decode()}.{signature}"

    def verify(self, raw: str, *, now: int, challenge_id: str | None = None) -> SessionToken:
        """Recover the token, or raise.

        `now` is a parameter rather than a clock read so the caller owns the time source — the
        gateway reads the clock once per request and every check below sees the same instant.
        Two checks against two `time.time()` calls can straddle an expiry boundary.
        """
        payload_part, _, signature = raw.partition(".")
        if not signature:
            raise TokenError("malformed session token: no signature")
        payload = payload_part.encode()
        if not hmac.compare_digest(self._sign(payload), signature):
            # compare_digest, not `==`: a timing-variable comparison on a signature leaks the
            # signature one byte at a time to a caller that can make many attempts, and a
            # laboratory can make many attempts.
            raise TokenError("session token signature does not verify")

        try:
            body = json.loads(urlsafe_b64decode(_pad(payload)))
        except (ValueError, TypeError) as error:
            raise TokenError(f"malformed session token body: {error}") from error
        if body.get("v") != _VERSION:
            raise TokenError(
                f"session token declares version {body.get('v')!r}; this gateway understands "
                f"{_VERSION}. Refused rather than interpreted under the wrong shape."
            )

        token = SessionToken(
            run_id=body["run_id"],
            miner_hotkey=body["miner_hotkey"],
            bundle_digest=body["bundle_digest"],
            validator_hotkey=body["validator_hotkey"],
            challenge_id=body["challenge_id"],
            allowed_models=tuple(body["allowed_models"]),
            maximum_rcc=body["maximum_rcc"],
            maximum_requests=body["maximum_requests"],
            maximum_search_calls=body["maximum_search_calls"],
            expires_at=body["expires_at"],
        )
        if now >= token.expires_at:
            raise TokenError(
                f"run {token.run_id}: token expired at {token.expires_at}, now {now}"
            )
        if challenge_id is not None and challenge_id != token.challenge_id:
            raise TokenError(
                f"run {token.run_id}: token is bound to challenge {token.challenge_id} but the "
                f"request names {challenge_id}. A replayable token would let one laboratory "
                "spend its whole ceiling on a single challenge, or spend a rival's."
            )
        return token

    def _sign(self, payload: bytes) -> str:
        return hmac.new(self.secret, payload, sha256).hexdigest()


def _encode(body: dict[str, Any]) -> bytes:
    """Deterministic JSON, base64url without padding.

    `sort_keys` and tight separators because the signature is over these bytes: two encoders
    that ordered keys differently would produce two valid signatures for the same token, and
    `protocol.canonical` is not used here because this object is not hashed into any
    consensus-visible commitment — it is a local capability, and JSON keeps it debuggable.
    """
    raw = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return urlsafe_b64encode(raw).rstrip(b"=")


def _pad(payload: bytes) -> bytes:
    return payload + b"=" * (-len(payload) % 4)
