"""Minting a round-scoped provider key from a miner's management key.

A miner funds their own laboratory, so a credential has to reach the validator. Until now
that credential was a runtime API key — which *is* spendable balance. This is the alternative, and
the asymmetry is the whole argument for it:

    a runtime key can spend and cannot mint
    a management key can mint and cannot spend

Measured against the live API rather than read from a doc. A management key on
`/v1/chat/completions` returns `401 User not found`; a runtime key on `/api/v1/keys` returns
`401 Invalid management key`. So the two are distinguishable for free, before anything runs, and a
leaked management key is not itself a loss of funds.

## What a minted key carries

Two bounds, both enforced by the provider rather than by us:

**A hard credit limit**, set to the miner's declared spend cap. Our own ledger already refuses to
exceed the round's RCC ceiling, but that ceiling is enforced by *our code* — a bug in it, or a
validator who patched their own gateway, spends the miner's balance. A minted limit means OpenRouter
refuses too. Two independent enforcers is the entire point; one of them being ours is why the other
one matters.

**An expiry.** Set past the round's end, so a key survives its round and dies afterwards even if
this validator crashes between minting and revoking. Revocation is still explicit — the expiry is
what makes the failure of revocation survivable rather than what replaces it.

## The secret exists once

`POST /api/v1/keys` returns the key string only at creation; afterwards only its hash and usage are
readable. So a mint that is not captured is a key that can never be used and can only be deleted by
hash — which is why the hash is recorded before anything else happens with the secret.

## Failure is refusal, not fallback

If minting fails, the submission is refused. There is nothing to fall back to: the management key
cannot make an inference call, so "use it directly" is not a degraded mode, it is a 401 on every
call and a laboratory scored as having produced nothing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

__all__ = [
    "MintedKey",
    "ProvisioningError",
    "is_management_key",
    "mint_round_key",
    "read_usage",
    "revoke",
]

_log = logging.getLogger(__name__)

_BASE = "https://openrouter.ai/api/v1/keys"
_TIMEOUT = 30.0


class ProvisioningError(RuntimeError):
    """A management key that will not mint, or a mint that cannot be trusted."""


@dataclass(frozen=True, slots=True)
class MintedKey:
    """A key created for one round, and the handle needed to revoke it."""

    #: The secret. Returned by the provider only at creation and held nowhere else.
    secret: str = field(repr=False)
    #: The provider's handle. This is what revocation and usage lookups take, and it is not a
    #: secret — which is why it, and not the key, is what gets written down.
    key_hash: str
    limit_usd: float
    expires_at: str

    def __repr__(self) -> str:
        # A dataclass repr in a traceback is how a credential reaches a log file, and round logs
        # are published (6.3, 22).
        return f"MintedKey(key_hash={self.key_hash[:12]}…, limit_usd={self.limit_usd})"


def _request(method: str, url: str, *, management_key: str, body: dict | None = None) -> Any:
    import httpx

    try:
        response = httpx.request(
            method,
            url,
            headers={
                "Authorization": f"Bearer {management_key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=_TIMEOUT,
        )
    except httpx.HTTPError as error:
        raise ProvisioningError(f"cannot reach the provisioning API: {error}") from error
    return response


def is_management_key(credential: str) -> bool:
    """Whether this credential can mint. Costs nothing and spends nothing.

    A miner who declares a management key and supplies a runtime one is told at admission rather
    than by every inference call failing — and the reverse, a runtime key declared as management,
    would otherwise mint nothing and refuse the whole submission for the wrong stated reason.
    """
    response = _request("GET", _BASE, management_key=credential)
    if response.status_code == 200:
        return True
    if response.status_code in (401, 403):
        return False
    raise ProvisioningError(
        f"the provisioning API answered {response.status_code} to a key listing, which is neither "
        "'this is a management key' nor 'this is not'. Refusing to guess."
    )


def mint_round_key(
    management_key: str,
    *,
    name: str,
    limit_usd: float,
    lifetime_hours: float = 30.0,
) -> MintedKey:
    """Create a key bounded by a credit limit and an expiry.

    `lifetime_hours` defaults past a full round rather than to it exactly. A key that expired at
    the execution deadline would kill a laboratory still finishing its last call inside its own
    window — the expiry is a backstop for a validator that never revoked, not a second deadline.
    """
    if limit_usd <= 0:
        raise ProvisioningError(
            f"refusing to mint a key with a limit of {limit_usd}. A zero or negative limit is "
            "not a safe default — it is either a key that cannot run the laboratory at all, or, "
            "if the provider treats it as absent, an uncapped key on the miner's account."
        )

    expires_at = (
        datetime.now(UTC) + timedelta(hours=lifetime_hours)
    ).isoformat().replace("+00:00", "Z")

    response = _request(
        "POST",
        _BASE,
        management_key=management_key,
        body={"name": name, "limit": limit_usd, "expires_at": expires_at},
    )
    if response.status_code // 100 != 2:
        # Any 2xx, not 200. Creation answers 201, and demanding 200 rejected a mint that had already
        # succeeded — then raised without deleting it, leaving a live key on the miner's account
        # that this code had just been told the hash of. Found on the first live call.
        _delete_orphan(management_key, response, why=f"HTTP {response.status_code}")
        raise ProvisioningError(
            f"the provisioning API refused to mint a key for {name}: HTTP "
            f"{response.status_code} {response.text[:300]}"
        )

    body = response.json()
    secret = str(body.get("key", ""))
    data = body.get("data", {})
    key_hash = str(data.get("hash", ""))
    if not secret or not key_hash:
        # A mint whose secret was not returned is a key that can never be used and can only be
        # deleted by hash. If the hash came back, delete it; otherwise say so, because an
        # unreferenced key on a miner's account is exactly what this design exists to avoid.
        if key_hash:
            revoke(management_key, key_hash)
            raise ProvisioningError(
                "the provisioning API returned a key with no secret; it has been deleted"
            )
        raise ProvisioningError(
            "the provisioning API returned neither a secret nor a hash, so a key may exist on the "
            "miner's account that this validator cannot name or delete. Check the account."
        )

    granted = data.get("limit")
    if granted is None or float(granted) > limit_usd:
        # The provider did not apply the cap we asked for. Deleted rather than used: an uncapped key
        # on someone else's account is the failure this whole path exists to prevent, and it is
        # worse than not running the laboratory at all.
        revoke(management_key, key_hash)
        raise ProvisioningError(
            f"asked for a {limit_usd} credit limit and the key came back with {granted!r}. It has "
            "been deleted — an uncapped key on a miner's account is worse than a laboratory that "
            "does not run."
        )

    _log.info(
        "minted %s for %s: limit %s, expires %s", key_hash[:12], name, granted, expires_at
    )
    return MintedKey(
        secret=secret, key_hash=key_hash, limit_usd=float(granted), expires_at=expires_at
    )


def _delete_orphan(management_key: str, response: Any, *, why: str) -> None:
    """Delete a key the provider may have created on a response we are about to reject.

    Every path that refuses a mint has to come through here first. A refusal that leaves the key
    behind is worse than the condition it refused: the miner has a live key created by a validator
    that then reported failure, so nobody is watching it.
    """
    try:
        key_hash = str(response.json().get("data", {}).get("hash", ""))
    except (ValueError, AttributeError):
        return
    if key_hash:
        _log.error(
            "the provisioning API returned %s but included key %s; deleting it rather than "
            "leaving a key nobody is watching on the miner's account.",
            why,
            key_hash[:12],
        )
        revoke(management_key, key_hash)


def revoke(management_key: str, key_hash: str) -> bool:
    """Delete a minted key. Returns whether the provider confirmed it.

    Never raises. Revocation runs in the cleanup path of a round that may already be failing, and a
    raise there would replace the round's actual error with this one. A failure is logged at error
    level and the key's expiry is what bounds the damage.
    """
    if not key_hash:
        return False
    try:
        response = _request("DELETE", f"{_BASE}/{key_hash}", management_key=management_key)
    except ProvisioningError as error:
        _log.error(
            "could not revoke %s: %s. The key remains until its expiry; that expiry is why it was "
            "set.",
            key_hash[:12],
            error,
        )
        return False
    if response.status_code != 200:
        _log.error(
            "revoking %s returned HTTP %d: %s. The key remains until its expiry.",
            key_hash[:12],
            response.status_code,
            response.text[:200],
        )
        return False
    _log.info("revoked %s", key_hash[:12])
    return True


def read_usage(management_key: str, key_hash: str) -> dict[str, float]:
    """What the provider says this key spent, in dollars.

    3.4.4 reconciles our receipts against the provider's own accounting. Per-key rather than
    per-account is what makes that exact: a miner running anything else on the same account makes an
    account-level total unreconcilable, and a reconciliation nobody can act on is one nobody reads.
    """
    response = _request("GET", f"{_BASE}/{key_hash}", management_key=management_key)
    if response.status_code != 200:
        raise ProvisioningError(
            f"cannot read usage for {key_hash[:12]}: HTTP {response.status_code} "
            f"{response.text[:200]}"
        )
    data = response.json().get("data", {})
    return {
        "usage_usd": float(data.get("usage") or 0.0),
        "limit_usd": float(data.get("limit") or 0.0),
        "limit_remaining_usd": float(data.get("limit_remaining") or 0.0),
    }
