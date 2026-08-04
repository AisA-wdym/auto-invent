"""Round-scoped key minting: architecture.md 3.4.2, 5.4.1, 23.

A miner funds their own laboratory, so a credential reaches the validator. Which credential is a
choice with a real asymmetry behind it, measured against the live API rather than read from a doc:

    a runtime key can spend and cannot mint
    a management key can mint and cannot spend

So a leaked management key is not itself a loss of funds, and a key minted from one is bounded twice
— by a provider-enforced credit limit and by an expiry — where a runtime key is bounded only by our
own ledger. Two independent enforcers is the point; one of them being our own code is why the other
one matters.

Every test here fakes only the transport. The contract it fakes was recorded from real calls: HTTP
**201** on creation (not 200 — demanding 200 rejected a mint that had already succeeded and left the
key behind), the secret returned only at creation, `data.hash` as the revocation handle.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from gateway.provisioning import (
    ProvisioningError,
    is_management_key,
    mint_round_key,
    read_usage,
    revoke,
)

pytestmark = pytest.mark.determinism

MANAGEMENT = "sk-or-v1-management"
HASH = "a" * 64


class FakeResponse:
    def __init__(self, status: int, body: Any = None) -> None:
        self.status_code = status
        self._body = body if body is not None else {}
        self.text = json.dumps(self._body)

    def json(self) -> Any:
        return self._body


def install(monkeypatch, handler) -> list[tuple[str, str, dict | None]]:
    """Record every call and answer it with `handler(method, url, body)`."""
    seen: list[tuple[str, str, dict | None]] = []

    def request(method: str, url: str, *, management_key: str, body: dict | None = None):
        seen.append((method, url, body))
        return handler(method, url, body)

    monkeypatch.setattr("gateway.provisioning._request", request)
    return seen


def minted_body(*, limit: float = 25.0, key: str = "sk-or-v1-minted", key_hash: str = HASH):
    return {"key": key, "data": {"hash": key_hash, "limit": limit, "limit_remaining": limit}}


# --------------------------------------------------------------------------
# Telling the two kinds apart, for free
# --------------------------------------------------------------------------


def test_a_key_that_lists_keys_is_a_management_key(monkeypatch):
    install(monkeypatch, lambda *_: FakeResponse(200, {"data": []}))
    assert is_management_key(MANAGEMENT)


def test_a_key_that_cannot_list_keys_is_not(monkeypatch):
    """A runtime key answers 401 `Invalid management key` here. Checking costs nothing and spends
    nothing, so a miner who declares the wrong kind is told at admission rather than by every
    inference call failing for a reason that does not name the cause."""
    install(
        monkeypatch,
        lambda *_: FakeResponse(401, {"error": {"message": "Invalid management key"}}),
    )
    assert not is_management_key("sk-or-v1-runtime")


def test_an_unexpected_status_refuses_to_guess(monkeypatch):
    """A 500 is neither "this is a management key" nor "this is not", and guessing either way is a
    submission refused for the wrong reason or a mint attempted against nothing."""
    install(monkeypatch, lambda *_: FakeResponse(503))
    with pytest.raises(ProvisioningError, match="Refusing to guess"):
        is_management_key(MANAGEMENT)


# --------------------------------------------------------------------------
# Minting
# --------------------------------------------------------------------------


def test_a_mint_carries_a_limit_and_an_expiry(monkeypatch):
    """Both are provider-enforced. The limit is the miner's declared cap; the expiry is what makes a
    validator that crashes between minting and revoking survivable rather than permanent."""
    seen = install(monkeypatch, lambda *_: FakeResponse(201, minted_body()))
    minted = mint_round_key(MANAGEMENT, name="uid7", limit_usd=25.0)

    assert minted.secret == "sk-or-v1-minted"
    assert minted.key_hash == HASH
    body = seen[0][2]
    assert body["limit"] == 25.0
    assert body["name"] == "uid7"
    assert body["expires_at"].endswith("Z")


def test_creation_answers_201_and_that_counts_as_success(monkeypatch):
    """The defect the first live call found. `!= 200` rejected a mint that had already succeeded —
    and raised without deleting it, leaving a live key on the miner's account whose hash the code
    had just been handed."""
    install(monkeypatch, lambda *_: FakeResponse(201, minted_body()))
    assert mint_round_key(MANAGEMENT, name="uid7", limit_usd=25.0).key_hash == HASH


def test_a_refused_mint_deletes_anything_the_provider_created(monkeypatch):
    """Every refusal path goes through the same cleanup. A refusal that leaves the key behind is
    worse than the condition it refused: the miner has a live key created by a validator that then
    reported failure, so nobody is watching it."""
    calls: list[str] = []

    def handler(method, url, body):
        calls.append(method)
        if method == "POST":
            return FakeResponse(409, {"data": {"hash": HASH}})
        return FakeResponse(200, {"deleted": True})

    install(monkeypatch, handler)
    with pytest.raises(ProvisioningError, match="refused to mint"):
        mint_round_key(MANAGEMENT, name="uid7", limit_usd=25.0)
    assert "DELETE" in calls, "a key the provider created was left on the miner's account"


def test_a_key_that_came_back_uncapped_is_deleted_rather_than_used(monkeypatch):
    """An uncapped key on someone else's account is the exact failure this path exists to prevent,
    and it is worse than a laboratory that does not run."""
    calls: list[str] = []

    def handler(method, url, body):
        calls.append(method)
        if method == "POST":
            return FakeResponse(201, {"key": "k", "data": {"hash": HASH, "limit": None}})
        return FakeResponse(200, {"deleted": True})

    install(monkeypatch, handler)
    with pytest.raises(ProvisioningError, match="uncapped key"):
        mint_round_key(MANAGEMENT, name="uid7", limit_usd=25.0)
    assert "DELETE" in calls


def test_a_key_capped_higher_than_asked_is_deleted(monkeypatch):
    """A cap the provider widened is not a cap. Lower is fine — that is the miner's own account
    limit binding first, which is stricter than what was asked for."""
    def handler(method, url, body):
        if method == "POST":
            return FakeResponse(201, minted_body(limit=1_000.0))
        return FakeResponse(200, {"deleted": True})

    install(monkeypatch, handler)
    with pytest.raises(ProvisioningError, match="came back with"):
        mint_round_key(MANAGEMENT, name="uid7", limit_usd=25.0)


def test_a_tighter_cap_than_asked_is_accepted(monkeypatch):
    install(monkeypatch, lambda *_: FakeResponse(201, minted_body(limit=10.0)))
    assert mint_round_key(MANAGEMENT, name="uid7", limit_usd=25.0).limit_usd == 10.0


def test_a_zero_limit_is_refused_before_any_call(monkeypatch):
    """Not a safe default in either direction: a key that cannot run the laboratory, or — if the
    provider reads zero as absent — an uncapped key."""
    seen = install(monkeypatch, lambda *_: FakeResponse(201, minted_body()))
    with pytest.raises(ProvisioningError, match="not a safe default"):
        mint_round_key(MANAGEMENT, name="uid7", limit_usd=0)
    assert seen == [], "a request was made for a mint that should have been refused outright"


def test_a_mint_with_no_secret_is_deleted_and_reported(monkeypatch):
    """The secret exists only in the creation response. One that did not arrive is a key that can
    never be used and can only be deleted by hash."""
    def handler(method, url, body):
        if method == "POST":
            return FakeResponse(201, {"data": {"hash": HASH, "limit": 25.0}})
        return FakeResponse(200, {"deleted": True})

    install(monkeypatch, handler)
    with pytest.raises(ProvisioningError, match="no secret; it has been deleted"):
        mint_round_key(MANAGEMENT, name="uid7", limit_usd=25.0)


def test_a_mint_with_neither_secret_nor_hash_says_so_loudly(monkeypatch):
    """The one case that cannot be cleaned up. A key may exist that this validator can neither name
    nor delete, and the message has to say to go and look."""
    install(monkeypatch, lambda *_: FakeResponse(201, {"data": {}}))
    with pytest.raises(ProvisioningError, match="Check the account"):
        mint_round_key(MANAGEMENT, name="uid7", limit_usd=25.0)


# --------------------------------------------------------------------------
# Revocation
# --------------------------------------------------------------------------


def test_revocation_never_raises(monkeypatch):
    """It runs in the cleanup path of a round that may already be failing, and a raise there would
    replace the round's actual error with this one."""
    install(monkeypatch, lambda *_: FakeResponse(404, {"error": {"message": "not found"}}))
    assert revoke(MANAGEMENT, HASH) is False


def test_revocation_survives_an_unreachable_provider(monkeypatch):
    def handler(*_args):
        raise ProvisioningError("network is down")

    install(monkeypatch, handler)
    assert revoke(MANAGEMENT, HASH) is False


def test_revoking_nothing_is_not_an_error(monkeypatch):
    """A laboratory that supplied a runtime key has no minted hash, and the cleanup path walks every
    laboratory."""
    seen = install(monkeypatch, lambda *_: FakeResponse(200))
    assert revoke(MANAGEMENT, "") is False
    assert seen == []


def test_a_confirmed_revocation_reports_true(monkeypatch):
    install(monkeypatch, lambda *_: FakeResponse(200, {"deleted": True}))
    assert revoke(MANAGEMENT, HASH) is True


# --------------------------------------------------------------------------
# The credential never reaches a log
# --------------------------------------------------------------------------


def test_a_minted_key_never_prints_its_secret(monkeypatch):
    """A dataclass repr in a traceback is how a credential reaches a log file, and round logs are
    published (6.3, 22)."""
    install(monkeypatch, lambda *_: FakeResponse(201, minted_body(key="sk-or-v1-verysecret")))
    minted = mint_round_key(MANAGEMENT, name="uid7", limit_usd=25.0)
    assert "verysecret" not in repr(minted)
    assert minted.key_hash[:12] in repr(minted)


# --------------------------------------------------------------------------
# 3.4.4: reconciliation, per key
# --------------------------------------------------------------------------


def test_usage_is_read_per_key(monkeypatch):
    """Per-key rather than per-account is what makes reconciliation exact: a miner running anything
    else on the same account makes an account total unreconcilable, and a reconciliation nobody can
    act on is one nobody reads."""
    install(
        monkeypatch,
        lambda *_: FakeResponse(
            200, {"data": {"usage": 3.5, "limit": 25, "limit_remaining": 21.5}}
        ),
    )
    assert read_usage(MANAGEMENT, HASH) == {
        "usage_usd": 3.5,
        "limit_usd": 25.0,
        "limit_remaining_usd": 21.5,
    }


def test_unreadable_usage_raises_rather_than_reporting_zero(monkeypatch):
    """Zero spend and unknown spend are different facts, and 3.4.4 acts on the difference."""
    install(monkeypatch, lambda *_: FakeResponse(500))
    with pytest.raises(ProvisioningError, match="cannot read usage"):
        read_usage(MANAGEMENT, HASH)
