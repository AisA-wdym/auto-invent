"""Session tokens: architecture.md 5.4.1.

The token is what a container holds *instead of* a credential. So these tests are about the
three things it must never permit: outliving its episode, being replayed against a second
challenge, and being forged.
"""

from __future__ import annotations

import pytest

from gateway.tokens import SessionToken, TokenError, TokenIssuer

pytestmark = pytest.mark.determinism

SECRET = b"a" * 32
NOW = 1_800_000_000


def issuer() -> TokenIssuer:
    return TokenIssuer(secret=SECRET)


def token(**over) -> SessionToken:
    fields = dict(
        run_id="run-1",
        miner_hotkey="5Fminer",
        bundle_digest="sha256:" + "d" * 64,
        validator_hotkey="5Gvalidator",
        challenge_id="sha256:" + "c" * 64,
        allowed_models=("openai/gpt-5", "anthropic/claude-sonnet-4.5"),
        maximum_rcc=400,
        maximum_requests=500,
        maximum_search_calls=100,
        expires_at=NOW + 1_800,
    )
    fields.update(over)
    return SessionToken(**fields)


def issued(**over) -> str:
    return issuer().issue(token(**over), episode_deadline=NOW + 3_600)


# --------------------------------------------------------------------------
# Round trip
# --------------------------------------------------------------------------


def test_a_token_survives_a_round_trip_unchanged():
    original = token()
    recovered = issuer().verify(issued(), now=NOW)
    assert recovered == original


def test_the_same_token_encodes_identically_every_time():
    """Sorted keys, because the signature is over the bytes: two encoders that ordered keys
    differently would produce two valid signatures for one token."""
    assert issued() == issued()


def test_the_token_carries_no_credential():
    """The whole point of 5.4.1. Asserted on the field set, so a key added later fails here."""
    body = token().body()
    forbidden = {"api_key", "key", "credential", "secret", "authorization", "openrouter_key"}
    assert not forbidden & set(body)
    assert not any("sk-or" in str(value) for value in body.values())


def test_every_field_is_covered_by_the_signature():
    """A field outside the signature is attacker-controlled while looking signed."""
    body = token().body()
    for name in (
        "run_id",
        "miner_hotkey",
        "bundle_digest",
        "validator_hotkey",
        "challenge_id",
        "allowed_models",
        "maximum_rcc",
        "maximum_requests",
        "maximum_search_calls",
        "expires_at",
    ):
        assert name in body, f"{name} is not in the signed body"


# --------------------------------------------------------------------------
# Forgery
# --------------------------------------------------------------------------


def test_a_tampered_body_does_not_verify():
    raw = issued()
    payload, _, signature = raw.partition(".")
    forged = payload[:-4] + "AAAA" + "." + signature
    with pytest.raises(TokenError, match="signature does not verify"):
        issuer().verify(forged, now=NOW)


def test_a_token_signed_with_another_secret_does_not_verify():
    """Restarting the gateway invalidates outstanding tokens, which is the safe direction."""
    other = TokenIssuer(secret=b"b" * 32)
    with pytest.raises(TokenError, match="signature does not verify"):
        other.verify(issued(), now=NOW)


def test_a_token_with_no_signature_is_refused():
    with pytest.raises(TokenError, match="no signature"):
        issuer().verify("just-a-payload", now=NOW)


def test_a_raised_ceiling_does_not_verify():
    """The attack the signature exists for: edit maximum_rcc, keep the signature."""
    import base64
    import json

    payload, _, signature = issued().partition(".")
    body = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    body["maximum_rcc"] = 10_000_000
    forged = base64.urlsafe_b64encode(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).rstrip(b"=")
    with pytest.raises(TokenError, match="signature does not verify"):
        issuer().verify(f"{forged.decode()}.{signature}", now=NOW)


def test_a_short_secret_is_refused_at_construction():
    """This secret authorises spending, so it needs a full HMAC block of entropy."""
    with pytest.raises(TokenError, match="at least 32"):
        TokenIssuer(secret=b"short")


# --------------------------------------------------------------------------
# Expiry
# --------------------------------------------------------------------------


def test_an_expired_token_is_refused():
    with pytest.raises(TokenError, match="expired"):
        issuer().verify(issued(), now=NOW + 1_801)


def test_a_token_at_exactly_its_expiry_is_refused():
    """`>=` rather than `>`: a token valid at the instant it expires is valid for a whole block."""
    with pytest.raises(TokenError, match="expired"):
        issuer().verify(issued(), now=NOW + 1_800)


def test_a_token_that_would_outlive_its_episode_is_refused_at_issue():
    """10 requires forced termination when the episode closes. A token valid past that is a
    container that can still spend after it was supposed to be killed."""
    with pytest.raises(TokenError, match="past the episode deadline"):
        issuer().issue(token(expires_at=NOW + 7_200), episode_deadline=NOW + 3_600)


def test_a_token_with_no_rcc_ceiling_is_refused_at_issue():
    with pytest.raises(TokenError, match="no RCC ceiling"):
        issuer().issue(token(maximum_rcc=0), episode_deadline=NOW + 3_600)


def test_an_empty_allowlist_is_refused_rather_than_read_as_unrestricted():
    """Gate 13.3 makes undeclared model use fatal; an empty list read as "anything" defeats it."""
    with pytest.raises(TokenError, match="empty allowlist"):
        issuer().issue(token(allowed_models=()), episode_deadline=NOW + 3_600)


# --------------------------------------------------------------------------
# Replay across challenges
# --------------------------------------------------------------------------


def test_a_token_is_refused_against_a_different_challenge():
    """7.1 scores every laboratory on the same challenge instances. A replayable token would let
    one spend its whole ceiling on the challenge it found easiest."""
    with pytest.raises(TokenError, match="bound to challenge"):
        issuer().verify(issued(), now=NOW, challenge_id="sha256:" + "e" * 64)


def test_a_token_verifies_against_its_own_challenge():
    recovered = issuer().verify(issued(), now=NOW, challenge_id="sha256:" + "c" * 64)
    assert recovered.run_id == "run-1"


def test_omitting_the_challenge_skips_the_binding_check():
    """`/v1/usage` has no challenge to check against, and refusing it would be worse than
    checking nothing there."""
    assert issuer().verify(issued(), now=NOW).challenge_id.endswith("c" * 64)


# --------------------------------------------------------------------------
# Model allowlist
# --------------------------------------------------------------------------


def test_a_declared_model_is_permitted():
    assert token().permits_model("openai/gpt-5")


def test_an_undeclared_model_is_not_permitted():
    assert not token().permits_model("openai/gpt-5-turbo-secret")


def test_the_allowlist_is_exact_rather_than_a_prefix_match():
    """A prefix match on `anthropic/` would admit a model the miner never declared."""
    assert not token().permits_model("anthropic/claude-opus-4")


# --------------------------------------------------------------------------
# Version
# --------------------------------------------------------------------------


def test_an_unknown_body_version_is_refused_rather_than_reinterpreted():
    import base64
    import hmac
    import json
    from hashlib import sha256

    body = token().body()
    body["v"] = 99
    payload = base64.urlsafe_b64encode(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).rstrip(b"=")
    signature = hmac.new(SECRET, payload, sha256).hexdigest()
    with pytest.raises(TokenError, match="version"):
        issuer().verify(f"{payload.decode()}.{signature}", now=NOW)
