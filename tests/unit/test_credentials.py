"""Credential separation: architecture.md 3.4.4.

The reason this file exists in this shape: with one provider surface, a swapped key
**authenticates and returns a completion**. The provider will not tell us. So every one of
these tests is asserting on a failure that the API would happily let through.
"""

from __future__ import annotations

import pytest

from gateway.credentials import (
    CredentialError,
    CredentialSet,
    MinerCredential,
    ValidatorCredential,
    load_validator_credential,
)
from protocol.receipts import CredentialOwner, Purpose


@pytest.fixture
def credentials() -> CredentialSet:
    credentials = CredentialSet(
        validator=ValidatorCredential(validator_hotkey="5Gvalidator", api_key="sk-or-validator")
    )
    credentials.admit(MinerCredential(miner_hotkey="5Fminer", api_key="sk-or-miner"))
    return credentials


# --------------------------------------------------------------------------
# The swap the provider would not catch
# --------------------------------------------------------------------------


def test_judging_cannot_be_billed_to_a_miner():
    """The failure this whole module exists for.

    A validator that could bill judging to a miner's key could exhaust a rival-sponsored
    laboratory's balance at will, and the equal-budget guarantee would be a fiction. Under one
    provider surface the request would have *succeeded*.
    """
    miner = MinerCredential(miner_hotkey="5Fminer", api_key="sk-or-miner")
    with pytest.raises(CredentialError, match="silently billed"):
        miner.assert_may_fund(Purpose.JUDGING)


def test_challenge_generation_cannot_be_billed_to_a_miner():
    miner = MinerCredential(miner_hotkey="5Fminer", api_key="sk-or-miner")
    with pytest.raises(CredentialError):
        miner.assert_may_fund(Purpose.CHALLENGE_GENERATION)


def test_research_cannot_be_billed_to_the_validator():
    """The mirror image: the validator subsidising a laboratory's research.

    Just as damaging in the other direction — a validator that paid for one miner's inference
    would be funding a competitor's entry.
    """
    validator = ValidatorCredential(validator_hotkey="5Gvalidator", api_key="sk-or-validator")
    with pytest.raises(CredentialError, match="subsidise or starve"):
        validator.assert_may_fund(Purpose.RESEARCH)


def test_search_cannot_be_billed_to_the_validator():
    validator = ValidatorCredential(validator_hotkey="5Gvalidator", api_key="sk-or-validator")
    with pytest.raises(CredentialError):
        validator.assert_may_fund(Purpose.SEARCH)


# --------------------------------------------------------------------------
# Purpose selects; there is no owner parameter to get wrong
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "purpose",
    [Purpose.CHALLENGE_GENERATION, Purpose.CRITIQUE, Purpose.JUDGING, Purpose.PRIOR_ART],
)
def test_every_validator_purpose_resolves_to_the_validator(credentials, purpose):
    assert credentials.for_purpose(purpose).owner is CredentialOwner.VALIDATOR


@pytest.mark.parametrize("purpose", [Purpose.RESEARCH, Purpose.SEARCH, Purpose.SIMULATION])
def test_every_miner_purpose_resolves_to_that_miner(credentials, purpose):
    resolved = credentials.for_purpose(purpose, miner_hotkey="5Fminer")
    assert resolved.owner is CredentialOwner.MINER
    assert resolved.miner_hotkey == "5Fminer"


def test_every_purpose_in_the_enum_has_a_declared_payer(credentials):
    """A purpose added without deciding who pays would otherwise default to somebody.

    Parameterising over the enum rather than over a hand-written list means adding a purpose to
    `protocol.receipts.Purpose` fails here until the payer is decided.
    """
    for purpose in Purpose:
        owner = credentials.owner_of(purpose)
        assert owner in (CredentialOwner.MINER, CredentialOwner.VALIDATOR)


def test_a_miner_purpose_with_no_miner_named_is_refused(credentials):
    """No default. Charging an unnamed research call to the validator is the silent failure."""
    with pytest.raises(CredentialError, match="no default"):
        credentials.for_purpose(Purpose.RESEARCH)


def test_an_unknown_miner_does_not_fall_back_to_the_validator(credentials):
    """The most tempting defect: a missing credential resolved to the one that exists."""
    with pytest.raises(CredentialError, match="did not decrypt"):
        credentials.for_purpose(Purpose.RESEARCH, miner_hotkey="5Fstranger")


def test_a_validator_purpose_that_names_a_miner_is_refused_rather_than_ignored(credentials):
    """Ignoring the name would hide a caller confusion that one provider surface makes silent."""
    with pytest.raises(CredentialError, match="validator-funded but names miner"):
        credentials.for_purpose(Purpose.JUDGING, miner_hotkey="5Fminer")


def test_the_two_credentials_are_distinct_types():
    """Point 1 of 3.4.4: "a parameter can be passed the wrong value, and a type cannot.\""""
    assert MinerCredential is not ValidatorCredential
    assert not issubclass(MinerCredential, ValidatorCredential)
    assert not issubclass(ValidatorCredential, MinerCredential)


def test_there_is_no_resolver_that_takes_an_owner():
    """Asserted structurally, because the absence of an API is the design.

    A `resolve(owner)` added later would satisfy every other test in this file and reintroduce
    exactly the confusion the two types exist to prevent.
    """
    forbidden = {"resolve", "get", "get_key", "credential_for", "by_owner", "__getitem__"}
    assert not forbidden & set(dir(CredentialSet))


# --------------------------------------------------------------------------
# A credential must never reach a log, a repr, or a traceback
# --------------------------------------------------------------------------


def test_a_miner_credential_does_not_print_its_key():
    """A repr reaches every traceback, and tracebacks reach CI transcripts."""
    credential = MinerCredential(miner_hotkey="5Fminer", api_key="sk-or-SECRET-VALUE")
    assert "SECRET-VALUE" not in repr(credential)
    assert "5Fminer" in repr(credential)


def test_a_validator_credential_does_not_print_its_key():
    credential = ValidatorCredential(validator_hotkey="5G", api_key="sk-or-SECRET-VALUE")
    assert "SECRET-VALUE" not in repr(credential)


def test_a_credential_set_does_not_print_any_key():
    credentials = CredentialSet(
        validator=ValidatorCredential(validator_hotkey="5G", api_key="sk-or-VSECRET")
    )
    credentials.admit(MinerCredential(miner_hotkey="5F", api_key="sk-or-MSECRET"))
    printed = repr(credentials)
    assert "VSECRET" not in printed
    assert "MSECRET" not in printed


def test_a_refusal_message_does_not_leak_the_key():
    """Exception text is the most-copied string in any system."""
    miner = MinerCredential(miner_hotkey="5Fminer", api_key="sk-or-SECRET-VALUE")
    with pytest.raises(CredentialError) as raised:
        miner.assert_may_fund(Purpose.JUDGING)
    assert "SECRET-VALUE" not in str(raised.value)


# --------------------------------------------------------------------------
# An absent credential fails loudly rather than falling back
# --------------------------------------------------------------------------


def test_an_empty_miner_key_is_refused_at_construction():
    with pytest.raises(CredentialError, match="bill the wrong account"):
        MinerCredential(miner_hotkey="5Fminer", api_key="")


def test_an_empty_validator_key_is_refused_at_construction():
    with pytest.raises(CredentialError, match="fiction"):
        ValidatorCredential(validator_hotkey="5G", api_key="")


def test_a_missing_validator_key_names_both_ways_to_supply_it():
    with pytest.raises(CredentialError, match="AI_VALIDATOR_OPENROUTER_KEY"):
        load_validator_credential("5G", environ={})


def test_the_key_file_form_strips_the_trailing_newline(tmp_path):
    """`echo key > file` ends in a newline, and a newline in an Authorization header is both an
    auth failure and a request-splitting primitive."""
    path = tmp_path / "key"
    path.write_text("sk-or-fromfile\n")
    credential = load_validator_credential(
        "5G", environ={"AI_VALIDATOR_OPENROUTER_KEY_FILE": str(path)}
    )
    assert credential.api_key == "sk-or-fromfile"


def test_a_key_file_that_does_not_exist_is_an_error_rather_than_a_fallback(tmp_path):
    """Falling through to the environment would silently use a key the operator did not mean."""
    with pytest.raises(CredentialError, match="names no file"):
        load_validator_credential(
            "5G",
            environ={
                "AI_VALIDATOR_OPENROUTER_KEY_FILE": str(tmp_path / "absent"),
                "AI_VALIDATOR_OPENROUTER_KEY": "sk-or-wrong-one",
            },
        )


def test_the_file_form_is_preferred_over_the_environment(tmp_path):
    path = tmp_path / "key"
    path.write_text("sk-or-preferred")
    credential = load_validator_credential(
        "5G",
        environ={
            "AI_VALIDATOR_OPENROUTER_KEY_FILE": str(path),
            "AI_VALIDATOR_OPENROUTER_KEY": "sk-or-ignored",
        },
    )
    assert credential.api_key == "sk-or-preferred"


def test_readmitting_a_miner_replaces_its_credential(credentials):
    """A miner that resubmits in a later round has a new key; the old one must not persist."""
    credentials.admit(MinerCredential(miner_hotkey="5Fminer", api_key="sk-or-round-two"))
    resolved = credentials.for_purpose(Purpose.RESEARCH, miner_hotkey="5Fminer")
    assert resolved.api_key == "sk-or-round-two"
