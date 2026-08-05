"""The hash-chained receipt.

architecture.md 13 makes budget and endpoint violations *hard gates*, and 27 requires 100%
receipt reconciliation. Both depend on the chain being tamper-evident, so these tests are
mostly attempts to alter a receipt without detection.
"""

from __future__ import annotations

import dataclasses

import pytest

from protocol.receipts import (
    GENESIS,
    Call,
    CredentialOwner,
    Purpose,
    Receipt,
    ReceiptBroken,
    Tool,
    chain_head,
    reconcile,
    verify_chain,
)

pytestmark = pytest.mark.determinism


def receipt(calls: int = 3) -> Receipt:
    made = Receipt(
        run_id="run-1",
        miner_hotkey="5Miner",
        bundle_digest="sha256:" + "a" * 64,
        challenge_id="sha256:" + "b" * 64,
        validator_hotkey="5Validator",
    )
    for index in range(calls):
        made.record(
            tool=Tool.LLM,
            provider="openrouter",
            request=f"request-{index}".encode(),
            response=f"response-{index}".encode(),
            rcc=10,
            purpose=Purpose.RESEARCH,
            credential_owner=CredentialOwner.MINER,
            model="anthropic/claude-sonnet-4.5",
            tokens_in=100,
            tokens_out=50,
        )
    return made


# --------------------------------------------------------------------------
# The chain is tamper-evident
# --------------------------------------------------------------------------


def test_a_well_formed_chain_verifies():
    verify_chain(receipt().calls)


def test_the_first_call_chains_to_genesis():
    """A literal rather than an empty string, so an unpopulated field is distinguishable."""
    assert receipt(1).calls[0].previous == GENESIS


def test_removing_a_call_breaks_every_link_after_it():
    """The property a log cannot provide."""
    calls = receipt(4).calls
    tampered = calls[:1] + calls[2:]
    with pytest.raises(ReceiptBroken):
        verify_chain(tampered)


def test_reordering_two_calls_is_detected():
    calls = receipt(3).calls
    swapped = [calls[0], calls[2], calls[1]]
    with pytest.raises(ReceiptBroken):
        verify_chain(swapped)


def test_altering_a_recorded_amount_is_detected():
    """Editing the RCC on one call changes its link and orphans the next."""
    calls = receipt(3).calls
    calls[1] = dataclasses.replace(calls[1], rcc=1)
    with pytest.raises(ReceiptBroken, match="reordered or an entry removed"):
        verify_chain(calls)


def test_altering_a_request_hash_is_detected():
    calls = receipt(3).calls
    calls[0] = dataclasses.replace(calls[0], request_hash="sha256:" + "0" * 64)
    with pytest.raises(ReceiptBroken):
        verify_chain(calls)


def test_a_sequence_gap_is_detected_even_though_the_links_would_verify():
    """The case links alone cannot see.

    A call dropped *before* it was ever linked leaves a consistent chain with a missing
    number — so sequence is checked as well as linkage.
    """
    original = receipt(3).calls
    # Rebuild a chain that is internally consistent but starts at seq 1.
    rebuilt: list[Call] = []
    previous = GENESIS
    for call in original[1:]:
        rebuilt.append(dataclasses.replace(call, previous=previous))
        previous = rebuilt[-1].link()
    with pytest.raises(ReceiptBroken, match="dropped before it was linked"):
        verify_chain(rebuilt)


def test_an_empty_chain_has_a_genesis_head():
    assert chain_head([]) == GENESIS


def test_the_link_body_is_named_explicitly_not_derived():
    """A field added later must not silently change every historical link.

    A chain whose definition moves cannot be verified against history, so the committed
    fields are listed rather than taken from the dataclass.
    """
    body = receipt(1).calls[0].link_body()
    assert set(body) == {
        "seq", "tool", "provider", "model", "revision", "request_hash", "response_hash",
        "rcc", "tokens_in", "tokens_out", "attempts", "purpose", "credential_owner", "previous",
    }


# --------------------------------------------------------------------------
# Bodies are hashed, never kept
# --------------------------------------------------------------------------


def test_request_and_response_content_is_not_retained():
    """The gateway must prove what was asked without archiving every prompt.

    Such an archive is a standing liability, and once published at disclosure it would be a
    corpus of every competitor's technique.
    """
    made = Receipt(
        run_id="r", miner_hotkey="m", bundle_digest="sha256:" + "a" * 64,
        challenge_id="sha256:" + "b" * 64, validator_hotkey="v",
    )
    secret = b"a prompt that reveals the laboratory's whole strategy"
    made.record(
        tool=Tool.LLM, provider="openrouter", request=secret, response=b"reply", rcc=1,
        purpose=Purpose.RESEARCH, credential_owner=CredentialOwner.MINER,
    )
    serialised = str(made.as_document())
    assert secret.decode() not in serialised
    assert "strategy" not in serialised


def test_the_caller_cannot_supply_a_hash_instead_of_the_bytes():
    """`record` takes bytes and hashes them internally.

    That is what guarantees the recorded digest is of what was actually sent, rather than of
    whatever the caller claimed to have sent.
    """
    import inspect

    signature = inspect.signature(Receipt.record)
    assert signature.parameters["request"].annotation == "bytes"
    assert signature.parameters["response"].annotation == "bytes"


# --------------------------------------------------------------------------
# Attempts: a retry storm is not free
# --------------------------------------------------------------------------


def test_attempts_are_recorded_so_failed_tries_are_charged():
    """A failure across three attempts costs three attempts' worth.

    Recording only the successful one would let a laboratory retry without limit at no cost,
    and would make the receipt disagree with the provider's invoice.
    """
    made = receipt(0)
    made.record(
        tool=Tool.LLM, provider="openrouter", request=b"q", response=b"a", rcc=30,
        purpose=Purpose.RESEARCH, credential_owner=CredentialOwner.MINER, attempts=3,
    )
    assert made.totals["attempts"] == 3


def test_zero_attempts_is_refused():
    with pytest.raises(ReceiptBroken, match="would make a retry free"):
        Call(
            seq=0, tool=Tool.LLM, provider="openrouter",
            request_hash="sha256:" + "a" * 64, response_hash="sha256:" + "b" * 64,
            rcc=1, purpose=Purpose.RESEARCH, credential_owner=CredentialOwner.MINER, attempts=0,
        )


def test_negative_rcc_is_refused():
    with pytest.raises(ReceiptBroken, match="negative RCC"):
        Call(
            seq=0, tool=Tool.LLM, provider="openrouter",
            request_hash="sha256:" + "a" * 64, response_hash="sha256:" + "b" * 64,
            rcc=-1, purpose=Purpose.RESEARCH, credential_owner=CredentialOwner.MINER,
        )


# --------------------------------------------------------------------------
# The check one provider surface removes: a swapped credential
# --------------------------------------------------------------------------


def test_billing_research_to_the_validator_is_refused():
    """architecture.md 3.4.4. With one provider, a swapped key *succeeds*.

    It authenticates, returns a completion, and silently bills the wrong party. The API will
    not object, so this is where it stops.
    """
    with pytest.raises(ReceiptBroken, match="must be billed to 'miner'"):
        Call(
            seq=0, tool=Tool.LLM, provider="openrouter",
            request_hash="sha256:" + "a" * 64, response_hash="sha256:" + "b" * 64,
            rcc=1, purpose=Purpose.RESEARCH, credential_owner=CredentialOwner.VALIDATOR,
        )


def test_billing_judging_to_a_miner_is_refused():
    """The direction that matters most: it would drain a rival-sponsored miner's balance."""
    with pytest.raises(ReceiptBroken, match="must be billed to 'validator'"):
        Call(
            seq=0, tool=Tool.LLM, provider="openrouter",
            request_hash="sha256:" + "a" * 64, response_hash="sha256:" + "b" * 64,
            rcc=1, purpose=Purpose.JUDGING, credential_owner=CredentialOwner.MINER,
        )


@pytest.mark.parametrize("purpose", list(Purpose))
def test_every_purpose_has_a_declared_payer(purpose):
    """Adding a purpose must force a decision about who pays for it.

    A purpose absent from the mapping would raise `KeyError` here rather than defaulting to
    someone.
    """
    from protocol.receipts import _PURPOSE_OWNER

    assert purpose in _PURPOSE_OWNER


def test_spend_is_reported_per_account():
    """What reconciliation compares."""
    made = receipt(2)
    by_owner = made.spend_by_owner()
    assert by_owner == {"miner": 20, "validator": 0}


# --------------------------------------------------------------------------
# Reconciliation against the provider
# --------------------------------------------------------------------------


def test_matching_totals_reconcile():
    reconcile(receipt(3), provider_reported_rcc=30)


def test_unaccounted_provider_spend_is_an_incident():
    """A call made outside the receipted path appears in the provider's accounting only."""
    with pytest.raises(ReceiptBroken, match="Unaccounted spend"):
        reconcile(receipt(3), provider_reported_rcc=45)


def test_a_receipt_claiming_more_than_the_provider_saw_is_also_an_incident():
    with pytest.raises(ReceiptBroken, match="Both are incidents"):
        reconcile(receipt(3), provider_reported_rcc=10)


def test_tolerance_defaults_to_zero():
    """A default tolerance is a standing allowance for the discrepancy being looked for."""
    import inspect

    assert inspect.signature(reconcile).parameters["tolerance"].default == 0


def test_an_explicit_tolerance_is_honoured():
    reconcile(receipt(3), provider_reported_rcc=31, tolerance=1)


# --------------------------------------------------------------------------
# The document
# --------------------------------------------------------------------------


def test_the_document_verifies_the_chain_before_emitting_it():
    made = receipt(3)
    made.calls[1] = dataclasses.replace(made.calls[1], rcc=999)
    with pytest.raises(ReceiptBroken):
        made.as_document()


def test_the_signing_payload_covers_the_chain_head():
    made = receipt(2)
    payload = made.signing_payload()
    assert chain_head(made.calls).encode() in payload or chain_head(made.calls) in str(payload)


def test_measured_totals_are_computed_not_claimed():
    """Validators replace self-reported usage with measured usage."""
    made = receipt(3)
    assert made.totals["rcc"] == 30
    assert made.totals["model_calls"] == 3
    assert made.totals["search_calls"] == 0
