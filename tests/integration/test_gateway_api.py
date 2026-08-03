"""The composed gateway, over HTTP: architecture.md 3.4, 5.4.1, 13.

Unit tests can show that each guard works when called. This shows the guards are *on the request
path* — which is a different claim, and the one that failed in the predecessor: a lifecycle route
returned 423 on every session because nothing opened the run, and every unit test passed.

So this drives the real ASGI app through a real client. The only substitution is the provider
transport, because a test suite that spent money on every run would be a test suite nobody runs.
"""

from __future__ import annotations

import json
import pathlib
import time
from dataclasses import dataclass, field
from typing import Any

import pytest
from fastapi.testclient import TestClient

from gateway.api import GatewayState, build_app
from gateway.credentials import CredentialSet, ValidatorCredential
from gateway.metering import Ledger, PriceTable
from gateway.tokens import TokenIssuer
from protocol.receipts import verify_chain

SEASON = json.loads(pathlib.Path("config/season.example.json").read_text())
RUNNER_SECRET = "runner-secret-for-tests"
CHALLENGE = "sha256:" + "c" * 64
MODEL = "openai/gpt-5"


@dataclass
class FakeUsage:
    prompt_tokens: int
    completion_tokens: int


@dataclass
class FakeMessage:
    content: str
    tool_calls: list = field(default_factory=list)
    annotations: list = field(default_factory=list)


@dataclass
class FakeChoice:
    message: FakeMessage


@dataclass
class FakeCompletion:
    choices: list
    usage: FakeUsage


@dataclass
class FakeTransport:
    """Stands in for `AsyncOpenAI(...).chat.completions`.

    Records every request so a test can assert on what would have been sent — which is how the
    "does the allowlist actually gate the wire" question gets answered without a network.
    """

    tokens_in: int = 100
    tokens_out: int = 50
    reply: str = "a portfolio"
    raise_with: Exception | None = None
    seen: list[dict[str, Any]] = field(default_factory=list)

    async def create(self, **request: Any) -> FakeCompletion:
        self.seen.append(request)
        if self.raise_with is not None:
            raise self.raise_with
        annotations = []
        if "extra_body" in request and "plugins" in request["extra_body"]:
            annotations = [
                {
                    "url_citation": {
                        "url": "https://example.org/paper",
                        "title": "A paper",
                        "content": "…",
                    }
                }
            ]
        return FakeCompletion(
            choices=[FakeChoice(message=FakeMessage(content=self.reply, annotations=annotations))],
            usage=FakeUsage(prompt_tokens=self.tokens_in, completion_tokens=self.tokens_out),
        )


@pytest.fixture
def transport() -> FakeTransport:
    return FakeTransport()


@pytest.fixture
def client(transport: FakeTransport) -> TestClient:
    state = GatewayState(
        credentials=CredentialSet(
            validator=ValidatorCredential(validator_hotkey="5Gvalidator", api_key="sk-or-validator")
        ),
        prices=PriceTable.from_season(SEASON),
        issuer=TokenIssuer(secret=b"t" * 32),
        ledger=Ledger(),
        runner_secret=RUNNER_SECRET,
        transport_factory=lambda _credential: transport,
    )
    return TestClient(build_app(state))


def admit(client: TestClient, **over) -> str:
    body = dict(
        run_id="run-1",
        miner_hotkey="5Fminer",
        bundle_digest="sha256:" + "d" * 64,
        validator_hotkey="5Gvalidator",
        challenge_id=CHALLENGE,
        api_key="sk-or-miner-key",
        allowed_models=[MODEL],
        maximum_rcc=100_000,
        maximum_requests=50,
        maximum_search_calls=10,
        expires_at=int(time.time()) + 1_800,
        episode_deadline=int(time.time()) + 3_600,
    )
    body.update(over)
    response = client.post(
        "/v1/runs", json=body, headers={"Authorization": f"Bearer {RUNNER_SECRET}"}
    )
    assert response.status_code == 200, response.text
    return response.json()["session_token"]


def llm(client: TestClient, token: str, **over) -> Any:
    body = dict(
        challenge_id=CHALLENGE,
        purpose="research",
        model_slug=MODEL,
        messages=[{"role": "user", "content": "invent something"}],
        max_tokens=1_024,
    )
    body.update(over)
    return client.post("/v1/llm", json=body, headers={"Authorization": f"Bearer {token}"})


# --------------------------------------------------------------------------
# The happy path exists, which is the claim the predecessor got wrong
# --------------------------------------------------------------------------


def test_a_laboratory_can_actually_make_a_call(client, transport):
    """The end-to-end claim: admit, call, get an answer, see the charge."""
    token = admit(client)
    response = llm(client, token)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["content"] == "a portfolio"
    assert body["rcc_charged"] > 0
    assert body["rcc_remaining"] < 100_000
    assert len(transport.seen) == 1


def test_search_returns_citations(client):
    """Gate 13.8 checks citations against searches actually made, so they have to come back."""
    token = admit(client)
    response = client.post(
        "/v1/search",
        json={"challenge_id": CHALLENGE, "query": "prior art", "model_slug": MODEL},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["results"][0]["url"] == "https://example.org/paper"


def test_usage_is_visible_to_the_laboratory(client):
    """A laboratory that cannot see its budget wastes it or stops early, and stopping early makes
    the comparison about caution rather than architecture."""
    token = admit(client)
    llm(client, token)
    usage = client.get("/v1/usage", headers={"Authorization": f"Bearer {token}"}).json()
    assert usage["rcc_spent"] > 0
    assert usage["maximum_rcc"] == 100_000
    assert usage["rcc_spent"] + usage["rcc_remaining"] == 100_000


def test_closing_a_run_returns_a_receipt_whose_chain_verifies(client):
    token = admit(client)
    llm(client, token)
    llm(client, token)
    response = client.post(
        "/v1/runs/run-1/close", headers={"Authorization": f"Bearer {RUNNER_SECRET}"}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["totals"]["requests"] == 2
    assert len(body["receipt"]["calls"]) == 2
    assert body["receipt"]["chain_head"] == body["chain_head"]


def test_health_reports_no_credential_material(client):
    admit(client)
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert not any("sk-or" in str(value) for value in body.values())


# --------------------------------------------------------------------------
# The laboratory cannot reach the validator's account
# --------------------------------------------------------------------------


@pytest.mark.parametrize("purpose", ["judging", "challenge_generation", "critique", "prior_art"])
def test_a_validator_funded_purpose_is_unreachable_from_the_sandbox(client, purpose):
    """3.4.4: a laboratory that could ask for `judging` would be asking the validator to pay.
    Refused at the edge, so the request never reaches a credential resolver at all."""
    token = admit(client)
    response = llm(client, token, purpose=purpose)
    assert response.status_code == 403
    assert "not available to a session token" in response.json()["detail"]


def test_no_route_returns_a_credential(client):
    """Asserted over the whole route table, because 6.3 publishes what a laboratory outputs."""
    token = admit(client)
    llm(client, token)
    for path in ("/health", "/v1/usage"):
        text = client.get(path, headers={"Authorization": f"Bearer {token}"}).text
        assert "sk-or" not in text
        assert "api_key" not in text


# --------------------------------------------------------------------------
# Hard gates on the request path
# --------------------------------------------------------------------------


def test_an_undeclared_model_is_refused_before_the_request_is_sent(client, transport):
    """Gate 13.3. Refused rather than sent-and-discounted, because a sent call was billed."""
    token = admit(client)
    response = llm(client, token, model_slug="anthropic/claude-opus-4")
    assert response.status_code == 403
    assert "declared model manifest" in response.json()["detail"]
    assert transport.seen == [], "the request must not have reached the provider"


def test_a_token_cannot_be_replayed_against_another_challenge(client):
    """7.1: every laboratory is scored on the same challenge instances."""
    token = admit(client)
    response = llm(client, token, challenge_id="sha256:" + "e" * 64)
    assert response.status_code == 403
    assert "bound to challenge" in response.json()["detail"]


def test_the_budget_ceiling_refuses_with_429_rather_than_403(client):
    """A laboratory needs to tell "you have spent your ceiling" from "that was forbidden": the
    first means submit what you have, the second means the request was wrong."""
    token = admit(client, maximum_rcc=200)
    statuses = [llm(client, token).status_code for _ in range(6)]
    assert 429 in statuses, statuses


def test_the_request_ceiling_is_enforced(client):
    token = admit(client, maximum_requests=2, maximum_rcc=10_000_000)
    assert llm(client, token).status_code == 200
    assert llm(client, token).status_code == 200
    assert llm(client, token).status_code == 429


def test_spend_never_exceeds_the_ceiling_by_more_than_one_call(client):
    """The reserve-then-settle guarantee, measured through the API."""
    token = admit(client, maximum_rcc=5_000)
    for _ in range(20):
        llm(client, token)
    usage = client.get("/v1/usage", headers={"Authorization": f"Bearer {token}"}).json()
    single_call_ceiling = 5_000 + PriceTable.from_season(SEASON).rcc_for_tokens(1_000, 1_024)
    assert usage["rcc_spent"] <= single_call_ceiling


# --------------------------------------------------------------------------
# Authentication
# --------------------------------------------------------------------------


def test_a_call_with_no_token_is_refused(client):
    response = client.post(
        "/v1/llm",
        json={
            "challenge_id": CHALLENGE,
            "model_slug": MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 10,
        },
    )
    assert response.status_code == 401


def test_a_forged_token_is_refused(client):
    admit(client)
    assert llm(client, "not.a.token").status_code == 401


def test_a_laboratory_cannot_admit_its_own_run(client):
    """The re-openable run gate. A container that could admit its own run could reset its spend."""
    token = admit(client)
    response = client.post(
        "/v1/runs",
        json={
            "run_id": "run-2",
            "miner_hotkey": "5Fminer",
            "bundle_digest": "sha256:" + "d" * 64,
            "validator_hotkey": "5Gvalidator",
            "challenge_id": CHALLENGE,
            "api_key": "sk-or-miner-key",
            "allowed_models": [MODEL],
            "maximum_rcc": 10_000_000,
            "maximum_requests": 999,
            "maximum_search_calls": 999,
            "expires_at": int(time.time()) + 1_800,
            "episode_deadline": int(time.time()) + 3_600,
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 403


def test_a_laboratory_cannot_close_its_own_run(client):
    token = admit(client)
    response = client.post(
        "/v1/runs/run-1/close", headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 403


def test_readmitting_an_open_run_is_refused(client):
    """Re-admission with a raised ceiling is the whole attack; 409 rather than a silent success."""
    admit(client)
    response = client.post(
        "/v1/runs",
        json={
            "run_id": "run-1",
            "miner_hotkey": "5Fminer",
            "bundle_digest": "sha256:" + "d" * 64,
            "validator_hotkey": "5Gvalidator",
            "challenge_id": CHALLENGE,
            "api_key": "sk-or-miner-key",
            "allowed_models": [MODEL],
            "maximum_rcc": 10_000_000,
            "maximum_requests": 999,
            "maximum_search_calls": 999,
            "expires_at": int(time.time()) + 1_800,
            "episode_deadline": int(time.time()) + 3_600,
        },
        headers={"Authorization": f"Bearer {RUNNER_SECRET}"},
    )
    assert response.status_code == 409
    assert "re-openable run gate" in response.json()["detail"]


def test_a_closed_run_cannot_spend_again(client):
    token = admit(client)
    client.post("/v1/runs/run-1/close", headers={"Authorization": f"Bearer {RUNNER_SECRET}"})
    response = llm(client, token)
    assert response.status_code == 409


# --------------------------------------------------------------------------
# The receipt records what happened, including failures
# --------------------------------------------------------------------------


def test_a_provider_error_is_still_recorded_and_still_charged(client, transport):
    """A provider *response* was billed. Releasing it would make retries free."""
    token = admit(client)
    llm(client, token)

    class Status(Exception):
        status_code = 400

    transport.raise_with = Status("bad request")
    assert llm(client, token).status_code == 502

    transport.raise_with = None
    closed = client.post(
        "/v1/runs/run-1/close", headers={"Authorization": f"Bearer {RUNNER_SECRET}"}
    ).json()
    assert len(closed["receipt"]["calls"]) == 2, "the failed call must appear on the receipt"


def test_the_receipt_attributes_every_call_to_the_miner(client):
    """3.4.4 point 3: `credential_owner` on every call, so per-account totals are comparable."""
    token = admit(client)
    llm(client, token)
    client.post(
        "/v1/search",
        json={"challenge_id": CHALLENGE, "query": "q", "model_slug": MODEL},
        headers={"Authorization": f"Bearer {token}"},
    )
    closed = client.post(
        "/v1/runs/run-1/close", headers={"Authorization": f"Bearer {RUNNER_SECRET}"}
    ).json()
    owners = {call["credential_owner"] for call in closed["receipt"]["calls"]}
    assert owners == {"miner"}


def test_the_receipt_chain_detects_a_removed_call(client):
    """Reconstructed from the returned document, then broken deliberately."""
    from protocol.receipts import Call, CredentialOwner, Purpose, ReceiptBroken, Tool

    token = admit(client)
    for _ in range(3):
        llm(client, token)
    closed = client.post(
        "/v1/runs/run-1/close", headers={"Authorization": f"Bearer {RUNNER_SECRET}"}
    ).json()

    calls = [
        Call(
            seq=body["seq"],
            tool=Tool(body["tool"]),
            provider=body["provider"],
            request_hash=body["request_hash"],
            response_hash=body["response_hash"],
            rcc=body["rcc"],
            purpose=Purpose(body["purpose"]),
            credential_owner=CredentialOwner(body["credential_owner"]),
            previous=body["previous"],
            model=body["model"],
            revision=body["revision"],
            tokens_in=body["tokens_in"],
            tokens_out=body["tokens_out"],
            attempts=body["attempts"],
        )
        for body in closed["receipt"]["calls"]
    ]
    verify_chain(calls)
    with pytest.raises(ReceiptBroken):
        verify_chain([calls[0], calls[2]])
