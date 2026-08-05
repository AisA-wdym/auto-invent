"""The RCG's HTTP surface. The only network address a laboratory can reach.

architecture.md 3.4 and 10. The sandbox has no route to the internet; it has a route to this
process. So every request here arrives from code a miner wrote, and is treated accordingly:
authenticated by session token, bounded by the ledger, and recorded on a receipt chain.

## Shape of a request

    laboratory --(session token)--> RCG --(miner's OpenRouter key)--> OpenRouter

The laboratory names a model and a purpose. It never names a credential, because it has none —
`gateway.credentials` selects the payer from the purpose, and the purpose a session token can
carry is restricted to the miner-funded set.

## What is deliberately *not* here

No endpoint that returns a credential, echoes one, or reports which one paid. Disclosure
publishes bundle source and portfolios, and a laboratory that could read its own key could print
it into its portfolio, at which point it is published.

No endpoint that raises a ceiling. `/v1/usage` reports remaining budget because a laboratory
that cannot see its budget will either waste it or stop early, but the ceiling itself comes from
the signed token and cannot be changed by anything reachable from the sandbox.

## Lifecycle routes are runner-authenticated, not token-authenticated

`admit` and `close` are called by the validator's runner, not by the laboratory, and they use a
separate bearer secret. A container that could admit its own run could re-admit it and reset its
spend, which is what a separate secret prevents.
"""

from __future__ import annotations

import hmac
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from fastapi import Body, Depends, FastAPI, Header, HTTPException, Request
from pydantic import Field

from gateway.adapters.openrouter import (
    AdapterError,
    ModelPin,
    OpenRouterAdapter,
    budget_error_status,
)
from gateway.credentials import CredentialError, CredentialSet, MinerCredential
from gateway.metering import BudgetExceeded, Ledger, PriceTable
from gateway.tokens import SessionToken, TokenError, TokenIssuer
from protocol.model_base import ProtocolModel
from protocol.receipts import Purpose, Receipt

__all__ = ["GatewayState", "build_app"]

_log = logging.getLogger(__name__)

#: Purposes a session token may request. The validator-funded purposes are absent by
#: construction: a laboratory that could ask for `judging` would be asking the validator to pay,
#: and `gateway.credentials` would refuse — but refusing at the edge means the request never
#: reaches a credential resolver at all.
_SANDBOX_PURPOSES = {
    "research": Purpose.RESEARCH,
    "search": Purpose.SEARCH,
    "simulation": Purpose.SIMULATION,
}


@dataclass
class GatewayState:
    """Everything the handlers need, assembled once at startup.

    Held on `app.state` rather than in module globals so a test can build an independent
    gateway, and so two gateways in one process cannot share a ledger.
    """

    credentials: CredentialSet
    prices: PriceTable
    issuer: TokenIssuer
    ledger: Ledger = field(default_factory=Ledger)
    #: run_id -> receipt under construction.
    receipts: dict[str, Receipt] = field(default_factory=dict)
    #: run_id -> the adapter bound to that run's miner credential.
    adapters: dict[str, OpenRouterAdapter] = field(default_factory=dict)
    #: Bearer secret for the runner-only lifecycle routes.
    runner_secret: str = field(default="", repr=False)
    base_url: str | None = None
    #: Injected in tests; `None` means build a real OpenRouter client per run.
    transport_factory: Any | None = field(default=None, repr=False)

    def adapter_for(self, token: SessionToken) -> OpenRouterAdapter:
        adapter = self.adapters.get(token.run_id)
        if adapter is None:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"run {token.run_id} has no adapter: it was never admitted, or it has been "
                    "closed. A closed run cannot spend again."
                ),
            )
        return adapter

    def receipt_for(self, token: SessionToken) -> Receipt:
        receipt = self.receipts.get(token.run_id)
        if receipt is None:
            raise HTTPException(status_code=409, detail=f"run {token.run_id} has no open receipt")
        return receipt


# --------------------------------------------------------------------------
# Request and response bodies
# --------------------------------------------------------------------------


class _Body(ProtocolModel):
    """Base for every request body.

    `ProtocolModel` lifts pydantic's `model_` namespace reservation, which `model_slug` and
    `model_snapshot` would otherwise trip on every import. `extra="forbid"` is added here rather
    than inherited: a laboratory that sent `maximum_rcc` in its request body should be told the
    field is not accepted, because the alternative is a miner who believes they raised their own
    ceiling and a silence that lets them believe it.
    """

    model_config = ProtocolModel.model_config | {"extra": "forbid"}


class Message(_Body):
    role: str
    content: str


class CompletionRequest(_Body):
    challenge_id: str
    purpose: str = "research"
    model_slug: str
    model_snapshot: str = ""
    messages: list[Message]
    max_tokens: int = Field(gt=0, le=200_000)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    tools: list[dict[str, Any]] | None = None
    response_format: dict[str, Any] | None = None


class CompletionResponse(_Body):
    content: str
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    rcc_charged: int
    rcc_remaining: int


class SearchRequest(_Body):
    challenge_id: str
    query: str
    model_slug: str
    model_snapshot: str = ""
    max_results: int = Field(default=10, ge=1, le=50)


class SearchResponse(_Body):
    content: str
    results: list[dict[str, Any]]
    rcc_charged: int
    rcc_remaining: int


class UsageResponse(_Body):
    rcc_spent: int
    rcc_remaining: int
    maximum_rcc: int


class AdmitRequest(_Body):
    """Runner-only. Opens a run's ledger and binds it to a miner credential."""

    run_id: str
    miner_hotkey: str
    bundle_digest: str
    validator_hotkey: str
    challenge_id: str
    api_key: str
    allowed_models: list[str]
    maximum_rcc: int = Field(gt=0)
    maximum_requests: int = Field(gt=0)
    maximum_search_calls: int = Field(ge=0)
    expires_at: int
    episode_deadline: int
    declared_spend_cap_usd: int = 0


class AdmitResponse(_Body):
    session_token: str


class CloseResponse(_Body):
    run_id: str
    totals: dict[str, int]
    chain_head: str
    receipt: dict[str, Any]


# --------------------------------------------------------------------------
# Authentication
# --------------------------------------------------------------------------


def _state(request: Request) -> GatewayState:
    return request.app.state.gateway


async def _session(
    request: Request, authorization: str = Header(default="")
) -> tuple[GatewayState, SessionToken]:
    """Verify the laboratory's session token.

    The clock is read exactly once and passed down, so every check in this request sees the same
    instant. Two `time.time()` calls can straddle an expiry boundary, and a token that is valid
    for the budget check but expired for the model check is a confusing partial failure.
    """
    state = _state(request)
    scheme, _, raw = authorization.partition(" ")
    if scheme.lower() != "bearer" or not raw:
        raise HTTPException(status_code=401, detail="expected `Authorization: Bearer <token>`")
    try:
        token = state.issuer.verify(raw, now=int(time.time()))
    except TokenError as error:
        raise HTTPException(status_code=401, detail=str(error)) from error
    return state, token


async def _runner(request: Request, authorization: str = Header(default="")) -> GatewayState:
    """Verify the validator runner's bearer secret.

    Separate from the session-token path on purpose. `compare_digest` because this secret guards
    ledger creation, and a container that could time the comparison could recover it.
    """
    state = _state(request)
    scheme, _, raw = authorization.partition(" ")
    if scheme.lower() != "bearer" or not raw:
        raise HTTPException(status_code=401, detail="runner routes need a bearer secret")
    if not state.runner_secret or not hmac.compare_digest(raw, state.runner_secret):
        raise HTTPException(status_code=403, detail="not the validator runner")
    return state


# --------------------------------------------------------------------------
# The app
# --------------------------------------------------------------------------


def build_app(state: GatewayState) -> FastAPI:
    """Compose the gateway. One factory so tests and production build the same object."""
    app = FastAPI(
        title="Research Compute Gateway",
        description="Brokers every external call a laboratory makes. architecture.md 3.4.",
        version="AIL-3.0",
    )
    app.state.gateway = state

    @app.get("/health")
    async def health() -> dict[str, Any]:
        """Liveness. Reports counts, never a credential or a key fingerprint."""
        return {
            "status": "ok",
            "protocol_version": "AIL-3.0",
            "open_runs": len(state.adapters),
            "admitted_miners": len(state.credentials.miners),
        }

    @app.post("/v1/runs", response_model=AdmitResponse)
    async def admit(
        body: AdmitRequest, gateway: GatewayState = Depends(_runner)
    ) -> AdmitResponse:
        """Open a run: admit its credential, open its ledger, issue its token.

        Runner-only. Idempotency is *not* offered: `Ledger.admit` refuses to reset an open run's
        spend, and a second admit that silently succeeded would be a budget reset with extra
        steps.
        """
        if body.run_id in gateway.adapters:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"run {body.run_id} is already open. Re-admitting would reset its spend, "
                    "which is a re-openable run gate."
                ),
            )
        try:
            credential = MinerCredential(
                miner_hotkey=body.miner_hotkey,
                api_key=body.api_key,
                declared_spend_cap_usd=body.declared_spend_cap_usd,
            )
        except CredentialError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        gateway.credentials.admit(credential)

        token = SessionToken(
            run_id=body.run_id,
            miner_hotkey=body.miner_hotkey,
            bundle_digest=body.bundle_digest,
            validator_hotkey=body.validator_hotkey,
            challenge_id=body.challenge_id,
            allowed_models=tuple(body.allowed_models),
            maximum_rcc=body.maximum_rcc,
            maximum_requests=body.maximum_requests,
            maximum_search_calls=body.maximum_search_calls,
            expires_at=body.expires_at,
        )
        try:
            raw = gateway.issuer.issue(token, episode_deadline=body.episode_deadline)
        except TokenError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        gateway.ledger.admit(
            body.run_id,
            maximum_rcc=body.maximum_rcc,
            maximum_requests=body.maximum_requests,
            maximum_search_calls=body.maximum_search_calls,
        )
        gateway.receipts[body.run_id] = Receipt(
            run_id=body.run_id,
            miner_hotkey=body.miner_hotkey,
            bundle_digest=body.bundle_digest,
            challenge_id=body.challenge_id,
            validator_hotkey=body.validator_hotkey,
        )
        transport = (
            gateway.transport_factory(credential) if gateway.transport_factory else None
        )
        gateway.adapters[body.run_id] = OpenRouterAdapter(
            credential=credential,
            prices=gateway.prices,
            ledger=gateway.ledger,
            base_url=gateway.base_url or OpenRouterAdapter.base_url,
            transport=transport,
        )
        return AdmitResponse(session_token=raw)

    @app.post("/v1/runs/{run_id}/close", response_model=CloseResponse)
    async def close(run_id: str, gateway: GatewayState = Depends(_runner)) -> CloseResponse:
        """Close a run and return its sealed receipt. Runner-only.

        The receipt is returned here rather than fetched later because this is the moment the
        chain is complete and the run's adapter goes away. `as_document` verifies the chain, so
        a close that returns is a close whose evidence held.
        """
        receipt = gateway.receipts.pop(run_id, None)
        gateway.adapters.pop(run_id, None)
        totals = gateway.ledger.close(run_id)
        if receipt is None:
            raise HTTPException(status_code=404, detail=f"no open run {run_id}")
        document = receipt.as_document()
        _log.info("run %s closed: %s", run_id, totals)
        return CloseResponse(
            run_id=run_id,
            totals=totals,
            chain_head=document["chain_head"],
            receipt=document,
        )

    @app.post("/v1/llm", response_model=CompletionResponse)
    async def llm(
        body: CompletionRequest = Body(...),
        session: tuple[GatewayState, SessionToken] = Depends(_session),
    ) -> CompletionResponse:
        gateway, token = session
        _assert_challenge(token, body.challenge_id)
        purpose = _purpose(body.purpose)
        adapter = gateway.adapter_for(token)
        try:
            outcome = await adapter.complete(
                run_id=token.run_id,
                receipt=gateway.receipt_for(token),
                purpose=purpose,
                pin=ModelPin(slug=body.model_slug, snapshot=body.model_snapshot),
                messages=[message.model_dump() for message in body.messages],
                max_tokens=body.max_tokens,
                temperature=body.temperature,
                allowed_models=frozenset(token.allowed_models),
                tools=body.tools,
                response_format=body.response_format,
            )
        except (BudgetExceeded, AdapterError, CredentialError) as error:
            raise HTTPException(
                status_code=budget_error_status(error), detail=str(error)
            ) from error
        return CompletionResponse(
            content=outcome.text,
            tool_calls=list(outcome.tool_calls),
            rcc_charged=outcome.rcc,
            rcc_remaining=gateway.ledger.remaining(token.run_id),
        )

    @app.post("/v1/search", response_model=SearchResponse)
    async def search(
        body: SearchRequest = Body(...),
        session: tuple[GatewayState, SessionToken] = Depends(_session),
    ) -> SearchResponse:
        gateway, token = session
        _assert_challenge(token, body.challenge_id)
        adapter = gateway.adapter_for(token)
        try:
            outcome = await adapter.search(
                run_id=token.run_id,
                receipt=gateway.receipt_for(token),
                purpose=Purpose.SEARCH,
                query=body.query,
                pin=ModelPin(slug=body.model_slug, snapshot=body.model_snapshot),
                max_results=body.max_results,
            )
        except (BudgetExceeded, AdapterError, CredentialError) as error:
            raise HTTPException(
                status_code=budget_error_status(error), detail=str(error)
            ) from error
        return SearchResponse(
            content=outcome.text,
            results=list(outcome.results),
            rcc_charged=outcome.rcc,
            rcc_remaining=gateway.ledger.remaining(token.run_id),
        )

    @app.get("/v1/usage", response_model=UsageResponse)
    async def usage(
        session: tuple[GatewayState, SessionToken] = Depends(_session),
    ) -> UsageResponse:
        """What the run has spent. Readable by the laboratory, changeable by nothing.

        Offered because a laboratory that cannot see its remaining budget either wastes it on a
        call that will be refused or stops early to be safe — and stopping early makes the
        comparison between laboratories about caution rather than architecture.
        """
        gateway, token = session
        return UsageResponse(
            rcc_spent=gateway.ledger.spent(token.run_id),
            rcc_remaining=gateway.ledger.remaining(token.run_id),
            maximum_rcc=token.maximum_rcc,
        )

    return app


def _assert_challenge(token: SessionToken, challenge_id: str) -> None:
    if challenge_id != token.challenge_id:
        raise HTTPException(
            status_code=403,
            detail=(
                f"this token is bound to challenge {token.challenge_id}; the request names "
                f"{challenge_id}. A token replayable across challenges could spend one "
                "challenge's ceiling on another."
            ),
        )


def _purpose(name: str) -> Purpose:
    purpose = _SANDBOX_PURPOSES.get(name)
    if purpose is None:
        raise HTTPException(
            status_code=403,
            detail=(
                f"purpose {name!r} is not available to a session token. A laboratory may fund "
                f"{sorted(_SANDBOX_PURPOSES)}; validator-funded purposes are not reachable from "
                "the sandbox at all."
            ),
        )
    return purpose
