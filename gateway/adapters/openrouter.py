"""The one provider adapter. OpenRouter for inference and for web search.

architecture.md 3.4.1. One surface for every model call in the subnet, so there is one metering
format, one allowlist, and one credential type. The *selection* behind it stays open — Anthropic,
OpenAI, Google, open-weight, or a miner's own fine-tune routed through.

Built on the OpenAI SDK against OpenRouter's base URL, which is the shape production subnets use
and which means the request encoding is maintained by someone else.

## Snapshots are what the gateway enforces, not slugs

`model_slug` is a route and a route can move — a provider repointing `anthropic/claude-sonnet-4.5`
mid-season changes what every laboratory is running, without anyone editing a manifest. So 5.3
pins `model_snapshot` and this adapter sends the *snapshot-qualified* route, refusing a call whose
snapshot is not the one the season fixed. A laboratory therefore cannot be silently upgraded, and
two laboratories compared on the same day were compared against the same weights.

## Usage is read from the provider, never from the model

Token counts come out of the response's `usage` block. That matters because RCC is charged from
them: a count the caller supplied would let a laboratory under-report its own spend, and a count
estimated locally would disagree with the invoice that section 27 reconciles against.

## Every error path settles or releases

A provider *response* — even a 429 or a 500 — may have been billed, so it settles. Only a request
that never reached the provider releases. Getting this backwards in either direction is a real
failure: release-on-billed-error makes retries free, and settle-on-never-sent charges for
nothing.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from gateway.credentials import MinerCredential, ValidatorCredential
from gateway.metering import BudgetExceeded, Ledger, PriceTable, estimate_rcc
from protocol.receipts import CredentialOwner, Purpose, Receipt, Tool

__all__ = [
    "AdapterError",
    "CallOutcome",
    "ModelPin",
    "OpenRouterAdapter",
    "UndeclaredModel",
]

_log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

#: Sent on every request. OpenRouter attributes traffic by these, which means a subnet operator
#: can see subnet spend separately from anything else on the same account.
_ATTRIBUTION_HEADERS = {
    "HTTP-Referer": "https://github.com/AisA-wdym/auto-invent",
    "X-Title": "auto-invent",
}


#: What an unmeasurable call is charged, in tokens. Deliberately large: a call whose cost we cannot
#: read cannot be reconciled against the provider's invoice (27), and the direction of the error has
#: to make it not worth provoking. Sized at roughly a full research prompt rather than at a context
#: window, so an isolated provider glitch does not consume a whole round's ceiling in one call.
_UNMEASURED_IN = 100_000
_UNMEASURED_OUT = 8_192


class AdapterError(RuntimeError):
    """A call that could not be made, or a response that could not be trusted."""


class UndeclaredModel(AdapterError):
    """Gate 13.3/13.4: a model the miner did not declare, or a moved snapshot."""


@dataclass(frozen=True, slots=True)
class ModelPin:
    """A declared model: the route plus the season-fixed snapshot of it."""

    slug: str
    snapshot: str

    def qualified(self) -> str:
        """What goes on the wire.

        OpenRouter has no `slug@version` syntax: a pinned version *is* a distinct slug, dated by the
        provider — `openai/gpt-4o` is the moving route and `openai/gpt-4o-2024-08-06` is the frozen
        one. So `model_snapshot` holds the immutable slug and `model_slug` holds the readable route,
        and this sends the snapshot whenever there is one.

        Two fields rather than one because they answer different questions. `model_slug` is what the
        season *meant* ("the Claude judge"), and it stays stable in a config a human edits across
        seasons. `model_snapshot` is what was *sent*, and it is what a receipt records and 27
        reconciles against — so a provider retiring a dated slug changes one field and leaves the
        other as the record of intent.

        The bare slug is sent when the snapshot is still the config's placeholder, so an operator
        part-way through provisioning gets a working system rather than a mysteriously empty one.
        `assert_pinned` is where a production deployment refuses that state, and both `--check`
        paths call it.
        """
        if not self.snapshot or self.snapshot.startswith("<"):
            return self.slug
        return self.snapshot

    def assert_pinned(self) -> None:
        """Refuse an unpinned model. Called by the validator at season start.

        Separate from `qualified` so the check happens once, loudly, at startup — rather than on
        every call, where it would be a per-request failure nobody could act on mid-round.
        """
        if not self.snapshot or self.snapshot.startswith("<"):
            raise UndeclaredModel(
                f"model {self.slug} has no season snapshot ({self.snapshot!r}). An unpinned route "
                "can be repointed by the provider mid-season, which changes what every "
                "laboratory is running without any manifest changing."
            )


@dataclass(frozen=True, slots=True)
class CallOutcome:
    """What a completed call produced, plus what it cost as measured."""

    text: str
    tokens_in: int
    tokens_out: int
    rcc: int
    model: str
    attempts: int
    #: Present only for tool-calling requests; empty otherwise.
    tool_calls: tuple[dict[str, Any], ...] = ()
    #: Search results, for a search call.
    results: tuple[dict[str, Any], ...] = ()


@dataclass
class OpenRouterAdapter:
    """Issues requests, meters them, and records them on a receipt.

    The credential is a constructor argument of the *typed* kind from `gateway.credentials`, so
    an adapter instance is bound to one payer for its lifetime. A single adapter that could
    switch payers per call would put the 3.4.4 invariant back in a parameter.
    """

    credential: MinerCredential | ValidatorCredential
    prices: PriceTable
    ledger: Ledger
    base_url: str = DEFAULT_BASE_URL
    #: Injected for tests and for the offline `--check` path. Signature mirrors
    #: `AsyncOpenAI.chat.completions.create` closely enough to substitute.
    transport: Any | None = field(default=None, repr=False)
    max_attempts: int = 3

    def __post_init__(self) -> None:
        if self.transport is None:
            self.transport = _build_client(self.credential.api_key, self.base_url)

    @property
    def owner(self) -> CredentialOwner:
        return self.credential.owner

    async def complete(
        self,
        *,
        run_id: str,
        receipt: Receipt,
        purpose: Purpose,
        pin: ModelPin,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float = 0.0,
        allowed_models: frozenset[str] | None = None,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> CallOutcome:
        """One completion: allowlist, reserve, send, settle, receipt.

        `allowed_models` is the per-run allowlist from the verified session token. Checked here
        as well as at the token layer because this is the last point before bytes leave the
        process, and gate 13.3 ("undeclared model use") is checked against what was *sent*.
        """
        self.credential.assert_may_fund(purpose)
        if allowed_models is not None and pin.slug not in allowed_models:
            raise UndeclaredModel(
                f"run {run_id} requested {pin.slug}, which is not in its declared model "
                f"manifest ({sorted(allowed_models)}). Gate 13.3 invalidates the response, so "
                "the call is refused rather than made and then discounted."
            )
        if not self.prices.permits(pin.slug):
            raise UndeclaredModel(
                f"{pin.slug} is not permitted by this season's allowed_model_slugs"
            )

        prompt_tokens = _estimate_prompt_tokens(messages)
        estimate = estimate_rcc(
            self.prices, kind="llm", prompt_tokens=prompt_tokens, max_tokens=max_tokens
        )
        reservation = self.ledger.reserve(run_id, estimate, kind="llm")

        request: dict[str, Any] = {
            "model": pin.qualified(),
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            request["tools"] = tools
        if response_format:
            request["response_format"] = response_format

        request_bytes = json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
        attempts = 0
        last_error: Exception | None = None

        while attempts < self.max_attempts:
            attempts += 1
            try:
                raw = await self.transport.create(**_split_for_transport(request))
            except _NeverSent as error:
                # Nothing reached the provider, so nothing was billed. Release and stop: a
                # request the adapter itself refused will be refused identically on retry.
                self.ledger.release(reservation)
                raise AdapterError(f"run {run_id}: request not sent: {error}") from error
            except Exception as error:  # noqa: BLE001 - provider errors are data, not bugs
                last_error = error
                status = getattr(error, "status_code", None)
                if status is not None and 400 <= status < 500 and status != 429:
                    # A 4xx that is not rate limiting will fail identically on retry, and each
                    # attempt is charged. Stop rather than burn the run's attempts.
                    self.ledger.settle(reservation, 0)
                    self._record(
                        receipt,
                        tool=Tool.LLM,
                        purpose=purpose,
                        pin=pin,
                        request_bytes=request_bytes,
                        response_bytes=str(error).encode(),
                        rcc=0,
                        tokens_in=0,
                        tokens_out=0,
                        attempts=attempts,
                    )
                    raise AdapterError(f"run {run_id}: provider refused: {error}") from error
                _log.warning(
                    "run %s attempt %d/%d failed (%s); retrying",
                    run_id,
                    attempts,
                    self.max_attempts,
                    error,
                )
                continue

            usage = _usage(raw)
            rcc = self.prices.rcc_for_tokens(usage["tokens_in"], usage["tokens_out"])
            self.ledger.settle(reservation, rcc)
            response_bytes = json.dumps(_response_body(raw), sort_keys=True).encode()
            self._record(
                receipt,
                tool=Tool.LLM,
                purpose=purpose,
                pin=pin,
                request_bytes=request_bytes,
                response_bytes=response_bytes,
                rcc=rcc,
                tokens_in=usage["tokens_in"],
                tokens_out=usage["tokens_out"],
                attempts=attempts,
            )
            return CallOutcome(
                text=_content(raw),
                tokens_in=usage["tokens_in"],
                tokens_out=usage["tokens_out"],
                rcc=rcc,
                model=pin.qualified(),
                attempts=attempts,
                tool_calls=tuple(_tool_calls(raw)),
            )

        # Every attempt produced a provider response, and each of those was billed. The
        # reservation covered one call's worth, so the exhausted retries are charged at the
        # estimate rather than released — a retry storm that cost money must cost budget.
        self.ledger.settle(reservation, estimate * attempts)
        self._record(
            receipt,
            tool=Tool.LLM,
            purpose=purpose,
            pin=pin,
            request_bytes=request_bytes,
            response_bytes=str(last_error).encode(),
            rcc=estimate * attempts,
            tokens_in=0,
            tokens_out=0,
            attempts=attempts,
        )
        raise AdapterError(
            f"run {run_id}: {attempts} attempts against {pin.slug} all failed; last error: "
            f"{last_error}. Charged {estimate * attempts} RCC because each attempt was billed."
        )

    async def search(
        self,
        *,
        run_id: str,
        receipt: Receipt,
        purpose: Purpose,
        query: str,
        pin: ModelPin,
        max_results: int = 10,
    ) -> CallOutcome:
        """Web search, through the same credential and the same meter.

        5.3: "search spend and inference spend are bounded by one ceiling rather than two." So
        this shares the run's RCC ledger and counts against `maximum_search_calls` as well.

        OpenRouter exposes search as a plugin on a completion rather than a separate endpoint,
        which is why this is a completion under the hood and why the receipt still records
        `Tool.SEARCH` — the receipt records what the call was *for*, and gate 13.8 checks
        citations against searches actually made.
        """
        self.credential.assert_may_fund(purpose)
        reservation = self.ledger.reserve(run_id, self.prices.rcc_per_search, kind="search")

        request = {
            "model": pin.qualified(),
            "messages": [{"role": "user", "content": query}],
            "plugins": [{"id": "web", "max_results": max_results}],
            "max_tokens": 2_048,
        }
        request_bytes = json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
        try:
            raw = await self.transport.create(**_split_for_transport(request))
        except Exception as error:  # noqa: BLE001
            self.ledger.settle(reservation, self.prices.rcc_per_search)
            self._record(
                receipt,
                tool=Tool.SEARCH,
                purpose=purpose,
                pin=pin,
                request_bytes=request_bytes,
                response_bytes=str(error).encode(),
                rcc=self.prices.rcc_per_search,
                tokens_in=0,
                tokens_out=0,
                attempts=1,
            )
            raise AdapterError(f"run {run_id}: search failed: {error}") from error

        usage = _usage(raw)
        # Charged at the flat search rate, not per token: a search's token cost is dominated by
        # whatever the provider chose to inject, which the laboratory did not control and should
        # therefore not be billed by the token for.
        rcc = self.prices.rcc_per_search
        self.ledger.settle(reservation, rcc)
        response_bytes = json.dumps(_response_body(raw), sort_keys=True).encode()
        self._record(
            receipt,
            tool=Tool.SEARCH,
            purpose=purpose,
            pin=pin,
            request_bytes=request_bytes,
            response_bytes=response_bytes,
            rcc=rcc,
            tokens_in=usage["tokens_in"],
            tokens_out=usage["tokens_out"],
            attempts=1,
        )
        return CallOutcome(
            text=_content(raw),
            tokens_in=usage["tokens_in"],
            tokens_out=usage["tokens_out"],
            rcc=rcc,
            model=pin.qualified(),
            attempts=1,
            results=tuple(_annotations(raw)),
        )

    def _record(
        self,
        receipt: Receipt,
        *,
        tool: Tool,
        purpose: Purpose,
        pin: ModelPin,
        request_bytes: bytes,
        response_bytes: bytes,
        rcc: int,
        tokens_in: int,
        tokens_out: int,
        attempts: int,
    ) -> None:
        receipt.record(
            tool=tool,
            provider="openrouter",
            request=request_bytes,
            response=response_bytes,
            rcc=rcc,
            purpose=purpose,
            credential_owner=self.owner,
            model=pin.slug,
            revision=pin.snapshot,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            attempts=attempts,
        )


class _NeverSent(AdapterError):
    """Raised by a transport when the request did not leave the process."""


#: Request keys that are OpenRouter extensions rather than OpenAI-standard parameters. The
#: OpenAI SDK rejects an unknown top-level keyword, so these travel in `extra_body` — which is
#: the SDK's documented escape hatch and puts them in the JSON body unchanged.
#:
#: Split at send time rather than at build time so the *receipt* hashes one flat request object
#: containing everything that went on the wire. Hashing only the standard half would mean a
#: laboratory could change `plugins` — turning search on, changing result counts — without the
#: receipt's request digest moving, and gate 13.5 checks endpoints against that digest.
_OPENROUTER_EXTENSIONS = frozenset({"plugins", "provider", "transforms", "route", "models"})


def _split_for_transport(request: dict[str, Any]) -> dict[str, Any]:
    """Move OpenRouter-only keys into `extra_body`, leaving standard keys top-level."""
    standard = {key: value for key, value in request.items() if key not in _OPENROUTER_EXTENSIONS}
    extensions = {key: value for key, value in request.items() if key in _OPENROUTER_EXTENSIONS}
    if extensions:
        standard["extra_body"] = extensions
    return standard


def _build_client(api_key: str, base_url: str) -> Any:
    """The real transport. Imported lazily so the pure layer never pulls in `openai`."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key, base_url=base_url, default_headers=_ATTRIBUTION_HEADERS)
    return client.chat.completions


def _estimate_prompt_tokens(messages: list[dict[str, Any]]) -> int:
    """A cheap over-estimate of the prompt size, for the reservation only.

    Four characters per token is the usual English rule of thumb and it under-estimates for
    code and for non-Latin scripts — so the result is scaled up by a quarter. The estimate only
    ever affects how early a near-exhausted run is refused; the *charge* is always the
    provider's reported count, so an inaccurate estimate here cannot mis-bill anyone.
    """
    characters = sum(len(str(message.get("content", ""))) for message in messages)
    return (characters // 4) * 5 // 4 + 16


def _usage(raw: Any) -> dict[str, int]:
    """Token counts from the provider's own accounting.

    A response with no usage block is charged as if it used the full context, because a call
    whose cost we cannot measure is a call we cannot reconcile — and treating it as free is what
    would make it worth provoking.
    """
    usage = getattr(raw, "usage", None) or (raw.get("usage") if isinstance(raw, dict) else None)
    if usage is None:
        _log.error("provider response carried no usage block; charging %d tokens", _UNMEASURED_IN)
        return {"tokens_in": _UNMEASURED_IN, "tokens_out": 0}

    if isinstance(usage, dict):
        prompt, completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
    else:
        prompt = getattr(usage, "prompt_tokens", None)
        completion = getattr(usage, "completion_tokens", None)

    # A *partial* usage block is charged as conservatively as an absent one. The first version read
    # each side with a zero default, so a provider or proxy that reported only completion tokens
    # made every input token free — and input dominates a research prompt. Absent and partial are
    # the same fact: we cannot measure it, and treating what we cannot measure as free is what makes
    # it worth provoking.
    if not isinstance(prompt, int) or isinstance(prompt, bool) or prompt < 0:
        _log.error("provider reported prompt_tokens=%r; charging %d", prompt, _UNMEASURED_IN)
        prompt = _UNMEASURED_IN
    if not isinstance(completion, int) or isinstance(completion, bool) or completion < 0:
        _log.error(
            "provider reported completion_tokens=%r; charging %d", completion, _UNMEASURED_OUT
        )
        completion = _UNMEASURED_OUT
    return {"tokens_in": prompt, "tokens_out": completion}


def _first_choice(raw: Any) -> Any:
    choices = getattr(raw, "choices", None)
    if choices is None and isinstance(raw, dict):
        choices = raw.get("choices")
    if not choices:
        raise AdapterError("provider response carried no choices")
    return choices[0]


def _message(raw: Any) -> Any:
    choice = _first_choice(raw)
    message = getattr(choice, "message", None)
    if message is None and isinstance(choice, dict):
        message = choice.get("message")
    if message is None:
        raise AdapterError("provider response choice carried no message")
    return message


def _content(raw: Any) -> str:
    message = _message(raw)
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")
    return content or ""


def _tool_calls(raw: Any) -> list[dict[str, Any]]:
    message = _message(raw)
    calls = getattr(message, "tool_calls", None)
    if calls is None and isinstance(message, dict):
        calls = message.get("tool_calls")
    if not calls:
        return []
    normalised: list[dict[str, Any]] = []
    for call in calls:
        function = getattr(call, "function", None) or (
            call.get("function") if isinstance(call, dict) else None
        )
        normalised.append(
            {
                "name": getattr(function, "name", None)
                or (function.get("name") if isinstance(function, dict) else ""),
                "arguments": getattr(function, "arguments", None)
                or (function.get("arguments") if isinstance(function, dict) else ""),
            }
        )
    return normalised


def _annotations(raw: Any) -> list[dict[str, Any]]:
    """Search citations, which OpenRouter attaches to the message as annotations."""
    message = _message(raw)
    annotations = getattr(message, "annotations", None)
    if annotations is None and isinstance(message, dict):
        annotations = message.get("annotations")
    results: list[dict[str, Any]] = []
    for annotation in annotations or []:
        citation = getattr(annotation, "url_citation", None) or (
            annotation.get("url_citation") if isinstance(annotation, dict) else None
        )
        if citation is None:
            continue
        results.append(
            {
                "url": getattr(citation, "url", None)
                or (citation.get("url") if isinstance(citation, dict) else ""),
                "title": getattr(citation, "title", None)
                or (citation.get("title") if isinstance(citation, dict) else ""),
                "content": getattr(citation, "content", None)
                or (citation.get("content") if isinstance(citation, dict) else ""),
            }
        )
    return results


def _response_body(raw: Any) -> dict[str, Any]:
    """The subset of the response the receipt hashes.

    Hashed rather than stored, and reduced first: hashing the raw SDK object would make the
    digest depend on SDK version, so two gateways on different releases would disagree about
    what the same response was. The receipt has to be verifiable across versions.
    """
    usage = _usage(raw)
    body: dict[str, Any] = {
        "content": _content(raw),
        "tokens_in": usage["tokens_in"],
        "tokens_out": usage["tokens_out"],
    }
    calls = _tool_calls(raw)
    if calls:
        body["tool_calls"] = calls
    return body


def budget_error_status(error: Exception) -> int:
    """HTTP status for an error, for the API layer.

    429 for a budget refusal rather than 403, because a laboratory should distinguish "you have
    spent your ceiling" from "you asked for something forbidden" — the first means stop and
    submit what you have, the second means the request was wrong.
    """
    if isinstance(error, BudgetExceeded):
        return 429
    if isinstance(error, UndeclaredModel):
        return 403
    return 502
