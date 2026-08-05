"""One validator-side model client for generation, critique, judging and prior art.

architecture.md 3.4.3: all four are validator costs on the validator's own OpenRouter key. Same
provider as the miners, a different account.

Shared rather than one client per subsystem, because all four need the same four things and
getting any of them wrong is silent:

1. **The validator's credential, selected by purpose.** Not passed in — `CredentialSet.for_purpose`
   picks it, and a miner-funded purpose cannot reach here at all.
2. **A receipt.** Validator spend is receipted like miner spend, because 3.4.4 point 3 makes
   per-account reconciliation the check that catches a call billed to the wrong side, and that
   comparison needs both sides recorded in the same format.
3. **Strict JSON.** Every one of the four wants structured output, and a model that returns prose
   wrapped in a code fence is the ordinary case rather than the exception.
4. **A refusal, not a guess, on unparseable output.** A judge whose JSON did not parse must be
   recorded as *absent* so its criterion weight redistributes (16.3, and
   `protocol.fixedpoint.apply_weights`). Substituting a default score would put a fabricated
   number into a miner's payment.

## Why the JSON handling is more than `json.loads`

Models wrap JSON in ```json fences, prepend "Here is the analysis:", and occasionally emit two
objects. `parse_json` strips fences and takes the first complete object by brace matching. That
is more tolerant than a protocol should normally be, and it is the right trade here: the
alternative is discarding a judge's genuine verdict over a formatting habit, which shifts weight
onto the remaining judges and makes the panel smaller than 16.1 requires for no real reason.

What it does *not* do is repair malformed JSON. A truncated object is a truncated verdict, and
guessing the missing half would be inventing a score.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from gateway.adapters.openrouter import AdapterError, ModelPin, OpenRouterAdapter
from gateway.credentials import CredentialSet
from gateway.metering import Ledger, PriceTable
from protocol.receipts import Purpose, Receipt

__all__ = [
    "ModelClient",
    "ModelReply",
    "UnparseableReply",
    "parse_json",
]

_log = logging.getLogger(__name__)

#: Ceiling on the validator's own spend per round, in RCC. The validator is not competing, so
#: this is an operational guard rather than a fairness one: a generation loop that retried
#: without limit would spend an operator's balance overnight.
DEFAULT_VALIDATOR_CEILING = 2_000_000


class UnparseableReply(AdapterError):
    """The model replied, but not with something we can read as the requested structure."""


@dataclass(frozen=True, slots=True)
class ModelReply:
    """One validator-funded call's result."""

    text: str
    parsed: Any
    rcc: int
    tokens_in: int
    tokens_out: int
    family: str
    model: str


@dataclass
class ModelClient:
    """Validator-funded model access. One instance per round.

    Holds one adapter per *family* rather than one overall, because each family is a different
    model pin and the receipt records which model answered — a judge panel whose members were
    indistinguishable on the receipt could not be audited against 16.1's family requirements.
    """

    credentials: CredentialSet
    prices: PriceTable
    ledger: Ledger
    receipt: Receipt
    run_id: str
    #: family -> the pinned model that family means this season.
    pins: Mapping[str, ModelPin]
    #: Injected in tests; `None` builds a real OpenRouter client.
    transport_factory: Any | None = field(default=None, repr=False)
    _adapters: dict[str, OpenRouterAdapter] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not self.pins:
            raise AdapterError(
                "no model pins: challenge generation, critique and judging all name a family, "
                "and a family with no pin cannot be called"
            )

    def families(self) -> tuple[str, ...]:
        return tuple(sorted(self.pins))

    def assert_all_pinned(self) -> None:
        """Refuse an unpinned family at startup.

        Called once by the validator's composition root. A route the provider can repoint
        mid-season changes what every laboratory is judged by, and 27 measures rerun correlation
        — so this is checked loudly at start rather than per call, where nobody could act on it.
        """
        for family, pin in sorted(self.pins.items()):
            try:
                pin.assert_pinned()
            except AdapterError as error:
                raise AdapterError(f"family {family!r}: {error}") from error

    def _adapter(self, family: str) -> OpenRouterAdapter:
        if family not in self.pins:
            raise AdapterError(
                f"family {family!r} has no pinned model; declared families are "
                f"{self.families()}"
            )
        adapter = self._adapters.get(family)
        if adapter is None:
            # `for_purpose` with no miner named: a validator purpose that named a miner would be
            # refused by `CredentialSet`, which is the 3.4.4 invariant doing its job here.
            credential = self.credentials.for_purpose(Purpose.JUDGING)
            transport = self.transport_factory(family) if self.transport_factory else None
            adapter = OpenRouterAdapter(
                credential=credential,
                prices=self.prices,
                ledger=self.ledger,
                transport=transport,
            )
            self._adapters[family] = adapter
        return adapter

    async def ask(
        self,
        *,
        family: str,
        purpose: Purpose,
        system: str,
        user: str,
        max_tokens: int = 8_192,
        temperature: float = 0.0,
        expect_json: bool = True,
    ) -> ModelReply:
        """One validator-funded call.

        `temperature=0.0` by default. Not because it makes the call deterministic — it does not,
        and 27 accepts cross-validator divergence — but because a temperature the caller did not
        choose should be the one that minimises variance. A generator that wants variety asks for
        it explicitly, which is visible in the receipt.
        """
        pin = self.pins.get(family)
        adapter = self._adapter(family)
        if pin is None:  # pragma: no cover - `_adapter` raises first
            raise AdapterError(f"family {family!r} has no pin")

        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        outcome = await adapter.complete(
            run_id=self.run_id,
            receipt=self.receipt,
            purpose=purpose,
            pin=pin,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            # No allowlist: the validator's own calls are constrained by the season's pinned
            # families, which `_adapter` already enforced. An allowlist here would duplicate
            # `self.pins` and could disagree with it.
            allowed_models=None,
            response_format={"type": "json_object"} if expect_json else None,
        )
        parsed = parse_json(outcome.text) if expect_json else None
        return ModelReply(
            text=outcome.text,
            parsed=parsed,
            rcc=outcome.rcc,
            tokens_in=outcome.tokens_in,
            tokens_out=outcome.tokens_out,
            family=family,
            model=pin.slug,
        )

    async def ask_many(
        self, requests: Sequence[Mapping[str, Any]]
    ) -> list[ModelReply | Exception]:
        """Several calls concurrently, returning failures as values rather than raising.

        Failures are returned rather than raised because of what the callers do with them. A
        judge panel of three that lost one member must score with two and redistribute the
        weight; a `gather` that propagated the first exception would lose the two
        verdicts that *did* arrive, and turn one model's rate limit into a criterion nobody
        scored.
        """
        import asyncio

        async def one(request: Mapping[str, Any]) -> ModelReply | Exception:
            try:
                return await self.ask(**request)
            except Exception as error:  # noqa: BLE001 - a failed call is data here
                _log.warning("validator call to %s failed: %s", request.get("family"), error)
                return error

        return list(await asyncio.gather(*(one(request) for request in requests)))


def parse_json(text: str) -> Any:
    """Read the first complete JSON object or array out of a model's reply.

    Tolerant of the three things models reliably do — fence it, preface it, follow it with
    commentary — and intolerant of anything that would require guessing at content.
    """
    stripped = text.strip()
    if not stripped:
        raise UnparseableReply("the model returned nothing at all")

    # Fences first: ```json {...} ``` is the single most common wrapper.
    if stripped.startswith("```"):
        stripped = stripped.removeprefix("```json").removeprefix("```").strip()
        if stripped.endswith("```"):
            stripped = stripped[:-3].strip()

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    extracted = _first_structure(stripped)
    if extracted is None:
        raise UnparseableReply(
            f"no JSON object or array in a {len(text)}-character reply. Recorded as absent "
            "rather than substituted: a default score is a fabricated number in someone's "
            f"payment. Reply began: {text[:120]!r}"
        )
    try:
        return json.loads(extracted)
    except json.JSONDecodeError as error:
        raise UnparseableReply(
            f"found a JSON-shaped span that does not parse ({error}). Not repaired: a truncated "
            "object is a truncated verdict, and completing it would be inventing content."
        ) from error


def _first_structure(text: str) -> str | None:
    """The first balanced `{...}` or `[...]`, ignoring braces inside strings.

    String-aware because a problem statement containing `{` — which a challenge about templating
    or formatting very well might — would otherwise unbalance the scan and truncate the object
    mid-field.
    """
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start < 0:
            continue
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            character = text[index]
            if escaped:
                escaped = False
                continue
            if character == "\\":
                escaped = True
                continue
            if character == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if character == opener:
                depth += 1
            elif character == closer:
                depth -= 1
                if depth == 0:
                    return text[start : index + 1]
    return None
