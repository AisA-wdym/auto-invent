"""RCC pricing and the budget ledger. Reserve before the call, settle after.

architecture.md 13.6 makes "budget exceeded" a hard gate, and 8's `resource_limits` gives every
laboratory the same ceiling. The ceiling is the whole basis of comparison: two laboratories that
spent different amounts were not asked the same question.

## Reserve-then-settle, and why check-then-add is not enough

The obvious shape — and the one the production gateway this design draws from uses — is:

    cost = ledger.spent(run)
    if cost >= LIMIT: refuse
    response = await provider.call(...)
    ledger.add(run, response.cost)

That is a time-of-check/time-of-use race. `maximum_parallel_calls` is 16 in the model manifest
(5.3), so sixteen requests can pass the check while spend is still under the limit and *then*
settle, all of them. The ceiling overshoots by up to fifteen calls' worth — and it overshoots
by more for the laboratory that parallelises hardest, which is to say the overshoot rewards
exactly the behaviour the ceiling exists to bound.

So spend is reserved *before* the request goes out, at a conservative estimate, and the
reservation is replaced by the true cost when the response comes back:

    reservation = ledger.reserve(run, estimate)   # refuses here, atomically
    try:
        response = await provider.call(...)
    finally:
        ledger.settle(reservation, actual_rcc)     # or release() if nothing was spent

`spent()` counts settled plus outstanding, so a concurrent caller sees the reservation. The
estimate has to be an over-estimate for the bound to hold — see `estimate_rcc`.

## Why RCC rather than dollars

A price in dollars moves under us: OpenRouter reprices a route mid-season and every laboratory's
effective ceiling changes without anyone editing a config. RCC is an integer unit fixed by the
season's price table, so the ceiling means the same thing on day 1 and day 90. The table is part
of the season config, so changing it is a visible, hashed change rather than a provider's
Tuesday.

Integers throughout, floor-rounded *up* on cost: a fractional RCC rounded down would let a
laboratory make many sub-unit calls for free.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

__all__ = [
    "BudgetExceeded",
    "Ledger",
    "PriceTable",
    "Reservation",
    "estimate_rcc",
]

_log = logging.getLogger(__name__)


class BudgetExceeded(RuntimeError):
    """A call that would take a run past a ceiling it was issued with."""


@dataclass(frozen=True, slots=True)
class PriceTable:
    """Season-fixed RCC prices, from `providers.miner_pricing`. Integers, flat per 1k tokens.

    ## Flat pricing is a mechanism choice, not a simplification

    Every model costs the same RCC per thousand tokens, whatever it costs in dollars. So RCC
    measures *reasoning volume*, and the equal ceiling of 8's `resource_limits` equalises how
    much thinking each laboratory may do — not how much money it may spend.

    The consequence is worth stating because it is deliberate and it cuts both ways. A
    laboratory that picks an expensive frontier model gets no less RCC headroom than one picking
    a cheap open-weight model; it simply burns more of its *own* dollars per RCC. Under
    dollar-denominated pricing the cheap-model laboratory would get several times the token
    budget for the same ceiling, and the tournament would be measuring model economics rather
    than research architecture. Flat pricing puts the comparison where 1 says it belongs: on the
    laboratory design.

    What it costs us is that the ceiling no longer bounds a miner's dollar spend. That is
    handled where it belongs — `declared_spend_cap_usd` on the miner's own key (5.4.1), which
    the miner controls and the protocol reconciles rather than trusts.
    """

    rcc_per_1k_in: int
    rcc_per_1k_out: int
    rcc_per_search: int
    #: Routes a miner may address at all. Empty means "whatever the model manifest declared" —
    #: the per-run allowlist in the session token is the binding one, and gate 13.3 catches an
    #: undeclared model regardless. A non-empty list is a season-wide restriction on top.
    allowed_model_slugs: frozenset[str] = frozenset()

    def rcc_for_tokens(self, tokens_in: int, tokens_out: int) -> int:
        """Cost of one completion, rounded up on each side.

        Ceiling division, not floor: a 400-token call at 300 RCC per 1k is 0.12 RCC, and
        flooring that to zero would make short calls free — so a laboratory could issue
        unboundedly many of them. Rounding up costs at most one RCC per call per side, which is
        under a percent of any ceiling worth setting.
        """
        if tokens_in < 0 or tokens_out < 0:
            raise BudgetExceeded(
                f"negative token count ({tokens_in} in, {tokens_out} out): a provider response "
                "that reports negative usage would credit RCC back to the run"
            )
        return -(-tokens_in * self.rcc_per_1k_in // 1_000) + -(
            -tokens_out * self.rcc_per_1k_out // 1_000
        )

    def permits(self, model_slug: str) -> bool:
        return not self.allowed_model_slugs or model_slug in self.allowed_model_slugs

    @classmethod
    def from_season(cls, season: dict) -> PriceTable:
        pricing = season["providers"]["miner_pricing"]
        return cls(
            rcc_per_1k_in=pricing["rcc_per_1k_in"],
            rcc_per_1k_out=pricing["rcc_per_1k_out"],
            rcc_per_search=pricing["rcc_per_search"],
            allowed_model_slugs=frozenset(pricing["allowed_model_slugs"]),
        )


@dataclass(frozen=True, slots=True)
class Reservation:
    """An outstanding hold on a run's budget. Must be settled or released."""

    run_id: str
    reservation_id: int
    rcc: int
    kind: str


@dataclass
class Ledger:
    """Per-run spend, with holds. The authority on consumption.

    The token (`gateway.tokens`) is the authority on the *limit*; this is the authority on the
    *spend*. Split that way because a token verifies from bytes and so survives a restart, while
    spend cannot be recovered from a token and must therefore fail closed: `admit` must be
    called for a run before it can spend, and a restarted gateway that never admitted a run
    refuses it rather than assuming zero.

    Guarded by a lock because the reservation check and the reservation write must be one
    step. Under `asyncio` a single-threaded event loop would make that true anyway, but the
    gateway runs under uvicorn with a threadpool for sync work and the runner drives it from
    another thread — so the lock is what makes the guarantee independent of who calls.
    """

    #: run_id -> settled RCC
    _settled: dict[str, int] = field(default_factory=dict)
    #: run_id -> {reservation_id: held RCC}
    _held: dict[str, dict[int, int]] = field(default_factory=dict)
    #: run_id -> ceilings, from the verified token
    _limits: dict[str, dict[str, int]] = field(default_factory=dict)
    #: run_id -> counts, for the request and search ceilings
    _counts: dict[str, dict[str, int]] = field(default_factory=dict)
    _next_id: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def admit(
        self, run_id: str, *, maximum_rcc: int, maximum_requests: int, maximum_search_calls: int
    ) -> None:
        """Open a run's ledger with the ceilings from its verified token."""
        with self._lock:
            if run_id in self._limits:
                # Re-admitting would reset the spend, which is a budget reset. A run that is
                # already open stays open with the spend it has.
                _log.warning("run %s is already admitted; ceilings unchanged", run_id)
                return
            self._limits[run_id] = {
                "rcc": maximum_rcc,
                "requests": maximum_requests,
                "search_calls": maximum_search_calls,
            }
            self._settled[run_id] = 0
            self._held[run_id] = {}
            self._counts[run_id] = {"requests": 0, "search_calls": 0}

    def close(self, run_id: str) -> dict[str, int]:
        """Finish a run and return its totals. Outstanding holds are an error.

        A hold still open at close is a call whose outcome nobody recorded — which means the
        provider may have billed for it. Logged loudly and counted as spent, because the safe
        direction is to charge for a call that may have happened rather than to forget it.
        """
        with self._lock:
            outstanding = self._held.pop(run_id, {})
            if outstanding:
                leaked = sum(outstanding.values())
                _log.error(
                    "run %s closed with %d outstanding reservation(s) totalling %d RCC. A hold "
                    "open at close is a call whose outcome was never recorded, so it is counted "
                    "as spent rather than forgotten.",
                    run_id,
                    len(outstanding),
                    leaked,
                )
                self._settled[run_id] = self._settled.get(run_id, 0) + leaked
            totals = {
                "rcc": self._settled.pop(run_id, 0),
                **self._counts.pop(run_id, {"requests": 0, "search_calls": 0}),
            }
            self._limits.pop(run_id, None)
            return totals

    def spent(self, run_id: str) -> int:
        """Settled plus outstanding. What a concurrent caller must see."""
        with self._lock:
            return self._settled.get(run_id, 0) + sum(self._held.get(run_id, {}).values())

    def remaining(self, run_id: str) -> int:
        with self._lock:
            limit = self._limits.get(run_id)
            if limit is None:
                return 0
            spent = self._settled.get(run_id, 0) + sum(self._held.get(run_id, {}).values())
            return max(0, limit["rcc"] - spent)

    def reserve(self, run_id: str, estimate: int, *, kind: str) -> Reservation:
        """Hold `estimate` RCC, or refuse. Atomic with the ceiling check.

        `kind` is `"llm"`, `"search"` or `"embedding"`; the request and search-call ceilings
        are counted here too, because a laboratory bounded only on RCC could make a very large
        number of very cheap calls and turn the gateway into its own denial of service.
        """
        with self._lock:
            limit = self._limits.get(run_id)
            if limit is None:
                # Fail closed. A gateway that lost its ledger cannot know what a run has spent,
                # and treating unknown as zero would hand a restarted run a fresh budget.
                raise BudgetExceeded(
                    f"run {run_id} is not admitted to the ledger. Its spend cannot be "
                    "accounted for, and treating an unknown run as having spent nothing would "
                    "reset its budget on every gateway restart."
                )

            counts = self._counts[run_id]
            if counts["requests"] >= limit["requests"]:
                raise BudgetExceeded(
                    f"run {run_id} has made {counts['requests']} requests, its ceiling. Gate "
                    "13.6 is exceeded, and a run bounded only on RCC could make unboundedly "
                    "many cheap calls."
                )
            if kind == "search" and counts["search_calls"] >= limit["search_calls"]:
                raise BudgetExceeded(
                    f"run {run_id} has made {counts['search_calls']} search calls, its ceiling"
                )

            spent = self._settled[run_id] + sum(self._held[run_id].values())
            if spent + estimate > limit["rcc"]:
                raise BudgetExceeded(
                    f"run {run_id} has {limit['rcc'] - spent} RCC remaining and this call is "
                    f"estimated at {estimate}. Refused before the request is sent: refusing "
                    "after would mean the provider already billed for it."
                )

            self._next_id += 1
            reservation = Reservation(
                run_id=run_id, reservation_id=self._next_id, rcc=estimate, kind=kind
            )
            self._held[run_id][reservation.reservation_id] = estimate
            counts["requests"] += 1
            if kind == "search":
                counts["search_calls"] += 1
            return reservation

    def settle(self, reservation: Reservation, actual_rcc: int) -> int:
        """Replace a hold with the measured cost. Returns the charged amount.

        A call that came back costing *more* than its estimate is charged in full and allowed
        to exceed the ceiling by that overshoot. The provider has already billed it; refusing
        to record it would make the receipt disagree with the invoice, which is the one thing
        reconciliation exists to catch. The next `reserve` sees the overshoot and refuses, so
        the breach is bounded to a single call.
        """
        with self._lock:
            held = self._held.get(reservation.run_id, {})
            estimate = held.pop(reservation.reservation_id, None)
            if estimate is None:
                raise BudgetExceeded(
                    f"reservation {reservation.reservation_id} for run {reservation.run_id} was "
                    "already settled or released; settling twice would double-charge"
                )
            run = reservation.run_id
            self._settled[run] = self._settled.get(run, 0) + actual_rcc
            limit = self._limits.get(reservation.run_id)
            if limit and self._settled[reservation.run_id] > limit["rcc"]:
                _log.warning(
                    "run %s settled at %d RCC against a ceiling of %d: the call cost more than "
                    "its estimate of %d. Recorded in full because the provider billed it; the "
                    "next reservation will be refused.",
                    reservation.run_id,
                    self._settled[reservation.run_id],
                    limit["rcc"],
                    estimate,
                )
            return actual_rcc

    def release(self, reservation: Reservation) -> None:
        """Drop a hold for a call that never reached the provider.

        Only for a request that failed before the provider saw it — a connection refused, a
        request the adapter rejected. A provider error *response* was billed and must be
        settled, not released.
        """
        with self._lock:
            self._held.get(reservation.run_id, {}).pop(reservation.reservation_id, None)


def estimate_rcc(
    prices: PriceTable, *, kind: str, prompt_tokens: int = 0, max_tokens: int = 0
) -> int:
    """A deliberate over-estimate of what a call will cost.

    The reservation must be an upper bound or the ceiling does not hold: an under-estimate lets
    a run reserve less than it spends, and the difference accumulates across every call.

    So output tokens are estimated at the caller's `max_tokens` — the most the provider can
    return — rather than at some historical average. That over-reserves for most calls, and the
    over-reservation is returned at `settle`, so the only cost is that a run near its ceiling is
    refused slightly early. Being refused one call early is a much smaller error than exceeding
    a ceiling that the comparison between laboratories depends on.
    """
    if kind == "search":
        return prices.rcc_per_search
    if kind == "embedding":
        # Embeddings are input-only, so the input rate is the whole cost. Priced off the same
        # rate rather than a separate config field: a field nobody sets would default to
        # something, and a default of zero is an unmetered call.
        return prices.rcc_for_tokens(prompt_tokens, 0)
    if kind != "llm":
        raise BudgetExceeded(f"unknown call kind {kind!r}: it has no price and cannot be metered")
    if max_tokens <= 0:
        raise BudgetExceeded(
            "an LLM call must declare max_tokens. Without a bound on the response there is no "
            "upper bound to reserve, and a reservation that is not an upper bound does not hold "
            "the ceiling."
        )
    return prices.rcc_for_tokens(prompt_tokens, max_tokens)
