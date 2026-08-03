"""RCC pricing and the budget ledger: architecture.md 13.6 and 8.

The ceiling is the basis of comparison — two laboratories that spent different amounts were not
asked the same question. So these tests are mostly about the ceiling holding under conditions
where the obvious implementation lets it slip.
"""

from __future__ import annotations

import json
import pathlib
import threading

import pytest

from gateway.metering import BudgetExceeded, Ledger, PriceTable, estimate_rcc

pytestmark = pytest.mark.determinism

SEASON = json.loads(pathlib.Path("config/season.example.json").read_text())
PRICES = PriceTable.from_season(SEASON)


def opened(maximum_rcc: int = 1_000, requests: int = 100, searches: int = 50) -> Ledger:
    ledger = Ledger()
    ledger.admit(
        "run", maximum_rcc=maximum_rcc, maximum_requests=requests, maximum_search_calls=searches
    )
    return ledger


# --------------------------------------------------------------------------
# Pricing: integers, rounded up
# --------------------------------------------------------------------------


def test_the_price_table_comes_from_the_season_config():
    pricing = SEASON["providers"]["miner_pricing"]
    assert PRICES.rcc_per_1k_in == pricing["rcc_per_1k_in"]
    assert PRICES.rcc_per_1k_out == pricing["rcc_per_1k_out"]


def test_a_short_call_is_not_free():
    """Floor rounding would price a 400-token call at zero, so a laboratory could make
    unboundedly many of them."""
    assert PRICES.rcc_for_tokens(1, 0) >= 1
    assert PRICES.rcc_for_tokens(0, 1) >= 1


def test_output_tokens_cost_more_than_input_tokens():
    """They cost the provider more, and a laboratory that reasoned at length should pay for it."""
    assert PRICES.rcc_for_tokens(0, 1_000) > PRICES.rcc_for_tokens(1_000, 0)


def test_cost_is_monotone_in_tokens():
    previous = -1
    for tokens in (0, 1, 100, 1_000, 10_000, 100_000):
        cost = PRICES.rcc_for_tokens(tokens, tokens)
        assert cost > previous
        previous = cost


def test_a_negative_token_count_is_refused_rather_than_crediting_budget():
    """A provider response reporting negative usage would otherwise refund RCC."""
    with pytest.raises(BudgetExceeded, match="negative token count"):
        PRICES.rcc_for_tokens(-1_000_000, 0)


def test_every_price_is_an_integer():
    """A float price would put a float on the path to a hashed receipt total."""
    assert isinstance(PRICES.rcc_per_1k_in, int)
    assert isinstance(PRICES.rcc_per_1k_out, int)
    assert isinstance(PRICES.rcc_for_tokens(1_234, 5_678), int)


def test_an_empty_allowlist_permits_any_declared_model():
    """The per-run token allowlist is the binding one; a season-wide list is an extra."""
    assert PRICES.permits("anything/at-all")


def test_a_non_empty_allowlist_excludes_what_it_omits():
    restricted = PriceTable(
        rcc_per_1k_in=300,
        rcc_per_1k_out=1_500,
        rcc_per_search=50,
        allowed_model_slugs=frozenset({"openai/gpt-5"}),
    )
    assert restricted.permits("openai/gpt-5")
    assert not restricted.permits("anthropic/claude-sonnet-4.5")


# --------------------------------------------------------------------------
# The ceiling holds, including under concurrency
# --------------------------------------------------------------------------


def test_a_reservation_that_would_exceed_the_ceiling_is_refused():
    ledger = opened(maximum_rcc=100)
    ledger.reserve("run", 90, kind="llm")
    with pytest.raises(BudgetExceeded, match="Refused before the request is sent"):
        ledger.reserve("run", 20, kind="llm")


def test_an_outstanding_reservation_is_visible_to_a_concurrent_caller():
    """The defect this module exists to avoid.

    Check-then-add lets every concurrent request see spend-under-limit and all proceed. With
    `maximum_parallel_calls` at 16 (5.3) the ceiling overshoots by up to fifteen calls — and it
    overshoots most for the laboratory that parallelises hardest, which is to say the overshoot
    rewards exactly what the ceiling exists to bound.
    """
    ledger = opened(maximum_rcc=100)
    ledger.reserve("run", 100, kind="llm")
    assert ledger.spent("run") == 100, "an unsettled hold must count as spent"
    assert ledger.remaining("run") == 0


def test_sixteen_concurrent_reservations_cannot_together_exceed_the_ceiling():
    """Measured rather than reasoned about, with real threads.

    Sixteen threads each try to reserve a tenth of the ceiling. At most ten can succeed; a
    check-then-add ledger would let all sixteen through.
    """
    ledger = opened(maximum_rcc=1_000)
    granted: list[int] = []
    refused: list[int] = []
    barrier = threading.Barrier(16)

    def attempt() -> None:
        barrier.wait()
        try:
            ledger.reserve("run", 100, kind="llm")
            granted.append(1)
        except BudgetExceeded:
            refused.append(1)

    threads = [threading.Thread(target=attempt) for _ in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(granted) == 10, f"expected exactly 10 grants, got {len(granted)}"
    assert len(refused) == 6
    assert ledger.spent("run") <= 1_000


def test_settling_below_the_estimate_returns_the_difference():
    """Over-reserving is only acceptable because the surplus comes back."""
    ledger = opened(maximum_rcc=1_000)
    reservation = ledger.reserve("run", 500, kind="llm")
    ledger.settle(reservation, 12)
    assert ledger.spent("run") == 12
    assert ledger.remaining("run") == 988


def test_releasing_a_request_that_was_never_sent_costs_nothing():
    ledger = opened(maximum_rcc=1_000)
    reservation = ledger.reserve("run", 500, kind="llm")
    ledger.release(reservation)
    assert ledger.spent("run") == 0


def test_settling_twice_is_refused_rather_than_double_charging():
    ledger = opened(maximum_rcc=1_000)
    reservation = ledger.reserve("run", 100, kind="llm")
    ledger.settle(reservation, 100)
    with pytest.raises(BudgetExceeded, match="already settled"):
        ledger.settle(reservation, 100)


def test_a_call_costing_more_than_its_estimate_is_recorded_in_full():
    """The provider already billed it. A receipt that disagreed with the invoice would defeat
    the reconciliation in 27, which is the check that catches spend outside the receipted path."""
    ledger = opened(maximum_rcc=100)
    reservation = ledger.reserve("run", 50, kind="llm")
    ledger.settle(reservation, 400)
    assert ledger.spent("run") == 400


def test_an_overshoot_bounds_the_breach_to_one_call():
    """After an overshoot the next reservation is refused, so it cannot compound."""
    ledger = opened(maximum_rcc=100)
    ledger.settle(ledger.reserve("run", 50, kind="llm"), 400)
    with pytest.raises(BudgetExceeded):
        ledger.reserve("run", 1, kind="llm")


# --------------------------------------------------------------------------
# Fail closed: an unknown run has no budget
# --------------------------------------------------------------------------


def test_a_run_that_was_never_admitted_cannot_spend():
    """A gateway that lost its ledger must not hand a restarted run a fresh budget."""
    with pytest.raises(BudgetExceeded, match="not admitted"):
        Ledger().reserve("unknown", 1, kind="llm")


def test_a_closed_run_cannot_spend_again():
    ledger = opened()
    ledger.close("run")
    with pytest.raises(BudgetExceeded, match="not admitted"):
        ledger.reserve("run", 1, kind="llm")


def test_readmitting_an_open_run_does_not_reset_its_spend():
    """Re-admission as a budget reset is the re-openable run gate, with extra steps."""
    ledger = opened(maximum_rcc=1_000)
    ledger.settle(ledger.reserve("run", 100, kind="llm"), 100)
    ledger.admit("run", maximum_rcc=1_000, maximum_requests=100, maximum_search_calls=50)
    assert ledger.spent("run") == 100


def test_remaining_is_zero_for_a_run_that_does_not_exist():
    assert Ledger().remaining("nope") == 0


# --------------------------------------------------------------------------
# Request and search ceilings, not only RCC
# --------------------------------------------------------------------------


def test_the_request_ceiling_binds_independently_of_rcc():
    """A run bounded only on RCC could make very many very cheap calls, which is a denial of
    service against the gateway rather than a budget breach."""
    ledger = Ledger()
    ledger.admit("run", maximum_rcc=1_000_000, maximum_requests=3, maximum_search_calls=50)
    for _ in range(3):
        ledger.reserve("run", 1, kind="llm")
    with pytest.raises(BudgetExceeded, match="its ceiling"):
        ledger.reserve("run", 1, kind="llm")


def test_the_search_ceiling_binds_separately_from_the_request_ceiling():
    ledger = Ledger()
    ledger.admit("run", maximum_rcc=1_000_000, maximum_requests=100, maximum_search_calls=2)
    ledger.reserve("run", 1, kind="search")
    ledger.reserve("run", 1, kind="search")
    with pytest.raises(BudgetExceeded, match="search calls"):
        ledger.reserve("run", 1, kind="search")


def test_search_calls_also_count_against_the_request_ceiling():
    """One ceiling over both, per 5.3: "bounded by one ceiling rather than two.\""""
    ledger = Ledger()
    ledger.admit("run", maximum_rcc=1_000_000, maximum_requests=2, maximum_search_calls=50)
    ledger.reserve("run", 1, kind="search")
    ledger.reserve("run", 1, kind="llm")
    with pytest.raises(BudgetExceeded):
        ledger.reserve("run", 1, kind="search")


# --------------------------------------------------------------------------
# Closing accounts for what was left open
# --------------------------------------------------------------------------


def test_closing_reports_totals():
    ledger = opened(maximum_rcc=1_000)
    ledger.settle(ledger.reserve("run", 100, kind="llm"), 70)
    ledger.settle(ledger.reserve("run", 50, kind="search"), 50)
    totals = ledger.close("run")
    assert totals == {"rcc": 120, "requests": 2, "search_calls": 1}


def test_a_hold_still_open_at_close_is_charged_rather_than_forgotten(caplog):
    """It is a call whose outcome nobody recorded, so the provider may have billed it. Charging
    for a call that may have happened is the safe direction."""
    ledger = opened(maximum_rcc=1_000)
    ledger.reserve("run", 250, kind="llm")
    with caplog.at_level("ERROR"):
        totals = ledger.close("run")
    assert totals["rcc"] == 250
    assert "outstanding reservation" in caplog.text


# --------------------------------------------------------------------------
# The estimate must be an upper bound
# --------------------------------------------------------------------------


def test_the_estimate_bounds_the_actual_cost_from_above():
    """If the estimate under-shoots, the ceiling does not hold — the shortfall accumulates
    across every call in the run."""
    estimate = estimate_rcc(PRICES, kind="llm", prompt_tokens=1_000, max_tokens=4_000)
    actual = PRICES.rcc_for_tokens(1_000, 4_000)
    assert estimate >= actual


def test_an_llm_call_without_a_token_bound_is_refused():
    """No bound on the response means no upper bound to reserve."""
    with pytest.raises(BudgetExceeded, match="must declare max_tokens"):
        estimate_rcc(PRICES, kind="llm", prompt_tokens=100, max_tokens=0)


def test_a_search_estimate_is_the_flat_rate():
    assert estimate_rcc(PRICES, kind="search") == PRICES.rcc_per_search


def test_an_embedding_is_priced_on_input_only():
    assert estimate_rcc(PRICES, kind="embedding", prompt_tokens=2_000) == PRICES.rcc_for_tokens(
        2_000, 0
    )


def test_an_unknown_call_kind_has_no_price_and_is_refused():
    """A kind with no price would otherwise be free, and an unmetered call is worth provoking."""
    with pytest.raises(BudgetExceeded, match="no price"):
        estimate_rcc(PRICES, kind="telepathy")


def test_two_ledgers_do_not_share_state():
    """A module-global ledger would make two gateways in one process share a budget."""
    first, second = opened(maximum_rcc=100), opened(maximum_rcc=100)
    first.settle(first.reserve("run", 100, kind="llm"), 100)
    assert second.remaining("run") == 100
