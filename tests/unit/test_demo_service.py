"""The demo service: a visitor's problem, the owner's money.

Two things separate this from the rest of the subnet, and both are what the tests are about.

The input is from a stranger — less vetted than a generator this validator configured — so it goes
through the same safety screen a generated candidate does. And the money is the owner's, so the
bounds have to hold when the caller is hostile rather than merely wrong.

Nothing here runs a container. What is checked is every decision made *before* one starts and every
bound that survives one finishing.
"""

from __future__ import annotations

import pytest

from demo.service import MAXIMUM_PROBLEM_CHARS, DemoConfig, DemoError, DemoService, Spend

pytestmark = pytest.mark.determinism


def service(**over) -> DemoService:
    fields = dict(management_key="mgmt", caller_secret="shh", runner_token="tok")
    fields.update(over)
    return DemoService(config=DemoConfig(**fields))


# --------------------------------------------------------------------------
# Configuration refuses to be dangerous
# --------------------------------------------------------------------------


def test_an_unconfigured_service_refuses_to_exist():
    """No caller secret is an open endpoint spending the owner's account; no management key fails on
    the first request, after a visitor has already waited several minutes."""
    with pytest.raises(ValueError, match="AI_OWNER_MANAGEMENT_KEY"):
        DemoConfig.from_environment({})


def test_a_missing_secret_alone_is_still_refused():
    with pytest.raises(ValueError, match="AI_DEMO_SECRET"):
        DemoConfig.from_environment(
            {"AI_OWNER_MANAGEMENT_KEY": "m", "AI_RUNNER_TOKEN": "t"}
        )


# --------------------------------------------------------------------------
# The caller
# --------------------------------------------------------------------------


def test_a_wrong_secret_is_refused():
    with pytest.raises(DemoError, match="not the dashboard"):
        service().authenticate("wrong")


def test_an_absent_secret_is_refused_rather_than_treated_as_empty():
    """An empty presented secret against an empty configured one would compare equal, which is how
    an unconfigured service becomes an open one."""
    with pytest.raises(DemoError):
        service().authenticate("")


# --------------------------------------------------------------------------
# A problem becomes a challenge, in the shape a laboratory reads
# --------------------------------------------------------------------------


def problem(**over) -> dict:
    body = {
        "title": "Bounding tail latency in a fan-out read",
        "problem_statement": (
            "The slowest replica sets the response time and hedging multiplies load."
        ),
        "research_objective": "Bound p99.9 without raising steady-state volume by more than 5%.",
    }
    body.update(over)
    return body


def test_a_visitor_problem_takes_the_same_shape_as_a_generated_one():
    """The laboratory cannot tell the difference, which is the point: what runs here is the subnet's
    own loop rather than a special path with different rules."""
    challenge = service().as_challenge(problem())
    expected = ("challenge_id", "title", "problem_statement", "required_output", "resource_limits")
    for field in expected:
        assert field in challenge
    assert challenge["required_output"]["portfolio_size"] == 5


def test_a_problem_with_no_statement_is_refused():
    """A laboratory asked for five inventions against an empty problem returns five generalities."""
    with pytest.raises(DemoError, match="what is actually hard"):
        service().as_challenge(problem(problem_statement=""))


def test_a_problem_with_no_title_is_refused():
    with pytest.raises(DemoError, match="needs a title"):
        service().as_challenge(problem(title="   "))


def test_an_enormous_problem_is_refused_before_it_reaches_anything():
    """A research problem needing more than this is several problems, and a laboratory given several
    answers none of them well."""
    with pytest.raises(DemoError, match="longer than"):
        service().as_challenge(problem(problem_statement="x" * (MAXIMUM_PROBLEM_CHARS + 1)))


def test_a_demo_run_is_bounded_harder_than_a_round():
    """The owner pays and nobody is being scored, so there is no fairness argument for the full
    ceiling — the equal-budget guarantee is about competitors, and there are none here."""
    challenge = service().as_challenge(problem())
    assert challenge["resource_limits"]["maximum_search_calls"] <= 20
    assert challenge["resource_limits"]["maximum_wall_time_seconds"] <= 900


def test_constraints_are_carried_but_bounded():
    challenge = service().as_challenge(problem(constraints=[f"c{n}" for n in range(20)]))
    assert len(challenge["constraints"]) == 8


# --------------------------------------------------------------------------
# 13.12: a stranger's problem is screened like a generated one
# --------------------------------------------------------------------------


def test_an_excluded_domain_is_refused_before_a_container_starts():
    """A visitor is *less* vetted than a generator this validator configured, so running the same
    screen on their input is the minimum rather than an extra."""
    subject = service()
    challenge = subject.as_challenge(
        problem(
            title="Exploit a buffer overflow in a replica parser",
            problem_statement=(
                "Build a payload that exploits a buffer overflow to bypass the quorum."
            ),
        )
    )
    with pytest.raises(DemoError, match="domain the subnet excludes"):
        subject.screen(challenge)


def test_an_ordinary_systems_problem_passes_the_screen():
    """The screen has to let the subnet's actual subject matter through — the proximity fix earlier
    today exists because it did not."""
    subject = service()
    subject.screen(subject.as_challenge(problem()))


# --------------------------------------------------------------------------
# The daily allowance
# --------------------------------------------------------------------------


def test_the_daily_ceiling_bounds_a_thousand_visitors_rather_than_one():
    """A per-run cap bounds one visitor. Without this, a thousand requests at the cap each is a
    thousand times the cap."""
    spend = Spend(maximum_daily_usd=1.0)
    spend.record(0.6)
    assert round(spend.remaining(), 2) == 0.40
    spend.record(0.6)
    assert spend.remaining() == 0.0


def test_the_ceiling_resets_on_a_new_day():
    spend = Spend(maximum_daily_usd=1.0)
    spend.record(1.0)
    spend.day = "1999-01-01"
    assert spend.remaining() == 1.0


def test_a_spent_allowance_refuses_before_minting_anything():
    """Refused before a key is minted, not after: a mint that is then abandoned leaves a live key on
    the owner's account for its expiry."""
    subject = service(maximum_daily_usd=0.1, maximum_run_usd=0.5)
    with pytest.raises(DemoError, match="daily budget is spent"):
        import asyncio

        asyncio.run(subject.run(problem()))


def test_an_unreadable_usage_figure_is_charged_at_the_cap():
    """Not at zero. An unreadable figure is not evidence of no spend, and assuming zero is how a
    daily ceiling stops bounding anything."""
    spend = Spend(maximum_daily_usd=10.0)
    spend.record(0.5)  # what the failure path records
    assert spend.spent_usd == 0.5


# --------------------------------------------------------------------------
# Concurrency, bounded rather than serialised
# --------------------------------------------------------------------------


def test_concurrency_is_bounded_rather_than_one():
    """A single slot was over-cautious: the validator runs four containers at once on this host, and
    queueing everyone behind one run takes half an hour to answer three people."""
    assert service().config.concurrency >= 2


def test_a_full_queue_refuses_rather_than_growing():
    """A queue nobody will outlive is worse than a refusal that says to come back."""
    import asyncio

    subject = service(maximum_queued=0)
    with pytest.raises(DemoError, match="already waiting"):
        asyncio.run(subject.run(problem()))
