"""The vertical slice, against real infrastructure. Opt-in, because it spends money.

    real container -> 13 hard gates -> canonicalise -> judge panels -> pairwise -> score -> weights

Everything else in the suite substitutes the provider and the container. This does not, and that is
the point: every defect the first run of this found was invisible to the rest of the suite.

## What it found the first time it ran

Recorded here because each was a defect no unit test could have caught, and the list is the argument
for keeping this file:

1. **The gateway image could not start.** `Dockerfile.gateway` never copied `config/`, so the season
   config it needs for the price table was absent. CI *built* the image and never ran it.
2. **`ail-miner init` wrote a laboratory with a syntax error.** The scaffold's `src/lab.py` was
   assembled from `repr()`'d fragments and interleaved a docstring into a function. Nothing had ever
   run the scaffold — whose entire purpose is to run on the first invocation.
3. **A five-idea portfolio truncated at `max_tokens=16_384`**, so the reference laboratory failed
   gate 13.1 with no output at all. A floor laboratory that fails 13.1 makes the floor zero.
4. **The RCC price table was out by a factor of 62.** One real portfolio call cost 25,114 RCC
   against a declared `maximum_rcc` of 400.
5. **The template declared a `model_snapshot` and did not send it**, so the receipt recorded an
   empty revision and gate 13.4 failed the floor laboratory.
6. **Gate 13.9 fired on "rate faster than the maximum"** — a sentence about rate limiting — because
   the pattern matched scoring vocabulary co-occurring rather than an imperative. A false positive
   on a fatal gate.
7. **Judge verdicts truncated at `max_tokens=1_536`**, because a reasoning model spends that budget
   thinking. The verdict was correctly recorded as an abstention, which dropped the panel below
   16.1's three families for a reason that looked like a model refusing to answer.

## Running it

    AI_VALIDATOR_OPENROUTER_KEY=... AI_TEST_MINER_OPENROUTER_KEY=... \\
      pytest -m live tests/integration/test_live_round.py

Needs Docker, a reachable gateway container, and both OpenRouter accounts. It costs a few dollars a
run and takes about ten minutes, so it is excluded from `make test` and marked `live`.
"""

from __future__ import annotations

import copy
import json
import os
import pathlib
import shutil
import subprocess
import time
import urllib.request

import pytest

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not os.environ.get("AI_VALIDATOR_OPENROUTER_KEY")
        or not os.environ.get("AI_TEST_MINER_OPENROUTER_KEY"),
        reason="needs both OpenRouter accounts; see the module docstring",
    ),
    pytest.mark.skipif(shutil.which("docker") is None, reason="needs Docker"),
]

GATEWAY = os.environ.get("AI_RCG_ENDPOINT_HOST", "http://127.0.0.1:8081")
RUNNER_SECRET = os.environ.get("AI_RUNNER_SECRET", "test-runner-secret")
LAB_IMAGE = os.environ.get("AI_LAB_IMAGE", "ail-ref-lab:test")

CHALLENGE_BODY = {
    "challenge_id": "sha256:" + "c" * 64,
    "domain": "distributed_coordination",
    "title": "Bounding tail latency in a fan-out read path",
    "problem_statement": (
        "A read fans out to twelve shards and returns when all twelve reply, so request latency is "
        "the latency of the slowest shard. At the 99th percentile the request is as slow as the "
        "worst shard on its worst day, and adding shards makes it worse. Practitioners hedge with "
        "duplicate requests, doubling load exactly when the system is already struggling."
    ),
    "research_objective": (
        "Bound 99th-percentile fan-out latency without more than 10% extra requests issued."
    ),
    "current_baseline": "Hedged requests after a fixed delay; tied requests with cancellation.",
    "known_attempts": ["Hedging after p95", "Tied requests"],
    "constraints": [
        "Must add at most 10% to total requests issued.",
        "Cannot assume shards report their own load honestly.",
    ],
    "forbidden_shortcuts": ["Reducing the shard count is not a solution."],
    "required_output": {
        "portfolio_size": 5,
        "ranked": True,
        "mechanism_required": True,
        "prior_art_comparison_required": True,
        "falsification_plan_required": True,
        "simulation_or_calculation_required": True,
    },
}


def _post(path: str, body: dict) -> dict:
    request = urllib.request.Request(  # noqa: S310 - a fixed local endpoint
        f"{GATEWAY}{path}",
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {RUNNER_SECRET}",
            "Content-Type": "application/json",
        },
    )
    return json.load(urllib.request.urlopen(request, timeout=1_200))  # noqa: S310


@pytest.fixture(scope="module")
def season() -> dict:
    return json.loads(pathlib.Path("config/season.example.json").read_text())


@pytest.fixture(scope="module")
def gateway_reachable() -> None:
    try:
        with urllib.request.urlopen(f"{GATEWAY}/health", timeout=10) as response:  # noqa: S310
            assert json.load(response)["status"] == "ok"
    except Exception as error:  # noqa: BLE001
        pytest.skip(f"no gateway at {GATEWAY}: {error}")


@pytest.fixture(scope="module")
def lab_bundle(tmp_path_factory) -> pathlib.Path:
    """A bundle written by `ail-miner init` and built into an image.

    Built from the scaffold rather than from a fixture, because "the scaffold runs" is one of the
    things this file exists to check — and it did not, the first time it was checked.
    """
    root = tmp_path_factory.mktemp("bundle") / "lab"
    from miner.cli.main import main as miner_main

    assert miner_main(["init", str(root)]) == 0

    for name, _content in (("manifest.json", None),):  # touch: the manifest must exist
        assert (root / name).is_file(), f"the scaffold did not write {name}"

    built = subprocess.run(  # noqa: S603
        ["docker", "build", "-q", "-t", LAB_IMAGE, "."],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if built.returncode != 0:
        pytest.skip(f"cannot build the scaffold image: {built.stderr[-400:]}")
    return root


@pytest.fixture(scope="module")
def executed(season, gateway_reachable, lab_bundle, tmp_path_factory):
    """Run the laboratory once, and share the result across the assertions below.

    Module-scoped deliberately: this costs money and takes minutes, so the tests that follow examine
    one run rather than each paying for their own.
    """
    import asyncio

    from validator.sandbox.container import Limits, SandboxRunner, assert_egress_confined
    from validator.sandbox.runner import Runner

    assert_egress_confined()

    pricing = season["providers"]["miner_pricing"]
    challenge = {
        **CHALLENGE_BODY,
        "resource_limits": {
            key: pricing[key]
            for key in ("maximum_rcc", "maximum_search_calls", "maximum_wall_time_seconds")
        },
    }
    manifest = json.loads((lab_bundle / "model_manifest.json").read_text())
    declared = {
        model["model_slug"]: model.get("model_snapshot", model["model_slug"])
        for model in manifest["models"]
    }

    digest = subprocess.run(  # noqa: S603
        ["docker", "inspect", LAB_IMAGE, "--format", "{{.Id}}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    runner = Runner(
        sandbox=SandboxRunner(),
        admit=lambda body: _post("/v1/runs", body)["session_token"],
        close=lambda run_id: _post(f"/v1/runs/{run_id}/close", {}),
        rcg_endpoint="http://rcg:8081",
        workspace=tmp_path_factory.mktemp("runs"),
    )

    now = int(time.time())
    result = asyncio.run(
        runner.execute(
            run_id=f"live-{now}",
            miner_hotkey="5Fminer",
            bundle_digest="sha256:" + "d" * 64,
            image_digest=digest,
            validator_hotkey="5Gtest",
            challenge=challenge,
            api_key=os.environ["AI_TEST_MINER_OPENROUTER_KEY"],
            allowed_models=list(declared),
            limits=Limits.from_season(
                season, wall_time_seconds=pricing["maximum_wall_time_seconds"]
            ),
            deadline="2026-12-31T00:00:00Z",
            expires_at=now + 1_700,
            episode_deadline=now + 1_750,
        )
    )
    return result, challenge, declared


# --------------------------------------------------------------------------
# The laboratory runs, and the controls hold while it does
# --------------------------------------------------------------------------


def test_the_scaffolded_laboratory_produces_a_portfolio(executed):
    """The floor laboratory has to work. If it does not, the qualification floor is zero and every
    miner clears it without doing anything."""
    result, _challenge, _declared = executed
    assert result.produced_output, f"{result.failure}\n{result.stderr_tail[-600:]}"
    assert result.exit_code == 0
    assert not result.timed_out


def test_it_returns_the_requested_number_of_ideas(executed):
    result, challenge, _ = executed
    ideas = result.portfolio["portfolio"]
    assert len(ideas) == challenge["required_output"]["portfolio_size"]


def test_every_idea_carries_a_mechanism_with_a_causal_chain(executed):
    """18.4 caps value and originality on a weak mechanism, so the floor must state one."""
    result, _, _ = executed
    for idea in result.portfolio["portfolio"]:
        mechanism = idea["mechanism"]
        assert mechanism["components"]
        assert len(mechanism["causal_explanation"]) > 80


def test_measured_usage_replaces_the_self_reported_claim(executed):
    """9.2. The template's own accounting was measured at 92 RCC against a real 36,466 — which is
    exactly why the claim is recorded as a claim and the measurement is used as the usage."""
    result, _, _ = executed
    assert result.measured_usage["rcc"] > 0
    assert result.claimed_usage != result.measured_usage
    assert result.usage_discrepancy()


def test_the_receipt_chain_verifies_and_bills_only_the_miner(executed):
    """3.4.4 point 3: per-account totals are what catch a call billed to the wrong side."""
    result, _, _ = executed
    assert result.receipt_calls
    assert result.chain_head
    assert {call["credential_owner"] for call in result.receipt_calls} == {"miner"}


def test_spend_stays_within_the_declared_ceiling(executed, season):
    result, _, _ = executed
    assert result.measured_usage["rcc"] <= season["providers"]["miner_pricing"]["maximum_rcc"]


# --------------------------------------------------------------------------
# Every hard gate passes on an honest laboratory
# --------------------------------------------------------------------------


def test_an_honest_laboratory_passes_all_thirteen_hard_gates(executed):
    """The assertion that found two defects: the template not sending its declared snapshot (13.4),
    and gate 13.9 firing on the phrase "rate faster than the maximum" (13.9).

    A gate is fatal, so a false positive here invalidates an honest response unappealably — which
    makes this the single most important assertion in the file."""
    from validator.scoring.gates import check_all

    result, challenge, declared = executed
    report = check_all(
        portfolio=result.portfolio,
        challenge=challenge,
        receipt_calls=result.receipt_calls,
        declared_models=declared,
        measured_rcc=result.measured_usage["rcc"],
        measured_search_calls=result.measured_usage["search_calls"],
        wall_seconds=result.wall_seconds,
    )
    assert report.valid, report.reason()


def test_canonicalisation_strips_presentation_and_substitutes_the_measurement(executed):
    from validator.canonicalizer.neutral import canonicalize

    result, _, _ = executed
    canonical = canonicalize(result.portfolio, measured_usage=result.measured_usage)
    assert "resource_usage_claim" not in canonical.body
    assert canonical.body["measured_resource_usage"] == result.measured_usage


# --------------------------------------------------------------------------
# The judge panels score it, with three families each
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def judged(executed, season):
    """Score the canonicalised portfolio on three criteria with the real panels."""
    import asyncio

    from gateway.adapters.openrouter import ModelPin
    from gateway.credentials import CredentialSet, ValidatorCredential
    from gateway.metering import Ledger, PriceTable
    from protocol.receipts import Receipt
    from validator.canonicalizer.neutral import canonicalize
    from validator.judge.panels import panels_from_season, pins_for
    from validator.judge.pointwise import aggregate, screen_portfolio
    from validator.model_client import ModelClient

    result, _, _ = executed
    canonical = canonicalize(result.portfolio, measured_usage=result.measured_usage)

    ledger = Ledger()
    ledger.admit("judge", maximum_rcc=40_000_000, maximum_requests=500, maximum_search_calls=0)
    panels = panels_from_season(season)
    client = ModelClient(
        credentials=CredentialSet(
            validator=ValidatorCredential("5Gval", os.environ["AI_VALIDATOR_OPENROUTER_KEY"])
        ),
        prices=PriceTable.from_season(season),
        ledger=ledger,
        receipt=Receipt("judge", "", "0" * 64, "0" * 64, "5Gval"),
        run_id="judge",
        pins={family: ModelPin(pin.slug, "") for family, pin in pins_for(panels).items()},
    )

    async def score() -> dict:
        out = {}
        for name in ("mechanism", "originality", "diversity"):
            scores = await screen_portfolio(client, panel=panels[name], portfolio=canonical.body)
            out[name] = (aggregate(scores), scores)
        return out

    return asyncio.run(score()), ledger, client


def test_every_criterion_is_scored_by_three_model_families(judged):
    """16.1 requires at least three, and the cap is on the family rather than the route."""
    scored, _ledger, _client = judged
    for name, ((_value, voters), scores) in scored.items():
        assert len({s.family for s in scores}) == 3, f"{name} had fewer than three families"
        lost = [(s.family, s.raw) for s in scores]
        assert voters >= 2, f"{name} lost more than one judge: {lost}"


def test_an_honest_portfolio_scores_above_the_bottom_anchors(judged):
    """Not a quality claim about the template — a check that the panel is discriminating at all. A
    panel returning 0 or 1 on a real frontier-model portfolio would be measuring nothing."""
    scored, _, _ = judged
    for name, ((value, _voters), _scores) in scored.items():
        assert value >= 500_000, f"{name} scored {value}, at or below 'plausible but incomplete'"


def test_validator_spend_is_billed_to_the_validator_alone(judged):
    """The other half of 3.4.4: judging is a validator cost and must never reach a miner's key."""
    _scored, ledger, client = judged
    assert ledger.spent("judge") > 0
    assert client.receipt.spend_by_owner()["miner"] == 0


# --------------------------------------------------------------------------
# An open finding, deliberately left failing rather than tuned away
# --------------------------------------------------------------------------


@pytest.mark.xfail(
    reason=(
        "MEASURED, NOT TUNED. Deleting every mechanism field from a real portfolio moved the "
        "mechanism panel from 916,666 to 833,333 ppm — a gap of 83,333 against the season's "
        "100,000 floor, so 7.4 step 5 condition 5 fails on a genuinely gutted portfolio.\n\n"
        "Two candidate causes, and they need different fixes:\n\n"
        "1. The degradation is too weak. `core_invention`, `problem_reframe` and `why_non_obvious` "
        "restate the mechanism in prose, so a judge reads it from the surrounding fields. The real "
        "probe should degrade those too.\n"
        "2. The threshold is miscalibrated for the panel's granularity. On a 5-anchor scale with 3 "
        "judges, one judge moving one anchor is 83,333 ppm — so a 100,000 floor requires *two* "
        "judges to move, and there is no way to express a smaller genuine difference.\n\n"
        "Lowering the threshold until this passes would be the worst available response: it would "
        "make condition 5 — the sharpest of the five, and the only one that tests the judge rather "
        "than the problem — satisfiable by noise. Left failing until degradation and threshold "
        "are derived together from measurement, which is 27's job."
    ),
    strict=False,
)
def test_the_panel_distinguishes_a_gutted_portfolio_from_a_whole_one(executed, season, judged):
    """7.4 step 5, condition 5. See the xfail reason — this is a recorded finding, not a bug."""
    import asyncio

    from validator.canonicalizer.neutral import canonicalize
    from validator.judge.panels import panels_from_season
    from validator.judge.pointwise import aggregate, screen_portfolio

    result, _, _ = executed
    _scored, _ledger, client = judged

    gutted = copy.deepcopy(result.portfolio)
    for idea in gutted["portfolio"]:
        idea["mechanism"] = {
            "components": [],
            "information_flow": "",
            "causal_explanation": "",
            "feedback_loops": [],
        }

    panels = panels_from_season(season)
    canonical = canonicalize(result.portfolio, measured_usage=result.measured_usage)
    damaged = canonicalize(gutted, measured_usage=result.measured_usage)

    async def gap() -> int:
        intact = aggregate(
            await screen_portfolio(client, panel=panels["mechanism"], portfolio=canonical.body)
        )[0]
        broken = aggregate(
            await screen_portfolio(client, panel=panels["mechanism"], portfolio=damaged.body)
        )[0]
        return intact - broken

    floor = season["challenge_generation"]["minimum_degradation_gap_ppm"]
    assert asyncio.run(gap()) >= floor
