"""The composition root's five steps, and the loop's refusal to start half-built.

`test_scheduler.py` holds the decisions and `test_driver.py` holds the loop. What is left untested
by both is the wiring in `validator/__main__.py`: the adapter that gives `Validator` the driver's
method names, the steps themselves, and `main`'s refusal to enter a loop that is not all built.

That refusal is the reason this file exists. Without it a deployment would publish a salt
commitment and a pack hash on a live chain, spend a generation budget, and abandon the round at the
first missing step — once a day, indefinitely, with two extrinsics a day of evidence that it tried.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from dataclasses import replace

import pytest

from chain.client import ChainError, FakeChain
from validator.__main__ import Validator, ValidatorSteps, build_driver, main
from validator.challenge_factory.store import StoreError
from validator.driver import IMPLEMENTATION
from validator.roundstate import RoundState, StandingEntry
from validator.scheduler import Step

pytestmark = pytest.mark.determinism

SEASON = json.loads(pathlib.Path("config/season.testnet.json").read_text())


def args(**over) -> argparse.Namespace:
    fields = dict(
        season=pathlib.Path("config/season.testnet.json"),
        netuid=1,
        network="finney",
        wallet="default",
        hotkey="default",
        rcg_endpoint="http://127.0.0.1:8081",
        redis_url="",
        publish_redis_url="",
        workspace=pathlib.Path("var/runs"),
        concurrency=2,
        runner_token="runner-token",
        check=False,
        once=False,
        log_level="INFO",
    )
    fields.update(over)
    return argparse.Namespace(**fields)


def validator(chain: FakeChain | None = None) -> Validator:
    return Validator(SEASON, chain=chain or FakeChain(netuid=1), args=args())


def state(**over) -> RoundState:
    fields = dict(
        date="2026-08-03",
        validator_hotkey="5Fvalidator",
        phase="AWAITING_RANDOMNESS",
        block=1_000,
    )
    fields.update(over)
    return RoundState(**fields)


# --------------------------------------------------------------------------
# The adapter
# --------------------------------------------------------------------------


def test_the_adapter_exposes_every_name_the_driver_looks_up():
    """The driver calls `getattr(steps, IMPLEMENTATION[step])`. A name it cannot find would fail
    at the moment the step is due — once a day, inside a window that does not reopen."""
    steps = ValidatorSteps(validator())
    for name in IMPLEMENTATION.values():
        assert callable(getattr(steps, name)), name


def test_the_adapter_is_needed_because_two_names_collide():
    """`Validator.commit_salt` is the chain call and takes a date and a salt; the *step* draws the
    salt and records it. Collapsing them would make one method mean both."""
    import inspect

    assert "date" in inspect.signature(Validator.commit_salt).parameters
    assert "state" in inspect.signature(Validator.commit_salt_step).parameters


def test_build_driver_wires_the_round_store_rather_than_the_challenge_store():
    """Two stores share one Redis on different prefixes. The driver needs the round one; handing it
    the challenge store would fail on the first `read`, at the first tick."""
    engine = build_driver(validator())
    assert engine.store is not None
    assert hasattr(engine.store, "read_public")


# --------------------------------------------------------------------------
# Which steps are missing, and what that stops
# --------------------------------------------------------------------------


def test_no_step_is_unimplemented():
    """The guard that kept the loop from starting while execute and score were placeholders. It
    stays, because it is what a half-built release trips over rather than discovering on chain."""
    assert validator().unimplemented_steps() == ()


def test_execute_refuses_a_round_with_no_stored_pack():
    """The pack's hash is on chain, so it cannot be regenerated. A round that lost it is over —
    executing against a freshly generated pack would test laboratories on problems no commitment
    vouches for."""
    with pytest.raises(StoreError, match="no pack is stored"):
        ValidatorSteps(validator()).execute(state(), block=1_000, deadline_block=2_000)


def test_score_refuses_to_re_run_executions_it_cannot_find():
    """Re-running now would give these laboratories a window nobody else had. The round is lost
    instead, which is the expensive answer and the only fair one."""
    with pytest.raises(StoreError, match="no executions are stored"):
        ValidatorSteps(validator()).score(state(), block=1_000, deadline_block=2_000)


def test_main_enters_the_loop_now_that_every_step_is_built(monkeypatch):
    """It refused with exit code 4 while execute and score were placeholders. One tick against a
    fake chain: the round has no salt commitment from this process, so it is abandoned by name —
    which is the loop working, not the loop failing."""

    def fake_chain(**_kwargs):
        # Advanced into the epoch the shipped anchor names, or the anchor check fails first.
        chain = FakeChain(netuid=1)
        anchor = int(SEASON["cycle"]["anchor_block"])
        chain.advance(anchor + 1_000 - chain.current_block())
        return chain

    monkeypatch.setattr("validator.__main__.BittensorChain", fake_chain)
    assert main(["--season", "config/season.testnet.json", "--netuid", "1", "--once"]) == 0


def test_check_still_passes_while_the_loop_cannot_run():
    """`--check` validates the configuration and says nothing about whether every step is built. The
    two answer different questions, and conflating them would make a config error and a missing step
    indistinguishable."""
    assert main(["--season", "config/season.testnet.json", "--check"]) == 0


# --------------------------------------------------------------------------
# The three steps that are built
# --------------------------------------------------------------------------


def test_commit_salt_records_the_salt_and_its_commitment_for_recovery():
    """The commitment binds a value only this process knows. A restart before generation without the
    salt leaves a commitment on chain that no seed can be derived against."""
    chain = FakeChain(netuid=1)
    produced = ValidatorSteps(validator(chain)).commit_salt(state(), block=chain.current_block())
    assert len(produced.salt_hex) == 64
    assert produced.salt_commitment.startswith("sha256:")
    assert len(produced.salt_commitment) == len("sha256:") + 64
    assert chain.live_commitments


def test_two_rounds_draw_different_salts():
    """A reused salt makes two days' seeds differ only by their date, which is a published value."""
    steps = ValidatorSteps(validator())
    first = steps.commit_salt(state(date="2026-08-03"), block=1_000)
    second = steps.commit_salt(state(date="2026-08-04"), block=1_000)
    assert first.salt_hex != second.salt_hex


def test_generate_refuses_when_the_recorded_salt_is_missing():
    """Rather than drawing a fresh one. The commitment on chain is to the old salt, so a fresh one
    would generate a pack whose commitment no peer can verify — and the failure would be silent."""
    with pytest.raises(ChainError, match="no recorded salt"):
        ValidatorSteps(validator()).generate(state(), block=1_000, deadline_block=2_000)


def test_submit_weights_refuses_an_empty_field_rather_than_burning_by_default():
    """An empty standings list would allocate everything to the burn uid, which is 20.4's outcome
    for a round where nobody qualified — a different claim from a round that was never scored."""
    with pytest.raises(ChainError, match="no standings"):
        ValidatorSteps(validator()).submit_weights(state(), block=1_000)


def test_submit_weights_allocates_from_the_published_standings():
    """Read rather than recomputed. A second computation here would eventually disagree with the
    numbers the round published, and the published ones are what a miner checked."""
    chain = FakeChain(netuid=1)
    subject = validator(chain)
    scored = replace(
        state(phase="AWAITING_WEIGHTS"),
        floor_ppm=500_000,
        standings=(
            StandingEntry(1, "5Fa", 712_000, 690_000, 8, 45_000, True, 0),
            StandingEntry(2, "5Fb", 480_000, 455_000, 8, 220_000, False, 0),
        ),
    )
    produced = ValidatorSteps(subject).submit_weights(scored, block=chain.current_block())
    assert chain.submitted, "no weight vector reached the chain"
    uids, weights, _version = chain.submitted[-1]
    assert 1 in uids
    assert sum(weights) > 0
    assert produced.burned is False


def test_a_laboratory_with_a_failed_gate_is_not_a_candidate():
    """13's gates are fatal and cannot be offset by a score. The flag is read from the published lab
    record rather than re-derived, because a second derivation would eventually disagree with
    what the miner was shown."""
    chain = FakeChain(netuid=1)
    from validator.roundstate import LabStatus

    scored = replace(
        state(phase="AWAITING_WEIGHTS"),
        floor_ppm=500_000,
        labs=(LabStatus(1, "5Fa", "complete", 8, 8, ("13.6 budget exceeded",), 400),),
        standings=(StandingEntry(1, "5Fa", 900_000, 900_000, 8, 0, True, 0),),
    )
    produced = ValidatorSteps(validator(chain)).submit_weights(scored, block=chain.current_block())
    # Nothing qualified, so 20.4's burn applies — and it applies because of the gate, not the score.
    assert produced.burned is True


# --------------------------------------------------------------------------
# The step set matches the scheduler's
# --------------------------------------------------------------------------


def test_every_scheduler_step_has_a_validator_method():
    """A step added to the scheduler with no `*_step` method would report as implemented — the
    `_not_implemented` marker is absent from a method that does not exist — and then fail at the
    moment it was due, which is once a day inside a window that does not reopen."""
    for step in Step:
        assert hasattr(Validator, f"{step.name.lower()}_step"), step.name
