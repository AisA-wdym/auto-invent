"""The round driver: the loop that turns scheduler decisions into work.

The scheduler's tests establish that the *decisions* are right. These establish that the loop obeys
them, which is a different claim and the one with the failure modes that matter in production: a
step run twice, a step recorded before it ran, an abandoned round re-decided every tick, a restart
that resumes at the wrong place.

Everything here runs against `FakeChain` and a recording step double, so a whole day passes in
milliseconds. That is the point of the split — the alternative is a live chain and a stopwatch, and
"we were down across the salt boundary" is not a state you can arrange that way.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import pytest

from chain.client import ChainError, FakeChain
from validator.cycle import CycleConfig
from validator.driver import IMPLEMENTATION, Driver, describe
from validator.roundstate import InMemoryRoundStore, RoundState
from validator.scheduler import Step

pytestmark = pytest.mark.determinism

DAY = 7_200


def cycle(**over) -> CycleConfig:
    fields = dict(
        blocks_per_day=DAY,
        submission_close_offset=-600,
        salt_commit_offset=-450,
        randomness_offset=-300,
        pack_commit_offset=-100,
        reveal_offset=0,
        execution_close_offset=4_200,
        weights_offset=6_900,
        anchor_block=0,
        anchor_date="2026-01-01",
    )
    fields.update(over)
    return CycleConfig(**fields)


@dataclass
class RecordingSteps:
    """Records every call and returns the state unchanged, unless told to raise.

    Returning the state rather than a fresh one matters: the driver appends the step name to
    whatever the step returned, so a step that dropped the accumulated list would silently reset a
    round's progress and the round would restart from COMMIT_SALT.
    """

    calls: list[tuple[str, str, int]] = field(default_factory=list)
    raise_on: str = ""
    #: What each step adds to the state, so a test can check the driver keeps a step's own writes.
    marks: dict[str, str] = field(default_factory=dict)

    def _record(self, name: str, state: RoundState, block: int) -> RoundState:
        self.calls.append((name, state.date, block))
        if self.raise_on == name:
            raise RuntimeError(f"{name} broke")
        if name in self.marks:
            return replace(state, pack_hash=self.marks[name])
        return state

    def commit_salt(self, state, *, block):
        return self._record("commit_salt", state, block)

    def generate(self, state, *, block, deadline_block):
        return self._record("generate", state, block)

    def execute(self, state, *, block, deadline_block):
        return self._record("execute", state, block)

    def score(self, state, *, block, deadline_block):
        return self._record("score", state, block)

    def submit_weights(self, state, *, block):
        return self._record("submit_weights", state, block)


def only(outcomes, round_id: str):
    """The outcome for one round.

    Needed because two rounds are usually live: at a block 400 before an epoch, today's round is
    committing its salt and yesterday's is abandoned for never having been started by this process.
    Indexing the list picks whichever happens to be first.
    """
    matched = [outcome for outcome in outcomes if outcome.round_id == round_id]
    assert len(matched) == 1, (round_id, outcomes)
    return matched[0]


def driver(*, block: int, steps: RecordingSteps | None = None, store=None, config=None):
    chain = FakeChain(netuid=1)
    chain.advance(block - chain.current_block())
    return Driver(
        chain=chain,
        cycle=config or cycle(),
        store=store if store is not None else InMemoryRoundStore(),
        steps=steps or RecordingSteps(),
        sleep=lambda _seconds: None,
    )


# --------------------------------------------------------------------------
# A whole day, one step at a time
# --------------------------------------------------------------------------


def test_a_full_day_runs_every_step_once_in_order():
    """The end-to-end claim, driven by advancing a fake chain rather than by calling steps directly.

    Each step is asserted to run exactly once. A driver that re-ran a step because progress had not
    been persisted would still produce all five names, in order, and this would still fail.
    """
    steps = RecordingSteps()
    store = InMemoryRoundStore()
    config = cycle()
    chain = FakeChain(netuid=1)
    chain.advance(3 * DAY - 500 - chain.current_block())
    engine = Driver(
        chain=chain, cycle=config, store=store, steps=steps, sleep=lambda _s: None
    )

    # One tick per 50 blocks through the round, which is finer than every boundary gap.
    for _ in range(160):
        engine.tick(chain.current_block())
        chain.advance(50)

    ran = [name for name, round_id, _ in steps.calls if round_id == "2026-01-04"]
    assert ran == ["commit_salt", "generate", "execute", "score", "submit_weights"]


def test_the_completed_steps_are_persisted_as_they_happen():
    """The recovery record. Written after each step, so a restart resumes rather than repeats."""
    steps = RecordingSteps()
    store = InMemoryRoundStore()
    engine = driver(block=3 * DAY - 400, steps=steps, store=store)

    engine.tick(3 * DAY - 400)
    state = store.read("2026-01-04")
    assert state is not None
    assert state.steps_done == ("COMMIT_SALT",)

    engine.tick(3 * DAY - 250)
    assert store.read("2026-01-04").steps_done == ("COMMIT_SALT", "GENERATE")


def test_a_step_keeps_what_it_wrote_to_the_state():
    """The driver appends to the state the step returned. A driver that rebuilt the state from the
    store instead would discard the pack hash the step had just committed."""
    steps = RecordingSteps(marks={"generate": "sha256:" + "ab" * 32})
    store = InMemoryRoundStore()
    engine = driver(block=3 * DAY - 400, steps=steps, store=store)
    engine.tick(3 * DAY - 400)
    engine.tick(3 * DAY - 250)
    assert store.read("2026-01-04").pack_hash == "sha256:" + "ab" * 32


def test_only_one_step_runs_per_round_per_tick():
    """The block height is read once at the top of a tick. A step that took twenty minutes leaves it
    stale, so acting on it again would decide a later step against an earlier chain."""
    steps = RecordingSteps()
    engine = driver(block=3 * DAY - 400, steps=steps)
    engine.tick(3 * DAY - 400)
    assert len(steps.calls) == 1


def test_the_phase_recorded_is_the_phase_of_the_round_that_ran():
    """Not the phase of the block. In the overlap those differ, and a status page showing the wrong
    one claims a round is executing while it is being scored."""
    store = InMemoryRoundStore()
    engine = driver(block=3 * DAY - 400, store=store)
    engine.tick(3 * DAY - 400)
    assert store.read("2026-01-04").phase == "AWAITING_RANDOMNESS"


# --------------------------------------------------------------------------
# Restart and recovery
# --------------------------------------------------------------------------


def test_a_restart_resumes_at_the_step_it_had_reached():
    steps = RecordingSteps()
    store = InMemoryRoundStore()
    store.write(
        RoundState(
            date="2026-01-04",
            validator_hotkey="5Gv",
            phase="AWAITING_RANDOMNESS",
            block=3 * DAY - 400,
            steps_done=("COMMIT_SALT",),
        )
    )
    driver(block=3 * DAY - 250, steps=steps, store=store).tick(3 * DAY - 250)
    assert [name for name, *_ in steps.calls] == ["generate"]


def test_a_validator_starting_mid_round_abandons_that_round_rather_than_joining_it():
    """No salt commitment means no derivable seed and nothing for a peer to check a pack against."""
    steps = RecordingSteps()
    store = InMemoryRoundStore()
    outcomes = driver(block=3 * DAY + 1_000, steps=steps, store=store).tick(3 * DAY + 1_000)
    assert steps.calls == []
    outcome = only(outcomes, "2026-01-04")
    assert outcome.kind == "abandoned"
    assert outcome.step is Step.COMMIT_SALT
    assert store.read("2026-01-04").abandoned


def test_an_abandoned_round_is_not_re_decided_on_every_tick():
    """Recorded rather than re-derived. A loop that re-decided it would log the same abandonment for
    the rest of the day, and would call the step implementation again if the reason ever changed."""
    store = InMemoryRoundStore()
    steps = RecordingSteps()
    engine = driver(block=3 * DAY + 1_000, steps=steps, store=store)
    first = only(engine.tick(3 * DAY + 1_000), "2026-01-04")
    second = only(engine.tick(3 * DAY + 1_050), "2026-01-04")
    assert first.kind == second.kind == "abandoned"
    assert first.step is Step.COMMIT_SALT
    # The second tick reports the stored reason and carries no step, because nothing was decided.
    assert second.step is None
    assert second.detail == first.detail
    assert steps.calls == []


def test_an_unrecognised_recorded_step_raises_rather_than_being_ignored():
    """A step name from a newer release is not dropped. Dropping it would re-run the step, and
    running COMMIT_SALT twice publishes two commitments for one round."""
    store = InMemoryRoundStore()
    store.write(
        RoundState(
            date="2026-01-04",
            validator_hotkey="5Gv",
            phase="AWAITING_RANDOMNESS",
            block=3 * DAY - 400,
            steps_done=("COMMIT_SALT", "REPLICATE_PEER"),
        )
    )
    with pytest.raises(ChainError, match="does not know"):
        driver(block=3 * DAY - 250, store=store).tick(3 * DAY - 250)


# --------------------------------------------------------------------------
# Failure
# --------------------------------------------------------------------------


def test_a_step_that_raises_abandons_the_round_rather_than_retrying():
    """Deliberately expensive: a transient RPC failure during COMMIT_SALT loses the day. A retry
    needs a window to retry inside, and every window here is a security boundary."""
    steps = RecordingSteps(raise_on="commit_salt")
    store = InMemoryRoundStore()
    outcomes = driver(block=3 * DAY - 400, steps=steps, store=store).tick(3 * DAY - 400)
    assert only(outcomes, "2026-01-04").kind == "failed"
    assert "commit_salt broke" in store.read("2026-01-04").abandoned


def test_a_failed_step_is_not_recorded_as_done():
    """The whole reason progress is written after the step. A step marked done before it ran makes a
    crash mid-step look like success."""
    steps = RecordingSteps(raise_on="generate")
    store = InMemoryRoundStore()
    engine = driver(block=3 * DAY - 400, steps=steps, store=store)
    engine.tick(3 * DAY - 400)
    engine.tick(3 * DAY - 250)
    state = store.read("2026-01-04")
    assert state.steps_done == ("COMMIT_SALT",)
    assert "generate broke" in state.abandoned


def test_a_failure_in_one_round_does_not_touch_the_other_live_round():
    """In the overlap two rounds are live. One abandoning must not take the other with it — that
    would turn a single bad day into two."""
    store = InMemoryRoundStore()
    store.write(
        RoundState(
            date="2026-01-04",
            validator_hotkey="5Gv",
            phase="SCORING",
            block=3 * DAY,
            steps_done=("COMMIT_SALT", "GENERATE", "EXECUTE", "SCORE"),
        )
    )
    steps = RecordingSteps(raise_on="submit_weights")
    block = 3 * DAY + 6_950
    outcomes = driver(block=block, steps=steps, store=store).tick(block)
    assert only(outcomes, "2026-01-04").kind == "failed"
    # The next round is separately live and separately decided. It is abandoned here for its own
    # reason — this process never committed its salt — and not because its neighbour failed.
    neighbour = only(outcomes, "2026-01-05")
    assert neighbour.kind == "abandoned"
    assert neighbour.step is Step.COMMIT_SALT
    assert "submit_weights" not in store.read("2026-01-05").abandoned


# --------------------------------------------------------------------------
# Two live rounds
# --------------------------------------------------------------------------


def test_both_live_rounds_are_advanced_on_one_tick():
    """Yesterday's weight submission and today's salt commitment fall due together."""
    store = InMemoryRoundStore()
    store.write(
        RoundState(
            date="2026-01-04",
            validator_hotkey="5Gv",
            phase="SCORING",
            block=3 * DAY,
            steps_done=("COMMIT_SALT", "GENERATE", "EXECUTE", "SCORE"),
        )
    )
    steps = RecordingSteps()
    block = 3 * DAY + 6_800
    outcomes = driver(block=block, steps=steps, store=store).tick(block)
    assert {outcome.round_id for outcome in outcomes} == {"2026-01-04", "2026-01-05"}
    assert ("commit_salt", "2026-01-05", block) in steps.calls


def test_the_older_round_is_advanced_first():
    """`live_rounds` is oldest first and the driver keeps that order. In the overlap the older round
    is the one with a hard chain deadline behind it."""
    store = InMemoryRoundStore()
    store.write(
        RoundState(
            date="2026-01-04",
            validator_hotkey="5Gv",
            phase="SCORING",
            block=3 * DAY,
            steps_done=("COMMIT_SALT", "GENERATE", "EXECUTE", "SCORE"),
        )
    )
    steps = RecordingSteps()
    block = 3 * DAY + 6_950
    outcomes = driver(block=block, steps=steps, store=store).tick(block)
    assert [outcome.round_id for outcome in outcomes] == ["2026-01-04", "2026-01-05"]
    assert steps.calls[0][1] == "2026-01-04"


# --------------------------------------------------------------------------
# The loop itself
# --------------------------------------------------------------------------


def test_the_loop_stops_after_the_tick_budget():
    """`--once` uses `max_ticks=1` rather than a separate single-shot path. Two code paths through a
    loop means one of them is the one nobody runs."""
    steps = RecordingSteps()
    engine = driver(block=3 * DAY - 400, steps=steps)
    outcomes = engine.run(max_ticks=1)
    assert len(steps.calls) == 1
    assert outcomes


def test_the_loop_does_not_sleep_after_a_step_ran():
    """The next step may already be due, so the height is re-read immediately. A loop that slept a
    poll interval after every step would add 24 seconds five times a day for nothing."""
    slept: list[float] = []
    chain = FakeChain(netuid=1)
    chain.advance(3 * DAY - 400 - chain.current_block())
    engine = Driver(
        chain=chain,
        cycle=cycle(),
        store=InMemoryRoundStore(),
        steps=RecordingSteps(),
        sleep=slept.append,
    )
    engine.run(max_ticks=2)
    assert slept == []


def test_the_loop_sleeps_towards_the_next_boundary_rather_than_polling_blindly():
    """Between the salt commit and the randomness there are 150 blocks with nothing to do. A fixed
    poll interval would wake 75 times."""
    slept: list[float] = []
    store = InMemoryRoundStore()
    store.write(
        RoundState(
            date="2026-01-04",
            validator_hotkey="5Gv",
            phase="AWAITING_RANDOMNESS",
            block=3 * DAY - 400,
            steps_done=("COMMIT_SALT",),
        )
    )
    chain = FakeChain(netuid=1)
    chain.advance(3 * DAY - 400 - chain.current_block())
    engine = Driver(
        chain=chain, cycle=cycle(), store=store, steps=RecordingSteps(), sleep=slept.append
    )
    engine.run(max_ticks=2)
    # 100 blocks to the randomness at twelve seconds each, capped at ten minutes.
    assert slept == [600.0]


def test_the_sleep_is_capped_so_a_stalled_chain_is_noticed():
    """The wake block is a floor, not a promise. A chain that stops producing would otherwise leave
    the validator asleep for the length of the gap it computed."""
    slept: list[float] = []
    chain = FakeChain(netuid=1)
    chain.advance(3 * DAY + 100 - chain.current_block())
    store = InMemoryRoundStore()
    store.write(
        RoundState(
            date="2026-01-04",
            validator_hotkey="5Gv",
            phase="EXECUTING",
            block=3 * DAY + 100,
            steps_done=("COMMIT_SALT", "GENERATE", "EXECUTE"),
        )
    )
    engine = Driver(
        cycle=cycle(), chain=chain, store=store, steps=RecordingSteps(), sleep=slept.append
    )
    engine.run(max_ticks=2)
    assert slept and max(slept) <= 600.0


def test_a_finished_round_produces_no_work_and_a_plain_poll():
    slept: list[float] = []
    store = InMemoryRoundStore()
    store.write(
        RoundState(
            date="2026-01-04",
            validator_hotkey="5Gv",
            phase="DONE",
            block=3 * DAY,
            steps_done=tuple(step.name for step in Step),
        )
    )
    chain = FakeChain(netuid=1)
    chain.advance(3 * DAY + 1_000 - chain.current_block())
    steps = RecordingSteps()
    engine = Driver(cycle=cycle(), chain=chain, store=store, steps=steps, sleep=slept.append)
    outcomes = engine.run(max_ticks=2)
    assert steps.calls == []
    assert {outcome.kind for outcome in outcomes} == {"complete"}
    assert slept == [24.0]


# --------------------------------------------------------------------------
# Construction and reporting
# --------------------------------------------------------------------------


def test_every_scheduler_step_has_an_implementation():
    """A step added to the scheduler without a method here would be attempted and would fail with an
    AttributeError mid-round, on a live chain, once. Asserted against the mapping directly rather
    than through a monkeypatched enum — patching the enum tests the patch."""
    assert set(IMPLEMENTATION) == set(Step)


def test_a_steps_object_missing_a_method_is_a_construction_error():
    """Checked at construction, because the moment a step is due is once a day inside a window that
    does not reopen."""

    class Incomplete:
        def commit_salt(self, state, *, block):  # pragma: no cover - never called
            return state

    with pytest.raises(ValueError, match="no callable 'generate'"):
        Driver(
            chain=FakeChain(netuid=1),
            cycle=cycle(),
            store=InMemoryRoundStore(),
            steps=Incomplete(),  # type: ignore[arg-type]
        )


def test_describe_names_every_live_round_and_what_happened_to_it():
    store = InMemoryRoundStore()
    outcomes = driver(block=3 * DAY - 400, store=store).tick(3 * DAY - 400)
    rendered = describe(outcomes)
    assert "2026-01-04" in rendered
    assert "COMMIT_SALT" in rendered


def test_describe_says_so_when_no_round_is_live():
    """Between rounds is a normal state, and an empty report reads as a broken validator."""
    assert "no round is live" in describe([])
