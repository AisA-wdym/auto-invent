"""The round scheduler: architecture.md 7.3 and 21.

What is being held here is not "the loop runs". It is that the loop cannot be made to run a step
outside its window — because every one of 7.3's three orderings is a statement about when something
happened, and a validator that catches up after a restart breaks all three while looking perfectly
healthy.

Each ordering has a test that puts the validator in the failure state directly rather than
describing it. A restart across the salt boundary is one call here and a stopwatch against a live
chain otherwise, which is the whole reason `decide` is a pure function.

The cycle-ordering refusals at the end were in the canonicaliser's test file, which is where nobody
looks for them. They validate the config this module reads, so they live here.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from validator.cycle import CycleConfig, CycleError, Phase
from validator.scheduler import (
    Abandon,
    Complete,
    Run,
    Step,
    Wait,
    decide,
    next_wake_block,
    windows,
)

pytestmark = pytest.mark.determinism

SEASON = json.loads(pathlib.Path("config/season.example.json").read_text())

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


def at(offset: int, *done: Step, config: CycleConfig | None = None):
    return decide(cycle=config or cycle(), offset=offset, done=done)


# --------------------------------------------------------------------------
# The windows are the intervals between 7.3's boundaries
# --------------------------------------------------------------------------


def test_the_windows_are_ordered_and_never_overlap():
    """An overlap would let two steps run at one offset — for GENERATE and EXECUTE that means a pack
    hash committed after a bundle has opened, which is the one ordering with no recovery."""
    spans = windows(cycle())
    for earlier, later in zip(spans, spans[1:], strict=False):
        assert earlier.closes <= later.opens, (earlier.step, later.step)
    assert spans[0].opens == cycle().round_opens()
    assert spans[-1].closes == cycle().round_closes()


def test_the_only_gap_between_windows_is_section_21s_commitment_margin():
    """`[pack_commit, reveal)` belongs to no step: the pack hash is on chain and nothing else may
    happen. Asserted as the *only* gap, because a second one would be a block on which nothing may
    run and nothing has expired — a round that stalls without any step being abandoned."""
    config = cycle()
    spans = windows(config)
    gaps = [
        (earlier.closes, later.opens)
        for earlier, later in zip(spans, spans[1:], strict=False)
        if earlier.closes != later.opens
    ]
    assert gaps == [(config.pack_commit_offset, config.reveal_offset)]


def test_a_round_in_the_commitment_margin_waits_rather_than_stalling_or_abandoning():
    decision = at(-50, Step.COMMIT_SALT, Step.GENERATE)
    assert isinstance(decision, Wait), decision
    assert decision.next_step is Step.EXECUTE


def test_every_window_says_why_it_closes_where_it_does():
    """The reason is what an abandonment message carries. "COMMIT_SALT's window closed" tells an
    operator it was late; it does not tell them that running it late is unsafe."""
    for window in windows(cycle()):
        assert len(window.because) > 30, window


def test_a_window_excludes_its_closing_block():
    """Half-open at the close, so "before the randomness" means strictly before. At the randomness
    offset itself the salt window is shut."""
    salt = windows(cycle())[0]
    assert salt.contains(-301)
    assert not salt.contains(-300)


# --------------------------------------------------------------------------
# The happy path, offset by offset
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("offset", "done", "step"),
    [
        (-450, (), Step.COMMIT_SALT),
        (-301, (), Step.COMMIT_SALT),
        (-300, (Step.COMMIT_SALT,), Step.GENERATE),
        (-101, (Step.COMMIT_SALT,), Step.GENERATE),
        (0, (Step.COMMIT_SALT, Step.GENERATE), Step.EXECUTE),
        (4_200, (Step.COMMIT_SALT, Step.GENERATE, Step.EXECUTE), Step.SCORE),
        (
            6_900,
            (Step.COMMIT_SALT, Step.GENERATE, Step.EXECUTE, Step.SCORE),
            Step.SUBMIT_WEIGHTS,
        ),
    ],
)
def test_each_step_runs_at_the_opening_of_its_window(offset, done, step):
    decision = at(offset, *done)
    assert isinstance(decision, Run), decision
    assert decision.step is step


def test_a_run_carries_the_offset_it_must_finish_by():
    """Passed to the stage rather than kept here. The sandbox runner is the case that matters: it
    can bound its own work if it knows the room it has, and cannot if it has to guess."""
    decision = at(0, Step.COMMIT_SALT, Step.GENERATE)
    assert isinstance(decision, Run)
    assert decision.deadline_offset == 4_200


def test_a_round_with_every_step_done_is_complete():
    assert isinstance(at(7_000, *Step), Complete)


def test_before_the_first_window_the_round_waits_rather_than_abandoning():
    """A round that has not started yet is the normal state for most of a day."""
    decision = at(-600)
    assert isinstance(decision, Wait), decision
    assert decision.until_offset == -450
    assert decision.next_step is Step.COMMIT_SALT


def test_a_step_finished_early_waits_for_the_next_window():
    """A salt committed in one block does not entitle the validator to generate early: generation
    needs the randomness, which does not exist yet."""
    decision = at(-400, Step.COMMIT_SALT)
    assert isinstance(decision, Wait), decision
    assert decision.until_offset == -300
    assert decision.next_step is Step.GENERATE


# --------------------------------------------------------------------------
# 7.3's three orderings, as failures
# --------------------------------------------------------------------------


def test_a_salt_not_committed_before_the_randomness_abandons_the_round():
    """The first ordering. A validator that was down across the boundary must not commit now: the
    commitment would look valid to every peer while having been chosen with the randomness in hand.

    This is the case a retry loop would get wrong, and it would get it wrong silently.
    """
    decision = at(-250)
    assert isinstance(decision, Abandon), decision
    assert decision.step is Step.COMMIT_SALT
    assert "ground against it" in decision.reason


def test_generation_that_missed_its_window_abandons_rather_than_generating_late():
    """The second ordering. Generating after the pack-commit boundary means the hash goes on chain
    late, or the pack is stored before its hash is committed — 7.4 step 6 forbids both."""
    decision = at(-50, Step.COMMIT_SALT)
    assert isinstance(decision, Abandon), decision
    assert decision.step is Step.GENERATE


def test_a_pack_not_generated_and_committed_before_reveal_abandons_the_round():
    """The third ordering, and the one with the worst failure mode: bundles are open, so a pack
    committed now could have been chosen to suit what the validator has already read.

    GENERATE covers the commitment, so this is the same abandonment as the second ordering seen from
    a later offset — and that is the point of fusing them. There is no state in which the pack
    exists and its hash does not.
    """
    decision = at(10, Step.COMMIT_SALT)
    assert isinstance(decision, Abandon), decision
    assert decision.step is Step.GENERATE
    assert "chosen to suit a submission" in decision.reason


def test_execution_that_overran_its_window_abandons_rather_than_scoring_what_finished():
    """Scoring a partial execution would rank laboratories on unequal windows. The round is worth
    nothing; saying so is better than publishing a ranking that means nothing."""
    decision = at(4_300, Step.COMMIT_SALT, Step.GENERATE)
    assert isinstance(decision, Abandon), decision
    assert decision.step is Step.EXECUTE


def test_weights_not_submitted_before_the_next_epoch_abandons_rather_than_submitting_late():
    """A vector landing in the next round's window cannot be attributed. Better no weights for a day
    than yesterday's weights attributed to today."""
    done = (Step.COMMIT_SALT, Step.GENERATE, Step.EXECUTE, Step.SCORE)
    decision = at(DAY, *done)
    assert isinstance(decision, Abandon), decision
    assert decision.step is Step.SUBMIT_WEIGHTS


# --------------------------------------------------------------------------
# Restart and recovery
# --------------------------------------------------------------------------


def test_a_validator_joining_mid_day_does_not_join_that_day():
    """It falls out of the window rule rather than being a special case, which is the point: the
    first not-done step is COMMIT_SALT, its window has closed, so the round is over.

    A validator that joined mid-round would have no salt commitment, so its seed would be
    underivable — and every peer checking its pack commitment would find nothing to check it
    against.
    """
    for offset in (-299, 0, 4_000, 6_950):
        decision = at(offset)
        assert isinstance(decision, Abandon), (offset, decision)
        assert decision.step is Step.COMMIT_SALT


def test_a_restart_mid_round_resumes_at_the_step_it_had_reached():
    """The recovery path. Round state is what makes this possible: the phase alone would not — an
    `AWAITING_RANDOMNESS` phase says nothing about whether a salt commitment was published."""
    decision = at(-200, Step.COMMIT_SALT)
    assert isinstance(decision, Run), decision
    assert decision.step is Step.GENERATE


def test_the_reported_fault_is_the_step_that_was_missed_not_the_window_that_is_open():
    """Ordering-first evaluation. At offset 5,000 the SCORE window is open, but with nothing done
    the answer is the abandonment of COMMIT_SALT — the actual fault. Reporting SCORE would send an
    operator to look at the judge panels."""
    decision = at(5_000)
    assert isinstance(decision, Abandon)
    assert decision.step is Step.COMMIT_SALT


def test_a_later_step_recorded_without_its_predecessor_is_reported_as_the_predecessor_missing():
    """Corrupt or hand-edited progress. `{SCORE}` without `EXECUTE` is not a state the scheduler can
    produce, so what matters is that it fails safe rather than skipping ahead."""
    decision = at(5_000, Step.SCORE)
    assert isinstance(decision, Abandon)
    assert decision.step is Step.COMMIT_SALT


# --------------------------------------------------------------------------
# Round identity: from the chain, with the calendar attached
# --------------------------------------------------------------------------


def test_a_block_belongs_to_a_round_only_once_the_round_is_named():
    """The defect this replaced. `blocks_from_epoch(block)` computed `block - epoch_start(block)`,
    always in [0, blocks_per_day), so it never produced a negative offset — every pre-reveal phase
    was unreachable and `phase_of` reported EXECUTING at the epoch start.

    Both functions were individually correct. The assumption underneath was wrong: a block does not
    belong to one round.
    """
    config = cycle()
    assert config.offset_in(1, DAY - 450) == -450
    assert config.offset_in(0, DAY - 450) == DAY - 450
    assert config.phase_of(config.offset_in(1, DAY - 450)) is Phase.AWAITING_RANDOMNESS
    assert config.phase_of(config.offset_in(0, DAY - 450)) is Phase.SCORING


def test_the_round_id_is_derived_from_the_chain_rather_than_the_clock():
    """Two validators either side of midnight must label the same round identically, or they cannot
    recognise each other's commitments and cannot compare packs."""
    config = cycle(anchor_block=0, anchor_date="2026-01-01")
    assert config.round_id(0) == "2026-01-01"
    assert config.round_id(1) == "2026-01-02"
    assert config.round_id(214) == "2026-08-03"


def test_an_anchor_part_way_through_an_epoch_is_refused():
    """It would label some rounds with yesterday's date and some with today's, depending on the
    block — which is worse than being uniformly wrong."""
    with pytest.raises(CycleError, match="does not name the start of an epoch"):
        cycle(anchor_block=3_600)


def test_an_anchor_that_is_not_a_date_is_refused():
    with pytest.raises(CycleError, match="not an ISO date"):
        cycle(anchor_date="soon")


def test_an_anchor_disagreeing_with_the_calendar_is_caught_at_deployment():
    """Every validator sharing a wrong anchor still agrees with every other, so this is not a
    consensus failure and nothing would surface it. It would be found months later by someone
    reading a date."""
    from datetime import date

    config = cycle(anchor_block=0, anchor_date="2020-01-01")
    with pytest.raises(CycleError, match="days out"):
        config.assert_anchor_is_plausible(block=214 * DAY, now=date(2026, 8, 3))


def test_a_correct_anchor_passes_the_calendar_check():
    from datetime import date

    cycle().assert_anchor_is_plausible(block=214 * DAY + 10, now=date(2026, 8, 3))


def test_the_calendar_check_tolerates_a_day_of_drift():
    """A round is a day long and a validator checking near a boundary is legitimately one day out
    from the calendar. Refusing that would make startup depend on the minute."""
    from datetime import date

    cycle().assert_anchor_is_plausible(block=214 * DAY, now=date(2026, 8, 4))


# --------------------------------------------------------------------------
# Two rounds live at once
# --------------------------------------------------------------------------


def test_one_round_is_live_through_most_of_a_day():
    config = cycle()
    assert config.live_rounds(3 * DAY + 1_000) == (3,)


def test_two_rounds_are_live_in_the_overlap_and_the_geometry_says_why():
    """The cycle spans more blocks than a day, so the tail of one round overlaps the head of the
    next. Yesterday's weights are due at +6,900 while today's salt is due at +6,750.

    Code that assumed one current round would either skip a salt commit or submit yesterday's
    weights against today's state, and both are silent.
    """
    config = cycle()
    assert config.round_closes() - config.round_opens() > config.blocks_per_day
    assert config.live_rounds(3 * DAY + 6_800) == (3, 4)
    assert config.offset_in(3, 3 * DAY + 6_800) == 6_800
    assert config.offset_in(4, 3 * DAY + 6_800) == -400


def test_never_more_than_two_rounds_are_live():
    """Swept across a whole day rather than sampled, because the count changing to three is a
    driver that would have to handle a case it does not."""
    config = cycle()
    for block in range(5 * DAY, 6 * DAY):
        assert len(config.live_rounds(block)) <= 2, block


def test_the_previous_round_has_closed_by_the_next_epoch_start():
    """The boundary that keeps the count at two."""
    config = cycle()
    assert 3 not in config.live_rounds(4 * DAY)


def test_the_two_live_rounds_run_different_steps_in_the_overlap():
    """Why the overlap is safe, checked at the block where both rounds are actually runnable.

    Measured rather than assumed: at +6,800 the older round is still inside its SCORE window, so it
    is waiting. The genuine contention is at +6,950 — yesterday's weight submission against today's
    generation, a chain write against a run of model calls. They contend for nothing.
    """
    config = cycle()
    block = 3 * DAY + 6_950
    older, newer = config.live_rounds(block)
    scored = (Step.COMMIT_SALT, Step.GENERATE, Step.EXECUTE, Step.SCORE)
    older_decision = decide(cycle=config, offset=config.offset_in(older, block), done=scored)
    newer_decision = decide(
        cycle=config, offset=config.offset_in(newer, block), done=(Step.COMMIT_SALT,)
    )
    assert isinstance(older_decision, Run) and older_decision.step is Step.SUBMIT_WEIGHTS
    assert isinstance(newer_decision, Run) and newer_decision.step is Step.GENERATE


def test_the_salt_window_closes_exactly_where_the_weights_window_opens():
    """A coincidence of the example config, recorded because the driver depends on it.

    Round n+1's salt window closes exactly where round n's weights window opens, so the two chain
    writes are never both runnable at the same block. That falls out of `salt_commit_offset` and
    `weights_offset` each sitting 300 blocks from an epoch boundary — it is not enforced by
    `assert_ordering`, and a config that moved either would make the driver issue two extrinsics
    from one hotkey in one loop iteration. Asserted here so that change is visible rather than
    discovered by a nonce collision on a live chain.
    """
    config = cycle()
    salt, weights = windows(config)[0], windows(config)[-1]
    assert config.blocks_per_day + salt.closes == weights.opens
    # The same fact in absolute blocks, which is the form the driver sees.
    assert config.epoch_start_of(4) + salt.closes == config.epoch_start_of(3) + weights.opens


# --------------------------------------------------------------------------
# When to wake up
# --------------------------------------------------------------------------


def test_a_runnable_step_means_wake_now():
    config = cycle()
    block = 3 * DAY - 450
    assert next_wake_block(cycle=config, block=block, progress={}) == block


def test_waiting_wakes_at_the_block_the_next_window_opens():
    """A fixed poll interval would either wake sixty times between boundaries or miss one. The
    boundary is a block height, so it can be computed."""
    config = cycle()
    block = 3 * DAY - 200
    progress = {3: (Step.COMMIT_SALT, Step.GENERATE)}
    # The pack is committed and the round is in section 21's margin; the next thing due is the
    # reveal at the epoch start, 200 blocks away.
    assert next_wake_block(cycle=config, block=block, progress=progress) == 3 * DAY


def test_a_round_that_is_over_produces_no_wake_block():
    """None means the next thing to happen is the next epoch. The scheduler does not return that
    itself: a round that does not exist yet has no state to reason about."""
    config = cycle()
    block = 3 * DAY + 1_000
    assert next_wake_block(cycle=config, block=block, progress={3: tuple(Step)}) is None


def test_the_wake_block_is_the_earliest_across_both_live_rounds():
    """In the overlap, the older round is runnable now — so waiting for the newer round's salt
    window would delay a weight submission past its deadline."""
    config = cycle()
    block = 3 * DAY + 6_800
    done = (Step.COMMIT_SALT, Step.GENERATE, Step.EXECUTE, Step.SCORE)
    assert next_wake_block(cycle=config, block=block, progress={3: done}) == block


def test_an_abandoned_round_does_not_hold_the_loop_awake():
    """An abandoned round has nothing left to do, so it must not produce a wake block — a loop that
    kept waking for it would spin for the rest of the day."""
    config = cycle()
    block = 3 * DAY + 1_000
    assert next_wake_block(cycle=config, block=block, progress={3: ()}) is None


# --------------------------------------------------------------------------
# 21: the cycle ordering, refused at load rather than discovered on the day it matters
# --------------------------------------------------------------------------


def test_the_example_season_cycle_validates():
    CycleConfig.from_season(SEASON).assert_ordering()


def test_a_salt_committed_after_the_randomness_is_refused():
    """The ordering 7.3 depends on: a validator that chose its salt with the randomness in hand
    could grind it until the seed produced a pack it liked."""
    with pytest.raises(CycleError, match="grind the salt"):
        cycle(salt_commit_offset=-200)


def test_a_pack_committed_after_reveal_is_refused():
    """The pack hash must be on chain before any bundle opens, or a validator could read a
    submission and regenerate its challenges to suit it."""
    with pytest.raises(CycleError, match="before any bundle opens"):
        cycle(pack_commit_offset=100)


def test_generation_before_the_randomness_is_refused():
    with pytest.raises(CycleError, match="seed needs the randomness"):
        cycle(randomness_offset=-50)


def test_submissions_closing_after_the_salt_commit_is_refused():
    """A miner could otherwise submit after seeing which validators had committed."""
    with pytest.raises(CycleError, match="Submissions must close first"):
        cycle(submission_close_offset=-400)


def test_weights_before_execution_closes_is_refused():
    with pytest.raises(CycleError, match="cannot be computed before execution"):
        cycle(weights_offset=3_000)


def test_a_cycle_that_overruns_its_day_is_refused():
    """It would submit weights for one round during the next one."""
    with pytest.raises(CycleError, match="overruns"):
        cycle(weights_offset=7_300)


@pytest.mark.parametrize(
    ("blocks", "expected"),
    [
        (-700, Phase.BEFORE_SUBMISSION_CLOSE),
        (-500, Phase.AWAITING_SALT_COMMIT),
        (-400, Phase.AWAITING_RANDOMNESS),
        (-200, Phase.GENERATING),
        (-50, Phase.AWAITING_REVEAL),
        (100, Phase.EXECUTING),
        (5_000, Phase.SCORING),
        (6_900, Phase.AWAITING_WEIGHTS),
        (7_000, Phase.DONE),
    ],
)
def test_each_block_offset_maps_to_its_phase(blocks, expected):
    assert cycle().phase_of(blocks) is expected


def test_the_epoch_start_is_derived_from_the_chain_rather_than_the_clock():
    """A day boundary from wall clock would put two validators in different days either side of
    midnight, generating packs for different dates and unable to compare."""
    config = cycle()
    assert config.epoch_start(7_250) == 7_200
    assert config.epoch_start(7_199) == 0
    assert config.epoch_start(14_400) == 14_400
