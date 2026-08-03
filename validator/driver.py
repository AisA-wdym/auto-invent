"""The loop: poll the chain, ask the scheduler, run what it says, record what happened.

`validator/scheduler.py` decides; this executes. The split is why the decision layer has no chain
and no clock, and it is also why this file has almost no logic — everything here is either an I/O
call or the persistence of an outcome.

## What the driver owns, and what it must not

It owns three things: reading the block height, loading and saving each round's progress, and
calling the step implementations in the order the scheduler gives. It owns no *rules*. In particular
it never decides that a step is late enough to skip or early enough to try — a driver that could
reorder steps would be a second place where 7.3's orderings live, and the second place is always the
one that is wrong.

## Progress is durable, and it is what recovery reads

Each completed step is appended to `RoundState.steps_done` and written before the next step is
attempted. Written *after* the step, never before: a step recorded before it ran would make a crash
mid-step look like a completed step, and for `COMMIT_SALT` that means a round whose seed cannot be
derived from any commitment on chain.

The write is also why a restart is cheap. The scheduler's `decide` needs only the step list, so a
validator that comes back up mid-round resumes at the step it had reached, and one that comes back
up too late abandons the round by name.

## Steps are attempted once per round

If a step raises, the round is abandoned. There is no retry, and that is a deliberate cost: a
transient RPC failure during `COMMIT_SALT` loses the day.

The alternative is worse. A retry needs a window to retry inside, and every window here is a
security boundary — retrying `COMMIT_SALT` until it succeeds is exactly how a salt ends up committed
after the randomness. A retry loop *within* a step's own window would be safe, but it belongs to the
step, which knows what it is calling and what a safe repeat looks like; the driver does not.

## Two rounds, in order

`live_rounds` returns oldest first and the driver walks them in that order. The order matters in the
overlap: yesterday's weight submission is due at the same time as today's salt commitment, and the
weight submission is the one with a hard chain deadline behind it.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Protocol

from chain.client import ChainClient, ChainError
from validator.cycle import CycleConfig
from validator.roundstate import RoundState, RoundStore
from validator.scheduler import Abandon, Complete, Run, Step, Wait, decide, next_wake_block

__all__ = ["IMPLEMENTATION", "Driver", "Outcome", "Steps", "describe"]

_log = logging.getLogger(__name__)

#: Which method on `Steps` runs each scheduler step. A module constant rather than a literal inside
#: `__post_init__` so the pairing can be asserted directly: a step added to the scheduler without an
#: implementation here would otherwise be attempted and fail with an AttributeError mid-round, on a
#: live chain, once.
IMPLEMENTATION: dict[Step, str] = {
    Step.COMMIT_SALT: "commit_salt",
    Step.GENERATE: "generate",
    Step.EXECUTE: "execute",
    Step.SCORE: "score",
    Step.SUBMIT_WEIGHTS: "submit_weights",
}

#: Steps the driver calls without a deadline, because their window's close is the deadline and the
#: work is a single chain write. Named rather than tested inline so adding a step cannot silently
#: land in the wrong branch.
_WITHOUT_DEADLINE = frozenset({Step.COMMIT_SALT, Step.SUBMIT_WEIGHTS})


class Steps(Protocol):
    """The six things a round does. Implemented by `Validator`, faked in tests.

    Each takes the round id and returns the state it produced, so the driver persists one object per
    step rather than reaching into the validator for fragments. A step that returns the state it
    wrote is also a step that can be checked: the test double returns a state and the driver's
    handling of it is exercised without a chain.
    """

    def commit_salt(self, state: RoundState, *, block: int) -> RoundState: ...

    def generate(self, state: RoundState, *, block: int, deadline_block: int) -> RoundState: ...

    def execute(self, state: RoundState, *, block: int, deadline_block: int) -> RoundState: ...

    def score(self, state: RoundState, *, block: int, deadline_block: int) -> RoundState: ...

    def submit_weights(self, state: RoundState, *, block: int) -> RoundState: ...


@dataclass(frozen=True, slots=True)
class Outcome:
    """What the driver did about one round on one tick. Returned so a caller can assert on it.

    `--once` prints these, and the driver tests read them. Without a return value the only evidence
    of a tick would be log lines, and asserting on log text is how a test starts failing because
    somebody improved a message.
    """

    round_id: str
    epoch_index: int
    offset: int
    #: `"ran"`, `"waiting"`, `"abandoned"`, `"complete"`, or `"failed"`.
    kind: str
    step: Step | None = None
    detail: str = ""


@dataclass
class Driver:
    """Polls the chain and advances every live round by at most one step per tick."""

    chain: ChainClient
    cycle: CycleConfig
    store: RoundStore
    steps: Steps
    #: How long to sleep between polls when there is nothing to do. Blocks are ~12s, so this is
    #: about two blocks: short enough not to miss a boundary by long, long enough not to hammer an
    #: RPC endpoint for a day at a time.
    poll_seconds: float = 24.0
    #: Injected so a test can drive a whole day without waiting for one. Not a default of `None`
    #: resolved later — a sleep that silently became a no-op would make a test of the loop's timing
    #: pass while the loop spun.
    sleep: Callable[[float], None] = time.sleep

    def __post_init__(self) -> None:
        missing = set(Step) - set(IMPLEMENTATION)
        if missing:
            raise ValueError(
                f"no implementation is mapped for {sorted(step.name for step in missing)}. Every "
                "scheduler step needs a method on Steps, or a round would abandon itself at that "
                "step with an attribute error."
            )
        for step, name in IMPLEMENTATION.items():
            if not callable(getattr(self.steps, name, None)):
                # Checked at construction rather than at the moment the step is due. The moment the
                # step is due is once a day, on a live chain, inside a window that does not reopen.
                raise ValueError(
                    f"the steps object has no callable {name!r} for {step.name}"
                )

    # ------------------------------------------------------------------
    # One tick
    # ------------------------------------------------------------------

    def tick(self, block: int) -> list[Outcome]:
        """Advance every live round by at most one step. Returns what happened to each.

        One step per round per tick, rather than looping until a round blocks. The block height is
        read once at the top, and a step that takes twenty minutes leaves it stale — so continuing
        to act on it would mean deciding a later step against an earlier chain. Re-reading the
        height is the next tick's job.
        """
        outcomes: list[Outcome] = []
        for index in self.cycle.live_rounds(block):
            outcomes.append(self._advance(index, block))
        return outcomes

    def preview(self, block: int) -> list[Outcome]:
        """What a tick *would* do, without doing it or writing anything.

        `--once` uses this when the loop cannot run, and an operator uses it to see where in the
        cycle a deployment sits. Read-only by construction: it never calls a step and never touches
        the store beyond the read `decide` needs, so running it against production is safe.
        """
        previews: list[Outcome] = []
        for index in self.cycle.live_rounds(block):
            round_id = self.cycle.round_id(index)
            offset = self.cycle.offset_in(index, block)
            state = self._load(round_id, block)
            if state.abandoned:
                previews.append(
                    Outcome(round_id, index, offset, "abandoned", detail=state.abandoned)
                )
                continue
            decision = decide(cycle=self.cycle, offset=offset, done=self._done(state))
            if isinstance(decision, Run):
                previews.append(Outcome(round_id, index, offset, "would run", step=decision.step))
            elif isinstance(decision, Wait):
                previews.append(
                    Outcome(
                        round_id, index, offset, "waiting", step=decision.next_step,
                        detail=f"until offset {decision.until_offset}",
                    )
                )
            elif isinstance(decision, Abandon):
                previews.append(
                    Outcome(
                        round_id, index, offset, "would abandon", step=decision.step,
                        detail=decision.reason,
                    )
                )
            else:
                previews.append(Outcome(round_id, index, offset, "complete"))
        return previews

    def _advance(self, epoch_index: int, block: int) -> Outcome:
        round_id = self.cycle.round_id(epoch_index)
        offset = self.cycle.offset_in(epoch_index, block)
        state = self._load(round_id, block)

        if state.abandoned:
            # Already given up on. Reported so `--once` can say so, and not re-decided: the
            # scheduler would return the same abandonment every tick for the rest of the day.
            return Outcome(round_id, epoch_index, offset, "abandoned", detail=state.abandoned)

        done = self._done(state)
        decision = decide(cycle=self.cycle, offset=offset, done=done)

        if isinstance(decision, Wait):
            return Outcome(
                round_id,
                epoch_index,
                offset,
                "waiting",
                step=decision.next_step,
                detail=f"until offset {decision.until_offset}",
            )

        if isinstance(decision, Complete):
            return Outcome(round_id, epoch_index, offset, "complete")

        if isinstance(decision, Abandon):
            self._abandon(state, block, decision.reason)
            return Outcome(
                round_id, epoch_index, offset, "abandoned", step=decision.step,
                detail=decision.reason,
            )

        return self._run(state, decision, epoch_index=epoch_index, block=block, offset=offset)

    def _run(
        self, state: RoundState, decision: Run, *, epoch_index: int, block: int, offset: int
    ) -> Outcome:
        deadline_block = (
            self.cycle.epoch_start_of(epoch_index)
            + decision.deadline_offset
            - self.cycle.reveal_offset
        )
        method = getattr(self.steps, IMPLEMENTATION[decision.step])
        _log.info(
            "round %s: running %s at block %d (offset %d), deadline block %d",
            state.date,
            decision.step.name,
            block,
            offset,
            deadline_block,
        )
        try:
            if decision.step in _WITHOUT_DEADLINE:
                produced = method(state, block=block)
            else:
                produced = method(state, block=block, deadline_block=deadline_block)
        except Exception as error:  # noqa: BLE001 - any failure ends the round; see the module note
            reason = (
                f"{decision.step.name} failed: {type(error).__name__}: {error}. The round is "
                "abandoned rather than retried, because every window here is a security boundary "
                "and a retry needs one to retry inside."
            )
            _log.exception("round %s: %s", state.date, reason)
            self._abandon(state, block, reason)
            return Outcome(
                state.date, epoch_index, offset, "failed", step=decision.step, detail=reason
            )

        # Recorded after the step, never before. A step marked done before it ran would make a crash
        # mid-step indistinguishable from a completed one — and for COMMIT_SALT that is a round
        # whose seed cannot be derived from anything on chain.
        self._save(
            replace(
                produced,
                steps_done=(*produced.steps_done, decision.step.name),
                block=block,
                updated_at_block=block,
                phase=self.cycle.phase_of(offset).name,
            )
        )
        return Outcome(state.date, epoch_index, offset, "ran", step=decision.step)

    # ------------------------------------------------------------------
    # The loop
    # ------------------------------------------------------------------

    def run(self, *, max_ticks: int | None = None) -> list[Outcome]:
        """Poll and advance until `max_ticks` ticks have run, or forever.

        `max_ticks` is how a test drives this and how `--once` uses it, rather than a separate
        single-shot path. Two code paths through a loop means one of them is the one nobody runs.

        A `ChainError` ends the loop rather than being swallowed. A validator that cannot read the
        chain cannot know which round it is in, and continuing on the last height it saw is how a
        step runs against a boundary that passed an hour ago.
        """
        ticks = 0
        history: list[Outcome] = []
        while max_ticks is None or ticks < max_ticks:
            block = self.chain.current_block()
            outcomes = self.tick(block)
            history.extend(outcomes)
            ticks += 1
            if max_ticks is not None and ticks >= max_ticks:
                break
            self._sleep_until_something_happens(block, outcomes)
        return history

    def _sleep_until_something_happens(self, block: int, outcomes: Sequence[Outcome]) -> None:
        """Sleep for a poll interval, or long enough to reach the next boundary.

        The wake block is a floor rather than a promise: block production is not uniform, so the
        sleep is capped at what the remaining blocks would take at nominal speed and the loop
        re-reads the height rather than assuming the boundary arrived.
        """
        if any(outcome.kind == "ran" for outcome in outcomes):
            # Something ran, so the next step may already be due. Re-read immediately.
            return
        wake = next_wake_block(
            cycle=self.cycle,
            block=block,
            progress={
                outcome.epoch_index: self._done(self._load(outcome.round_id, block))
                for outcome in outcomes
                if outcome.kind == "waiting"
            },
        )
        if wake is None or wake <= block:
            self.sleep(self.poll_seconds)
            return
        # Twelve seconds a block, floored at one poll interval so a one-block wait does not become a
        # busy loop, and capped so a long wait still re-checks that the chain is producing at all.
        seconds = min(max((wake - block) * 12.0, self.poll_seconds), 600.0)
        _log.debug("nothing due until block %d; sleeping %.0fs", wake, seconds)
        self.sleep(seconds)

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def _load(self, round_id: str, block: int) -> RoundState:
        """This round's state, or a fresh one.

        A round with no state is the normal case once a day. It is created here rather than by the
        first step so that an abandonment before any step has run still has somewhere to be
        written — otherwise the most important thing the driver can record would be the one thing it
        could not.
        """
        existing = self.store.read(round_id)
        if existing is not None:
            return existing
        return RoundState(
            date=round_id,
            validator_hotkey=self.chain.hotkey(),
            phase="BEFORE_SUBMISSION_CLOSE",
            block=block,
            updated_at_block=block,
        )

    def _done(self, state: RoundState) -> tuple[Step, ...]:
        """The recorded steps, as `Step` values.

        A name this version does not know raises rather than being dropped. A dropped step would be
        re-run: an unrecognised `COMMIT_SALT` written by a newer release would become a second salt
        commitment for the same round, which is precisely what a precommitment must not allow.
        """
        parsed: list[Step] = []
        for name in state.steps_done:
            try:
                parsed.append(Step[name])
            except KeyError as error:
                raise ChainError(
                    f"round {state.date} records a completed step {name!r} that this release does "
                    "not know. It is not ignored: an unrecognised step would be run again, and "
                    "running COMMIT_SALT twice publishes two commitments for one round. Upgrade "
                    "the validator or clear the round."
                ) from error
        return tuple(parsed)

    def _abandon(self, state: RoundState, block: int, reason: str) -> None:
        _log.error("round %s abandoned at block %d: %s", state.date, block, reason)
        self._save(replace(state, abandoned=reason, block=block, updated_at_block=block))

    def _save(self, state: RoundState) -> None:
        self.store.write(state)


def describe(outcomes: Sequence[Outcome]) -> str:
    """One line per outcome, for `--once` and for an operator reading a log tail."""
    if not outcomes:
        return "no round is live at this block"
    lines: list[str] = []
    for outcome in outcomes:
        step = outcome.step.name if outcome.step else "-"
        detail = f" — {outcome.detail}" if outcome.detail else ""
        lines.append(
            f"  {outcome.round_id}  offset {outcome.offset:+6d}  {outcome.kind:<9} {step}{detail}"
        )
    return "\n".join(lines)
