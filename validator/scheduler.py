"""What to do next, decided from a block height and a record of what is already done.

This is the layer that was missing. Every stage of a round was built and reachable from the
composition root, and nothing decided *when* to run them — so the validator could validate its
configuration and report a phase, and could not run a day.

## Why the decision is separated from the doing

The scheduler is pure: it takes a block height and a set of completed steps and returns what should
happen. It touches no chain, no clock, no store and no container. The driver in `__main__` does all
of that.

The split is not tidiness. Every rule worth having here is a rule about *ordering under failure* — a
validator that restarted mid-round, or was down across a boundary, or is looking at a chain that
stalled for an hour. Those are exactly the situations that are hard to reproduce against a live
chain and trivial to enumerate against a function. With the decision inlined into the loop, the only
way to test "we were down through the salt commit and must not commit late" is to run a validator
and stop it at the right moment; here it is one call.

## The rule that carries the security argument

**A step runs inside its window or not at all.** If a step's window has closed and the step is not
done, the round is abandoned — loudly, by name, with the reason.

Never caught up, because catching up is precisely what breaks the three orderings. A salt
committed after the randomness is drawn is worse than no salt commitment: the commitment exists, it
looks valid to every peer, and the validator could have ground it against the randomness it had
already seen. The same holds for a pack hash committed after reveal — it is a commitment to a pack
the validator chose knowing what it was being tested against.

So there is no retry, no grace period, and no "best effort". A round that misses a boundary is over.

## Joining mid-day is not joining

A validator that starts with no record of today and finds the salt window already closed does not
join today. It waits for the next epoch. This falls out of the rule above rather than being a
special case, which is the point: the first not-done step is `COMMIT_SALT`, its window has closed,
so the round is abandoned.

## Two rounds at once

`CycleConfig.live_rounds` returns up to two, and the driver decides for each independently. That is
why `decide` takes a round's offset rather than a block: at one block, yesterday's round is at
+6,950 and today's is at −250, and one function that tried to serve both from a block height would
have to pick.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum

from validator.cycle import CycleConfig, Phase

__all__ = [
    "Abandon",
    "Complete",
    "Decision",
    "Run",
    "Step",
    "Wait",
    "decide",
    "next_wake_block",
    "windows",
]

_log = logging.getLogger(__name__)


class Step(Enum):
    """The five things a round does, in the order it does them.

    Coarser than `Phase`. A phase is where the clock is; a step is a unit of work that either
    happened or did not, and the difference matters for recovery — `AWAITING_RANDOMNESS` tells a
    restarting validator nothing about whether it already published a salt commitment.

    The values are the ordering. `Step.SCORE > Step.EXECUTE` is meaningful.

    `GENERATE` covers planning, generation, filtering, committing the pack hash and storing the
    pack, which the schedule lists as two boundaries. They are one step because the pipeline
    requires
    the hash on chain *before* the pack reaches Redis — so there is no legal place to persist a
    generated-but-uncommitted pack, and a step recorded as done with its only copy in memory would
    be destroyed by the restart the record exists to survive.
    """

    COMMIT_SALT = 1
    GENERATE = 2
    EXECUTE = 3
    SCORE = 4
    SUBMIT_WEIGHTS = 5


@dataclass(frozen=True, slots=True)
class Window:
    """The half-open block-offset range in which a step may run.

    Half-open at the close, so a step whose window closes at `randomness_offset` may run at
    `randomness_offset - 1` and not at `randomness_offset`. The boundary block belongs to the next
    window, which is what makes "before the randomness" mean strictly before.
    """

    step: Step
    opens: int
    closes: int

    #: Why this window closes where it does. Carried on the object rather than kept in a comment,
    #: because it is what an abandonment message has to say — an operator reading "COMMIT_SALT's
    #: window closed" needs to know that committing late is unsafe, not merely late.
    because: str

    def contains(self, offset: int) -> bool:
        return self.opens <= offset < self.closes


def windows(cycle: CycleConfig) -> tuple[Window, ...]:
    """Each step's window, derived from the cycle config.

    Derived rather than configured separately: a second set of offsets would be a second thing to
    keep consistent with 7.3's ordering, and `assert_ordering` already refuses a config that breaks
    it. The windows are the intervals *between* the boundaries it validates.

    They do not tile the round: `[pack_commit_offset, reveal_offset)` belongs to no step. That gap
    is the margin between the commitment and the reveal — the hundred blocks in which the
    pack hash is on chain and nothing else may happen. `decide` returns `Wait` there, which is
    correct: a round in the margin has nothing to do and nothing has expired.
    """
    return (
        Window(
            Step.COMMIT_SALT,
            cycle.salt_commit_offset,
            cycle.randomness_offset,
            "the salt must be committed before the randomness exists, or it could have been ground "
            "against it",
        ),
        Window(
            # Opens one block *after* the randomness, not on it. The seed mixes the hash of the
            # randomness block, and a block's hash is not final while that block is the head — so a
            # step scheduled at `randomness_offset` reads the hash of the block it is standing on
            # and is refused. Whether the driver polled on that block or the next was a race, and
            # losing it abandoned the whole round.
            Step.GENERATE,
            cycle.randomness_offset + 1,
            cycle.pack_commit_offset,
            "generation needs the randomness settled, and the pack hash must be on chain before "
            "any bundle is opened — a pack committed later could have been chosen to suit a "
            "submission",
        ),
        Window(
            Step.EXECUTE,
            cycle.reveal_offset,
            cycle.execution_close_offset,
            "every laboratory gets the same execution window; running past it would give one more "
            "time than the others",
        ),
        Window(
            Step.SCORE,
            cycle.execution_close_offset,
            cycle.weights_offset,
            "scoring must finish before the weight vector is due",
        ),
        Window(
            Step.SUBMIT_WEIGHTS,
            cycle.weights_offset,
            cycle.round_closes(),
            "a vector submitted after the next epoch begins lands in the next round's window, "
            "where the chain cannot tell which round it was for",
        ),
    )


# --------------------------------------------------------------------------
# Decisions
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Run:
    """Run this step now. Its window is open and every earlier step is done."""

    step: Step
    #: The block by which it must have finished. Passed on to the step so a stage that can bound its
    #: own work — the sandbox runner, above all — knows how much room it has rather than guessing.
    deadline_offset: int


@dataclass(frozen=True, slots=True)
class Wait:
    """Nothing to do for this round yet."""

    #: The offset at which the next step's window opens.
    until_offset: int
    next_step: Step


@dataclass(frozen=True, slots=True)
class Abandon:
    """This round cannot be completed. It is over; the next one is unaffected."""

    step: Step
    reason: str


@dataclass(frozen=True, slots=True)
class Complete:
    """Every step is done."""


Decision = Run | Wait | Abandon | Complete


def decide(
    *,
    cycle: CycleConfig,
    offset: int,
    done: Iterable[Step],
) -> Decision:
    """What this round should do at this offset, given what it has already done.

    Steps are considered in order and the first one not done is the answer. That ordering is load
    bearing: a step whose predecessor never ran is never reached, so the only way to arrive at
    `GENERATE` is with `COMMIT_SALT` done — and if the salt was never committed, what is reported is
    the abandonment of `COMMIT_SALT`, which is the actual fault, rather than a `GENERATE` window
    that happens to be open.
    """
    finished = set(done)
    for window in windows(cycle):
        if window.step in finished:
            continue
        if offset < window.opens:
            return Wait(until_offset=window.opens, next_step=window.step)
        if offset >= window.closes:
            return Abandon(
                step=window.step,
                reason=(
                    f"{window.step.name} did not run: its window closed at offset "
                    f"{window.closes} and the chain is at {offset}. It is not run late, because "
                    f"{window.because}."
                ),
            )
        return Run(step=window.step, deadline_offset=window.closes)
    return Complete()


def next_wake_block(
    *,
    cycle: CycleConfig,
    block: int,
    progress: Mapping[int, Iterable[Step]],
) -> int | None:
    """The earliest block at which any live round's decision could change, or None if none can.

    The driver uses this instead of a fixed poll interval, so it wakes when a boundary arrives
    rather than sixty times in between. It is a floor and not a promise: block production is not
    uniform, so the driver polls again at this height and recomputes rather than assuming the
    boundary has been reached.

    None means every live round is either complete or abandoned, and the next thing to happen is the
    next epoch — which the caller derives, because a scheduler that returned "the next epoch start"
    would be claiming knowledge of a round that does not exist yet.
    """
    candidates: list[int] = []
    for index in cycle.live_rounds(block):
        offset = cycle.offset_in(index, block)
        decision = decide(cycle=cycle, offset=offset, done=progress.get(index, ()))
        if isinstance(decision, Run):
            # Something is runnable now. There is nothing to wait for.
            return block
        if isinstance(decision, Wait):
            candidates.append(
                cycle.epoch_start_of(index) + decision.until_offset - cycle.reveal_offset
            )
    if not candidates:
        return None
    return min(candidates)


def phase_for(cycle: CycleConfig, offset: int) -> Phase:
    """The phase a round is in at an offset.

    A thin pass-through, and it exists so the driver never has to hold both an offset and a block:
    reporting a phase against the wrong round's offset is how a status page ends up claiming a round
    is executing while it is being scored.
    """
    return cycle.phase_of(offset)
