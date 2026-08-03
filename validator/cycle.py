"""The daily cycle: architecture.md 21. Seven timed chain interactions, in one place.

    submission_close  T-600   miners can no longer commit
    salt_commit       T-450   the validator commits its salt — before the randomness exists
    randomness        T-300   the post-deadline block hash is drawn
    pack_commit       T-100   the pack hash goes on chain, before the pack is stored
    reveal            T+0     sealed bundles open; execution begins
    execution_close   T+4200  containers are terminated
    weights           T+6900  the vector is submitted

Offsets are block numbers relative to the day's epoch start, from `cycle` in the season config.

## Why the ordering is the security property

Three orderings carry the whole of 7.3's guarantee, and each is enforced by a phase boundary rather
than by a comment:

**salt_commit < randomness.** A validator that chose its salt after seeing the block hash could
grind
the salt until the derived seed produced a pack it liked. Committing first removes the choice.

**randomness < pack_commit.** The seed needs the randomness, so generation cannot begin before it.

**pack_commit < reveal.** The pack hash is on chain before any bundle is opened, so a validator
cannot see a submission and then regenerate its challenges to suit it.

`Phase.of` derives the current phase from a block height, and `assert_ordering` refuses a config
whose offsets break any of the three. A misordered config is a validator that looks correct and has
no guarantee at all, so it is refused at load rather than discovered on a day it matters.

## Blocks, not wall clock

Every boundary is a block height. Wall clock would drift between validators and would make "before
the randomness" a question about NTP; a block height is the same fact for everyone reading the same
chain.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

__all__ = ["CycleConfig", "CycleError", "Phase"]

_log = logging.getLogger(__name__)


class CycleError(ValueError):
    """A cycle configuration whose ordering does not hold."""


class Phase(Enum):
    """Where in the day we are. Ordered, so a comparison means "before" or "after"."""

    BEFORE_SUBMISSION_CLOSE = 0
    AWAITING_SALT_COMMIT = 1
    AWAITING_RANDOMNESS = 2
    GENERATING = 3
    AWAITING_REVEAL = 4
    EXECUTING = 5
    SCORING = 6
    AWAITING_WEIGHTS = 7
    DONE = 8


@dataclass(frozen=True, slots=True)
class CycleConfig:
    """The day's block offsets, validated on construction."""

    blocks_per_day: int
    submission_close_offset: int
    salt_commit_offset: int
    randomness_offset: int
    pack_commit_offset: int
    reveal_offset: int
    execution_close_offset: int
    weights_offset: int

    def __post_init__(self) -> None:
        self.assert_ordering()

    @classmethod
    def from_season(cls, season: Mapping[str, object]) -> CycleConfig:
        cycle = season["cycle"]  # type: ignore[index]
        return cls(
            blocks_per_day=int(cycle["blocks_per_day"]),  # type: ignore[index]
            submission_close_offset=int(cycle["submission_close_offset"]),  # type: ignore[index]
            salt_commit_offset=int(cycle["salt_commit_offset"]),  # type: ignore[index]
            randomness_offset=int(cycle["randomness_offset"]),  # type: ignore[index]
            pack_commit_offset=int(cycle["pack_commit_offset"]),  # type: ignore[index]
            reveal_offset=int(cycle["reveal_offset"]),  # type: ignore[index]
            execution_close_offset=int(cycle["execution_close_offset"]),  # type: ignore[index]
            weights_offset=int(cycle["weights_offset"]),  # type: ignore[index]
        )

    def assert_ordering(self) -> None:
        """Refuse a config that breaks any of 7.3's three orderings.

        Each check names what the ordering buys, because a config is edited by an operator who did
        not write this and a message that only said "invalid offsets" would not tell them what they
        had broken.
        """
        if self.salt_commit_offset >= self.randomness_offset:
            raise CycleError(
                f"salt_commit_offset ({self.salt_commit_offset}) is not before randomness_offset "
                f"({self.randomness_offset}). A validator that commits its salt after seeing the "
                "randomness can grind the salt until the derived seed produces a pack it likes, "
                "which is the whole of what committing first prevents."
            )
        if self.randomness_offset >= self.pack_commit_offset:
            raise CycleError(
                f"randomness_offset ({self.randomness_offset}) is not before pack_commit_offset "
                f"({self.pack_commit_offset}). The seed needs the randomness, so generation cannot "
                "begin before it is drawn."
            )
        if self.pack_commit_offset >= self.reveal_offset:
            raise CycleError(
                f"pack_commit_offset ({self.pack_commit_offset}) is not before reveal_offset "
                f"({self.reveal_offset}). The pack hash must be on chain before any bundle opens, "
                "or a validator could read a submission and regenerate its challenges to suit it."
            )
        if self.submission_close_offset >= self.salt_commit_offset:
            raise CycleError(
                f"submission_close_offset ({self.submission_close_offset}) is not before "
                f"salt_commit_offset ({self.salt_commit_offset}). Submissions must close first, or "
                "a miner could submit after seeing which validators had committed."
            )
        if not self.reveal_offset < self.execution_close_offset < self.weights_offset:
            raise CycleError(
                f"reveal ({self.reveal_offset}), execution_close "
                f"({self.execution_close_offset}) and weights ({self.weights_offset}) must be in "
                "that order: weights cannot be computed before execution has closed."
            )
        if self.weights_offset >= self.blocks_per_day + self.reveal_offset:
            raise CycleError(
                f"weights_offset ({self.weights_offset}) does not leave the day's "
                f"{self.blocks_per_day} blocks before the next epoch begins. A cycle that overruns "
                "its day would submit weights for one round during the next one."
            )

    def phase_of(self, blocks_from_epoch: int) -> Phase:
        """Which phase a block height falls in, relative to the day's epoch start.

        `blocks_from_epoch` is measured from the reveal point, so it is negative before reveal —
        matching the negative offsets in the config, which is what makes the config readable as a
        timeline around the moment execution starts.
        """
        if blocks_from_epoch < self.submission_close_offset:
            return Phase.BEFORE_SUBMISSION_CLOSE
        if blocks_from_epoch < self.salt_commit_offset:
            return Phase.AWAITING_SALT_COMMIT
        if blocks_from_epoch < self.randomness_offset:
            return Phase.AWAITING_RANDOMNESS
        if blocks_from_epoch < self.pack_commit_offset:
            return Phase.GENERATING
        if blocks_from_epoch < self.reveal_offset:
            return Phase.AWAITING_REVEAL
        if blocks_from_epoch < self.execution_close_offset:
            return Phase.EXECUTING
        if blocks_from_epoch < self.weights_offset:
            return Phase.SCORING
        if blocks_from_epoch == self.weights_offset:
            return Phase.AWAITING_WEIGHTS
        return Phase.DONE

    def epoch_start(self, block: int) -> int:
        """The block at which the current day's cycle began.

        Floor-divided by `blocks_per_day`, so every validator on the same chain agrees which day it
        is without coordinating. A day boundary derived from wall clock would put two validators in
        different days for a few minutes either side of midnight, and they would generate packs for
        different dates and be unable to compare.
        """
        return (block // self.blocks_per_day) * self.blocks_per_day

    def blocks_from_epoch(self, block: int) -> int:
        return block - self.epoch_start(block) + self.reveal_offset
