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

## Two rounds are live at once, and that is not a bug

The cycle spans from `salt_commit_offset` to the next epoch start — 7,650 blocks with the example
config — which is more than the 7,200 blocks in a day. So the tail of one round overlaps the head of
the next, by design: yesterday's weights are still due at +6,900 when today's salt must be committed
at 7,200 − 450 = +6,750.

`live_rounds` returns every round whose window contains a block, and the geometry guarantees at most
two. The overlap is safe because the two rounds want different things in it — one wants a weight
submission, the other wants a commitment — but code that assumed a single current round would either
skip a salt commit or submit yesterday's weights against today's state, and both are silent.

## A round's identity comes from the chain, with the calendar attached

The round id is an ISO date, because that is what a commitment carries, what Redis is keyed by and
what a reader wants to see. But the *date* cannot come from a clock: two validators either side of
midnight would label the same round differently, fail to recognise each other's commitments, and
generate packs they could not compare.

So the identity is the epoch index — `block // blocks_per_day`, a fact about the chain — and the
date is derived from it through an anchor in the season config, which every validator on the subnet
shares. The anchor is one (block, date) pair; the mapping is exact because `blocks_per_day` blocks
is exactly one day. A wrong anchor shifts every validator's labels by the same amount, so they still
agree with each other — but they would disagree with the calendar, which is why
`assert_anchor_is_plausible` exists and is called from `--check` rather than from here.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from enum import Enum

__all__ = ["CycleConfig", "CycleError", "Phase"]

_log = logging.getLogger(__name__)

#: Nominal seconds per block. Read only to turn an epoch length into a round label; nothing that
#: decides anything reads it, because every boundary here is a block height precisely so it does
#: not depend on how fast blocks actually arrive.
_BLOCK_SECONDS = 12.0


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
    #: A block that is an epoch start, and the calendar date of the round that begins there. Shared
    #: through the season config so every validator derives the same label from the same chain. No
    #: default: a guessed anchor produces plausible dates that are silently wrong.
    anchor_block: int
    anchor_date: str

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
            anchor_block=int(cycle["anchor_block"]),  # type: ignore[index]
            anchor_date=str(cycle["anchor_date"]),  # type: ignore[index]
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
        self._assert_anchor()
        if self.weights_offset >= self.blocks_per_day + self.reveal_offset:
            raise CycleError(
                f"weights_offset ({self.weights_offset}) does not leave the day's "
                f"{self.blocks_per_day} blocks before the next epoch begins. A cycle that overruns "
                "its day would submit weights for one round during the next one."
            )

    def _assert_anchor(self) -> None:
        """The anchor must name an epoch start and a real date.

        An anchor part-way through an epoch would make `round_id` disagree with `epoch_index` by a
        fraction of a day, which rounds to either zero or one depending on the block — so half the
        rounds would be labelled with yesterday's date.
        """
        if self.anchor_block % self.blocks_per_day != 0:
            raise CycleError(
                f"anchor_block ({self.anchor_block}) is not a multiple of blocks_per_day "
                f"({self.blocks_per_day}), so it does not name the start of an epoch. Every "
                "round's date is derived from it, and an anchor part-way through a day would label "
                "some rounds with yesterday's date and some with today's."
            )
        try:
            date.fromisoformat(self.anchor_date)
        except ValueError as error:
            raise CycleError(
                f"anchor_date ({self.anchor_date!r}) is not an ISO date: {error}"
            ) from error

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

    def epoch_index(self, block: int) -> int:
        """Which day this block belongs to, counted from the chain's genesis."""
        return block // self.blocks_per_day

    def epoch_start_of(self, epoch_index: int) -> int:
        return epoch_index * self.blocks_per_day

    def offset_in(self, epoch_index: int, block: int) -> int:
        """Where a block sits relative to a given round's reveal point.

        Negative before that round's reveal. The round has to be named: a block does not belong to
        one round, so an offset cannot be computed from a block alone. Deriving the round from the
        block instead — `block - epoch_start(block) + reveal_offset` — can only return offsets in
        [0, blocks_per_day), which makes every pre-reveal phase unreachable.
        """
        return block - self.epoch_start_of(epoch_index) + self.reveal_offset

    def round_opens(self) -> int:
        """The offset of a round's first action. Before this, the round has nothing to do."""
        return self.salt_commit_offset

    def round_closes(self) -> int:
        """The offset at which a round is over: the next epoch start.

        Weights are due at `weights_offset` and this is the last block on which they may still be
        submitted. Later than that and the submission would land during the next round's own weights
        window, where the chain cannot tell which round it was for.
        """
        return self.blocks_per_day + self.reveal_offset

    def live_rounds(self, block: int) -> tuple[int, ...]:
        """Every round whose window contains this block, oldest first.

        At most two, and the arithmetic says why: the window spans
        `round_closes() - round_opens()` blocks and rounds start every `blocks_per_day`, so the
        overlap is one round deep for any sane config. The candidates are this block's own epoch and
        the next one — the previous epoch's window closes exactly at this epoch's start.
        """
        current = self.epoch_index(block)
        live = [
            index
            for index in (current, current + 1)
            if self.round_opens() <= self.offset_in(index, block) < self.round_closes()
        ]
        return tuple(live)

    #: Blocks in a real calendar day at nominal block time. A season with this many or more labels
    #: rounds by date; a shorter one labels by hour, because several of its epochs share a date
    #: and a date-only label would collide.
    CALENDAR_DAY_BLOCKS = 7_200

    def epoch_seconds(self) -> float:
        """Wall-clock time one epoch spans, at nominal block time."""
        return self.blocks_per_day * _BLOCK_SECONDS

    def round_moment(self, epoch_index: int) -> datetime:
        """When a round begins. What the plausibility check compares against a clock."""
        anchor_index = self.anchor_block // self.blocks_per_day
        return datetime.fromisoformat(self.anchor_date).replace(tzinfo=UTC) + timedelta(
            seconds=(epoch_index - anchor_index) * self.epoch_seconds()
        )

    def round_id(self, epoch_index: int) -> str:
        """The round's label, derived from the epoch index through the season anchor.

        `2026-08-04` on a mainnet season; `2026-08-04T20` on a compressed one. Both sort lexically
        in chronological order, which `recent()` and the dashboard's key scan depend on.

        ## Why the hour appears

        The label was always a date, assuming one epoch is one calendar day. That holds for
        `blocks_per_day = 7200` and breaks for any testnet wanting a round to finish in under a day:
        with a 300-block epoch, twenty-four rounds fall on one date, every one computes the same id,
        and they collide in the commitment, in Redis and on the dashboard.

        Worse, `assert_anchor_is_plausible` compares the label to today and allows one day of
        drift — so a compressed season starts, runs two hours, then refuses to boot because its
        labels have outrun the calendar: a validator that works this afternoon and will not start
        tomorrow.

        So granularity follows the epoch length rather than being assumed. Nothing about the mainnet
        label changes, which matters because a label is what a commitment carries and what every
        stored round is keyed by.
        """
        moment = self.round_moment(epoch_index)
        if self.blocks_per_day >= self.CALENDAR_DAY_BLOCKS:
            return moment.date().isoformat()
        # Hour granularity and no finer: an epoch under an hour would need minutes, and a cycle that
        # short cannot fit a generation call, let alone an execution window.
        return moment.strftime("%Y-%m-%dT%H")

    def assert_anchor_is_plausible(self, *, block: int, now: date) -> None:
        """Check the anchor against the calendar. Called from `--check`, not from construction.

        Every validator with the same anchor agrees with every other, whatever the anchor says —
        so a wrong anchor is not a consensus failure, it is a labelling failure, and it would be
        found months later by someone reading a date. This catches it at deployment.

        Not called from `__post_init__` because it needs a clock and a chain height, and this module
        must be constructible from a config alone.
        """
        # Compared as a moment, not a parsed label: a compressed season's label is not a date, and
        # `date.fromisoformat` raised on it — so the check that exists to catch a wrong anchor was
        # itself the thing that would not start.
        derived = self.round_moment(self.epoch_index(block))
        drift = abs((derived.date() - now).days)
        if drift > 1:
            raise CycleError(
                f"the anchor puts block {block} in round "
                f"{self.round_id(self.epoch_index(block))}, but today is "
                f"{now.isoformat()} — {drift} days out. Every validator sharing this anchor "
                "would agree with each other and disagree with the calendar. Fix "
                "anchor_block/anchor_date in the season config: for this chain, anchor_block "
                f"{self.epoch_start_of(self.epoch_index(block))} is round {now.isoformat()}."
            )
