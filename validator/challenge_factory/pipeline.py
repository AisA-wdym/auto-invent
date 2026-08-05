"""The six steps of 7.4, in order, with the commitment before the store.

    domain sampler → generator → linter → critic → safety → dedup → discrimination → commitment

The ordering is a cost gradient and a correctness constraint at once.

**Cost.** The linter is free, the critic is one call, the safety filter is free, dedup is one
embedding, and the discrimination probe is four laboratory runs plus a judge panel. So they run
cheapest-first, and a candidate rejected by the linter never reaches the probe. Reordering this
would multiply generation cost by roughly the reject rate.

**Correctness.** The commitment comes after all filtering and before the store. Committing earlier
would commit to a pack still being filtered; storing earlier would leave a window in which the
stored pack and the committed hash could differ.

## What happens when a slot cannot be filled

A slot whose candidates are all rejected leaves the pack short, and the pack size is committed. The
pipeline does not silently ship nineteen challenges: `build_pack` raises unless every slot filled,
because a nineteen-challenge day scored against a twenty-challenge commitment cannot be verified by
anyone, and a validator that quietly shipped short packs would drift away from its peers in a way
that looks like a scoring disagreement.

What it does instead is report exactly which slots failed and why, so the operator can see whether
the cause is a rate limit (retry the day) or a generator that has stopped producing acceptable
problems (a real problem, and the earliest signal of it).

## Rejection reasons are kept, not discarded

Every rejected candidate keeps its reason. 22 publishes generation outcomes, and the reasons are
what make a pack auditable: a reader can see that six candidates died on `answer_leakage` and form
a view about whether the generator is degrading. Discarding them would leave only the survivors,
which is the one view that cannot show a problem.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from protocol.canonical import digest_object
from protocol.seeds import challenge_id
from validator.challenge_factory.critic import CriticVerdict, review
from validator.challenge_factory.dedup import fingerprint, is_duplicate
from validator.challenge_factory.discriminator import (
    DiscriminationVerdict,
    ReferenceProbe,
    assess,
)
from validator.challenge_factory.generator import (
    GeneratorConfig,
    generate_for_slot,
)
from validator.challenge_factory.linter import lint
from validator.challenge_factory.safety import screen
from validator.challenge_factory.store import ChallengeStore, StoredPack
from validator.challenge_factory.taxonomy import EXCLUDED_DOMAINS, Slot, Taxonomy
from validator.model_client import ModelClient

__all__ = [
    "PackResult",
    "PipelineError",
    "Rejection",
    "build_pack",
    "commit_and_store",
    "pack_hash",
]

_log = logging.getLogger(__name__)


class PipelineError(RuntimeError):
    """The day's pack could not be built as specified."""


@dataclass(frozen=True, slots=True)
class Rejection:
    """One candidate that did not make it, and the step that stopped it."""

    slot_index: int
    attempt: int
    step: str
    reason: str


@dataclass(frozen=True, slots=True)
class PackResult:
    """A completed day's pack, with everything needed to audit how it was built."""

    date: str
    challenges: tuple[Mapping[str, Any], ...]
    generation_protocol_version: str
    #: family -> how many surviving challenges it wrote. Published; the per-slot split is not.
    challenges_per_generator: Mapping[str, int]
    rejections: tuple[Rejection, ...]
    #: Total validator RCC spent generating the pack.
    rcc: int
    #: Whether 7.4 step 5 actually ran. False means the pack reached this point without the
    #: strongest filter in the pipeline having been applied.
    #:
    #: Recorded rather than assumed because it *was* assumed: `build_pack` took `probe=None` as a
    #: default and skipped the step, the validator's composition passed no probe, and the resulting
    #: `PackResult` looked exactly like a probed one. A pack that skipped the discrimination check
    #: may contain problems no laboratory can be distinguished on — and twenty of those is a day
    #: whose ranking is noise. `commit_and_store` refuses one unless the season says otherwise.
    discrimination_probed: bool = True

    def hash(self) -> str:
        return pack_hash(
            date=self.date,
            challenges=self.challenges,
            generation_protocol_version=self.generation_protocol_version,
        )

    def stored(self) -> StoredPack:
        return StoredPack(
            date=self.date,
            pack_hash=self.hash(),
            challenges=self.challenges,
            generation_protocol_version=self.generation_protocol_version,
            challenges_per_generator=self.challenges_per_generator,
        )

    def rejections_by_step(self) -> dict[str, int]:
        """How many candidates each step rejected. The operator's health signal.

        A shift in this distribution is the earliest visible sign of trouble: a rising `dedup`
        count means the generator is repeating itself, and a rising `discrimination` count means
        the problems are getting easier or the panel is getting worse.
        """
        counts: dict[str, int] = {}
        for rejection in self.rejections:
            counts[rejection.step] = counts.get(rejection.step, 0) + 1
        return counts


def pack_hash(
    *, date: str, challenges: Sequence[Mapping[str, Any]], generation_protocol_version: str
) -> str:
    """The hash committed on chain in 7.4 step 6.

    Over the date, the protocol version and the challenges — and *not* over the per-slot generator
    attribution. 7.4 step 6 is explicit: "The commitment names the counts per generator but not
    which slot came from which family", because publishing the split per slot would tell a
    laboratory which half of the pack to expect from whom, and the point of two families is that it
    cannot know.

    The challenge bodies themselves carry `generator_family`, so the attribution is recoverable
    *after* publication — which is when it should be recoverable, and not before.
    """
    return digest_object(
        {
            "date": date,
            "generation_protocol_version": generation_protocol_version,
            "challenges": [dict(challenge) for challenge in challenges],
        }
    )


async def _accept_first(
    client: ModelClient,
    *,
    slot: Slot,
    config: GeneratorConfig,
    taxonomy: Taxonomy,
    history: Sequence[tuple[str, Any]],
    probe: ReferenceProbe | None,
    rejections: list[Rejection],
) -> tuple[Mapping[str, Any], int] | None:
    """The first candidate for one slot that survives every step, with its RCC cost.

    Candidates are tried in generation order rather than scored and ranked. Ranking would need a
    quality metric over problems, and the only one available is the discrimination probe — which is
    the most expensive step, so ranking would mean probing every candidate rather than one. First-
    acceptable is the right trade: every survivor has passed the same bar, and the bar is what
    matters rather than which survivor is marginally better.
    """
    candidates = await generate_for_slot(
        client, slot=slot, config=config, excluded=taxonomy.excluded_domains | EXCLUDED_DOMAINS
    )
    spent = sum(candidate.rcc for candidate in candidates)

    for candidate in candidates:
        # Step 2: the linter. Free, so first.
        result = lint(
            candidate.body,
            excluded_domains=taxonomy.excluded_domains | EXCLUDED_DOMAINS,
        )
        if not result.accepted:
            rejections.append(
                Rejection(slot.index, candidate.attempt, "linter", result.reason())
            )
            continue

        # The safety filter, before the critic: also free, and it saves a critic call on a
        # candidate that could never be used.
        safety = screen(
            candidate.body, excluded_domains=taxonomy.excluded_domains | EXCLUDED_DOMAINS
        )
        if not safety.safe:
            rejections.append(
                Rejection(slot.index, candidate.attempt, "safety", safety.reason())
            )
            continue

        # Step 4: dedup. Free against stored fingerprints; the embedding comparison needs one call
        # and is left to the caller to supply via `history`.
        duplicate = is_duplicate(
            candidate.body, history=history, threshold_ppm=config.duplicate_threshold_ppm
        )
        if duplicate.is_duplicate:
            rejections.append(
                Rejection(slot.index, candidate.attempt, "dedup", duplicate.reason())
            )
            continue

        # Step 3: the critic. One call, so after everything free.
        verdict: CriticVerdict = await review(client, candidate)
        spent += verdict.rcc
        if not verdict.accepted:
            rejections.append(
                Rejection(slot.index, candidate.attempt, "critic", verdict.reason())
            )
            continue

        # Step 5: the discrimination probe. Four laboratory runs plus a panel, so last.
        if probe is not None:
            outcome = await probe.probe(candidate.body)
            discrimination: DiscriminationVerdict = assess(
                outcome,
                minimum_spread_ppm=config.minimum_reference_spread_ppm,
                minimum_degradation_gap_ppm=config.minimum_degradation_gap_ppm,
                maximum_instability_ppm=config.maximum_judge_instability_ppm,
            )
            if not discrimination.discriminates:
                rejections.append(
                    Rejection(
                        slot.index, candidate.attempt, "discrimination", discrimination.reason()
                    )
                )
                continue

        body = dict(candidate.body)
        body["challenge_id"] = challenge_id(body)
        return body, spent

    return None


async def build_pack(
    client: ModelClient,
    *,
    date: str,
    slots: Sequence[Slot],
    taxonomy: Taxonomy,
    config: GeneratorConfig,
    store: ChallengeStore | None = None,
    probe: ReferenceProbe | None,
) -> PackResult:
    """Run 7.4 steps 1–5 for every slot. Does not commit and does not store.

    Split from `commit_and_store` deliberately: this function can fail, and a failure here must
    leave nothing on chain and nothing in the store. Combining them would mean a partial pack could
    already have been committed when a later slot failed.

    `probe` is required and may be `None`. Required so that running without the discrimination check
    is a decision the caller writes down rather than a default it inherits; `None` is permitted
    because the probe needs reference-laboratory runs and a judge panel, and a testnet may
    legitimately generate packs before those exist. The absence is recorded on the result and
    `commit_and_store` refuses to commit it unless the season config permits it.

    Slots run sequentially rather than concurrently. Within a slot the candidates are concurrent
    (see `generate_for_slot`), but the slots themselves are not, because each one's dedup check must
    see the challenges the earlier slots accepted — twenty concurrent slots could accept twenty
    near-identical problems, each of which was unique against *history* at the moment it was
    checked.
    """
    if not slots:
        raise PipelineError("no slots planned; there would be nothing to generate")

    if probe is None:
        _log.warning(
            "generating the pack for %s without the discrimination probe (7.4 step 5). Every "
            "candidate that clears the linter, the safety filter, dedup and the critic will be "
            "accepted, including problems on which no laboratory can be distinguished. The result "
            "is marked unprobed and cannot be committed unless the season permits it.",
            date,
        )

    history = list(store.fingerprints()) if store is not None else []
    rejections: list[Rejection] = []
    challenges: list[Mapping[str, Any]] = []
    unfilled: list[Slot] = []
    total_rcc = 0

    for slot in slots:
        accepted = await _accept_first(
            client,
            slot=slot,
            config=config,
            taxonomy=taxonomy,
            history=history,
            probe=probe,
            rejections=rejections,
        )
        if accepted is None:
            unfilled.append(slot)
            continue
        body, spent = accepted
        total_rcc += spent
        challenges.append(body)
        # Added to the in-round history immediately, so a later slot in the same day cannot accept
        # a near-duplicate of this one.
        history.append((str(body["challenge_id"]), fingerprint(body)))
        _log.info(
            "slot %d filled by %s (%s), %d rejected before it",
            slot.index,
            slot.generator_family,
            slot.domain,
            sum(1 for rejection in rejections if rejection.slot_index == slot.index),
        )

    if unfilled:
        summary = ", ".join(
            f"slot {slot.index} ({slot.domain}, {slot.generator_family})" for slot in unfilled
        )
        by_step = {}
        for rejection in rejections:
            by_step[rejection.step] = by_step.get(rejection.step, 0) + 1
        raise PipelineError(
            f"{len(unfilled)} of {len(slots)} slots could not be filled: {summary}. Rejections by "
            f"step: {by_step}. A short pack is not shipped: the pack size is committed, so "
            f"{len(challenges)} challenges scored against a {len(slots)}-challenge commitment "
            "cannot be verified by anyone, and a validator quietly shipping short packs would "
            "diverge from its peers in a way that looks like a scoring disagreement. If the "
            "rejections are concentrated on the critic or the provider, retry the day; if on "
            "dedup, the generator has begun repeating itself."
        )

    per_generator: dict[str, int] = {}
    for body in challenges:
        family = str(body.get("generator_family", ""))
        per_generator[family] = per_generator.get(family, 0) + 1

    declared = {family: int(entry["slots"]) for family, entry in config.generators.items()}
    if per_generator != declared:
        # Cannot normally happen — the slot plan fixes the counts — but if it did, the commitment
        # in 7.4 step 6 would state counts that do not match the pack, and the commitment is the
        # thing a third party checks.
        raise PipelineError(
            f"generator counts came out {per_generator} against the declared {declared}. The "
            "commitment states these counts, so a mismatch would put a false claim on chain."
        )

    return PackResult(
        date=date,
        challenges=tuple(challenges),
        generation_protocol_version=config.protocol_version,
        challenges_per_generator=per_generator,
        rejections=tuple(rejections),
        rcc=total_rcc,
        discrimination_probed=probe is not None,
    )


def commit_and_store(
    result: PackResult,
    *,
    publish: Any,
    store: ChallengeStore,
    salt_commitment: str,
    ttl_days: int,
    allow_unprobed: bool = False,
) -> str:
    """Step 6: commit the hash on chain, *then* store the pack. Returns the hash.

    The ordering is the whole point of this function existing separately, and it is enforced by
    sequence rather than by a flag: `publish` is called first and its failure propagates, so there
    is no path on which the pack is stored without a commitment. `store.write_pack` then refuses
    any hash other than the committed one, so the two cannot disagree even if this function were
    called wrongly.

    `publish` is the chain write, passed in rather than imported, so this module has no dependency
    on the chain and the ordering is testable without one.
    """
    from protocol.commitments import PackCommitment

    if not result.discrimination_probed and not allow_unprobed:
        raise PipelineError(
            f"refusing to commit the pack for {result.date}: it was generated without the "
            "discrimination probe (7.4 step 5), so nothing has established that its problems "
            "separate laboratories at all. A committed pack is scored against every laboratory in "
            "the cohort, and an unprobed one may contain problems on which they are all equal — "
            "which produces a day's ranking made of noise rather than a day with no ranking. Set "
            "challenge_generation.allow_unprobed_packs in the season config to permit this on a "
            "testnet, where it is a deliberate degradation rather than an oversight."
        )
    if not result.discrimination_probed:
        _log.warning(
            "committing an UNPROBED pack for %s because the season permits it. 7.4 step 5 did not "
            "run: no problem in this pack has been shown to discriminate between laboratories.",
            result.date,
        )

    digest = result.hash()
    commitment = PackCommitment(
        round_id=result.date,
        pack_hash=digest,
        salt_commitment=salt_commitment,
        challenge_count=len(result.challenges),
        generation_protocol_version=result.generation_protocol_version,
    )

    block = publish(commitment.encode())
    _log.info(
        "pack hash %s committed at block %s (%d challenges, %s)",
        digest[:16],
        block,
        len(result.challenges),
        dict(result.challenges_per_generator),
    )

    store.write_pack(result.stored(), committed_hash=digest, ttl_days=ttl_days)
    store.record_fingerprints(
        [
            (str(challenge["challenge_id"]), fingerprint(challenge))
            for challenge in result.challenges
        ],
        ttl_days=ttl_days,
    )
    return digest
