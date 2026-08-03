"""Rolling scores to a weight vector: architecture.md 20.

```
qualification floor  --20.1-->  only laboratories above the reference-lab floor
softmax at tau       --20.2-->  p_i = exp((S_i - S_min)/tau) / sum_j exp((S_j - S_min)/tau)
cap and redistribute --20.3-->  no laboratory above 15-20%
nobody qualifies     --20.4-->  100% to the burn UID
```

## Softmax rather than proportional, and what tau controls

Proportional allocation makes the *ratio* of two scores decide the split, so a laboratory at
0.60 earns twice a laboratory at 0.30 whether the field is tight or spread. Softmax makes the
*gap* decide it, which is what a bounded score needs: at tau = 0.1, a 0.10 gap is roughly a
2.7x weight ratio regardless of where in the range the two sit.

That is why the distribution is a policy decision made once, here, with a temperature and a
cap — and why the Bradley-Terry conversion upstream reports rank position rather than rescaling
strengths. If the fit also spread the field, tau would be tuning something already tuned.

## The floor is a threshold on the score, not a rank

20.1: a laboratory qualifies only if "its rolling score exceeds the reference-lab floor". *Being
best among weak miners is not enough.* So the floor is absolute: if every competitor is worse
than direct frontier-model use, none of them qualifies and the emission burns. A rank-based
floor — "the top N qualify" — would always pay someone, which is precisely the outcome 20.4
exists to refuse.

## Redistribution after capping has to iterate

Capping one laboratory and handing its overflow to the others can push a second over the cap.
One pass would leave that second one above the limit, so the redistribution repeats until no
laboratory exceeds it. The loop is bounded by the number of competitors, since each pass either
caps at least one more or terminates.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from protocol.fixedpoint import PPM

__all__ = [
    "Allocation",
    "Candidate",
    "WeightsConfig",
    "WeightsError",
    "allocate",
]

_log = logging.getLogger(__name__)

#: Bittensor weights are u16. The vector is emitted in this range, and the conversion is the
#: last step so every earlier stage stays in ppm.
_U16_MAX = 65_535


class WeightsError(ValueError):
    """A weight vector that cannot be produced as configured."""


@dataclass(frozen=True, slots=True)
class WeightsConfig:
    """Allocation parameters from the signed season."""

    temperature_ppm: int
    maximum_weight_ppm: int
    minimum_valid_challenges: int
    burn_uid: int
    reference_floor_lab: str

    def __post_init__(self) -> None:
        if self.temperature_ppm <= 0:
            raise WeightsError(
                "temperature must be positive: at zero the softmax becomes winner-take-all, "
                "which section 20.3's cap exists to prevent"
            )
        if not 0 < self.maximum_weight_ppm <= PPM:
            raise WeightsError(f"maximum weight {self.maximum_weight_ppm} is outside (0, {PPM}]")

    @classmethod
    def from_season(cls, season: Mapping[str, object]) -> WeightsConfig:
        allocation = season["weight_allocation"]  # type: ignore[index]
        return cls(
            temperature_ppm=int(allocation["temperature_ppm"]),  # type: ignore[index]
            maximum_weight_ppm=int(allocation["maximum_weight_ppm"]),  # type: ignore[index]
            minimum_valid_challenges=int(
                allocation["minimum_valid_challenges"]  # type: ignore[index]
            ),
            burn_uid=int(allocation["burn_uid"]),  # type: ignore[index]
            reference_floor_lab=str(
                allocation.get("reference_floor_lab", "")  # type: ignore[union-attr]
            ),
        )


@dataclass(frozen=True, slots=True)
class Candidate:
    """One laboratory's standing at allocation time."""

    uid: int
    rolling_score_ppm: int
    valid_challenges: int
    hard_gates_passed: bool
    artifacts_available: bool

    def disqualification(self, config: WeightsConfig, floor_ppm: int) -> str:
        """Why this laboratory does not qualify, or `""` if it does.

        A reason string rather than a bool, because 22 publishes hard-gate outcomes: a
        laboratory excluded without a stated reason cannot tell a failed gate from an
        unavailable artifact from simply being below the floor, and those need different fixes.
        """
        if not self.hard_gates_passed:
            return "a hard gate failed"
        if not self.artifacts_available:
            return "the bundle or its model artifacts are no longer available"
        if self.valid_challenges < config.minimum_valid_challenges:
            return (
                f"{self.valid_challenges} valid challenges, below the minimum "
                f"{config.minimum_valid_challenges}"
            )
        if self.rolling_score_ppm <= floor_ppm:
            return (
                f"rolling score {self.rolling_score_ppm} does not exceed the reference floor "
                f"{floor_ppm}"
            )
        return ""


@dataclass(frozen=True, slots=True)
class Allocation:
    """The vector, and the reasoning behind it.

    `excluded` and `capped` are kept because 22 publishes the weight vector, and a vector with
    no accompanying explanation is a number to be trusted rather than audited.
    """

    uids: list[int]
    weights: list[int]
    burned: bool
    excluded: Mapping[int, str]
    capped: tuple[int, ...]

    @property
    def weights_ppm(self) -> dict[int, int]:
        return {
            uid: weight * PPM // _U16_MAX
            for uid, weight in zip(self.uids, self.weights, strict=True)
        }


def _softmax_ppm(scores: Mapping[int, int], temperature_ppm: int) -> dict[int, int]:
    """`exp((S_i - S_min)/tau)` normalised, in ppm.

    Shifted by the minimum before exponentiating. Mathematically the shift cancels, but
    numerically it is what keeps `exp` in range: without it a score expressed in ppm divided by
    a ppm temperature is an exponent in the hundreds, which overflows to infinity and yields
    `nan` weights.

    Iterates in sorted uid order. Floating-point addition is not associative, so summing the
    denominator in dict order would make the vector depend on insertion order — and section 27
    measures same-bundle rerun correlation, which that would break.
    """
    ordered = sorted(scores)
    lowest = min(scores.values())
    # Both in ppm, so the ratio is dimensionless and tau reads as a fraction of full range.
    exponents = {uid: (scores[uid] - lowest) / temperature_ppm for uid in ordered}
    weights = {uid: math.exp(exponents[uid]) for uid in ordered}
    total = sum(weights[uid] for uid in ordered)
    if total <= 0 or not math.isfinite(total):
        raise WeightsError(f"the softmax denominator is {total}, which cannot be normalised")
    return {uid: int(weights[uid] * PPM / total) for uid in ordered}


def _effective_cap(declared_ppm: int, qualifiers: int) -> int:
    """The cap actually applied, which is looser than declared in a small field.

    A cap of C can only be satisfied by N laboratories when `N * C >= PPM`. At the declared
    17.5% that needs six qualifiers. With fewer, *every* laboratory lands exactly on the cap —
    and because Bittensor renormalises the vector it receives, `[cap, cap]` becomes a 50/50
    split. The cap would then have flattened the field rather than limiting concentration:
    two laboratories with very different scores would receive identical emission, and the
    incentive to be the better of the two would be gone.

    A test caught this by measuring two fields with different score gaps and finding them
    allocated identically.

    So the applied cap is `max(declared, PPM / N)` — the tightest value the field can actually
    absorb. For six or more qualifiers that is the declared cap and nothing changes. Below
    six it relaxes, which is the honest reading of section 20.3: the cap is stated as a
    *concentration* control, and concentration is only meaningful relative to a field. A small
    field's protection is the qualification floor, not a ceiling it cannot satisfy.

    Relaxing the cap to `PPM / N` is not enough, and it took a second measurement to see why.
    With two qualifiers, capping the leader leaves exactly one receiver for the overflow, and
    that receiver's headroom equals the overflow — so it lands precisely on the cap too. The
    result is 50/50 for *any* cap at or below half. Flattening with two laboratories is not a
    tuning problem; it is arithmetic.

    So below the satisfiable threshold the cap does not bind at all, and the softmax stands. A
    small field's protection against a bad allocation is the qualification floor, which has
    already established that every qualifier beats direct frontier-model use. The cap starts
    binding the moment the field is large enough for it to mean something.

    The single-qualifier case does not come here. It is handled separately and *does* apply
    the declared cap, because one laboratory taking the whole emission is the clearest
    concentration risk there is.
    """
    needed = -(-PPM // declared_ppm)  # ceil
    if qualifiers >= needed:
        return declared_ppm
    _log.info(
        "%d qualifiers cannot absorb a %d ppm cap (it needs %d); the cap does not bind this "
        "round. Capping a field this small forces an even split whatever the cap value, which "
        "would remove the incentive to be the better laboratory. The qualification floor is the "
        "protection here.",
        qualifiers,
        declared_ppm,
        needed,
    )
    return PPM


def _cap_and_redistribute(
    weights_ppm: Mapping[int, int], cap_ppm: int
) -> tuple[dict[int, int], tuple[int, ...]]:
    """Hold every laboratory at or below the cap, redistributing the overflow (20.3).

    Iterated rather than single-pass: handing one laboratory's overflow to the others can push a
    second over the cap, and one pass would leave it there. Bounded by the number of
    competitors, since each pass caps at least one more or terminates.

    If every laboratory is at the cap the total falls short of one whole — with a 17.5% cap that
    needs six qualifiers to reach 100%. The shortfall is left rather than forced: scaling
    everyone back up would breach the cap, and Bittensor normalises the vector it receives, so
    the *relative* allocation is what matters and is correct.
    """
    current = dict(weights_ppm)
    capped: set[int] = set()

    for _ in range(len(current) + 1):
        over = {uid: value for uid, value in current.items() if value > cap_ppm}
        if not over:
            break
        overflow = sum(value - cap_ppm for value in over.values())
        for uid in over:
            current[uid] = cap_ppm
            capped.add(uid)
        receivers = [uid for uid in sorted(current) if uid not in capped]
        if not receivers:
            # Everyone is capped. The remainder is simply not allocated, which is correct:
            # forcing it out would breach the cap the season declared.
            _log.info(
                "every qualifier is at the %d ppm cap; %d ppm is left unallocated rather than "
                "breaching it",
                cap_ppm,
                overflow,
            )
            break
        # Distributed by existing share, so redistribution preserves the ranking among the
        # uncapped rather than flattening them.
        base = sum(current[uid] for uid in receivers)
        if base <= 0:
            share = overflow // len(receivers)
            for uid in receivers:
                current[uid] += share
        else:
            for uid in receivers:
                current[uid] += overflow * current[uid] // base

    return current, tuple(sorted(capped))


def allocate(
    candidates: Sequence[Candidate],
    *,
    reference_floor_ppm: int,
    config: WeightsConfig,
) -> Allocation:
    """The weight vector for one round (20.1 through 20.5).

    `reference_floor_ppm` is the reference laboratory's own rolling score — Reference A by
    default, the direct frontier-model baseline. It is passed in rather than derived here
    because the reference labs are scored by the same pipeline as everyone else: a floor
    computed separately would be a second scoring path, and the two would drift.
    """
    excluded: dict[int, str] = {}
    qualified: dict[int, int] = {}

    for candidate in sorted(candidates, key=lambda c: c.uid):
        reason = candidate.disqualification(config, reference_floor_ppm)
        if reason:
            excluded[candidate.uid] = reason
        else:
            qualified[candidate.uid] = candidate.rolling_score_ppm

    if not qualified:
        # 20.4. Paying nobody is a valid outcome and the whole point of an absolute floor:
        # if no laboratory beats direct frontier-model use, the subnet has not demonstrated
        # that competing architectures add value, and emitting anyway would pay for that.
        _log.warning(
            "no laboratory exceeds the reference floor %d; burning 100%% of emission. "
            "Being best among weak miners is not enough (section 20.1).",
            reference_floor_ppm,
        )
        return Allocation(
            uids=[config.burn_uid],
            weights=[_U16_MAX],
            burned=True,
            excluded=excluded,
            capped=(),
        )

    if len(qualified) == 1:
        # A single qualifier would take everything under a softmax. The cap still applies, and
        # the remainder burns: one laboratory above the floor has earned its capped share, and
        # the rest has been earned by nobody.
        only = next(iter(qualified))
        capped_share = min(config.maximum_weight_ppm, PPM)
        remainder = PPM - capped_share
        _log.info(
            "one qualifier; allocating %d ppm and burning the remaining %d rather than lifting "
            "a single laboratory above the declared cap",
            capped_share,
            remainder,
        )
        uids = sorted({only, config.burn_uid})
        shares = {only: capped_share, config.burn_uid: remainder}
        return Allocation(
            uids=uids,
            weights=[shares.get(uid, 0) * _U16_MAX // PPM for uid in uids],
            burned=False,
            excluded=excluded,
            capped=(only,) if capped_share < PPM else (),
        )

    softmaxed = _softmax_ppm(qualified, config.temperature_ppm)
    final, capped = _cap_and_redistribute(
        softmaxed, _effective_cap(config.maximum_weight_ppm, len(qualified))
    )

    uids = sorted(final)
    return Allocation(
        uids=uids,
        weights=[final[uid] * _U16_MAX // PPM for uid in uids],
        burned=False,
        excluded=excluded,
        capped=capped,
    )
