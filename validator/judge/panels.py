"""Judge panels: architecture.md 16.1 and 16.2. Three families, eight roles, one hard cap.

## The family cap, and the reading that would make it vacuous

"No single provider family may control more than 40% of a semantic criterion." 16.1 then spends a
paragraph on how to read that, because there are two readings and one is useless:

Every judge is reached through OpenRouter, so "provider" in the *routing* sense is always
`openrouter` — a cap on that is a cap on nothing. What matters is who **trained** the model
behind the route. Three routes to three Anthropic snapshots is one family and breaches the cap,
however
many distinct slugs it uses, because two versions of one model share their failure modes.

So the cap is evaluated on the declared `family` field, which is why that field is required in the
season config rather than derived from the slug. Deriving it would seem tidier and would be exactly
the mistake: a miner-hosted fine-tune routed through OpenRouter has a slug that says nothing about
what trained it.

## Why cross-family judging follows from 7.2.1 for free

The challenge generators are GPT and Claude; the judge panel requires at least three families
including both. So a problem written by GPT is judged by a panel containing Claude, and the reverse.
"No family both sets a problem and unilaterally decides the answer" — and this needs no extra
mechanism, only the cap.

## An absent judge is absent, never a default

A judge whose call failed or whose JSON did not parse is recorded as **not having voted**. Its share
of the criterion redistributes across the judges that did vote
(`protocol.fixedpoint.apply_weights`).
The alternative — substituting a neutral score — puts a number nobody produced into a miner's
payment, and does it in the direction that looks harmless: a 0.5 on a criterion where the real
answer was 0.9 costs the miner exactly as much as an error in the other direction would gain them.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from gateway.adapters.openrouter import ModelPin
from protocol.fixedpoint import PPM

__all__ = [
    "JUDGE_ROLES",
    "Panel",
    "PanelError",
    "PanelJudge",
    "assert_family_cap",
    "panels_from_season",
]

_log = logging.getLogger(__name__)


class PanelError(ValueError):
    """A panel that does not satisfy 16.1."""


#: 16.2's eight roles, each with the question it answers. The question wording goes into the prompt,
#: so a judge is told what it is deciding rather than only which criterion it is scoring — a judge
#: asked to "score mechanism" invents its own rubric, and two judges then score different things.
JUDGE_ROLES: Mapping[str, str] = {
    "constraint_fit": (
        "Did the answer satisfy the actual research objective and the stated constraints? Not "
        "whether it is a good idea — whether it is an answer to *this* problem."
    ),
    "originality": (
        "Is the mechanism materially different from the prior art supplied, rather than renamed or "
        "superficially recombined? Judge against the prior-art report, not against your impression "
        "of what exists."
    ),
    "value": (
        "Would this plausibly create meaningful scientific, technical or commercial value? Judge "
        "the magnitude claim, treating any quantity marked [unverified] as unverified."
    ),
    "mechanism": (
        "Is there a coherent causal or technical mechanism explaining how this works? A description "  # noqa: E501 - prompt text; a break would change what the judge reads
        "of what it achieves is not a mechanism; a chain of steps where each follows from the last "
        "is."
    ),
    "diversity": (
        "Do the five ideas represent substantially different research directions, or five "
        "variations on one? Ideas listed in the duplicate_clusters field have already been "
        "identified as one lineage."
    ),
    "self_selection": (
        "Did the laboratory rank its strongest idea first? Judge the ranking, not the ideas: a "
        "laboratory with five weak ideas that ranked them correctly scores well here."
    ),
    "falsifiability": (
        "Does this produce discriminating predictions or a meaningful kill test — an experiment "
        "whose outcome would change what you believe? A test that cannot fail is not a test."
    ),
    "cost_reliability": (
        "Did the laboratory turn its best idea into a concrete research path, simulation or "
        "calculation, at a cost it stated and can defend?"
    ),
}


@dataclass(frozen=True, slots=True)
class PanelJudge:
    """One judge: a model family and the pinned route that means it this season."""

    family: str
    pin: ModelPin

    def __post_init__(self) -> None:
        if not self.family:
            raise PanelError(
                "a judge with no declared family cannot be counted against the 16.1 cap, and the "
                "cap is on the family that trained the model rather than on the route"
            )


@dataclass(frozen=True, slots=True)
class Panel:
    """The judges for one criterion, with the reliability floor 19 requires of them."""

    criterion: str
    judges: tuple[PanelJudge, ...]
    reliability_floor_ppm: int

    def families(self) -> tuple[str, ...]:
        return tuple(sorted({judge.family for judge in self.judges}))

    def assert_valid(self, *, minimum_families: int, family_cap_ppm: int) -> None:
        if self.criterion not in JUDGE_ROLES:
            raise PanelError(
                f"panel criterion {self.criterion!r} is not one of 16.2's roles "
                f"({sorted(JUDGE_ROLES)}). A criterion with no declared role has no rubric, so "
                "each judge would invent one."
            )
        if len(self.families()) < minimum_families:
            raise PanelError(
                f"criterion {self.criterion!r} has {len(self.families())} model families "
                f"({list(self.families())}); 16.1 requires at least {minimum_families}. Fewer "
                "means the criterion's failure modes are shared rather than independent."
            )
        assert_family_cap(self.criterion, self.judges, family_cap_ppm=family_cap_ppm)


def assert_family_cap(
    criterion: str, judges: Sequence[PanelJudge], *, family_cap_ppm: int
) -> None:
    """16.1's cap, counted on families rather than on routes or slugs.

    Two snapshots of one family are one family. Counted by grouping on the declared `family` field,
    so three Anthropic snapshots contribute three votes to one family and breach a 40% cap on any
    panel of fewer than eight judges — which is the intended outcome, and the one a slug-derived
    reading would miss.
    """
    if not judges:
        raise PanelError(f"criterion {criterion!r} has no judges at all")

    counts: dict[str, int] = {}
    for judge in judges:
        counts[judge.family] = counts.get(judge.family, 0) + 1

    for family, count in sorted(counts.items()):
        share = count * PPM // len(judges)
        if share > family_cap_ppm:
            raise PanelError(
                f"criterion {criterion!r} gives {family!r} {count} of {len(judges)} votes "
                f"({share / 10_000:.1f}%), over the {family_cap_ppm / 10_000:.1f}% cap in 16.1. "
                "Two snapshots of one family are one family: they share their failure modes, so "
                "the cap is on what trained the model rather than on how many slugs it has."
            )


def panels_from_season(season: Mapping[str, Any]) -> dict[str, Panel]:
    """Build and validate every declared panel.

    Validated here, at load, rather than at first use. A panel that breaches the cap must stop a
    validator from starting — discovering it mid-round would mean the criterion had already been
    scored by a panel the protocol forbids, and those scores cannot be retracted from a weight
    vector already on chain.
    """
    judging = season["judging"]
    minimum_families = int(judging["minimum_families"])
    family_cap_ppm = int(judging["family_cap_ppm"])

    panels: dict[str, Panel] = {}
    for declared in judging["panels"]:
        panel = Panel(
            criterion=str(declared["criterion"]),
            judges=tuple(
                PanelJudge(
                    family=str(judge["family"]),
                    pin=ModelPin(
                        slug=str(judge["model_slug"]), snapshot=str(judge["model_snapshot"])
                    ),
                )
                for judge in declared["judges"]
            ),
            reliability_floor_ppm=int(declared["reliability_floor_ppm"]),
        )
        panel.assert_valid(
            minimum_families=minimum_families, family_cap_ppm=family_cap_ppm
        )
        panels[panel.criterion] = panel

    if not panels:
        raise PanelError(
            "the season declares no judge panels, so no semantic criterion can be scored and every "
            "laboratory would receive the same score"
        )

    missing = sorted(set(JUDGE_ROLES) - set(panels))
    if missing:
        # Not fatal: a season may deliberately score a subset, and `apply_weights` redistributes
        # over the criteria present so an omitted criterion is not scored zero. Logged because the
        # usual cause is an incomplete config rather than a decision.
        _log.warning(
            "no panel declared for %s. Their criterion weights redistribute over the criteria "
            "that are declared, so they are not scored zero — but check this was intended.",
            missing,
        )
    return panels


def pins_for(panels: Mapping[str, Panel]) -> dict[str, ModelPin]:
    """family -> pin, for the shared `ModelClient`.

    One pin per family across all panels. A family appearing in two panels with two different pins
    is refused: it would mean "the Claude judge" meant two different models depending on the
    criterion, and the 16.1 cap counts them as one family — so the cap and the routing would
    disagree about what a family is.
    """
    pins: dict[str, ModelPin] = {}
    for panel in panels.values():
        for judge in panel.judges:
            existing = pins.get(judge.family)
            if existing is not None and existing != judge.pin:
                raise PanelError(
                    f"family {judge.family!r} is pinned to {existing.slug} in one panel and "
                    f"{judge.pin.slug} in another. The 16.1 cap treats them as one family, so two "
                    "routes under one family name would make the cap and the routing disagree."
                )
            pins[judge.family] = judge.pin
    return pins
