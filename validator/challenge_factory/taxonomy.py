"""The domain plan for a day's pack. Pure, seeded, and reproducible.

architecture.md 7.2. Twenty challenges per validator per day, stratified across the active
taxonomy, with ten slots owned by each generator family.

## Two independent assignments over the same twenty slots

A slot has a **domain** (which of the eight areas the problem is about) and a **family** (which
generator writes it). Both are derived from the daily seed, and — this is the part that matters —
they are derived *independently*.

If domains and families were correlated, the two-generator design would break in a way that
looks fine. Suppose GPT always drew the algorithm slots and Claude always drew the architecture
slots. Then a laboratory good at algorithms scores well on GPT's half for reasons that have
nothing to do with generator style, and 7.2.1's measurement — "a laboratory that scores far
better on one family's problems has learned a generator, not a domain" — measures the domain
instead. The signal it exists to give would be there from day one and mean nothing.

So `plan` shuffles the domain multiset with one label and the family multiset with another,
from the same seed. Two independent deals over the same slots.

## Stratification counts are declared, not computed

The season config states how many slots each domain gets, and those must sum to the pack size.
Computing an even split instead would look tidier and would be wrong: 7.2 gives AI-agent
architecture four slots and mechanism design two, because the subnet cares about them unequally.
An even split would silently overrule that.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from protocol.seeds import _seeded_stream, slot_assignments

__all__ = [
    "EXCLUDED_DOMAINS",
    "Slot",
    "TaxonomyError",
    "Taxonomy",
    "plan",
]


class TaxonomyError(ValueError):
    """A taxonomy that cannot produce a valid pack plan."""


#: Domains 2 excludes from V1 scoring, restated here so the linter and the safety filter can
#: check against one list. Duplicated from the season config on purpose: a season that omitted
#: the exclusions would otherwise silently permit them, and these are the categories where a
#: wrong answer causes harm outside the subnet.
EXCLUDED_DOMAINS = frozenset(
    {
        "clinical_or_medical_treatment",
        "wet_lab_chemistry_or_biology",
        "physical_engineering_requiring_measurement",
        "legal_or_policy",
        "weapons_malware_or_exploits",
        "broad_artistic_ideation",
        "long_horizon_real_world_outcomes",
    }
)


@dataclass(frozen=True, slots=True)
class Slot:
    """One position in the day's pack: what it is about, and who writes it."""

    index: int
    domain: str
    generator_family: str
    critic_family: str

    def __post_init__(self) -> None:
        if not self.critic_family:
            raise TaxonomyError(
                f"slot {self.index}: no critic family. A slot with no reviewer would put an "
                "unreviewed problem into a committed pack that the whole cohort is scored on."
            )
        # Generator and critic may be the same family. They were originally required to differ
        # (7.2.2), and the owner has since decided the critique should not reach across model
        # families — so the season config supplies `critic_family` and this only checks that one
        # exists. See `critic.py` for what that costs.


@dataclass(frozen=True, slots=True)
class Taxonomy:
    """The season's active taxonomy and its stratification."""

    domains: tuple[str, ...]
    challenges_per_day: int
    #: domain -> slot count. Must sum to `challenges_per_day`.
    stratification: Mapping[str, int]
    excluded_domains: frozenset[str]

    @classmethod
    def from_season(cls, season: Mapping[str, object]) -> Taxonomy:
        block = season["taxonomy"]  # type: ignore[index]
        taxonomy = cls(
            domains=tuple(block["domains"]),  # type: ignore[index]
            challenges_per_day=int(block["challenges_per_day"]),  # type: ignore[index]
            stratification=dict(block["stratification"]),  # type: ignore[index]
            excluded_domains=frozenset(block.get("excluded_domains", ())),  # type: ignore[union-attr]
        )
        taxonomy.validate()
        return taxonomy

    def validate(self) -> None:
        """Check the config before a day depends on it.

        Every failure here would otherwise surface as a pack of the wrong size *after* the salt
        was committed and the randomness drawn — at which point the day cannot be re-planned
        without breaking 7.3's ordering.
        """
        if self.challenges_per_day < 1:
            raise TaxonomyError("a pack with no challenges leaves every laboratory equal")

        total = sum(self.stratification.values())
        if total != self.challenges_per_day:
            raise TaxonomyError(
                f"stratification sums to {total} but the pack is {self.challenges_per_day}. "
                "Reconciling by scaling would silently overrule the declared emphasis; 7.2 "
                "gives agent architecture four slots and mechanism design two because the "
                "subnet cares about them unequally."
            )

        unknown = set(self.stratification) - set(self.domains)
        if unknown:
            raise TaxonomyError(
                f"stratification names domains not in the taxonomy: {sorted(unknown)}"
            )

        forbidden = (set(self.domains) | set(self.stratification)) & (
            self.excluded_domains | EXCLUDED_DOMAINS
        )
        if forbidden:
            raise TaxonomyError(
                f"the taxonomy includes domains excluded from V1 scoring: {sorted(forbidden)}. "
                "These are the categories where a plausible-but-wrong invention causes harm "
                "outside the subnet, and 2 excludes them from scoring rather than from mention."
            )

        for domain, count in sorted(self.stratification.items()):
            if count < 1:
                raise TaxonomyError(
                    f"domain {domain!r} is stratified at {count}: a domain with no slots should "
                    "be removed from the stratification rather than declared at zero, so the "
                    "config states what it means"
                )


def plan(
    seed: bytes, *, taxonomy: Taxonomy, generators: Sequence[Mapping[str, object]]
) -> tuple[Slot, ...]:
    """The day's twenty slots, fixed by the seed before any generation happens.

    7.4 step 1: "Slot assignment is fixed by the daily seed before generation begins, so a
    validator cannot decide after the fact which family produced which surviving problem." That
    is what stops a validator generating both halves, keeping whichever half suits a submission
    it has seen, and attributing the survivors freely.

    Families come from `protocol.seeds.slot_assignments`, which deals an exact multiset — ten and
    ten, not ten and ten on average. Domains are dealt here by the same method and a different
    label, so the two deals are independent.
    """
    families = slot_assignments(seed, generators)
    if len(families) != taxonomy.challenges_per_day:
        raise TaxonomyError(
            f"generators declare {len(families)} slots but the taxonomy declares "
            f"{taxonomy.challenges_per_day}. The pack size has one definition, and two "
            "disagreeing ones mean some slot is either unassigned or generated twice."
        )

    critics = {
        str(generator["family"]): str(generator["critic_family"]) for generator in generators
    }
    missing = set(families) - set(critics)
    if missing:
        raise TaxonomyError(f"no critic family declared for generator(s) {sorted(missing)}")

    domains = _dealt_domains(seed, taxonomy)
    return tuple(
        Slot(
            index=index,
            domain=domains[index],
            generator_family=family,
            critic_family=critics[family],
        )
        for index, family in enumerate(families)
    )


def _dealt_domains(seed: bytes, taxonomy: Taxonomy) -> tuple[str, ...]:
    """Domains for each slot: the declared multiset, shuffled by the seed.

    Sorted before shuffling so the starting order does not depend on the config file's key
    order. Without that, two validators with the same seed and the same stratification written
    in a different order would produce different packs — and the pack hash is committed, so
    they would be unable to explain the difference.
    """
    pool: list[str] = []
    for domain, count in sorted(taxonomy.stratification.items()):
        pool.extend([domain] * count)

    # A distinct label from the family deal. Sharing one stream would correlate the two, and a
    # correlation makes 7.2.1's generator-overfit signal measure domain skill instead.
    stream = _seeded_stream(seed, b"domain-assignment")
    for index in range(len(pool) - 1, 0, -1):
        swap = next(stream) % (index + 1)
        pool[index], pool[swap] = pool[swap], pool[index]
    return tuple(pool)
