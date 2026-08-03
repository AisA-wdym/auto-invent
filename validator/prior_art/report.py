"""Stage 3: prior-art and originality analysis. architecture.md 15.

The defining sentence: **"The validator must never assert that an idea is absolutely
unprecedented."** Everything here is shaped by that.

## Why absolute novelty is not claimable, and what is

A validator searches papers, patents, repositories, products and standards through one provider's
web search. What comes back is *what was findable*, which is a strict subset of what exists — and
the subset shifts with indexing, phrasing, paywalls and the day. An idea that returns nothing is an
idea nothing was found for. Calling that "novel" converts a search result into a claim about the
world, and the claim is false often enough to matter: the ideas most likely to return nothing are
the ones whose vocabulary is unusual, which correlates with genuine novelty *and* with a bad query.

So `PriorArtReport` reports `novelty_confidence` — a bounded statement about the search, not about
the world — and the field that actually drives 18.2's originality criterion is `renaming_only`,
which is a claim about *found* art rather than about absent art. "This mechanism is the one in that
paper with different names" is checkable. "Nothing like this exists" is not.

## `renaming_only` is the finding that carries weight

16.2's Originality judge asks whether the mechanism is "materially different from prior art, rather
than renamed or superficially recombined". That is answerable from what was found: given a paper
whose mechanism matches, the question is whether the miner's differences are substantive or
lexical. This module gathers the evidence for that question; the judge answers it.

The asymmetry is deliberate. A high `novelty_confidence` never *raises* an originality score on its
own — it only fails to lower it. Otherwise a laboratory could earn originality by being hard to
search for, and being hard to search for is cheap: unusual synonyms, invented terminology, a
mechanism described only by its effects.

## Verified difference versus claimed difference

9.2 has every idea state `nearest_prior_art[].material_difference` — the miner's account of how its
idea differs. 15's report carries both `claimed_difference` and `verified_difference`, and keeping
them apart is the point: a miner that describes its difference accurately and a miner that
overstates it look identical if only one field is kept.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from protocol.fixedpoint import PPM, clamp_ppm

__all__ = [
    "Match",
    "PriorArtReport",
    "SearchResult",
    "assess_renaming",
    "novelty_confidence_ppm",
]

_log = logging.getLogger(__name__)

#: Similarity at or above which a match is treated as describing the same mechanism. Not a
#: rejection threshold — an idea may legitimately build on very similar art — but the level at
#: which `renaming_only` becomes the live question.
SAME_MECHANISM_PPM = 780_000


@dataclass(frozen=True, slots=True)
class SearchResult:
    """One thing the search found. What a provider returned, before any judgement."""

    source: str
    url: str
    excerpt: str
    #: Which corpus it came from, from 15's list. Kept because the corpora are not equivalent: a
    #: granted patent claiming the mechanism is a different fact from a blog post describing it.
    corpus: str = "web"


@dataclass(frozen=True, slots=True)
class Match:
    """One prior-art match against one idea, with both accounts of the difference.

    Every ratio is a ppm integer, like the rest of the subnet. `similarity` in 15's JSON is written
    as a decimal; it is carried here as ppm and rendered on the way out, because this value enters
    the originality criterion and a float would make the boundary depend on rounding.
    """

    source: str
    similarity_ppm: int
    shared_mechanism: str
    #: The miner's account, from 9.2's `nearest_prior_art[].material_difference`.
    claimed_difference: str
    #: What the validator could confirm from the found text. Empty when it could confirm nothing —
    #: which is different from confirming there is no difference, and is recorded as empty rather
    #: than as a denial.
    verified_difference: str
    corpus: str = "web"

    def describes_same_mechanism(self) -> bool:
        return self.similarity_ppm >= SAME_MECHANISM_PPM


@dataclass(frozen=True, slots=True)
class PriorArtReport:
    """15's report for one idea. Never asserts absolute novelty.

    There is deliberately no `is_novel` field and no `novel: bool`. A boolean would be read as a
    claim about the world, and the validator is not in a position to make one — it can only report
    what a search returned.
    """

    idea_id: str
    nearest_matches: tuple[Match, ...]
    #: Confidence that the search *did not find* closely matching art. A statement about the
    #: search, bounded by it.
    novelty_confidence_ppm: int
    #: True when the differences the miner claimed are lexical rather than mechanical.
    renaming_only: bool
    #: How the search was performed, so a reader can judge what it could have found.
    queries: tuple[str, ...] = ()
    corpora_searched: tuple[str, ...] = ()
    notes: tuple[str, ...] = field(default=())

    def as_document(self) -> dict[str, Any]:
        """15's JSON shape, with ratios rendered as decimals at the boundary only.

        The ppm integers are what the scoring path uses; these decimals are for publication (22).
        Converting here rather than storing floats keeps every comparison integral.
        """
        return {
            "idea_id": self.idea_id,
            "nearest_matches": [
                {
                    "source": match.source,
                    "similarity": match.similarity_ppm / PPM,
                    "shared_mechanism": match.shared_mechanism,
                    "claimed_difference": match.claimed_difference,
                    "verified_difference": match.verified_difference,
                    "corpus": match.corpus,
                }
                for match in self.nearest_matches
            ],
            "novelty_confidence": self.novelty_confidence_ppm / PPM,
            "renaming_only": self.renaming_only,
            "queries": list(self.queries),
            "corpora_searched": list(self.corpora_searched),
            "notes": list(self.notes),
        }

    def closest(self) -> Match | None:
        return max(self.nearest_matches, key=lambda m: m.similarity_ppm, default=None)


def novelty_confidence_ppm(
    matches: Sequence[Match], *, queries_run: int, corpora_searched: int
) -> int:
    """How confident the *search* is that it found no close match. Never a claim about the world.

    Three inputs, and the two beyond similarity are what keep this honest:

    * **The closest match.** A match at 0.95 similarity means confidence near zero, whatever else
      happened.
    * **How many queries ran.** One query returning nothing is weak evidence; eight queries with
      different phrasings returning nothing is stronger. Without this term, a search that failed to
      execute would score the same as a thorough one that found nothing — and a failed search is
      exactly the case where the answer must not be "novel".
    * **How many corpora were reached.** Papers, patents and repositories fail independently. A
      search that reached only the open web has not looked where patents are.

    So a shallow search caps out well below full confidence. That is the intended shape: the
    validator earns the right to say "we found nothing" by having looked.
    """
    if matches:
        closest = max(match.similarity_ppm for match in matches)
        # Linear in the closest match. A found match is the dominant evidence and no amount of
        # searching elsewhere offsets it.
        from_similarity = clamp_ppm(PPM - closest)
    else:
        from_similarity = PPM

    # Effort ceiling. Eight queries across four corpora reaches full confidence; anything less caps
    # it proportionally. Deliberately hard to saturate — the failure mode being guarded against is a
    # validator whose search silently returned nothing.
    query_factor = min(PPM, queries_run * PPM // 8)
    corpus_factor = min(PPM, corpora_searched * PPM // 4)
    effort = min(query_factor, corpus_factor)

    if effort == 0:
        # No search happened. Zero confidence, not full confidence — this is the direction the whole
        # module exists to get right.
        _log.warning(
            "novelty confidence requested with %d queries across %d corpora; reporting zero. A "
            "search that did not run found nothing for a reason that is not novelty.",
            queries_run,
            corpora_searched,
        )
        return 0

    return clamp_ppm(from_similarity * effort // PPM)


def assess_renaming(
    match: Match, *, mechanism_terms: Sequence[str], prior_terms: Sequence[str]
) -> tuple[bool, str]:
    """Whether a claimed difference is lexical rather than mechanical.

    The heuristic: if the miner's mechanism and the prior art share their *structural* terms and
    differ only in *naming*, the difference is a rename. Structural terms are the ones that describe
    what happens — verbs and relations — and naming terms are the labels on the parts.

    This returns evidence rather than a verdict on the criterion. 16.2's Originality judge decides;
    this tells it what the overlap is, so the judge is reasoning about a measurement rather than
    forming its own impression of two texts it has to hold in mind at once.

    Deliberately conservative: it reports a rename only when the overlap is near total *and* the
    claimed difference contains no mechanical vocabulary at all. A false "renaming_only" would zero
    an originality score on a real invention.
    """
    if not match.describes_same_mechanism():
        return False, (
            f"the closest match is {match.similarity_ppm / PPM:.2f} similar, below the "
            f"{SAME_MECHANISM_PPM / PPM:.2f} threshold at which renaming becomes the question"
        )

    own = {term.lower() for term in mechanism_terms if term}
    prior = {term.lower() for term in prior_terms if term}
    if not own or not prior:
        return False, "not assessable: one side has no extracted mechanism terms"

    overlap = len(own & prior) * PPM // len(own | prior)
    claimed = match.claimed_difference.lower()
    # Vocabulary that would indicate a *mechanical* difference rather than a lexical one. A claimed
    # difference containing none of these is describing new labels.
    mechanical = any(
        word in claimed
        for word in (
            "instead of",
            "rather than",
            "whereas",
            "we replace",
            "eliminates the",
            "removes the need",
            "inverts",
            "reverses",
            "decouples",
            "without requiring",
            "asynchronous",
            "synchronous",
            "bound",
            "invariant",
            "feedback",
            "amortis",
            "amortiz",
        )
    )

    if overlap >= 900_000 and not mechanical:
        return True, (
            f"the mechanism shares {overlap / PPM:.2f} of its structural terms with {match.source} "
            "and the claimed difference names no mechanical change — so the difference is in the "
            "labels rather than in what happens"
        )
    return False, (
        f"shares {overlap / PPM:.2f} of its structural terms with {match.source}; the claimed "
        f"difference {'does' if mechanical else 'does not'} describe a mechanical change"
    )


def build_report(
    *,
    idea_id: str,
    matches: Sequence[Match],
    queries: Sequence[str],
    corpora: Sequence[str],
    renaming: bool = False,
    notes: Sequence[str] = (),
) -> PriorArtReport:
    """Assemble a report, computing confidence from the search that was actually performed.

    A single constructor so `novelty_confidence_ppm` cannot be bypassed. A caller that set the
    confidence directly could report full confidence on a search that never ran, which is the one
    error this module is built to prevent.
    """
    ordered = tuple(sorted(matches, key=lambda match: -match.similarity_ppm))
    return PriorArtReport(
        idea_id=idea_id,
        nearest_matches=ordered,
        novelty_confidence_ppm=novelty_confidence_ppm(
            ordered, queries_run=len(queries), corpora_searched=len(set(corpora))
        ),
        renaming_only=renaming,
        queries=tuple(queries),
        corpora_searched=tuple(sorted(set(corpora))),
        notes=tuple(notes),
    )
