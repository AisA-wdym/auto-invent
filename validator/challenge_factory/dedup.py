"""Step 4 of 7.4: duplicate detection over a 90-day window. Pure, no store.

Two comparisons, because they catch different things and either alone is defeated:

**Mechanism fingerprints** are a shingled hash of the problem's structural content. They catch a
problem rewritten in different words about the same mechanism — which is what a generator does
naturally when it has produced a good problem before, since the underlying idea is what it
remembers.

**Embeddings** catch a problem that is semantically the same with different structural vocabulary.
Fingerprints miss that entirely: "bound the tail latency of a fan-out read" and "keep the slowest
of N parallel fetches under a deadline" share almost no tokens.

A candidate is a duplicate if *either* exceeds its threshold. Requiring both would mean a
generator only has to defeat one — and defeating fingerprints is as easy as a thesaurus.

## Why the fingerprint is not a hash of the text

Hashing the problem statement would make one changed word a different problem. So the fingerprint
is a set of hashed 4-grams over normalised content words, compared by Jaccard similarity: a
paraphrase keeps most of its n-grams, and a genuinely different problem shares few.

Normalisation drops stopwords and lowercases, but does **not** stem. Stemming would collapse
"bounding" and "bounded" — helpful — and also "optimise" and "optimal", which in this domain are
different claims: one is an operation and the other a property, and a problem about achieving
optimality is not the problem of running an optimiser.

## Integers throughout

Similarities are parts-per-million integers, like every other ratio in the subnet. The threshold
is compared as an integer, so the accept/reject boundary is exactly reproducible rather than
depending on how a float rounded — and this decision changes which problems enter a pack whose
hash goes on chain.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from protocol.fixedpoint import PPM

_log = logging.getLogger(__name__)

__all__ = [
    "DuplicateVerdict",
    "Fingerprint",
    "cosine_ppm",
    "fingerprint",
    "is_duplicate",
    "jaccard_ppm",
]

#: Words carrying no structural information about a problem. Short and deliberately not a full
#: English stopword list: this domain's vocabulary overlaps a general list in places that matter
#: ("state", "order", "set", "map" are all technical terms here), so removing a standard list
#: would erase exactly the tokens that distinguish two problems.
_NOISE = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can", "could", "do",
        "does", "for", "from", "had", "has", "have", "how", "in", "into", "is", "it", "its",
        "may", "might", "must", "no", "not", "of", "on", "or", "should", "so", "such", "than",
        "that", "the", "their", "them", "then", "there", "these", "they", "this", "those", "to",
        "was", "we", "were", "what", "when", "where", "which", "while", "why", "will", "with",
        "would", "you", "your",
    }
)

#: Shingle width. Four is wide enough that a coincidental match is unlikely and narrow enough
#: that a paraphrase still shares shingles. At two, unrelated problems in one domain collide on
#: common phrases; at eight, changing one word in five destroys every shingle containing it.
_SHINGLE = 4

_WORD = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True, slots=True)
class Fingerprint:
    """A challenge's structural signature: hashed shingles plus its domain."""

    domain: str
    shingles: frozenset[int]

    def similarity_ppm(self, other: Fingerprint) -> int:
        """Jaccard similarity in ppm, or zero across domains.

        Cross-domain comparison returns zero rather than a small number. Two problems in
        different domains are not duplicates however much vocabulary they share, and the
        stratification in 7.2 means each domain has its own small pool — so a domain-blind
        comparison would spend its budget on pairs that can never be duplicates.
        """
        if self.domain != other.domain:
            return 0
        return jaccard_ppm(self.shingles, other.shingles)


@dataclass(frozen=True, slots=True)
class DuplicateVerdict:
    """Whether a candidate duplicates something already used, and against what."""

    is_duplicate: bool
    #: The highest similarity found, in ppm.
    peak_ppm: int
    #: Which stored challenge it matched most closely.
    nearest_id: str = ""
    #: `"fingerprint"` or `"embedding"` — which comparison found it.
    detected_by: str = ""

    def reason(self) -> str:
        if not self.is_duplicate:
            return ""
        return (
            f"{self.peak_ppm / 10_000:.2f}% similar to {self.nearest_id} by {self.detected_by}"
        )


def fingerprint(body: Mapping[str, object]) -> Fingerprint:
    """The structural signature of a challenge.

    Built from the fields that describe the *problem*, not from the whole object. Including
    `resource_limits` would make two problems with the same budget look similar, and every problem
    in a season has the same budget — so every pair would score a floor of similarity and the
    threshold would have to be raised to compensate, blunting it.
    """
    parts: list[str] = []
    for key in ("title", "problem_statement", "research_objective", "current_baseline"):
        value = body.get(key)
        if isinstance(value, str):
            parts.append(value)
    for key in ("constraints", "forbidden_shortcuts", "known_attempts"):
        value = body.get(key)
        if isinstance(value, Sequence) and not isinstance(value, str | bytes):
            parts.extend(entry for entry in value if isinstance(entry, str))

    words = [word for word in _WORD.findall(" ".join(parts).lower()) if word not in _NOISE]
    return Fingerprint(
        domain=str(body.get("domain", "")),
        shingles=frozenset(_shingles(words)),
    )


def _shingles(words: Sequence[str]) -> Iterable[int]:
    """Hashed overlapping n-grams.

    Hashed to 64 bits rather than kept as tuples: a 90-day window at twenty problems a day is
    1,800 challenges, each with a few hundred shingles, and integers compare and store far more
    cheaply than tuples of strings. Collisions at 64 bits are negligible against 10^6 shingles.

    A document shorter than the shingle width yields its whole word list as one shingle rather
    than nothing — otherwise a two-word title would fingerprint as empty, and an empty fingerprint
    is similar to nothing at all, so the shortest problems would never be detected as duplicates.
    """
    if len(words) < _SHINGLE:
        if not words:
            return
        yield _hash(words)
        return
    for index in range(len(words) - _SHINGLE + 1):
        yield _hash(words[index : index + _SHINGLE])


def _hash(words: Sequence[str]) -> int:
    digest = hashlib.blake2b(" ".join(words).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def jaccard_ppm(left: frozenset[int], right: frozenset[int]) -> int:
    """Intersection over union, in ppm, floor-rounded.

    Two empty sets are *not* similar. An empty fingerprint means the text carried no structural
    content, and calling two contentless problems identical would reject the second one for
    resembling the first, when what should happen is that both fail the linter.
    """
    if not left or not right:
        return 0
    union = len(left | right)
    return len(left & right) * PPM // union


def cosine_ppm(left: Sequence[float], right: Sequence[float]) -> int:
    """Cosine similarity in ppm, clamped to [0, PPM].

    Negative similarity is clamped to zero because a negative cosine means "semantically
    opposite", which is not a degree of duplication — and a signed value would let an unrelated
    pair register as *less* than zero and pull an average down.

    Floats are unavoidable here: an embedding is what the provider returns. They stay out of
    anything hashed — the ppm integer is what the threshold compares and what gets stored.
    """
    if len(left) != len(right):
        raise ValueError(
            f"embeddings of {len(left)} and {len(right)} dimensions cannot be compared; this "
            "usually means two different embedding models, whose vectors are not commensurable"
        )
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    if left_norm == 0.0 or right_norm == 0.0:
        return 0
    similarity = dot / (left_norm * right_norm)
    return max(0, min(PPM, int(similarity * PPM)))


def is_duplicate(
    candidate: Mapping[str, object],
    *,
    history: Sequence[tuple[str, Fingerprint]],
    threshold_ppm: int,
    candidate_embedding: Sequence[float] | None = None,
    embedding_history: Sequence[tuple[str, Sequence[float]]] = (),
) -> DuplicateVerdict:
    """Compare a candidate against the window by both methods.

    Either method exceeding the threshold is a duplicate. Both are computed even after one
    matches, so `peak_ppm` reports the strongest signal — which is what an operator watches: a
    steadily rising peak across a season means the generator supply is narrowing before it starts
    producing outright duplicates.
    """
    own = fingerprint(candidate)
    peak = 0
    nearest = ""
    detector = ""

    for identifier, stored in history:
        similarity = own.similarity_ppm(stored)
        if similarity > peak:
            peak, nearest, detector = similarity, identifier, "fingerprint"

    compared = 0
    incommensurable = 0
    if candidate_embedding is not None:
        for identifier, stored_vector in embedding_history:
            try:
                similarity = cosine_ppm(candidate_embedding, stored_vector)
            except ValueError:
                # Mismatched dimensions mean a stored vector from a different embedding model. One
                # stale entry must not abort a day's dedup, so it is skipped — but *counted*.
                incommensurable += 1
                continue
            compared += 1
            if similarity > peak:
                peak, nearest, detector = similarity, identifier, "embedding"

        # The case the silent version hid: every stored vector is from an older model, so the
        # embedding comparison ran against nothing and dedup quietly became fingerprint-only. The
        # verdict would then report `detected_by="fingerprint"` with no indication that the check
        # designed to catch paraphrase-with-different-vocabulary had not run at all.
        if embedding_history and compared == 0:
            _log.error(
                "embedding dedup compared nothing: all %d stored vectors are incommensurable with "
                "the candidate's, so only the fingerprint check ran. A paraphrase with different "
                "structural vocabulary would not be caught. Re-embed the dedup window.",
                incommensurable,
            )
        elif incommensurable:
            _log.warning(
                "skipped %d of %d stored embeddings as incommensurable; %d were compared",
                incommensurable,
                incommensurable + compared,
                compared,
            )

    return DuplicateVerdict(
        is_duplicate=peak >= threshold_ppm,
        peak_ppm=peak,
        nearest_id=nearest,
        detected_by=detector,
    )
