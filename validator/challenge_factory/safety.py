"""The safety and prohibited-domain filter of 7.4, between the critic and dedup.

architecture.md 2 excludes seven domains from V1 scoring. They are excluded for two different
reasons, and the filter has to serve both:

**Cannot be evaluated here.** Clinical treatment, wet-lab chemistry, physical engineering
requiring measurement, and long-horizon real-world outcomes. A judge panel of language models
cannot tell a good answer from a plausible one, so the score would be noise dressed as
measurement.

**Should not be produced here.** Weapons, malware and exploits. This one is not about
evaluability — a language model can assess an exploit design perfectly well. It is that the
subnet's output is *published* (6.3), so a portfolio of five ranked exploit designs becomes a
public document with the subnet's name on it.

The second reason is why this filter is separate from the linter rather than another of its
checks. The linter's failures are advice to a generator; this one's are refusals.

## Deterministic, and deliberately over-inclusive at the boundary

Keyword matching, no model. A model would classify better, and would also be a call that can fail
open — a rate-limited safety classifier that returned "unknown" would have to be treated as
"unsafe" to be safe, which makes an outage into a zero-challenge day.

So: keywords, with context requirements to keep the false-positive rate tolerable. "Exploit" alone
is a normal English word in this domain ("exploit locality of reference"); "exploit" near
"vulnerability" or "payload" is not. Where the check does fire on something harmless, the cost is
one discarded candidate out of three to five in that slot.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

__all__ = ["SafetyVerdict", "screen"]

_log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SafetyVerdict:
    """Whether a candidate may be used, and which exclusion it fell under."""

    safe: bool
    #: The excluded domain from 2, if one applies.
    excluded_domain: str = ""
    #: The phrase that triggered it, for the audit trail 22 publishes.
    trigger: str = ""

    def reason(self) -> str:
        if self.safe:
            return ""
        return f"excluded domain {self.excluded_domain!r} (matched {self.trigger!r})"


def _pattern(*words: str) -> re.Pattern[str]:
    return re.compile(r"\b(?:" + "|".join(words) + r")\b", re.IGNORECASE)


#: Terms that alone place a candidate in an excluded domain. These are unambiguous in this
#: context: a research problem about "chemotherapy dosing" is a clinical problem whatever its
#: framing, and no software-architecture problem needs the word.
_UNAMBIGUOUS: Mapping[str, re.Pattern[str]] = {
    "clinical_or_medical_treatment": _pattern(
        "diagnosis", "diagnose", "chemotherapy", "dosage", "dosing", "patient outcomes",
        "clinical trial", "treatment protocol", "prescribe", "prescription", "triage patients",
        "medical device", "therapeutic",
    ),
    "wet_lab_chemistry_or_biology": _pattern(
        "wet lab", "in vitro", "in vivo", "assay", "reagent", "cell culture", "pcr",
        "gene editing", "crispr", "protein synthesis", "compound synthesis",
    ),
    "physical_engineering_requiring_measurement": _pattern(
        "wind tunnel", "tensile strength", "material fatigue", "thermal expansion",
        "oscilloscope", "physical prototype", "machined", "torque", "load-bearing beam",
    ),
    "legal_or_policy": _pattern(
        "statute", "legislation", "case law", "regulatory filing", "legal liability",
        "constitutional", "jurisprudence", "draft a policy", "compliance ruling",
    ),
    "broad_artistic_ideation": _pattern(
        "screenplay", "novel plot", "poem", "song lyrics", "brand slogan", "artistic style",
        "musical composition",
    ),
}

@dataclass(frozen=True, slots=True)
class _ContextualRule:
    """A term that is ordinary in this domain and only concerning in combination.

    `context_makes_it_concerning` says which way the context reads, and it is a property of the
    **rule** rather than something derived from the matched word. Deriving it from the match was a
    real defect: the inversion was keyed on the literal string `"malware"`, so a rule that also
    matched `"ransomware"` took the non-inverted branch, and "design ransomware that spreads
    across a network" passed the filter. A per-rule flag cannot fail that way.
    """

    domain: str
    trigger: re.Pattern[str]
    context: re.Pattern[str]
    #: True: the term is fine alone and concerning *with* the context ("exploit" + "buffer
    #: overflow"). False: the term is concerning alone and made fine *by* the context
    #: ("malware" + "detection").
    context_makes_it_concerning: bool


_CONTEXTUAL: Sequence[_ContextualRule] = (
    _ContextualRule(
        domain="weapons_malware_or_exploits",
        trigger=_pattern("exploit", "exploits", "exploitation", "exploiting"),
        context=_pattern(
            "vulnerability", "vulnerabilities", "payload", "shellcode", "privilege escalation",
            "remote code execution", "zero-day", "cve", "buffer overflow", "rootkit",
        ),
        # "Exploit locality of reference" is ordinary; "exploit a buffer overflow" is not.
        context_makes_it_concerning=True,
    ),
    _ContextualRule(
        domain="weapons_malware_or_exploits",
        trigger=_pattern("malware", "ransomware", "botnet", "keylogger", "rootkit", "trojan"),
        # The field's legitimate work is overwhelmingly defensive — detection, classification,
        # sandboxing — so defensive framing is what makes this fine, and its absence is what makes
        # it concerning.
        context=_pattern(
            "detect", "detection", "detecting", "classify", "classification", "defence",
            "defense", "defensive", "quarantine", "mitigate", "mitigation", "analysis",
            "analyse", "analyze", "sandbox", "forensic", "triage",
        ),
        context_makes_it_concerning=False,
    ),
    _ContextualRule(
        domain="weapons_malware_or_exploits",
        trigger=_pattern("weapon", "weapons", "munition", "warhead", "targeting system"),
        # No ordinary use in these domains, so any occurrence is concerning. Expressed as a
        # context that always matches rather than as an unambiguous term, so the rule stays next
        # to the others it belongs with.
        context=re.compile(r""),
        context_makes_it_concerning=True,
    ),
    _ContextualRule(
        domain="long_horizon_real_world_outcomes",
        trigger=_pattern("revenue", "market share", "adoption", "user growth"),
        context=_pattern(
            "over (?:the next )?(?:five|ten|\\d+) years", "by 20[3-9]\\d", "decade",
            "long[- ]term outcome",
        ),
        # Revenue as a constraint is fine; revenue measured over five years cannot be evaluated.
        context_makes_it_concerning=True,
    ),
)


def screen(
    candidate: Mapping[str, object], *, excluded_domains: frozenset[str] = frozenset()
) -> SafetyVerdict:
    """Screen one candidate against 2's exclusions. Deterministic and total.

    `excluded_domains` from the season config is checked as well as the built-in list, so an
    operator can exclude more without editing code — but never fewer, because the built-in list is
    unioned rather than replaced. A season that could shrink the exclusions could re-enable the
    domains where a published wrong answer causes harm outside the subnet.
    """
    declared = str(candidate.get("domain", ""))
    if declared and declared in excluded_domains:
        return SafetyVerdict(safe=False, excluded_domain=declared, trigger="declared domain")

    text = _searchable(candidate)

    for domain, pattern in _UNAMBIGUOUS.items():
        match = pattern.search(text)
        if match:
            _log.info("candidate screened out: %s (%r)", domain, match.group(0))
            return SafetyVerdict(safe=False, excluded_domain=domain, trigger=match.group(0))

    for rule in _CONTEXTUAL:
        match = rule.trigger.search(text)
        if not match:
            continue
        has_context = bool(rule.context.search(text))
        concerning = has_context if rule.context_makes_it_concerning else not has_context
        if concerning:
            _log.info("candidate screened out: %s (%r)", rule.domain, match.group(0))
            return SafetyVerdict(
                safe=False, excluded_domain=rule.domain, trigger=match.group(0)
            )

    return SafetyVerdict(safe=True)


def _searchable(candidate: Mapping[str, object]) -> str:
    """Every string in the candidate, flattened.

    Everything, not just the problem statement: a generator that put an excluded requirement in a
    constraint rather than in the statement would otherwise pass, and the constraint is what the
    laboratory is actually held to.
    """
    parts: list[str] = []

    def walk(value: object) -> None:
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, Mapping):
            for entry in value.values():
                walk(entry)
        elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
            for entry in value:
                walk(entry)

    walk(candidate)
    return " ".join(parts)
