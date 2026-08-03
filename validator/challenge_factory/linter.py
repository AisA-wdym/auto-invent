"""Step 2 of 7.4: the deterministic linter. No model, no clock, no network.

Eight requirements from 7.4 step 2. Every one is checked mechanically, and every rejection names
which requirement failed — because the generator gets told, and "invalid" teaches it nothing
while "no falsifiable constraint" teaches it what to write next time.

## Deterministic on purpose, and this is the only step that is

Steps 3 and 5 use models, so two validators running them get different answers. This step does
not: the same candidate is accepted or rejected identically by every validator, on every host,
forever. That matters for 27's asymmetry — cross-validator rank correlation may be as low as
0.60 because divergence is expected, but same-bundle rerun correlation must be 0.80 or above.
A non-deterministic linter would put noise into the *pack*, which is upstream of every score, and
noise upstream of everything cannot be separated from signal anywhere downstream.

So this module is in `tools/check_purity.py`. No clock, no RNG, no network.

## Why heuristics rather than a model for "needs invention"

Requirement eight — "a meaningful need for invention rather than simple factual retrieval" — is
genuinely a judgement call, and a model would judge it better. It is checked here anyway, with
crude signals, because the *model* check happens next in step 3 and this step exists to be cheap
and certain. A candidate that fails a crude check would fail the model check too, and failing it
here costs no tokens.

The crude checks are deliberately conservative: they reject only what is unambiguously
retrieval-shaped ("what is the time complexity of quicksort"), and let everything arguable
through to the critic. A linter that rejected borderline candidates would silently narrow the
problem space toward whatever its heuristics happened to like.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

__all__ = [
    "LintResult",
    "Requirement",
    "lint",
]


class Requirement:
    """The eight requirements of 7.4 step 2, as stable identifiers.

    Strings rather than an enum so a rejection reason can be stored, published (22 publishes
    generation outcomes) and compared across protocol versions without an enum import.
    """

    PROBLEM = "clearly stated problem"
    OBJECTIVE = "defined research objective"
    CONSTRAINTS = "explicit constraints"
    OUTPUT = "expected output structure"
    CONTEXT = "sufficient technical context"
    SCOPE = "feasible research scope"
    BUDGET = "fixed budget"
    INVENTION = "needs invention rather than retrieval"


#: Minimum characters for the two long-form fields. Deliberately low: the purpose is to catch a
#: generator that emitted a placeholder or a single clause, not to enforce an essay. A high floor
#: would reward padding, and a padded problem statement is worse than a terse one.
_MINIMUM_PROBLEM = 200
_MINIMUM_OBJECTIVE = 60

#: Wording that makes a problem answerable by looking something up. Matched as whole phrases with
#: word boundaries so "what is the time complexity" is caught while "understand what is at stake
#: when complexity grows" is not.
_RETRIEVAL_PHRASES = (
    r"what is the (?:time|space) complexity of",
    r"who (?:invented|created|first published)",
    r"when was .{0,40} (?:published|released|invented)",
    r"list the (?:steps|stages) (?:of|in) the .{0,40} algorithm",
    r"define the term",
    r"what does .{0,30} stand for",
    r"summari[sz]e the .{0,40} paper",
)

#: Wording that asks for something the subnet cannot evaluate — a physical measurement, a private
#: dataset, or an outcome that takes months to observe. 7.4 step 3 gives these to the critic, but
#: the phrasings below are unambiguous enough to reject without spending a token.
_UNEVALUABLE_PHRASES = (
    r"\bin the (?:lab|laboratory)\b",
    r"\bwet[- ]lab\b",
    r"\bphysical (?:prototype|measurement|apparatus|rig)\b",
    r"\bclinical trial\b",
    r"\bour internal (?:dataset|logs|telemetry)\b",
    r"\bproprietary dataset\b",
    r"\bover the next (?:six months|year|quarter)\b",
)

_RETRIEVAL = tuple(re.compile(pattern, re.IGNORECASE) for pattern in _RETRIEVAL_PHRASES)
_UNEVALUABLE = tuple(re.compile(pattern, re.IGNORECASE) for pattern in _UNEVALUABLE_PHRASES)

#: Words that make a constraint checkable. A "constraint" reading "should be efficient" is not a
#: constraint; one reading "must run in under 200ms" is.
_MEASURABLE = re.compile(
    r"\d|\bmust\b|\bmust not\b|\bat most\b|\bat least\b|\bwithin\b|\bno more than\b"
    r"|\bexactly\b|\bcannot\b|\bwithout\b|\bbounded\b|\blimited to\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class LintResult:
    """Whether a candidate passes, and every requirement it failed.

    *Every* failure, not the first. A generator told one problem at a time makes one fix at a
    time, and each fix costs another candidate; told all four, it can fix all four.
    """

    accepted: bool
    failures: tuple[str, ...] = ()
    notes: tuple[str, ...] = field(default=())

    def reason(self) -> str:
        return "; ".join(self.failures) if self.failures else ""


def lint(
    candidate: Mapping[str, object], *, excluded_domains: frozenset[str] = frozenset()
) -> LintResult:
    """Check one candidate against 7.4 step 2. Deterministic and total.

    Never raises on a malformed candidate. A generator returning something that is not a
    challenge at all is the *ordinary* case this filters, and an exception there would abort a
    day's generation over one bad completion.
    """
    failures: list[str] = []
    notes: list[str] = []

    statement = _text(candidate, "problem_statement")
    objective = _text(candidate, "research_objective")
    constraints = _list(candidate, "constraints")
    forbidden = _list(candidate, "forbidden_shortcuts")
    baseline = _text(candidate, "current_baseline")
    attempts = _list(candidate, "known_attempts")
    required = candidate.get("required_output")
    limits = candidate.get("resource_limits")
    domain = _text(candidate, "domain")
    title = _text(candidate, "title")

    # 1. A clearly stated problem.
    if len(statement) < _MINIMUM_PROBLEM:
        failures.append(
            f"{Requirement.PROBLEM}: problem_statement is {len(statement)} characters, under "
            f"{_MINIMUM_PROBLEM}. A laboratory cannot invent against a problem it must guess at."
        )
    if not title:
        failures.append(f"{Requirement.PROBLEM}: no title")

    # 2. A defined research objective, distinct from the statement.
    if len(objective) < _MINIMUM_OBJECTIVE:
        failures.append(
            f"{Requirement.OBJECTIVE}: research_objective is {len(objective)} characters, under "
            f"{_MINIMUM_OBJECTIVE}"
        )
    elif objective.strip() == statement.strip():
        # A generator that copied the statement into the objective has stated the problem twice
        # and the goal never, which reads as complete and is not.
        failures.append(
            f"{Requirement.OBJECTIVE}: research_objective repeats problem_statement verbatim, so "
            "the problem is stated twice and the goal never"
        )

    # 3. Explicit constraints, and at least one that can be checked.
    if len(constraints) < 2:
        failures.append(
            f"{Requirement.CONSTRAINTS}: {len(constraints)} constraint(s). Without constraints "
            "every idea fits, so constraint_fit cannot discriminate between portfolios."
        )
    elif not any(_MEASURABLE.search(constraint) for constraint in constraints):
        failures.append(
            f"{Requirement.CONSTRAINTS}: no constraint is checkable — none contains a number, a "
            "bound, or a prohibition. 'Should be efficient' cannot be scored for fit."
        )
    if not forbidden:
        # 8's `forbidden_shortcuts` is what stops the obvious non-answer counting as an answer.
        failures.append(
            f"{Requirement.CONSTRAINTS}: no forbidden_shortcuts. Without them the cheapest "
            "restatement of the baseline is a valid submission."
        )

    # 4. Expected output structure.
    if not isinstance(required, Mapping):
        failures.append(f"{Requirement.OUTPUT}: required_output is missing or not an object")
    else:
        size = required.get("portfolio_size")
        if not isinstance(size, int) or size < 1:
            failures.append(
                f"{Requirement.OUTPUT}: required_output.portfolio_size is {size!r}; 9.2 expects a "
                "ranked Top-5 and the rank weights in 18.1 are defined over five positions"
            )
        for flag in (
            "ranked",
            "mechanism_required",
            "prior_art_comparison_required",
            "falsification_plan_required",
        ):
            if flag not in required:
                failures.append(f"{Requirement.OUTPUT}: required_output.{flag} is not declared")

    # 5. Sufficient technical context.
    if not baseline:
        failures.append(
            f"{Requirement.CONTEXT}: no current_baseline. Originality is judged against what "
            "already exists, and a challenge that does not say what exists cannot support that."
        )
    if not attempts:
        notes.append(
            "no known_attempts: permitted, but the critic should check that prior art is "
            "discoverable, since a laboratory cannot be marked down for missing what is not there"
        )

    # 6. A feasible research scope.
    unevaluable = [
        pattern.pattern
        for pattern in _UNEVALUABLE
        for text in (statement, objective, " ".join(constraints))
        if pattern.search(text)
    ]
    if unevaluable:
        failures.append(
            f"{Requirement.SCOPE}: requires something the subnet cannot evaluate "
            f"({unevaluable[0]}) — a physical measurement, private data, or a months-long outcome"
        )
    if domain and domain in excluded_domains:
        failures.append(
            f"{Requirement.SCOPE}: domain {domain!r} is excluded from V1 scoring by 2"
        )

    # 7. A fixed budget.
    if not isinstance(limits, Mapping):
        failures.append(f"{Requirement.BUDGET}: resource_limits is missing or not an object")
    else:
        for key in ("maximum_wall_time_seconds", "maximum_rcc", "maximum_search_calls"):
            value = limits.get(key)
            if not isinstance(value, int) or value < 1:
                failures.append(
                    f"{Requirement.BUDGET}: resource_limits.{key} is {value!r}. An unbounded "
                    "ceiling means two laboratories were not asked the same question."
                )

    # 8. Needs invention rather than retrieval.
    retrieval = [
        pattern.pattern
        for pattern in _RETRIEVAL
        for text in (title, statement, objective)
        if pattern.search(text)
    ]
    if retrieval:
        failures.append(
            f"{Requirement.INVENTION}: answerable by retrieval ({retrieval[0]}). 1 rewards "
            "invention, and a problem a search answers rewards search."
        )

    return LintResult(accepted=not failures, failures=tuple(failures), notes=tuple(notes))


def _text(candidate: Mapping[str, object], key: str) -> str:
    value = candidate.get(key)
    return value.strip() if isinstance(value, str) else ""


def _list(candidate: Mapping[str, object], key: str) -> tuple[str, ...]:
    """String entries of a list field, ignoring anything that is not a string.

    A generator that returned `constraints: "must be fast"` — a bare string rather than a list —
    would otherwise iterate character by character and report sixteen constraints, every one a
    single letter, and pass the count check.
    """
    value = candidate.get(key)
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return ()
    return tuple(entry.strip() for entry in value if isinstance(entry, str) and entry.strip())
