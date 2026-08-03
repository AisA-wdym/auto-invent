"""Step 3 of 7.4: the critic that reviews a generated problem before it can enter a pack.

The critic family comes from the season config, per slot. It is **not** required to differ from
the generator's, and by default it does not.

## Why not cross-family, and what that costs

architecture.md 7.2.2 originally required crossing the families — a GPT-written problem reviewed by
Claude, and the reverse — on the argument that a model shares its own generator's blind spots. The
owner has since decided the critique should not reach across model families, so the config sets
each family as its own critic.

The cost is real and worth recording rather than glossing. Measured on a live run before the
change: a weak GPT generator produced two problems whose flaw was vagueness ("fails to define what
'digital mechanism design' means operationally"), and the Claude critic named it. A same-family
critic is closer to self-review, and self-review is weakest at exactly that fault — a model finds
its own phrasing clear because it knows what it meant.

What still catches those problems:

* **The deterministic linter** (step 2), which is unaffected. It is the only step that is
  identical on every validator, and it rejects on structure rather than judgement.
* **The discrimination probe** (step 5), which is the strongest filter in the pipeline and is also
  unaffected. A vague problem fails it on condition 3 — judges scoring the same portfolio far
  apart across repeats — and that is a *measurement* of vagueness rather than an opinion about it.

So the critic is now the weaker of three filters rather than one of two. That is a defensible
place for it, and it is why this module was not simply deleted: the critic is still one model call
against four laboratory runs, and it rejects candidates the linter cannot see.

## Eight checks, reported as a list of faults rather than eight booleans

7.4 step 3 lists eight failure modes. A critic that graded on a scale would let a weak problem
through with a mediocre score and force the pipeline to pick a threshold, so the verdict is
categorical: any fault named is a rejection.

The *shape* of that verdict was corrected after measuring it. The first version asked for one
boolean per check — "true if the problem is FREE of that fault" — and both families inverted it.
The clearest evidence: a critic wrote "well-defined with clear constraints, does not reveal the
intended solution" and simultaneously flagged `requires_physical_experiment` on a **sealed-bid
auction** problem. There is no physical experiment in an auction; the model was answering "is this
fault present?" using the value the prompt had defined to mean the opposite.

That is a double negative across eight fields, and any single inversion rejects the candidate — so
the error rate compounds eightfold. Asking instead for a list of the faults actually found removes
the polarity question: a model naming `triviality` cannot mean the reverse of naming it. An empty
list means no faults, which is also the natural way to say it.

## A critic that cannot be read is not an acceptance

If the critic's reply does not parse, the candidate is **rejected**, not accepted. That is the
conservative direction and it is deliberate: the cost of wrongly rejecting a good candidate is one
more generation call, and the slot has others. The cost of wrongly accepting a bad one is a
challenge in a committed pack that every laboratory is scored on and that may not discriminate at
all — twenty of those and the day's ranking is noise.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from protocol.receipts import Purpose
from validator.challenge_factory.generator import Candidate
from validator.model_client import ModelClient

__all__ = ["CHECKS", "CriticVerdict", "review", "review_all"]

_log = logging.getLogger(__name__)

#: The eight checks of 7.4 step 3, each with what a failure means. The wording goes into the
#: prompt, so the critic is told what each check is for rather than only its name — a critic asked
#: about "ambiguity" with no further guidance flags stylistic imprecision, which is not the
#: property that matters.
CHECKS: Mapping[str, str] = {
    "ambiguity": (
        "Could two competent laboratories read this problem as asking for different things? Not "
        "'is the prose imprecise' — could they disagree about what a valid answer even is?"
    ),
    "internal_contradiction": (
        "Do any two constraints, or a constraint and the objective, make each other unsatisfiable?"
    ),
    "triviality": (
        "Would every competent laboratory produce essentially the same answer? If you can name "
        "the answer from the problem statement, it is trivial."
    ),
    "answer_leakage": (
        "Does the problem, its baseline, or its constraints reveal the intended solution? A "
        "constraint that only one mechanism can satisfy has named that mechanism."
    ),
    "unavailable_private_data": (
        "Does answering require data the laboratory cannot obtain — internal logs, a proprietary "
        "dataset, unpublished measurements?"
    ),
    "requires_physical_experiment": (
        "Does answering require building or measuring something physical?"
    ),
    "unevaluable_relevance": (
        "Could a judge tell a relevant answer from an irrelevant one? A problem whose success "
        "criteria cannot be stated cannot be scored."
    ),
    "resembles_recent": (
        "Does this closely resemble a well-known benchmark problem or a standard textbook "
        "exercise? Such a problem rewards recall over invention."
    ),
}

_SYSTEM = """\
You review candidate research problems for an autonomous invention benchmark. You did not write \
this problem and you are not being asked to improve it — you decide whether it is fit to score \
laboratories against.

Be strict. A flawed problem that gets used is worse than a good problem that gets discarded: \
every laboratory in the cohort is scored on it, and a problem that fails to discriminate turns \
a whole day's ranking into noise. Rejecting costs one more generation call.

Return one JSON object and nothing else.
"""

_USER = """\
Review this candidate problem.

## The candidate

{candidate}

## The faults to look for

{checks}

## Required JSON shape

{{
  "faults": [],
  "reasoning": "one or two sentences on the most serious fault, or on why the problem is sound",
  "accept": true
}}

`faults` lists the name of every fault you actually found, taken from this list exactly:

{names}

An empty list means you found none of them. Do not list a fault you did not find, and do not omit
one you did.

Set `accept` to false if `faults` is non-empty, true if it is empty.
"""


@dataclass(frozen=True, slots=True)
class CriticVerdict:
    """One critic's decision on one candidate."""

    accepted: bool
    #: The faults the critic named, from `CHECKS`. Empty means it found none.
    faults: tuple[str, ...]
    reasoning: str
    critic_family: str
    rcc: int
    #: Faults the critic named that are not in `CHECKS`. Kept rather than dropped — see `_verdict`.
    unrecognised: tuple[str, ...] = ()

    def failed_checks(self) -> tuple[str, ...]:
        return self.faults

    def reason(self) -> str:
        if not self.faults:
            return self.reasoning
        return f"found {', '.join(self.faults)}: {self.reasoning}"


async def review(client: ModelClient, candidate: Candidate) -> CriticVerdict:
    """Review one candidate with the critic family its slot declares.

    Read off the slot rather than taken as an argument, so the critic for a given slot is fixed by
    the seeded plan and the season config. A caller cannot choose it per candidate — which would
    let a validator retry a rejected candidate against a more permissive reviewer.
    """
    import json

    critic_family = candidate.slot.critic_family
    checks = "\n".join(f"- **{name}**: {question}" for name, question in CHECKS.items())
    names = ", ".join(sorted(CHECKS))

    try:
        reply = await client.ask(
            family=critic_family,
            purpose=Purpose.CRITIQUE,
            system=_SYSTEM,
            user=_USER.format(
                candidate=json.dumps(dict(candidate.body), indent=2, sort_keys=True),
                checks=checks,
                names=names,
            ),
            max_tokens=2_048,
        )
    except Exception as error:  # noqa: BLE001 - a critic failure is a rejection, not a crash
        _log.warning(
            "slot %d: critic %s failed (%s); candidate rejected",
            candidate.slot.index,
            critic_family,
            error,
        )
        return CriticVerdict(
            accepted=False,
            faults=(),
            reasoning=(
                f"the critic could not be read ({error}). Rejected rather than accepted: the "
                "slot has other candidates, and an unreviewed problem in a committed pack is "
                "scored against every laboratory in the cohort."
            ),
            critic_family=critic_family,
            rcc=0,
        )

    return _verdict(reply.parsed, critic_family, reply.rcc, candidate)


def _verdict(
    parsed: Any, critic_family: str, rcc: int, candidate: Candidate
) -> CriticVerdict:
    """Read a critic's reply, treating anything unreadable as a rejection."""
    if not isinstance(parsed, Mapping):
        return CriticVerdict(
            accepted=False,
            faults=(),
            reasoning=f"critic returned {type(parsed).__name__} rather than an object",
            critic_family=critic_family,
            rcc=rcc,
        )

    reasoning = parsed.get("reasoning")
    raw_faults = parsed.get("faults")
    if raw_faults is None or isinstance(raw_faults, str) or not isinstance(raw_faults, Sequence):
        # A reply with no fault list is unreadable, not an acceptance. Reading a missing list as
        # "found nothing" would let any malformed reply accept a candidate, and the reply shape is
        # the one thing we can check without trusting the critic's judgement.
        return CriticVerdict(
            accepted=False,
            faults=(),
            reasoning=(
                f"critic named no `faults` list ({raw_faults!r}); a missing list is unreadable "
                "rather than empty, and reading it as 'found nothing' would let a malformed reply "
                "accept anything"
            ),
            critic_family=critic_family,
            rcc=rcc,
        )

    named = [str(entry).strip().lower().replace(" ", "_") for entry in raw_faults if entry]
    faults = tuple(sorted({name for name in named if name in CHECKS}))
    unrecognised = tuple(sorted({name for name in named if name not in CHECKS}))
    if unrecognised:
        # Counted as faults even though we cannot map them. A critic that found something real and
        # named it in its own words has still found something real, and discarding it would turn a
        # genuine objection into an acceptance.
        _log.info(
            "slot %d: critic %s named unrecognised fault(s) %s; counted as faults",
            candidate.slot.index,
            critic_family,
            list(unrecognised),
        )

    stated = parsed.get("accept")
    clean = not faults and not unrecognised
    # The critic's own `accept` and its fault list can disagree — a model that names a fault and
    # then accepts anyway is common. Both must agree to accept, which resolves every disagreement
    # toward rejection: the slot has other candidates, and a bad problem in a committed pack is
    # scored against the whole cohort.
    accepted = bool(stated) and clean
    if bool(stated) and not clean:
        _log.info(
            "slot %d: critic %s set accept=true while naming %s; rejected",
            candidate.slot.index,
            critic_family,
            list(faults + unrecognised),
        )
    elif not stated and clean:
        _log.info(
            "slot %d: critic %s named no faults but set accept=false; rejected on its own verdict",
            candidate.slot.index,
            critic_family,
        )

    return CriticVerdict(
        accepted=accepted,
        faults=faults,
        reasoning=str(reasoning) if isinstance(reasoning, str) else "",
        critic_family=critic_family,
        rcc=rcc,
        unrecognised=unrecognised,
    )


async def review_all(
    client: ModelClient, candidates: Sequence[Candidate]
) -> list[tuple[Candidate, CriticVerdict]]:
    """Review many candidates concurrently, pairing each with its verdict."""
    import asyncio

    verdicts = await asyncio.gather(
        *(review(client, candidate) for candidate in candidates), return_exceptions=False
    )
    return list(zip(candidates, verdicts, strict=True))
