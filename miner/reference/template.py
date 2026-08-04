"""The reference laboratory: a four-stage pipeline. architecture.md 25.

**This is the qualification floor, not a demo.** 20.1 makes it the bar every miner must exceed: if
no laboratory beats it, the subnet has not shown that competing architectures add value, and 20.4
burns the emission rather than paying for the appearance of competition.

Two consequences follow, and both shape this file.

**It has to be genuinely good.** A deliberately weak reference would make the floor easy and the
subnet would pay for laboratories that beat nothing.

**It is also the scaffold `ail-miner init` writes.** A miner's first laboratory is this one, so it
has to run on the first try: a scaffold that fails is indistinguishable from a broken environment,
and a miner cannot debug the difference.

## The four stages, and what each is for

    survey  → prior art from the web, so novelty is claimed against something
    diverge → candidate directions, deliberately before any depth
    deepen  → the full portfolio, written with the survey and the directions in hand
    audit   → a self-critique that names faults, then one revision pass

The single-call version this replaces produced five variations on the first idea a model thinks of,
because a model conditions each idea on the ones before it. Divergence *before* depth is the whole
reason the stages are separate — and it is the cheapest architectural idea there is, which is why
the floor has to include it. A miner earns emission by beating a laboratory that already does the
obvious things, not by discovering them.

## Why it searches

`/v1/search` is metered against the same ceiling as inference (5.3), and gate 13.8 resolves every
URL a portfolio cites. A laboratory that cites without searching either fabricates or recalls, and
both fail — recall is worse, because it is confident. The survey stage exists so that "the nearest
prior art is X" is a statement about something that was actually retrieved.

## The budget governor

Every stage checks `/v1/usage` before it runs and skips itself if the remaining budget would not
cover it. That is not politeness: gate 13.6 is fatal, and a run that spends its ceiling
mid-portfolio produces no portfolio at all — which scores zero rather than scoring less. Degrading
in stages is strictly better than being cut off, and a laboratory that cannot afford the audit still
returns a portfolio written with a survey behind it.

Stages are skipped from the *back*, because the portfolio is the deliverable and everything else
exists to improve it.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import urllib.error
import urllib.request
from typing import Any

__all__ = ["SCAFFOLD", "build_prompt", "run", "scaffold"]

#: Output ceiling for the portfolio call. See the comment at the request site for the measurement
#: that set it.
_MAXIMUM_OUTPUT_TOKENS = 40_000

#: The system prompt. States what a portfolio is *for*, because a model told only the output schema
#: fills the fields without understanding which ones carry the judgement.
_SYSTEM = """\
You are an invention laboratory. You receive one research problem and return five ranked \
inventions.

You are judged on eight criteria, and knowing them changes what you should write:

- mechanism: a causal chain where each step follows from the last. A description of what the idea \
achieves is not a mechanism.
- originality: materially different from prior art, not renamed or recombined. Name the nearest \
prior work and say what is actually different about yours.
- value: who benefits, by how much, and why that magnitude is plausible.
- constraint_fit: an answer to *this* problem under *these* constraints, not a good idea in general.
- diversity: five genuinely different directions. Five variations on one idea scores as one idea.
- self_selection: rank your strongest first. You are scored on whether the ranking is right.
- falsifiability: a prediction whose failure would change what you believe, and the cheapest test \
that could produce that failure.
- cost_reliability: a concrete next step with a defensible cost.

Write plainly. Presentation is stripped before judging: emphasis, branding and unsupported \
percentages are removed or marked unverified, so confident phrasing earns nothing and plainness \
costs nothing.

Return one JSON object matching the schema given. No prose outside it.
"""

_TEMPLATE = """\
## The problem

{problem_statement}

## Research objective

{research_objective}

## Current baseline

{current_baseline}

## Already tried

{known_attempts}

## Constraints — every one must hold

{constraints}

## Forbidden shortcuts — these do not count as answers

{forbidden_shortcuts}

## Prior art, retrieved for this problem

{prior_art}

## Candidate directions already proposed

{directions}

## How to work

The divergence above was produced separately so that you are not selecting from your own first \
instinct. Use it: select from those directions rather than starting over, and add one only if it \
is mechanically unlike all of them.

1. Discard directions that violate a constraint or take a forbidden shortcut.
2. Keep the five that differ most from each other, not the five you like best. Two strong ideas \
that share a mechanism are worth less here than one strong and one different.
3. For each, work out the mechanism as a causal chain. Compare it against the retrieved prior art \
above and put its URL in `nearest_prior_art[].url`. Cite only URLs that appear above — the \
validator resolves every one, and a fabricated citation fails a hard gate for the whole portfolio. \
Where the closest prior art was named in the problem statement rather than retrieved, say so and \
leave the URL empty; an invented URL is worse than an absent one.
4. Name the assumption you are least sure of and the cheapest test that would kill the idea if it \
were wrong.
5. Rank them. Put the one you would fund first.

## Required output

{schema}
"""

#: 9.2's portfolio schema, as an example rather than a JSON Schema. A model given an example produces  # noqa: E501 - embedded scaffold content; wrapping changes the file a miner receives
#: conforming output far more reliably than one given a schema, and gate 13.2 checks the fields.
_SCHEMA_EXAMPLE = {
    "challenge_id": "copied from the challenge",
    "laboratory_summary": {
        "research_strategy": "how you searched the space",
        "search_scope": "what you considered and what you ruled out",
        "major_assumptions": ["assumptions common to the whole portfolio"],
    },
    "portfolio": [
        {
            "rank": 1,
            "title": "under 100 characters",
            "problem_reframe": "the problem as you understand it, if you reframed it",
            "core_invention": "what is new, in two or three sentences",
            "mechanism": {
                "components": ["the parts"],
                "information_flow": "what moves between them",
                "causal_explanation": "why this produces the effect, step by step",
                "feedback_loops": ["loops, if any"],
            },
            "nearest_prior_art": [
                {
                    "source": "the closest existing work",
                    # A retrieved URL, or empty. The field exists because without it the model has
                    # nowhere to put what the survey found and cites from recall instead — which is
                    # what the survey stage exists to replace, and which makes gate 13.8 pass
                    # vacuously because there is nothing for the validator to resolve.
                    "url": "the URL from the retrieved prior art above, or empty if it was named in the problem rather than retrieved",  # noqa: E501 - embedded scaffold content; wrapping changes the file a miner receives
                    "similarity": "what it shares",
                    "material_difference": "what is genuinely different — mechanically, not in name",  # noqa: E501 - embedded scaffold content; wrapping changes the file a miner receives
                }
            ],
            "why_non_obvious": "why a competent engineer would not reach this first",
            "expected_value": {
                "beneficiary": "who",
                "value_created": "what they gain",
                "magnitude_hypothesis": "how much, and why that number",
            },
            "assumptions": ["what must be true"],
            "weakest_assumption": "the one you are least sure of",
            "failure_modes": ["how it breaks"],
            "falsifiable_predictions": ["a prediction whose failure would change your mind"],
            "cheapest_kill_test": "the least expensive experiment that could kill it",
            "simulation_or_calculation": {
                "method": "what you calculated",
                "result": "what it gave",
                "artifact_refs": [],
            },
            "development_path": ["the next three steps"],
            "estimated_probability_of_value": 0.4,
            "estimated_validation_cost_rcc": 25,
        }
    ],
    "portfolio_map": {
        "idea_families": ["the distinct families in your five"],
        "differences": ["what separates them"],
    },
    "self_selection": {"why_rank_1": "why that one first", "confidence": 0.7},
    "resource_usage_claim": {"rcc": 0, "search_calls": 0, "model_calls": 0},
}


def build_prompt(challenge: dict[str, Any], *, survey: str = "", directions: str = "") -> str:
    """The portfolio request, carrying whatever the earlier stages produced.

    Both extras default to empty and both are optional by design: a run whose survey or divergence
    stage was skipped for budget still writes a portfolio, one stage shorter. Making them required
    would turn a degraded run into no run.
    """
    return _TEMPLATE.format(
        prior_art=survey.strip() or "(no prior-art survey was available for this run)",
        directions=directions.strip() or "(no candidate directions were generated for this run)",
        problem_statement=challenge.get("problem_statement", ""),
        research_objective=challenge.get("research_objective", ""),
        current_baseline=challenge.get("current_baseline", "not stated"),
        known_attempts="\n".join(
            f"- {entry}" for entry in challenge.get("known_attempts", [])
        )
        or "- none stated",
        constraints="\n".join(f"- {entry}" for entry in challenge.get("constraints", [])),
        forbidden_shortcuts="\n".join(
            f"- {entry}" for entry in challenge.get("forbidden_shortcuts", [])
        )
        or "- none stated",
        schema=json.dumps(_SCHEMA_EXAMPLE, indent=2),
    )


def _declared_model() -> tuple[str, str]:
    """The primary model from this bundle's `model_manifest.json`, as (slug, snapshot).

    The manifest is the declaration the validator checks against (5.3), so the laboratory calls what
    it declared rather than what an environment variable happens to say. `AIL_MODEL_SLUG` still
    overrides, for local experimentation — but it overrides *both* fields, so an override cannot
    produce the slug/snapshot mismatch that gate 13.4 exists to catch.
    """
    override = os.environ.get("AIL_MODEL_SLUG", "").strip()
    if override:
        return override, os.environ.get("AIL_MODEL_SNAPSHOT", override).strip()

    manifest_path = os.environ.get("AIL_MODEL_MANIFEST", "/app/model_manifest.json")
    try:
        with open(manifest_path) as handle:
            models = json.load(handle)["models"]
    except (OSError, KeyError, json.JSONDecodeError) as error:
        # No fallback slug. Guessing one would call a model this bundle never declared, which fails
        # gate 13.3 — so failing here, before spending anything, is both cheaper and clearer.
        raise SystemExit(
            f"cannot read a declared model from {manifest_path}: {error}. Every externally invoked "
            "model must be declared before submission closes (5.3), and calling an undeclared one "
            "invalidates the response under gate 13.3."
        ) from error
    primary = models[0]
    return primary["model_slug"], primary.get("model_snapshot", primary["model_slug"])


def _parse_portfolio(content: str) -> dict[str, Any] | None:
    """Read a portfolio out of a model reply, tolerantly. None if there is not one.

    Three things models reliably do to JSON, all handled: wrap it in a ```json fence, preface it
    with a sentence, and — the one that actually bit — run out of output tokens partway through,
    leaving a structurally invalid document.

    Truncation is *not* repaired. A portfolio cut off mid-idea is genuinely incomplete, and
    inventing the rest would be fabricating content the model never produced. What is recovered is
    the case where the JSON is complete but wrapped or prefaced, which is a formatting habit rather
    than a failure. The distinction matters because gate 13.2 checks required fields, and a repaired
    portfolio would be a miner's own laboratory lying to the validator on their behalf.
    """
    text = content.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if parsed is None:
        # Take the first balanced object, ignoring braces inside strings — a problem statement about
        # templating legitimately contains one.
        start = text.find("{")
        if start < 0:
            return None
        depth, in_string, escaped = 0, False, False
        for index in range(start, len(text)):
            character = text[index]
            if escaped:
                escaped = False
                continue
            if character == "\\":
                escaped = True
                continue
            if character == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(text[start : index + 1])
                    except json.JSONDecodeError:
                        return None
                    break
        else:
            # Never closed: the reply ran out of tokens. Not repaired — see the docstring.
            return None
    return parsed if isinstance(parsed, dict) else None


#: Wall-clock seconds for one RCG call. The episode ceiling is 1,800s and the pipeline makes four
#: calls plus searches, so no single call may own the whole budget — a survey that hangs must not
#: cost the portfolio.
_CALL_TIMEOUT = 420

#: Reserve kept for the portfolio call. Stages are skipped from the back when the remaining budget
#: falls below what the next one plus this reserve would cost, because the portfolio is the
#: deliverable and gate 13.1 is fatal without one.
_PORTFOLIO_RESERVE_RCC = 120_000

#: Rough cost of each stage, for the governor. Deliberately over-estimates: skipping a stage that
#: would have fitted costs some quality, and attempting one that does not fit costs the run.
_STAGE_COST_RCC = {"survey": 40_000, "diverge": 40_000, "audit": 90_000}

_SURVEY_SYSTEM = """\
You are a research librarian. You are given a problem and you report what already exists.

Report only what you actually found in the search results. If the results do not cover something, \
say so. An invented citation is worse than an absent one: the validator resolves every URL, and a \
fabricated one fails a hard gate for the whole portfolio.
"""

_DIVERGE_SYSTEM = """\
You are a research strategist. You are given a problem and a survey of prior art, and you propose \
candidate directions of attack.

Propose directions that differ in *mechanism*, not in wording. Two directions that would fail for \
the same reason are one direction. For each, name the mechanism in one sentence and name the \
assumption that, if false, kills it.
"""

_AUDIT_SYSTEM = """\
You are an adversarial reviewer. You are given a research portfolio and you list what is wrong \
with it.

Return a list of faults found. Do not return a verdict, a score, or reassurance — a reviewer that \
reports "this looks good" has reported nothing. If a mechanism is vague, say which step is \
missing. If two ideas share a mechanism, say which two. If a citation looks recalled rather than \
retrieved, say which one.
"""


class _Rcg:
    """The laboratory's only peer. Every call is metered and receipted (3.4).

    A small class rather than free functions so the endpoint, token and challenge id are held once.
    A laboratory that had to thread them through every call would eventually thread the wrong
    challenge id, and the session token is bound to one — so the call would be refused at the
    gateway rather than silently misattributed, but only after the tokens were spent.
    """

    def __init__(self, endpoint: str, token: str, challenge_id: str) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.token = token
        self.challenge_id = challenge_id

    def _post(self, path: str, body: dict, timeout: int) -> dict:
        request = urllib.request.Request(
            f"{self.endpoint}{path}",
            data=json.dumps(body).encode(),
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return json.load(response)

    def usage(self) -> dict:
        """Spend so far, so the governor can decide. Returns zeros if it cannot be read.

        Zeros mean "no evidence of spend", which makes the governor optimistic — the wrong
        direction. It is accepted because the alternative is aborting a run over a monitoring call,
        and gate 13.6 is enforced by the gateway regardless: the ceiling is not advisory and the
        governor only decides how gracefully the run approaches it.
        """
        try:
            request = urllib.request.Request(
                f"{self.endpoint}/v1/usage",
                headers={"Authorization": f"Bearer {self.token}"},
            )
            with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
                return json.load(response)
        except (urllib.error.URLError, TimeoutError, ValueError, OSError) as error:
            print(f"usage unreadable ({error}); assuming budget remains", file=sys.stderr)
            return {"rcc_spent": 0, "rcc_remaining": 10 ** 9, "maximum_rcc": 10 ** 9}

    def ask(
        self,
        *,
        system: str,
        user: str,
        model: str,
        snapshot: str,
        max_tokens: int,
        temperature: float = 0.7,
        as_json: bool = False,
    ) -> str:
        body = {
            "challenge_id": self.challenge_id,
            "purpose": "research",
            "model_slug": model,
            "model_snapshot": snapshot,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if as_json:
            body["response_format"] = {"type": "json_object"}
        return str(self._post("/v1/llm", body, _CALL_TIMEOUT).get("content", ""))

    def search(self, *, query: str, model: str, snapshot: str, max_results: int = 8) -> dict:
        return self._post(
            "/v1/search",
            {
                "challenge_id": self.challenge_id,
                "query": query,
                "model_slug": model,
                "model_snapshot": snapshot,
                "max_results": max_results,
            },
            _CALL_TIMEOUT,
        )


def _survey(rcg: _Rcg, challenge: dict, *, model: str, snapshot: str) -> str:
    """Stage 1: what already exists, retrieved rather than recalled.

    Two searches, not one: a problem statement and a research objective are different queries, and a
    single merged query returns the union of what neither asked for. Failures are swallowed to an
    empty survey — a search outage should cost the portfolio its prior-art grounding, not its
    existence.
    """
    queries = [
        str(challenge.get("title", ""))[:200],
        str(challenge.get("research_objective", ""))[:200],
    ]
    findings = []
    for query in [item for item in queries if item.strip()]:
        try:
            found = rcg.search(query=query, model=model, snapshot=snapshot)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as error:
            print(f"search failed for {query[:60]!r}: {error}", file=sys.stderr)
            continue
        findings.append(str(found.get("content", "")))
        for result in found.get("results", [])[:8]:
            title = str(result.get("title", ""))[:160]
            url = str(result.get("url", ""))
            if url:
                findings.append(f"- {title} :: {url}")
    if not findings:
        return ""

    joined = "\n".join(findings)[:20_000]
    try:
        return rcg.ask(
            system=_SURVEY_SYSTEM,
            user=(
                f"Problem: {challenge.get('title', '')}\n\n"
                f"{challenge.get('problem_statement', '')}\n\n"
                f"Search results:\n{joined}\n\n"
                "Summarise what already exists, with the URLs you were given. Name the closest "
                "existing approach and say specifically where it falls short of the objective."
            ),
            model=model,
            snapshot=snapshot,
            max_tokens=4_000,
            temperature=0.3,
        )
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as error:
        print(f"survey synthesis failed: {error}", file=sys.stderr)
        return joined[:4_000]


def _diverge(rcg: _Rcg, challenge: dict, survey: str, *, model: str, snapshot: str) -> str:
    """Stage 2: candidate directions, before any of them is written up.

    The stage the single-call version could not have. Asking one model for five finished ideas gets
    five variations on its first one; asking for directions first and depth second is the cheapest
    architectural idea there is, and the floor should already include it.
    """
    try:
        return rcg.ask(
            system=_DIVERGE_SYSTEM,
            user=(
                f"Problem: {challenge.get('title', '')}\n\n"
                f"{challenge.get('problem_statement', '')}\n\n"
                f"Objective: {challenge.get('research_objective', '')}\n\n"
                f"Constraints: {'; '.join(str(c) for c in challenge.get('constraints', []))}\n\n"
                f"Prior art:\n{survey or '(no survey was available)'}\n\n"
                "Propose eight candidate directions. For each: the mechanism in one sentence, the "
                "assumption that kills it if false, and how it differs mechanically from the "
                "others."
            ),
            model=model,
            snapshot=snapshot,
            max_tokens=6_000,
            temperature=0.9,
        )
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as error:
        print(f"divergence failed: {error}", file=sys.stderr)
        return ""


def _audit(rcg: _Rcg, portfolio_text: str, *, model: str, snapshot: str) -> str:
    """Stage 4: faults found, as a list.

    A *list of faults*, never a boolean or a verdict. The validator's own critic was written the
    other way round once — "return true if the problem is free of that fault" — and both model
    families inverted it, marking a portfolio sound while describing what was wrong with it in the
    same breath. Asking for faults cannot be inverted: an empty list is an empty list.
    """
    try:
        return rcg.ask(
            system=_AUDIT_SYSTEM,
            user=(
                f"Portfolio:\n{portfolio_text[:40_000]}\n\n"
                "List every fault. Duplicate mechanisms, vague causal chains, unfalsifiable "
                "claims, citations that look recalled rather than retrieved, and constraints the "
                "problem stated that an idea ignores."
            ),
            model=model,
            snapshot=snapshot,
            max_tokens=4_000,
            temperature=0.2,
        )
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as error:
        print(f"audit failed: {error}", file=sys.stderr)
        return ""


def run() -> int:
    """The laboratory's entry point, inside the container.

    Reads 9.1's standard input, runs the four stages through the RCG with the session token, writes
    9.2's portfolio to `/output`. No credential: the token authorises the RCG to spend on this run's
    behalf and authorises nothing else (5.4.1).
    """
    challenge_path = os.environ.get("AIL_CHALLENGE_PATH", "/input/challenge.json")
    output_dir = os.environ.get("AIL_OUTPUT_DIR", "/output")
    endpoint = os.environ.get("AIL_RCG_ENDPOINT", "")
    token = os.environ.get("AIL_SESSION_TOKEN", "")
    # Read from the bundle's own manifest rather than hardcoded, and *both* fields are sent.
    # Sending the slug without the snapshot is what an earlier version did, and it failed gate 13.4
    # for the reference laboratory: the receipt recorded an empty revision while the manifest
    # declared one, which reads as a model-revision mismatch. A qualification floor that fails a
    # hard gate is a floor of zero, which every miner clears without doing anything.
    model, snapshot = _declared_model()

    if not endpoint or not token:
        print("AIL_RCG_ENDPOINT and AIL_SESSION_TOKEN are required", file=sys.stderr)
        return 2

    with open(challenge_path) as handle:
        standard_input = json.load(handle)
    challenge = standard_input["challenge"]
    os.makedirs(output_dir, exist_ok=True)

    rcg = _Rcg(endpoint, token, str(challenge["challenge_id"]))

    def affordable(stage: str) -> bool:
        """Whether the remaining budget covers this stage and still leaves the portfolio room.

        Checked before each stage rather than once at the start: the stages before it may have cost
        more than estimated, and a governor that decided everything up front would be deciding on a
        number that is already stale by the time it matters.
        """
        remaining = int(rcg.usage().get("rcc_remaining", 0))
        needed = _STAGE_COST_RCC[stage] + _PORTFOLIO_RESERVE_RCC
        if remaining < needed:
            print(
                f"skipping {stage}: {remaining} RCC remains and it needs {needed} with the "
                "portfolio's reserve. The portfolio is the deliverable; gate 13.1 is fatal without "
                "one and gate 13.6 is fatal for overspending, so stages are dropped from the back.",
                file=sys.stderr,
            )
        return remaining >= needed

    survey = _survey(rcg, challenge, model=model, snapshot=snapshot) if affordable("survey") else ""
    directions = (
        _diverge(rcg, challenge, survey, model=model, snapshot=snapshot)
        if affordable("diverge")
        else ""
    )

    # Stage 3. Not guarded by `affordable`: it is the deliverable, and a run that cannot afford it
    # has already failed. Attempting it and being refused at the ceiling produces a 429 with a
    # readable reason, which is better evidence than declining to try.
    try:
        content = rcg.ask(
            system=_SYSTEM,
            user=build_prompt(challenge, survey=survey, directions=directions),
            model=model,
            snapshot=snapshot,
            # 40,000, not 16,384. Measured: a five-idea portfolio against claude-sonnet-5 hit the
            # 16,384 ceiling mid-JSON at 27,277 characters — about 70% of the way through — and the
            # template then failed gate 13.1 with no output at all. A reference laboratory that
            # routinely fails 13.1 makes the qualification floor zero, which every miner clears
            # trivially. The ceiling has to fit the answer the prompt asks for.
            max_tokens=_MAXIMUM_OUTPUT_TOKENS,
            as_json=True,
        )
    except urllib.error.HTTPError as error:
        # A 429 is the budget ceiling, which is a normal end to a run rather than a crash. Reported
        # rather than raised so the run produces a readable outcome: the validator cannot otherwise
        # tell "spent its budget" from "the container broke".
        detail = error.read().decode(errors="replace")[:400]
        print(f"RCG refused: {error.code} {detail}", file=sys.stderr)
        return 1
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        print(f"could not reach the RCG at {endpoint}: {error}", file=sys.stderr)
        return 1

    # The raw reply is written before it is parsed. A parse failure otherwise discards the only
    # evidence of what went wrong, and 6.3 publishes this directory — so a miner debugging a failed
    # round can see what their model actually said rather than only that it was unreadable.
    with open(os.path.join(output_dir, "raw_reply.txt"), "w") as handle:
        handle.write(content)

    # Stage 4, and one revision. Both are skipped rather than attempted if the budget will not cover
    # them, because a revision that is cut off mid-JSON destroys a portfolio that already parsed.
    if _parse_portfolio(content) is not None and affordable("audit"):
        faults = _audit(rcg, content, model=model, snapshot=snapshot)
        if faults.strip():
            with open(os.path.join(output_dir, "audit.txt"), "w") as handle:
                handle.write(faults)
            try:
                revised = rcg.ask(
                    system=_SYSTEM,
                    user=(
                        f"Here is your portfolio:\n{content[:60_000]}\n\n"
                        f"A reviewer found these faults:\n{faults}\n\n"
                        "Return the corrected portfolio in the same JSON schema. Fix what is "
                        "listed. Where a fault is wrong, keep your version — the reviewer did not "
                        "see the problem statement. Do not add ideas; replace a duplicated "
                        "mechanism rather than appending to it."
                    ),
                    model=model,
                    snapshot=snapshot,
                    max_tokens=_MAXIMUM_OUTPUT_TOKENS,
                    temperature=0.4,
                    as_json=True,
                )
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as error:
                print(f"revision failed, keeping the audited draft: {error}", file=sys.stderr)
                revised = ""
            # Kept only if it still parses. A revision that came back truncated would otherwise
            # replace a portfolio that was fine with one that is not — the audit stage must not be
            # able to make the run worse than not running it.
            if revised and _parse_portfolio(revised) is not None:
                content = revised
                with open(os.path.join(output_dir, "raw_reply.txt"), "w") as handle:
                    handle.write(content)

    portfolio = _parse_portfolio(content)
    if portfolio is None:
        print(
            f"the model's reply ({len(content)} chars) is not a readable portfolio; "
            "the raw text is in raw_reply.txt",
            file=sys.stderr,
        )
        return 1

    # The challenge id is overwritten rather than trusted from the model. A model that copied it
    # wrongly would produce a portfolio the validator cannot attribute to a challenge, and the
    # laboratory knows the correct value.
    portfolio["challenge_id"] = challenge["challenge_id"]

    with open(os.path.join(output_dir, "portfolio.json"), "w") as handle:
        json.dump(portfolio, handle, indent=2, sort_keys=True)
    print(f"wrote {len(portfolio.get('portfolio', []))} ideas", file=sys.stderr)
    return 0


def _vendored_lab() -> str:
    """This module's source, truncated before the scaffold definition.

    Derived rather than reconstructed. Everything above `SCAFFOLD` is stdlib-only — `json`, `os`,
    `sys`, `urllib` — so a verbatim copy runs inside the miner's container with nothing installed,
    and it cannot drift from the implementation it is supposed to be a copy of.

    The alternative, which shipped once, was to rebuild the file from `repr()`'d constants and
    hand-joined function bodies. That produced a syntax error nobody saw, because the only thing
    that would have caught it is running the scaffold — which is exactly what the scaffold exists
    to guarantee works.
    """
    source = pathlib.Path(__file__).read_text()
    cut = source.index("def _vendored_lab()")
    header = (
        '"""Reference A, vendored into your bundle so it runs with nothing installed.\n\n'
        "Copied rather than imported on purpose: 6.1 fixes your bundle at the deadline, and an\n"
        "import of a moving package is a dependency that can change after submission.\n\n"
        "Edit freely. Beating this laboratory is what earns emission (20.1) — a bundle that only\n"
        "runs it unchanged scores at the qualification floor and is paid nothing.\n"
        '"""\n\n'
    )
    # Drop the original module docstring; the header above replaces it with a miner-facing one.
    body = source[source.index('"""', source.index('"""') + 3) + 3 : cut].lstrip("\n")
    return header + body


def scaffold() -> dict[str, str]:
    """Every file `ail-miner init` writes, including the vendored laboratory."""
    return {**SCAFFOLD, "src/lab.py": _vendored_lab()}


#: What `ail-miner init` writes, apart from `src/lab.py` — see `scaffold()`. A laboratory that runs
#: on the first invocation, because a miner cannot distinguish a broken scaffold from a broken
#: environment.
SCAFFOLD: dict[str, str] = {
    "manifest.json": json.dumps(
        {
            "protocol_version": "AIL-3.0",
            "bundle_id": "YOUR_HOTKEY/lab-alpha",
            "bundle_version": "0.1.0",
            "round_id": "YYYY-MM-DD",
            "entrypoint": "/app/run_lab",
            "container_digest": "sha256:REPLACE_WITH_YOUR_IMAGE_DIGEST",
            "source_archive_hash": "sha256:FILLED_BY_AIL_MINER_SEAL",
            "lockfile_hash": "sha256:FILLED_BY_AIL_MINER_SEAL",
            "sbom_hash": "sha256:FILLED_BY_AIL_MINER_SEAL",
            "license": "Apache-2.0",
            "supported_domains": [
                "software_architecture",
                "algorithms",
                "ai_agent_architecture",
            ],
            "output_schema": "research_portfolio_v1",
        },
        indent=2,
    )
    + "\n",
    "model_manifest.json": json.dumps(
        {
            "models": [
                {
                    "alias": "primary",
                    "provider": "openrouter",
                    "model_slug": "anthropic/claude-sonnet-5",
                    "model_snapshot": "anthropic/claude-sonnet-5",
                    "parameters": {"temperature": 0.7, "max_tokens": 16384},
                    "role": "idea_generation",
                }
            ],
            "routing_config_hash": "sha256:FILLED_BY_AIL_MINER_SEAL",
            "maximum_parallel_calls": 4,
        },
        indent=2,
    )
    + "\n",
    "Dockerfile": """\
# Reference A. Non-root, no credential, one dependency.
FROM python:3.12-slim

# uid 1000 because the validator runs the container as 1000:1000 (architecture.md 10). A container
# whose files are owned by root cannot write its own output directory.
RUN useradd --uid 1000 --create-home lab
USER 1000:1000
WORKDIR /app

COPY --chown=1000:1000 src/ /app/src/
COPY --chown=1000:1000 requirements.lock /app/
# The model manifest, because the laboratory reads its own declaration rather than hardcoding a
# slug — see `_declared_model`. A bundle that called a model it had not declared would fail gate
# 13.3, and one that sent a slug without its snapshot fails 13.4.
COPY --chown=1000:1000 model_manifest.json /app/

# The laboratory reaches only the RCG, so it needs no HTTP client beyond the standard library.
# Nothing is installed: fewer dependencies is fewer things that can move between submission and
# execution, and 6.1 fixes dependencies at the deadline.

ENTRYPOINT ["python", "-u", "/app/src/entrypoint.py"]
""",
    "requirements.lock": "# Reference A uses only the standard library.\n",
    "LICENSE": """\
Apache License 2.0

Replace this with the full licence text for whatever licence you choose. 12 checks that a valid
licence is present, and 6.3 publishes your source after execution closes — so the licence you pick
governs what other miners may do with your design.
""",
    "SBOM.json": json.dumps(
        {
            "bomFormat": "CycloneDX",
            "specVersion": "1.5",
            "version": 1,
            "components": [
                {"type": "library", "name": "python", "version": "3.12"},
            ],
        },
        indent=2,
    )
    + "\n",
    "src/entrypoint.py": '''\
"""Reference A's entry point. Reads the challenge, calls the RCG, writes the portfolio."""

import sys

from lab import run

if __name__ == "__main__":
    sys.exit(run())
''',
    # `src/lab.py` is not written here. It is derived from this module's own source by
    # `_vendored_lab()` — see `scaffold()`. The first version hand-assembled it from repr'd
    # constants and joined function lines, and shipped a file with a syntax error: the module
    # docstring landed inside `build_prompt`. Nothing ran it, so nothing caught it, and the
    # scaffold's whole purpose is to run on the first invocation.
    "README.md": """\
# My invention laboratory

Reference A: one frontier model, one carefully structured request. This is the **qualification
floor** (architecture.md 20.1) — a bundle that runs it unchanged scores at the floor and is paid
nothing. Emission is earned by beating it.

## Build and check

    docker build -t my-lab .
    docker images --digests my-lab          # put the digest in manifest.json
    ail-miner validate .

## Seal and submit

    ail-miner seal . --out ../sealed --spend-cap 25
    ail-miner submit ../sealed --round YYYY-MM-DD --url https://... --netuid N

## What to change

The four reference architectures in the subnet repository show what beating this looks like:
independent idea islands, a planner–researcher–critic loop, and an evolutionary lab. All of them
spend the same RCC ceiling — what differs is how they spend it.

Your credential never enters this bundle. It travels in a separate sealed envelope (5.4.1), because
the bundle itself is published after execution closes (6.3).
""",
}
