"""Reference A — Frontier Single Agent. architecture.md 25.

**This is the qualification floor, not a demo.** 20.1 makes it the bar every miner must exceed: if
no laboratory beats direct frontier-model use, the subnet has not shown that competing architectures
add value, and 20.4 burns the emission rather than paying for the appearance of competition.

Two consequences follow, and both shape this file.

**It has to be genuinely good.** A deliberately weak reference would make the floor easy and the
subnet would pay for laboratories that beat nothing. So this uses the best available model, a
carefully constructed single prompt, and the full portfolio structure 9.2 asks for. What it does
*not* do is orchestrate: one model, one call chain, no islands, no critics, no evolution. That is
the point — it measures what a frontier model does when asked well, and a miner earns emission by
beating it with architecture.

**It is also the scaffold `ail-miner init` writes.** A miner's first laboratory is this one, so it
has to run on the first try: a scaffold that fails is indistinguishable from a broken environment,
and a miner cannot debug the difference.

## Why the prompt is structured rather than a single instruction

The naive version — "invent five things, here is the problem" — produces five variations on the
first idea a model thinks of, because the model conditions each subsequent idea on the ones before.
So this asks for *divergence first*: candidate directions, then selection, then depth on the
survivors. That is one model doing what a multi-agent laboratory does with separate agents, and it
is deliberately the strongest single-agent form — the floor should be hard to clear by architecture
alone if the architecture adds nothing.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any

__all__ = ["SCAFFOLD", "build_prompt", "run"]

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

## How to work

First diverge, then select, then deepen. Do not write the first idea you have five times.

1. Consider at least ten distinct mechanisms that could address this. Vary what you change: the \
data structure, the protocol, the failure model, the place a decision is made, the thing being \
traded off.
2. Discard the ones that violate a constraint or take a forbidden shortcut.
3. Keep the five that differ most from each other, not the five you like best. Two strong ideas \
that share a mechanism are worth less here than one strong and one different.
4. For each, work out the mechanism as a causal chain, find the nearest prior art you know of and \
state what is genuinely different, then name the assumption you are least sure of and the cheapest \
test that would kill the idea if it were wrong.
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
                    "source": "the closest thing you know of",
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


def build_prompt(challenge: dict[str, Any]) -> str:
    """The single structured request. Deterministic given a challenge."""
    return _TEMPLATE.format(
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


def run() -> int:
    """The laboratory's entry point, inside the container.

    Reads 9.1's standard input, calls the RCG with the session token, writes 9.2's portfolio to
    `/output`. No credential: the token authorises the RCG to spend on this run's behalf and
    authorises nothing else (5.4.1).
    """
    challenge_path = os.environ.get("AIL_CHALLENGE_PATH", "/input/challenge.json")
    output_dir = os.environ.get("AIL_OUTPUT_DIR", "/output")
    endpoint = os.environ.get("AIL_RCG_ENDPOINT", "")
    token = os.environ.get("AIL_SESSION_TOKEN", "")
    model = os.environ.get("AIL_MODEL_SLUG", "anthropic/claude-sonnet-5")

    if not endpoint or not token:
        print("AIL_RCG_ENDPOINT and AIL_SESSION_TOKEN are required", file=sys.stderr)
        return 2

    with open(challenge_path) as handle:
        standard_input = json.load(handle)
    challenge = standard_input["challenge"]

    request = json.dumps(
        {
            "challenge_id": challenge["challenge_id"],
            "purpose": "research",
            "model_slug": model,
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": build_prompt(challenge)},
            ],
            "max_tokens": 16_384,
            "temperature": 0.7,
            "response_format": {"type": "json_object"},
        }
    ).encode()

    try:
        response = urllib.request.urlopen(  # noqa: S310 - endpoint is the injected RCG
            urllib.request.Request(
                f"{endpoint.rstrip('/')}/v1/llm",
                data=request,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
            ),
            timeout=900,
        )
        body = json.load(response)
    except urllib.error.HTTPError as error:
        # A 429 is the budget ceiling, which is a normal end to a run rather than a failure. Written
        # as an empty portfolio so the run produces a readable outcome: a crash here yields no file
        # at all, and the validator cannot tell "spent its budget" from "the container broke".
        detail = error.read().decode(errors="replace")[:400]
        print(f"RCG refused: {error.code} {detail}", file=sys.stderr)
        return 1
    except (urllib.error.URLError, TimeoutError) as error:
        print(f"could not reach the RCG at {endpoint}: {error}", file=sys.stderr)
        return 1

    try:
        portfolio = json.loads(body["content"])
    except (json.JSONDecodeError, KeyError) as error:
        print(f"the model did not return a portfolio: {error}", file=sys.stderr)
        return 1

    # The challenge id is overwritten rather than trusted from the model. A model that copied it
    # wrongly would produce a portfolio the validator cannot attribute to a challenge, and the
    # laboratory knows the correct value.
    portfolio["challenge_id"] = challenge["challenge_id"]

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "portfolio.json"), "w") as handle:
        json.dump(portfolio, handle, indent=2, sort_keys=True)
    print(f"wrote {len(portfolio.get('portfolio', []))} ideas", file=sys.stderr)
    return 0


#: What `ail-miner init` writes. A laboratory that runs on the first invocation, because a miner
#: cannot distinguish a broken scaffold from a broken environment.
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
    "src/lab.py": (
        '"""Reference A, vendored into the scaffold so it runs without installing this repo.\n\n'
        "Copied rather than imported on purpose: a miner's bundle must be self-contained, because\n"
        "6.1 fixes it at the deadline and an import of a moving package is a dependency that can\n"
        "change after submission.\n\n"
        "Edit freely. Beating this laboratory is what earns emission (20.1) — a bundle that only\n"
        "runs it unchanged scores at the qualification floor and is paid nothing.\n"
        '"""\n\n'
        "# The implementation is `miner/reference/template.py` in the subnet repository.\n"
        "# `ail-miner init` writes a working copy here.\n"
        "from __future__ import annotations\n\n"
        "import json\nimport os\nimport sys\nimport urllib.error\nimport urllib.request\n\n"
        f"_SYSTEM = {_SYSTEM!r}\n\n"
        f"_TEMPLATE = {_TEMPLATE!r}\n\n"
        f"_SCHEMA_EXAMPLE = {_SCHEMA_EXAMPLE!r}\n\n"
        "\n".join(
            line
            for line in (
                "def build_prompt(challenge):",
                "    return _TEMPLATE.format(",
                '        problem_statement=challenge.get("problem_statement", ""),',
                '        research_objective=challenge.get("research_objective", ""),',
                '        current_baseline=challenge.get("current_baseline", "not stated"),',
                '        known_attempts="\\n".join(f"- {e}" for e in challenge.get("known_attempts", [])) or "- none stated",',  # noqa: E501 - embedded scaffold content; wrapping changes the file a miner receives
                '        constraints="\\n".join(f"- {e}" for e in challenge.get("constraints", [])),',  # noqa: E501 - embedded scaffold content; wrapping changes the file a miner receives
                '        forbidden_shortcuts="\\n".join(f"- {e}" for e in challenge.get("forbidden_shortcuts", [])) or "- none stated",',  # noqa: E501 - embedded scaffold content; wrapping changes the file a miner receives
                "        schema=json.dumps(_SCHEMA_EXAMPLE, indent=2),",
                "    )",
            )
        )
        + "\n\n\n"
        + "\n".join(
            (
                "def run():",
                '    challenge_path = os.environ.get("AIL_CHALLENGE_PATH", "/input/challenge.json")',  # noqa: E501 - embedded scaffold content; wrapping changes the file a miner receives
                '    output_dir = os.environ.get("AIL_OUTPUT_DIR", "/output")',
                '    endpoint = os.environ["AIL_RCG_ENDPOINT"]',
                '    token = os.environ["AIL_SESSION_TOKEN"]',
                '    model = os.environ.get("AIL_MODEL_SLUG", "anthropic/claude-sonnet-5")',
                "",
                "    with open(challenge_path) as handle:",
                '        challenge = json.load(handle)["challenge"]',
                "",
                "    request = json.dumps({",
                '        "challenge_id": challenge["challenge_id"],',
                '        "purpose": "research",',
                '        "model_slug": model,',
                '        "messages": [',
                '            {"role": "system", "content": _SYSTEM},',
                '            {"role": "user", "content": build_prompt(challenge)},',
                "        ],",
                '        "max_tokens": 16384,',
                '        "temperature": 0.7,',
                '        "response_format": {"type": "json_object"},',
                "    }).encode()",
                "",
                "    try:",
                "        response = urllib.request.urlopen(",
                "            urllib.request.Request(",
                '                f"{endpoint.rstrip(chr(47))}/v1/llm",',
                "                data=request,",
                "                headers={",
                '                    "Authorization": f"Bearer {token}",',
                '                    "Content-Type": "application/json",',
                "                },",
                "            ),",
                "            timeout=900,",
                "        )",
                "        body = json.load(response)",
                "    except urllib.error.HTTPError as error:",
                '        print(f"RCG refused: {error.code}", file=sys.stderr)',
                "        return 1",
                "",
                '    portfolio = json.loads(body["content"])',
                '    portfolio["challenge_id"] = challenge["challenge_id"]',
                "    os.makedirs(output_dir, exist_ok=True)",
                '    with open(os.path.join(output_dir, "portfolio.json"), "w") as handle:',
                "        json.dump(portfolio, handle, indent=2, sort_keys=True)",
                "    return 0",
            )
        )
        + "\n"
    ),
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
