"""One laboratory, one challenge, one portfolio. architecture.md 9.

The v3.0 shape, and the thing that most distinguishes it from what came before: there is no
conversation. The laboratory receives one structured request and returns one structured Top-5
portfolio. No simulated user, no turns, no persona.

That matters beyond simplicity. A conversational protocol makes the *validator* part of the content
path — every turn it generates is an input the laboratory reacts to, so two validators asking
differently get different work, and the owner of the conversation script decides what gets rewarded.
A single structured request removes the validator from the content path entirely: it delivers a
challenge it committed to a hash of, and reads a file.

## The runner is the only thing that touches the challenge

7.5 is emphatic that Redis is the validator's store and the laboratory must never reach it. This is
where that is honoured: `prepare` writes the challenge to a file, mounted read-only, and the
container's only network peer is the RCG. The laboratory cannot ask for a challenge, cannot list
them, and cannot see which one another laboratory got.

## Measured usage replaces the claim, always

9.2 lets a laboratory report `resource_usage_claim`, and ends "Validators replace self-reported
usage with RCG-measured usage." So `finish` takes the claim, records it *as a claim*, and returns
the receipt totals as the usage. The claim is kept rather than discarded because the difference
between claim and measurement is evidence: a laboratory that consistently under-reports is either
mis-instrumented or probing what the validator checks, and both are worth seeing.

## Every failure is an outcome, not an exception

A laboratory that crashes, hangs, writes nothing, or writes something that is not a portfolio has
*failed a hard gate* (13.1, 13.2, 13.7). It has not caused an error in the validator. So `execute`
returns a `LabResult` in every case, and the only exceptions it raises are about the validator's own
environment — no docker, no gateway, no network. Getting this backwards would let one broken
laboratory stop a round for everyone.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from validator.sandbox.container import ContainerError, Limits, RunOutcome, SandboxRunner

__all__ = [
    "LabResult",
    "Runner",
    "RunnerError",
    "standard_input",
]

_log = logging.getLogger(__name__)

#: Cap on the portfolio file. 10 requires "output-size limits", and the reason is concrete: this
#: file is parsed, canonicalised, sent to judge models and published. A laboratory that wrote a
#: gigabyte of JSON would exhaust the validator's memory during parsing, before any gate could
#: reject it — so the limit is applied by reading a bounded prefix rather than by checking size
#: after loading.
MAXIMUM_OUTPUT_BYTES = 4 * 1024 * 1024

PORTFOLIO_FILENAME = "portfolio.json"


class RunnerError(RuntimeError):
    """The validator's own environment is not fit to run a laboratory."""


@dataclass(frozen=True, slots=True)
class LabResult:
    """One laboratory's attempt at one challenge.

    `portfolio` is None whenever nothing usable came back. `failure` then says why, in terms a
    hard gate can consume — the gate decides the consequence, this only reports what happened.
    """

    run_id: str
    miner_hotkey: str
    challenge_id: str
    portfolio: dict[str, Any] | None
    failure: str = ""
    #: What the RCG measured. Replaces `resource_usage_claim` per 9.2.
    measured_usage: dict[str, int] = field(default_factory=dict)
    #: Every call on the receipt chain, as `Call.link_body()` dicts. Carried here rather than left
    #: for the caller to re-fetch, because gates 13.3, 13.4, 13.5 and 13.11 are all decided from it
    #: — and a caller that had to fetch it separately could omit it. An earlier version of the
    #: validator's composition passed `receipt_calls=()`, which failed gate 13.11 for *every*
    #: laboratory (a portfolio with no inference behind it) while 13.3 and 13.5 passed vacuously.
    receipt_calls: tuple[dict[str, Any], ...] = ()
    #: What the laboratory said it used. Kept as evidence, never used as usage.
    claimed_usage: dict[str, int] = field(default_factory=dict)
    #: The receipt chain head, binding this result to its evidence.
    chain_head: str = ""
    wall_seconds: float = 0.0
    timed_out: bool = False
    exit_code: int = 0
    stderr_tail: str = ""

    @property
    def produced_output(self) -> bool:
        return self.portfolio is not None

    def usage_discrepancy(self) -> dict[str, int]:
        """Claimed minus measured, per field present in both.

        Not a gate on its own — a laboratory may count differently in good faith. It is a signal,
        and the reason the claim is kept: a consistent understatement across rounds is either
        mis-instrumentation or an experiment in what the validator checks.
        """
        return {
            key: self.claimed_usage[key] - self.measured_usage.get(key, 0)
            for key in sorted(self.claimed_usage)
            if key in self.measured_usage
        }


def standard_input(
    *, challenge: dict[str, Any], run_id: str, deadline: str, rcg_endpoint: str
) -> dict[str, Any]:
    """9.1's standard input, exactly.

    `artifact_directory` is where a laboratory writes simulation artefacts it wants to cite —
    9.2's `simulation_or_calculation.artifact_refs`. It is the same directory the portfolio goes
    in, so a citation can be checked against a file that exists rather than trusted.
    """
    return {
        "type": "research_challenge",
        "protocol_version": "AIL-3.0",
        "challenge": challenge,
        "runtime": {
            "run_id": run_id,
            "deadline": deadline,
            "rcg_endpoint": rcg_endpoint,
            "artifact_directory": "/output",
        },
    }


@dataclass
class Runner:
    """Drives one laboratory through one challenge, and returns what happened.

    Takes the gateway as a callable pair rather than importing it, so a run can be driven against a
    real RCG over HTTP or an in-process one in a test without this module knowing which.
    """

    sandbox: SandboxRunner
    #: Called with the admission body; returns the session token. The runner-authenticated route.
    admit: Any
    #: Called with the run id; returns the close response including the receipt.
    close: Any
    rcg_endpoint: str
    workspace: Path

    async def execute(
        self,
        *,
        run_id: str,
        miner_hotkey: str,
        bundle_digest: str,
        image_digest: str,
        validator_hotkey: str,
        challenge: dict[str, Any],
        api_key: str,
        allowed_models: list[str],
        limits: Limits,
        deadline: str,
        expires_at: int,
        episode_deadline: int,
        declared_spend_cap_usd: int = 0,
    ) -> LabResult:
        """Admit, run, read, close. Returns a result in every case.

        The ordering matters at one point: the run is **closed** even when the container failed. A
        run left open holds its ledger entry and its receipt for the rest of the process, and its
        outstanding reservations would be counted as leaked spend at shutdown rather than against
        the run that actually made them.
        """
        challenge_id = str(challenge.get("challenge_id", ""))
        if not challenge_id:
            raise RunnerError(
                "the challenge has no challenge_id. It is what the session token is bound to and "
                "what every receipt references, so a run without one cannot be reconciled."
            )

        run_directory = self.workspace / run_id
        input_directory = run_directory / "input"
        output_directory = run_directory / "output"
        try:
            input_directory.mkdir(parents=True, exist_ok=True)
            # World-writable output: the container runs as uid 1000 and the validator may not. A
            # directory the container cannot write to produces an empty output and a hard-gate
            # failure that looks like the miner's fault and is ours.
            output_directory.mkdir(parents=True, exist_ok=True)
            output_directory.chmod(0o777)
        except OSError as error:
            raise RunnerError(f"cannot prepare a workspace for {run_id}: {error}") from error

        challenge_file = input_directory / "challenge.json"
        challenge_file.write_text(
            json.dumps(
                standard_input(
                    challenge=challenge,
                    run_id=run_id,
                    deadline=deadline,
                    rcg_endpoint=self.rcg_endpoint,
                ),
                indent=2,
                sort_keys=True,
            )
        )

        token = await _maybe_await(
            self.admit(
                {
                    "run_id": run_id,
                    "miner_hotkey": miner_hotkey,
                    "bundle_digest": bundle_digest,
                    "validator_hotkey": validator_hotkey,
                    "challenge_id": challenge_id,
                    "api_key": api_key,
                    "allowed_models": allowed_models,
                    "maximum_rcc": int(challenge["resource_limits"]["maximum_rcc"]),
                    "maximum_requests": _request_ceiling(challenge),
                    "maximum_search_calls": int(
                        challenge["resource_limits"]["maximum_search_calls"]
                    ),
                    "expires_at": expires_at,
                    "episode_deadline": episode_deadline,
                    "declared_spend_cap_usd": declared_spend_cap_usd,
                }
            )
        )

        outcome: RunOutcome | None = None
        container_failure = ""
        try:
            outcome = self.sandbox.run(
                image_digest=image_digest,
                run_id=run_id,
                limits=limits,
                input_host_path=challenge_file,
                output_host_path=output_directory,
                session_token=token,
                rcg_endpoint=self.rcg_endpoint,
            )
        except ContainerError as error:
            # An environment failure, not a miner failure. Re-raised after closing the run, because
            # scoring a laboratory zero for the validator's broken docker would be wrong.
            container_failure = str(error)

        # Closed unconditionally, so its ledger entry and receipt do not outlive it.
        closed = await _maybe_await(self.close(run_id))
        measured, receipt_calls, chain_head = _read_close(closed, run_id)

        if container_failure:
            raise RunnerError(container_failure)
        assert outcome is not None  # noqa: S101 - either set or raised above

        portfolio, failure, claimed = _read_portfolio(output_directory)
        if outcome.timed_out:
            # 13.7. Stated even when a portfolio was written: a laboratory that wrote output and
            # then hung still exceeded its limit, and 13 makes that fatal regardless of quality.
            failure = (
                f"exceeded its wall clock of {limits.wall_time_seconds}s (gate 13.7)"
                + (f"; also: {failure}" if failure else "")
            )
            portfolio = None
        elif outcome.exit_code != 0 and portfolio is None:
            failure = failure or f"the container exited {outcome.exit_code} without a portfolio"

        return LabResult(
            run_id=run_id,
            miner_hotkey=miner_hotkey,
            challenge_id=challenge_id,
            portfolio=portfolio,
            failure=failure,
            measured_usage=measured,
            claimed_usage=claimed,
            receipt_calls=receipt_calls,
            chain_head=chain_head,
            wall_seconds=outcome.duration_seconds,
            timed_out=outcome.timed_out,
            exit_code=outcome.exit_code,
            stderr_tail=outcome.stderr_tail,
        )


def _read_close(
    closed: Any, run_id: str
) -> tuple[dict[str, int], tuple[dict[str, Any], ...], str]:
    """Read the gateway's close response, refusing a shape we cannot score against.

    Every field here decides a hard gate, so a missing one is raised rather than defaulted.
    Reading `totals` with a `{}` default would report zero spend and *pass* gate 13.6; reading
    `calls` with a `()` default would fail gate 13.11 for every laboratory. Neither direction is
    safe, so there is no default at all — a gateway that returns an unreadable close is an
    operational failure of ours, not a scoring outcome for the miner.
    """
    if not isinstance(closed, Mapping):
        raise RunnerError(
            f"run {run_id}: the gateway returned {type(closed).__name__} from close, not an "
            "object. The response decides four hard gates and cannot be guessed at."
        )
    totals = closed.get("totals")
    if not isinstance(totals, Mapping):
        raise RunnerError(
            f"run {run_id}: the close response carries no `totals`. Defaulting to zero spend "
            "would pass gate 13.6 for a laboratory whose usage we never measured."
        )
    receipt = closed.get("receipt")
    calls = receipt.get("calls") if isinstance(receipt, Mapping) else None
    if not isinstance(calls, Sequence) or isinstance(calls, str | bytes):
        raise RunnerError(
            f"run {run_id}: the close response carries no receipt call list. Defaulting to an "
            "empty list would fail gate 13.11 for every laboratory — a portfolio with no inference "
            "behind it — while passing 13.3 and 13.5 vacuously."
        )
    return (
        {key: int(value) for key, value in totals.items() if isinstance(value, int)},
        tuple(dict(call) for call in calls if isinstance(call, Mapping)),
        str(closed.get("chain_head", "")),
    )


def _request_ceiling(challenge: dict[str, Any]) -> int:
    """A request ceiling derived from the RCC ceiling when the challenge does not state one.

    8's `resource_limits` names RCC and search calls but not a request count, and the ledger needs
    one — a run bounded only on RCC could make very many very cheap calls. Derived rather than
    defaulted to a constant so it scales with the budget: a challenge with a larger RCC ceiling
    legitimately supports more calls, and a fixed constant would bind before the RCC did on a
    generous day and never on a tight one.
    """
    limits = challenge["resource_limits"]
    declared = limits.get("maximum_requests")
    if isinstance(declared, int) and declared > 0:
        return declared
    # Two requests per RCC of budget, floored at 50. Deliberately loose: this is a denial-of-service
    # bound on the gateway, not a fairness bound — RCC is what fairness is measured in.
    return max(50, int(limits["maximum_rcc"]) * 2)


def _read_portfolio(output_directory: Path) -> tuple[dict[str, Any] | None, str, dict[str, int]]:
    """Read and bound-check the portfolio. Returns (portfolio, failure, claimed usage).

    Reads a bounded prefix rather than the whole file. `Path.read_text` on a file a laboratory
    chose the size of is an unbounded allocation in the validator, and the limit has to apply
    before the bytes are in memory — checking `stat().st_size` afterwards is too late, and checking
    it before is a race a laboratory controls.
    """
    path = output_directory / PORTFOLIO_FILENAME
    if not path.is_file():
        # 13.1/13.2. The most common miner failure by far, and it needs to name the path so a
        # miner reading the published outcome knows where the file was expected.
        return None, f"no portfolio written at {PORTFOLIO_FILENAME} (gate 13.1)", {}

    try:
        with path.open("rb") as handle:
            raw = handle.read(MAXIMUM_OUTPUT_BYTES + 1)
    except OSError as error:
        return None, f"could not read the portfolio: {error}", {}

    if len(raw) > MAXIMUM_OUTPUT_BYTES:
        return (
            None,
            f"the portfolio exceeds the {MAXIMUM_OUTPUT_BYTES}-byte output limit (10). It is "
            "parsed, canonicalised, judged and published, so an unbounded file would exhaust the "
            "validator before any gate could reject it.",
            {},
        )

    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        return None, f"the portfolio is not valid JSON: {error} (gate 13.1)", {}

    if not isinstance(parsed, dict):
        return None, f"the portfolio is a {type(parsed).__name__}, not an object (gate 13.1)", {}

    claimed = parsed.get("resource_usage_claim")
    claimed_usage = (
        {key: int(value) for key, value in claimed.items() if isinstance(value, int)}
        if isinstance(claimed, dict)
        else {}
    )
    return parsed, "", claimed_usage


async def _maybe_await(value: Any) -> Any:
    """Await a coroutine, or pass a plain value through.

    So the gateway can be an in-process object with sync methods in a test and an async HTTP client
    in production, without the runner branching on which.
    """
    import inspect

    if inspect.isawaitable(value):
        return await value
    return value
