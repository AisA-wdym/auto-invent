"""Running a round: every prepared laboratory against every challenge in the pack.

`Runner.execute` drives one laboratory through one challenge. This is the loop around it, and almost
all of it is about what happens when something fails, because with N laboratories and twenty
challenges there are 20N chances for one of them to.

## The failure rule, stated once

**A failure of one execution is a failed execution, not a failed round.** A container that never
starts, a gateway that refuses admission, a portfolio that will not parse — each produces an
`Execution` with a recorded failure and a gate report that says so, and the loop continues. The only
thing that ends the round is the round's own deadline.

That is not leniency. A gate failure is fatal *to that laboratory's score* (13), and the report
carries it; what must not happen is one miner's broken bundle costing every other miner their day.

## Concurrency is bounded, and the bound is the sandbox's

Executions run concurrently because twenty challenges at up to thirty minutes each is ten hours
serially and the execution window is fourteen. The bound is a semaphore rather than "all of them at
once": each execution is a container with a CPU and memory reservation, and oversubscribing the host
makes every laboratory slower — which is unfair in a way that is invisible in the results, since
wall time is gate 13.7 and a laboratory that timed out because sixteen others were running looks
exactly like one that is slow.

## The deadline is a block, and it is checked between executions

`execution_close` is a block height. Before starting each execution the loop asks whether the chain
has passed it; if so it stops and the remaining executions are recorded as not attempted. Recorded,
not silently absent: a laboratory with six results out of twenty scored as though it had twenty is
a laboratory penalised for the validator running late.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from validator.sandbox.container import Limits
from validator.sandbox.runner import LabResult, Runner, RunnerError
from validator.scoring.gates import GateReport, GateResult, check_all
from validator.submissions import Prepared

__all__ = [
    "Execution",
    "RoundExecution",
    "as_document",
    "execute_round",
    "from_document",
]

_log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Execution:
    """One laboratory's attempt at one challenge, with its gate verdicts."""

    uid: int
    hotkey: str
    challenge_id: str
    result: LabResult | None
    gates: GateReport
    #: Set when the execution never happened at all — the deadline passed, or the runner raised
    #: before a container existed. Distinct from a `LabResult` with a failure, which is an execution
    #: that ran and produced nothing.
    not_attempted: str = ""

    @property
    def valid(self) -> bool:
        return self.result is not None and self.gates.valid

    @property
    def measured_rcc(self) -> int:
        return int(self.result.measured_usage.get("rcc", 0)) if self.result else 0

    @property
    def failed_gates(self) -> tuple[str, ...]:
        return tuple(
            f"{result.gate} {result.detail}".strip()
            for result in self.gates.results
            if not result.passed
        )


@dataclass(frozen=True, slots=True)
class RoundExecution:
    """Everything a round's execution phase produced."""

    executions: tuple[Execution, ...] = ()
    #: uid -> the executions for that laboratory, in pack order.
    by_uid: Mapping[int, tuple[Execution, ...]] = field(default_factory=dict)
    stopped_at_deadline: bool = False

    def for_uid(self, uid: int) -> tuple[Execution, ...]:
        return tuple(self.by_uid.get(uid, ()))


def _deadline_report(reason: str) -> GateReport:
    """A gate report for an execution that never ran.

    An empty report would read as `valid` — `GateReport.valid` is "every result passed", and no
    results all pass vacuously. That would score a laboratory that never ran as one that passed
    every gate, which is the most expensive possible direction for this mistake to go.
    """
    return GateReport((GateResult(gate="execution", passed=False, detail=reason),))


async def execute_round(
    *,
    laboratories: Sequence[Prepared],
    challenges: Sequence[Mapping[str, Any]],
    runner: Runner,
    validator_hotkey: str,
    round_id: str,
    limits: Limits,
    allowed_models: Sequence[str],
    excluded_domains: frozenset[str],
    deadline_block: int,
    current_block: Callable[[], int],
    episode_seconds: int,
    concurrency: int = 4,
    now: Callable[[], float] = time.time,
) -> RoundExecution:
    """Run every laboratory against every challenge, bounded by concurrency and the deadline."""
    if not laboratories or not challenges:
        _log.warning(
            "round %s has %d laboratories and %d challenges; nothing to execute",
            round_id,
            len(laboratories),
            len(challenges),
        )
        return RoundExecution()

    semaphore = asyncio.Semaphore(max(1, concurrency))
    stopped = False

    async def one(lab: Prepared, challenge: Mapping[str, Any]) -> Execution:
        nonlocal stopped
        challenge_id = str(challenge.get("challenge_id", ""))
        async with semaphore:
            # Checked inside the semaphore, immediately before starting: outside it, a task could
            # wait an hour for a slot and then start a container the round no longer has room for.
            if current_block() >= deadline_block:
                stopped = True
                return Execution(
                    uid=lab.uid,
                    hotkey=lab.hotkey,
                    challenge_id=challenge_id,
                    result=None,
                    gates=_deadline_report(
                        f"not attempted: the execution window closed at block {deadline_block}"
                    ),
                    not_attempted="execution window closed",
                )
            return await _execute_one(
                lab=lab,
                challenge=challenge,
                runner=runner,
                validator_hotkey=validator_hotkey,
                round_id=round_id,
                limits=limits,
                allowed_models=allowed_models,
                excluded_domains=excluded_domains,
                episode_seconds=episode_seconds,
                now=now,
            )

    tasks = [one(lab, challenge) for lab in laboratories for challenge in challenges]
    results: list[Execution] = list(await asyncio.gather(*tasks))

    by_uid: dict[int, list[Execution]] = {}
    for execution in results:
        by_uid.setdefault(execution.uid, []).append(execution)

    _log.info(
        "round %s executed: %d of %d valid across %d laboratories%s",
        round_id,
        sum(1 for execution in results if execution.valid),
        len(results),
        len(laboratories),
        " (stopped at the deadline)" if stopped else "",
    )
    return RoundExecution(
        executions=tuple(results),
        by_uid={uid: tuple(items) for uid, items in by_uid.items()},
        stopped_at_deadline=stopped,
    )


async def _execute_one(
    *,
    lab: Prepared,
    challenge: Mapping[str, Any],
    runner: Runner,
    validator_hotkey: str,
    round_id: str,
    limits: Limits,
    allowed_models: Sequence[str],
    excluded_domains: frozenset[str],
    episode_seconds: int,
    now: Callable[[], float],
) -> Execution:
    challenge_id = str(challenge.get("challenge_id", ""))
    run_id = f"{round_id}-{lab.uid}-{challenge_id[-12:]}"
    deadline = int(now()) + episode_seconds

    try:
        result = await runner.execute(
            run_id=run_id,
            miner_hotkey=lab.hotkey,
            bundle_digest=lab.bundle_digest,
            image_digest=lab.image_digest,
            validator_hotkey=validator_hotkey,
            challenge=dict(challenge),
            api_key=lab.api_key,
            allowed_models=list(allowed_models),
            limits=limits,
            deadline=str(deadline),
            expires_at=deadline,
            episode_deadline=deadline,
            declared_spend_cap_usd=lab.declared_spend_cap_usd,
        )
    except RunnerError as error:
        # The runner raises when a run could not be set up at all — no workspace, no admission, no
        # challenge id. There is no `LabResult` to gate, so the failure is recorded as an execution
        # that did not happen rather than as a portfolio that was empty.
        _log.warning("uid %d / %s: %s", lab.uid, challenge_id, error)
        return Execution(
            uid=lab.uid,
            hotkey=lab.hotkey,
            challenge_id=challenge_id,
            result=None,
            gates=_deadline_report(f"not attempted: {error}"),
            not_attempted=str(error),
        )
    except Exception as error:  # noqa: BLE001 - one execution must not end a round
        _log.exception("uid %d / %s failed unexpectedly", lab.uid, challenge_id)
        return Execution(
            uid=lab.uid,
            hotkey=lab.hotkey,
            challenge_id=challenge_id,
            result=None,
            gates=_deadline_report(f"not attempted: {type(error).__name__}: {error}"),
            not_attempted=f"{type(error).__name__}: {error}",
        )

    gates = check_all(
        portfolio=result.portfolio,
        challenge=dict(challenge),
        receipt_calls=result.receipt_calls,
        measured_rcc=int(result.measured_usage.get("rcc", 0)),
        measured_search_calls=int(result.measured_usage.get("search_calls", 0)),
        declared_models=_declared_models(lab.manifest),
        wall_seconds=result.wall_seconds,
        timed_out=result.timed_out,
        excluded_domains=excluded_domains,
    )
    if not gates.valid:
        _log.info(
            "uid %d / %s failed %s",
            lab.uid,
            challenge_id,
            [result_.gate for result_ in gates.results if not result_.passed],
        )
    return Execution(
        uid=lab.uid,
        hotkey=lab.hotkey,
        challenge_id=challenge_id,
        result=result,
        gates=gates,
    )


def _declared_models(manifest: Mapping[str, Any]) -> dict[str, str]:
    """The bundle's locked model manifest, as gate 13.4 wants it.

    Read from the manifest rather than from anything the laboratory said at runtime. 5.3 locks the
    manifest at sealing, and a laboratory that could declare its own models at runtime could declare
    whichever ones it actually used.
    """
    declared = manifest.get("model_manifest", {})
    if isinstance(declared, Mapping):
        return {str(key): str(value) for key, value in declared.items()}
    return {}


# --------------------------------------------------------------------------
# Between two steps: executions have to survive the execution-close boundary
# --------------------------------------------------------------------------


def as_document(execution: RoundExecution, *, refused: Sequence[Any] = ()) -> dict[str, Any]:
    """The whole execution phase as a plain object, for the store.

    Scoring is a separate scheduler step that runs after the execution-close block, so what these
    containers produced has to survive the gap between them — and it cost real money to produce, so
    losing it to a restart loses the round outright.

    Portfolios are kept verbatim. They are what a judge reads, they are what 22 publishes, and a
    reduced copy would mean the published artefact and the judged artefact were different objects.
    """
    return {
        "stopped_at_deadline": execution.stopped_at_deadline,
        "refused": [
            {"uid": item.uid, "hotkey": item.hotkey, "reason": item.reason} for item in refused
        ],
        "executions": [
            {
                "uid": item.uid,
                "hotkey": item.hotkey,
                "challenge_id": item.challenge_id,
                "not_attempted": item.not_attempted,
                "gates": [
                    {"gate": verdict.gate, "passed": verdict.passed, "detail": verdict.detail}
                    for verdict in item.gates.results
                ],
                "result": None
                if item.result is None
                else {
                    "run_id": item.result.run_id,
                    "miner_hotkey": item.result.miner_hotkey,
                    "challenge_id": item.result.challenge_id,
                    "portfolio": item.result.portfolio,
                    "failure": item.result.failure,
                    "measured_usage": dict(item.result.measured_usage),
                    "receipt_calls": [dict(call) for call in item.result.receipt_calls],
                    "claimed_usage": dict(item.result.claimed_usage),
                    "chain_head": item.result.chain_head,
                    "wall_seconds": item.result.wall_seconds,
                    "timed_out": item.result.timed_out,
                    "exit_code": item.result.exit_code,
                    "stderr_tail": item.result.stderr_tail,
                },
            }
            for item in execution.executions
        ],
    }


def from_document(body: Mapping[str, Any]) -> RoundExecution:
    """Rebuild what `as_document` wrote.

    Missing fields raise rather than defaulting. A document written by a different release is a
    document this one cannot judge, and defaulting `gates` to empty would make every execution read
    as valid — `GateReport.valid` is "every result passed", and no results all pass vacuously.
    """
    executions: list[Execution] = []
    for item in body["executions"]:
        raw = item["result"]
        result = (
            None
            if raw is None
            else LabResult(
                run_id=str(raw["run_id"]),
                miner_hotkey=str(raw["miner_hotkey"]),
                challenge_id=str(raw["challenge_id"]),
                portfolio=raw["portfolio"],
                failure=str(raw.get("failure", "")),
                measured_usage={str(k): int(v) for k, v in raw["measured_usage"].items()},
                receipt_calls=tuple(dict(call) for call in raw["receipt_calls"]),
                claimed_usage={str(k): int(v) for k, v in raw.get("claimed_usage", {}).items()},
                chain_head=str(raw.get("chain_head", "")),
                wall_seconds=float(raw.get("wall_seconds", 0.0)),
                timed_out=bool(raw.get("timed_out", False)),
                exit_code=int(raw.get("exit_code", 0)),
                stderr_tail=str(raw.get("stderr_tail", "")),
            )
        )
        executions.append(
            Execution(
                uid=int(item["uid"]),
                hotkey=str(item["hotkey"]),
                challenge_id=str(item["challenge_id"]),
                result=result,
                gates=GateReport(
                    tuple(
                        GateResult(
                            gate=str(verdict["gate"]),
                            passed=bool(verdict["passed"]),
                            detail=str(verdict.get("detail", "")),
                        )
                        for verdict in item["gates"]
                    )
                ),
                not_attempted=str(item.get("not_attempted", "")),
            )
        )

    by_uid: dict[int, list[Execution]] = {}
    for execution in executions:
        by_uid.setdefault(execution.uid, []).append(execution)
    return RoundExecution(
        executions=tuple(executions),
        by_uid={uid: tuple(items) for uid, items in by_uid.items()},
        stopped_at_deadline=bool(body.get("stopped_at_deadline", False)),
    )
