"""`ail-miner run`: your laboratory, in the validator's sandbox, before you submit.

`ail-miner validate` checks everything that can be checked without running, and then prints a line
telling you to test the rest with a local run. Until now there was no local run — the miner was left
to reconstruct the container flags, the network confinement, the gateway and the session token by
hand, which is exactly the work the validator already does and exactly the work a miner should not
have to redo.

## The same sandbox, not a similar one

This uses `validator.sandbox` — the same `docker_command`, the same dropped capabilities, the same
internal network, the same read-only root — and `validator.scoring.gates.check_all`, the same
thirteen gates. A rehearsal harness that built its own environment would be a second definition of
what a run is, and the two would drift; the first a miner would hear about it is a bundle that
passed at home and failed on chain.

The one thing that differs is the gateway: this runs one in-process, on your own key, on a port
bound to the sandbox network. It meters and receipts identically because it is the same
`gateway.api` app.

## It costs you money, and it says so

Every call goes to the provider on the key you supply. A full rehearsal against twenty challenges
costs what a round costs — the whole point is that it is the same work — so the default is one
challenge and the rest is opt-in.

## What it cannot tell you

It runs your laboratory against *your* challenges. The real pack is sealed until execution closes,
generated from a seed nobody controls, and half of it comes from a generator family you did not
choose. A bundle that scores well here has demonstrated that it runs, spends within its ceiling,
finishes in time and passes every gate. It has not demonstrated that it is good, and this prints
that distinction rather than a score, because a number here would be read as a prediction.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import socket
import sys
import threading
import time
from contextlib import closing
from pathlib import Path
from typing import Any

from protocol.season import load_season

__all__ = ["rehearse", "sample_challenges"]

#: A challenge to rehearse against when the miner supplies none. Deliberately in the shape the
#: validator generates rather than a toy: the fields a laboratory reads are the fields gates 13.2,
#: 13.9 and 13.10 check against, so a sample missing them would rehearse a run that cannot fail.
_SAMPLE: dict[str, Any] = {
    "challenge_id": "sha256:" + "5a" * 32,
    "domain": "distributed_systems",
    "title": "Bounding tail latency in a fan-out read without hedging",
    "problem_statement": (
        "A coordinator issues one read to each of N replicas and must return once it has enough "
        "responses to be correct. The slowest replica sets the response time. Hedged requests "
        "reduce the tail but multiply load, and under load the hedges themselves become the cause "
        "of the tail they were added to fix."
    ),
    "research_objective": (
        "Bound the 99.9th percentile of coordinator response time without increasing steady-state "
        "request volume by more than 5%."
    ),
    "current_baseline": (
        "Request hedging after a p95 timer, which raises load by 15-30% under skew."
    ),
    "known_attempts": [
        "Tied requests with cross-replica cancellation",
        "Adaptive hedging thresholds driven by a latency histogram",
    ],
    "constraints": [
        "Steady-state request volume may rise by at most 5%",
        "No change to the replication protocol's correctness conditions",
        "Must degrade safely when a replica is partitioned rather than slow",
    ],
    "forbidden_shortcuts": [
        "Reducing the quorum size",
        "Returning stale reads without saying so",
        "Assuming replica latencies are independent",
    ],
    # Gate 13.2 checks the portfolio size against this. The sample omitted it and the gate
    # correctly refused to check — which is the gate working, and a sample that cannot be checked
    # rehearsing nothing.
    "required_output": {
        "portfolio_size": 5,
        "required_fields": [
            "mechanism",
            "why_non_obvious",
            "prior_art_comparison",
            "falsification_plan",
        ],
    },
    "resource_limits": {
        "maximum_rcc": 1_000_000,
        "maximum_search_calls": 100,
        "maximum_wall_time_seconds": 1_800,
    },
}


def sample_challenges() -> list[dict[str, Any]]:
    """The built-in rehearsal pack. One challenge, because each one costs real money."""
    return [dict(_SAMPLE)]


def _free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as probe:
        probe.bind(("0.0.0.0", 0))
        return int(probe.getsockname()[1])


def _load_challenges(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return sample_challenges()
    body = json.loads(path.read_text())
    challenges = body.get("challenges", body) if isinstance(body, dict) else body
    if not isinstance(challenges, list) or not challenges:
        raise ValueError(
            f"{path} holds no challenges. Expected a list, or an object with a `challenges` list — "
            "the shape a published round uses, so you can rehearse against a real past pack."
        )
    return [dict(entry) for entry in challenges]


def _start_gateway(*, api_key: str, port: int, runner_secret: str) -> tuple[Any, threading.Thread]:
    """A real gateway, in this process, on the miner's own key.

    The same `gateway.api` app the validator runs. Building a stub instead would mean the rehearsal
    metered differently from the round, and the number a miner tuned against would be the wrong one.
    """
    import uvicorn

    from gateway.api import GatewayState, build_app
    from gateway.credentials import CredentialSet, MinerCredential, ValidatorCredential
    from gateway.metering import PriceTable
    from gateway.tokens import TokenIssuer

    season = load_season(Path("config/season.example.json"))
    # The miner's key is admitted as a *miner* credential, and the validator slot holds the same
    # key. Not a shortcut: `CredentialSet` refuses to fund a miner purpose from the validator's
    # credential and vice versa, so a rehearsal with an empty validator slot could not be
    # constructed — and one that lied about which slot was which would rehearse the wrong
    # enforcement. Here the two accounts are genuinely the same account: the miner is paying for
    # their own rehearsal.
    credentials = CredentialSet(
        validator=ValidatorCredential(validator_hotkey="rehearsal", api_key=api_key)
    )
    credentials.admit(MinerCredential(miner_hotkey="rehearsal", api_key=api_key))
    state = GatewayState(
        credentials=credentials,
        prices=PriceTable.from_season(season),
        issuer=TokenIssuer(secret=secrets.token_bytes(32)),
        runner_secret=runner_secret,
    )
    config = uvicorn.Config(
        build_app(state), host="0.0.0.0", port=port, log_level="warning", access_log=False
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if getattr(server, "started", False):
            return server, thread
        time.sleep(0.1)
    raise RuntimeError("the rehearsal gateway did not start")


def rehearse(
    bundle: Path,
    *,
    image: str,
    api_key: str,
    challenges_path: Path | None = None,
    limit: int = 1,
    workspace: Path | None = None,
) -> int:
    """Run the bundle against challenges in the real sandbox and report every gate.

    Returns a process exit code: zero only if every rehearsed challenge produced a portfolio that
    passed all thirteen gates.
    """
    from validator.sandbox.container import Limits, SandboxRunner, ensure_network
    from validator.sandbox.runner import Runner
    from validator.scoring.gates import check_all

    manifest_path = bundle / "manifest.json"
    if not manifest_path.is_file():
        print(f"no manifest.json under {bundle}", file=sys.stderr)
        return 2
    # Parsed and discarded: what matters here is that it *is* readable JSON, because `ail-miner
    # seal` hashes it and a manifest that will not parse is a bundle that cannot be sealed.
    json.loads(manifest_path.read_text())

    challenges = _load_challenges(challenges_path)[:limit]
    season = load_season(Path("config/season.example.json"))
    workspace = workspace or Path("var/rehearsal")

    runner_secret = secrets.token_hex(16)
    port = _free_port()
    ensure_network()
    server, _thread = _start_gateway(api_key=api_key, port=port, runner_secret=runner_secret)

    # The container reaches the gateway across the sandbox bridge, so the endpoint it is given is
    # the host's address on that network rather than localhost — inside the container, localhost is
    # the container.
    endpoint = f"http://{_bridge_address()}:{port}"
    print(f"gateway on {endpoint}, {len(challenges)} challenge(s), image {image}\n")

    from gateway.client import GatewayClient

    client = GatewayClient(endpoint=f"http://127.0.0.1:{port}", runner_token=runner_secret)
    pricing = season["providers"]["miner_pricing"]
    limits = Limits.from_season(
        season, wall_time_seconds=int(pricing["maximum_wall_time_seconds"])
    )

    runner = Runner(
        sandbox=SandboxRunner(),
        admit=client.admit,
        close=client.close,
        rcg_endpoint=endpoint,
        workspace=workspace,
    )

    failures = 0
    try:
        for index, challenge in enumerate(challenges, start=1):
            challenge.setdefault("resource_limits", _SAMPLE["resource_limits"])
            print(f"[{index}/{len(challenges)}] {challenge.get('title', '')[:70]}")
            wall = int(challenge["resource_limits"]["maximum_wall_time_seconds"])
            deadline = int(time.time()) + wall
            result = asyncio.run(
                runner.execute(
                    run_id=f"rehearsal-{index}",
                    miner_hotkey="rehearsal",
                    bundle_digest="rehearsal",
                    image_digest=_image_digest(image),
                    validator_hotkey="rehearsal",
                    challenge=challenge,
                    api_key=api_key,
                    allowed_models=[str(slug) for slug in pricing["allowed_model_slugs"]],
                    limits=limits,
                    deadline=str(deadline),
                    expires_at=deadline,
                    episode_deadline=deadline,
                )
            )
            report = check_all(
                portfolio=result.portfolio,
                challenge=challenge,
                receipt_calls=result.receipt_calls,
                measured_rcc=int(result.measured_usage.get("rcc", 0)),
                measured_search_calls=int(result.measured_usage.get("search_calls", 0)),
                declared_models=_declared_models(bundle),
                wall_seconds=result.wall_seconds,
                timed_out=result.timed_out,
            )
            failures += _report(result, report, workspace / f"rehearsal-{index}")
    finally:
        server.should_exit = True

    print()
    if failures:
        print(
            f"{failures} of {len(challenges)} rehearsal(s) would not have scored.",
            file=sys.stderr,
        )
        return 1
    _print_caveat()
    return 0


def _report(result: Any, report: Any, run_directory: Path) -> int:
    """Print one rehearsal's outcome. Returns 1 if it would not have scored."""
    measured = result.measured_usage
    print(f"    portfolio      {'yes' if result.portfolio else 'NO'}")
    print(f"    wall clock     {result.wall_seconds:.0f}s")
    print(f"    RCC measured   {int(measured.get('rcc', 0)):,}")
    print(f"    search calls   {int(measured.get('search_calls', 0))}")
    print(f"    model calls    {len(result.receipt_calls)}")

    failed = [item for item in report.results if not item.passed]
    if failed:
        print(f"    gates          {len(failed)} FAILED")
        for item in failed:
            print(f"      x {item.gate} {item.detail}")
    else:
        print(f"    gates          all {len(report.results)} passed")
    if result.stderr_tail:
        # The last few lines, not the last one. Docker's final line is "Run 'docker run --help'",
        # which is the half that says nothing — the line above it is the one that says what broke.
        print("    stderr")
        for line in result.stderr_tail.strip().splitlines()[-6:]:
            print(f"      {line[:140]}")
    print(f"    output         {run_directory / 'output'}")
    print()
    return 0 if (result.portfolio and not failed) else 1


def _print_caveat() -> None:
    """What a passing rehearsal does and does not mean.

    Printed every time, because the failure mode of a harness like this is a miner reading "PASS" as
    a prediction of their score. It is not one: this ran against challenges the miner chose.
    """
    print("Every gate passed. That means it runs, stays inside its ceiling, finishes in time,")
    print("and produces a portfolio the validator can read.")
    print()
    print("It does not mean it will score. The real pack is sealed until execution closes, is")
    print("generated from a seed nobody controls, and half of it comes from a generator family")
    print("you did not choose. Rehearse against a published past round for a harder test:")
    print("  ail-miner run . --challenges <a published pack>.json")


def _bridge_address() -> str:
    """The host's address on the sandbox bridge, as seen from inside a container.

    Not `localhost`: inside the container that is the container. The gateway binds `0.0.0.0` on the
    host and the container reaches it on the bridge gateway address, which is what the validator's
    own compose file arranges with a service name.
    """
    import subprocess

    from validator.sandbox.container import DEFAULT_NETWORK as NETWORK_NAME

    result = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [
            "docker",
            "network",
            "inspect",
            NETWORK_NAME,
            "--format",
            "{{(index .IPAM.Config 0).Gateway}}",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    address = result.stdout.strip()
    if not address:
        raise RuntimeError(
            f"cannot read the gateway address of the {NETWORK_NAME} network. The sandbox network "
            "is internal and has no route out, so the rehearsal gateway has to be reachable on it."
        )
    return address


def _image_digest(image: str) -> str:
    import subprocess

    result = subprocess.run(  # noqa: S603 - fixed argv, no shell
        ["docker", "image", "inspect", "--format", "{{.Id}}", image],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"no local image {image!r}. Build it first — `docker build -t {image} .` — because the "
            "sandbox runs by digest and there is nothing to run without one."
        )
    return result.stdout.strip()


def _declared_models(bundle: Path) -> dict[str, str]:
    """The same reader the validator uses, so the rehearsal cannot disagree with the round.

    Both had the same bug and the rehearsal is what found it: 5.3's manifest is a list of model
    records, so a `dict` check returned `{}` and gate 13.3 failed for a laboratory that had declared
    exactly what it called.
    """
    from validator.execution import declared_models

    return declared_models(bundle)


def api_key_from_environment() -> str:
    """The miner's own provider key, for the rehearsal gateway to spend.

    Read from the environment rather than taken as an argument: an argument lands in shell history,
    and this is the same key `ail-miner seal` refuses to read for the same reason.
    """
    for name in ("AIL_MINER_API_KEY", "OPENROUTER_API_KEY"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    raise RuntimeError(
        "set AIL_MINER_API_KEY (or OPENROUTER_API_KEY) to the key the rehearsal should spend. It "
        "is not a command-line argument on purpose — that would put it in your shell history."
    )
