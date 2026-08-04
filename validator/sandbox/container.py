"""The container a miner's laboratory runs in. architecture.md 10.

Every control in 10's list, expressed as `docker run` arguments, with the reason each one is there
next to it. The flag set follows what production agent subnets have converged on — the arguments
below are close to ORO's, which is a system that has run adversarial miner code for a long time
and whose comments record which flags were added after which incident.

## The one control that is not a flag

"No arbitrary internet; RCG-only outbound access" cannot be expressed as a `docker run` argument.
`--network none` gives no internet *and* no RCG, which means the laboratory cannot work; any
`--network` that reaches the RCG reaches whatever else is on that network. So egress is a property
of the **network**, created once by `ensure_network` as an internal bridge with the RCG attached
and nothing else — and `assert_egress_confined` checks it rather than assuming it.

This is the control that failed in the predecessor. The rule was written as "the sandbox may reach
the gateway", implemented as a uid match on the request, and a container could therefore reach the
gateway *as another miner* by claiming a different uid. The lesson is the shape of the fix: the
confinement is topological (there is no route) rather than authorisational (the route exists and we
check who you say you are).

## Why the image is pinned by digest and never by tag

`bundle_manifest.container_digest` is a `sha256:` digest, and `run` refuses a tag. A tag is
mutable: a miner that submitted `lab:latest` before the deadline could push different bytes after
it, and 6.1's whole point is that source, prompts and model versions cannot change after
submission closes. A digest is the content, so there is nothing to re-point.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "ContainerError",
    "Limits",
    "RunOutcome",
    "SandboxRunner",
    "assert_egress_confined",
    "docker_command",
    "ensure_network",
]

_log = logging.getLogger(__name__)

#: The network a laboratory is attached to. Internal — no route to the host's networks, no NAT to
#: the internet. The RCG is attached to it separately by the operator's compose file.
DEFAULT_NETWORK = "auto-invent-sandbox"

#: Where the runner writes the challenge and reads the portfolio. 9.1 names `/output`.
INPUT_PATH = "/input/challenge.json"
OUTPUT_DIR = "/output"


class ContainerError(RuntimeError):
    """The container could not be started, or its environment cannot be vouched for."""


@dataclass(frozen=True, slots=True)
class Limits:
    """The resource class from the season config, plus the episode's wall clock."""

    memory_bytes: int
    cpu_shares: int
    pids_limit: int
    workspace_bytes: int
    wall_time_seconds: int

    @classmethod
    def from_season(cls, season: dict, *, wall_time_seconds: int, name: str = "standard") -> Limits:
        for entry in season["resource_classes"]:
            if entry["name"] == name:
                return cls(
                    memory_bytes=int(entry["memory_bytes"]),
                    cpu_shares=int(entry["cpu_shares"]),
                    pids_limit=int(entry["pids_limit"]),
                    workspace_bytes=int(entry["workspace_bytes"]),
                    wall_time_seconds=wall_time_seconds,
                )
        raise ContainerError(
            f"no resource class named {name!r} in the season config. Every laboratory must run "
            "under the same declared limits, so falling back to a default would mean two "
            "laboratories were not asked the same question."
        )


@dataclass(frozen=True, slots=True)
class RunOutcome:
    """What a container run produced."""

    exit_code: int
    #: Tail of stderr, bounded. The whole stream is not kept — see `SandboxRunner.run`.
    stderr_tail: str
    timed_out: bool
    #: Wall seconds the container ran. Measured by the runner, not self-reported.
    duration_seconds: float


def docker_command(
    *,
    image_digest: str,
    container_name: str,
    network: str,
    limits: Limits,
    input_host_path: Path,
    output_host_path: Path,
    session_token: str,
    rcg_endpoint: str,
) -> list[str]:
    """The full `docker run` argument list, with each control's reason.

    Built as a list and returned rather than executed, so a test can assert on the exact flags. The
    controls in 10 are the difference between running adversarial code and being run by it, and a
    missing flag is invisible until it matters — so they are checked as data.
    """
    for label, path in (("input", input_host_path), ("output", output_host_path)):
        if not path.is_absolute():
            # Docker reads a relative bind source as a *volume name*, not a path, so
            # `-v var/runs/x:/input` silently means "the volume called var/runs/x" and fails with
            # `invalid volume specification`. The validator's own default workspace is `var/runs`,
            # so every container would have failed to start — found by running the miner rehearsal
            # harness, which uses this same function.
            raise ContainerError(
                f"the {label} path {path} is relative. Docker reads a relative bind source as a "
                "volume name rather than a host path, so the mount would not be the directory you "
                "meant and the container would fail to start."
            )

    if not image_digest.startswith("sha256:"):
        raise ContainerError(
            f"refusing to run {image_digest!r}: only a sha256 digest may be run, never a tag. A "
            "tag is mutable, so a miner could submit `lab:latest` before the deadline and push "
            "different bytes after it — which is exactly what 6.1 exists to prevent."
        )

    return [
        "docker",
        "run",
        "--rm",
        "--name",
        container_name,
        # --- Egress. The network is internal and holds only the RCG; see `ensure_network`. ---
        "--network",
        network,
        # No DNS beyond the network's own resolver. A laboratory that could resolve arbitrary names
        # would still have no route, but it could use DNS itself as a covert channel — every
        # lookup leaves the host, whatever the answer is.
        "--dns-opt",
        "ndots:1",
        # --- Identity. Non-root, and no way to become root. ---
        "--user",
        "1000:1000",
        "--cap-drop=ALL",
        "--security-opt",
        "no-new-privileges=true",
        # --- Filesystem. Read-only root, one writable workspace, one writable output. ---
        "--read-only",
        # `noexec` on the workspace: a laboratory may write files there, but not write a binary and
        # run it. Writing an executable is a normal thing for research code to want (a compiled
        # helper), and it is also the standard way to run something the image does not contain.
        f"--tmpfs=/workspace:rw,noexec,nosuid,size={limits.workspace_bytes}",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=67108864",  # noqa: S108 - the path is the point
        "-v",
        f"{input_host_path}:{INPUT_PATH}:ro",
        "-v",
        f"{output_host_path}:{OUTPUT_DIR}:rw",
        # --- Resources. ---
        "--memory",
        str(limits.memory_bytes),
        # Swap equal to memory means *no* swap. Without this, a laboratory that exceeded its memory
        # limit would swap instead of being killed, and would run far slower while consuming host
        # I/O that every other laboratory on the box shares — so one miner's leak degrades
        # everyone's measured wall time.
        "--memory-swap",
        str(limits.memory_bytes),
        "--cpu-shares",
        str(limits.cpu_shares),
        "--pids-limit",
        str(limits.pids_limit),
        # File descriptors. A laboratory at high concurrency holds several sockets per worker, and
        # the default 1024 silently caps throughput below the declared parallelism — which would
        # make the limit a laboratory's *concurrency choice* rather than the declared one.
        "--ulimit",
        "nofile=4096:4096",
        # --- Logging. Bounded, because stdout is miner-controlled. ---
        # A laboratory in a tight print loop can emit gigabytes. The output that matters is the
        # portfolio in /output and the stderr tail we keep ourselves; the daemon's log sink only
        # exists for live debugging, so it is capped rather than trusted.
        "--log-driver",
        "json-file",
        "--log-opt",
        "max-size=10m",
        "--log-opt",
        "max-file=1",
        # --- What the laboratory is told. A token, never a credential. ---
        "-e",
        f"AIL_SESSION_TOKEN={session_token}",
        "-e",
        f"AIL_RCG_ENDPOINT={rcg_endpoint}",
        "-e",
        f"AIL_CHALLENGE_PATH={INPUT_PATH}",
        "-e",
        f"AIL_OUTPUT_DIR={OUTPUT_DIR}",
        image_digest,
    ]


def ensure_network(name: str = DEFAULT_NETWORK) -> None:
    """Create the sandbox network as internal, or verify an existing one is internal.

    `--internal` is the control. It removes the NAT rule that would otherwise give containers on
    the bridge a route to the internet, so a laboratory has no outbound path except to another
    container on the same network — which is the RCG and nothing else.

    An existing network that is *not* internal is a hard failure rather than something to repair.
    Repairing it would mean deleting a network other containers may be attached to, and a
    validator that silently recreated its sandbox network could disconnect the RCG mid-round.
    """
    if shutil.which("docker") is None:
        raise ContainerError("docker is not on PATH; the validator cannot run laboratories")

    inspect = subprocess.run(  # noqa: S603 - fixed argv, no shell
        ["docker", "network", "inspect", name, "--format", "{{.Internal}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if inspect.returncode == 0:
        if inspect.stdout.strip() != "true":
            raise ContainerError(
                f"the network {name!r} exists but is not internal, so every laboratory attached to "
                "it has a route to the internet and could make model calls outside the meter. "
                f"Remove it (`docker network rm {name}`) and let the validator recreate it, after "
                "checking what else is attached."
            )
        return

    created = subprocess.run(  # noqa: S603
        ["docker", "network", "create", "--internal", name],
        capture_output=True,
        text=True,
        check=False,
    )
    if created.returncode != 0:
        raise ContainerError(f"could not create the internal network {name!r}: {created.stderr}")
    _log.info("created internal sandbox network %s", name)


def assert_egress_confined(network: str = DEFAULT_NETWORK) -> None:
    """Check that the sandbox network holds only the RCG, and is internal.

    Checked rather than assumed, because this is the control whose failure is invisible. A
    laboratory with an unintended route does not error — it succeeds, off the meter, and the only
    symptom is a receipt that does not reconcile.

    The predecessor's version of this rule was authorisational: the gateway checked which miner a
    request claimed to be from. A container could therefore reach the gateway as *another* miner by
    claiming a different uid. The fix is topological, and this is the check that it holds.
    """
    inspect = subprocess.run(  # noqa: S603
        [
            "docker",
            "network",
            "inspect",
            network,
            "--format",
            "{{.Internal}} {{range $k, $v := .Containers}}{{$v.Name}} {{end}}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if inspect.returncode != 0:
        raise ContainerError(
            f"cannot inspect the sandbox network {network!r}: {inspect.stderr.strip()}. Running a "
            "laboratory on a network we cannot describe would mean running it with unknown egress."
        )

    parts = inspect.stdout.split()
    if not parts or parts[0] != "true":
        raise ContainerError(
            f"the sandbox network {network!r} is not internal: laboratories attached to it can "
            "reach the internet directly and spend outside the meter"
        )

    attached = [name for name in parts[1:] if not name.startswith("ail-lab-")]
    unexpected = [name for name in attached if "rcg" not in name and "gateway" not in name]
    if unexpected:
        # Not fatal — an operator may legitimately attach a proxy or a log shipper — but it is the
        # one configuration that silently widens what a laboratory can reach.
        _log.warning(
            "the sandbox network %s has %d non-RCG container(s) attached: %s. A laboratory can "
            "reach each of them. Verify that none of them has outbound access.",
            network,
            len(unexpected),
            unexpected,
        )


@dataclass
class SandboxRunner:
    """Runs one laboratory container under 10's controls.

    `subprocess` rather than the Docker SDK: the SDK's `containers.run` hides the argument list, and
    the argument list is the security boundary. Building argv explicitly means `docker_command` can
    be asserted on flag by flag, which is the only way a missing control gets caught.
    """

    network: str = DEFAULT_NETWORK
    #: Bytes of stderr to keep. Bounded because stderr is miner-controlled and this string is
    #: stored, logged and published (22).
    stderr_bytes: int = 16_384
    #: Grace period after the wall clock before SIGKILL. `docker run` is attached, so killing the
    #: client leaves the container running on the daemon — hence the explicit `docker kill`.
    kill_grace_seconds: float = 10.0

    def run(
        self,
        *,
        image_digest: str,
        run_id: str,
        limits: Limits,
        input_host_path: Path,
        output_host_path: Path,
        session_token: str,
        rcg_endpoint: str,
    ) -> RunOutcome:
        """Run to completion, to the wall clock, or to a kill. Never raises on miner failure.

        A laboratory that crashes, hangs or returns nothing is an ordinary outcome that must be
        scored (as a hard-gate failure), not an exception that stops the round. The only exceptions
        raised here are about *our* environment: no docker, no network.
        """
        import time

        container_name = f"ail-lab-{run_id}"
        argv = docker_command(
            image_digest=image_digest,
            container_name=container_name,
            network=self.network,
            limits=limits,
            input_host_path=input_host_path,
            output_host_path=output_host_path,
            session_token=session_token,
            rcg_endpoint=rcg_endpoint,
        )

        started = time.monotonic()
        timed_out = False
        try:
            completed = subprocess.run(  # noqa: S603 - argv is built above, no shell
                argv,
                capture_output=True,
                timeout=limits.wall_time_seconds,
                check=False,
            )
            exit_code = completed.returncode
            stderr = completed.stderr or b""
        except subprocess.TimeoutExpired as expired:
            timed_out = True
            exit_code = 124
            stderr = expired.stderr or b""
            # The attached client is dead; the container is not. Killing by name is what actually
            # stops it — without this, a hung laboratory keeps its memory and CPU for the rest of
            # the round and degrades every laboratory measured after it.
            self._kill(container_name)

        duration = time.monotonic() - started
        return RunOutcome(
            exit_code=exit_code,
            stderr_tail=stderr[-self.stderr_bytes :].decode("utf-8", errors="replace"),
            timed_out=timed_out,
            duration_seconds=duration,
        )

    def _kill(self, container_name: str) -> None:
        killed = subprocess.run(  # noqa: S603
            ["docker", "kill", container_name],
            capture_output=True,
            text=True,
            timeout=self.kill_grace_seconds,
            check=False,
        )
        if killed.returncode != 0:
            # Already gone is the common case and is fine. Anything else is an orphan holding
            # resources, which the operator needs to know about.
            if "No such container" not in killed.stderr:
                _log.error(
                    "could not kill %s after its wall clock expired: %s. It may still be running "
                    "and consuming resources that every later laboratory shares.",
                    container_name,
                    killed.stderr.strip(),
                )
        else:
            _log.info("killed %s after exceeding its wall clock", container_name)


#: Fields a laboratory must never receive. Asserted by `tests/adversarial/`, and listed here so the
#: environment block above can be checked against something rather than eyeballed.
FORBIDDEN_ENVIRONMENT = frozenset(
    {
        "AI_VALIDATOR_OPENROUTER_KEY",
        "AI_TEST_MINER_OPENROUTER_KEY",
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "AI_RUNNER_SECRET",
        "AIL_REDIS_URL",
    }
)


def environment_of(argv: list[str]) -> dict[str, str]:
    """The `-e` pairs in a command, for checking what a container is told."""
    found: dict[str, str] = {}
    for index, token in enumerate(argv):
        if token == "-e" and index + 1 < len(argv):
            key, _, value = argv[index + 1].partition("=")
            found[key] = value
    return found
