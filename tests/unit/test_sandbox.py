"""The sandbox: architecture.md 9 and 10.

The controls in 10 are the difference between running adversarial code and being run by it, and a
missing flag is invisible until it matters. So the argument list is asserted on as data — which is
the reason `docker_command` returns argv instead of executing it.
"""

from __future__ import annotations

import json

import pytest

from validator.sandbox.container import (
    FORBIDDEN_ENVIRONMENT,
    ContainerError,
    Limits,
    docker_command,
    environment_of,
)
from validator.sandbox.runner import (
    MAXIMUM_OUTPUT_BYTES,
    PORTFOLIO_FILENAME,
    _read_portfolio,
    _request_ceiling,
    standard_input,
)

pytestmark = pytest.mark.determinism

SEASON = json.loads(__import__("pathlib").Path("config/season.example.json").read_text())
DIGEST = "sha256:" + "a" * 64


def limits() -> Limits:
    return Limits.from_season(SEASON, wall_time_seconds=1_800)


def argv(tmp_path, **over) -> list[str]:
    kwargs = dict(
        image_digest=DIGEST,
        container_name="ail-lab-run-1",
        network="auto-invent-sandbox",
        limits=limits(),
        input_host_path=tmp_path / "challenge.json",
        output_host_path=tmp_path / "output",
        session_token="a.signed.token",
        rcg_endpoint="http://rcg:8081",
    )
    kwargs.update(over)
    return docker_command(**kwargs)


# --------------------------------------------------------------------------
# Every control in 10, asserted as a flag
# --------------------------------------------------------------------------


def test_the_container_runs_as_a_non_root_user(tmp_path):
    command = argv(tmp_path)
    assert "--user" in command
    assert command[command.index("--user") + 1] == "1000:1000"


def test_every_capability_is_dropped(tmp_path):
    assert "--cap-drop=ALL" in argv(tmp_path)


def test_privilege_escalation_is_disabled(tmp_path):
    command = argv(tmp_path)
    assert "no-new-privileges=true" in command


def test_the_root_filesystem_is_read_only(tmp_path):
    assert "--read-only" in argv(tmp_path)


def test_the_workspace_is_writable_but_not_executable(tmp_path):
    """A laboratory may write files; it may not write a binary and run it. Writing an executable
    is a normal thing for research code to want, and also the standard way to run something the
    image does not contain."""
    tmpfs = [flag for flag in argv(tmp_path) if flag.startswith("--tmpfs=/workspace")]
    assert tmpfs, "no writable workspace"
    assert "noexec" in tmpfs[0]
    assert "nosuid" in tmpfs[0]


def test_the_workspace_is_size_bounded(tmp_path):
    tmpfs = next(flag for flag in argv(tmp_path) if flag.startswith("--tmpfs=/workspace"))
    assert f"size={limits().workspace_bytes}" in tmpfs


def test_memory_is_capped(tmp_path):
    command = argv(tmp_path)
    assert command[command.index("--memory") + 1] == str(limits().memory_bytes)


def test_swap_equals_memory_so_there_is_no_swap(tmp_path):
    """Without this a laboratory that exceeded its memory limit would swap rather than be killed,
    consuming host I/O that every other laboratory shares — so one miner's leak degrades
    everyone's measured wall time."""
    command = argv(tmp_path)
    assert command[command.index("--memory-swap") + 1] == command[command.index("--memory") + 1]


def test_process_count_is_capped(tmp_path):
    command = argv(tmp_path)
    assert command[command.index("--pids-limit") + 1] == str(limits().pids_limit)


def test_file_descriptors_are_raised_above_the_default(tmp_path):
    """The default 1024 silently caps throughput below the declared parallelism, which would make
    the limit a laboratory's concurrency choice rather than the declared one."""
    command = argv(tmp_path)
    assert "nofile=4096:4096" in command


def test_the_log_sink_is_bounded(tmp_path):
    """A laboratory in a tight print loop can emit gigabytes, and stdout is miner-controlled."""
    command = argv(tmp_path)
    assert "max-size=10m" in command


def test_the_container_is_attached_to_the_sandbox_network(tmp_path):
    command = argv(tmp_path)
    assert command[command.index("--network") + 1] == "auto-invent-sandbox"


def test_the_container_is_removed_after_the_run(tmp_path):
    assert "--rm" in argv(tmp_path)


def test_the_container_is_named_so_a_hung_run_can_be_killed(tmp_path):
    """`docker run` is attached, so killing the client leaves the container on the daemon."""
    command = argv(tmp_path)
    assert command[command.index("--name") + 1] == "ail-lab-run-1"


def test_the_challenge_is_mounted_read_only(tmp_path):
    mounts = [
        argv(tmp_path)[index + 1]
        for index, flag in enumerate(argv(tmp_path))
        if flag == "-v"
    ]
    challenge_mount = next(mount for mount in mounts if "challenge.json" in mount)
    assert challenge_mount.endswith(":ro")


def test_the_output_directory_is_writable(tmp_path):
    mounts = [
        argv(tmp_path)[index + 1]
        for index, flag in enumerate(argv(tmp_path))
        if flag == "-v"
    ]
    assert any(mount.endswith("/output:rw") for mount in mounts)


# --------------------------------------------------------------------------
# The image is pinned by digest, never by tag
# --------------------------------------------------------------------------


def test_a_tagged_image_is_refused(tmp_path):
    """A tag is mutable: a miner could submit `lab:latest` before the deadline and push different
    bytes after it, which is exactly what 6.1 exists to prevent."""
    with pytest.raises(ContainerError, match="never a tag"):
        argv(tmp_path, image_digest="miner/lab:latest")


def test_a_bare_name_is_refused(tmp_path):
    with pytest.raises(ContainerError, match="never a tag"):
        argv(tmp_path, image_digest="miner/lab")


def test_a_digest_is_accepted(tmp_path):
    assert DIGEST in argv(tmp_path)


# --------------------------------------------------------------------------
# The laboratory receives a token and never a credential
# --------------------------------------------------------------------------


def test_the_container_receives_a_session_token(tmp_path):
    assert environment_of(argv(tmp_path))["AIL_SESSION_TOKEN"] == "a.signed.token"


def test_the_container_receives_no_credential(tmp_path):
    """5.4.1: "the key is never written into the sandbox's filesystem, environment, or any
    response the laboratory can read"."""
    environment = environment_of(argv(tmp_path))
    assert not FORBIDDEN_ENVIRONMENT & set(environment)


def test_no_environment_value_looks_like_a_provider_key(tmp_path):
    """Checked on values as well as names, because the name is the easy half."""
    for value in environment_of(argv(tmp_path)).values():
        assert "sk-or" not in value
        assert "sk-ant" not in value


def test_the_container_is_not_told_where_redis_is(tmp_path):
    """7.5: a laboratory that could reach the store could read every problem in the pack."""
    environment = environment_of(argv(tmp_path))
    assert not any("REDIS" in name for name in environment)
    assert not any("redis" in value for value in environment.values())


def test_the_container_is_not_told_the_runner_secret(tmp_path):
    """It guards the routes that open and close a run; a laboratory with it could reset its own
    spend."""
    assert "AI_RUNNER_SECRET" not in environment_of(argv(tmp_path))


# --------------------------------------------------------------------------
# Resource classes come from the season, never from a default
# --------------------------------------------------------------------------


def test_limits_come_from_the_season_config():
    parsed = limits()
    declared = SEASON["resource_classes"][0]
    assert parsed.memory_bytes == declared["memory_bytes"]
    assert parsed.pids_limit == declared["pids_limit"]


def test_an_unknown_resource_class_is_refused_rather_than_defaulted():
    """Falling back to a default would mean two laboratories were not asked the same question."""
    with pytest.raises(ContainerError, match="same declared limits"):
        Limits.from_season(SEASON, wall_time_seconds=1_800, name="enormous")


# --------------------------------------------------------------------------
# 9.1: the standard input
# --------------------------------------------------------------------------


def test_the_standard_input_matches_9_1():
    body = standard_input(
        challenge={"challenge_id": "sha256:" + "c" * 64},
        run_id="run-1",
        deadline="2026-08-03T12:00:00Z",
        rcg_endpoint="http://rcg:8081",
    )
    assert body["type"] == "research_challenge"
    assert body["protocol_version"] == "AIL-3.0"
    assert set(body["runtime"]) == {"run_id", "deadline", "rcg_endpoint", "artifact_directory"}
    assert body["runtime"]["artifact_directory"] == "/output"


def test_the_standard_input_carries_no_conversation():
    """The v3.0 shape: one structured request, no turns, no persona. A conversational protocol
    would put the validator into the content path."""
    body = standard_input(
        challenge={}, run_id="r", deadline="d", rcg_endpoint="e"
    )
    assert not any(
        key in body for key in ("messages", "turns", "persona", "user", "conversation")
    )


# --------------------------------------------------------------------------
# 9.2: reading the portfolio, and the output limit
# --------------------------------------------------------------------------


def test_a_missing_portfolio_names_the_expected_path(tmp_path):
    """A miner reading the published outcome needs to know where the file was expected."""
    parsed, failure, claimed = _read_portfolio(tmp_path)
    assert parsed is None
    assert PORTFOLIO_FILENAME in failure
    assert claimed == {}


def test_invalid_json_is_a_gate_failure_rather_than_an_exception(tmp_path):
    (tmp_path / PORTFOLIO_FILENAME).write_text("{not json")
    parsed, failure, _ = _read_portfolio(tmp_path)
    assert parsed is None
    assert "not valid JSON" in failure


def test_a_portfolio_that_is_a_list_is_rejected(tmp_path):
    (tmp_path / PORTFOLIO_FILENAME).write_text("[]")
    parsed, failure, _ = _read_portfolio(tmp_path)
    assert parsed is None
    assert "not an object" in failure


def test_an_oversized_portfolio_is_refused_without_being_loaded(tmp_path):
    """The limit has to apply before the bytes are in memory: `read_text` on a file the laboratory
    chose the size of is an unbounded allocation in the validator."""
    (tmp_path / PORTFOLIO_FILENAME).write_bytes(b"[" + b"0," * MAXIMUM_OUTPUT_BYTES + b"0]")
    parsed, failure, _ = _read_portfolio(tmp_path)
    assert parsed is None
    assert "output limit" in failure


def test_a_valid_portfolio_is_read_with_its_usage_claim(tmp_path):
    (tmp_path / PORTFOLIO_FILENAME).write_text(
        json.dumps(
            {
                "challenge_id": "sha256:" + "c" * 64,
                "portfolio": [],
                "resource_usage_claim": {"rcc": 397, "search_calls": 84, "model_calls": 163},
            }
        )
    )
    parsed, failure, claimed = _read_portfolio(tmp_path)
    assert failure == ""
    assert parsed is not None
    assert claimed == {"rcc": 397, "search_calls": 84, "model_calls": 163}


def test_a_non_integer_usage_claim_is_dropped_rather_than_coerced(tmp_path):
    (tmp_path / PORTFOLIO_FILENAME).write_text(
        json.dumps({"portfolio": [], "resource_usage_claim": {"rcc": "lots"}})
    )
    _, _, claimed = _read_portfolio(tmp_path)
    assert claimed == {}


# --------------------------------------------------------------------------
# The request ceiling
# --------------------------------------------------------------------------


def test_a_declared_request_ceiling_is_used():
    challenge = {"resource_limits": {"maximum_rcc": 400, "maximum_requests": 77}}
    assert _request_ceiling(challenge) == 77


def test_the_derived_ceiling_scales_with_the_budget():
    """A fixed constant would bind before RCC did on a generous day and never on a tight one."""
    small = _request_ceiling({"resource_limits": {"maximum_rcc": 100}})
    large = _request_ceiling({"resource_limits": {"maximum_rcc": 4_000}})
    assert large > small


def test_the_derived_ceiling_has_a_floor():
    """A tiny budget must still permit enough calls to attempt the challenge."""
    assert _request_ceiling({"resource_limits": {"maximum_rcc": 1}}) >= 50
