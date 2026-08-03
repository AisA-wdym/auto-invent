"""The deployment topology *is* the egress control. architecture.md 10.

`validator/sandbox/container.py` checks at runtime that the sandbox network is internal, and
`assert_egress_confined` checks what is attached to it. Both run against whatever the operator
actually deployed. This checks the file we *ship*, because a compose file that put Redis on the
sandbox network would hand every laboratory the whole challenge pack — and the runtime check warns
rather than refuses there, since a validator legitimately may run Redis on a private host.

A configuration file is not usually worth testing. This one is, because it encodes a security
property that is invisible when wrong: a laboratory with an unintended route does not error. It
succeeds, off the meter, and the only symptom is a receipt that will not reconcile.
"""

from __future__ import annotations

import pathlib

import pytest

yaml = pytest.importorskip("yaml")

COMPOSE = pathlib.Path("ops/docker-compose.yml")


@pytest.fixture(scope="module")
def compose() -> dict:
    return yaml.safe_load(COMPOSE.read_text())


def test_the_sandbox_network_is_internal(compose):
    """`internal: true` removes the NAT rule, so nothing on the bridge has a route to the internet.

    Without it a laboratory could reach OpenRouter directly, spend outside the meter, and pass every
    gate — because the receipt would simply not mention those calls.
    """
    assert compose["networks"]["sandbox"]["internal"] is True


def test_the_gateway_is_on_both_networks(compose):
    """Which is what makes it the only path out, and why it holds the credential rather than the
    laboratory: `--network none` gives a laboratory no internet *and* no gateway."""
    assert set(compose["services"]["rcg"]["networks"]) == {"sandbox", "upstream"}


def test_redis_is_not_on_the_sandbox_network(compose):
    """7.5. A laboratory that could reach the store could read every problem in the pack — including
    the ones it has not been given, and other rounds' packs."""
    assert "sandbox" not in compose["services"]["redis"]["networks"]


def test_redis_is_bound_to_loopback_only(compose):
    """The binding is the actual protection; the network split is the second layer."""
    for published in compose["services"]["redis"].get("ports", []):
        assert str(published).startswith("127.0.0.1:"), published


def test_redis_does_not_evict_under_memory_pressure(compose):
    """A pack whose hash is already committed on chain must not be evicted: it cannot be
    regenerated, because the seed's randomness has passed."""
    assert "noeviction" in compose["services"]["redis"]["command"]


def test_the_validator_is_not_on_the_sandbox_network(compose):
    """It has no reason to be reachable by a laboratory, and it holds the runner secret."""
    assert "sandbox" not in compose["services"]["validator"]["networks"]


def test_the_gateway_port_is_not_published_to_the_host(compose):
    """Publishing it would put a spending endpoint on the host. The validator reaches it by name."""
    assert "ports" not in compose["services"]["rcg"]


def test_the_openrouter_key_is_a_secret_rather_than_an_environment_value(compose):
    """A projected secret file can be permission-restricted; an environment variable is readable by
    anything that can list the process."""
    assert compose["services"]["rcg"]["secrets"] == ["openrouter"]
    assert "AI_VALIDATOR_OPENROUTER_KEY" not in compose["services"]["rcg"]["environment"]


def test_no_service_carries_a_literal_credential(compose):
    """Every secret is a reference. A literal here would be committed."""
    for name, service in compose["services"].items():
        for key, value in (service.get("environment") or {}).items():
            assert "sk-or" not in str(value), f"{name}.{key}"
            assert "sk-ant" not in str(value), f"{name}.{key}"


def test_the_secrets_directory_is_gitignored():
    """The one mistake that cannot be undone: a committed key is a published key.

    Checked on the effective rules rather than on the first line, and then confirmed against git
    itself — a `.gitignore` that looks right and does not match is the failure mode that matters.
    """
    import subprocess

    ignore = pathlib.Path("ops/secrets/.gitignore")
    assert ignore.is_file()
    rules = [
        line.strip()
        for line in ignore.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert "*" in rules, f"nothing is ignored: {rules}"
    assert "!.gitignore" in rules, "the rule file would ignore itself and never be committed"

    # git's own answer, which is the only one that counts.
    result = subprocess.run(  # noqa: S603
        ["git", "check-ignore", "ops/secrets/openrouter.key"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, "git does not ignore ops/secrets/openrouter.key"


def test_the_laboratory_image_is_never_run_by_tag_in_the_shipped_config():
    """A tag can be repointed after the deadline, which 6.1 exists to prevent. Laboratories are not
    in compose at all — the validator starts them — so this asserts they stay out of it."""
    compose_text = COMPOSE.read_text()
    assert "user_lab" not in compose_text
    assert "ail-lab" not in compose_text


def test_the_documented_entry_points_exist():
    """Every command in the docs should be runnable. A README that names a flag that does not exist
    is worse than no README, because it costs an operator a debugging session to discover."""
    import subprocess

    for module in ("validator", "gateway"):
        result = subprocess.run(  # noqa: S603
            [".venv/bin/python", "-m", module, "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, f"python -m {module} --help failed: {result.stderr[:400]}"
        assert "--check" in result.stdout, f"{module} does not offer --check"


def test_the_miner_cli_offers_every_documented_command():
    import subprocess

    result = subprocess.run(  # noqa: S603
        [".venv/bin/python", "-m", "miner.cli.main", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr[:400]
    for command in ("init", "validate", "seal", "submit"):
        assert command in result.stdout, f"ail-miner {command} is documented but absent"


@pytest.mark.parametrize(
    "document", ["README.md", "docs/miner.md", "docs/validator.md", "docs/incentive.md"]
)
def test_every_internal_documentation_link_resolves(document):
    """A broken link in a subnet's docs is how a miner concludes the project is abandoned."""
    import re

    text = pathlib.Path(document).read_text()
    base = pathlib.Path(document).parent
    for target in re.findall(r"\]\(([^)#]+)(?:#[^)]*)?\)", text):
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        resolved = (base / target).resolve()
        assert resolved.exists(), f"{document} links to {target}, which does not exist"


# --------------------------------------------------------------------------
# The two season configs, and why one of them must fail
# --------------------------------------------------------------------------


def test_the_mainnet_shaped_config_refuses_while_the_probe_is_unwired():
    """`--check` against `season.example.json` must FAIL, and that is the point.

    The example config is mainnet-shaped: it sets `allow_unprobed_packs` to false, so a validator
    cannot commit a pack that skipped 7.4 step 5 — the strongest filter in the pipeline. Until a
    probe is wired, refusing is the honest answer, and a check that passed here would be claiming
    the build can run a mainnet round.

    This test is expected to *fail* on the day a probe lands, which is when it should be deleted.
    """
    import subprocess

    result = subprocess.run(  # noqa: S603
        [
            ".venv/bin/python",
            "-m",
            "validator",
            "--check",
            "--season",
            "config/season.example.json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1, (
        "the mainnet-shaped config now passes — wire the probe check or delete this test"
    )
    assert "discrimination probe" in result.stderr


def test_the_testnet_config_passes_because_it_declares_the_degradation():
    """The difference between the two configs is a declaration, not a capability.

    A testnet standing up reference laboratories legitimately needs to generate packs before the
    probe exists. Setting the flag makes that a decision on record rather than an oversight.
    """
    import subprocess

    result = subprocess.run(  # noqa: S603
        [
            ".venv/bin/python",
            "-m",
            "validator",
            "--check",
            "--season",
            "config/season.testnet.json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr[:600]
    assert "probe      ABSENT" in result.stdout
    assert "permitted by the season" in result.stdout


@pytest.mark.parametrize("name", ["season.example.json", "season.testnet.json"])
def test_every_shipped_season_config_validates_against_the_schema(name):
    """Two configs mean two chances to drift from the schema.

    Validated through the generated model, which is what the rest of the suite uses — rather than
    adding a `jsonschema` dependency for one assertion. `make schema` already guarantees the model
    and the schema agree, so this transitively checks the config against the schema.
    """
    import json

    from protocol.models.season_config import SeasonConfig

    SeasonConfig.model_validate(json.loads(pathlib.Path("config", name).read_text()))


@pytest.mark.parametrize("name", ["season.example.json", "season.testnet.json"])
def test_no_shipped_season_config_carries_a_float(name):
    """Every season config is hashed and anchored; a float makes the anchor unreproducible."""
    import json

    from protocol.canonical import assert_no_floats

    assert_no_floats(json.loads(pathlib.Path("config", name).read_text()))


def test_the_testnet_config_differs_from_the_example_only_in_declared_ways():
    """A testnet that quietly diverged on scoring would make its results uncomparable.

    The permitted differences are the identifier, the two size reductions that let a round finish in
    a sitting, and the one declared degradation. Everything in the reward path — criterion weights,
    rank weights, scoring, judging, weight allocation — must match, or the testnet is measuring a
    different mechanism than the one being tested.
    """
    import json

    example = json.loads(pathlib.Path("config/season.example.json").read_text())
    testnet = json.loads(pathlib.Path("config/season.testnet.json").read_text())

    for block in ("criterion_weights_ppm", "rank_weights_ppm", "scoring", "judging"):
        assert testnet[block] == example[block], f"{block} diverges from the reference season"

    # Weight allocation may relax `minimum_valid_challenges`, because the pack is smaller — but
    # nothing else, since the rest is the emission mechanism itself.
    for key, value in example["weight_allocation"].items():
        if key == "minimum_valid_challenges":
            continue
        assert testnet["weight_allocation"][key] == value, f"weight_allocation.{key} diverges"
