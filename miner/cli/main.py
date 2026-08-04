"""Miner CLI: `ail-miner`, or `python -m miner.cli.main`.

Four commands, matching what a miner actually has to do:

    ail-miner init      scaffold a laboratory that runs
    ail-miner validate  check a bundle against every gate that can be checked offline
    ail-miner seal      build the sealed submission and the sealed credential envelope
    ail-miner submit    publish the commitment on chain

## `validate` exists so a miner does not learn from a score

Thirteen hard gates invalidate a response, and a miner who discovers a gate failure from a published
score has lost a day. Most of those gates are checkable before submission: the manifest, the digest
form, the licence, the lockfile, the output schema, the absence of embedded secrets. So `validate`
runs every offline check the validator's Stage 0 (12) will run, and names each failure with the gate
it corresponds to.

It cannot check the two that need a run — budget and wall clock — and it says so rather than
implying a clean bill of health.

## The credential is sealed separately, and the CLI enforces it

5.4.1: the credential travels in a distinct envelope because 6.3 publishes the bundle, and a
credential inside the published object would be published with it. `seal` therefore writes two
files, and refuses to proceed if it finds a provider key anywhere inside the bundle — which is the
mistake this separation exists to make impossible, and the one a miner is most likely to make by
committing a `.env`.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import re
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path

from protocol.canonical import digest_bytes
from protocol.commitments import CommitmentError, SubmissionCommitment

_log = logging.getLogger("ail-miner")

#: Anything that looks like a provider credential. Checked across the whole bundle before sealing,
#: because a key committed into source is published by 6.3 and cannot be un-published.
_SECRET_SHAPES = tuple(
    re.compile(pattern)
    for pattern in (
        r"sk-or-v1-[A-Za-z0-9]{32,}",  # OpenRouter
        r"sk-ant-[A-Za-z0-9\-_]{32,}",  # Anthropic
        r"sk-proj-[A-Za-z0-9\-_]{32,}",  # OpenAI project
        r"sk-[A-Za-z0-9]{40,}",  # generic OpenAI-style
        r"AIza[A-Za-z0-9\-_]{30,}",  # Google
        r"ghp_[A-Za-z0-9]{30,}",  # GitHub
        r"hf_[A-Za-z0-9]{30,}",  # Hugging Face
    )
)

#: Files never worth scanning for secrets and never worth shipping. Skipped rather than flagged: a
#: virtualenv contains other people's test fixtures, and flagging those would bury the real finding.
_SKIP = frozenset({".git", ".venv", "__pycache__", "node_modules", ".mypy_cache", ".pytest_cache"})

_REQUIRED_MANIFEST = (
    "protocol_version",
    "bundle_id",
    "bundle_version",
    "entrypoint",
    "container_digest",
    "source_archive_hash",
    "lockfile_hash",
    "license",
    "supported_domains",
    "output_schema",
)


@dataclass(frozen=True, slots=True)
class Finding:
    """One problem with a bundle, tied to the gate or section it will fail."""

    reference: str
    detail: str

    def __str__(self) -> str:
        return f"{self.reference}: {self.detail}"


def validate_bundle(root: Path) -> list[Finding]:
    """Every offline check 12's Stage 0 will run. Returns findings, does not raise.

    Returns a list so a miner sees all of it at once. Raising on the first would mean one fix per
    invocation, and the point of this command is to compress a week of one-fix-per-day into one run.
    """
    findings: list[Finding] = []

    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        findings.append(Finding("5.2", "no manifest.json at the bundle root"))
        return findings

    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as error:
        findings.append(Finding("5.2", f"manifest.json is not valid JSON: {error}"))
        return findings

    for field in _REQUIRED_MANIFEST:
        if not manifest.get(field):
            findings.append(Finding("5.2", f"manifest is missing {field}"))

    version = str(manifest.get("protocol_version", ""))
    if version and version != "AIL-3.0":
        findings.append(
            Finding(
                "12",
                f"protocol_version is {version!r}; this season is AIL-3.0. An unsupported version "
                "is excluded before execution, so the bundle would never run.",
            )
        )

    digest = str(manifest.get("container_digest", ""))
    if digest and not digest.startswith("sha256:"):
        findings.append(
            Finding(
                "10",
                f"container_digest is {digest!r}, not a sha256 digest. A tag can be "
                "repointed after the deadline, and 6.1 fixes the bundle at submission.",
            )
        )
    elif digest and len(digest) != len("sha256:") + 64:
        findings.append(
            Finding("10", f"container_digest is {len(digest)} characters; a sha256 digest is 71")
        )

    if not (root / "requirements.lock").is_file():
        findings.append(Finding("12", "no requirements.lock; 12 checks the dependency lock"))
    if not (root / "SBOM.json").is_file():
        findings.append(Finding("12", "no SBOM.json"))
    if not (root / "LICENSE").is_file():
        findings.append(Finding("12", "no LICENSE; 12 checks for a valid licence"))

    models_path = root / "model_manifest.json"
    if not models_path.is_file():
        findings.append(Finding("5.3", "no model_manifest.json"))
    else:
        findings.extend(_check_models(models_path))

    findings.extend(_scan_for_secrets(root))
    return findings


def _check_models(path: Path) -> list[Finding]:
    """5.3: every externally invoked model declared, pinned, and reached through OpenRouter."""
    findings: list[Finding] = []
    try:
        manifest = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        return [Finding("5.3", f"model_manifest.json is not valid JSON: {error}")]

    models = manifest.get("models")
    if not isinstance(models, list) or not models:
        return [
            Finding(
                "13.3",
                "model_manifest declares no models. Gate 13.3 makes undeclared model use fatal, so "
                "an empty manifest means every call the laboratory makes invalidates its answer.",
            )
        ]

    for index, model in enumerate(models):
        if not isinstance(model, dict):
            findings.append(Finding("5.3", f"models[{index}] is not an object"))
            continue
        where = f"models[{index}] ({model.get('alias', 'unnamed')})"
        if model.get("provider") != "openrouter":
            findings.append(
                Finding(
                    "3.4.1",
                    f"{where} declares provider {model.get('provider')!r}. Every model is reached "
                    "through OpenRouter; a call to anything else fails gate 13.5.",
                )
            )
        if not model.get("model_slug"):
            findings.append(Finding("5.3", f"{where} has no model_slug"))
        revision = str(model.get("revision", ""))
        if model.get("hf_repo") and len(revision) != 40:
            findings.append(
                Finding(
                    "5.3",
                    f"{where} declares hf_repo with a {len(revision)}-character revision. A "
                    "40-character commit SHA is required: an abbreviation becomes ambiguous as the "
                    "repository grows, and pinning exists so the artifact cannot move.",
                )
            )
    return findings


def _scan_for_secrets(root: Path) -> list[Finding]:
    """12's "absence of embedded secrets", checked before the bundle is sealed.

    Scanned before sealing rather than after, because after is too late in the only sense that
    matters: 6.3 publishes the bundle, and a published key cannot be un-published.
    """
    findings: list[Finding] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or _SKIP & set(path.parts):
            continue
        try:
            text = path.read_text(errors="ignore")
        except OSError as error:
            # Reported, not skipped. This is the one check standing between a credential committed
            # by habit and 6.3 publishing it, and a file it could not read is a file it did not
            # check — which is not a file that is clean. Skipping silently would let `seal` proceed
            # on a bundle containing an unread file, and a published key cannot be un-published.
            findings.append(
                Finding(
                    "12",
                    f"could not read {path.relative_to(root)} to scan for credentials ({error}). "
                    "A file that was not scanned is not a file that is clean, and 6.3 publishes "
                    "the bundle after execution closes.",
                )
            )
            continue
        for pattern in _SECRET_SHAPES:
            match = pattern.search(text)
            if match:
                findings.append(
                    Finding(
                        "12",
                        f"{path.relative_to(root)} contains something shaped like a provider "
                        f"credential ({match.group(0)[:12]}…). 6.3 publishes the bundle after "
                        "execution closes, so this would become public. The credential belongs in "
                        "the sealed envelope (5.4.1), which is never published.",
                    )
                )
                break
    return findings


def _archive(root: Path, destination: Path) -> str:
    """Tar the bundle deterministically and return its digest.

    Deterministic because the digest goes in the manifest and on chain: sorted entries, zeroed
    mtimes, normalised ownership and mode. Without that, two archives of identical source hash
    differently and a miner cannot reproduce the digest they committed to.
    """
    entries = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and not _SKIP & set(path.parts)
    )
    # gzip stamps the current time into its header, so `tarfile.open(..., "w:gz")` produces
    # different bytes on every run and the digest a miner committed could not be rebuilt.
    # `mtime=0` removes the only non-deterministic field. Caught by the determinism test.
    with (
        destination.open("wb") as raw,
        gzip.GzipFile(
            # `filename=""` as well as `mtime=0`: given a fileobj, gzip otherwise copies
            # `fileobj.name` into the header, so an archive written to a different path
            # hashed differently while holding identical content.
            filename="", fileobj=raw, mode="wb", compresslevel=9, mtime=0
        ) as compressed,
        tarfile.open(fileobj=compressed, mode="w") as archive,
    ):
        for path in entries:
            info = archive.gettarinfo(path, arcname=str(path.relative_to(root)))
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mode = 0o644 if not path.is_dir() else 0o755
            with path.open("rb") as handle:
                archive.addfile(info, handle)
    return digest_bytes(destination.read_bytes())


class SealError(RuntimeError):
    """The submission could not be packaged as the validator will expect to receive it."""


#: The three names `validator.submissions` looks for in the unpacked archive. Named here rather
#: than imported so the miner CLI does not depend on the validator package — but they are the same
#: three strings, and a test asserts that they still are.
ARCHIVE_MEMBERS = ("manifest.json", "credential_envelope.json", "image.tar")


def _save_image(container_digest: str, destination: Path) -> str:
    """`docker save` the image the manifest pins, and confirm it is the image that was pinned.

    The validator loads this tar and checks what came out against `container_digest` (13.4). Saving
    by digest rather than by tag is the same rule the runner enforces: a tag is mutable, and 6.1
    fixes the container at the deadline.
    """
    import subprocess

    if not container_digest.startswith("sha256:") or len(container_digest) != 71:
        raise SealError(
            f"container_digest is {container_digest!r}. It must be a full sha256 image digest — "
            "`docker inspect <tag> --format '{{.Id}}'` prints one. A tag cannot be sealed: it is "
            "mutable, so the bytes could change after the deadline."
        )
    try:
        result = subprocess.run(
            ["docker", "save", "-o", str(destination), container_digest],
            capture_output=True,
            text=True,
            timeout=900,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise SealError(f"could not run `docker save`: {error}") from error
    if result.returncode != 0:
        raise SealError(
            f"`docker save {container_digest}` failed: {result.stderr.strip()[:300]}. Build the "
            "image first, and check the digest in manifest.json names the image you built."
        )
    return container_digest


def _pack(sealed: Path, destination: Path) -> str:
    """Build the artifact the validator downloads, and return the digest of its bytes.

    ## Why this is at submit time and not at seal time

    `bundle_digest` on the commitment is checked by `validator.artifacts.fetch_and_verify` against
    the **bytes of the downloaded archive**. The credential envelope travels inside that archive,
    and its capsule is filled by the miner *after* sealing — so the archive's bytes are not final
    until then, and a digest computed at seal time could never match.

    That was the defect. `seal` hashed the *manifest object* with `digest_object(manifest)` and
    `submit` published that as `bundle_digest`, while the validator hashed the downloaded file.
    The two can never agree, so every submission this CLI produced would have been refused at
    `fetch_and_verify` with a digest mismatch — and nothing caught it, because the end-to-end test
    builds its archive by hand rather than through these commands.

    Deterministic in the same way `_archive` is: sorted names, zeroed mtimes, normalised ownership.
    The digest goes on chain, so a miner must be able to rebuild the same bytes and get the same
    answer.
    """
    members = [(name, sealed / name) for name in ARCHIVE_MEMBERS]
    missing = [name for name, path in members if not path.is_file()]
    if missing:
        raise SealError(
            f"{sealed} is missing {', '.join(missing)}. The validator unpacks the archive and "
            "looks for exactly these, so an archive without them cannot be run. Re-run "
            "`ail-miner seal`."
        )
    # The source archive rides along too: 6.3 publishes source after execution closes, and the
    # validator keeps whatever it fetched. It is not required, so it is added only if sealed.
    source = sealed / "source.tar.gz"
    if source.is_file():
        members.append(("source.tar.gz", source))

    # gzip stamps the current time into its header, so `tarfile.open(..., "w:gz")` produces
    # different bytes on every run and the digest a miner committed could not be rebuilt.
    # `mtime=0` removes the only non-deterministic field. Caught by the determinism test.
    with (
        destination.open("wb") as raw,
        gzip.GzipFile(
            # `filename=""` as well as `mtime=0`: given a fileobj, gzip otherwise copies
            # `fileobj.name` into the header, so an archive written to a different path
            # hashed differently while holding identical content.
            filename="", fileobj=raw, mode="wb", compresslevel=9, mtime=0
        ) as compressed,
        tarfile.open(fileobj=compressed, mode="w") as archive,
    ):
        for name, path in sorted(members):
            info = archive.gettarinfo(path, arcname=name)
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mode = 0o644
            with path.open("rb") as handle:
                archive.addfile(info, handle)
    return digest_bytes(destination.read_bytes())


def command_init(args: argparse.Namespace) -> int:
    """Scaffold a laboratory that runs. Reference A, in its simplest form.

    A scaffold that *runs* rather than a set of stubs, because a miner whose first invocation fails
    cannot tell a broken scaffold from a broken environment.
    """
    root: Path = args.path
    if root.exists() and any(root.iterdir()):
        print(f"{root} is not empty; refusing to overwrite", file=sys.stderr)
        return 1
    (root / "src").mkdir(parents=True, exist_ok=True)

    from miner.reference.template import scaffold

    for name, content in scaffold().items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)

    print(f"scaffolded a laboratory at {root}")
    print("\nnext:")
    print(f"  cd {root} && docker build -t my-lab .")
    print("  ail-miner validate .")
    print("  ail-miner seal . --out ../sealed")
    return 0


def command_validate(args: argparse.Namespace) -> int:
    findings = validate_bundle(args.path)
    if findings:
        print(f"{len(findings)} finding(s):\n", file=sys.stderr)
        for finding in findings:
            print(f"  x {finding}", file=sys.stderr)
        return 1

    print("every offline check passed.\n")
    # Named explicitly rather than left implied. A miner told "validation passed" would reasonably
    # conclude the bundle cannot fail a gate, and two gates cannot be checked without running.
    print("Not checkable offline, and still fatal if breached:")
    print("  13.6 budget exceeded — measured by the RCG during execution")
    print("  13.7 time limit exceeded — measured by the runner's wall clock")
    print("  13.8 fabricated citation — the validator resolves every URL you cite")
    print("\nRehearse against those before you submit:")
    print("  docker build -t my-lab:dev .")
    print("  export AIL_MINER_API_KEY=sk-or-...")
    print("  ail-miner run . --image my-lab:dev")
    return 0


def command_run(args: argparse.Namespace) -> int:
    """Rehearse the bundle in the validator's own sandbox (see `miner/cli/rehearse.py`).

    The command `validate` has been telling miners to run and which never existed. Everything it
    needs — the container flags, the network confinement, a metering gateway, a session token — is
    work the validator already does, and a miner reconstructing it by hand would be building a
    second definition of what a run is.
    """
    from miner.cli.rehearse import api_key_from_environment, rehearse

    try:
        api_key = api_key_from_environment()
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        return 2

    try:
        return rehearse(
            args.path,
            image=args.image,
            api_key=api_key,
            challenges_path=args.challenges,
            limit=args.limit,
        )
    except (RuntimeError, ValueError, OSError) as error:
        print(f"rehearsal could not start: {error}", file=sys.stderr)
        return 2


def command_seal(args: argparse.Namespace) -> int:
    """Build the sealed bundle and the separate sealed credential envelope (6.1, 5.4.1)."""
    findings = validate_bundle(args.path)
    blocking = [f for f in findings if f.reference == "12" and "credential" in f.detail]
    if blocking:
        # A secret in the bundle is the one finding that blocks sealing outright. Everything else is
        # a gate the miner may choose to fail; this one publishes their key.
        print("refusing to seal — the bundle contains credential material:\n", file=sys.stderr)
        for finding in blocking:
            print(f"  x {finding}", file=sys.stderr)
        return 1
    if findings and not args.force:
        print(f"{len(findings)} finding(s); pass --force to seal anyway:\n", file=sys.stderr)
        for finding in findings:
            print(f"  x {finding}", file=sys.stderr)
        return 1

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    source_path = out / "source.tar.gz"
    source_hash = _archive(args.path, source_path)

    manifest = json.loads((args.path / "manifest.json").read_text())
    manifest["source_archive_hash"] = source_hash
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))

    # The image, as bytes. `validator.submissions` looks for `image.tar` inside the fetched archive
    # and `docker load`s it, because 6.1 fixes the container at the deadline and a *tag* the
    # validator pulled at reveal could have been re-pointed after it. Nothing produced this file
    # before, so no bundle this command wrote could ever be run.
    try:
        image_digest = _save_image(str(manifest.get("container_digest", "")), out / "image.tar")
    except SealError as error:
        print(f"cannot package the image: {error}", file=sys.stderr)
        return 1

    envelope = {
        "provider": "openrouter",
        # Which kind of credential you will put in `key_capsule`. Declared rather than sniffed,
        # because the two are handled differently and the validator verifies the declaration before
        # anything runs — a mismatch is refused by name instead of by every call failing.
        #
        #   "management" — a provisioning key. It *cannot spend*: the validator mints a runtime
        #                  key capped at declared_spend_cap_usd, uses it, and deletes it.
        #                  The minted key also expires on its own, so a validator that crashes
        #                  before revoking still leaves nothing live for long.
        #   "runtime"    — an ordinary sk-or-v1 key. It *is* spendable balance, bounded only
        #                  by whatever cap you set on it. Provision a dedicated one per round.
        "credential_kind": args.credential_kind,
        "declared_spend_cap_usd": args.spend_cap,
        # The key itself is not written here. `seal` deliberately does not read it: a command that
        # took a credential as an argument would put it in the shell history, and one that read it
        # from a file would leave a copy on disk beside the bundle it must never be inside.
        "key_capsule": "",
        "nonce": "",
        "capsule_digest": "",
    }
    envelope_path = out / "credential_envelope.json"
    if envelope_path.is_file() and not args.force:
        # A re-seal would blank a capsule the miner has already filled, and the loss is silent:
        # `submit` would refuse with "empty key_capsule" and the miner would refill it, not knowing
        # the image or manifest beside it had also changed underneath.
        print(
            f"{envelope_path} already exists. Sealing again would blank a capsule you may have "
            "already filled — pass --force to overwrite, or seal to a fresh --out.",
            file=sys.stderr,
        )
        return 1
    envelope_path.write_text(json.dumps(envelope, indent=2, sort_keys=True))

    print(f"sealed to {out}")
    print(f"  image              {image_digest}")
    print(f"  source archive     {source_hash}")
    print("\nThe credential envelope is never published (5.4.1, 6.3) — but it does travel inside")
    print("the submitted archive, which is why `submit` builds that archive rather than `seal`:")
    print("the bundle digest is over the archive's bytes, and those are not final until the")
    print("capsule is in it.")
    print(f"\nFill key_capsule in {envelope_path} with your timelock-encrypted "
          f"{args.credential_kind} key, then run `ail-miner submit`.")
    if args.credential_kind == "management":
        print("\nA management key cannot spend. The validator mints a runtime key capped at your")
        print(f"${args.spend_cap} for the round, then deletes it — and it expires on its own")
        print("if the validator never gets that far.")
        print("Get one at openrouter.ai/settings/provisioning-keys.")
        print("It can create and delete keys on your account, so fund that account for this only.")
    else:
        print("\nProvision a dedicated, spend-capped key for the round — the RCG enforces the")
        print("round ceiling regardless, but a runtime key *is* spendable balance, so the cap you")
        print("set on it is the only bound a validator defect runs into.")
        print("Consider --credential-kind management instead: a management key cannot spend at")
        print("all, and the per-round cap is then enforced by OpenRouter rather than only by us.")
    return 0


def command_submit(args: argparse.Namespace) -> int:
    """Publish the on-chain commitment (6.1 step 6)."""
    envelope = json.loads((args.sealed / "credential_envelope.json").read_text())

    if not envelope.get("key_capsule"):
        print(
            "the credential envelope has an empty key_capsule. The validator decrypts this to run "
            "your laboratory on your own account (3.4.2); without it your bundle cannot be run and "
            "will score nothing.",
            file=sys.stderr,
        )
        return 1

    # The round goes into the manifest here, from the same `--round` the commitment carries.
    #
    # It was not written anywhere. `ail-miner init` scaffolds `"round_id": "YYYY-MM-DD"`, `validate`
    # passed it, `seal` kept it, and `submit --round 2026-08-04` set only the *commitment's* round —
    # so the validator compared the two, found the placeholder, and refused with "the manifest is
    # sealed for round 'YYYY-MM-DD'". Every bundle the documented commands produced was refused, and
    # the miner's only clue was a value they were never asked to set.
    #
    # One source of truth rather than a check: asking a miner to keep two copies in step is asking
    # for the mismatch. Written before `_pack` because the manifest is inside the archive whose
    # bytes become `bundle_digest`.
    try:
        manifest_path = args.sealed / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("round_id") != args.round:
            manifest["round_id"] = args.round
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            print(f"manifest round_id set to {args.round}")
    except (OSError, json.JSONDecodeError) as error:
        print(f"cannot read {args.sealed / 'manifest.json'}: {error}", file=sys.stderr)
        return 1

    # Built here rather than at seal time: the digest is over these bytes, and the capsule that is
    # inside them was filled after sealing. See `_pack`.
    archive_path = args.sealed / "bundle.tar.gz"
    try:
        bundle_digest = _pack(args.sealed, archive_path)
    except SealError as error:
        print(f"cannot build the artifact: {error}", file=sys.stderr)
        return 1

    try:
        commitment = SubmissionCommitment(
            round_id=args.round,
            bundle_digest=bundle_digest,
            capsule_digest=str(envelope.get("capsule_digest", "")),
            artifact_url=args.url,
        )
    except CommitmentError as error:
        print(f"cannot build the commitment: {error}", file=sys.stderr)
        return 1

    payload = commitment.encode()
    print(f"artifact  {archive_path} ({archive_path.stat().st_size} bytes)")
    print(f"digest    {bundle_digest}")
    print(f"Publish that exact file at {args.url} — the validator downloads it and refuses any")
    print("bytes that do not hash to the digest above.")
    if args.dry_run:
        print(f"\nwould publish ({len(payload.encode())} bytes):\n  {payload}")
        return 0

    from chain.client import BittensorChain, ChainError

    chain = BittensorChain(
        netuid=args.netuid,
        wallet_name=args.wallet,
        hotkey_name=args.hotkey,
        network=args.network,
    )
    try:
        block = chain.publish_commitment(payload)
    except ChainError as error:
        print(f"could not publish: {error}", file=sys.stderr)
        return 3

    print(f"submitted at block {block}")
    print(f"  {payload}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ail-miner", description="Build, check and submit an autonomous invention laboratory."
    )
    parser.add_argument("--log-level", default="WARNING")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="scaffold a laboratory that runs")
    init.add_argument("path", type=Path, nargs="?", default=Path("invention-lab"))
    init.set_defaults(handler=command_init)

    validate = sub.add_parser("validate", help="run every check the validator can run offline")
    validate.add_argument("path", type=Path, nargs="?", default=Path("."))
    validate.set_defaults(handler=command_validate)

    run_cmd = sub.add_parser(
        "run", help="rehearse in the validator's sandbox, against real gates and a real meter"
    )
    run_cmd.add_argument("path", type=Path, nargs="?", default=Path("."))
    run_cmd.add_argument(
        "--image", required=True, help="the local image tag to run, e.g. my-lab:dev"
    )
    run_cmd.add_argument(
        "--challenges",
        type=Path,
        default=None,
        help=(
            "a pack to rehearse against; defaults to one built-in challenge. A published past "
            "round is the harder test"
        ),
    )
    run_cmd.add_argument(
        "--limit",
        type=int,
        default=1,
        help="how many challenges to run. Each one costs what it costs in a real round",
    )
    run_cmd.set_defaults(handler=command_run)

    seal = sub.add_parser("seal", help="build the sealed bundle and credential envelope")
    seal.add_argument("path", type=Path, nargs="?", default=Path("."))
    seal.add_argument(
        "--credential-kind",
        choices=("management", "runtime"),
        default="management",
        help=(
            "management: a provisioning key that cannot spend; the validator mints a capped, "
            "expiring key per round. runtime: an ordinary key, which is spendable balance"
        ),
    )
    seal.add_argument("--out", type=Path, default=Path("sealed"))
    seal.add_argument(
        "--spend-cap",
        type=int,
        default=50,
        help=(
            "declared_spend_cap_usd (5.4.1). Default 50: one challenge at the season ceiling costs "
            "roughly $2 against a frontier model, and a twenty-challenge day is about $40 — so 25 "
            "would have run out mid-round."
        ),
    )
    seal.add_argument("--force", action="store_true", help="seal despite non-credential findings")
    seal.set_defaults(handler=command_seal)

    submit = sub.add_parser("submit", help="publish the on-chain commitment")
    submit.add_argument("sealed", type=Path, nargs="?", default=Path("sealed"))
    submit.add_argument("--round", required=True, help="round id, e.g. 2026-08-03")
    submit.add_argument("--url", required=True, help="https:// or ipfs:// URL of the sealed bundle")
    submit.add_argument("--netuid", type=int, required=True)
    submit.add_argument("--network", default="finney")
    submit.add_argument("--wallet", default="default")
    submit.add_argument("--hotkey", default="default")
    submit.add_argument(
        "--dry-run", action="store_true", help="print the commitment, publish nothing"
    )
    submit.set_defaults(handler=command_submit)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(message)s")
    try:
        return int(args.handler(args))
    except (OSError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
