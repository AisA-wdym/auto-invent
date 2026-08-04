"""What `ail-miner` hands over, against what the validator expects to receive.

This is the seam that was broken and that nothing watched. `tests/integration/test_round_end_to_end`
builds its artifact **by hand** in the shape `validator.submissions` wants, and the miner CLI built
a different one — so both sides had passing tests and no submission produced by the documented
commands could ever have been run:

* `submit` published `digest_object(manifest)` as `bundle_digest`, while `fetch_and_verify` checks
  the sha256 of the **downloaded bytes**. Those can never agree.
* the archive held the source tree, with `manifest.json` and `credential_envelope.json` written
  *beside* it rather than inside, and no `image.tar` at all.

So these tests assert the contract itself rather than either side's idea of it.
"""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

import pytest

from miner.cli.main import ARCHIVE_MEMBERS, SealError, _pack
from validator.submissions import IMAGE_NAME, MANIFEST_NAME

pytestmark = pytest.mark.determinism


def _sealed(root: Path, *, members: tuple[str, ...] = ARCHIVE_MEMBERS) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for name in members:
        if name.endswith(".json"):
            (root / name).write_text(json.dumps({"name": name}, sort_keys=True))
        else:
            (root / name).write_bytes(b"\x00" * 64)
    return root


def test_the_names_the_miner_writes_are_the_names_the_validator_looks_for() -> None:
    """A rename on either side is a submission nobody can run, so the two lists are pinned here."""
    assert MANIFEST_NAME in ARCHIVE_MEMBERS
    assert IMAGE_NAME in ARCHIVE_MEMBERS
    assert "credential_envelope.json" in ARCHIVE_MEMBERS


def test_the_archive_holds_every_member_the_validator_unpacks(tmp_path: Path) -> None:
    sealed = _sealed(tmp_path / "sealed")
    archive = tmp_path / "bundle.tar.gz"
    _pack(sealed, archive)

    with tarfile.open(archive) as tar:
        names = set(tar.getnames())
    assert set(ARCHIVE_MEMBERS) <= names


def test_the_returned_digest_is_over_the_archive_bytes(tmp_path: Path) -> None:
    """The property the whole fix turns on: this is what `fetch_and_verify` will recompute."""
    sealed = _sealed(tmp_path / "sealed")
    archive = tmp_path / "bundle.tar.gz"
    digest = _pack(sealed, archive)

    assert digest == "sha256:" + hashlib.sha256(archive.read_bytes()).hexdigest()


def test_packing_is_deterministic_so_a_miner_can_reproduce_the_committed_digest(
    tmp_path: Path,
) -> None:
    sealed = _sealed(tmp_path / "sealed")
    first = _pack(sealed, tmp_path / "one.tar.gz")
    second = _pack(sealed, tmp_path / "two.tar.gz")
    assert first == second


def test_the_credential_envelope_travels_inside_the_archive(tmp_path: Path) -> None:
    """`_open_credential` reads it from the unpacked root, so beside the archive is nowhere."""
    sealed = _sealed(tmp_path / "sealed")
    (sealed / "credential_envelope.json").write_text(json.dumps({"key_capsule": "deadbeef"}))
    archive = tmp_path / "bundle.tar.gz"
    _pack(sealed, archive)

    with tarfile.open(archive) as tar:
        handle = tar.extractfile("credential_envelope.json")
        assert handle is not None
        assert json.loads(handle.read())["key_capsule"] == "deadbeef"


def test_a_missing_member_is_named_rather_than_shipped(tmp_path: Path) -> None:
    sealed = _sealed(tmp_path / "sealed", members=(MANIFEST_NAME, "credential_envelope.json"))
    with pytest.raises(SealError, match=IMAGE_NAME):
        _pack(sealed, tmp_path / "bundle.tar.gz")


# --------------------------------------------------------------------------
# The round the manifest claims, and the round the commitment claims
# --------------------------------------------------------------------------


def test_submit_writes_the_round_into_the_manifest(tmp_path, monkeypatch, capsys):
    """`init` scaffolds `"round_id": "YYYY-MM-DD"`, `validate` passes it, `seal` keeps it, and
    `submit --round` set only the *commitment's* round — so the validator compared the two, found
    the placeholder, and refused. Every bundle the documented commands produced was refused, and the
    miner's only clue was a value they were never asked to set.

    One source of truth rather than a check: asking a miner to keep two copies in step is asking for
    the mismatch.
    """
    import argparse
    import importlib

    cli = importlib.import_module("miner.cli.main")

    sealed = tmp_path / "sealed"
    sealed.mkdir()
    (sealed / "manifest.json").write_text(json.dumps({"round_id": "YYYY-MM-DD", "bundle_id": "x"}))
    (sealed / "credential_envelope.json").write_text(
        json.dumps({"key_capsule": "ab", "capsule_digest": "sha256:" + "cd" * 32})
    )
    (sealed / "image.tar").write_bytes(b"img")

    def fake_pack(sealed_dir, dest):
        dest.write_bytes(b"artifact")
        return "sha256:" + "ef" * 32

    monkeypatch.setattr(cli, "_pack", fake_pack)
    args = argparse.Namespace(
        sealed=sealed,
        round="2026-08-04",
        url="https://example.test/a.tar.gz",
        dry_run=True,
        netuid=542,
        wallet="w",
        hotkey="h",
        network="test",
    )
    assert cli.command_submit(args) == 0
    written = json.loads((sealed / "manifest.json").read_text())
    assert written["round_id"] == "2026-08-04"
    assert "round_id set to 2026-08-04" in capsys.readouterr().out


def test_a_manifest_already_on_the_right_round_is_left_alone(tmp_path, monkeypatch):
    """Rewriting it every time would change the archive bytes, and therefore `bundle_digest`, for a
    resubmission that changed nothing."""
    import argparse
    import importlib

    cli = importlib.import_module("miner.cli.main")

    sealed = tmp_path / "sealed"
    sealed.mkdir()
    body = {"round_id": "2026-08-04", "bundle_id": "x"}
    (sealed / "manifest.json").write_text(json.dumps(body))
    before = (sealed / "manifest.json").read_bytes()
    (sealed / "credential_envelope.json").write_text(
        json.dumps({"key_capsule": "ab", "capsule_digest": "sha256:" + "cd" * 32})
    )
    def fake_pack(sealed_dir, dest):
        dest.write_bytes(b"artifact")
        return "sha256:" + "ef" * 32

    monkeypatch.setattr(cli, "_pack", fake_pack)
    args = argparse.Namespace(
        sealed=sealed, round="2026-08-04", url="https://example.test/a.tar.gz",
        dry_run=True, netuid=542, wallet="w", hotkey="h", network="test",
    )
    assert cli.command_submit(args) == 0
    assert (sealed / "manifest.json").read_bytes() == before
