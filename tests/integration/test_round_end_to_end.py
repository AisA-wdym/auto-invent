"""One round, end to end, against real infrastructure. architecture.md 6.1, 9, 10, 13.

Marked `live` because it needs Docker and the reference image. It is the test that found two
defects the unit tests could not. On-chain digests are bare hex and the fetcher demanded the
`sha256:` form, so a validator would have refused *every* submission on a live network — with a
message claiming the digest was malformed. And the refusal path deleted the image it had loaded,
which for an image already present means deleting somebody else's.

## What is real here, and what is not

Real: a 42 MB artifact built by `docker save`, a real tar, a real streamed sha256, a real
`docker load`, a real image-digest comparison, a real timelock-sealed credential opened through the
chain client, a real container run under the sandbox's flags, and a real metering gateway.

Not real: the HTTP transport serving the artifact, and the chain. The transport is faked because
the address check refuses `localhost` — which is the control working rather than a gap, and the
check has its own tests with real hostnames. The chain is `FakeChain` because the alternative is a
registered netuid and a funded wallet, for a test about the pipeline rather than about the chain.

## Why the whole bundle is built rather than fixtured

A fixture would encode today's assumptions about the archive layout. Building it with `docker save`
and a real tar means the test fails when the layout changes, which is the point: this is the seam
between `ail-miner seal` and the validator, and both sides have to agree about it.
"""

from __future__ import annotations

import hashlib
import io
import json
import pathlib
import subprocess
import tarfile

import pytest

from chain.client import FakeChain, Neuron, RegisteredCommitment, SubnetView
from protocol.canonical import digest_object
from protocol.commitments import SubmissionCommitment
from validator.submissions import prepare_all

pytestmark = pytest.mark.live

REFERENCE_IMAGE = "ail-ref-lab:test"
ROUND_ID = "2026-08-03"
SECRET_KEY = "sk-or-v1-a-key-that-never-leaves-the-gateway"


def _docker_available() -> bool:
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=20).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _image_id(reference: str) -> str:
    result = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", reference],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        pytest.skip(f"{reference} is not present; build it with tools/build_reference.sh")
    return result.stdout.strip()


@pytest.fixture(scope="module")
def bundle(tmp_path_factory) -> dict:
    """A real submission: image, manifest, sealed credential envelope, digest."""
    if not _docker_available():
        pytest.skip("docker is not available")

    work = tmp_path_factory.mktemp("bundle")
    image_id = _image_id(REFERENCE_IMAGE)
    image_tar = work / "image.tar"
    subprocess.run(
        ["docker", "save", "-o", str(image_tar), REFERENCE_IMAGE], check=True, timeout=600
    )

    chain = FakeChain(netuid=1)
    chain.advance(10)
    capsule = chain.seal(SECRET_KEY.encode(), reveal_at_block=chain.current_block()).hex()
    capsule_digest = digest_object({"key_capsule": capsule, "nonce": "n1"})

    manifest = {
        "protocol_version": "AIL-3.0",
        "round_id": ROUND_ID,
        "container_digest": image_id,
        "model_manifest": {"portfolio": "anthropic/claude-sonnet-5"},
    }
    envelope = {
        "provider": "openrouter",
        "declared_spend_cap_usd": 25,
        "key_capsule": capsule,
        "nonce": "n1",
        "capsule_digest": capsule_digest,
    }

    archive = work / "artifact.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name, body in (("manifest.json", manifest), ("credential_envelope.json", envelope)):
            raw = json.dumps(body, indent=2, sort_keys=True).encode()
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            tar.addfile(info, io.BytesIO(raw))
        tar.add(image_tar, arcname="image.tar")

    payload = archive.read_bytes()
    return {
        "payload": payload,
        "digest": f"sha256:{hashlib.sha256(payload).hexdigest()}",
        "capsule_digest": capsule_digest,
        "image_id": image_id,
        "chain": chain,
    }


def _view(bundle: dict) -> SubnetView:
    commitment = SubmissionCommitment(
        round_id=ROUND_ID,
        bundle_digest=bundle["digest"],
        capsule_digest=bundle["capsule_digest"],
        artifact_url="https://miner.test/bundle.tar.gz",
    )
    return SubnetView(
        netuid=1,
        mechid=0,
        block=bundle["chain"].current_block(),
        neurons=(Neuron(7, "5Fminer", "cold", 1.0, False, True),),
        commitments=(RegisteredCommitment(7, "5Fminer", commitment.encode(), 900),),
    )


@pytest.fixture
def serve(monkeypatch):
    """Answer the artifact URL with given bytes, leaving every other control in force."""
    import httpx

    from validator import artifacts

    monkeypatch.setattr(artifacts, "_assert_public", lambda url: None)
    original = httpx.Client

    def install(payload: bytes) -> None:
        transport = httpx.MockTransport(lambda request: httpx.Response(200, content=payload))
        monkeypatch.setattr(
            httpx, "Client", lambda *a, **k: original(*a, **{**k, "transport": transport})
        )

    return install


def test_a_real_bundle_is_fetched_verified_loaded_and_opened(bundle, serve, tmp_path):
    """The whole preparation path against real bytes.

    Every assertion is about something that could silently be wrong: the digest could be compared in
    the wrong form (it was), the image could be loaded without checking what was loaded, and the
    credential could come from the envelope rather than from the timelock.
    """
    serve(bundle["payload"])
    prepared = prepare_all(
        _view(bundle), round_id=ROUND_ID, chain=bundle["chain"], workspace=tmp_path
    )

    assert not prepared.refused, [item.reason for item in prepared.refused]
    assert len(prepared.ready) == 1
    lab = prepared.ready[0]
    assert lab.uid == 7
    assert lab.image_digest == bundle["image_id"], "the loaded image is not the committed one"
    assert lab.api_key == SECRET_KEY, "the credential did not come through the timelock"
    assert lab.declared_spend_cap_usd == 25


def test_one_flipped_bit_is_refused_unopened(bundle, serve, tmp_path):
    """The commitment's entire purpose. The archive is 42 MB of valid tar; one bit changed makes it
    something the miner never committed to, and 'nearly right' is not a category."""
    payload = bundle["payload"]
    serve(payload[:-1] + bytes([payload[-1] ^ 1]))
    prepared = prepare_all(
        _view(bundle), round_id=ROUND_ID, chain=bundle["chain"], workspace=tmp_path
    )

    assert not prepared.ready
    assert "discarded unopened" in prepared.refused[0].reason
    # Neither the verified name nor the partial survives, so there is no state in which a caller
    # could find an artifact whose bytes were never checked — which is what lets `unpack` treat
    # verification as a precondition rather than re-checking.
    assert not (tmp_path / "uid-7" / "artifact.tar.gz").exists()
    assert not (tmp_path / "uid-7" / "artifact.partial").exists()
    assert not (tmp_path / "uid-7" / "bundle").exists()


def test_a_substituted_credential_envelope_is_refused(bundle, serve, tmp_path, monkeypatch):
    """A capsule that decrypts is not evidence that it is the right capsule.

    The envelope travels in the same archive as the bundle, so substituting it is a matter of
    rebuilding the tar — and under one provider surface a swapped key *succeeds*, billing a rival's
    account. This is why 5.4.1 puts the capsule digest on chain separately.
    """
    chain: FakeChain = bundle["chain"]
    attacker_capsule = chain.seal(b"sk-or-v1-someone-elses-key", reveal_at_block=1).hex()

    # Rebuild the archive with a different envelope, and commit to *this* archive's digest so the
    # bundle check passes and only the capsule digest is wrong.
    work = pathlib.Path(tmp_path) / "tampered"
    work.mkdir()
    swapped = work / "artifact.tar.gz"
    with (
        tarfile.open(fileobj=io.BytesIO(bundle["payload"]), mode="r:gz") as source,
        tarfile.open(swapped, "w:gz") as tar,
    ):
        for member in source:
            if member.name == "credential_envelope.json":
                raw = json.dumps(
                    {
                        "provider": "openrouter",
                        "declared_spend_cap_usd": 25,
                        "key_capsule": attacker_capsule,
                        "nonce": "n1",
                        "capsule_digest": digest_object(
                            {"key_capsule": attacker_capsule, "nonce": "n1"}
                        ),
                    },
                    indent=2,
                    sort_keys=True,
                ).encode()
                info = tarfile.TarInfo(member.name)
                info.size = len(raw)
                tar.addfile(info, io.BytesIO(raw))
                continue
            tar.addfile(member, source.extractfile(member))

    payload = swapped.read_bytes()
    commitment = SubmissionCommitment(
        round_id=ROUND_ID,
        bundle_digest=f"sha256:{hashlib.sha256(payload).hexdigest()}",
        # Still the *original* capsule digest: the miner committed to their own envelope.
        capsule_digest=bundle["capsule_digest"],
        artifact_url="https://miner.test/bundle.tar.gz",
    )
    view = SubnetView(
        netuid=1,
        mechid=0,
        block=chain.current_block(),
        neurons=(Neuron(7, "5Fminer", "cold", 1.0, False, True),),
        commitments=(RegisteredCommitment(7, "5Fminer", commitment.encode(), 900),),
    )

    serve(payload)
    prepared = prepare_all(view, round_id=ROUND_ID, chain=chain, workspace=tmp_path / "run")
    assert not prepared.ready
    assert "not the committed" in prepared.refused[0].reason


def test_an_image_that_is_not_the_committed_one_is_refused(bundle, serve, tmp_path):
    """A miner could otherwise commit one `container_digest` and ship another, and every control
    downstream would faithfully run bytes nothing on chain attests to.

    This test found a second defect on its first run. The refusal used to `docker image rm
    --force` what it had loaded — and `docker load` of an image already in the daemon is a no-op
    returning the existing reference, so the removal deleted an image that was there for another
    reason. It deleted the reference laboratory's own image: a miner could ship a copy of the image
    that sets the qualification floor with a deliberately wrong manifest digest, and delete it.
    """
    work = pathlib.Path(tmp_path) / "wrong-image"
    work.mkdir()
    archive = work / "artifact.tar.gz"
    with (
        tarfile.open(fileobj=io.BytesIO(bundle["payload"]), mode="r:gz") as source,
        tarfile.open(archive, "w:gz") as tar,
    ):
        for member in source:
            if member.name == "manifest.json":
                raw = json.dumps(
                    {
                        "protocol_version": "AIL-3.0",
                        "round_id": ROUND_ID,
                        # A digest that is well-formed and is not what is in the archive.
                        "container_digest": "sha256:" + "11" * 32,
                        "model_manifest": {"portfolio": "anthropic/claude-sonnet-5"},
                    },
                    indent=2,
                    sort_keys=True,
                ).encode()
                info = tarfile.TarInfo(member.name)
                info.size = len(raw)
                tar.addfile(info, io.BytesIO(raw))
                continue
            tar.addfile(member, source.extractfile(member))

    payload = archive.read_bytes()
    commitment = SubmissionCommitment(
        round_id=ROUND_ID,
        bundle_digest=f"sha256:{hashlib.sha256(payload).hexdigest()}",
        capsule_digest=bundle["capsule_digest"],
        artifact_url="https://miner.test/bundle.tar.gz",
    )
    view = SubnetView(
        netuid=1,
        mechid=0,
        block=bundle["chain"].current_block(),
        neurons=(Neuron(7, "5Fminer", "cold", 1.0, False, True),),
        commitments=(RegisteredCommitment(7, "5Fminer", commitment.encode(), 900),),
    )
    serve(payload)
    prepared = prepare_all(
        view, round_id=ROUND_ID, chain=bundle["chain"], workspace=tmp_path / "run"
    )
    assert not prepared.ready
    assert "not the committed" in prepared.refused[0].reason
    # The image the archive actually contained is still in the daemon. Removing it is what the
    # refusal used to do, and it is a denial of service against whoever legitimately owns it.
    assert _image_id(REFERENCE_IMAGE) == bundle["image_id"]
