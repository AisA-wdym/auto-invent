"""Fetching and opening a miner's bundle: architecture.md 6.1, 10.

This is the only code in the repository that processes stranger-chosen bytes *outside* the
sandbox, so every control here is one the sandbox would otherwise have provided. Each test below is
a specific attack, run rather than described — a real tar with a real `../` member, a real hostname
resolving to a real link-local address.

The SSRF tests use `localhost` and `169.254.169.254` because they are the two that matter: one
reaches this validator's own gateway, which holds both API keys, and the other reaches the cloud
metadata service, which answers unauthenticated HTTP with instance credentials.
"""

from __future__ import annotations

import hashlib
import tarfile

import pytest

from validator.artifacts import ArtifactError, FetchLimits, fetch_and_verify, unpack

pytestmark = pytest.mark.determinism


def digest_of(raw: bytes) -> str:
    return f"sha256:{hashlib.sha256(raw).hexdigest()}"


# --------------------------------------------------------------------------
# Where a URL may point
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com/bundle.tar.gz",
        "ftp://example.com/bundle.tar.gz",
        "file:///etc/passwd",
    ],
)
def test_only_https_is_fetched(url, tmp_path):
    """A plain HTTP body can be replaced in transit. The digest would catch it — but only after this
    process had downloaded and unarchived attacker-chosen bytes, which is the thing to avoid."""
    with pytest.raises(ArtifactError, match="only https is fetched"):
        fetch_and_verify(url, expected_digest=digest_of(b""), into=tmp_path)


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1"])
def test_a_url_resolving_to_this_host_is_refused(host, tmp_path):
    """`127.0.0.1` reaches the gateway, which holds both API keys, from a process that already holds
    them — so the ask is not "leak a key" but "make the validator fetch its own admin surface"."""
    with pytest.raises(ArtifactError, match="not a public address"):
        fetch_and_verify(
            f"https://{host}/bundle.tar.gz", expected_digest=digest_of(b""), into=tmp_path
        )


def test_the_cloud_metadata_address_is_refused(tmp_path):
    """169.254.169.254 answers unauthenticated HTTP with instance credentials on every major
    provider, and a miner chooses this hostname — so it is one DNS record away."""
    with pytest.raises(ArtifactError, match="not a public address"):
        fetch_and_verify(
            "https://169.254.169.254/latest/meta-data/",
            expected_digest=digest_of(b""),
            into=tmp_path,
        )


def test_a_private_range_is_refused(tmp_path):
    """10/8 and 192.168/16 reach the operator's own network, which the validator has no business
    fetching a miner's bundle from."""
    with pytest.raises(ArtifactError, match="not a public address"):
        fetch_and_verify(
            "https://10.0.0.5/bundle.tar.gz", expected_digest=digest_of(b""), into=tmp_path
        )


def test_a_committed_digest_that_is_not_a_digest_is_refused_before_any_request(tmp_path):
    """Checked first. Without a digest there is nothing to check the download against, and an
    unchecked download of miner-chosen bytes is arbitrary code — so no request is made at all."""
    with pytest.raises(ArtifactError, match="not a sha256 digest"):
        fetch_and_verify(
            "https://example.invalid/x.tar.gz", expected_digest="latest", into=tmp_path
        )


# --------------------------------------------------------------------------
# What an archive may contain
# --------------------------------------------------------------------------


def build_tar(tmp_path, members):
    """A real tar with real members. `members` is (name, content, kind)."""
    path = tmp_path / "archive.tar"
    with tarfile.open(path, "w") as tar:
        for name, content, kind in members:
            info = tarfile.TarInfo(name)
            if kind == "file":
                payload = content.encode()
                info.size = len(payload)
                import io

                tar.addfile(info, io.BytesIO(payload))
            elif kind == "symlink":
                info.type = tarfile.SYMTYPE
                info.linkname = content
                tar.addfile(info)
            elif kind == "device":
                info.type = tarfile.CHRTYPE
                info.devmajor, info.devminor = 1, 3
                tar.addfile(info)
    return path


def test_a_normal_archive_extracts(tmp_path):
    archive = build_tar(tmp_path, [("manifest.json", "{}", "file")])
    root = unpack(archive, into=tmp_path / "out")
    assert (root / "manifest.json").read_text() == "{}"


def test_a_member_escaping_with_dot_dot_is_refused(tmp_path):
    """The classic. `../../etc/cron.d/x` written by a process that runs as the validator."""
    archive = build_tar(tmp_path, [("../escaped.txt", "x", "file")])
    with pytest.raises(ArtifactError):
        unpack(archive, into=tmp_path / "out")
    assert not (tmp_path / "escaped.txt").exists()


def test_an_absolute_member_is_neutralised_rather_than_refused(tmp_path):
    """Measured, not assumed: Python's `data` filter makes an absolute path relative instead of
    raising, so `/tmp/absolute.txt` lands at `out/tmp/absolute.txt`.

    That is safe — the write stays inside the destination — and it is worth pinning, because the
    obvious assumption is that it raises like `..` does. A future filter change that started writing
    to the real `/tmp` would break this test rather than the host.
    """
    archive = build_tar(tmp_path, [("/tmp/absolute.txt", "x", "file")])
    root = unpack(archive, into=tmp_path / "out")
    assert (root / "tmp" / "absolute.txt").read_text() == "x"
    assert (root / "tmp" / "absolute.txt").resolve().is_relative_to(root.resolve())


def test_a_symlink_pointing_outside_is_refused(tmp_path):
    """A symlink is the traversal that survives a naive path check: the member name is innocent and
    the *target* is not, so a later write through it lands wherever it points."""
    archive = build_tar(tmp_path, [("link", "/etc/passwd", "symlink")])
    with pytest.raises(ArtifactError):
        unpack(archive, into=tmp_path / "out")


def test_a_device_node_is_refused(tmp_path):
    archive = build_tar(tmp_path, [("null", "", "device")])
    with pytest.raises(ArtifactError):
        unpack(archive, into=tmp_path / "out")


def test_too_many_members_is_refused(tmp_path):
    """A million empty files is not a size problem, it is an inode problem — and it is invisible to
    a byte cap."""
    archive = build_tar(tmp_path, [(f"f{index}", "x", "file") for index in range(20)])
    with pytest.raises(ArtifactError, match="more than 5 entries"):
        unpack(archive, into=tmp_path / "out", limits=FetchLimits(maximum_members=5))


def test_an_expansion_bomb_is_refused_by_declared_size(tmp_path):
    """The download cap does not bound this: a tar of zeros compresses about a thousand to one, so a
    2 MB download is a 2 GB extraction."""
    archive = build_tar(tmp_path, [("big", "x" * 10_000, "file")])
    with pytest.raises(ArtifactError, match="expands past"):
        unpack(archive, into=tmp_path / "out", limits=FetchLimits(maximum_extracted_bytes=100))


def test_something_that_is_not_a_tar_is_refused_rather_than_crashing(tmp_path):
    junk = tmp_path / "junk.tar"
    junk.write_bytes(b"not a tar at all")
    with pytest.raises(ArtifactError, match="not a readable tar"):
        unpack(junk, into=tmp_path / "out")


# --------------------------------------------------------------------------
# The digest, and the ordering around it
# --------------------------------------------------------------------------


def test_a_verified_download_lands_under_its_final_name(tmp_path, monkeypatch):
    """And an unverified one never does. There is no state in which the artifact path exists and its
    bytes are unchecked, which is what lets `unpack` treat verification as a precondition."""
    payload = b"a bundle" * 100
    _serve(monkeypatch, payload)
    verified = fetch_and_verify(
        "https://example.test/bundle.tar.gz",
        expected_digest=digest_of(payload),
        into=tmp_path,
    )
    assert verified.name == "artifact.tar.gz"
    assert verified.read_bytes() == payload
    assert not (tmp_path / "artifact.partial").exists()


def test_a_mismatched_digest_discards_the_download_unopened(tmp_path, monkeypatch):
    _serve(monkeypatch, b"different bytes")
    with pytest.raises(ArtifactError, match="is discarded unopened"):
        fetch_and_verify(
            "https://example.test/bundle.tar.gz",
            expected_digest=digest_of(b"what was committed"),
            into=tmp_path,
        )
    assert not (tmp_path / "artifact.tar.gz").exists()
    assert not (tmp_path / "artifact.partial").exists()


def test_the_byte_cap_stops_the_transfer_rather_than_checking_afterwards(tmp_path, monkeypatch):
    """A response with no Content-Length would otherwise fill the disk before any size check ran."""
    _serve(monkeypatch, b"x" * 5_000)
    with pytest.raises(ArtifactError, match="exceeded 1000 bytes"):
        fetch_and_verify(
            "https://example.test/bundle.tar.gz",
            expected_digest=digest_of(b"x" * 5_000),
            into=tmp_path,
            limits=FetchLimits(maximum_download_bytes=1_000),
        )
    assert not (tmp_path / "artifact.partial").exists()


def test_a_non_200_is_a_refusal_rather_than_an_empty_bundle(tmp_path, monkeypatch):
    _serve(monkeypatch, b"", status=404)
    with pytest.raises(ArtifactError, match="HTTP 404"):
        fetch_and_verify(
            "https://example.test/gone.tar.gz", expected_digest=digest_of(b""), into=tmp_path
        )


def test_a_redirect_to_a_private_address_is_refused(tmp_path, monkeypatch):
    """The reason redirects are followed by hand. `follow_redirects=True` would let a public URL
    bounce to a private one with the check having run only on the first."""
    _serve(monkeypatch, b"", status=302, location="https://127.0.0.1/bundle.tar.gz")
    with pytest.raises(ArtifactError, match="not a public address"):
        fetch_and_verify(
            "https://example.test/bundle.tar.gz",
            expected_digest=digest_of(b""),
            into=tmp_path,
        )


def test_a_redirect_loop_is_capped(tmp_path, monkeypatch):
    _serve(monkeypatch, b"", status=302, location="https://example.test/again")
    with pytest.raises(ArtifactError, match="redirected more than"):
        fetch_and_verify(
            "https://example.test/bundle.tar.gz",
            expected_digest=digest_of(b""),
            into=tmp_path,
            limits=FetchLimits(maximum_redirects=2),
        )


# --------------------------------------------------------------------------
# The two digest forms, which is what a real run found
# --------------------------------------------------------------------------


@pytest.mark.parametrize("committed", ["prefixed", "bare"])
def test_a_digest_is_accepted_in_either_form(committed, tmp_path, monkeypatch):
    """Found by building a real bundle and running it through `prepare_all`, not by reading.

    Two representations of one value, both deliberate: `digest_object` and the 5.2 manifests write
    `sha256:<hex>`, and `protocol.commitments` strips the prefix because an on-chain commitment pays
    for every byte. Comparing across them with `==` refused *every* submission — with a message
    saying the digest was malformed, which is the one explanation that is not true.
    """
    payload = b"a bundle"
    prefixed = digest_of(payload)
    _serve(monkeypatch, payload)
    verified = fetch_and_verify(
        "https://example.test/bundle.tar.gz",
        expected_digest=prefixed if committed == "prefixed" else prefixed.removeprefix("sha256:"),
        into=tmp_path,
    )
    assert verified.read_bytes() == payload


def test_a_wrong_digest_is_still_refused_in_either_form(tmp_path, monkeypatch):
    """The fix must not have made the comparison permissive."""
    _serve(monkeypatch, b"actual bytes")
    for form in (digest_of(b"other"), digest_of(b"other").removeprefix("sha256:")):
        with pytest.raises(ArtifactError, match="discarded unopened"):
            fetch_and_verify(
                "https://example.test/b.tar.gz", expected_digest=form, into=tmp_path
            )


def test_a_truncated_digest_is_refused_in_either_form(tmp_path):
    """An abbreviated digest becomes ambiguous as the set of artifacts grows."""
    for form in ("sha256:abcd", "abcd"):
        with pytest.raises(ArtifactError, match="not a sha256 digest"):
            fetch_and_verify(
                "https://example.test/b.tar.gz", expected_digest=form, into=tmp_path
            )


# --------------------------------------------------------------------------
# A fake transport, so these run without a network
# --------------------------------------------------------------------------


def _serve(monkeypatch, payload: bytes, *, status: int = 200, location: str = "") -> None:
    """Answer every request with one response, and skip the address check.

    The address check is *not* disabled — it is narrowed to let `example.test` through, which does
    not resolve, and to apply the real rule to everything else. Disabling it outright would make the
    redirect test below pass while checking nothing, which is exactly the failure this whole file
    exists to catch elsewhere.
    """
    import httpx

    from validator import artifacts

    real = artifacts._assert_public

    def narrowed(url: str) -> None:
        if url.startswith("https://example.test/"):
            return
        real(url)

    monkeypatch.setattr(artifacts, "_assert_public", narrowed)

    def handler(request: httpx.Request) -> httpx.Response:
        headers = {"location": location} if location else {}
        return httpx.Response(status, content=payload, headers=headers)

    transport = httpx.MockTransport(handler)
    original = httpx.Client

    def client(*args, **kwargs):
        kwargs["transport"] = transport
        return original(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", client)


def test_the_fake_transport_does_not_skip_the_check_for_the_tests_that_need_it(tmp_path):
    """The fake above narrows `_assert_public` rather than removing it. This asserts the real rule
    is in force by default, so a test that forgot to install the fake fails on the check rather than
    reaching the network."""
    with pytest.raises(ArtifactError, match="not a public address|cannot resolve"):
        fetch_and_verify(
            "https://127.0.0.1/x.tar.gz", expected_digest=digest_of(b""), into=tmp_path
        )
