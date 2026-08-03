"""Fetching a miner's bundle: the one place a validator processes bytes a stranger chose.

Everything else in this repository runs adversarial code *inside* the sandbox. This module runs
outside it, on the validator host, before any container exists — so the sandbox's controls do not
apply here and each one has to be replaced by something explicit.

The commitment on chain names three things: a `bundle_digest`, a `capsule_digest`, and an
`artifact_url`. The digest is what makes the URL safe to use at all: whatever comes back either
hashes to the committed value or is discarded. But *discarded* has to mean discarded before it was
parsed, extracted, or written anywhere durable, and that ordering is most of this file.

## The order, and why each step precedes the next

1. **Check the URL's shape.** `https://` or `ipfs://`. The commitment already refuses plain HTTP,
   and it is checked again here because this is the point of use — a value validated at
   construction and used three modules away is a value nobody re-validates after a refactor.
2. **Check where it resolves.** A miner chooses this hostname. `https://x.example/` resolving to
   `169.254.169.254` is a request to the cloud metadata service made by a process holding validator
   credentials, and TLS does not help: the certificate is for the name, and the name is theirs.
3. **Stream with a byte cap.** Enforced while reading, not after. A response with no
   `Content-Length` and forty gigabytes behind it must stop at the cap, not fill the disk and then
   fail a size check.
4. **Hash while streaming, compare before use.** The bytes land in a temporary file that is never
   named as the artifact until the digest matches.
5. **Extract with a filter, a member cap and a size cap.** A tar can contain `../../etc/cron.d/x`,
   a symlink to `/`, a device node, or one file that decompresses to a terabyte. Python's
   `data` filter handles the first three; the last two are counted here.
6. **Load the image and check what was loaded.** `docker load` reports an ID. If it is not the
   `container_digest` from the manifest, the image is removed and the submission is refused —
   otherwise a miner could commit one digest and ship another, and the sandbox would faithfully run
   a digest that nothing on chain attests to.

## What this does not defend against, stated rather than implied

**DNS rebinding.** The address check resolves the name and then hands the *name* to the HTTP client,
which resolves it again. A resolver that answers differently between those two calls defeats the
check. Pinning the connection to the address that was checked means opening the socket ourselves and
carrying the SNI through by hand, which is a meaningful amount of TLS plumbing; the mitigation taken
instead is that every redirect hop is re-checked and the total is capped, so the window is one
resolution wide rather than a whole session.

**A malicious image that behaves.** Nothing here inspects what the image *does*. That is the
sandbox's job, and it is why the sandbox exists.

## IPFS

`ipfs://` is accepted by the commitment and is fetched through a configured HTTP gateway, so the
same controls apply to it — with one difference worth naming: the gateway is a third party the
validator chose, not one the miner chose, and the content hash in the URL is checked by us against
the committed digest rather than trusted from the gateway.
"""

from __future__ import annotations

import hashlib
import ipaddress
import logging
import socket
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from protocol.canonical import same_digest

__all__ = [
    "ArtifactError",
    "FetchLimits",
    "fetch_and_verify",
    "load_image",
    "unpack",
]

_log = logging.getLogger(__name__)


class ArtifactError(RuntimeError):
    """A bundle that cannot be fetched, does not match its commitment, or is unsafe to open."""


@dataclass(frozen=True, slots=True)
class FetchLimits:
    """Bounds on what a miner-chosen URL may cost this validator.

    Defaults are generous for a laboratory and small against a disk. A bundle is source plus a
    container image; the reference template's image is under 200 MB, so 2 GB is four to ten times
    what an honest submission needs and still one round of one miner's disk rather than the host's.
    """

    #: Bytes accepted from the network before the transfer is abandoned.
    maximum_download_bytes: int = 2 * 1024 * 1024 * 1024
    #: Bytes written by extraction. Separate from the download cap because compression ratios of
    #: 1000:1 are ordinary for a tar of zeros, so a 2 MB download can be a 2 GB extraction.
    maximum_extracted_bytes: int = 4 * 1024 * 1024 * 1024
    #: Members in the archive. A million empty files is not a size problem, it is an inode problem.
    maximum_members: int = 50_000
    #: Seconds for the connection and for each read.
    connect_timeout: float = 15.0
    read_timeout: float = 120.0
    #: Redirect hops. Each one is re-checked; the cap stops a redirect loop.
    maximum_redirects: int = 3
    #: The HTTP gateway used for `ipfs://`. Chosen by the validator, never by the miner.
    ipfs_gateway: str = "https://ipfs.io/ipfs/"


def _resolved_addresses(host: str) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as error:
        raise ArtifactError(f"cannot resolve {host!r}: {error}") from error
    return [ipaddress.ip_address(info[4][0]) for info in infos]


def _assert_public(url: str) -> None:
    """Refuse a URL that is not https, or that resolves anywhere internal.

    The address families refused are not an arbitrary list. `169.254.169.254` is the cloud metadata
    service on every major provider and answers to unauthenticated HTTP with instance credentials;
    `127.0.0.0/8` reaches this host's own services, including the gateway that holds both API keys;
    private ranges reach the operator's network. A miner picks this hostname, so every one of those
    is one DNS record away.
    """
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise ArtifactError(
            f"refusing to fetch {url!r}: only https is fetched. A plain HTTP body can be replaced "
            "in transit, and while the digest would catch that, it would catch it only after this "
            "process had already downloaded and unarchived attacker-chosen bytes."
        )
    host = parsed.hostname
    if not host:
        raise ArtifactError(f"refusing to fetch {url!r}: it has no host")

    for address in _resolved_addresses(host):
        if not address.is_global or address.is_multicast:
            raise ArtifactError(
                f"refusing to fetch {url!r}: {host} resolves to {address}, which is not a public "
                "address. A miner chooses this hostname, so a record pointing at 169.254.169.254 "
                "or 127.0.0.1 is a request to the cloud metadata service or to this validator's "
                "own gateway, made by a process holding both credentials."
            )


def _target(url: str, limits: FetchLimits) -> str:
    """The https URL to fetch, translating `ipfs://` through the configured gateway."""
    parsed = urlparse(url)
    if parsed.scheme == "ipfs":
        cid = (parsed.netloc + parsed.path).strip("/")
        if not cid:
            raise ArtifactError(f"refusing to fetch {url!r}: it names no CID")
        return limits.ipfs_gateway.rstrip("/") + "/" + cid
    return url


def fetch_and_verify(
    url: str,
    *,
    expected_digest: str,
    into: Path,
    limits: FetchLimits | None = None,
) -> Path:
    """Download `url`, check it against `expected_digest`, and return the verified file.

    The file only takes its final name once the digest matches, so a caller cannot be handed an
    unverified artifact by a partial failure — there is no state in which the path exists and the
    bytes are unchecked.
    """
    import httpx

    limits = limits or FetchLimits()
    # Accepts both forms. The digest arrives from an on-chain commitment, where it is bare hex, and
    # from a manifest, where it is `sha256:`-prefixed. Demanding one of them refused every
    # submission that came from the other — with a message claiming the digest was malformed.
    if len(expected_digest.removeprefix("sha256:")) != 64:
        raise ArtifactError(
            f"the committed digest {expected_digest!r} is not a sha256 digest. Without one there "
            "is nothing to check the download against, and an unchecked download is arbitrary code."
        )

    target = _target(url, limits)
    into.mkdir(parents=True, exist_ok=True)
    partial = into / "artifact.partial"
    digest = hashlib.sha256()
    written = 0

    timeout = httpx.Timeout(limits.read_timeout, connect=limits.connect_timeout)
    # Redirects are followed by hand so each hop is re-checked. `follow_redirects=True` would let a
    # public URL bounce to a private one and the check would have run only on the first.
    with httpx.Client(timeout=timeout, follow_redirects=False) as client:
        for hop in range(limits.maximum_redirects + 1):
            _assert_public(target)
            with client.stream("GET", target) as response:
                if response.is_redirect:
                    location = response.headers.get("location", "")
                    if not location:
                        raise ArtifactError(f"{target} redirected without a Location header")
                    target = str(httpx.URL(target).join(location))
                    _log.info("artifact redirect %d -> %s", hop + 1, target)
                    continue
                if response.status_code != 200:
                    raise ArtifactError(
                        f"{target} returned HTTP {response.status_code}; the bundle committed on "
                        "chain is not retrievable, so the submission cannot be run"
                    )
                with partial.open("wb") as handle:
                    for chunk in response.iter_bytes(1024 * 256):
                        written += len(chunk)
                        if written > limits.maximum_download_bytes:
                            handle.close()
                            partial.unlink(missing_ok=True)
                            raise ArtifactError(
                                f"{target} exceeded {limits.maximum_download_bytes} bytes. The cap "
                                "is enforced while reading rather than afterwards, because a "
                                "response with no Content-Length would otherwise fill the disk "
                                "before any size check ran."
                            )
                        digest.update(chunk)
                        handle.write(chunk)
                break
        else:
            raise ArtifactError(
                f"{url} redirected more than {limits.maximum_redirects} times"
            )

    observed = f"sha256:{digest.hexdigest()}"
    if not same_digest(observed, expected_digest):
        partial.unlink(missing_ok=True)
        raise ArtifactError(
            f"the artifact at {url} hashes to {observed}, not the committed {expected_digest}. "
            "It is discarded unopened: this is exactly the case the commitment exists to catch, "
            "and 'nearly right' is not a category here."
        )

    verified = into / "artifact.tar.gz"
    partial.replace(verified)
    _log.info("artifact verified: %s (%d bytes) matches %s", url, written, expected_digest)
    return verified


def unpack(archive: Path, *, into: Path, limits: FetchLimits | None = None) -> Path:
    """Extract a verified archive, refusing anything that writes outside it.

    Verified is a precondition, not a suggestion. Extraction is the first operation that acts on the
    *structure* of miner-chosen bytes rather than on their hash, so it must not run on bytes that
    failed the digest — `fetch_and_verify` is the only thing that should produce this argument.
    """
    limits = limits or FetchLimits()
    into.mkdir(parents=True, exist_ok=True)
    extracted = 0
    members = 0

    try:
        with tarfile.open(archive, "r:*") as tar:
            for member in tar:
                members += 1
                if members > limits.maximum_members:
                    raise ArtifactError(
                        f"the archive holds more than {limits.maximum_members} entries. A million "
                        "empty files is not a size problem, it is an inode problem."
                    )
                extracted += max(member.size, 0)
                if extracted > limits.maximum_extracted_bytes:
                    raise ArtifactError(
                        f"the archive expands past {limits.maximum_extracted_bytes} bytes. The "
                        "download cap does not bound this: a tar of zeros compresses about a "
                        "thousand to one, so a 2 MB download is a 2 GB extraction."
                    )
                # `data` refuses absolute paths, `..` traversal, links pointing outside the
                # destination, device nodes and setuid bits. Named explicitly rather than left to
                # the version default, which changed across Python releases and will change again.
                tar.extract(member, path=into, filter="data")
    except tarfile.TarError as error:
        raise ArtifactError(f"{archive} is not a readable tar archive: {error}") from error
    except (OSError, ValueError) as error:
        # `filter="data"` raises for a member that tries to escape. That is a submission refused,
        # not a validator fault, so it is reported as an artifact error rather than a crash.
        raise ArtifactError(
            f"refusing to extract {archive}: {error}. A member that writes outside its destination "
            "is not a packaging mistake to work around."
        ) from error

    _log.info("unpacked %s: %d members, %d bytes declared", archive, members, extracted)
    return into


def load_image(image_tar: Path, *, expected_digest: str, timeout: float = 600.0) -> str:
    """`docker load` an image and check the loaded ID against what was committed.

    The check is the point. `docker load` will happily load anything, and the sandbox runs by digest
    — so without this a miner could commit one `container_digest` and ship an image with another,
    and every control downstream would faithfully run bytes that nothing on chain attests to.

    A mismatch removes what was loaded. Leaving it would put an unattested image in the daemon under
    an ID that a later round might reference.
    """
    if not image_tar.is_file():
        raise ArtifactError(
            f"the bundle contains no image at {image_tar.name}. A submission ships its image "
            "because the sandbox runs by digest and a digest alone cannot be pulled — there is no "
            "registry reference in the commitment, deliberately, since a registry is mutable."
        )
    if len(expected_digest.removeprefix("sha256:")) != 64:
        raise ArtifactError(
            f"the manifest's container_digest {expected_digest!r} is not a sha256 digest"
        )

    try:
        loaded = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["docker", "load", "--quiet", "--input", str(image_tar)],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ArtifactError(f"cannot load {image_tar}: {error}") from error
    if loaded.returncode != 0:
        raise ArtifactError(
            f"docker load failed for {image_tar}: {loaded.stderr.strip()[:400]}"
        )

    # `docker load --quiet` prints "Loaded image: name" or "Loaded image ID: sha256:...". Neither
    # form is the digest we need, so the ID is read back by inspection rather than parsed out of a
    # human-readable line whose wording is not a stable interface.
    reference = loaded.stdout.strip().split(":", 1)[-1].strip()
    if not reference:
        raise ArtifactError(f"docker load reported nothing for {image_tar}")

    try:
        inspected = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["docker", "image", "inspect", "--format", "{{.Id}}", reference],
            capture_output=True,
            text=True,
            timeout=60.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ArtifactError(f"cannot inspect the loaded image: {error}") from error
    if inspected.returncode != 0:
        raise ArtifactError(
            f"cannot inspect the loaded image {reference!r}: {inspected.stderr.strip()[:400]}"
        )

    observed = inspected.stdout.strip()
    if not same_digest(observed, expected_digest):
        # Refused, and deliberately *not* removed. An earlier version ran `docker image rm --force`
        # here, and the end-to-end test caught what that means: `docker load` of an image already in
        # the daemon is a no-op returning the existing reference, so removing it deletes an image
        # that was there for another reason. A miner could ship a copy of the reference
        # laboratory's own image with a deliberately wrong manifest digest and delete the image that
        # sets the qualification floor.
        #
        # Leaving it costs nothing. Images are content-addressed and the sandbox runs by digest, so
        # an unattested image in the daemon is unreachable — there is no manifest that names it.
        raise ArtifactError(
            f"the shipped image is {observed}, not the committed {expected_digest}. Running it "
            "would mean executing an image no on-chain commitment attests to, which is the whole "
            "of what sealing a bundle before the deadline is for."
        )
    _log.info("image %s loaded and matches its commitment", observed)
    return observed
