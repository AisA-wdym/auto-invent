"""Turning on-chain commitments into laboratories that can be run.

Between "there is a commitment" and "there is a container to execute" sit six things that can
each go wrong quietly: the commitment can be for another round, the uid can have changed hands, the
artifact can not match its digest, the archive can not contain what it claims, the image can not be
the one committed, and the credential envelope can not open. Every one of those is a submission that
is *refused*, individually and by name — never a round that fails.

## One admitted laboratory failing must not end the round

That is the shape of this module. `prepare_all` returns what it prepared and what it refused, and
the caller runs the first list and publishes the second. A submission that raises out of here would
take every other miner's day with it, turning one miner's packaging mistake into everybody's burn.

## The uid is resolved once, at one height

`SubnetView` is a snapshot for exactly this reason. Reading the neuron set at one block and the
commitments at another means that when a miner deregisters in between, its commitment resolves to a
uid now owned by someone else — and that someone else is scored on a bundle they did not submit. So
the view is passed in, not fetched here, and the uid comes from the same object as the commitment.

## The credential is opened as late as possible and never written down

`chain.unseal` produces the miner's provider key. It goes into `Prepared.api_key`, is handed to
the gateway's admission call, and exists nowhere else — not in the round state, not in a log line,
not on disk. `Prepared.__repr__` is overridden because a dataclass repr in a traceback is how a key
reaches a log file that gets published.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chain.client import ChainClient, ChainError, SubnetView
from protocol.commitments import SubmissionCommitment
from validator.artifacts import ArtifactError, FetchLimits, fetch_and_verify, load_image, unpack

__all__ = ["Prepared", "Refused", "Preparation", "prepare_all", "submissions_for"]

_log = logging.getLogger(__name__)

#: What a bundle archive must contain. Named here because it is the interface between `ail-miner
#: seal` and this module, and an archive missing one of them is a submission refused rather than a
#: validator error.
MANIFEST_NAME = "manifest.json"
IMAGE_NAME = "image.tar"


@dataclass(frozen=True, slots=True)
class Prepared:
    """A laboratory ready to run: verified image, opened credential, resolved identity."""

    uid: int
    hotkey: str
    bundle_digest: str
    image_digest: str
    manifest: dict[str, Any]
    api_key: str
    declared_spend_cap_usd: int
    root: Path

    def __repr__(self) -> str:
        # The default dataclass repr prints every field, and one of them is a provider credential.
        # A traceback anywhere below this point would put it in a log, and logs from a round are
        # published (6.3, 22).
        return (
            f"Prepared(uid={self.uid}, hotkey={self.hotkey!r}, "
            f"bundle_digest={self.bundle_digest!r}, image_digest={self.image_digest!r}, "
            "api_key=<redacted>)"
        )


@dataclass(frozen=True, slots=True)
class Refused:
    """A submission that will not be run, and the reason a miner can act on."""

    uid: int
    hotkey: str
    reason: str


@dataclass(frozen=True, slots=True)
class Preparation:
    """The outcome of preparing a round's submissions."""

    ready: tuple[Prepared, ...] = ()
    refused: tuple[Refused, ...] = ()
    #: Digests seen more than once, with the uid that kept each. See `prepare_all`.
    duplicates: tuple[tuple[str, int], ...] = field(default=())


def submissions_for(
    view: SubnetView, *, round_id: str
) -> list[tuple[int, str, SubmissionCommitment]]:
    """Every submission commitment for one round, as (uid, hotkey, commitment).

    Filtered by `round_id` rather than by recency. The commitments pallet keeps one slot per hotkey
    and overwrites it, so a miner who submitted yesterday and not today still has yesterday's
    commitment on chain — and running it would score a laboratory against a pack it was not
    submitted for, which is both unfair and unmeasurable.
    """
    found: list[tuple[int, str, SubmissionCommitment]] = []
    for registered, decoded in view.parsed_commitments():
        if not isinstance(decoded, SubmissionCommitment):
            continue
        if decoded.round_id != round_id:
            _log.debug(
                "uid %d committed for round %s, not %s; skipped",
                registered.uid,
                decoded.round_id,
                round_id,
            )
            continue
        found.append((registered.uid, registered.hotkey, decoded))
    return found


def prepare_all(
    view: SubnetView,
    *,
    round_id: str,
    chain: ChainClient,
    workspace: Path,
    limits: FetchLimits | None = None,
) -> Preparation:
    """Fetch, verify and open every submission for a round.

    Deduplicates by `bundle_digest` before doing any work. Two hotkeys committing identical bytes is
    the cheapest sybil there is — register twice, submit the same laboratory, take two shares of a
    softmax — and it is also indistinguishable from an honest fork, which is why the response is to
    keep one rather than to punish. The one kept is the lowest uid, because uid order is
    registration order: the original keeps its place and the copy is refused by name.
    """
    limits = limits or FetchLimits()
    ready: list[Prepared] = []
    refused: list[Refused] = []
    duplicates: list[tuple[str, int]] = []
    kept_by_digest: dict[str, int] = {}

    for uid, hotkey, commitment in sorted(submissions_for(view, round_id=round_id)):
        first = kept_by_digest.get(commitment.bundle_digest)
        if first is not None:
            duplicates.append((commitment.bundle_digest, first))
            refused.append(
                Refused(
                    uid,
                    hotkey,
                    f"identical bundle to uid {first} ({commitment.bundle_digest}). One copy is "
                    "run. Two hotkeys submitting the same bytes take two shares of one result, "
                    "and that is true whether it is a sybil or an honest fork.",
                )
            )
            continue
        kept_by_digest[commitment.bundle_digest] = uid

        try:
            ready.append(
                _prepare_one(
                    uid=uid,
                    hotkey=hotkey,
                    commitment=commitment,
                    chain=chain,
                    workspace=workspace / f"uid-{uid}",
                    limits=limits,
                )
            )
        except (ArtifactError, ChainError, ValueError, OSError) as error:
            # One submission's failure is one submission refused. A raise here would take the round
            # down with it and turn one miner's packaging mistake into everybody's burn.
            _log.warning("uid %d refused: %s", uid, error)
            refused.append(Refused(uid, hotkey, str(error)))

    _log.info(
        "round %s: %d laboratories ready, %d refused", round_id, len(ready), len(refused)
    )
    return Preparation(tuple(ready), tuple(refused), tuple(duplicates))


def _prepare_one(
    *,
    uid: int,
    hotkey: str,
    commitment: SubmissionCommitment,
    chain: ChainClient,
    workspace: Path,
    limits: FetchLimits,
) -> Prepared:
    """One submission, in the order the checks have to happen in."""
    archive = fetch_and_verify(
        commitment.artifact_url,
        expected_digest=commitment.bundle_digest,
        into=workspace,
        limits=limits,
    )
    root = unpack(archive, into=workspace / "bundle", limits=limits)

    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise ArtifactError(
            f"the bundle contains no {MANIFEST_NAME}. It names the entrypoint, the image digest "
            "and the model manifest, so without it there is nothing to run and nothing to check."
        )
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactError(f"{MANIFEST_NAME} is not readable JSON: {error}") from error
    if not isinstance(manifest, dict):
        raise ArtifactError(f"{MANIFEST_NAME} is a {type(manifest).__name__}, not an object")

    if manifest.get("round_id") not in (round_id_of(commitment), None, ""):
        # A bundle sealed for another round. The digest matches because the miner committed these
        # bytes — they simply committed the wrong ones, and running them would test a laboratory
        # against a pack it was not built for.
        raise ArtifactError(
            f"the manifest is sealed for round {manifest.get('round_id')!r}, but the commitment is "
            f"for {round_id_of(commitment)!r}"
        )

    image_digest = load_image(
        root / IMAGE_NAME, expected_digest=str(manifest.get("container_digest", ""))
    )

    api_key, spend_cap = _open_credential(
        root, commitment=commitment, chain=chain
    )
    return Prepared(
        uid=uid,
        hotkey=hotkey,
        bundle_digest=commitment.bundle_digest,
        image_digest=image_digest,
        manifest=manifest,
        api_key=api_key,
        declared_spend_cap_usd=spend_cap,
        root=root,
    )


def round_id_of(commitment: SubmissionCommitment) -> str:
    return commitment.round_id


def _open_credential(
    root: Path, *, commitment: SubmissionCommitment, chain: ChainClient
) -> tuple[str, int]:
    """Open the sealed credential envelope, checking it is the one committed to.

    The capsule digest is checked *before* unsealing. A capsule that opens is not evidence that
    it is the right capsule: the envelope travels beside the bundle in the same archive, so a
    substituted envelope would decrypt perfectly and bill a different account — which under one
    provider surface succeeds silently, which is why 5.4.1 puts the digest on chain.
    """
    envelope_path = root / "credential_envelope.json"
    if not envelope_path.is_file():
        raise ArtifactError(
            "the bundle contains no credential_envelope.json. The laboratory runs on the miner's "
            "own account (3.4.2), so without it there is nothing to spend and nothing to run."
        )
    try:
        envelope = json.loads(envelope_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactError(f"credential_envelope.json is not readable JSON: {error}") from error

    capsule = str(envelope.get("key_capsule", ""))
    if not capsule:
        raise ArtifactError(
            "the credential envelope has an empty key_capsule, so this laboratory has no account "
            "to spend from. `ail-miner seal` leaves it empty on purpose and `ail-miner submit` "
            "refuses to publish without it."
        )

    from protocol.canonical import digest_object, same_digest

    observed = digest_object({"key_capsule": capsule, "nonce": str(envelope.get("nonce", ""))})
    if not same_digest(observed, commitment.capsule_digest):
        raise ArtifactError(
            f"the credential envelope hashes to {observed}, not the committed "
            f"{commitment.capsule_digest}. A capsule that decrypts is not evidence that it is the "
            "right capsule — it travels in the same archive as the bundle, so a substituted one "
            "would open perfectly and bill a different account."
        )

    try:
        opened = chain.unseal(bytes.fromhex(capsule))
    except ValueError as error:
        raise ArtifactError(f"the key capsule is not hex: {error}") from error

    api_key = opened.decode("utf-8", errors="strict").strip()
    if not api_key:
        raise ArtifactError("the key capsule opened to an empty string")

    try:
        spend_cap = int(envelope.get("declared_spend_cap_usd", 0))
    except (TypeError, ValueError):
        spend_cap = 0
    return api_key, spend_cap
