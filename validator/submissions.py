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
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chain.client import ChainClient, ChainError, SubnetView
from protocol.commitments import SubmissionCommitment
from validator.artifacts import ArtifactError, FetchLimits, fetch_and_verify, load_image, unpack

__all__ = [
    "Prepared",
    "Preparation",
    "Refused",
    "prepare_all",
    "revoke_minted",
    "submissions_for",
]

#: What a miner may declare their credential to be. Not a free-text field: the two are handled
#: differently, and an unrecognised value is refused rather than defaulted.
_CREDENTIAL_KINDS = frozenset({"runtime", "management"})

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
    #: Set when `api_key` was minted from a management key for this round. Carries what revocation
    #: needs — the provider's hash, and the management key that can delete it. Both are held so the
    #: round can revoke without re-reading the envelope, which by then has been unpacked into a
    #: workspace that may already be gone.
    minted_key_hash: str = ""
    management_key: str = field(default="", repr=False)

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

    opened = _open_credential(root, commitment=commitment, chain=chain, uid=uid)
    return Prepared(
        uid=uid,
        hotkey=hotkey,
        bundle_digest=commitment.bundle_digest,
        image_digest=image_digest,
        manifest=manifest,
        api_key=opened.api_key,
        declared_spend_cap_usd=opened.spend_cap_usd,
        root=root,
        minted_key_hash=opened.minted_key_hash,
        management_key=opened.management_key,
    )


def round_id_of(commitment: SubmissionCommitment) -> str:
    return commitment.round_id


@dataclass(frozen=True, slots=True)
class _Opened:
    """What came out of the credential envelope, ready to spend."""

    api_key: str = field(repr=False)
    spend_cap_usd: int
    minted_key_hash: str = ""
    management_key: str = field(default="", repr=False)


def _open_credential(
    root: Path, *, commitment: SubmissionCommitment, chain: ChainClient, uid: int
) -> _Opened:
    """Open the sealed credential envelope, checking it is the one committed to.

    The capsule digest is checked *before* unsealing. A capsule that opens is not evidence that it
    is the right capsule: the envelope travels beside the bundle in the same archive, so a
    substituted envelope would decrypt perfectly and bill a different account — which under one
    provider surface succeeds silently, which is why 5.4.1 puts the digest on chain.

    ## Two kinds of credential, and the miner says which

    `credential_kind` is `"runtime"` or `"management"`. A runtime key is spendable balance and is
    used directly, as before. A management key cannot spend at all — it can only mint — so a
    round-scoped key is minted from it, capped at the declared spend cap and set to expire.

    The declaration is *verified*, not trusted: one free listing call distinguishes them. A miner
    who declares one and supplies the other is told here, once, rather than by every inference call
    failing for a reason that does not name the cause.
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
        opened_bytes = chain.unseal(bytes.fromhex(capsule))
    except ValueError as error:
        raise ArtifactError(f"the key capsule is not hex: {error}") from error

    credential = opened_bytes.decode("utf-8", errors="strict").strip()
    if not credential:
        raise ArtifactError("the key capsule opened to an empty string")

    try:
        spend_cap = int(envelope.get("declared_spend_cap_usd", 0))
    except (TypeError, ValueError):
        spend_cap = 0

    kind = str(envelope.get("credential_kind", "runtime")).lower()
    if kind not in _CREDENTIAL_KINDS:
        raise ArtifactError(
            f"credential_kind is {kind!r}; it must be one of {sorted(_CREDENTIAL_KINDS)}. It is "
            "not defaulted, because the two are handled differently and guessing wrong means "
            "either an unminted management key that cannot make a single call, or a runtime key "
            "used with no provider-side cap."
        )

    if kind == "runtime":
        return _Opened(api_key=credential, spend_cap_usd=spend_cap)

    return _mint_for_round(credential, spend_cap_usd=spend_cap, uid=uid)


def _mint_for_round(management_key: str, *, spend_cap_usd: int, uid: int) -> _Opened:
    """Turn a management key into a round-scoped, capped, expiring runtime key."""
    from gateway.provisioning import ProvisioningError, is_management_key, mint_round_key

    try:
        if not is_management_key(management_key):
            raise ArtifactError(
                "the envelope declares credential_kind 'management', but this credential cannot "
                "list keys — it is a runtime key. Either declare 'runtime', or supply a management "
                "key from https://openrouter.ai/settings/provisioning-keys. Checked here because "
                "the alternative is every inference call failing for a reason that does not name "
                "the cause."
            )
        if spend_cap_usd <= 0:
            raise ArtifactError(
                "credential_kind 'management' needs a positive declared_spend_cap_usd: it becomes "
                "the minted key's hard credit limit, and there is no safe value to guess. Zero "
                "would either mint a key that cannot run the laboratory or, if the provider reads "
                "it as absent, an uncapped key on the miner's account."
            )
        minted = mint_round_key(
            management_key, name=f"auto-invent-uid{uid}", limit_usd=float(spend_cap_usd)
        )
    except ProvisioningError as error:
        # Refused, not degraded. A management key cannot make an inference call, so there is no
        # fallback mode — using it directly is a 401 on every call and a laboratory scored as
        # having produced nothing, which reads as the miner's fault.
        raise ArtifactError(
            f"could not mint a round key from the management key: {error}"
        ) from error

    return _Opened(
        api_key=minted.secret,
        spend_cap_usd=spend_cap_usd,
        minted_key_hash=minted.key_hash,
        management_key=management_key,
    )


def revoke_minted(laboratories: Sequence[Prepared]) -> int:
    """Delete every key minted for a round. Returns how many were confirmed gone.

    Called when execution closes, whatever the round's outcome. A key that outlives its round is
    spendable balance sitting on a miner's account with a validator holding the secret — which is
    the thing minting was meant to avoid, so failing to revoke undoes the whole point.

    Never raises: this runs in the cleanup path of a round that may already be failing, and every
    minted key also carries an expiry precisely so a missed revocation is survivable.
    """
    from gateway.provisioning import revoke

    revoked = 0
    for lab in laboratories:
        if not lab.minted_key_hash or not lab.management_key:
            continue
        if revoke(lab.management_key, lab.minted_key_hash):
            revoked += 1
        else:
            _log.error(
                "uid %d: minted key %s was not revoked. It expires on its own, which is why the "
                "expiry is set, but until then the miner has a live key this validator created.",
                lab.uid,
                lab.minted_key_hash[:12],
            )
    if revoked:
        _log.info("revoked %d round-scoped key(s)", revoked)
    return revoked
