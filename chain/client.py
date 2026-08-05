"""The only module that imports `bittensor`. Everything else takes `ChainClient`.

## Why an interface at all

Two reasons, and the second is the one that decided it.

First, testability: the daily cycle in 21 has seven timed chain interactions, and a validator
that could only be tested against a live chain would be tested rarely. `FakeChain` makes the
whole orchestration testable in milliseconds.

Second, and more important: **the SDK redesigns.** This subnet was designed against the
`subtensor.set_weights(netuid, wallet, uids, weights, ...)` shape that every production subnet
in this ecosystem uses. Bittensor 11 replaced it with an intent model — `client.execute(
SetWeights(...), wallet)` — with commitments carried on the metagraph and timelock encryption
native. That is a better fit for this subnet than what it replaced, but a codebase that had
spread SDK calls across the validator, the miner CLI and the registry would have needed changing
in all three. Here it needs changing in one file.

So the interface is deliberately narrow: eight methods, named for what this subnet needs rather
than for what the chain offers.

## What bittensor 11 gives us that we no longer have to build

* `bt.set_weights(...)` conforms weights to the subnet's hyperparameters, quantises to u16,
  chooses plaintext or timelocked commit-reveal by reading whether the subnet enables it, and
  raises on failure rather than returning `False`. The reference subnets all hand-roll that; the
  hand-rolled versions are where their weight bugs live.
* `bittensor.timelock` is drand timelock encryption with round arithmetic. 6.1 needs exactly
  that, and building it on `bittensor-drand` directly would mean owning the round maths.
* `Metagraph.commitments` returns every hotkey's commitment *with the metagraph*, including
  whether a timelocked payload has been opened on chain, so reading submissions costs no extra
  query per hotkey.

## The failure mode this interface is shaped around

A validator that cannot reach the chain must **not** proceed on stale state. Weight submission
that silently reuses yesterday's vector pays yesterday's winners for today's work, and a
metagraph read that silently returns a cached copy runs laboratories that have deregistered. So
every read either succeeds or raises, and no method returns a plausible default.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, runtime_checkable

from protocol.commitments import CommitmentError, decode

__all__ = [
    "ChainClient",
    "ChainError",
    "BittensorChain",
    "FakeChain",
    "Neuron",
    "RegisteredCommitment",
    "SubnetView",
]

_log = logging.getLogger(__name__)

#: Bytes per commitment field. The pallet's `Data` enum is `Raw0`…`Raw128`, so 128 is the ceiling
#: and `Raw129` does not exist. Named because a longer payload is split rather than refused.
_COMMITMENT_FIELD_BYTES = 128


class ChainError(RuntimeError):
    """The chain could not be reached, or refused what we asked."""


@dataclass(frozen=True, slots=True)
class Neuron:
    """One registered participant, reduced to what this subnet uses.

    A projection rather than the SDK's neuron because the SDK's has thirty fields and we depend
    on five. Depending on five means an SDK that renames the other twenty-five does not touch us.
    """

    uid: int
    hotkey: str
    coldkey: str
    #: Stake on *this subnet*, in alpha. Not TAO: under dTAO a subnet's stake is denominated in its
    #: own alpha, and the SDK's `Balance.tao` raises `UnitMismatchError` rather than converting —
    #: correctly, because converting needs a price quote and a stake figure silently valued at a
    #: stale price is worse than one labelled in the unit it is actually in.
    #:
    #: It was `stake_tao` and read `.tao`, so *every* metagraph read raised on any real dTAO subnet.
    #: Found reading subnet 542's own metagraph.
    stake_alpha: float
    validator_permit: bool
    active: bool


@dataclass(frozen=True, slots=True)
class RegisteredCommitment:
    """A hotkey's on-chain commitment, with the block it was published at.

    `block` is carried because `protocol.commitments.verify_salt_timing` needs it: a salt
    commitment is only evidence if it predates the randomness it was mixed with, and the
    commitment's own bytes cannot say when they were written.
    """

    uid: int
    hotkey: str
    raw: str
    block: int
    #: True while a timelocked payload has not yet been opened on chain.
    encrypted: bool = False
    reveal_round: int | None = None


@dataclass(frozen=True, slots=True)
class SubnetView:
    """One consistent snapshot: block height plus every neuron and commitment at it.

    A snapshot rather than live accessors, because the alternative is a validator that reads the
    neuron set at one block and the commitments at another. When a miner deregisters between
    those two reads, its commitment resolves to a uid now owned by someone else — and that
    someone else gets scored on a bundle they did not submit.
    """

    netuid: int
    mechid: int
    block: int
    neurons: tuple[Neuron, ...]
    commitments: tuple[RegisteredCommitment, ...]
    #: The subnet owner's coldkey, for 23's owner-eligibility rule. Carried on the snapshot
    #: rather than fetched where it is used, so the owner and the neuron set are read at one
    #: height — an owner coldkey read at a different block than the neurons it is compared against
    #: would, on the block an owner transfers the subnet, exclude the wrong hotkeys.
    #:
    #: Empty when the runtime does not expose it. Empty means *no exclusion*, the permissive
    #: direction, so `owner_linked_hotkeys` says so rather than returning a quietly empty set.
    owner_coldkey: str = ""

    def owner_linked_hotkeys(self) -> frozenset[str]:
        """Every hotkey whose coldkey is the subnet owner's.

        23: "owner-linked miner UIDs should be ineligible". By *coldkey*, not by hotkey: an owner
        who wanted to self-mine would register a fresh hotkey rather than resubmit the one that owns
        the subnet, and a hotkey-only comparison would catch nobody. The coldkey pays for a
        registration and is therefore what links them.
        """
        if not self.owner_coldkey:
            return frozenset()
        return frozenset(
            neuron.hotkey for neuron in self.neurons if neuron.coldkey == self.owner_coldkey
        )

    def uid_of(self, hotkey: str) -> int | None:
        for neuron in self.neurons:
            if neuron.hotkey == hotkey:
                return neuron.uid
        return None

    def hotkeys(self) -> tuple[str, ...]:
        return tuple(neuron.hotkey for neuron in self.neurons)

    def validators(self) -> tuple[Neuron, ...]:
        return tuple(neuron for neuron in self.neurons if neuron.validator_permit)

    def parsed_commitments(self) -> list[tuple[RegisteredCommitment, Any]]:
        """Every commitment this protocol understands, with the rest logged and dropped.

        Other subnets' commitments and other protocol versions appear on this channel, so
        skipping is normal. It is logged at debug rather than silently, because "the miner never
        submitted" and "the miner submitted something malformed" need different responses and
        look identical from a filtered list.
        """
        parsed: list[tuple[RegisteredCommitment, Any]] = []
        for commitment in self.commitments:
            if commitment.encrypted:
                # Still sealed. Expected before the reveal point in 6.2, and not an error.
                continue
            try:
                parsed.append((commitment, decode(commitment.raw)))
            except CommitmentError as error:
                _log.debug(
                    "uid %d commitment not for this protocol (%s)", commitment.uid, error
                )
        return parsed


@runtime_checkable
class ChainClient(Protocol):
    """What the subnet needs from a chain. Eight methods, no more."""

    def current_block(self) -> int:
        """Head block height. Raises rather than returning a stale value."""
        ...

    def view(self) -> SubnetView:
        """One consistent snapshot of neurons and commitments."""
        ...

    def view_at(self, block: int) -> SubnetView:
        """The same, at a past height. For verifying when a salt was committed."""
        ...

    def block_hash(self, block: int) -> bytes:
        """The hash of one block. 7.3's post-deadline randomness.

        A read of a *past* block, always: the seed mixes a hash that exists only after the salt was
        committed, so asking for the head's hash would be asking for a value that is still changing.
        """
        ...

    def publish_commitment(self, payload: str) -> int:
        """Write this hotkey's commitment. Returns the block it landed in."""
        ...

    def submit_weights(self, uids: list[int], weights: list[int]) -> None:
        """Publish a weight vector. Raises on failure; never returns a bool."""
        ...

    def seal(self, plaintext: bytes, *, reveal_at_block: int) -> bytes:
        """Timelock-encrypt so it opens no earlier than that block."""
        ...

    def unseal(self, ciphertext: bytes) -> bytes:
        """Open a timelocked payload, or raise if its round has not arrived."""
        ...

    def hotkey(self) -> str:
        """This process's own hotkey."""
        ...


# --------------------------------------------------------------------------
# The real one
# --------------------------------------------------------------------------


@dataclass
class BittensorChain:
    """`ChainClient` over bittensor 11.

    The SDK client is built lazily and shared. Construction in bittensor 11 is cold — no socket,
    thread or loop exists until first use — so building one in `__init__` would still be free,
    but lazy construction means importing this module does not require a reachable endpoint, and
    the `--check` paths depend on that.
    """

    netuid: int
    wallet_name: str
    hotkey_name: str
    network: str = "finney"
    mechid: int = 0
    version_key: int = 0
    #: `bt.set_weights` retries transient pool rejections itself; this is on top of that, for a
    #: node that is briefly unreachable rather than briefly busy.
    submit_retries: int = 2
    _client: Any = field(default=None, repr=False)
    #: Cached subnet owner coldkey; `None` until first read, `""` when the runtime hides it.
    _owner: str | None = field(default=None, repr=False)
    _wallet: Any = field(default=None, repr=False)

    def _bt(self) -> Any:
        import bittensor

        return bittensor

    def _connected(self) -> Any:
        if self._client is None:
            self._client = self._bt().subtensor(self.network)
        return self._client

    def _signer(self) -> Any:
        if self._wallet is None:
            self._wallet = self._bt().Wallet(self.wallet_name, self.hotkey_name)
        return self._wallet

    def hotkey(self) -> str:
        """This process's own hotkey.

        A missing keyfile arrives here as a bare `FileNotFoundError` from deep inside the SDK, four
        frames below anything this repo wrote. On a first deployment that is the most likely failure
        of all, and a traceback naming `keyfiles.py` does not say which wallet was expected or how
        to make one — so it is turned into a message that does.
        """
        try:
            return str(self._signer().hotkey.ss58_address)
        except FileNotFoundError as error:
            raise ChainError(
                f"no hotkey for wallet {self.wallet_name!r}/{self.hotkey_name!r}: {error}. A "
                "validator signs its salt commitment, its pack commitment and its weight vector "
                "with this key, so there is nothing it can do without one. Create or import it "
                "with `btcli wallet new_hotkey`, or point --wallet/--hotkey at an existing pair."
            ) from error
        except Exception as error:  # noqa: BLE001 - the SDK raises a wide family here
            raise ChainError(
                f"cannot load the hotkey for {self.wallet_name!r}/{self.hotkey_name!r}: "
                f"{type(error).__name__}: {error}"
            ) from error

    def current_block(self) -> int:
        try:
            return int(self._connected().block)
        except Exception as error:  # noqa: BLE001 - SDK raises a wide family
            raise ChainError(
                f"could not read the head block from {self.network}: {error}. Proceeding on a "
                "stale height would run the wrong cycle phase."
            ) from error

    def view(self) -> SubnetView:
        return self._view(None)

    def view_at(self, block: int) -> SubnetView:
        return self._view(block)

    def _owner_coldkey(self) -> str:
        """The subnet owner's coldkey, for 23's owner-eligibility rule.

        Cached after the first successful read: it changes only when a subnet is transferred, and a
        query per metagraph read would add a round trip to the hottest read the validator makes.

        A runtime that does not expose it yields `""`, which means *no exclusion* — the permissive
        direction. Logged at warning rather than raised, because refusing to run a round over a
        missing storage item would take the subnet down over a control that is one of several, and
        `prepare_all` says out loud when the exclusion is not in force.
        """
        if self._owner is not None:
            return self._owner
        try:
            import bittensor as bt

            value = self._connected().query(bt.storage.SubtensorModule.SubnetOwner, [self.netuid])
            self._owner = str(value or "")
        except Exception as error:  # noqa: BLE001 - the SDK raises a wide family here
            _log.warning(
                "cannot read the owner coldkey of netuid %d (%s): 23's owner-eligibility rule "
                "cannot be applied this round, and an owner-linked laboratory would be scored like "
                "any other.",
                self.netuid,
                error,
            )
            self._owner = ""
        return self._owner

    def block_hash(self, block: int) -> bytes:
        """7.3's randomness, read from the chain rather than generated.

        Refuses a future or current height. The head's hash is not yet final — asking for it would
        return whatever the node has provisionally, and two validators asking at slightly different
        moments would seed their generation differently from a value that was supposed to be shared.
        """
        head = self.current_block()
        if block >= head:
            raise ChainError(
                f"block {block} is not in the past (head is {head}); its hash is not final. The "
                "daily seed mixes a settled hash, or two validators reading at different moments "
                "would derive different seeds from what is meant to be one shared value."
            )
        try:
            # `block_info(n).hash`, not `get_block_hash(n)` — bittensor 11.0.1 has no such method,
            # so this raised `AttributeError` and 7.3's randomness could never be read at all.
            # Every round dies at GENERATE: the seed mixes this hash, so without it the pack cannot
            # be derived from anything the validator committed to, and the day is lost after the
            # salt commitment has already been published.
            digest = self._connected().block_info(block).hash
        except Exception as error:  # noqa: BLE001 - the SDK raises a different type per transport
            raise ChainError(f"cannot read the hash of block {block}: {error}") from error
        if not isinstance(digest, str) or not digest.startswith("0x"):
            raise ChainError(
                f"the node returned {digest!r} for the hash of block {block}, which is not a 0x "
                "hex string. A seed derived from a mis-parsed hash is a seed no peer can reproduce."
            )
        try:
            return bytes.fromhex(digest[2:])
        except ValueError as error:
            raise ChainError(
                f"the hash of block {block} is not hex: {digest!r}"
            ) from error

    def _view(self, block: int | None) -> SubnetView:
        client = self._connected()
        source = client.at(block) if block is not None else client
        try:
            metagraph = source.subnets.metagraph(self.netuid, mechid=self.mechid)
        except TypeError as error:
            # A runtime that predates per-mechanism metagraphs does not accept the keyword. Retrying
            # without it is correct *only* when we wanted mechanism 0, because on such a runtime
            # mechanism 0 is the whole subnet. For any other mechid the retry would silently return
            # a different mechanism's neuron set — a wrong cohort, scored and paid as if it were
            # ours. The narrowness matters: `TypeError` can come from anywhere inside the call, so
            # the message is checked too — catching it unconditionally would turn any internal
            # TypeError into a silent read of mechanism 0.
            if self.mechid != 0 or "mechid" not in str(error):
                raise ChainError(
                    f"could not read the metagraph for netuid {self.netuid} mechanism "
                    f"{self.mechid}: {error}. Retrying without the mechanism would read another "
                    "mechanism's neuron set, which would be scored and paid as if it were ours."
                ) from error
            _log.info(
                "this runtime does not accept a mechid; reading netuid %d as a single mechanism",
                self.netuid,
            )
            try:
                metagraph = source.subnets.metagraph(self.netuid)
            except Exception as retry_error:  # noqa: BLE001
                raise ChainError(
                    f"could not read the metagraph for netuid {self.netuid}: {retry_error}"
                ) from retry_error
        except Exception as error:  # noqa: BLE001
            raise ChainError(
                f"could not read the metagraph for netuid {self.netuid}: {error}"
            ) from error

        projected = _project(
            metagraph,
            netuid=self.netuid,
            mechid=self.mechid,
            owner_coldkey=self._owner_coldkey(),
        )
        _assert_mechanism(metagraph, expected=self.mechid, netuid=self.netuid)
        if projected.commitments:
            return projected
        # The metagraph exposes `commitments` and `neuron.commitment`, and bittensor 11.0.1
        # populates neither: both come back empty on a subnet that demonstrably has commitments in
        # storage. `_project` then found none, `parsed_commitments()` returned none, and the
        # validator concluded that no miner had submitted and that no peer had committed a salt. A
        # round like that runs, scores nobody, and looks exactly like a quiet day.
        #
        # Read directly only when the metagraph yields nothing, so an SDK that starts populating
        # the field keeps doing it in one round trip and this stays a fallback.
        return replace(projected, commitments=self._commitments(source, projected))

    def _commitments(self, source: Any, view: SubnetView) -> tuple[RegisteredCommitment, ...]:
        """Every registered hotkey's commitment, read from the Commitments pallet.

        One `query_map` over the netuid rather than a query per hotkey: a 256-neuron subnet would
        otherwise be 256 round trips on every view, and `view()` is called at every round step.

        A commitment from a hotkey that holds no uid is dropped. A commitment is evidence *by a
        uid* — it is how a submission is attributed and paid — so one that cannot be resolved to a
        uid has nobody to attribute it to.
        """
        from bittensor._generated import storage

        try:
            rows = list(source.query_map(storage.Commitments.CommitmentOf, [self.netuid]))
        except Exception as error:  # noqa: BLE001 - the SDK raises a wide family here
            raise ChainError(
                f"could not read commitments for netuid {self.netuid}: {error}. Proceeding as "
                "though there were none would score a round in which every submission was ignored."
            ) from error

        uid_of = {neuron.hotkey: neuron.uid for neuron in view.neurons}
        commitments: list[RegisteredCommitment] = []
        for hotkey, record in rows:
            hotkey = str(hotkey)
            uid = uid_of.get(hotkey)
            if uid is None:
                _log.debug("commitment from %s, which holds no uid on this subnet", hotkey[:12])
                continue
            raw = _commitment_text(record)
            if not raw:
                continue
            commitments.append(
                RegisteredCommitment(
                    uid=uid, hotkey=hotkey, raw=raw, block=int((record or {}).get("block", 0) or 0)
                )
            )
        _log.info("netuid %d: %d commitment(s) read", self.netuid, len(commitments))
        return tuple(sorted(commitments, key=lambda entry: entry.uid))

    def publish_commitment(self, payload: str) -> int:
        """Write a commitment through the Commitments pallet.

        A raw call rather than an intent because bittensor 11 ships no commitment *write* intent —
        `bt.Commitment` is the read type. The payload is the compact text from
        `protocol.commitments`, so nothing about its meaning lives here.

        ## Three things measured against the live pallet, each of which broke this

        **`calls.CommitmentInfo` is `typing.Any`.** Constructing it raises `TypeError: Any cannot be
        instantiated`. The parameter is only *annotated* as that type; the call is built from a
        plain mapping and encoded by the runtime's own metadata.

        **The variant is `Raw{n}`, sized.** There is no plain `Raw`: the pallet's `Data` enum runs
        `Raw0`…`Raw128` and the variant name states the byte count. `Raw129` does not exist.

        **The value is bytes, not hex.** Hex-encoding doubled the length, so a 32-byte field was
        rejected as 64 bytes.

        All three failed together, which is why nothing surfaced until a commitment was published
        against a real chain: every miner submission, every salt commitment and every pack hash goes
        through this method.

        ## Why the payload is split

        A commitment runs to about 240 bytes — a round id, two digests and an artifact URL — and one
        field holds at most 128. So it is chunked, one `Raw{n}` field per chunk, and the pallet
        keeps the fields in order. Split rather than shortened because every part is load
        bearing: the digests bind the bundle and the URL is where it lives.
        """
        client = self._connected()
        raw = payload.encode()
        size = _COMMITMENT_FIELD_BYTES
        chunks = [raw[index : index + size] for index in range(0, len(raw), size)]
        if not chunks:
            raise ChainError("refusing to publish an empty commitment")
        try:
            from bittensor._generated import calls

            call = calls.Commitments.set_commitment(
                netuid=self.netuid,
                info={"fields": [{f"Raw{len(chunk)}": chunk} for chunk in chunks]},
            )
            # Signed by the **hotkey**, not the coldkey. The pallet's origin check is against a
            # registered neuron's hotkey: signing with the wallet (which signs with the coldkey)
            # returns `success=False, "Account is not allowed to make commitments to the chain"`.
            # A commitment *is* the hotkey's statement, so this is the right key on the merits
            # too — but it was found by the chain refusing it, not by reading the pallet.
            result = client.submit_call(call, self._signer().hotkey)
        except Exception as error:  # noqa: BLE001
            raise ChainError(
                f"could not publish a {len(raw)}-byte commitment on netuid {self.netuid}: "
                f"{type(error).__name__}: {error}"
            ) from error
        if not getattr(result, "success", True):
            # `submit_call` reports failure in the result rather than raising, so a commitment the
            # chain refused would otherwise be logged as published — and the round would proceed
            # believing a hash was on chain that is not.
            raise ChainError(
                f"the chain refused the commitment on netuid {self.netuid}: "
                f"{getattr(result, 'message', result)}"
            )
        block = getattr(result, "block", None) or self.current_block()
        _log.info(
            "commitment published at block %d in %d field(s): %s",
            block,
            len(chunks),
            payload[:80],
        )
        return block

    def submit_weights(self, uids: list[int], weights: list[int]) -> None:
        """Publish the weight vector.

        `bt.set_weights` conforms to the subnet's hyperparameters, quantises to u16, chooses
        plaintext or timelocked commit-reveal by reading whether the subnet enables it, and
        raises `ChainError` on failure. Left to the SDK because getting it wrong is invisible: a
        mis-normalised vector is still a valid vector.

        Weights arrive here as ppm integers from `validator.weights`. They are handed over as
        floats because that is the SDK's parameter type and it re-quantises anyway — the integer
        arithmetic upstream is what makes the *ranking* reproducible, and re-quantising a
        normalised vector cannot reorder it.
        """
        if not uids:
            raise ChainError(
                "refusing to submit an empty weight vector: it would zero every miner's "
                "incentive, which is not the same as burning and is never what was meant"
            )
        if len(uids) != len(weights):
            raise ChainError(f"{len(uids)} uids against {len(weights)} weights")

        bittensor = self._bt()
        total = sum(weights)
        if total <= 0:
            raise ChainError("weight vector sums to zero; nothing would be emitted or burned")
        normalised = [weight / total for weight in weights]

        last: Exception | None = None
        for attempt in range(1, self.submit_retries + 2):
            try:
                bittensor.set_weights(
                    self.netuid,
                    normalised,
                    uids=uids,
                    wallet=self._signer(),
                    mechid=self.mechid,
                    version_key=self.version_key,
                    network=self.network,
                )
                _log.info(
                    "weights submitted: %d uids, netuid %d, mechid %d",
                    len(uids),
                    self.netuid,
                    self.mechid,
                )
                return
            except Exception as error:  # noqa: BLE001
                last = error
                _log.warning(
                    "set_weights attempt %d/%d failed: %s",
                    attempt,
                    self.submit_retries + 1,
                    error,
                )
        raise ChainError(
            f"weight submission failed after {self.submit_retries + 1} attempts: {last}. The "
            "cycle must not proceed as if it succeeded — an unsubmitted vector leaves the "
            "previous one in force, which pays yesterday's ranking for today's work."
        ) from last

    def seal(self, plaintext: bytes, *, reveal_at_block: int) -> bytes:
        """Timelock-encrypt for 6.1.

        The reveal *round* is derived from wall-clock time, not from a block height, because
        drand rounds are time-based. So the block is converted through the chain's block time —
        and rounded up, so a payload opens no *earlier* than intended. Opening early is the only
        direction that breaks 6.2, since a validator that could open a rival's bundle before the
        deadline could absorb its design.
        """
        from bittensor import timelock

        client = self._connected()
        try:
            # `block_time` is a *method* on bittensor 11, not a property. `float(client.block_time)`
            # raised `TypeError: float() argument must be ... not 'method'` — so sealing failed on
            # every call and no miner could produce a credential capsule at all. Found by sealing
            # one against the real chain. Called, and the result checked, because a `block_time`
            # returning something unusable would otherwise produce a reveal round of zero: a
            # capsule openable immediately, which is precisely what 6.2 forbids.
            block_seconds = float(client.block_time())
            if not block_seconds > 0:
                raise ChainError(
                    f"the chain reports a block time of {block_seconds}, so a reveal round cannot "
                    "be derived. A zero would seal a payload that opens immediately, and 6.2 "
                    "depends on a bundle not opening before the deadline."
                )
            ahead = max(0, reveal_at_block - self.current_block())
        except ChainError:
            raise
        except Exception as error:  # noqa: BLE001
            raise ChainError(f"could not compute a reveal round: {error}") from error
        seconds = ahead * block_seconds
        sealed = timelock.encrypt(plaintext, seconds)
        return bytes(sealed)

    def unseal(self, ciphertext: bytes) -> bytes:
        """Open a timelocked payload at reveal, or raise if it is not yet openable."""
        from bittensor import timelock
        from bittensor.timelock import TimelockNotReady

        try:
            return timelock.Timelocked.parse(ciphertext).decrypt()
        except TimelockNotReady as error:
            raise ChainError(
                f"the sealed payload does not open until round {error.reveal_round} "
                f"({error.remaining} remaining). Reveal has not arrived; running now would mean "
                "running a bundle whose commitment cannot yet be checked."
            ) from error
        except Exception as error:  # noqa: BLE001
            raise ChainError(f"could not open a sealed payload: {error}") from error


def _assert_mechanism(metagraph: Any, *, expected: int, netuid: int) -> None:
    """Check the metagraph we got is the mechanism we asked for.

    The SDK accepts `mechid` as a keyword and reports it back on the result, so this is a cheap
    confirmation that the two agree. It exists because the alternative failure is invisible: reading
    mechanism 0 when mechanism 1 was wanted returns a perfectly well-formed metagraph with the wrong
    neurons in it, and every score computed from it would be attributed to the wrong hotkeys.

    A metagraph that does not report a mechanism at all is accepted with a log rather than refused —
    an older runtime has no field to report, and refusing would make this client unusable there.
    """
    reported = getattr(metagraph, "mechid", None)
    if reported is None:
        _log.debug("metagraph for netuid %d reports no mechanism; cannot confirm", netuid)
        return
    if int(reported) != expected:
        raise ChainError(
            f"asked netuid {netuid} for mechanism {expected} and received {reported}. Scoring "
            "against another mechanism's neuron set would attribute every result to the wrong "
            "hotkeys."
        )


def _project(
    metagraph: Any, *, netuid: int, mechid: int, owner_coldkey: str = ""
) -> SubnetView:
    """Reduce an SDK metagraph to `SubnetView`.

    Tolerant of missing attributes because this is the one place the SDK's shape reaches us, and
    an attribute that moved should degrade one field rather than fail the whole read.
    """
    neurons: list[Neuron] = []
    commitments: list[RegisteredCommitment] = []
    block = int(getattr(metagraph, "block", 0) or 0)

    for entry in getattr(metagraph, "neurons", ()) or ():
        stake = getattr(entry, "total_stake", None)
        neurons.append(
            Neuron(
                uid=int(entry.uid),
                hotkey=str(entry.hotkey),
                coldkey=str(getattr(entry, "coldkey", "")),
                stake_alpha=_alpha_of(stake),
                validator_permit=bool(getattr(entry, "validator_permit", False)),
                active=bool(getattr(entry, "active", True)),
            )
        )
        commitment = getattr(entry, "commitment", None)
        if commitment is not None:
            commitments.append(_project_commitment(int(entry.uid), commitment))

    # Belt and braces: the metagraph also exposes a uid->commitment map, and a runtime that
    # populates that but not the per-neuron field would otherwise yield no commitments at all.
    for uid, commitment in (getattr(metagraph, "commitments", {}) or {}).items():
        if any(existing.uid == int(uid) for existing in commitments):
            continue
        commitments.append(_project_commitment(int(uid), commitment))

    return SubnetView(
        netuid=netuid,
        mechid=mechid,
        block=block,
        neurons=tuple(sorted(neurons, key=lambda neuron: neuron.uid)),
        commitments=tuple(sorted(commitments, key=lambda entry: entry.uid)),
        owner_coldkey=owner_coldkey,
    )


def _alpha_of(stake: Any) -> float:
    """A stake balance as a float, in whatever unit the runtime denominates it.

    `Balance.alpha` on a dTAO subnet, `Balance.tao` on a root or pre-dTAO one. Tried in that order
    and neither is required, because this figure is a diagnostic here — it is not what decides
    anything — and a metagraph read must not fail over it. What *did* fail over it was
    `float(stake.tao)` on subnet alpha, which raises rather than converting.
    """
    if stake is None:
        return 0.0
    for unit in ("alpha", "tao"):
        try:
            return float(getattr(stake, unit))
        except Exception:  # noqa: BLE001 - the SDK raises UnitMismatchError, and may change which
            continue
    try:
        return float(stake)
    except (TypeError, ValueError):
        return 0.0


def _commitment_text(record: Any) -> str:
    """Rejoin a stored commitment's `Raw` fields into the text that was published.

    The inverse of the chunking in `publish_commitment`. A payload longer than one field is stored
    as several, in order, so they are concatenated before decoding — reading only the first would
    silently truncate every commitment this subnet writes to its first 128 bytes, which for a
    submission is the round id and half a digest.

    A field that is not valid UTF-8, or a record shaped differently, yields `""` rather than
    raising: other subnets and other protocol versions share this channel, and
    `parsed_commitments` already drops what it cannot decode.
    """
    stored = ((record or {}).get("info") or {}).get("fields") or []
    payload = bytearray()
    for entry in stored:
        if not isinstance(entry, dict):
            continue
        for name, value in entry.items():
            if not str(name).startswith("Raw"):
                continue
            if isinstance(value, str):
                try:
                    payload.extend(bytes.fromhex(value.removeprefix("0x")))
                except ValueError:
                    return ""
            elif isinstance(value, bytes | bytearray):
                payload.extend(value)
    try:
        return payload.decode()
    except UnicodeDecodeError:
        return ""


def _project_commitment(uid: int, commitment: Any) -> RegisteredCommitment:
    return RegisteredCommitment(
        uid=uid,
        hotkey=str(getattr(commitment, "hotkey", "")),
        raw=str(getattr(commitment, "data", "") or ""),
        block=int(getattr(commitment, "block", 0) or 0),
        encrypted=bool(getattr(commitment, "encrypted", False)),
        reveal_round=getattr(commitment, "reveal_round", None),
    )


# --------------------------------------------------------------------------
# The test double
# --------------------------------------------------------------------------


@dataclass
class FakeChain:
    """An in-memory chain. Not a mock — a small honest implementation.

    Deliberately *not* a `Mock`: a mock asserts on calls, and what the cycle tests need to assert
    on is state. This advances blocks, stores commitments per hotkey with the block they landed
    in, overwrites on a second write (which is the pallet's real behaviour, and the reason
    `PackCommitment` carries the salt forward), and keeps history so `view_at` works.

    `seal`/`unseal` are XOR with a block gate rather than real timelock encryption. That is
    honest about being a stub — it makes the *timing* rule testable, which is the part the cycle
    depends on, and no test here should be able to convince anyone the cryptography works.
    """

    netuid: int = 1
    mechid: int = 0
    own_hotkey: str = "5Fvalidator"
    block: int = 1_000
    neurons: list[Neuron] = field(default_factory=list)
    #: hotkey -> (payload, block)
    live_commitments: dict[str, tuple[str, int]] = field(default_factory=dict)
    #: block -> snapshot of live_commitments as it was after that block
    history: dict[int, dict[str, tuple[str, int]]] = field(default_factory=dict)
    submitted: list[tuple[list[int], list[int], int]] = field(default_factory=list)
    fail_submit: bool = False
    fail_reads: bool = False
    #: 23's owner-eligibility rule. Settable so a test can register an owner-linked miner and watch
    #: it be excluded — the rule is unobservable against a fake that has no owner.
    owner_coldkey: str = ""

    def advance(self, blocks: int = 1) -> int:
        self.history[self.block] = dict(self.live_commitments)
        self.block += blocks
        return self.block

    def register(self, hotkey: str, *, validator: bool = False, stake: float = 1_000.0) -> Neuron:
        neuron = Neuron(
            uid=len(self.neurons),
            hotkey=hotkey,
            coldkey=f"cold-{hotkey}",
            stake_alpha=stake,
            validator_permit=validator,
            active=True,
        )
        self.neurons.append(neuron)
        return neuron

    def current_block(self) -> int:
        if self.fail_reads:
            raise ChainError("fake chain is unreachable")
        return self.block

    def view(self) -> SubnetView:
        return self._build(self.live_commitments, self.block)

    def block_hash(self, block: int) -> bytes:
        """Deterministic from the height, and only for the past.

        Deterministic so a test can assert that two derivations of a seed agree; past-only so the
        rule the real client enforces is the rule a test exercises. A fake that answered for the
        head would let a test pass against a sequence the chain cannot produce.
        """
        if self.fail_reads:
            raise ChainError("fake chain is unreachable")
        if block >= self.block:
            raise ChainError(
                f"block {block} is not in the past (head is {self.block}); its hash is not final"
            )
        return hashlib.blake2b(
            f"fake-block-{block}".encode(), digest_size=32
        ).digest()

    def view_at(self, block: int) -> SubnetView:
        if self.fail_reads:
            raise ChainError("fake chain is unreachable")
        # Nearest recorded height at or before the request, which is what an archive node does.
        candidates = [height for height in self.history if height <= block]
        if not candidates:
            return self._build({}, block)
        return self._build(self.history[max(candidates)], block)

    def _build(self, commitments: dict[str, tuple[str, int]], block: int) -> SubnetView:
        if self.fail_reads:
            raise ChainError("fake chain is unreachable")
        by_hotkey = {neuron.hotkey: neuron for neuron in self.neurons}
        registered = [
            RegisteredCommitment(
                uid=by_hotkey[hotkey].uid, hotkey=hotkey, raw=payload, block=landed
            )
            for hotkey, (payload, landed) in sorted(commitments.items())
            if hotkey in by_hotkey
        ]
        return SubnetView(
            netuid=self.netuid,
            mechid=self.mechid,
            block=block,
            neurons=tuple(self.neurons),
            commitments=tuple(sorted(registered, key=lambda entry: entry.uid)),
            owner_coldkey=self.owner_coldkey,
        )

    def publish_commitment(self, payload: str) -> int:
        if self.fail_reads:
            raise ChainError("fake chain is unreachable")
        # Overwrites, exactly as the pallet does. A test that assumed both a salt and a pack
        # commitment could coexist would pass against a store that appended, and fail on chain.
        self.live_commitments[self.own_hotkey] = (payload, self.block)
        self.history[self.block] = dict(self.live_commitments)
        return self.block

    def submit_weights(self, uids: list[int], weights: list[int]) -> None:
        if self.fail_submit:
            raise ChainError("fake chain refuses weight submission")
        if not uids:
            raise ChainError("refusing an empty weight vector")
        if len(uids) != len(weights):
            raise ChainError(f"{len(uids)} uids against {len(weights)} weights")
        self.submitted.append((list(uids), list(weights), self.block))

    def seal(self, plaintext: bytes, *, reveal_at_block: int) -> bytes:
        header = reveal_at_block.to_bytes(8, "big")
        return header + bytes(byte ^ 0x5A for byte in plaintext)

    def unseal(self, ciphertext: bytes) -> bytes:
        reveal_at = int.from_bytes(ciphertext[:8], "big")
        if self.block < reveal_at:
            raise ChainError(
                f"sealed until block {reveal_at}, now {self.block}: opening early would let a "
                "validator absorb a rival's current-round design"
            )
        return bytes(byte ^ 0x5A for byte in ciphertext[8:])

    def hotkey(self) -> str:
        return self.own_hotkey
