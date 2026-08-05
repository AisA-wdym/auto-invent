"""Two credentials, two types, and no way to confuse them.

architecture.md 3.4.4. Every model call in the subnet goes through OpenRouter: miner research
on the miner's key, challenge generation and judging on the validator's. Same provider, two
accounts, and the separation is what keeps the equal-budget guarantee real — a validator that
could bill its judging to a miner's key could exhaust a rival-sponsored laboratory at will.

## Why this module exists at all

When the two sides used different providers, a swapped credential failed on the first request:
wrong endpoint, wrong request shape, wrong error. With one provider a swapped key **succeeds**.
It authenticates, returns a completion, and silently bills the wrong party. Nothing surfaces
until someone reconciles an invoice.

So the invariant is structural rather than remembered:

1. **Two types, not one store keyed by owner.** `MinerCredential` and `ValidatorCredential` are
   distinct types. There is no `resolve(owner)` — because a parameter can be passed the wrong
   value, and a type cannot.
2. **Purpose selects the credential.** `for_purpose` maps each purpose to the one account that
   may pay for it, and a mismatch raises before the request is built rather than after it
   succeeds.
3. **The laboratory never receives either.** It gets a session token bound to one run and one
   challenge, carrying no credential at all.

Point 3 is why this module holds the key and the sandbox does not. A laboratory with its own
key could call outside the meter, spend past the ceiling, or return the key in its own output —
and disclosure publishes that output.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from protocol.receipts import CredentialOwner, Purpose

__all__ = [
    "CredentialError",
    "CredentialSet",
    "MinerCredential",
    "ValidatorCredential",
    "load_validator_credential",
]

_log = logging.getLogger(__name__)

#: Purposes each account may fund. The same table `protocol.receipts` enforces on the receipt,
#: expressed here at the point of *selection* rather than of recording — so a mismatch is
#: refused before a request is built, and refused again when the receipt is written. Two
#: independent checks on the one property a single provider surface cannot check for us.
_MINER_PURPOSES = frozenset({Purpose.RESEARCH, Purpose.SEARCH, Purpose.SIMULATION})
_VALIDATOR_PURPOSES = frozenset(
    {Purpose.CHALLENGE_GENERATION, Purpose.CRITIQUE, Purpose.JUDGING, Purpose.PRIOR_ART}
)


class CredentialError(RuntimeError):
    """A credential that cannot be used for the purpose requested."""


@dataclass(frozen=True, slots=True)
class MinerCredential:
    """A miner's OpenRouter key, decrypted from its sealed envelope at reveal.

    `field(repr=False)` on the secret because a credential reaches logs and exception messages,
    and a `repr` that printed it would put it in every traceback and CI transcript.
    """

    miner_hotkey: str
    api_key: str = field(repr=False)
    declared_spend_cap_usd: int = 0

    owner = CredentialOwner.MINER

    def __post_init__(self) -> None:
        if not self.api_key:
            raise CredentialError(
                f"no API key for miner {self.miner_hotkey}: its laboratory cannot be run, and "
                "running it on the validator's key would bill the wrong account"
            )

    def assert_may_fund(self, purpose: Purpose) -> None:
        if purpose not in _MINER_PURPOSES:
            raise CredentialError(
                f"purpose {purpose.value!r} may not be funded by a miner's credential. With one "
                "provider surface this call would have succeeded and silently billed "
                f"{self.miner_hotkey}, so it is refused before the request is built."
            )


@dataclass(frozen=True, slots=True)
class ValidatorCredential:
    """The validator's own OpenRouter key, for generation, critique, judging and prior art."""

    validator_hotkey: str
    api_key: str = field(repr=False)

    owner = CredentialOwner.VALIDATOR

    def __post_init__(self) -> None:
        if not self.api_key:
            raise CredentialError(
                "no validator API key: challenge generation and judging cannot run, and "
                "falling back to a miner's key would make the equal-budget guarantee a fiction"
            )

    def assert_may_fund(self, purpose: Purpose) -> None:
        if purpose not in _VALIDATOR_PURPOSES:
            raise CredentialError(
                f"purpose {purpose.value!r} may not be funded by the validator's credential. "
                "Research is the miner's cost; billing it here would let a validator subsidise "
                "or starve a laboratory at will."
            )


@dataclass(slots=True)
class CredentialSet:
    """Holds both, and hands out exactly one per purpose.

    The only object with access to both keys, and it exposes no way to ask for one by name. A
    caller states *what it is doing*; the set decides which account pays. That inverts the usual
    shape — `get_key("miner")` — and the inversion is the point: a purpose is a fact about the
    call, while an owner name is a claim the caller could get wrong.
    """

    validator: ValidatorCredential
    #: Per-miner keys, populated at reveal as each envelope is decrypted.
    miners: dict[str, MinerCredential] = field(default_factory=dict)

    def admit(self, credential: MinerCredential) -> None:
        """Register a miner's credential for this round."""
        self.miners[credential.miner_hotkey] = credential
        _log.info("credential admitted for miner %s", credential.miner_hotkey)

    def for_purpose(
        self, purpose: Purpose, *, miner_hotkey: str | None = None
    ) -> MinerCredential | ValidatorCredential:
        """The one credential that may fund `purpose`.

        A miner purpose requires `miner_hotkey`, and requires it to be a miner whose envelope
        actually decrypted. Falling back to the validator's key for an unknown miner is the
        exact failure this module exists to prevent, so an unknown miner raises.
        """
        if purpose in _VALIDATOR_PURPOSES:
            if miner_hotkey is not None:
                # A validator-funded call that names a miner is a caller confusion, and under
                # one provider surface it would have worked. Refused rather than ignored.
                raise CredentialError(
                    f"purpose {purpose.value!r} is validator-funded but names miner "
                    f"{miner_hotkey!r}. Ignoring the name would hide which account was meant."
                )
            self.validator.assert_may_fund(purpose)
            return self.validator

        if purpose not in _MINER_PURPOSES:
            raise CredentialError(f"purpose {purpose.value!r} has no declared payer")
        if miner_hotkey is None:
            raise CredentialError(
                f"purpose {purpose.value!r} is miner-funded and no miner was named. There is no "
                "default: charging an unnamed research call to the validator would make every "
                "budget comparison meaningless."
            )
        credential = self.miners.get(miner_hotkey)
        if credential is None:
            raise CredentialError(
                f"no credential admitted for miner {miner_hotkey!r}. Its envelope did not "
                "decrypt, or it was never submitted. Running its laboratory on the validator's "
                "key would bill the wrong account."
            )
        credential.assert_may_fund(purpose)
        return credential

    def owner_of(self, purpose: Purpose) -> CredentialOwner:
        """Which account a purpose bills, for the receipt."""
        if purpose in _VALIDATOR_PURPOSES:
            return CredentialOwner.VALIDATOR
        if purpose in _MINER_PURPOSES:
            return CredentialOwner.MINER
        raise CredentialError(f"purpose {purpose.value!r} has no declared payer")


def load_validator_credential(
    hotkey: str, *, environ: dict[str, str] | None = None
) -> ValidatorCredential:
    """Read the validator's key from a mounted secret or the environment.

    A file path is preferred and checked first, because a projected secret file can be
    permission-restricted where an environment variable is readable by anything that can list
    the process. The environment form exists because it is what most deployments actually do,
    and refusing it would push operators toward worse workarounds.
    """
    env = dict(os.environ if environ is None else environ)

    path = env.get("AI_VALIDATOR_OPENROUTER_KEY_FILE")
    if path:
        secret = Path(path)
        if not secret.is_file():
            raise CredentialError(f"AI_VALIDATOR_OPENROUTER_KEY_FILE names no file at {path}")
        # Trailing newline stripped: a file written by `echo` ends in one, and a newline in an
        # Authorization header is both an authentication failure and a request-splitting
        # primitive.
        return ValidatorCredential(validator_hotkey=hotkey, api_key=secret.read_text().strip())

    key = env.get("AI_VALIDATOR_OPENROUTER_KEY", "").strip()
    if not key:
        raise CredentialError(
            "set AI_VALIDATOR_OPENROUTER_KEY_FILE or AI_VALIDATOR_OPENROUTER_KEY. Challenge "
            "generation and judging are validator costs and have no fallback: using a miner's "
            "key for them would make the equal-budget guarantee a fiction."
        )
    return ValidatorCredential(validator_hotkey=hotkey, api_key=key)
