"""The protocol layer: everything both sides must agree on, byte for byte.

Deliberately free of network, clock and randomness — `tools/check_purity.py` enforces that on the
modules whose non-determinism would change a score.
"""

from protocol.canonical import canonical_bytes, digest_bytes, digest_object
from protocol.fixedpoint import PPM, apply_weights, assert_sums_to_one, clamp_ppm
from protocol.receipts import Call, CredentialOwner, Purpose, Receipt, Tool, verify_chain
from protocol.seeds import challenge_id, daily_seed, salt_commitment, verify_salt

__all__ = [
    "PPM",
    "Call",
    "CredentialOwner",
    "Purpose",
    "Receipt",
    "Tool",
    "apply_weights",
    "assert_sums_to_one",
    "canonical_bytes",
    "challenge_id",
    "clamp_ppm",
    "daily_seed",
    "digest_bytes",
    "digest_object",
    "salt_commitment",
    "verify_chain",
    "verify_salt",
]
