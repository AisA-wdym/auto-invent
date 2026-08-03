"""Chain access. One narrow interface (`ChainClient`), one SDK-touching implementation.

A deviation from architecture.md 24, which has no `chain/` package: the validator and the miner
CLI both write and read the same commitments, and the same weight-vector conformance rules apply
to both. Putting that in each of them would define the wire format twice, and two definitions
drift. `protocol/commitments.py` holds the format (pure); this holds the transport.
"""

from chain.client import (
    BittensorChain,
    ChainClient,
    ChainError,
    FakeChain,
    Neuron,
    RegisteredCommitment,
    SubnetView,
)

__all__ = [
    "BittensorChain",
    "ChainClient",
    "ChainError",
    "FakeChain",
    "Neuron",
    "RegisteredCommitment",
    "SubnetView",
]
