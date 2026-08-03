"""Validator-generated daily challenge packs: architecture.md 7.

Six steps, cheapest-first, with the chain commitment before the store:

    plan (seeded) → generate → lint → safety → dedup → critic → discriminate → commit → store
"""

from validator.challenge_factory.pipeline import (
    PackResult,
    PipelineError,
    Rejection,
    build_pack,
    commit_and_store,
    pack_hash,
)
from validator.challenge_factory.taxonomy import Slot, Taxonomy, TaxonomyError, plan

__all__ = [
    "PackResult",
    "PipelineError",
    "Rejection",
    "Slot",
    "Taxonomy",
    "TaxonomyError",
    "build_pack",
    "commit_and_store",
    "pack_hash",
    "plan",
]
