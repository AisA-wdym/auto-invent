"""Stage 2: answer canonicalization. architecture.md 14."""

from validator.canonicalizer.neutral import (
    CanonicalPortfolio,
    Removal,
    canonicalize,
    strip_text,
)

__all__ = ["CanonicalPortfolio", "Removal", "canonicalize", "strip_text"]
