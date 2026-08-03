"""Stage 3: prior-art and originality analysis. architecture.md 15.

Never asserts that an idea is absolutely unprecedented — only what a search found.
"""

from validator.prior_art.report import (
    Match,
    PriorArtReport,
    assess_renaming,
    build_report,
    novelty_confidence_ppm,
)

__all__ = [
    "Match",
    "PriorArtReport",
    "assess_renaming",
    "build_report",
    "novelty_confidence_ppm",
]
