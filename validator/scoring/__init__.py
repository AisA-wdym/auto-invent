"""Hard gates and the scoring model. architecture.md 13, 18."""

from validator.scoring.criteria import ScoringConfig, challenge_score, collapse_duplicates
from validator.scoring.daily import DailyConfig, daily_score, rolling_score
from validator.scoring.gates import Gate, GateReport, check_all

__all__ = [
    "DailyConfig",
    "Gate",
    "GateReport",
    "ScoringConfig",
    "challenge_score",
    "check_all",
    "collapse_duplicates",
    "daily_score",
    "rolling_score",
]
