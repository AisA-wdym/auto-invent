"""LLM judge panels and the evaluation funnel. architecture.md 16-19."""

from validator.judge.bradley_terry import Outcome, Pairing, fit, strengths_to_ppm
from validator.judge.pairwise import PairVerdict, combine_orders, compare_pair, swiss_pairings
from validator.judge.panels import JUDGE_ROLES, Panel, PanelError, panels_from_season, pins_for
from validator.judge.pointwise import PointwiseScore, aggregate, screen_portfolio

__all__ = [
    "JUDGE_ROLES",
    "Outcome",
    "PairVerdict",
    "Pairing",
    "Panel",
    "PanelError",
    "PointwiseScore",
    "aggregate",
    "combine_orders",
    "compare_pair",
    "fit",
    "panels_from_season",
    "pins_for",
    "screen_portfolio",
    "strengths_to_ppm",
    "swiss_pairings",
]
