"""Shared, unclipped Power Score scale for ESPN and Sleeper."""
import math
import numpy as np
import pandas as pd

FORMULA_VERSION = "power-v2"
METHODOLOGY = (
    "Power Score: 100 is the neutral benchmark; scores have no imposed floor or ceiling. "
    "All-play performance gains weight as the season progresses. "
    "Historical comparisons are approximate: league competition, roster values, "
    "season timing, and sample size affect scores. Different formula versions "
    "are not directly comparable. Formula: " + FORMULA_VERSION + "."
)


def power_scores(performance: pd.Series, roster: pd.Series, week: int) -> pd.Series:
    """Fixed units: 0.25 performance or 25% of median roster value = one strength unit."""
    if week < 1:
        raise ValueError("Power Score requires a completed week")
    if not performance.index.equals(roster.index):
        raise ValueError("Performance and roster indexes must match")
    if not np.isfinite(performance).all() or not np.isfinite(roster).all():
        raise ValueError("Power Score inputs must be finite")
    median = float(roster.median())
    if median <= 0:
        raise ValueError("Median roster value must be positive")
    weight = 0.20 + 0.25 * 2 ** (-(week - 1) / 5)
    performance_strength = (performance - 0.50) / 0.25
    roster_strength = (roster / median - 1) / 0.25
    return (100 + 30 * ((1 - weight) * performance_strength + weight * roster_strength)).round(1)


def win_probability(left: float, right: float) -> float:
    """Difference-based odds remain valid for zero and negative Power Scores.

    A 50-point advantage gives 75% odds; this is a heuristic, not a fitted model.
    """
    z = (float(left) - float(right)) * math.log(3) / 50
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    exp_z = math.exp(z)
    return exp_z / (1 + exp_z)
