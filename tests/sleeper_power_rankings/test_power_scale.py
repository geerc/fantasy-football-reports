import pandas as pd
import pytest
from sleeper_rankings.power import power_scores, win_probability


def test_neutral_benchmark_and_fixed_units():
    scores = power_scores(pd.Series([.5, .75, .5]), pd.Series([100, 100, 125]), 1)
    assert list(scores) == [100, 116.5, 113.5]


def test_no_clipping_and_no_min_max_stretch():
    scores = power_scores(pd.Series([.49, .5, .51]), pd.Series([100, 100, 100]), 5)
    assert scores.max() - scores.min() < 2
    extreme = power_scores(pd.Series([0, .5, 1]), pd.Series([-1000, 100, 1000]), 1)
    assert extreme.iloc[0] < 0
    assert extreme.iloc[2] > 200


def test_growing_performance_weight_and_roster_unit_invariance():
    perf, roster = pd.Series([.75, .5, .5]), pd.Series([100, 125, 100])
    early, late = power_scores(perf, roster, 1), power_scores(perf, roster, 15)
    assert late.iloc[0] > early.iloc[0]
    assert late.iloc[1] < early.iloc[1]
    pd.testing.assert_series_equal(early, power_scores(perf, roster * 1000, 1))


def test_probability_uses_gaps_and_supports_unbounded_scores():
    assert win_probability(125, 75) == pytest.approx(.75)
    assert win_probability(0, 0) == .5
    assert win_probability(-75, -125) == pytest.approx(.75)
    assert win_probability(10000, -10000) == pytest.approx(1)
    assert win_probability(-10000, 10000) == pytest.approx(0)


def test_invalid_inputs_fail_clearly():
    with pytest.raises(ValueError, match="positive"):
        power_scores(pd.Series([.5]), pd.Series([0]), 1)
    with pytest.raises(ValueError, match="finite"):
        power_scores(pd.Series([float('nan')]), pd.Series([100]), 1)
