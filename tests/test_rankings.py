from pathlib import Path

import pandas as pd
import pytest

from conftest import FakeLeague, FakeTeam
from rankings import (
    add_weekly_change,
    fuzzy_merge,
    generate_expected_standings,
    generate_playoff_probabilities,
    generate_power_rankings,
)


def test_fuzzy_merge_does_not_mutate_inputs_and_handles_nulls():
    left = pd.DataFrame([{"name": "Josh Allen", "team": "BUF"}, {"name": None, "team": None}])
    right = pd.DataFrame([{"player": "Josh Allen", "club": "BUF", "owner": "A"}])
    left_columns = list(left.columns)
    right_columns = list(right.columns)

    result = fuzzy_merge(left, right, left_name="name", right_name="player", left_team="team", right_team="club")

    assert result.loc[0, "owner"] == "A"
    assert list(left.columns) == left_columns
    assert list(right.columns) == right_columns


def test_fuzzy_merge_validates_columns():
    with pytest.raises(ValueError, match="missing required columns"):
        fuzzy_merge(pd.DataFrame({"name": ["A"]}), pd.DataFrame({"player": ["A"], "club": ["X"]}), left_name="name", right_name="player", left_team="team", right_team="club")


def test_generate_power_rankings_reports_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Player values not found"):
        generate_power_rankings(FakeLeague([]), 3, tmp_path / "missing.csv")


def test_generate_power_rankings_matches_alias_and_handles_equal_scores(tmp_path, fake_player):
    alpha = FakeTeam("Alpha", roster=[fake_player("Hollywood Brown", nfl_team="KC")])
    beta = FakeTeam("Beta", roster=[fake_player("Other Player", nfl_team="BUF")])
    league = FakeLeague([alpha, beta], power_scores={"Alpha": 10, "Beta": 10})
    values_path = tmp_path / "values.csv"
    pd.DataFrame([
        {"Player Name": "Marquise Brown", "Pos": "WR1", "Value": "100", "NFL_Team": "KC"},
        {"Player Name": "Other Player", "Pos": "RB1", "Value": "100", "NFL_Team": "BUF"},
    ]).to_csv(values_path, index=False)

    rankings, players = generate_power_rankings(league, 1, values_path)

    assert set(rankings["Team"]) == {"Alpha", "Beta"}
    assert rankings["Power Score"].notna().all()
    assert "Hollywood Brown" in set(players["Player Name"])


def test_weekly_change_supports_new_and_renamed_teams():
    current = pd.DataFrame({"Team": ["Renamed Team", "Alpha"], "Power Score": [90, 80], "Performance Rank": [1, 2], "KTC Value Rank": [1, 2]}, index=[1, 2])
    previous = pd.DataFrame({"Team": ["Old Team", "Alpha"], "Power Score": [90, 80]}, index=[1, 2])

    result = add_weekly_change(current, previous)

    assert result.loc[1, "Weekly Change"] == "NEW"
    assert "#111827" in result.loc[2, "Weekly Change"]
    assert "—" in result.loc[2, "Weekly Change"]


def test_expected_standings_use_dynamic_regular_season_length():
    alpha = FakeTeam("Alpha", wins=2, losses=1)
    beta = FakeTeam("Beta", wins=1, losses=2)
    alpha.schedule = [beta, beta, beta, beta]
    beta.schedule = [alpha, alpha, alpha, alpha]
    rankings = pd.DataFrame({"Team": ["Alpha", "Beta"], "Power Score": [75, 25]})

    result = generate_expected_standings(FakeLeague([alpha, beta]), rankings, week=2, regular_season_weeks=4)

    alpha_result = result[result["Team"] == "Alpha"].iloc[0]
    assert alpha_result["Projected Wins"] == 3.5
    assert alpha_result["Projected Losses"] == 1.5


def test_playoff_simulation_is_seeded_dynamic_and_preserves_total_probability(matchup):
    teams = [FakeTeam(name, wins=0, scores=[100 + index]) for index, name in enumerate(["Renamed A", "B", "C", "D"])]
    scoreboards = {
        3: [matchup(teams[0], teams[1]), matchup(teams[2], teams[3])],
        4: [matchup(teams[0], teams[2]), matchup(teams[1], teams[3])],
    }
    league = FakeLeague(teams, scoreboards=scoreboards)
    rankings = pd.DataFrame(
        {"Team": ["Renamed A", "B", "C", "D"], "Power Score": [80, 60, 40, 20]}
    )

    first = generate_playoff_probabilities(league, rankings, week=2, regular_season_weeks=4, playoff_teams=2, simulations=500, seed=123)
    second = generate_playoff_probabilities(league, rankings, week=2, regular_season_weeks=4, playoff_teams=2, simulations=500, seed=123)

    pd.testing.assert_frame_equal(first, second)
    total_probability = first["Playoffs"].str.rstrip("%").astype(float).sum()
    assert total_probability == pytest.approx(200, abs=0.05)
    assert set(first["Team"]) == {"Renamed A", "B", "C", "D"}


def test_playoff_simulation_weights_matchups_by_power_score(matchup):
    strong = FakeTeam("Strong", scores=[100])
    weak = FakeTeam("Weak", scores=[100])
    league = FakeLeague([strong, weak], scoreboards={2: [matchup(strong, weak)]})
    rankings = pd.DataFrame({"Team": ["Strong", "Weak"], "Power Score": [90, 10]})

    result = generate_playoff_probabilities(
        league, rankings, week=1, regular_season_weeks=2, playoff_teams=1, simulations=5000, seed=123
    )

    probabilities = result.set_index("Team")["Playoffs"].str.rstrip("%").astype(float)
    assert probabilities["Strong"] > 85
    assert probabilities["Weak"] < 15
