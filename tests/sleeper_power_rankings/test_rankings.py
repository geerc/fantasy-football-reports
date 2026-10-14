import pandas as pd

from sleeper_rankings.models import LeagueData, Team
import pytest

from sleeper_rankings.rankings import add_weekly_change, performance_score, playoff_probabilities, power_rankings, roster_values


def fixture_data():
    first = Team(1, "a", "Alpha", "A", ["1"], wins=1, scores=[120], opponents=[2])
    second = Team(2, "b", "Beta", "B", ["2"], losses=1, scores=[100], opponents=[1])
    players = {
        "1": {"full_name": "Josh Allen", "position": "QB"},
        "2": {"full_name": "Lamar Jackson", "position": "QB"},
    }
    return LeagueData({"settings": {"playoff_week_start": 15, "playoff_teams": 1}}, [first, second], {}, players)


def test_winner_has_higher_performance_score():
    data = fixture_data()
    assert performance_score(data, data.teams[0], 1) > performance_score(data, data.teams[1], 1)


def test_power_rankings_match_sleeper_ids_to_values():
    data = fixture_data()
    values = pd.DataFrame([
        {"name": "Josh Allen", "position": "QB", "nfl_team": "BUF", "value": 9000},
        {"name": "Lamar Jackson", "position": "QB", "nfl_team": "BAL", "value": 8000},
    ])
    result = power_rankings(data, 1, values)
    assert result.iloc[0]["Team"] == "Alpha"
    assert list(result["roster_id"]) == [1, 2]


def test_performance_ignores_record_and_head_to_head_schedule():
    teams = [
        Team(1, "a", "Alpha", "A", [], wins=0, losses=2, scores=[120, 90], opponents=[2, 3]),
        Team(2, "b", "Beta", "B", [], wins=2, scores=[100, 110], opponents=[1, 3]),
        Team(3, "c", "Gamma", "C", [], wins=1, losses=1, scores=[80, 95], opponents=[2, 1]),
    ]
    data = LeagueData({"settings": {}}, teams, {}, {})
    before = performance_score(data, teams[0], 2)
    teams[0].wins = 99
    teams[0].opponents = [3, 2]
    assert performance_score(data, teams[0], 2) == before


def test_all_play_ties_split_credit_evenly():
    teams = [
        Team(1, "a", "Alpha", "A", [], scores=[100], opponents=[2]),
        Team(2, "b", "Beta", "B", [], scores=[100], opponents=[1]),
        Team(3, "c", "Gamma", "C", [], scores=[80], opponents=[None]),
    ]
    data = LeagueData({"settings": {}}, teams, {}, {})
    assert performance_score(data, teams[0], 1) == performance_score(data, teams[1], 1)
    assert performance_score(data, teams[0], 1) > performance_score(data, teams[2], 1)


def test_recent_scores_receive_more_weight():
    teams = [
        Team(1, "a", "Improving", "A", [], scores=[80, 120], opponents=[2, 2]),
        Team(2, "b", "Fading", "B", [], scores=[120, 80], opponents=[1, 1]),
    ]
    data = LeagueData({"settings": {}}, teams, {}, {})
    assert performance_score(data, teams[0], 2) > performance_score(data, teams[1], 2)


def test_unmatched_player_does_not_increase_team_value():
    team = Team(1, "a", "Alpha", "A", ["1", "2"])
    players = {
        "1": {"full_name": "Josh Allen", "position": "QB", "team": "BUF"},
        "2": {"full_name": "Imaginary Star", "position": "QB", "team": "BUF"},
    }
    data = LeagueData({"roster_positions": ["QB", "BN"]}, [team], {}, players)
    values = pd.DataFrame([{"name": "Josh Allen", "position": "QB", "nfl_team": "BUF", "value": 9000}])
    assert roster_values(data, values)[1] == pytest.approx(6750)


def test_roster_value_prioritizes_starters_and_discounts_depth():
    players = {
        "1": {"full_name": "Starter One", "position": "RB", "team": "AAA"},
        "2": {"full_name": "Starter Two", "position": "RB", "team": "BBB"},
        "3": {"full_name": "Bench Player", "position": "RB", "team": "CCC"},
    }
    team = Team(1, "a", "Alpha", "A", ["1", "2", "3"])
    data = LeagueData({"roster_positions": ["RB", "RB", "BN"]}, [team], {}, players)
    values = pd.DataFrame([
        {"name": "Starter One", "position": "RB", "nfl_team": "AAA", "value": 9000},
        {"name": "Starter Two", "position": "RB", "nfl_team": "BBB", "value": 7000},
        {"name": "Bench Player", "position": "RB", "nfl_team": "CCC", "value": 1000},
    ])
    assert roster_values(data, values)[1] == pytest.approx(6250)


def test_player_matching_requires_compatible_position_and_team():
    team = Team(1, "a", "Alpha", "A", ["1"])
    players = {"1": {"full_name": "John Smith", "position": "RB", "team": "BUF"}}
    data = LeagueData({"roster_positions": ["RB"]}, [team], {}, players)
    values = pd.DataFrame([
        {"name": "John Smith", "position": "WR", "nfl_team": "BUF", "value": 9000},
        {"name": "John Smith", "position": "RB", "nfl_team": "MIA", "value": 8000},
    ])
    assert roster_values(data, values)[1] == 0


def test_identical_inputs_receive_identical_non_extreme_power_scores():
    teams = [
        Team(1, "a", "Alpha", "A", ["1"], scores=[100], opponents=[2]),
        Team(2, "b", "Beta", "B", ["2"], scores=[100], opponents=[1]),
    ]
    players = {
        "1": {"full_name": "Player One", "position": "QB", "team": "AAA"},
        "2": {"full_name": "Player Two", "position": "QB", "team": "BBB"},
    }
    data = LeagueData({"settings": {}, "roster_positions": ["QB"]}, teams, {}, players)
    values = pd.DataFrame([
        {"name": "Player One", "position": "QB", "nfl_team": "AAA", "value": 5000},
        {"name": "Player Two", "position": "QB", "nfl_team": "BBB", "value": 5000},
    ])
    result = power_rankings(data, 1, values)
    assert list(result["Power Score"]) == [50, 50]


def test_incomplete_week_is_rejected():
    data = fixture_data()
    values = pd.DataFrame([
        {"name": "Josh Allen", "position": "QB", "nfl_team": "BUF", "value": 9000},
        {"name": "Lamar Jackson", "position": "QB", "nfl_team": "BAL", "value": 8000},
    ])
    with pytest.raises(ValueError, match="Week 2 is incomplete"):
        power_rankings(data, 2, values)


def test_weekly_change_uses_roster_id_not_mutable_team_name():
    current = pd.DataFrame([{"Team": "New name", "roster_id": 2, "Power Score": 90}, {"Team": "Alpha", "roster_id": 1, "Power Score": 80}], index=[1, 2])
    previous = pd.DataFrame([{"Team": "Alpha", "roster_id": 1}, {"Team": "Old name", "roster_id": 2}], index=[1, 2])
    result = add_weekly_change(current, previous)
    assert list(result["Weekly Change"]) == ["↑ 1", "↓ 1"]


def test_playoff_probabilities_are_seeded_and_sum_to_available_spots():
    teams = [
        Team(1, "a", "Alpha", "A", [], wins=2, scores=[120, 110], opponents=[2, 3, 2, 4]),
        Team(2, "b", "Beta", "B", [], wins=1, scores=[100, 105], opponents=[1, 4, 1, 3]),
        Team(3, "c", "Gamma", "C", [], wins=1, scores=[95, 115], opponents=[4, 1, 4, 2]),
        Team(4, "d", "Delta", "D", [], wins=0, scores=[90, 85], opponents=[3, 2, 3, 1]),
    ]
    data = LeagueData({"settings": {"playoff_week_start": 5, "playoff_teams": 2}}, teams, {}, {})
    rankings = pd.DataFrame(
        {"roster_id": [1, 2, 3, 4], "Power Score": [80, 60, 40, 20]},
        index=[1, 2, 3, 4],
    )

    first = playoff_probabilities(data, rankings, week=2, simulations=500, seed=123)
    second = playoff_probabilities(data, rankings, week=2, simulations=500, seed=123)

    pd.testing.assert_frame_equal(first, second)
    total = first["Playoffs"].str.rstrip("%").astype(float).sum()
    assert total == pytest.approx(200, abs=0.05)
