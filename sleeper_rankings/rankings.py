from __future__ import annotations

import math
import random
import re
import unicodedata

import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz

from .models import LeagueData, Team


PERFORMANCE_HALF_LIFE = 6.0
RECENT_HALF_LIFE = 2.0
STARTER_WEIGHT = 0.75
DEPTH_WEIGHT = 0.25


def _weighted_mean(values: list[float], half_life: float) -> float:
    if not values:
        return 0.5
    weights = np.exp(-math.log(2) * np.arange(len(values) - 1, -1, -1) / half_life)
    return float(np.average(values, weights=weights))


def _cdf(value: float) -> float:
    return (1 + math.erf(value / math.sqrt(2))) / 2


def _weekly_strength(values: list[float], value: float) -> float:
    """Map a weekly score to a robust 0-1 strength without forced endpoints."""
    median = float(np.median(values))
    mad = float(np.median(np.abs(np.asarray(values, dtype=float) - median)))
    scale = mad * 1.4826
    if scale == 0:
        scale = float(np.std(values))
    return 0.5 if scale == 0 else _cdf((value - median) / scale)


def _relative_score(series: pd.Series, minimum_spread: float) -> pd.Series:
    """Produce stable 10-90 ratings while preserving meaningful input gaps."""
    values = series.astype(float)
    observed_spread = float(values.std(ddof=0))
    if observed_spread == 0:
        return pd.Series(50.0, index=series.index)
    spread = max(observed_spread, minimum_spread)
    z = ((values - float(values.mean())) / spread).clip(-2.5, 2.5)
    return z.map(lambda item: 10 + 80 * _cdf(float(item)))


def performance_score(data: LeagueData, team: Team, week: int) -> float:
    all_play: list[float] = []
    scoring: list[float] = []
    for number in range(week):
        weekly = [float(candidate.scores[number]) for candidate in data.teams]
        score = float(team.scores[number])
        wins = sum(score > opponent for opponent in weekly if opponent != score)
        ties = sum(score == opponent for opponent in weekly) - 1
        all_play.append((wins + ties * 0.5) / max(len(weekly) - 1, 1))
        scoring.append(_weekly_strength(weekly, score))
    season_all_play = _weighted_mean(all_play, PERFORMANCE_HALF_LIFE)
    season_scoring = _weighted_mean(scoring, PERFORMANCE_HALF_LIFE)
    recent_form = _weighted_mean(scoring, RECENT_HALF_LIFE)
    return round(season_all_play * 0.50 + season_scoring * 0.30 + recent_form * 0.20, 6)


def _key(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _team_key(value: object) -> str:
    aliases = {"jac": "jax", "la": "lar", "was": "wsh"}
    key = _key(value)
    return aliases.get(key, key)


def _match_value(player: dict, values: pd.DataFrame) -> float:
    position = _key(player.get("position"))
    if position in {"def", "dst", "k", "pk"}:
        return 0.0
    name = player.get("full_name") or " ".join(filter(None, [player.get("first_name"), player.get("last_name")]))
    name_key = _key(name)
    team_key = _team_key(player.get("team"))
    candidates = values[values["_position"] == position]
    if team_key:
        same_team = candidates[candidates["_team"] == team_key]
        if same_team.empty:
            return 0.0
        candidates = same_team
    exact = candidates[candidates["_name"] == name_key]
    if len(exact) == 1:
        return float(exact.iloc[0]["value"])
    scored = sorted(
        ((fuzz.ratio(name_key, candidate), index) for index, candidate in candidates["_name"].items()),
        reverse=True,
    )
    if not scored or scored[0][0] < 92 or (len(scored) > 1 and scored[0][0] - scored[1][0] < 3):
        return 0.0
    return float(values.loc[scored[0][1], "value"])


def _eligible(position: str, slot: str) -> bool:
    groups = {
        "FLEX": {"RB", "WR", "TE"},
        "WRRB_FLEX": {"RB", "WR"},
        "REC_FLEX": {"WR", "TE"},
        "SUPER_FLEX": {"QB", "RB", "WR", "TE"},
        "IDP_FLEX": {"DL", "LB", "DB"},
    }
    return position == slot or position in groups.get(slot, set())


def _roster_value(data: LeagueData, team: Team, values: pd.DataFrame) -> float:
    players = []
    for player_id in team.players:
        player = data.players.get(str(player_id)) or {}
        position = str(player.get("position") or "").upper()
        if position in {"DEF", "DST", "K", "PK"}:
            continue
        players.append((position, _match_value(player, values)))

    excluded = {"BN", "BENCH", "IR", "TAXI", "DEF", "DST", "K", "PK"}
    slots = [str(slot).upper() for slot in data.league.get("roster_positions", []) if str(slot).upper() not in excluded]
    if not slots:
        slots = [position for position, _ in players]
    slots.sort(key=lambda slot: sum(_eligible(position, slot) for position, _ in players))
    unused = set(range(len(players)))
    starters = []
    for slot in slots:
        choices = [index for index in unused if _eligible(players[index][0], slot)]
        if not choices:
            starters.append(0.0)
            continue
        selected = max(choices, key=lambda index: players[index][1])
        unused.remove(selected)
        starters.append(players[selected][1])
    depth = [players[index][1] for index in unused]
    starter_value = sum(starters) / len(starters) if starters else 0.0
    depth_value = sum(depth) / len(depth) if depth else 0.0
    return starter_value * STARTER_WEIGHT + depth_value * DEPTH_WEIGHT


def roster_values(data: LeagueData, values: pd.DataFrame) -> dict[int, float]:
    prepared = values.copy()
    prepared["_name"] = prepared["name"].map(_key)
    prepared["_position"] = prepared["position"].map(_key)
    prepared["_team"] = prepared["nfl_team"].map(_team_key)
    return {team.roster_id: _roster_value(data, team, prepared) for team in data.teams}


def _validate_completed_scores(data: LeagueData, week: int) -> None:
    if week < 1:
        raise ValueError("Power rankings require at least one completed week")
    incomplete = [team.name for team in data.teams if len(team.scores) < week or not np.isfinite(team.scores[week - 1])]
    if incomplete:
        raise ValueError(f"Week {week} is incomplete for: {', '.join(incomplete)}")


def power_rankings(data: LeagueData, week: int, values: pd.DataFrame) -> pd.DataFrame:
    _validate_completed_scores(data, week)
    roster_value = roster_values(data, values)
    rows = [{"Team": team.name, "roster_id": team.roster_id, "Performance Score": performance_score(data, team, week), "Roster Value": roster_value[team.roster_id]} for team in data.teams]
    frame = pd.DataFrame(rows)
    value_weight = 0.20 + 0.25 * math.exp(-math.log(2) * (week - 1) / 5)
    performance_rating = _relative_score(frame["Performance Score"], 0.15)
    typical_value = max(abs(float(frame["Roster Value"].median())) * 0.10, 250.0)
    value_rating = _relative_score(frame["Roster Value"], typical_value)
    frame["Power Score"] = (performance_rating * (1 - value_weight) + value_rating * value_weight).round().astype(int)
    frame["Performance Rank"] = frame["Performance Score"].rank(ascending=False, method="min").astype(int)
    frame["Team Value Rank"] = frame["Roster Value"].rank(ascending=False, method="min").astype(int)
    frame = frame.sort_values(["Power Score", "Performance Score"], ascending=False).reset_index(drop=True)
    frame.index = range(1, len(frame) + 1)
    return frame


def add_weekly_change(current: pd.DataFrame, previous: pd.DataFrame) -> pd.DataFrame:
    current = current.copy()
    prior_positions = {int(row.roster_id): rank for rank, row in previous.iterrows()}
    changes = []
    for rank, row in current.iterrows():
        previous_rank = prior_positions.get(int(row.roster_id))
        if previous_rank is None:
            changes.append("New")
        elif previous_rank > rank:
            changes.append(f"↑ {previous_rank - rank}")
        elif previous_rank < rank:
            changes.append(f"↓ {rank - previous_rank}")
        else:
            changes.append("—")
    current.insert(current.columns.get_loc("Power Score") + 1, "Weekly Change", changes)
    return current


def luck_index(data: LeagueData, week: int) -> pd.DataFrame:
    rows = []
    for team in data.teams:
        total = 0.0
        history = np.array(team.scores[:week], dtype=float)
        mean, std = (history.mean(), history.std()) if len(history) else (0, 0)
        for number, score in enumerate(history):
            opponent_id = team.opponents[number]
            if opponent_id is None:
                continue
            all_scores = sorted((item.scores[number] for item in data.teams), reverse=True)
            rank = all_scores.index(score) + 1
            opponent_score = data.by_roster[opponent_id].scores[number]
            won = score > opponent_score
            schedule = (rank - 1) / (len(data.teams) - 1) if won else -(len(data.teams) - rank) / (len(data.teams) - 1)
            historical = np.clip((score - mean) / std / 2, -1, 1) if std else 0
            margin = np.clip((score - opponent_score) / max(min(score, opponent_score), 1) / 0.1, -1, 1)
            close_game = np.sign(margin) * (1 - abs(margin))
            total += schedule * 0.65 + historical * 0.25 + close_game * 0.10
        rows.append({"Team": team.name, "Luck Index": round(total, 2)})
    frame = pd.DataFrame(rows).sort_values("Luck Index", ascending=False).reset_index(drop=True)
    frame.index = range(1, len(frame) + 1)
    return frame


def projected_standings(data: LeagueData, rankings: pd.DataFrame, week: int) -> pd.DataFrame:
    scores = rankings.set_index("roster_id")["Power Score"].to_dict()
    regular_weeks = int(data.league["settings"].get("playoff_week_start", 15)) - 1
    rows = []
    median_game = bool(data.league["settings"].get("league_average_match"))
    ordered_strength = sorted(scores.values())
    for team in data.teams:
        expected = float(team.wins)
        remaining = 0
        for number in range(week, regular_weeks):
            if number >= len(team.opponents) or team.opponents[number] is None:
                continue
            opponent_score = scores[team.opponents[number]]
            denominator = scores[team.roster_id] + opponent_score
            expected += 0.5 if denominator == 0 else scores[team.roster_id] / denominator
            remaining += 1
            if median_game:
                below = sum(value < scores[team.roster_id] for value in ordered_strength)
                equal = sum(value == scores[team.roster_id] for value in ordered_strength) - 1
                expected += (below + equal * 0.5) / max(len(ordered_strength) - 1, 1)
                remaining += 1
        rows.append({"Team": team.name, "Projected Wins": round(expected, 2), "Projected Losses": round(team.losses + remaining - (expected - team.wins), 2)})
    frame = pd.DataFrame(rows).sort_values("Projected Wins", ascending=False).reset_index(drop=True)
    frame.index = range(1, len(frame) + 1)
    return frame


def playoff_probabilities(data: LeagueData, rankings: pd.DataFrame, week: int, simulations: int, seed: int | None) -> pd.DataFrame:
    rng = random.Random(seed)
    teams = data.teams
    index = {team.roster_id: number for number, team in enumerate(teams)}
    playoff_teams = int(data.league["settings"].get("playoff_teams", len(teams) // 2))
    regular_weeks = int(data.league["settings"].get("playoff_week_start", 15)) - 1
    strength = rankings.set_index("roster_id")["Power Score"].to_dict()
    median_game = bool(data.league["settings"].get("league_average_match"))
    counts = np.zeros((len(teams), playoff_teams), dtype=int)
    for _ in range(simulations):
        wins = np.array([team.wins for team in teams], dtype=float)
        points = np.array([sum(team.scores[:week]) for team in teams]) / 1_000_000
        for number in range(week, regular_weeks):
            seen = set()
            for team in teams:
                if number >= len(team.opponents) or team.opponents[number] is None:
                    continue
                pair = tuple(sorted((team.roster_id, team.opponents[number])))
                if pair in seen:
                    continue
                seen.add(pair)
                left, right = pair
                denominator = strength[left] + strength[right]
                probability = 0.5 if denominator == 0 else strength[left] / denominator
                wins[index[left if rng.random() < probability else right]] += 1
            if median_game:
                # Rank a noisy weekly strength draw; the top half earns the median-game win.
                weekly_order = sorted(teams, key=lambda item: strength[item.roster_id] + rng.gauss(0, 20), reverse=True)
                for item in weekly_order[: len(teams) // 2]:
                    wins[index[item.roster_id]] += 1
        order = np.argsort(wins + points)[::-1]
        for position, team_index in enumerate(order[:playoff_teams]):
            counts[team_index, position] += 1
    rows = []
    for number, team in enumerate(teams):
        row = {"Team": team.name, "Playoffs": counts[number].sum() / simulations * 100}
        row.update({f"Seed {seed_number + 1}": counts[number, seed_number] / simulations * 100 for seed_number in range(playoff_teams)})
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values(["Playoffs", "Seed 1"], ascending=False).reset_index(drop=True)
    for column in frame.columns[1:]:
        frame[column] = frame[column].map(lambda value: f"{value:.2f}%")
    frame.index = range(1, len(frame) + 1)
    return frame
