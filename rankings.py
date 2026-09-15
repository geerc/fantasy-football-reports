import random
import re
from typing import Optional

import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz, process


PLAYER_ALIASES = {"Marquise Brown": "Hollywood Brown"}


def _require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def fuzzy_merge(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    left_name: str,
    right_name: str,
    left_team: str,
    right_team: str,
    threshold: int = 90,
) -> pd.DataFrame:
    _require_columns(left, [left_name, left_team], "left data")
    _require_columns(right, [right_name, right_team], "right data")
    left = left.copy()
    right = right.copy()
    left["combined_key"] = left[left_name].fillna("").astype(str).str.strip() + " " + left[left_team].fillna("").astype(str).str.strip()
    right["combined_key"] = right[right_name].fillna("").astype(str).str.strip() + " " + right[right_team].fillna("").astype(str).str.strip()
    choices = right["combined_key"]
    matches = left["combined_key"].apply(
        lambda value: process.extractOne(value, choices, scorer=fuzz.token_sort_ratio, score_cutoff=threshold)
    )
    left["Best Match"] = matches.apply(lambda match: match[0] if match else None)
    left["Match Score"] = matches.apply(lambda match: match[1] if match else None)
    merged = left.merge(right, left_on="Best Match", right_on="combined_key", how="left")
    return merged.drop(columns=["combined_key_x", "combined_key_y"])


def generate_power_rankings(league, week: int, values_path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not values_path.exists():
        raise FileNotFoundError(f"Player values not found for week {week}: {values_path}")
    values = pd.read_csv(values_path)
    _require_columns(values, ["Player Name", "Pos", "Value", "NFL_Team"], str(values_path))
    values["Player Name"] = values["Player Name"].replace(PLAYER_ALIASES)
    values["Value"] = pd.to_numeric(values["Value"], errors="coerce")
    values["Pos"] = values["Pos"].astype("string").str.extract(r"(\D+)")[0].str.strip()
    values = values.dropna(subset=["Player Name", "NFL_Team", "Value"])
    values = values[~values["Pos"].isin(["DST", "PK"])]

    rosters = pd.DataFrame(
        [
            [team.team_name, player.name, player.position, player.proTeam]
            for team in league.teams
            for player in team.roster
        ],
        columns=["Team", "Player", "Position", "NFL_Team"],
    ).dropna(subset=["Player", "NFL_Team"])
    rosters = rosters[~rosters["Position"].isin(["D/ST", "PK"])]

    matched = fuzzy_merge(
        values[["Player Name", "Pos", "Value", "NFL_Team"]],
        rosters[["Player", "Team", "NFL_Team"]],
        left_name="Player Name",
        right_name="Player",
        left_team="NFL_Team",
        right_team="NFL_Team",
        threshold=85,
    )
    players = matched[["Player Name", "Pos", "Value", "Team"]].drop_duplicates("Player Name")
    unmatched = players[players["Team"].isna()]
    if not unmatched.empty:
        print(f"Warning: {len(unmatched)} player values could not be matched to a roster.")
    players = players.dropna(subset=["Team"])

    counts = players.groupby("Team", as_index=False).size()
    team_values = players.groupby("Team", as_index=False)["Value"].sum().merge(counts, on="Team")
    team_values["Value"] = team_values["Value"] / team_values["size"]

    raw = pd.DataFrame(
        [(re.sub(r"Team\((.*?)\)", r"\1", str(team)), score) for score, team in league.power_rankings(week=week)],
        columns=["Team", "Performance Score"],
    )
    raw["Performance Score"] = pd.to_numeric(
        raw["Performance Score"].astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce"
    )
    result = raw.merge(team_values[["Team", "Value"]], on="Team", how="left", validate="one_to_one")
    if result[["Performance Score", "Value"]].isna().any().any():
        missing = result.loc[result["Value"].isna(), "Team"].tolist()
        raise ValueError(f"Incomplete ranking inputs; missing player values for: {', '.join(missing)}")

    def normalize(series: pd.Series) -> pd.Series:
        span = series.max() - series.min()
        return pd.Series(0.5, index=series.index) if span == 0 else (series - series.min()) / span

    value_weight = round(0.5585 * np.exp(-0.1147 * week), 2)
    result["Power Score"] = normalize(result["Performance Score"]) * (1 - value_weight) + normalize(result["Value"]) * value_weight
    result["Performance Rank"] = result["Performance Score"].rank(ascending=False, method="min").astype(int)
    result["KTC Value Rank"] = result["Value"].rank(ascending=False, method="min").astype(int)
    result["Power Score"] = (result["Power Score"] * 100).round()
    result = result.sort_values("Power Score", ascending=False)[["Team", "Power Score", "Performance Rank", "KTC Value Rank"]]
    result.index = range(1, len(result) + 1)
    return result, players


def add_weekly_change(current: pd.DataFrame, previous: pd.DataFrame) -> pd.DataFrame:
    current = current.copy()
    current_positions = {team: rank for rank, team in current["Team"].items()}
    previous_positions = {team: rank for rank, team in previous["Team"].items()}
    def display(team: str) -> str:
        if team not in previous_positions:
            return "NEW"
        change = previous_positions[team] - current_positions[team]
        if change > 0:
            return f'<span style="color: #15803d; font-weight: 700;">↑ {change}</span>'
        if change < 0:
            return f'<span style="color: #b91c1c; font-weight: 700;">↓ {abs(change)}</span>'
        return '<span style="color: #111827;">—</span>'
    current.insert(2, "Weekly Change", current["Team"].map(display))
    return current


def generate_expected_standings(league, rankings: pd.DataFrame, week: int, regular_season_weeks: int) -> pd.DataFrame:
    scores = rankings.set_index("Team")["Power Score"].to_dict()
    rows = []
    for team in league.teams:
        probabilities = []
        opponent_scores = []
        for week_number, opponent in enumerate(team.schedule, start=1):
            if week_number <= week or week_number > regular_season_weeks:
                continue
            denominator = scores[team.team_name] + scores[opponent.team_name]
            probabilities.append(0.5 if denominator == 0 else scores[team.team_name] / denominator)
            opponent_scores.append(scores[opponent.team_name])
        rows.append([
            team.team_name,
            round(team.wins + sum(probabilities), 2),
            round(team.losses + len(probabilities) - sum(probabilities), 2),
            round(sum(opponent_scores) / len(opponent_scores)) if opponent_scores else None,
        ])
    result = pd.DataFrame(rows, columns=["Team", "Projected Wins", "Projected Losses", "SOS"])
    if not (9 < week < regular_season_weeks):
        result = result.drop(columns="SOS")
    result = result.sort_values("Projected Wins", ascending=False)
    result.index = range(1, len(result) + 1)
    return result


def generate_playoff_probabilities(league, rankings: pd.DataFrame, week: int, regular_season_weeks: int, playoff_teams: int, simulations: int, seed: Optional[int]) -> pd.DataFrame:
    team_names = [team.team_name for team in league.teams]
    team_index = {name: index for index, name in enumerate(team_names)}
    strength = rankings.set_index("Team")["Power Score"].to_dict()
    matchups = []
    for week_number in range(week + 1, regular_season_weeks + 1):
        for matchup in league.scoreboard(week_number):
            if matchup.home_team and matchup.away_team:
                matchups.append((team_index[matchup.home_team.team_name], team_index[matchup.away_team.team_name]))
    rng = random.Random(seed)
    seed_counts = np.zeros((len(team_names), playoff_teams), dtype=int)
    base_wins = np.array([team.wins for team in league.teams], dtype=float)
    tie_breakers = np.array([sum(team.scores) for team in league.teams]) / 1_000_000
    for _ in range(simulations):
        totals = base_wins + tie_breakers
        for home, away in matchups:
            home_strength = strength[team_names[home]]
            away_strength = strength[team_names[away]]
            denominator = home_strength + away_strength
            home_probability = 0.5 if denominator == 0 else home_strength / denominator
            totals[home if rng.random() < home_probability else away] += 1
        order = np.argsort(totals)[::-1]
        for seed_number, index in enumerate(order[:playoff_teams]):
            seed_counts[index, seed_number] += 1
    percentages = seed_counts / simulations * 100
    columns = [f"{number}{'st' if number == 1 else 'nd' if number == 2 else 'rd' if number == 3 else 'th'} Seed" for number in range(1, playoff_teams + 1)]
    result = pd.DataFrame(percentages, columns=columns)
    result.insert(0, "Playoffs", percentages.sum(axis=1))
    result.insert(0, "Team", team_names)
    result = result.sort_values(["Playoffs", *columns], ascending=False)
    for column in ["Playoffs", *columns]:
        result[column] = result[column].round(2).astype(str) + "%"
    result.index = range(1, len(result) + 1)
    return result
