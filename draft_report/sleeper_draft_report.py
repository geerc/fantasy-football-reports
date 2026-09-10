import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
from io import StringIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from tabulate import tabulate

try:
    from .report import write_report_atomic
except ImportError:  # Support direct execution from the draft_report directory.
    from report import write_report_atomic


SLEEPER_API = "https://api.sleeper.app/v1"
FANTASYPROS_KICKERS = "https://www.fantasypros.com/nfl/projections/k.php?week=draft"
FANTASYPROS_DEFENSES = "https://www.fantasypros.com/nfl/projections/dst.php?week=draft"
FANTASYPROS_MOCK_ADP = "https://draftwizard.fantasypros.com/football/adp/mock-drafts/overall/{filters}"
KTC_REDRAFT_RANKINGS = "https://keeptradecut.com/fantasy-rankings?page={page}&filters=QB|WR|RB|TE|DST|PK&format={format}"
CBS_PROJECTIONS = "https://www.cbssports.com/fantasy/football/stats/{position}/{season}/restofseason/projections/nonppr/"
NFL_REGULAR_SEASON_GAMES = 17
RADAR_POSITIONS = ("QB", "RB", "WR", "TE", "FLEX", "REC_FLEX", "WRRB_FLEX", "K", "DST")
SIMULATION_EXCLUDED_POSITIONS = {"K", "DST"}
NFL_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
FLEX_ELIGIBILITY = {
    "FLEX": {"RB", "WR", "TE"},
    "SUPER_FLEX": {"QB", "RB", "WR", "TE"},
    "REC_FLEX": {"WR", "TE"},
    "WRRB_FLEX": {"WR", "RB"},
}
POSITION_ALIASES = {"DEF": "DST", "PK": "K"}
NFL_TEAM_ALIASES = {
    "JAC": "JAX", "KCC": "KC", "LVR": "LV", "NEP": "NE", "NOS": "NO",
    "SFO": "SF", "TBB": "TB", "WSH": "WAS",
}
DEFAULT_VOR_BASELINE = {"QB": 13, "RB": 35, "WR": 36, "TE": 13, "K": 8, "DST": 3}
DEFAULT_AI_MODEL = "gpt-5.6-terra"
DEFAULT_AI_REASONING_EFFORT = "low"
LOCAL_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
PROJECTION_CACHE_DIR = Path(__file__).resolve().parent / "cache"
NFL_TEAM_CODES = {
    "arizona cardinals": "ARI", "atlanta falcons": "ATL", "baltimore ravens": "BAL",
    "buffalo bills": "BUF", "carolina panthers": "CAR", "chicago bears": "CHI",
    "cincinnati bengals": "CIN", "cleveland browns": "CLE", "dallas cowboys": "DAL",
    "denver broncos": "DEN", "detroit lions": "DET", "green bay packers": "GB",
    "houston texans": "HOU", "indianapolis colts": "IND", "jacksonville jaguars": "JAX",
    "kansas city chiefs": "KC", "las vegas raiders": "LV", "los angeles chargers": "LAC",
    "los angeles rams": "LAR", "miami dolphins": "MIA", "minnesota vikings": "MIN",
    "new england patriots": "NE", "new orleans saints": "NO", "new york giants": "NYG",
    "new york jets": "NYJ", "philadelphia eagles": "PHI", "pittsburgh steelers": "PIT",
    "san francisco 49ers": "SF", "seattle seahawks": "SEA", "tampa bay buccaneers": "TB",
    "tennessee titans": "TEN", "washington commanders": "WAS",
}
CBS_DST_CODES = {
    "Arizona": "ARI", "Atlanta": "ATL", "Baltimore": "BAL", "Buffalo": "BUF",
    "Carolina": "CAR", "Chicago": "CHI", "Cincinnati": "CIN", "Cleveland": "CLE",
    "Dallas": "DAL", "Denver": "DEN", "Detroit": "DET", "Green Bay": "GB",
    "Houston": "HOU", "Indianapolis": "IND", "Jacksonville": "JAX", "Kansas City": "KC",
    "L.A. Chargers": "LAC", "L.A. Rams": "LAR", "Las Vegas": "LV", "Miami": "MIA",
    "Minnesota": "MIN", "N.Y. Giants": "NYG", "N.Y. Jets": "NYJ", "New England": "NE",
    "New Orleans": "NO", "Philadelphia": "PHI", "Pittsburgh": "PIT", "San Francisco": "SF",
    "Seattle": "SEA", "Tampa Bay": "TB", "Tennessee": "TEN", "Washington": "WAS",
}
PLAYER_NAME_ALIASES = {
    "chigokonkwo": "chigoziemokonkwo",
    "kennygainwell": "kennethgainwell",
}


@dataclass(frozen=True)
class PlayerProjection:
    name: str
    position: str
    team: str
    points: float
    points_vor: float
    vor_rank: int
    adp: float = math.nan
    ktc_value: float = math.nan


def api_get(path, *, session=requests):
    response = session.get(f"{SLEEPER_API}{path}", timeout=30)
    response.raise_for_status()
    return response.json()


def normalize_name(value):
    value = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    value = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", value.lower())
    normalized = re.sub(r"[^a-z0-9]", "", value)
    return PLAYER_NAME_ALIASES.get(normalized, normalized)


def normalize_position(value):
    return POSITION_ALIASES.get(str(value).upper(), str(value).upper())


def normalize_nfl_team(value):
    team = str(value or "").upper()
    return NFL_TEAM_ALIASES.get(team, team)


def run_ffanalytics(*, season, scoring_settings, output_path, rscript="Rscript"):
    script = Path(__file__).with_name("ffanalytics_projections.R")
    scoring_path = Path(output_path).with_suffix(".scoring.json")
    scoring_path.write_text(json.dumps(scoring_settings), encoding="utf-8")
    try:
        subprocess.run([rscript, str(script), str(season), str(scoring_path), str(output_path)], check=True)
    finally:
        scoring_path.unlink(missing_ok=True)


def cached_projection_path(*, season, scoring_settings, cache_dir=None):
    cache_dir = Path(cache_dir or PROJECTION_CACHE_DIR)
    scoring_json = json.dumps(scoring_settings, sort_keys=True, separators=(",", ":"))
    scoring_hash = hashlib.sha256(scoring_json.encode("utf-8")).hexdigest()[:12]
    return cache_dir / f"ffanalytics-{season}-{scoring_hash}.csv"


def get_projection_path(*, season, scoring_settings, rscript, refresh=False, cache_dir=None):
    path = cached_projection_path(
        season=season, scoring_settings=scoring_settings, cache_dir=cache_dir,
    )
    if path.exists() and not refresh:
        print(f"Using cached ffanalytics projections from {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp.csv")
    try:
        run_ffanalytics(
            season=season, scoring_settings=scoring_settings,
            output_path=temporary_path, rscript=rscript,
        )
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    print(f"Cached ffanalytics projections at {path}")
    return path


def load_ffanalytics(path):
    frame = pd.read_csv(path)
    required = {"first_name", "last_name", "team", "position", "points", "points_vor", "rank"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"ffanalytics output is missing columns: {', '.join(sorted(missing))}")
    frame["name"] = (frame["first_name"].fillna("") + " " + frame["last_name"].fillna("")).str.strip()
    frame["position"] = frame["position"].map(normalize_position)
    return frame[["name", "team", "position", "points", "points_vor", "rank"]]


def fetch_fantasypros_position(position, *, session=requests):
    url = FANTASYPROS_KICKERS if position == "K" else FANTASYPROS_DEFENSES
    response = session.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    tables = pd.read_html(StringIO(response.text))
    table = next((item for item in tables if any("Player" in str(column) for column in item.columns)), None)
    if table is None:
        raise ValueError(f"FantasyPros {position} projections table was not found")
    table.columns = [column[-1] if isinstance(column, tuple) else column for column in table.columns]
    player_column = next(column for column in table.columns if "Player" in str(column))
    points_column = next((column for column in table.columns if str(column).strip().upper() in {"FPTS", "POINTS"}), None)
    if points_column is None:
        raise ValueError(f"FantasyPros {position} projections did not contain a points column")
    extracted = table[player_column].astype(str).str.extract(r"^(.*?)\s+([A-Z]{2,3})$", expand=True)
    result = pd.DataFrame({
        "name": extracted[0].fillna(table[player_column]).str.strip(),
        "team": extracted[1].fillna(""),
        "position": position,
        "points": pd.to_numeric(table[points_column], errors="coerce"),
    }).dropna(subset=["points"])
    return result


def fantasypros_adp_settings(league, team_count):
    roster_positions = [normalize_position(position) for position in league.get("roster_positions", [])]
    roster_format = "2qb" if "SUPER_FLEX" in roster_positions or roster_positions.count("QB") > 1 else "default"
    receptions = float((league.get("scoring_settings") or {}).get("rec", 0) or 0)
    scoring = "ppr" if receptions == 1 else "half" if receptions == 0.5 else "std"
    supported_team_counts = {4, 6, 8, 10, 12, 14, 16}
    filters = [roster_format, scoring]
    if team_count in supported_team_counts:
        filters.append(f"{team_count}-teams")
    return {
        "roster_format": roster_format,
        "scoring": scoring,
        "team_count": team_count if team_count in supported_team_counts else None,
        "url": FANTASYPROS_MOCK_ADP.format(filters="-".join(filters)),
    }


def overall_pick_from_round_slot(value, team_count):
    round_slot = float(value)
    round_number = math.floor(round_slot)
    slot = round((round_slot - round_number) * 100)
    if slot == 0:
        slot = team_count
    return float((round_number - 1) * team_count + slot)


def fetch_fantasypros_adp(league, team_count, *, session=requests):
    settings = fantasypros_adp_settings(league, team_count)
    response = session.get(settings["url"], timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    table = next(
        (
            item for item in pd.read_html(StringIO(response.text))
            if {"Player", "Avg Pick"}.issubset({str(column) for column in item.columns})
        ),
        None,
    )
    if table is None:
        raise ValueError("FantasyPros league-adjusted ADP table was not found")
    result = pd.DataFrame({
        "name": table["Player"].astype(str).str.strip(),
        "position": table["Position"].astype(str).str.extract(r"^([A-Z]+)", expand=False).map(normalize_position),
        "adp": table["Avg Pick"].map(lambda value: overall_pick_from_round_slot(value, team_count)),
    })
    return result.dropna(subset=["name", "position", "adp"]), settings


def ktc_ranking_settings(league):
    roster_positions = [normalize_position(position) for position in league.get("roster_positions", [])]
    superflex = "SUPER_FLEX" in roster_positions or roster_positions.count("QB") > 1
    return {
        "format": 2 if superflex else 1,
        "roster_format": "superflex" if superflex else "1qb",
    }


def fetch_ktc_rankings(league, *, session=requests, pages=8):
    settings = ktc_ranking_settings(league)
    records = []
    headers = {"User-Agent": "fantasy-football-reports/0.1"}
    for page in range(pages):
        response = session.get(
            KTC_REDRAFT_RANKINGS.format(page=page, format=settings["format"]),
            timeout=30,
            headers=headers,
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        for row in soup.select("div.onePlayer"):
            name = row.select_one("div.player-name a")
            position = row.select_one("p.position")
            value = row.select_one("div.value")
            if all((name, position, value)):
                team = row.select_one("div.player-name span.player-team")
                records.append({
                    "name": name.get_text(strip=True),
                    "team": normalize_nfl_team(team.get_text(strip=True) if team else ""),
                    "position": normalize_position(re.sub(r"\d+$", "", position.get_text(strip=True))),
                    "ktc_value": pd.to_numeric(value.get_text(strip=True).replace(",", ""), errors="coerce"),
                })
    frame = pd.DataFrame(records).dropna(subset=["name", "ktc_value"]).drop_duplicates(
        ["name", "position"], keep="first",
    )
    if frame.empty:
        raise ValueError("KeepTradeCut returned no fantasy-ranking values")
    return frame, settings


def add_ktc_values(projections, values):
    result = projections.copy()
    result["ktc_key"] = result.apply(
        lambda row: f"{normalize_name(row['name'])}:{normalize_position(row['position'])}", axis=1,
    )
    values = values.copy()
    values["ktc_key"] = values.apply(
        lambda row: f"{normalize_name(row['name'])}:{normalize_position(row['position'])}", axis=1,
    )
    values = values.drop_duplicates("ktc_key", keep="first")[["ktc_key", "ktc_value"]]
    return result.merge(values, on="ktc_key", how="left").drop(columns="ktc_key")


def ktc_player_frame(values):
    frame = values.copy()
    frame["points"] = frame["ktc_value"]
    frame["points_vor"] = frame["ktc_value"]
    frame["rank"] = frame["ktc_value"].rank(method="min", ascending=False).astype(int)
    return frame[["name", "team", "position", "points", "points_vor", "rank", "ktc_value"]]


def add_adp_to_projections(projections, adp):
    result = projections.copy()
    result["adp_key"] = result.apply(
        lambda row: f"{normalize_name(row['name'])}:{normalize_position(row['position'])}", axis=1,
    )
    adp = adp.copy()
    adp["adp_key"] = adp.apply(
        lambda row: f"{normalize_name(row['name'])}:{normalize_position(row['position'])}", axis=1,
    )
    adp = adp.drop_duplicates("adp_key", keep="first")[["adp_key", "adp"]]
    return result.merge(adp, on="adp_key", how="left").drop(columns="adp_key")


def fetch_cbs_position(position, season, *, session=requests):
    response = session.get(
        CBS_PROJECTIONS.format(position=position, season=season),
        timeout=30, headers={"User-Agent": "Mozilla/5.0"},
    )
    response.raise_for_status()
    tables = pd.read_html(StringIO(response.text))
    if not tables:
        raise ValueError(f"CBS {position} projections table was not found")
    table = tables[0]
    player_column = table.columns[0]
    points_column = next(
        (column for column in table.columns if str(column[-1] if isinstance(column, tuple) else column).startswith("fpts")),
        None,
    )
    if points_column is None:
        raise ValueError(f"CBS {position} projections did not contain a fantasy-points column")
    points = pd.to_numeric(table[points_column], errors="coerce")
    if position == "K":
        parsed = table[player_column].astype(str).str.split(r"\s{2,}", regex=True)
        result = pd.DataFrame({
            "name": parsed.map(lambda parts: parts[3] if len(parts) >= 6 else ""),
            "team": parsed.map(lambda parts: parts[5] if len(parts) >= 6 else ""),
            "position": position,
            "points": points,
        })
    else:
        names = table[player_column].astype(str).str.strip()
        result = pd.DataFrame({
            "name": names,
            "team": names.map(CBS_DST_CODES).fillna(""),
            "position": position,
            "points": points,
        })
    return result[(result["name"] != "") & result["points"].notna()].reset_index(drop=True)


def combine_supplemental_projections(*frames):
    combined = pd.concat(frames, ignore_index=True)
    combined["match_key"] = combined.apply(
        lambda row: f"{normalize_name(row['name'])}:{normalize_position(row['position'])}", axis=1,
    )
    return combined.drop_duplicates("match_key", keep="last").drop(columns="match_key")


def add_supplemental_vor_and_rerank(projections, supplemental, baseline=DEFAULT_VOR_BASELINE):
    supplemental = supplemental.copy()
    additions = []
    existing_positions = set(projections["position"].map(normalize_position))
    for position, group in supplemental.groupby("position"):
        position = normalize_position(position)
        if position in existing_positions and position != "K":
            continue
        group = group.sort_values("points", ascending=False).reset_index(drop=True)
        replacement_index = min(baseline[position], len(group)) - 1
        replacement = group.iloc[replacement_index]["points"] if replacement_index >= 0 else 0
        group["points_vor"] = group["points"] - replacement
        additions.append(group)
    combined = pd.concat(
        [projections.drop(columns=["rank"], errors="ignore"), *additions], ignore_index=True,
    )
    combined["rank"] = combined["points_vor"].rank(method="dense", ascending=False).astype(int)
    return combined


def add_kicker_vor_and_rerank(projections, kickers, baseline=DEFAULT_VOR_BASELINE):
    return add_supplemental_vor_and_rerank(projections, kickers, baseline)


def projection_index(frame):
    index = {}
    for row in frame.itertuples(index=False):
        ktc_value = (
            float(row.ktc_value) if pd.notna(row.ktc_value) else math.nan
        ) if hasattr(row, "ktc_value") else float(row.points)
        item = PlayerProjection(
            name=str(row.name), position=normalize_position(row.position), team=str(row.team),
            points=float(row.points), points_vor=float(row.points_vor), vor_rank=int(row.rank),
            adp=float(row.adp) if hasattr(row, "adp") and pd.notna(row.adp) else math.nan,
            ktc_value=ktc_value,
        )
        index[(normalize_name(item.name), item.position)] = item
        if item.position == "DST" and item.name.lower() in NFL_TEAM_CODES:
            index[(normalize_name(NFL_TEAM_CODES[item.name.lower()]), item.position)] = item
        if item.position == "DST" and item.team and item.team.lower() != "nan":
            index[(normalize_name(item.team), item.position)] = item
    return index


def generate_dummy_picks(frame, *, teams, rounds, seed):
    if teams < 2 or rounds < 1:
        raise ValueError("Dummy drafts require at least two teams and one round")
    pool = frame.dropna(subset=["name", "position", "points", "rank"]).copy()
    pool = pool[pool["position"].map(normalize_position).isin(RADAR_POSITIONS)]
    required = teams * rounds
    if len(pool) < required:
        raise ValueError(f"Only {len(pool)} projected players are available for {required} draft picks")
    random = np.random.default_rng(seed)
    # Mostly preserve VOR order while allowing plausible draft-day variation.
    pool["dummy_order"] = pd.to_numeric(pool["rank"]) + random.normal(0, 12, len(pool))
    pool = pool.sort_values(["dummy_order", "rank", "name"]).head(required).reset_index(drop=True)
    picks = []
    for overall, row in enumerate(pool.itertuples(index=False), 1):
        round_number = (overall - 1) // teams + 1
        slot_in_round = (overall - 1) % teams
        draft_slot = slot_in_round + 1 if round_number % 2 else teams - slot_in_round
        first_name, _, last_name = str(row.name).partition(" ")
        picks.append({
            "player_id": f"dummy-{overall}", "roster_id": draft_slot,
            "round": round_number, "draft_slot": draft_slot, "pick_no": overall,
            "metadata": {
                "first_name": first_name, "last_name": last_name,
                "position": normalize_position(row.position), "team": str(row.team),
            },
        })
    return picks


def write_dummy_draft(path, *, league_id, season, teams, rounds, seed, picks):
    payload = {
        "league_id": str(league_id), "season": int(season), "type": "snake",
        "teams": int(teams), "rounds": int(rounds), "seed": int(seed), "picks": picks,
    }
    write_report_atomic(Path(path), json.dumps(payload, indent=2) + "\n")


def pick_name_and_position(pick):
    metadata = pick.get("metadata") or {}
    name = metadata.get("first_name", "") + " " + metadata.get("last_name", "")
    if not name.strip():
        name = metadata.get("full_name") or metadata.get("player_id") or pick.get("player_id", "")
    return name.strip(), normalize_position(metadata.get("position", ""))


def find_projection(index, pick, name, position):
    projection = index.get((normalize_name(name), position))
    if projection is None and position == "DST":
        team = (pick.get("metadata") or {}).get("team", "")
        projection = index.get((normalize_name(team), position))
    return projection


def eligible(player_position, slot):
    slot = normalize_position(slot)
    return player_position == slot or player_position in FLEX_ELIGIBILITY.get(slot, set())


def starter_slots(roster_positions, *, exclude=()):
    excluded = {normalize_position(position) for position in exclude}
    slots = []
    for position in roster_positions:
        slot = normalize_position(position)
        if slot in {"BN", "IR", "TAXI", *excluded}:
            continue
        slots.append("QB" if slot == "SUPER_FLEX" else slot)
    native = [slot for slot in slots if slot not in FLEX_ELIGIBILITY]
    flexible = [slot for slot in slots if slot in FLEX_ELIGIBILITY]
    return native + flexible


def optimize_lineup(players, roster_positions, score_attribute="points"):
    slots = starter_slots(roster_positions)
    available = [
        player for player in players
        if math.isfinite(float(getattr(player, score_attribute)))
    ]
    selected = []
    for slot in slots:
        candidates = [player for player in available if eligible(player.position, slot)]
        if not candidates:
            raise ValueError(f"Unable to fill starting lineup slot {slot} from valued drafted players")
        player = max(candidates, key=lambda item: float(getattr(item, score_attribute)))
        selected.append((slot, player))
        available.remove(player)
    return sum(float(getattr(player, score_attribute)) for _, player in selected), selected


def team_name(roster_id, rosters, users):
    roster = next(item for item in rosters if int(item["roster_id"]) == int(roster_id))
    user = users.get(str(roster.get("owner_id")), {})
    metadata = user.get("metadata") or {}
    return metadata.get("team_name") or user.get("display_name") or f"Roster {roster_id}"


def league_context(league):
    settings = league.get("settings") or {}
    scoring = league.get("scoring_settings") or {}
    reception_points = float(scoring.get("rec", 0) or 0)
    if reception_points == 1:
        reception_scoring = "full PPR"
    elif reception_points == 0.5:
        reception_scoring = "half PPR"
    elif reception_points == 0:
        reception_scoring = "standard/non-PPR"
    else:
        reception_scoring = f"{reception_points:g} points per reception"
    return {
        "format": "best ball" if int(settings.get("best_ball", 0) or 0) else "managed lineup",
        "reception_scoring": reception_scoring,
        "starting_lineup_slots": [
            normalize_position(position)
            for position in league.get("roster_positions", [])
            if position not in {"BN", "IR", "TAXI"}
        ],
    }


def draft_impact_score(pick_no, vor_difference, total_picks):
    """Weight VOR misses more heavily when they occur earlier in the draft."""
    if total_picks <= 0:
        return float(vor_difference)
    early_pick_weight = 1 + max(0, total_picks - pick_no) / total_picks
    return vor_difference * early_pick_weight


def player_availability_concern(data):
    injury_status = str(data.get("injury_status") or "").strip()
    roster_status = str(data.get("status") or "").strip()
    concern_statuses = {"doubtful", "ir", "out", "pup", "questionable", "suspended"}
    statuses = []
    for status in (injury_status, roster_status):
        if status.lower() in concern_statuses and status.lower() not in {
            item.lower() for item in statuses
        }:
            statuses.append(status)
    if not statuses:
        return None
    name = data.get("full_name") or (
        f"{data.get('first_name', '')} {data.get('last_name', '')}".strip()
    ) or "Unknown player"
    position = normalize_position(data.get("position", ""))
    body_part = str(data.get("injury_body_part") or "").strip()
    details = ", ".join(statuses)
    if body_part:
        details = f"{details} — {body_part}"
    return f"{name}{f' ({position})' if position else ''}: {details}"


def build_team_results(*, league, rosters, users, picks, projections, player_catalog=None):
    drafted_by_roster = {}
    unmatched = []
    for pick in picks:
        name, position = pick_name_and_position(pick)
        projection = find_projection(projections, pick, name, position)
        if projection is None:
            unmatched.append(f"{name} ({position or 'unknown'})")
            continue
        drafted_by_roster.setdefault(int(pick["roster_id"]), []).append((pick, projection))
    total_picks = max((int(pick["pick_no"]) for pick in picks), default=0)
    reach_value_cutoff = math.floor(total_picks * 0.75)
    results = []
    for roster in rosters:
        roster_id = int(roster["roster_id"])
        drafted = drafted_by_roster.get(roster_id, [])
        roster_players = [item[1] for item in drafted]
        availability_concerns = []
        if player_catalog is not None:
            roster_players = []
            for player_id in roster.get("players") or []:
                data = player_catalog.get(str(player_id), {})
                concern = player_availability_concern(data)
                if concern:
                    availability_concerns.append(concern)
                name = data.get("full_name") or f"{data.get('first_name', '')} {data.get('last_name', '')}".strip()
                position = normalize_position(data.get("position", ""))
                projection = projections.get((normalize_name(name), position))
                if projection is None and position == "DST":
                    projection = projections.get((normalize_name(data.get("team", player_id)), position))
                if projection is None:
                    unmatched.append(f"{name or player_id} ({position or 'unknown'})")
                else:
                    roster_players.append(projection)
        ktc_score, starters = optimize_lineup(
            roster_players, league["roster_positions"], score_attribute="ktc_value",
        )
        position_totals = {position: 0.0 for position in RADAR_POSITIONS}
        for slot, player in starters:
            slot = normalize_position(slot)
            if slot in position_totals:
                position_totals[slot] += player.ktc_value
        deltas = [
            (int(pick["pick_no"]), player, player.adp - int(pick["pick_no"]))
            for pick, player in drafted
            if int(pick["pick_no"]) <= reach_value_cutoff and not math.isnan(player.adp)
        ]
        results.append({
            "roster_id": roster_id,
            "team": team_name(roster_id, rosters, users),
            "ktc_value": ktc_score,
            "position_totals": position_totals,
            "_players": roster_players,
            "roster_construction": dict(sorted(Counter(player.position for player in roster_players).items())),
            "availability_concerns": availability_concerns,
            "roster": [
                {
                    "name": player.name,
                    "position": player.position,
                }
                for player in sorted(roster_players, key=lambda player: (player.position, -player.points, player.name))
            ],
            "reach": max(
                deltas,
                key=lambda item: draft_impact_score(item[0], item[2], total_picks),
            ) if deltas else None,
            "value": min(
                deltas,
                key=lambda item: draft_impact_score(item[0], item[2], total_picks),
            ) if deltas else None,
        })
    results.sort(key=lambda item: item["ktc_value"])
    for rank, result in enumerate(reversed(results), 1):
        result["rank"] = rank
    unmatched = list(dict.fromkeys(unmatched))
    return results, unmatched


def radar_positions_for_league(roster_positions):
    league_positions = {
        normalize_position(position)
        for position in roster_positions
        if normalize_position(position) not in {"BN", "IR", "TAXI"}
    }
    if "SUPER_FLEX" in league_positions:
        league_positions.remove("SUPER_FLEX")
        league_positions.add("QB")
    return tuple(
        position
        for position in RADAR_POSITIONS
        if position in league_positions
    )


def simulation_slots(roster_positions):
    return starter_slots(roster_positions, exclude=SIMULATION_EXCLUDED_POSITIONS)


def partial_lineup(players, slots):
    available = [player for player in players if math.isfinite(player.ktc_value)]
    selected = []
    missing = []
    for slot_index, slot in enumerate(slots):
        candidates = [player for player in available if eligible(player.position, slot)]
        if not candidates:
            missing.append(slot)
            continue
        player = max(candidates, key=lambda item: item.ktc_value)
        selected.append((slot_index, slot, player))
        available.remove(player)
    return sum(player.ktc_value for _, _, player in selected), selected, missing


def replacement_values(results, slots):
    by_slot = {slot: [] for slot in set(slots)}
    for result in results:
        _, selected, _ = partial_lineup(result["_players"], slots)
        for _, slot, player in selected:
            by_slot[slot].append(player.ktc_value)
    overall = [value for values in by_slot.values() for value in values]
    fallback = float(np.mean(overall)) if overall else 0.0
    return {
        slot: float(np.mean(values)) if values else fallback
        for slot, values in by_slot.items()
    }


def weekly_lineup_value(players, slots, bye_teams, replacements):
    available = [player for player in players if normalize_nfl_team(player.team) not in bye_teams]
    score, _, missing = partial_lineup(available, slots)
    return score + sum(replacements[slot] for slot in missing)


def fetch_nfl_byes(season, regular_season_weeks, *, session=requests):
    all_teams = set(NFL_TEAM_CODES.values())
    byes = {week: set() for week in range(1, regular_season_weeks + 1)}
    for week in range(1, regular_season_weeks + 1):
        response = session.get(
            NFL_SCOREBOARD,
            params={"dates": season, "seasontype": 2, "week": week, "limit": 100},
            timeout=30,
        )
        response.raise_for_status()
        playing = {
            normalize_nfl_team(team["team"]["abbreviation"])
            for event in response.json().get("events", [])
            for team in event["competitions"][0]["competitors"]
        }
        if not playing:
            raise ValueError(f"NFL schedule returned no games for regular-season week {week}")
        absent = all_teams - playing
        if 4 <= week <= 14:
            byes[week] = absent
    return byes


def fetch_league_schedule(league_id, regular_season_weeks):
    schedule = {}
    for week in range(1, regular_season_weeks + 1):
        groups = {}
        for entry in api_get(f"/league/{league_id}/matchups/{week}"):
            if entry.get("matchup_id") is not None:
                groups.setdefault(int(entry["matchup_id"]), []).append(int(entry["roster_id"]))
        schedule[week] = [tuple(rosters) for rosters in groups.values() if len(rosters) == 2]
        if not schedule[week]:
            raise ValueError(f"Sleeper returned no head-to-head schedule for week {week}")
    return schedule


def simulate_projected_standings(
    *, results, roster_positions, schedule, bye_teams, playoff_teams,
    league_median=False, simulations=100000, seed=2026, weekly_variance=0.18,
):
    slots = simulation_slots(roster_positions)
    replacements = replacement_values(results, slots)
    roster_ids = [result["roster_id"] for result in results]
    index = {roster_id: item for item, roster_id in enumerate(roster_ids)}
    strengths = {
        week: np.array([
            weekly_lineup_value(result["_players"], slots, bye_teams.get(week, set()), replacements)
            for result in results
        ])
        for week in schedule
    }
    rng = np.random.default_rng(seed)
    wins = np.zeros((simulations, len(results)), dtype=np.float32)
    points = np.zeros_like(wins)
    for week, matchups in schedule.items():
        means = strengths[week]
        scores = np.maximum(0, rng.normal(means, np.maximum(means * weekly_variance, 1), (simulations, len(results))))
        points += scores
        for first, second in matchups:
            left, right = index[first], index[second]
            wins[:, left] += scores[:, left] > scores[:, right]
            wins[:, right] += scores[:, right] > scores[:, left]
            ties = scores[:, left] == scores[:, right]
            wins[ties, left] += 0.5
            wins[ties, right] += 0.5
        if league_median:
            median = np.median(scores, axis=1)
            wins += (scores > median[:, None]).astype(np.float32)
            wins += 0.5 * (scores == median[:, None])
    tie_break = points / np.maximum(points.max(axis=1, keepdims=True), 1) * 0.001
    order = np.argsort(-(wins + tie_break), axis=1)
    made_playoffs = np.zeros_like(wins)
    rows = np.arange(simulations)[:, None]
    made_playoffs[rows, order[:, :min(playoff_teams, len(results))]] = 1
    frame = pd.DataFrame({
        "Team": [result["team"] for result in results],
        "Projected Wins": wins.mean(axis=0),
        "Playoff Probability": made_playoffs.mean(axis=0) * 100,
    }).sort_values(["Projected Wins", "Playoff Probability"], ascending=False).reset_index(drop=True)
    frame.insert(0, "Rank", range(1, len(frame) + 1))
    frame.attrs["simulations"] = simulations
    return frame


def rank_radar_values(results, positions=RADAR_POSITIONS):
    team_count = len(results)
    for item in results:
        item["team_count"] = team_count
        item["radar_positions"] = tuple(positions)
    for position in positions:
        values = pd.Series([item["position_totals"].get(position, 0) for item in results])
        if values.max() == 0:
            for item in results:
                item.setdefault("position_ranks", {})[position] = None
                item.setdefault("radar", {})[position] = 0
            continue
        ranks = values.rank(method="min", ascending=False).astype(int)
        for item, rank in zip(results, ranks):
            item.setdefault("position_ranks", {})[position] = int(rank)
            item.setdefault("radar", {})[position] = team_count + 1 - int(rank)


def render_radar(result, path):
    positions = result.get("radar_positions", RADAR_POSITIONS)
    labels = [
        f'{position}\n#{result["position_ranks"][position]}'
        if result["position_ranks"][position] is not None
        else f"{position}\nN/A"
        for position in positions
    ]
    values = [result["radar"][position] for position in positions]
    team_count = result["team_count"]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]
    figure, axis = plt.subplots(figsize=(3.25, 3.25), subplot_kw={"polar": True})
    axis.plot(angles, values, color="#2563eb", linewidth=2)
    axis.fill(angles, values, color="#60a5fa", alpha=0.35)
    axis.set_xticks(angles[:-1], labels)
    axis.set_ylim(0, team_count)
    axis.set_yticklabels([])
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=120, bbox_inches="tight", transparent=True)
    plt.close(figure)


def pick_summary(item):
    if item is None:
        return "N/A"
    pick_no, player, difference = item
    return f"{player.name} — pick {pick_no}, ADP {player.adp:.1f} ({difference:+.1f})"


def position_rank_summary(result):
    return ", ".join(
        f"{position} #{rank}" if rank is not None else f"{position} N/A"
        for position, rank in result.get("position_ranks", {}).items()
    ) or "N/A"


def response_markdown_with_citations(response, footnote_prefix="source"):
    for item in getattr(response, "output", []):
        if getattr(item, "type", None) != "message":
            continue
        for content in getattr(item, "content", []):
            if getattr(content, "type", None) != "output_text":
                continue
            text = content.text
            citations = []
            for annotation in getattr(content, "annotations", []):
                if getattr(annotation, "type", None) != "url_citation":
                    continue
                citations.append((
                    annotation.start_index,
                    annotation.end_index,
                    annotation.title or "source",
                    annotation.url,
                ))
            safe_prefix = re.sub(r"[^a-zA-Z0-9-]+", "-", footnote_prefix).strip("-") or "source"
            source_numbers = {}
            sources = []
            numbered_citations = []
            for start, end, title, url in sorted(citations):
                if url not in source_numbers:
                    source_numbers[url] = len(sources) + 1
                    sources.append((title, url))
                numbered_citations.append((start, end, source_numbers[url]))
            for start, end, source_number in sorted(numbered_citations, reverse=True):
                text = f"{text[:start]}[^{safe_prefix}-{source_number}]{text[end:]}"
            if sources:
                footnotes = [
                    f"[^{safe_prefix}-{number}]: [{title}]({url})"
                    for number, (title, url) in enumerate(sources, 1)
                ]
                text = f"{text.strip()}\n\n" + "\n".join(footnotes)
            return text.strip()
    return response.output_text.strip()


def commentary_tone(overall_rank, league_size):
    if league_size <= 1:
        return "balanced: give comparable weight to the roster's strongest feature and biggest concern"
    rank_percentile = (overall_rank - 1) / (league_size - 1)
    if rank_percentile <= 0.25:
        return "strongly positive: emphasize why this is an elite roster, while naming one credible concern"
    if rank_percentile <= 0.50:
        return "positive-leaning: emphasize the strengths, but explain at least one meaningful weakness"
    if rank_percentile <= 0.75:
        return "critical-leaning: emphasize the weaknesses, but identify at least one legitimate strength"
    return "strongly critical: emphasize why this roster trails the league, while acknowledging one real strength"


def generate_ai_commentary(
    *, league, results, api_key, model, client=None, workers=4,
    reasoning_effort=DEFAULT_AI_REASONING_EFFORT, full_overview=False,
):
    if client is None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required to generate roster analysis")
        client = OpenAI(api_key=api_key)
    shared_instructions = (
        "Write as an entertaining fantasy-football fan reacting to a roster after the draft. Use an informal, "
        "conversational voice—not a formal analyst voice or scouting report. Be playful, opinionated, and willing "
        "to exaggerate for entertainment: hype up strong teams and roast weak teams, while keeping factual claims "
        "grounded. Do not recite the overall league rank because the report already displays it. Research the "
        "roster's players using current, reputable "
        "fantasy football sources such as ESPN, FantasyPros, CBS Sports, and official NFL or team coverage. In "
        "4-6 punchy sentences, explain where the roster construction thrives, where it falls short, and which "
        "players or position groups drive that verdict. Implicitly account for the supplied league format and "
        "scoring rules; best ball roster construction and risk tolerance differ from a managed-lineup leagues. "
        "Use overall rank as the source of truth for the analysis's sentiment and follow the supplied editorial "
        "tone: higher-ranked teams should read more positively and lower-ranked teams more critically. Every "
        "team must still receive at least one genuine strength and one genuine concern. Use positional ranks, "
        "biggest value, and biggest reach as evidence rather than merely repeating them. The "
        "biggest value and biggest reach are calculated from FantasyPros ADP adjusted for this league's team "
        "count, PPR scoring, and superflex/2-QB roster format; discuss them as ADP values, never as VOR. Be "
        "engaging, colorful, and bombastic—celebrate sharp drafting like a league-winning heist and roast bad "
        "decisions like draft-night disasters. Distinguish sourced facts from your analysis, never invent facts, and "
        "cite web-derived claims; the application will format those citations as footnotes at the end of the team summary. "
    )
    instructions = shared_instructions + (
        "Write 4-6 punchy sentences of prose without a heading or bullet list."
        if full_overview else
        "Return exactly 3-5 concise Markdown bullet points covering the most important positive or negative aspects "
        "of this roster's construction. Do not add a heading or introductory sentence."
    )
    def generate_for_team(result):
        statistics = {
            "league": league["name"],
            "league_context": league_context(league),
            "team": result["team"],
            "overall_rank": result["rank"],
            "league_size": result["team_count"],
            "editorial_tone": commentary_tone(result["rank"], result["team_count"]),
            "ranking_source": {
                "provider": "KeepTradeCut fantasy rankings",
                **result.get("ktc_settings", {}),
            },
            "adp_settings": result.get("adp_settings", {}),
            "roster_construction": result.get("roster_construction", {}),
            "roster": result.get("roster", []),
            "position_ranks": {
                position: rank
                for position, rank in result["position_ranks"].items()
                if rank is not None
            },
            "biggest_reach": pick_summary(result["reach"]),
            "biggest_value": pick_summary(result["value"]),
        }
        response = client.responses.create(
            model=model,
            instructions=instructions,
            input=json.dumps(statistics),
            store=False,
            text={"verbosity": "low"},
            tools=[{"type": "web_search"}],
            tool_choice="auto",
            reasoning={"effort": reasoning_effort},
        )
        commentary = response_markdown_with_citations(
            response, footnote_prefix=f"team-{result['roster_id']}"
        )
        if not commentary:
            raise ValueError(f"OpenAI returned empty commentary for {result['team']}")
        result["commentary"] = commentary
        return result["team"]

    worker_count = min(max(1, int(workers)), max(1, len(results)))
    if worker_count == 1:
        for result in results:
            print(f"Generated AI commentary for {generate_for_team(result)}")
        return
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(generate_for_team, result) for result in results]
        for future in as_completed(futures):
            print(f"Generated AI commentary for {future.result()}")

def standings_markdown(standings):
    display = standings.copy()
    display["Rank"] = display["Rank"].astype(int)
    display["Projected Wins"] = display["Projected Wins"].map(lambda value: f"{value:.1f}")
    display["Playoff Probability"] = display["Playoff Probability"].map(lambda value: f"{value:.1f}%")
    return tabulate(display, headers="keys", tablefmt="pipe", showindex=False)


def render_report(*, league, results, standings):
    sections = [
        "+++", f'title = "{league["season"]} Post-Draft Rankings"', f'date = "{date.today()}"',
        "draft = false", "+++", "", f"# {league['name']} Post-Draft Rankings", "",
    ]
    for result in results:
        image_name = f"team-{result['roster_id']}-radar.png"
        sections.extend([
            f"## #{result['rank']} {result['team']}", "",
            f"![{result['team']} positional strength radar chart]({image_name})", "",
        ])
        if result.get("full_overview"):
            sections.extend([
                f"**Biggest Reach:** {pick_summary(result['reach'])}", "",
                f"**Biggest Value:** {pick_summary(result['value'])}", "",
                result["commentary"], "",
            ])
        else:
            concerns = result.get("availability_concerns") or []
            concern_summary = "; ".join(concerns) if concerns else "None currently flagged by Sleeper"
            sections.extend([
                "### Roster analysis", "",
                f"- **Position-group rankings:** {position_rank_summary(result)}", "",
                f"- **Biggest Reach:** {pick_summary(result['reach'])}", "",
                f"- **Biggest Value:** {pick_summary(result['value'])}", "",
                f"- **Injury/suspension monitor:** {concern_summary}", "",
                result["commentary"], "",
            ])
    sections.extend([
        "## Projected Standings", "",
        f"Based on {int(standings.attrs.get('simulations', 100000)):,} schedule simulations using bye-adjusted "
        "KTC starting-lineup value; K and D/ST are excluded.", "",
        standings_markdown(standings), "",
    ])
    return "\n".join(sections)


def render_report_html(*, markdown_content, league, output_path):
    import markdown
    from sleeper_rankings.render import CSS

    body = re.sub(r"^\+\+\+\n.*?\n\+\+\+\n", "", markdown_content, count=1, flags=re.DOTALL)
    report_html = markdown.markdown(body, extensions=["footnotes", "tables"])
    report_html = report_html.replace("<table>", '<table class="rankings-table">')
    title = f"{league['season']} Post-Draft Rankings"
    page = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="Post-draft fantasy football rankings for {league['name']}">
<title>{title}</title><link rel="stylesheet" href="assets/site.css"></head>
<body><header class="hero"><div class="wrap"><a class="report-home" href="/">← Home</a><p class="eyebrow">{league['season']} · Draft report</p><h1>{title}</h1><p>{league['name']} · Preseason roster analysis</p></div></header>
    <main class="wrap report-prose">{report_html}</main></body></html>'''
    write_report_atomic(output_path, page)
    assets = output_path.parent / "assets"
    assets.mkdir(exist_ok=True)
    write_report_atomic(assets / "site.css", CSS + DRAFT_REPORT_CSS)


DRAFT_REPORT_CSS = """
.report-home{display:inline-block;margin-bottom:28px;color:#fff;text-decoration:none;font-weight:800}.report-home:hover,.report-home:focus{text-decoration:underline;text-underline-offset:4px}
.report-prose{padding:38px 0 72px}.report-prose>h1:first-of-type{display:none}
.report-prose>h1:not(:first-of-type),.report-prose>h2{margin-top:50px;border-top:1px solid var(--line);padding-top:32px}
.report-prose .rankings-table th:not(:nth-child(2)),.report-prose .rankings-table td:not(:nth-child(2)){text-align:center!important}
.report-prose img{width:min(330px,100%);height:auto;display:block;margin:18px auto}
.report-prose .footnote{font-size:.82rem;color:var(--muted)}
"""


def parse_args(argv=None):
    load_dotenv(LOCAL_ENV_FILE)
    parser = argparse.ArgumentParser(description="Generate a Sleeper post-draft rankings report.")
    parser.add_argument("league_id", help="Sleeper league ID")
    parser.add_argument("--output", type=Path, help="Exact output directory (default: reports/YEAR/LEAGUE[_ai])")
    parser.add_argument("--dummy-draft", type=Path, help="Use draft picks from a dummy draft JSON file")
    parser.add_argument("--ai-commentary", action="store_true", help="Use full AI overviews instead of concise AI bullets")
    parser.add_argument("--ai-model", default=os.getenv("OPENAI_MODEL", DEFAULT_AI_MODEL), help="OpenAI model used for commentary")
    parser.add_argument(
        "--ai-reasoning-effort",
        choices=("none", "low", "medium", "high", "xhigh", "max"),
        default=os.getenv("OPENAI_REASONING_EFFORT", DEFAULT_AI_REASONING_EFFORT),
        help="OpenAI reasoning effort used for commentary",
    )
    parser.add_argument("--ai-workers", type=int, default=int(os.getenv("OPENAI_AI_WORKERS", "4")), help="Concurrent AI commentary requests")
    parser.add_argument("--simulations", type=int, default=100000, help="Projected-standings simulations")
    parser.add_argument("--simulation-seed", type=int, default=2026, help="Projected-standings random seed")
    return parser.parse_args(argv)


def safe_directory_name(value):
    cleaned = re.sub(r"[\\/:*?\"<>|]+", "-", str(value)).strip(" .")
    return cleaned or "league"


def report_directory_name(league_name, full_overview):
    return safe_directory_name(league_name) + ("_ai" if full_overview else "")


def run(args):
    league = api_get(f"/league/{args.league_id}")
    dummy = None
    if args.dummy_draft:
        dummy = json.loads(args.dummy_draft.read_text(encoding="utf-8"))
        if str(dummy.get("league_id")) != str(args.league_id):
            raise ValueError("Dummy draft league ID does not match the requested Sleeper league")
        picks = dummy.get("picks") or []
    else:
        drafts = api_get(f"/league/{args.league_id}/drafts")
        complete = [draft for draft in drafts if draft.get("status") == "complete"]
        if not complete:
            raise ValueError("Sleeper league has no completed draft; use --dummy-draft for a preview")
        draft = max(complete, key=lambda item: item.get("last_picked") or item.get("start_time") or 0)
        picks = api_get(f"/draft/{draft['draft_id']}/picks")
    rosters = api_get(f"/league/{args.league_id}/rosters")
    users = {str(user["user_id"]): user for user in api_get(f"/league/{args.league_id}/users")}
    player_catalog = None if args.dummy_draft else api_get("/players/nfl")
    season = int(league["season"])
    ktc_values, ktc_settings = fetch_ktc_rankings(league)
    base = ktc_player_frame(ktc_values)
    if dummy is not None and not picks:
        picks = generate_dummy_picks(
            base, teams=int(dummy.get("teams", len(rosters))),
            rounds=int(dummy.get("rounds", len(league["roster_positions"]))),
            seed=int(dummy.get("seed", 2026)),
        )
    adp, adp_settings = fetch_fantasypros_adp(league, len(rosters))
    projections = projection_index(add_adp_to_projections(base, adp))
    results, unmatched = build_team_results(
        league=league, rosters=rosters, users=users, picks=picks, projections=projections,
        player_catalog=player_catalog,
    )
    rank_radar_values(results, radar_positions_for_league(league["roster_positions"]))
    for result in results:
        result["adp_settings"] = adp_settings
        result["ktc_settings"] = ktc_settings
    settings = league.get("settings") or {}
    regular_season_weeks = int(settings.get("playoff_week_start", 15)) - 1
    schedule = fetch_league_schedule(args.league_id, regular_season_weeks)
    byes = fetch_nfl_byes(season, regular_season_weeks)
    standings = simulate_projected_standings(
        results=results, roster_positions=league["roster_positions"], schedule=schedule,
        bye_teams=byes, playoff_teams=int(settings.get("playoff_teams", 6)),
        league_median=bool(int(settings.get("league_average_match", 0) or 0)),
        simulations=args.simulations, seed=args.simulation_seed,
    )
    generate_ai_commentary(
        league=league, results=results, api_key=os.getenv("OPENAI_API_KEY"), model=args.ai_model,
        workers=getattr(args, "ai_workers", 4), full_overview=args.ai_commentary,
        reasoning_effort=getattr(args, "ai_reasoning_effort", DEFAULT_AI_REASONING_EFFORT),
    )
    for result in results:
        result["full_overview"] = args.ai_commentary
    report_name = report_directory_name(league["name"], args.ai_commentary)
    output_dir = (
        args.output or Path(__file__).resolve().parent / "reports" / str(season) / report_name
    ).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        render_radar(result, output_dir / f"team-{result['roster_id']}-radar.png")
    markdown_content = render_report(league=league, results=results, standings=standings)
    write_report_atomic(output_dir / "index.md", markdown_content)
    render_report_html(
        markdown_content=markdown_content, league=league, output_path=output_dir / "index.html",
    )
    draft_rankings = [
        {"rank": int(result["rank"]), "roster_id": int(result["roster_id"]), "Team": result["team"]}
        for result in sorted(results, key=lambda item: int(item["rank"]))
    ]
    write_report_atomic(output_dir / "rankings.json", json.dumps(draft_rankings, indent=2) + "\n")
    if unmatched:
        print(f"Warning: {len(unmatched)} drafted player(s) had no KTC value: {', '.join(unmatched)}", file=sys.stderr)
    print(f"Report written to {output_dir / 'index.md'}")
    return output_dir / "index.md"


def main(argv=None):
    try:
        run(parse_args(argv))
    except (OSError, subprocess.CalledProcessError, requests.RequestException, ValueError, OpenAIError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
