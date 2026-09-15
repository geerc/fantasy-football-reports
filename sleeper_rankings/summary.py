from __future__ import annotations

import json
import os

from openai import OpenAI

from .models import LeagueData


def matchup_payload(data: LeagueData, week: int) -> list[dict]:
    by_roster = data.by_roster
    groups: dict[int, list[dict]] = {}
    for row in data.matchups.get(week, []):
        if row.get("matchup_id") is not None:
            groups.setdefault(int(row["matchup_id"]), []).append(row)
    payload = []
    for rows in groups.values():
        if len(rows) != 2:
            continue
        matchup = []
        for row in rows:
            player_points = row.get("players_points") or {}
            starters = set(row.get("starters") or [])
            players = []
            for player_id, points in player_points.items():
                player = data.players.get(str(player_id)) or {}
                players.append({
                    "name": player.get("full_name") or player_id,
                    "position": player.get("position"),
                    "points": points,
                    "starter": player_id in starters,
                })
            matchup.append({
                "team": by_roster[int(row["roster_id"])].name,
                "score": row.get("custom_points") if row.get("custom_points") is not None else row.get("points"),
                "players": players,
            })
        payload.append({"teams": matchup})
    return payload


def generate_summary(data: LeagueData, week: int, model: str) -> str | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    response = OpenAI(api_key=api_key).responses.create(
        model=model,
        instructions=(
            "Write a lively, concise weekly recap for the SYPIP fantasy football league using only the supplied "
            "matchup data. Use 300–450 words and this structure: an opening paragraph identifying the week's main "
            "storyline; a 'Game of the Week' section for the closest matchup; a 'Statement Win' section for the "
            "largest margin; a 'Weekly Heroes' section highlighting 3–5 exceptional starters; a 'Bench Regrets' "
            "section mentioning only bench players who outscored a starter at the same position and stating both "
            "scores; and a short closing line looking ahead without inventing predictions. Use exact team names, "
            "player names, and scores. Be playful and lightly teasing, but never insulting. Vary the language and "
            "avoid clichés such as 'thrilling week' or describing every game as dramatic. Sleeper projections, "
            "injuries, transactions, standings implications, and information outside the supplied data are "
            "unavailable; do not infer or invent them. Use Markdown section headings, but do not add a top-level title."
        ),
        input=json.dumps(matchup_payload(data, week)),
    )
    return response.output_text
