from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from .api import SleeperClient
from .archive import build_archive, entry_path, with_previous
from .loader import completed_week, load_league
from .rankings import add_weekly_change, luck_index, playoff_probabilities, power_rankings, projected_standings
from .render import render_preseason, render_site
from .summary import generate_summary
from .values import weekly_values


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Generate a Sleeper power rankings site")
    result.add_argument("--config", type=Path, default=Path("leagues/sypip.json"))
    result.add_argument("--league-id")
    result.add_argument("--week", type=int)
    result.add_argument("--output", type=Path, default=Path("dist"))
    result.add_argument("--skip-ai", action="store_true")
    result.add_argument("--overwrite", action="store_true", help="Manually replace an existing week; requires --week")
    result.add_argument("--content", type=Path, default=Path("content/reports"))
    result.add_argument("--archive-only", action="store_true", help="Render saved entries without generating new reports")
    return result


def run(args: argparse.Namespace) -> Path:
    if args.overwrite and (args.week is None or os.getenv("GITHUB_EVENT_NAME") == "schedule"):
        raise ValueError("--overwrite requires an explicit --week and cannot run on a schedule")
    load_dotenv()
    config = json.loads(args.config.read_text(encoding="utf-8")) if args.config.exists() else {}
    if args.archive_only:
        return build_archive(args.content, args.output, config)
    league_id = args.league_id or os.getenv("SLEEPER_LEAGUE_ID") or config.get("league_id")
    if not league_id:
        raise ValueError("A Sleeper league ID is required")
    client = SleeperClient()
    league = client.league(str(league_id))
    week = args.week if args.week is not None else completed_week(league)
    if week < 1:
        return render_preseason(
            output=args.output,
            title=config.get("title", f'{league["name"]} Power Rankings'),
            league_name=league["name"],
            season=league["season"],
        )
    destination = entry_path(args.content, league_id, league["season"], week)
    if destination.exists() and not args.overwrite:
        raise ValueError(f"Report already exists: {destination}; refusing to overwrite it")
    data = load_league(client, str(league_id), through_week=week)
    if not any(team.players for team in data.teams):
        raise ValueError("The league has no populated rosters yet")
    snapshot_path = destination / "values.csv"
    values = weekly_values(snapshot_path, overwrite=args.overwrite, dynasty_weight=float(config.get("dynasty_weight", 0)))
    rankings = power_rankings(data, week, values)
    rankings = with_previous(rankings, args.content, league_id, league["season"], week)
    standings = projected_standings(data, rankings, week)
    luck = luck_index(data, week)
    playoffs = None
    regular_weeks = int(data.league["settings"].get("playoff_week_start", 15)) - 1
    if 5 <= week < regular_weeks:
        playoffs = playoff_probabilities(data, rankings, week, int(config.get("simulations", 100000)), config.get("random_seed"))
    ai_enabled = bool(config.get("ai_recap", False)) and not args.skip_ai
    summary = generate_summary(data, week, os.getenv("OPENAI_MODEL", "gpt-5-mini")) if ai_enabled else None
    header_image = Path(config.get("header_image", "")) if config.get("header_image") else None
    render_site(output=destination, title=config.get("title", f'{data.league["name"]} Power Rankings'), league_name=data.league["name"], season=data.league["season"], week=week, rankings=rankings, summary=summary, playoffs=playoffs, standings=standings, luck=luck, header_image=header_image)
    (destination / "rankings.json").write_text(rankings.to_json(orient="records"))
    if not snapshot_path.exists():
        values.to_csv(snapshot_path, index=False)
    (destination / "report.json").write_text(json.dumps({"league_id": str(league_id), "season": league["season"], "week": week}))
    return build_archive(args.content, args.output, config)


def main(argv=None) -> int:
    try:
        path = run(parser().parse_args(argv))
    except (OSError, ValueError) as error:
        print(f"Error: {error}")
        return 2
    print(f"Site written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
