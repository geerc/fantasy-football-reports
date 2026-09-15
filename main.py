import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from espn_api.football import League

from config import AppConfig
from doritostats import luck_index
from rankings import add_weekly_change, generate_expected_standings, generate_playoff_probabilities, generate_power_rankings
from report import render_report, write_report_atomic
from summary import generate_ai_summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Generate the weekly ESPN fantasy football report.")
    parser.add_argument("--year", type=int, help="ESPN season year (defaults to the current year)")
    parser.add_argument("--week", type=int, help="Report week (defaults to the last completed ESPN week)")
    parser.add_argument("--simulations", type=int, default=100_000, help="Number of playoff simulations")
    parser.add_argument("--seed", type=int, help="Random seed for reproducible simulations")
    parser.add_argument("--skip-ai", action="store_true", help="Generate the report without an AI recap")
    parser.add_argument("--overwrite", action="store_true", help="Manually replace an existing week; requires --week")
    parser.add_argument("--output", type=Path, help="Override the output Markdown path")
    return parser.parse_args(argv)


def league_setting(league, names, default):
    settings = getattr(league, "settings", None)
    for name in names:
        value = getattr(settings, name, None)
        if isinstance(value, int) and value > 0:
            return value
    return default


def generate_luck_index(league, week):
    rows = []
    for team in league.teams:
        value = sum(luck_index.get_weekly_luck_index(league, team, item) for item in range(1, week + 1))
        rows.append([team.team_name, round(value, 2)])
    result = pd.DataFrame(rows, columns=["Team", "Luck Index"]).sort_values("Luck Index", ascending=False)
    result.index = range(1, len(result) + 1)
    return result


def run(args):
    overwrite = getattr(args, "overwrite", False)
    if overwrite and (args.week is None or os.getenv("GITHUB_EVENT_NAME") == "schedule"):
        raise ValueError("--overwrite requires an explicit --week and cannot run on a schedule")
    if args.simulations < 1:
        raise ValueError("--simulations must be greater than zero")
    config = AppConfig.from_env(year=args.year, week=args.week, simulations=args.simulations, random_seed=args.seed)
    league = League(config.league_id, config.year, config.espn_s2, config.swid)
    week = config.week if config.week is not None else league.nfl_week - 1
    if week < 1:
        raise ValueError("No completed fantasy week is available; pass --week explicitly if appropriate")
    output_path = args.output.expanduser().resolve() if args.output else config.report_root / f"{config.year}Week{week}" / "index.md"
    if output_path.exists() and not overwrite:
        raise ValueError(f"Report already exists: {output_path}; use --week and --overwrite for a manual correction")

    regular_season_weeks = league_setting(league, ["reg_season_count", "regular_season_matchup_period_count"], 15)
    playoff_teams = league_setting(league, ["playoff_team_count"], max(1, len(league.teams) // 2))
    values_path = config.player_values_dir / f"KTC_values_week{week}.csv"
    print(f"Generating {config.year} week {week} report for {league}...")

    rankings, _ = generate_power_rankings(league, week, values_path)
    display_rankings = rankings
    if week > 1:
        previous_path = config.player_values_dir / f"KTC_values_week{week - 1}.csv"
        if previous_path.exists():
            previous_rankings, _ = generate_power_rankings(league, week - 1, previous_path)
            display_rankings = add_weekly_change(rankings, previous_rankings)
        else:
            print(f"Warning: {previous_path.name} is missing; weekly change will be omitted.")

    expected = generate_expected_standings(league, rankings, week, regular_season_weeks)
    playoffs = None
    if 6 <= week < regular_season_weeks:
        playoffs = generate_playoff_probabilities(league, rankings, week, regular_season_weeks, playoff_teams, config.simulations, config.random_seed)
    luck = generate_luck_index(league, week)

    summary = None
    if not args.skip_ai:
        try:
            summary = generate_ai_summary(league, week, config.names_file, values_path, config.player_values_dir / f"KTC_values_week{week - 1}.csv" if week > 1 else None, config.openai_api_key, config.ai_model)
        except Exception as error:
            print(f"Warning: AI summary failed and will be omitted: {error}", file=sys.stderr)

    output_path = args.output.expanduser().resolve() if args.output else config.report_root / f"{config.year}Week{week}" / "index.md"
    content = render_report(year=config.year, week=week, rankings=display_rankings, summary=summary, playoff_probabilities=playoffs, expected_standings=expected, luck_index=luck)
    write_report_atomic(output_path, content)
    print(f"Report written to {output_path}")
    return output_path


def main(argv=None):
    try:
        run(parse_args(argv))
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Cancelled.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
