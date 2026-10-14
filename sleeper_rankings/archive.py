"""Persist reviewed weekly output; rebuilding never refetches historical data."""
import json
import shutil
from html import escape
from pathlib import Path

import pandas as pd

from .rankings import add_weekly_change
from .render import CSS, render_preseason, refresh_saved_presentation


def entry_path(content, league_id, season, week):
    if not str(league_id).isdigit() or not str(season).isdigit() or not 1 <= week <= 18:
        raise ValueError("Invalid league, season, or week")
    return content / str(league_id) / str(season) / f"week-{week:02d}"


def with_previous(current, content, league_id, season, week):
    if week > 1:
        previous = entry_path(content, league_id, season, week - 1) / "rankings.json"
        if previous.exists():
            frame = pd.DataFrame(json.loads(previous.read_text()))
            frame.index = range(1, len(frame) + 1)
            return add_weekly_change(current, frame)
    result = current.copy()
    result["Weekly Change"] = "—" if week == 1 else "No prior snapshot"
    return result


def build_archive(content: Path, output: Path, config: dict):
    output.mkdir(parents=True, exist_ok=True)
    entries = []
    for metadata in sorted(content.glob("*/*/week-*/report.json"), reverse=True):
        record = json.loads(metadata.read_text())
        relative = metadata.parent.relative_to(content)
        target = output / "reports" / relative
        target.mkdir(parents=True, exist_ok=True)
        (target / "index.html").write_text(refresh_saved_presentation((metadata.parent / "index.html").read_text()))
        shutil.copytree(metadata.parent / "assets", target / "assets", dirs_exist_ok=True)
        (target / "assets/site.css").write_text(CSS)
        label = f'{record["season"]} · Week {record["week"]}'
        entries.append(f'<li><a href="reports/{relative.as_posix()}/"><span>{escape(label)}</span><span class="arrow" aria-hidden="true">→</span></a></li>')
    if not entries:
        return render_preseason(output=output, title=config.get("title", ""), league_name="", season="")
    title = escape(config.get("title", "Weekly reports"))
    (output / "assets").mkdir(exist_ok=True)
    (output / "assets/site.css").write_text(CSS)
    (output / "index.html").write_text(f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title><link rel="stylesheet" href="assets/site.css"></head><body><header class="hero"><div class="wrap"><p class="eyebrow">The archive</p><h1>{title}</h1><p>Every week. Every ranking.</p></div></header><main class="wrap"><section><h2>Weekly reports</h2><p class="archive-intro">Choose a week to explore the rankings and recap.</p><ul class="report-list">{"".join(entries)}</ul></section></main></body></html>')
    return output / "index.html"
