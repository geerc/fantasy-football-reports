"""Persist reviewed weekly output; rebuilding never refetches historical data."""
import json
import shutil
from html import escape
from pathlib import Path

import pandas as pd

from .rankings import add_weekly_change
from .render import CSS


def entry_path(content, league_id, season, week):
    if not str(league_id).isdigit() or not str(season).isdigit() or not 1 <= week <= 18:
        raise ValueError("Invalid league, season, or week")
    return content / str(league_id) / str(season) / f"week-{week:02d}"


def with_previous(current, content, league_id, season, week):
    if week == 1:
        previous = (
            content.parent
            / "draft-reports"
            / str(league_id)
            / str(season)
            / "post-draft"
            / "rankings.json"
        )
    else:
        previous = entry_path(content, league_id, season, week - 1) / "rankings.json"
    if previous.exists():
        frame = pd.DataFrame(json.loads(previous.read_text()))
        if "rank" in frame.columns:
            frame = frame.sort_values("rank")
        frame.index = range(1, len(frame) + 1)
        return add_weekly_change(current, frame)
    result = current.copy()
    result["Weekly Change"] = "No draft snapshot" if week == 1 else "No prior snapshot"
    return result


def build_archive(content: Path, output: Path, config: dict):
    output.mkdir(parents=True, exist_ok=True)
    weekly_entries = []
    for metadata in sorted(content.glob("*/*/week-*/report.json"), reverse=True):
        record = json.loads(metadata.read_text())
        relative = metadata.parent.relative_to(content)
        target = output / "reports" / relative
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(metadata.parent / "index.html", target / "index.html")
        shutil.copytree(metadata.parent / "assets", target / "assets", dirs_exist_ok=True)
        label = f'{record["season"]} · Week {record["week"]}'
        weekly_entries.append(
            f'<article class="report-card"><p class="eyebrow">Weekly power rankings</p>'
            f'<h3><a href="reports/{relative.as_posix()}/">{escape(label)}</a></h3>'
            '<p>Results, roster value, projected standings, and luck.</p></article>'
        )
    draft_entries = []
    draft_content = content.parent / "draft-reports"
    for metadata in sorted(draft_content.glob("*/*/*/report.json"), reverse=True):
        record = json.loads(metadata.read_text())
        relative = metadata.parent.relative_to(draft_content)
        target = output / "draft-reports" / relative
        shutil.copytree(metadata.parent, target, dirs_exist_ok=True)
        label = escape(record.get("title", f'{record["season"]} Post-Draft Rankings'))
        status = escape(record.get("status", "Draft"))
        draft_entries.append(
            f'<article class="report-card"><p class="eyebrow">Draft report · {status}</p>'
            f'<h3><a href="draft-reports/{relative.as_posix()}/">{label}</a></h3>'
            '<p>Projected starters, positional rankings, draft values, and roster analysis.</p></article>'
        )
    title = escape(config.get("title", "Weekly reports"))
    (output / "assets").mkdir(exist_ok=True)
    (output / "assets/site.css").write_text(CSS + ARCHIVE_CSS)
    draft_html = "".join(draft_entries) or '<p class="note">No draft reports are available yet.</p>'
    weekly_html = "".join(weekly_entries) or '<p class="note">Weekly rankings begin after Week 1.</p>'
    (output / "index.html").write_text(f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="Fantasy football reports for {title}"><title>{title}</title>
<link rel="stylesheet" href="assets/site.css"></head><body>
<header class="hero"><div class="wrap"><p class="eyebrow">League report center</p><h1>{title}</h1>
<p>Draft analysis, weekly power rankings, projected standings, and season-long league intelligence.</p></div></header>
<main class="wrap archive"><section><div class="section-title"><div><p class="eyebrow">Preseason</p><h2>Draft reports</h2></div></div>
<div class="report-grid">{draft_html}</div></section>
<section><div class="section-title"><div><p class="eyebrow">In season</p><h2>Weekly power rankings</h2></div></div>
<div class="report-grid">{weekly_html}</div></section></main></body></html>''')
    return output / "index.html"


ARCHIVE_CSS = """
.archive{padding-top:42px}.report-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:18px}
.report-card{border:1px solid var(--line);border-radius:14px;padding:22px;background:#fff;box-shadow:0 8px 24px #17231d0a}
.report-card h3{font:800 1.45rem/1.15 Georgia,serif;margin:.3rem 0 .65rem}.report-card a{color:var(--ink)}
.report-card p:last-child{color:var(--muted);margin-bottom:0}
"""
