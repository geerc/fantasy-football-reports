import os
import tempfile
from datetime import date

from tabulate import tabulate


def _table(frame, centered_columns=()):
    alignments = ("right",) + tuple("center" if column in centered_columns else None for column in frame.columns)
    return tabulate(frame, headers="keys", tablefmt="pipe", showindex=True, colalign=alignments)


def render_report(*, year, week, rankings, summary, playoff_probabilities, expected_standings, luck_index):
    sections = ["+++", f'title = "Week {week} {year} Report"', f'date = "{date.today()}"', "draft = false", "hideSummary = true", "hideMeta = true", "+++", "", "# POWER RANKINGS", "", _table(rankings, {"Performance Rank", "KTC Value Rank"})]
    if summary:
        sections.extend(["", summary])
    if playoff_probabilities is not None:
        sections.extend(["", "## Current Playoff Probabilities", "", tabulate(playoff_probabilities, headers="keys", tablefmt="pipe", showindex=True)])
    sections.extend(["", f"## Projected Standings (as of week {week})", "", _table(expected_standings, {"Projected Wins", "Projected Losses"}), "", "## LUCK INDEX", "", _table(luck_index, {"Luck Index"}), ""])
    return "\n".join(sections)


def write_report_atomic(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(content)
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
