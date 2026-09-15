import pandas as pd

from report import render_report, write_report_atomic


def test_report_rendering_handles_optional_sections():
    rankings = pd.DataFrame({"Team": ["Alpha"], "Power Score": [100], "Performance Rank": [1], "KTC Value Rank": [1]})
    standings = pd.DataFrame({"Team": ["Alpha"], "Projected Wins": [10.5], "Projected Losses": [4.5]})
    luck = pd.DataFrame({"Team": ["Alpha"], "Luck Index": [0.25]})
    content = render_report(year=2026, week=1, rankings=rankings, summary=None, playoff_probabilities=None, expected_standings=standings, luck_index=luck)

    assert "Week 1 2026 Report" in content
    assert "Current Playoff Probabilities" not in content
    assert "Projected Standings" in content
    assert "cover" not in content
    assert ".jpeg" not in content
    assert "hideSummary = true" in content
    assert "hideMeta = true" in content
    separators = [line for line in content.splitlines() if line.startswith("|") and "---" in line]
    ranking_alignments = [cell.strip() for cell in separators[0].strip("|").split("|")]
    standings_alignments = [cell.strip() for cell in separators[1].strip("|").split("|")]
    luck_alignments = [cell.strip() for cell in separators[2].strip("|").split("|")]
    assert all(value.startswith(":") and value.endswith(":") for value in ranking_alignments[3:5])
    assert all(value.startswith(":") and value.endswith(":") for value in standings_alignments[2:4])
    assert luck_alignments[2].startswith(":") and luck_alignments[2].endswith(":")


def test_atomic_report_write_creates_parent_directory(tmp_path):
    destination = tmp_path / "2026Week1" / "index.md"

    write_report_atomic(destination, "new report")

    assert destination.read_text() == "new report"
    assert list(destination.parent.glob(".index.md.*")) == []
