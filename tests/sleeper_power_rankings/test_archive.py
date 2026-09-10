import json

import pandas as pd
import pytest

from sleeper_rankings.archive import build_archive, entry_path, with_previous
from sleeper_rankings.render import render_site


def test_saved_rank_not_recomputed_and_missing_history(tmp_path):
    current = pd.DataFrame([{"roster_id": 2, "Team": "Renamed", "Power Score": 90}, {"roster_id": 1, "Team": "A", "Power Score": 80}], index=[1, 2])
    assert with_previous(current, tmp_path, "123", "2026", 2)["Weekly Change"].tolist() == ["No prior snapshot"] * 2
    path = entry_path(tmp_path, "123", "2026", 1)
    path.mkdir(parents=True)
    (path / "rankings.json").write_text(json.dumps([{"roster_id": 1}, {"roster_id": 2}]))
    assert with_previous(current, tmp_path, "123", "2026", 2)["Weekly Change"].tolist() == ["↑ 1", "↓ 1"]


def test_week_one_uses_post_draft_ranking_snapshot(tmp_path):
    content = tmp_path / "content" / "reports"
    current = pd.DataFrame(
        [{"roster_id": 2, "Team": "Renamed", "Power Score": 90}, {"roster_id": 1, "Team": "A", "Power Score": 80}],
        index=[1, 2],
    )
    draft = content.parent / "draft-reports" / "123" / "2026" / "post-draft"
    draft.mkdir(parents=True)
    (draft / "rankings.json").write_text(json.dumps([
        {"rank": 1, "roster_id": 1, "Team": "A"},
        {"rank": 2, "roster_id": 2, "Team": "Old name"},
    ]))

    assert with_previous(current, content, "123", "2026", 1)["Weekly Change"].tolist() == ["↑ 1", "↓ 1"]


def test_week_one_labels_missing_draft_snapshot(tmp_path):
    current = pd.DataFrame([{"roster_id": 1, "Team": "A"}], index=[1])
    content = tmp_path / "content" / "reports"
    assert with_previous(current, content, "123", "2026", 1)["Weekly Change"].tolist() == ["No draft snapshot"]


def test_archive_preserves_both_weeks_without_network(tmp_path):
    content, output = tmp_path / "content", tmp_path / "dist"
    frame = pd.DataFrame([{"Team": "Alpha", "Power Score": 50}], index=[1])
    for week in [1, 2]:
        path = entry_path(content, "123", "2026", week)
        render_site(output=path, title="Test", league_name="Test", season="2026", week=week, rankings=frame, summary=None, playoffs=None, standings=frame, luck=frame)
        (path / "report.json").write_text(json.dumps({"season": "2026", "week": week}))
    index = build_archive(content, output, {"title": "Test"})
    assert "Week 1" in index.read_text() and "Week 2" in index.read_text()
    assert (output / "reports/123/2026/week-01/index.html").exists()
    assert (output / "reports/123/2026/week-02/assets/site.css").exists()


def test_archive_homepage_includes_draft_report_and_empty_weekly_state(tmp_path):
    content, output = tmp_path / "content" / "reports", tmp_path / "dist"
    draft = tmp_path / "content" / "draft-reports" / "123" / "2026" / "post-draft"
    (draft / "assets").mkdir(parents=True)
    (draft / "index.html").write_text("<h1>Draft report</h1>")
    (draft / "assets" / "chart.png").write_bytes(b"png")
    (draft / "report.json").write_text(json.dumps({
        "season": "2026", "title": "2026 Post-Draft Rankings", "status": "Draft",
    }))

    index = build_archive(content, output, {"title": "SYPIP Power Rankings"})
    homepage = index.read_text()

    assert "Draft reports" in homepage
    assert "Weekly rankings begin after Week 1" in homepage
    assert "2026 Post-Draft Rankings" in homepage
    assert (output / "draft-reports/123/2026/post-draft/index.html").exists()
    assert (output / "draft-reports/123/2026/post-draft/assets/chart.png").exists()


def test_invalid_archive_keys_rejected(tmp_path):
    with pytest.raises(ValueError):
        entry_path(tmp_path, "../bad", "2026", 1)


def test_cli_refuses_existing_week(tmp_path, monkeypatch):
    from sleeper_rankings import cli
    class Client:
        def league(self, league_id):
            return {"name": "Test", "season": "2026"}
    monkeypatch.setattr(cli, "SleeperClient", Client)
    entry_path(tmp_path, "123", "2026", 1).mkdir(parents=True)
    args = cli.parser().parse_args(["--league-id", "123", "--week", "1", "--content", str(tmp_path)])
    with pytest.raises(ValueError, match="refusing to overwrite"):
        cli.run(args)


def test_archive_only_never_calls_sleeper(tmp_path, monkeypatch):
    from sleeper_rankings import cli
    def forbidden():
        raise AssertionError("Production archive build must not fetch data")
    monkeypatch.setattr(cli, "SleeperClient", forbidden)
    args = cli.parser().parse_args(["--archive-only", "--content", str(tmp_path / "empty"), "--output", str(tmp_path / "site")])
    homepage = cli.run(args).read_text()
    assert "League report center" in homepage
    assert "No draft reports are available yet" in homepage
    assert "Weekly rankings begin after Week 1" in homepage


def test_overwrite_requires_week_and_rejects_schedule(monkeypatch):
    from sleeper_rankings import cli
    with pytest.raises(ValueError, match="explicit --week"):
        cli.run(cli.parser().parse_args(["--overwrite"]))
    monkeypatch.setenv("GITHUB_EVENT_NAME", "schedule")
    with pytest.raises(ValueError, match="cannot run on a schedule"):
        cli.run(cli.parser().parse_args(["--overwrite", "--week", "2"]))


def test_manual_overwrite_passes_existing_entry_guard(tmp_path, monkeypatch):
    from sleeper_rankings import cli
    class Client:
        def league(self, league_id):
            return {"name": "Test", "season": "2026"}
    monkeypatch.setattr(cli, "SleeperClient", Client)
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
    def reached_generation(*args, **kwargs):
        raise RuntimeError("generation reached")
    monkeypatch.setattr(cli, "load_league", reached_generation)
    entry_path(tmp_path, "123", "2026", 1).mkdir(parents=True)
    args = cli.parser().parse_args(["--league-id", "123", "--week", "1", "--overwrite", "--content", str(tmp_path)])
    with pytest.raises(RuntimeError, match="generation reached"):
        cli.run(args)
