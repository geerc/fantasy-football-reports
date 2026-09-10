import os
from types import SimpleNamespace

import pandas as pd
import pytest

import draft_report.sleeper_draft_report as draft_report
from draft_report.sleeper_draft_report import (
    PlayerProjection,
    add_ktc_values,
    add_kicker_vor_and_rerank,
    fantasypros_adp_settings,
    fetch_ktc_rankings,
    build_team_results,
    combine_supplemental_projections,
    draft_impact_score,
    generate_ai_commentary,
    generate_dummy_picks,
    cached_projection_path,
    commentary_tone,
    league_context,
    ktc_ranking_settings,
    normalize_name,
    normalize_nfl_team,
    overall_pick_from_round_slot,
    player_availability_concern,
    radar_positions_for_league,
    rank_radar_values,
    optimize_lineup,
    pick_summary,
    projection_index,
    render_report,
    report_directory_name,
    render_report_html,
    response_markdown_with_citations,
    safe_directory_name,
    simulate_projected_standings,
    simulation_slots,
    weekly_lineup_value,
)


def player(name, position, points, rank=1):
    return PlayerProjection(
        name, position, "NFL", points, points - 100, rank,
        adp=float(rank), ktc_value=float(points),
    )


def test_parse_args_loads_local_environment(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_MODEL=test-model\n", encoding="utf-8")
    monkeypatch.setattr(draft_report, "LOCAL_ENV_FILE", env_file)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    args = draft_report.parse_args(["123"])

    assert args.ai_model == "test-model"
    assert os.environ["OPENAI_MODEL"] == "test-model"
    assert args.ai_workers == 4
    assert draft_report.DEFAULT_AI_MODEL == "gpt-5.6-terra"
    assert args.ai_reasoning_effort == "low"
    assert args.simulations == 100000
    assert args.simulation_seed == 2026


def test_projection_cache_is_scoped_by_season_and_scoring(tmp_path):
    ppr = cached_projection_path(season=2026, scoring_settings={"rec": 1}, cache_dir=tmp_path)
    standard = cached_projection_path(season=2026, scoring_settings={"rec": 0}, cache_dir=tmp_path)
    next_season = cached_projection_path(season=2027, scoring_settings={"rec": 1}, cache_dir=tmp_path)

    assert ppr.parent == tmp_path
    assert ppr != standard
    assert ppr != next_season
    assert ppr.name.startswith("ffanalytics-2026-")


def test_normalize_name_handles_suffixes_and_punctuation():
    assert normalize_name("D'Andre Swift Jr.") == "dandreswift"


def test_normalize_name_handles_sleeper_projection_aliases():
    assert normalize_name("Kenny Gainwell") == normalize_name("Kenneth Gainwell")
    assert normalize_name("Chig Okonkwo") == normalize_name("Chigoziem Okonkwo")


def test_normalize_nfl_team_handles_ktc_and_espn_aliases():
    assert normalize_nfl_team("KCC") == "KC"
    assert normalize_nfl_team("SFO") == "SF"
    assert normalize_nfl_team("WSH") == "WAS"


def test_optimize_lineup_maximizes_legal_flex_lineup():
    players = [
        player("QB", "QB", 300), player("RB1", "RB", 250), player("RB2", "RB", 200),
        player("WR1", "WR", 240), player("WR2", "WR", 230), player("TE", "TE", 180),
    ]

    score, starters = optimize_lineup(players, ["QB", "RB", "WR", "TE", "FLEX", "BN"])

    assert score == 1200
    assert {item.name for _, item in starters} == {"QB", "RB1", "WR1", "WR2", "TE"}
    assert ("RB", "RB1") in [(slot, item.name) for slot, item in starters]
    assert ("FLEX", "WR2") in [(slot, item.name) for slot, item in starters]


def test_optimize_lineup_treats_superflex_as_second_qb_and_fills_native_slots_first():
    players = [
        player("QB1", "QB", 9000), player("QB2", "QB", 7000),
        player("RB1", "RB", 9941), player("RB2", "RB", 6208),
        player("RB3", "RB", 6026),
    ]

    score, starters = optimize_lineup(
        players, ["QB", "RB", "RB", "FLEX", "SUPER_FLEX"], score_attribute="ktc_value",
    )

    assignments = [(slot, item.name) for slot, item in starters]
    assert assignments == [
        ("QB", "QB1"), ("RB", "RB1"), ("RB", "RB2"),
        ("QB", "QB2"), ("FLEX", "RB3"),
    ]
    assert score == 38175


def test_optimize_lineup_rejects_incomplete_roster():
    with pytest.raises(ValueError, match="Unable to fill"):
        optimize_lineup([player("Only QB", "QB", 10)], ["QB", "RB"])


def test_kicker_vor_is_merged_before_overall_reranking():
    base = pd.DataFrame([
        {"name": "Quarterback", "team": "BUF", "position": "QB", "points": 300, "points_vor": 50, "rank": 1},
    ])
    kickers = pd.DataFrame([
        {"name": "K One", "team": "DAL", "position": "K", "points": 140},
        {"name": "K Two", "team": "BAL", "position": "K", "points": 120},
    ])

    result = add_kicker_vor_and_rerank(base, kickers, baseline={"K": 2})

    assert result.loc[result["name"] == "K One", "points_vor"].item() == 20
    assert result.loc[result["name"] == "Quarterback", "rank"].item() == 1
    assert result.loc[result["name"] == "K One", "rank"].item() == 2


def test_supplemental_projections_prefer_later_source_values():
    cbs = pd.DataFrame([
        {"name": "K One", "team": "DAL", "position": "K", "points": 140},
        {"name": "K Two", "team": "BAL", "position": "K", "points": 120},
    ])
    fantasypros = pd.DataFrame([
        {"name": "K One", "team": "DAL", "position": "K", "points": 145},
    ])

    result = combine_supplemental_projections(cbs, fantasypros)

    assert len(result) == 2
    assert result.loc[result["name"] == "K One", "points"].item() == 145


def test_fantasypros_adp_settings_match_superflex_ppr_team_count():
    settings = fantasypros_adp_settings({
        "roster_positions": ["QB", "SUPER_FLEX", "BN"],
        "scoring_settings": {"rec": 1},
    }, 10)

    assert settings["roster_format"] == "2qb"
    assert settings["scoring"] == "ppr"
    assert settings["team_count"] == 10
    assert settings["url"].endswith("/2qb-ppr-10-teams")
    assert overall_pick_from_round_slot(2.03, 10) == 13
    assert overall_pick_from_round_slot(4.0, 10) == 40


def test_ktc_ranking_settings_match_sleeper_starting_lineup():
    assert ktc_ranking_settings({"roster_positions": ["QB", "SUPER_FLEX", "BN"]}) == {
        "format": 2, "roster_format": "superflex",
    }
    assert ktc_ranking_settings({"roster_positions": ["QB", "FLEX", "BN"]}) == {
        "format": 1, "roster_format": "1qb",
    }
    assert ktc_ranking_settings({"roster_positions": ["QB", "QB", "BN"]})["format"] == 2


def test_fetch_ktc_rankings_uses_superflex_page_and_normalizes_kicker():
    class Response:
        content = b"""<div class='onePlayer'><div class='player-name'><a>Josh Allen</a></div>
        <p class='position'>QB1</p><div class='value'>9,744</div></div>
        <div class='onePlayer'><div class='player-name'><a>Kicker One</a></div>
        <p class='position'>PK1</p><div class='value'>500</div></div>"""

        def raise_for_status(self):
            return None

    class Session:
        def __init__(self):
            self.urls = []

        def get(self, url, **kwargs):
            self.urls.append(url)
            return Response()

    session = Session()
    result, settings = fetch_ktc_rankings(
        {"roster_positions": ["QB", "SUPER_FLEX"]}, session=session, pages=1,
    )

    assert settings["format"] == 2
    assert "format=2" in session.urls[0]
    assert result.set_index("name").loc["Josh Allen", "ktc_value"] == 9744
    assert result.set_index("name").loc["Kicker One", "position"] == "K"


def test_add_ktc_values_matches_normalized_player_and_position():
    projections = pd.DataFrame([
        {"name": "D'Andre Swift Jr.", "position": "RB", "points": 200},
    ])
    values = pd.DataFrame([
        {"name": "Dandre Swift", "position": "RB", "ktc_value": 7000},
    ])

    result = add_ktc_values(projections, values)

    assert result["ktc_value"].item() == 7000


def test_team_results_are_worst_to_best_and_reach_value_use_actual_pick():
    frame = pd.DataFrame([
        {"name": "Alpha QB", "team": "BUF", "position": "QB", "points": 300, "points_vor": 10, "rank": 10, "adp": 10},
        {"name": "Alpha RB", "team": "DAL", "position": "RB", "points": 200, "points_vor": 5, "rank": 20, "adp": 20},
        {"name": "Beta QB", "team": "KC", "position": "QB", "points": 350, "points_vor": 30, "rank": 1, "adp": 1},
        {"name": "Beta RB", "team": "SF", "position": "RB", "points": 250, "points_vor": 20, "rank": 2, "adp": 2},
    ])
    picks = [
        {"roster_id": 1, "pick_no": 1, "metadata": {"first_name": "Alpha", "last_name": "QB", "position": "QB"}},
        {"roster_id": 1, "pick_no": 20, "metadata": {"first_name": "Alpha", "last_name": "RB", "position": "RB"}},
        {"roster_id": 2, "pick_no": 2, "metadata": {"first_name": "Beta", "last_name": "QB", "position": "QB"}},
        {"roster_id": 2, "pick_no": 30, "metadata": {"first_name": "Beta", "last_name": "RB", "position": "RB"}},
    ]
    rosters = [{"roster_id": 1, "owner_id": "a"}, {"roster_id": 2, "owner_id": "b"}]
    users = {
        "a": {"display_name": "Alpha Manager", "metadata": {"team_name": "Alpha Team"}},
        "b": {"display_name": "Beta Manager", "metadata": {}},
    }
    league = {"roster_positions": ["QB", "RB"]}

    results, unmatched = build_team_results(
        league=league, rosters=rosters, users=users, picks=picks, projections=projection_index(frame),
    )

    assert unmatched == []
    assert [item["team"] for item in results] == ["Alpha Team", "Beta Manager"]
    assert [item["rank"] for item in results] == [2, 1]
    assert results[0]["reach"][1].name == "Alpha QB"
    assert results[0]["reach"][2] == 9
    assert results[0]["value"][1].name == "Alpha RB"
    assert results[0]["value"][2] == 0
    assert results[0]["ktc_value"] == 500
    assert results[0]["roster_construction"] == {"QB": 1, "RB": 1}
    assert results[0]["roster"] == [
        {"name": "Alpha QB", "position": "QB"},
        {"name": "Alpha RB", "position": "RB"},
    ]


def test_team_results_rank_by_ktc_value_instead_of_projected_points():
    frame = pd.DataFrame([
        {"name": "Lower Projection", "team": "BUF", "position": "QB", "points": 250,
         "points_vor": 10, "rank": 2, "adp": 2, "ktc_value": 9000},
        {"name": "Higher Projection", "team": "KC", "position": "QB", "points": 350,
         "points_vor": 20, "rank": 1, "adp": 1, "ktc_value": 7000},
    ])
    picks = [
        {"roster_id": 1, "pick_no": 1, "metadata": {"full_name": "Lower Projection", "position": "QB"}},
        {"roster_id": 2, "pick_no": 2, "metadata": {"full_name": "Higher Projection", "position": "QB"}},
    ]
    rosters = [{"roster_id": 1, "owner_id": "a"}, {"roster_id": 2, "owner_id": "b"}]

    results, _ = build_team_results(
        league={"roster_positions": ["QB"]}, rosters=rosters,
        users={"a": {"display_name": "KTC Favorite"}, "b": {"display_name": "Projection Favorite"}},
        picks=picks, projections=projection_index(frame),
    )

    assert [item["team"] for item in results] == ["Projection Favorite", "KTC Favorite"]
    assert [item["rank"] for item in results] == [2, 1]


def test_draft_impact_score_gives_early_picks_more_weight():
    assert draft_impact_score(10, 50, 240) > draft_impact_score(220, 75, 240)
    assert draft_impact_score(10, -50, 240) < draft_impact_score(220, -75, 240)


def test_reach_and_value_ignore_final_quarter_of_draft():
    frame = pd.DataFrame([
        {"name": "Early Reach", "team": "BUF", "position": "QB", "points": 300, "points_vor": 10, "rank": 20, "adp": 20},
        {"name": "Early Value", "team": "DAL", "position": "RB", "points": 200, "points_vor": 5, "rank": 1, "adp": 1},
        {"name": "Late Outlier", "team": "KC", "position": "WR", "points": 100, "points_vor": 1, "rank": 100, "adp": 100},
        {"name": "Other Player", "team": "SF", "position": "TE", "points": 100, "points_vor": 1, "rank": 4, "adp": 4},
    ])
    picks = [
        {"roster_id": 1, "pick_no": 1, "metadata": {"first_name": "Early", "last_name": "Reach", "position": "QB"}},
        {"roster_id": 1, "pick_no": 2, "metadata": {"first_name": "Early", "last_name": "Value", "position": "RB"}},
        {"roster_id": 1, "pick_no": 4, "metadata": {"first_name": "Late", "last_name": "Outlier", "position": "WR"}},
        {"roster_id": 2, "pick_no": 3, "metadata": {"first_name": "Other", "last_name": "Player", "position": "TE"}},
    ]
    results, _ = build_team_results(
        league={"roster_positions": ["QB", "RB", "WR"]},
        rosters=[{"roster_id": 1, "owner_id": "a"}],
        users={"a": {"display_name": "Manager"}}, picks=picks,
        projections=projection_index(frame),
    )

    assert results[0]["reach"][1].name == "Early Reach"
    assert results[0]["value"][1].name == "Early Value"


def test_player_availability_concern_uses_sleeper_status():
    assert player_availability_concern({
        "full_name": "Injured Player", "position": "RB",
        "injury_status": "Questionable", "injury_body_part": "Hamstring",
    }) == "Injured Player (RB): Questionable — Hamstring"
    assert player_availability_concern({
        "full_name": "Suspended Player", "position": "WR", "status": "Suspended",
    }) == "Suspended Player (WR): Suspended"
    assert player_availability_concern({"full_name": "Healthy Player", "status": "Active"}) is None


def test_team_results_use_full_sleeper_roster():
    frame = pd.DataFrame([
        {"name": "Kept QB", "team": "BUF", "position": "QB", "points": 300, "points_vor": 20, "rank": 2, "adp": 2},
        {"name": "Drafted RB", "team": "DAL", "position": "RB", "points": 200, "points_vor": 10, "rank": 10, "adp": 10},
    ])
    rosters = [{"roster_id": 1, "owner_id": "a", "players": ["kept", "drafted"]}]
    catalog = {
        "kept": {"full_name": "Kept QB", "position": "QB", "team": "BUF"},
        "drafted": {"full_name": "Drafted RB", "position": "RB", "team": "DAL"},
    }
    picks = [
        {
            "roster_id": 1, "pick_no": 5,
            "metadata": {"first_name": "Drafted", "last_name": "RB", "position": "RB"},
        },
        {
            "roster_id": 2, "pick_no": 8,
            "metadata": {"first_name": "Drafted", "last_name": "RB", "position": "RB"},
        },
    ]

    results, unmatched = build_team_results(
        league={"roster_positions": ["QB", "RB"]}, rosters=rosters,
        users={"a": {"display_name": "Manager"}}, picks=picks,
        projections=projection_index(frame), player_catalog=catalog,
    )

    assert unmatched == []
    assert results[0]["ktc_value"] == 500
    assert results[0]["reach"][1].name == "Drafted RB"


def test_dummy_draft_is_reproducible_randomized_snake():
    frame = pd.DataFrame([
        {
            "name": f"Player {index}", "team": "NFL", "position": "RB",
            "points": 300 - index, "points_vor": 100 - index, "rank": index,
        }
        for index in range(1, 25)
    ])

    first = generate_dummy_picks(frame, teams=4, rounds=3, seed=17)
    second = generate_dummy_picks(frame, teams=4, rounds=3, seed=17)
    different = generate_dummy_picks(frame, teams=4, rounds=3, seed=18)

    assert first == second
    assert [pick["draft_slot"] for pick in first] == [1, 2, 3, 4, 4, 3, 2, 1, 1, 2, 3, 4]
    assert [pick["metadata"] for pick in first] != [pick["metadata"] for pick in different]


def test_radar_uses_positional_rank_among_teams():
    results = [
        {"position_totals": {position: 100 for position in ("QB", "RB", "WR", "TE", "K", "DST")}},
        {"position_totals": {position: 200 for position in ("QB", "RB", "WR", "TE", "K", "DST")}},
    ]
    rank_radar_values(results, ("QB", "RB", "WR", "TE", "K", "DST"))
    assert set(results[0]["position_ranks"].values()) == {2}
    assert set(results[1]["position_ranks"].values()) == {1}
    assert set(results[0]["radar"].values()) == {1}
    assert set(results[1]["radar"].values()) == {2}


def test_radar_all_zero_position_stays_at_zero():
    totals = {position: 100 for position in ("QB", "RB", "WR", "TE", "K", "DST")}
    totals["K"] = 0
    results = [{"position_totals": totals}]

    rank_radar_values(results, ("QB", "RB", "WR", "TE", "K", "DST"))

    assert results[0]["position_ranks"]["K"] is None
    assert results[0]["radar"]["K"] == 0
    assert results[0]["position_ranks"]["QB"] == 1
    assert results[0]["radar"]["QB"] == 1


def test_radar_omits_kicker_and_defense_when_league_does_not_use_them():
    assert radar_positions_for_league(["QB", "RB", "WR", "TE", "FLEX", "BN"]) == (
        "QB", "RB", "WR", "TE", "FLEX",
    )
    assert radar_positions_for_league(["QB", "RB", "WR", "TE", "K", "DEF", "BN"]) == (
        "QB", "RB", "WR", "TE", "K", "DST",
    )
    assert radar_positions_for_league(["QB", "SUPER_FLEX", "RB", "BN"]) == (
        "QB", "RB",
    )


def test_flex_starters_are_scored_in_a_dedicated_equal_slot_bucket():
    frame = pd.DataFrame([
        {"name": "Team One RB", "team": "BUF", "position": "RB", "points": 200,
         "points_vor": 1, "rank": 1, "adp": 1, "ktc_value": 8000},
        {"name": "Team One WR", "team": "BUF", "position": "WR", "points": 190,
         "points_vor": 1, "rank": 2, "adp": 2, "ktc_value": 7000},
        {"name": "Team Two RB", "team": "KC", "position": "RB", "points": 180,
         "points_vor": 1, "rank": 3, "adp": 3, "ktc_value": 6000},
        {"name": "Team Two WR", "team": "KC", "position": "WR", "points": 170,
         "points_vor": 1, "rank": 4, "adp": 4, "ktc_value": 5000},
    ])
    picks = [
        {"roster_id": 1, "pick_no": 1, "metadata": {"full_name": "Team One RB", "position": "RB"}},
        {"roster_id": 1, "pick_no": 2, "metadata": {"full_name": "Team One WR", "position": "WR"}},
        {"roster_id": 2, "pick_no": 3, "metadata": {"full_name": "Team Two RB", "position": "RB"}},
        {"roster_id": 2, "pick_no": 4, "metadata": {"full_name": "Team Two WR", "position": "WR"}},
    ]
    rosters = [{"roster_id": 1, "owner_id": "a"}, {"roster_id": 2, "owner_id": "b"}]

    results, _ = build_team_results(
        league={"roster_positions": ["RB", "FLEX"]}, rosters=rosters,
        users={"a": {"display_name": "One"}, "b": {"display_name": "Two"}},
        picks=picks, projections=projection_index(frame),
    )

    assert [item["position_totals"]["RB"] for item in results] == [6000, 8000]
    assert [item["position_totals"]["FLEX"] for item in results] == [5000, 7000]


def test_report_contains_only_structured_rankings_and_statistics():
    item = {
        "rank": 1, "roster_id": 7, "team": "Champions",
        "reach": (3, player("Reach", "WR", 100, rank=12), 9),
        "value": (30, player("Value", "RB", 100, rank=10), -20),
        "position_ranks": {"QB": 2, "RB": 1},
        "availability_concerns": ["Risky Player (WR): Questionable — Knee"],
        "commentary": "- Strong quarterback room.\n- Thin at running back.",
        "full_overview": False,
    }
    standings = pd.DataFrame([{
        "Rank": 1, "Team": "Champions", "Projected Wins": 10.25, "Playoff Probability": 85.5,
    }])
    content = render_report(
        league={"season": "2026", "name": "League"}, results=[item], standings=standings,
    )

    assert "## #1 Champions" in content
    assert "Projected starter points" not in content
    assert "team-7-radar.png" in content
    assert pick_summary(item["reach"]) in content
    assert "Position-group rankings:** QB #2, RB #1" in content
    assert "Risky Player (WR): Questionable — Knee" in content
    assert "- Strong quarterback room." in content
    assert "## Projected Standings" in content
    assert "|      1 | Champions" in content


def test_report_html_wraps_markdown_and_writes_site_styles(tmp_path):
    output = tmp_path / "index.html"

    render_report_html(
        markdown_content=(
            "+++\ntitle = \"Draft\"\n+++\n\n# Draft\n\n"
            "# *Cellar Dwellars*\n\n## #1 Team\n\nAnalysis."
        ),
        league={"season": "2026", "name": "Test League"}, output_path=output,
    )

    html = output.read_text()
    assert "2026 Post-Draft Rankings" in html
    assert "<h1><em>Cellar Dwellars</em></h1>" in html
    assert "<h2>#1 Team</h2>" in html
    css = (tmp_path / "assets/site.css").read_text()
    assert ".report-prose>h1:first-of-type{display:none}" in css


def test_report_html_renders_projected_standings_as_a_table(tmp_path):
    output = tmp_path / "index.html"
    standings = pd.DataFrame([{
        "Rank": 1, "Team": "Champions", "Projected Wins": 10.25, "Playoff Probability": 85.5,
    }])
    item = {
        "rank": 1, "roster_id": 7, "team": "Champions", "reach": None, "value": None,
        "position_ranks": {"QB": 1}, "availability_concerns": [],
        "commentary": "- Strong roster.", "full_overview": False,
    }
    markdown_content = render_report(
        league={"season": "2026", "name": "League"}, results=[item], standings=standings,
    )

    render_report_html(
        markdown_content=markdown_content,
        league={"season": "2026", "name": "League"}, output_path=output,
    )

    html = output.read_text()
    assert '<table class="rankings-table">' in html
    assert "Playoff Probability</th>" in html
    assert "85.5%</td>" in html


def test_league_context_distinguishes_best_ball_and_ppr():
    context = league_context({
        "settings": {"best_ball": 1},
        "scoring_settings": {"rec": 0.5},
        "roster_positions": ["QB", "RB", "FLEX", "BN"],
    })

    assert context == {
        "format": "best ball",
        "reception_scoring": "half PPR",
        "starting_lineup_slots": ["QB", "RB", "FLEX"],
    }


def test_commentary_tone_tracks_overall_rank_without_becoming_one_sided():
    assert commentary_tone(1, 12).startswith("strongly positive")
    assert commentary_tone(4, 12).startswith("positive-leaning")
    assert commentary_tone(8, 12).startswith("critical-leaning")
    assert commentary_tone(12, 12).startswith("strongly critical")
    assert "concern" in commentary_tone(1, 12)
    assert "strength" in commentary_tone(12, 12)


def test_ai_commentary_is_opt_in_and_uses_roster_research_context():
    calls = []

    class FakeResponses:
        def create(self, **kwargs):
            calls.append(kwargs)
            statistics = __import__("json").loads(kwargs["input"])
            return SimpleNamespace(output_text=f'{statistics["team"]} is ranked #{statistics["overall_rank"]}.')

    client = SimpleNamespace(responses=FakeResponses())
    results = [{
        "roster_id": 7,
        "team": "Alpha",
        "rank": 2,
        "team_count": 12,
        "adp_settings": {"roster_format": "2qb", "scoring": "ppr", "team_count": 12},
        "roster_construction": {"QB": 2, "RB": 6, "WR": 7, "TE": 2},
        "roster": [{"name": "Example Player", "position": "WR"}],
        "position_ranks": {"QB": 3, "RB": 8, "K": None},
        "reach": (3, player("Reach", "WR", 100, rank=12), 9),
        "value": (30, player("Value", "RB", 100, rank=10), -20),
    }]

    generate_ai_commentary(
        league={
            "name": "League", "settings": {"best_ball": 1},
            "scoring_settings": {"rec": 1}, "roster_positions": ["QB", "RB", "WR", "FLEX", "BN"],
        }, results=results, api_key=None,
        model="test-model", client=client,
    )

    assert results[0]["commentary"] == "Alpha is ranked #2."
    assert calls[0]["model"] == "test-model"
    assert calls[0]["store"] is False
    assert calls[0]["tools"] == [{"type": "web_search"}]
    assert calls[0]["tool_choice"] == "auto"
    assert calls[0]["reasoning"] == {"effort": "low"}
    statistics = __import__("json").loads(calls[0]["input"])
    assert statistics["position_ranks"] == {"QB": 3, "RB": 8}
    assert statistics["roster_construction"] == {"QB": 2, "RB": 6, "WR": 7, "TE": 2}
    assert statistics["league_context"]["format"] == "best ball"
    assert "projected_starter_points_per_game" not in statistics
    assert statistics["adp_settings"]["roster_format"] == "2qb"
    assert statistics["editorial_tone"].startswith("strongly positive")
    assert statistics["roster"][0]["name"] == "Example Player"
    assert "do not recite" in calls[0]["instructions"].lower()
    assert "informal" in calls[0]["instructions"]
    assert "never as VOR" in calls[0]["instructions"]
    assert "source of truth" in calls[0]["instructions"]
    assert "exactly 3-5" in calls[0]["instructions"]


def test_web_citations_are_rendered_as_clickable_footnotes():
    annotation = SimpleNamespace(
        type="url_citation", start_index=15, end_index=21,
        title="FantasyPros analysis", url="https://www.fantasypros.com/example",
    )
    content = SimpleNamespace(
        type="output_text", text="A strong claim source.", annotations=[annotation],
    )
    response = SimpleNamespace(
        output=[SimpleNamespace(type="message", content=[content])], output_text=content.text,
    )

    assert response_markdown_with_citations(response, footnote_prefix="team-7") == (
        "A strong claim [^team-7-1].\n\n"
        "[^team-7-1]: [FantasyPros analysis](https://www.fantasypros.com/example)"
    )


def test_ai_commentary_requires_api_key_without_injected_client():
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        generate_ai_commentary(
            league={"name": "League"}, results=[], api_key=None, model="test-model",
        )


def test_report_includes_commentary_only_when_present():
    item = {
        "rank": 1, "roster_id": 7, "team": "Champions",
        "reach": None, "value": None, "commentary": "A concise statistical assessment.",
        "full_overview": True,
    }
    standings = pd.DataFrame([{
        "Rank": 1, "Team": "Champions", "Projected Wins": 10, "Playoff Probability": 100,
    }])
    content = render_report(
        league={"season": "2026", "name": "League"}, results=[item], standings=standings,
    )

    assert "A concise statistical assessment." in content


def test_simulation_excludes_kicker_and_defense_and_replaces_bye_players():
    slots = simulation_slots(["QB", "RB", "SUPER_FLEX", "K", "DEF", "BN"])
    assert slots == ["QB", "RB", "QB"]
    roster = [
        PlayerProjection("QB One", "QB", "BUF", 0, 0, 1, ktc_value=9000),
        PlayerProjection("QB Two", "QB", "KC", 0, 0, 2, ktc_value=8000),
        PlayerProjection("RB One", "RB", "DAL", 0, 0, 3, ktc_value=7000),
    ]
    assert weekly_lineup_value(roster, slots, {"BUF"}, {"QB": 6000, "RB": 5000}) == 21000


def test_projected_standings_use_schedule_and_return_playoff_odds():
    strong = PlayerProjection("Strong QB", "QB", "BUF", 0, 0, 1, ktc_value=9000)
    weak = PlayerProjection("Weak QB", "QB", "DAL", 0, 0, 2, ktc_value=3000)
    results = [
        {"roster_id": 1, "team": "Strong", "_players": [strong]},
        {"roster_id": 2, "team": "Weak", "_players": [weak]},
    ]
    standings = simulate_projected_standings(
        results=results, roster_positions=["QB", "K"], schedule={1: [(1, 2)], 2: [(1, 2)]},
        bye_teams={1: set(), 2: set()}, playoff_teams=1, simulations=5000, seed=7,
    )
    assert standings.iloc[0]["Team"] == "Strong"
    assert standings.iloc[0]["Projected Wins"] > standings.iloc[1]["Projected Wins"]
    assert standings.iloc[0]["Playoff Probability"] > 90


def test_safe_directory_name_keeps_league_names_and_removes_path_characters():
    assert safe_directory_name("SYPIP") == "SYPIP"
    assert safe_directory_name("League / One") == "League - One"
    assert report_directory_name("SYPIP", False) == "SYPIP"
    assert report_directory_name("SYPIP", True) == "SYPIP_ai"
