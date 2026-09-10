# Draft Report

Generates post-draft rankings for Sleeper fantasy football leagues. Teams are ranked by the KeepTradeCut redraft value of their strongest legal starting lineup, automatically using Superflex or 1QB rankings to match the league. The report includes league-ranked roster-slot radar charts, draft reach/value picks, AI-written roster analysis, and bye-aware Monte Carlo projected standings.

## Setup

```shell
python3.12 -m venv venv
venv/bin/pip install -r requirements.txt -r requirements-dev.txt
```

## Generate a report

```shell
venv/bin/python -m draft_report.sleeper_draft_report 1388595531374157824
```

Before the real draft is complete, use the included seeded dummy snake draft:

```shell
venv/bin/python -m draft_report.sleeper_draft_report 1388595531374157824 \
  --dummy-draft draft_report/tests/fixtures/dummy_sleeper_draft.json
```

By default, reports and radar images are written to `draft_report/reports/<season>/<league name>/`. Full-overview reports use `<league name>_ai/`. Re-running the same league and mode overwrites that directory; other leagues and modes remain separate. Use `--output PATH` to choose an exact destination.

## AI roster analysis

All reports include AI-written roster analysis. Create a local `.env` file in the repository root:

```shell
cp .env.example .env
```

Open `.env`, replace the placeholder with your key, and save it. The default report contains 3–5 researched bullets per team:

```shell
venv/bin/sleeper-draft-report 1388595531374157824
```

Use the full AI overview mode for longer prose commentary:

```shell
venv/bin/sleeper-draft-report 1388595531374157824 --ai-commentary
```

The command automatically loads `.env` from the repository root. The file is excluded by `.gitignore`, so it will not be committed. Commentary defaults to `gpt-5.6-terra` with low reasoning effort. You can override these with `OPENAI_MODEL` and `OPENAI_REASONING_EFFORT` in `.env`, or with `--ai-model MODEL` and `--ai-reasoning-effort EFFORT`. Commentary generation uses OpenAI web search to research current player context, adds clickable source citations, and may incur both model-token and web-search charges. Four teams are researched concurrently by default; use `--ai-workers NUMBER` or `OPENAI_AI_WORKERS` to adjust concurrency if your API rate limits require it. Projected standings default to 100,000 simulations; `--simulations` and `--simulation-seed` control that run.
