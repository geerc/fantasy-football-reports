# Draft Report

Generates commentary-free post-draft rankings for Sleeper fantasy football leagues. Teams are ranked by the KeepTradeCut redraft value of their strongest legal starting lineup, automatically using Superflex or 1QB rankings to match the league. `ffanalytics` season projections remain the source for the informational projected-points-per-game figure. The report includes league-ranked roster-slot radar charts plus each team's biggest draft reach and value.

## Setup

```shell
python3 -m venv venv
venv/bin/pip install -r requirements.txt -r requirements-dev.txt
Rscript -e 'if (!requireNamespace("remotes", quietly = TRUE)) install.packages("remotes", repos = "https://cloud.r-project.org"); remotes::install_github("FantasyFootballAnalytics/ffanalytics", upgrade = "never")'
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

By default, reports and radar images are written to `draft_report/reports/<season>/` inside this repository. Use `--output PATH` to choose another destination. ffanalytics projections are cached by season and scoring rules after the first run; use `--refresh-projections` when you want fresh projections, or `--projections PATH` to use a specific CSV.

## Optional AI commentary

Reports remain commentary-free by default. To add a concise statistical assessment for each team, create a local `.env` file in the repository root:

```shell
cp .env.example .env
```

Open `.env`, replace the placeholder with your key, and save it. Then enable commentary when generating the report:

```shell
venv/bin/sleeper-draft-report 1388595531374157824 --ai-commentary
```

The command automatically loads `.env` from the repository root. The file is excluded by `.gitignore`, so it will not be committed. Commentary defaults to `gpt-5.6-terra` with low reasoning effort. You can override these with `OPENAI_MODEL` and `OPENAI_REASONING_EFFORT` in `.env`, or with `--ai-model MODEL` and `--ai-reasoning-effort EFFORT`. Commentary generation uses OpenAI web search to research current player context, adds clickable source citations, and may incur both model-token and web-search charges. Four teams are researched concurrently by default; use `--ai-workers NUMBER` or `OPENAI_AI_WORKERS` to adjust concurrency if your API rate limits require it.
