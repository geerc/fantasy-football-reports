from __future__ import annotations

from datetime import datetime, timezone
from html import escape
from pathlib import Path
import re
import shutil

import pandas as pd
from bs4 import BeautifulSoup
from markdown_it import MarkdownIt

from .power import METHODOLOGY


HOME_NAV = '<nav class="report-nav" aria-label="Report navigation"><a class="home-button" href="../../../../">← Home</a></nav>'


def render_recap(summary: str) -> str:
    """Render Markdown while treating embedded HTML as untrusted text."""
    markdown = MarkdownIt("commonmark", {"html": False})
    tokens = markdown.parse(summary)
    for token in tokens:
        if token.type in {"heading_open", "heading_close"}:
            token.tag = "h3" if int(token.tag[1:]) <= 3 else "h4"
    return markdown.renderer.render(tokens, markdown.options, {})


def refresh_saved_presentation(html: str) -> str:
    """Upgrade archived presentation without recalculating saved report data."""
    soup = BeautifulSoup(html, "html.parser")
    main = soup.find("main")
    if main is not None and not soup.select_one(".report-nav"):
        main.insert(0, BeautifulSoup(HOME_NAV, "html.parser"))
    for prose in soup.select(".recap .prose"):
        if prose.get("data-format") == "markdown":
            continue
        # The old renderer escaped Markdown and represented newlines as <br>.
        if any(prose.find(tag) for tag in ("p", "h3", "ul", "ol")):
            continue
        for br in prose.find_all("br"):
            br.replace_with("\n")
        rendered = render_recap(prose.get_text())
        prose.clear()
        prose.append(BeautifulSoup(rendered, "html.parser"))
        prose["data-format"] = "markdown"
    return str(soup)


def _table(frame: pd.DataFrame, hidden: tuple[str, ...] = ()) -> str:
    display = frame.drop(columns=[column for column in hidden if column in frame], errors="ignore").copy()
    display.insert(0, "Rank", display.index)
    html = display.to_html(index=False, border=0, classes="rankings-table", escape=True, justify="left")
    html = re.sub(r"<td>(↑[^<]*)</td>", r'<td><span class="movement up">\1</span></td>', html)
    html = re.sub(r"<td>(↓[^<]*)</td>", r'<td><span class="movement down">\1</span></td>', html)
    html = re.sub(r"<td>(—)</td>", r'<td><span class="movement same">\1</span></td>', html)
    return html


def render_site(*, output: Path, title: str, league_name: str, season: str, week: int, rankings: pd.DataFrame, summary: str | None, playoffs: pd.DataFrame | None, standings: pd.DataFrame, luck: pd.DataFrame, header_image: Path | None = None) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    assets = output / "assets"
    assets.mkdir(exist_ok=True)
    hero_class = "hero"
    if header_image is not None and header_image.is_file():
        shutil.copy2(header_image, assets / "header.jpg")
        hero_class += " has-image"
    summary_html = f'<section class="recap"><h2>Week {week} recap</h2><div class="prose" data-format="markdown">{render_recap(summary)}</div></section>' if summary else '<p class="note">AI recap omitted. Add <code>OPENAI_API_KEY</code> to enable it.</p>'
    playoffs_html = f'<section><h2>Playoff probabilities</h2>{_table(playoffs)}</section>' if playoffs is not None else ""
    generated = datetime.now(timezone.utc).strftime("%B %d, %Y at %H:%M UTC")
    html = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="Weekly Sleeper fantasy football power rankings for {escape(league_name)}">
<title>{escape(title)}</title><link rel="stylesheet" href="assets/site.css"></head>
<body><header class="{hero_class}"><div class="wrap"><p class="eyebrow">{escape(season)} · Week {week}</p><h1>{escape(title)}</h1><p>{escape(league_name)} · Powered by Sleeper results and KeepTradeCut roster values</p></div></header>
<main class="wrap">{HOME_NAV}<section><div class="section-title"><div><p class="eyebrow">The table</p><h2>Power rankings</h2></div><p class="note">{escape(METHODOLOGY)}</p></div>{_table(rankings, ("roster_id", "Performance Score", "Roster Value"))}</section>
{summary_html}{playoffs_html}<section><h2>Projected standings</h2>{_table(standings)}</section><section><h2>Luck index</h2><p class="note">Positive means fortunate; negative means unlucky. Based on opponent draw, score consistency, and close-game outcomes.</p>{_table(luck)}</section>
<footer>Generated {generated}. Data updates when the scheduled GitHub workflow runs.</footer></main></body></html>'''
    (output / "index.html").write_text(html, encoding="utf-8")
    (assets / "site.css").write_text(CSS, encoding="utf-8")
    return output / "index.html"


CSS = """
:root{--ink:#17231d;--muted:#617067;--paper:#f5f1e7;--card:#fffdf7;--line:#d8d3c6;--accent:#ee5b2b;--green:#1d6949}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.55 ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}.wrap{width:min(1120px,calc(100% - 32px));margin:auto}.hero{padding:72px 0 64px;background:var(--ink);color:#fff;border-bottom:8px solid var(--accent)}.hero.has-image{position:relative;background:linear-gradient(#17231dcc,#17231dcc),url("header.jpg") center/cover no-repeat}.hero h1{font:800 clamp(2.8rem,8vw,6.5rem)/.92 Georgia,serif;max-width:900px;margin:.12em 0}.hero p:last-child{color:#cad3cd}.eyebrow{text-transform:uppercase;letter-spacing:.16em;font-weight:800;font-size:.78rem;color:var(--accent)}main{padding:36px 0 72px}section{margin:0 0 48px;background:var(--card);border:1px solid var(--line);border-radius:18px;padding:clamp(18px,4vw,38px);box-shadow:0 14px 40px #17231d0d}h2{font:800 clamp(1.8rem,5vw,3rem)/1 Georgia,serif;margin:.2em 0 .7em}.section-title{display:flex;align-items:end;justify-content:space-between;gap:24px}.note{color:var(--muted);font-size:.92rem}.rankings-table{width:100%;border-collapse:collapse;display:block;overflow-x:auto}.rankings-table th{background:var(--ink);color:#fff;text-align:left;font-size:.78rem;letter-spacing:.06em;text-transform:uppercase}.rankings-table th,.rankings-table td{padding:13px 15px;border-bottom:1px solid var(--line);white-space:nowrap}.rankings-table tr:nth-child(even) td{background:#f5f1e780}.rankings-table td:first-child{font-weight:900;color:var(--accent)}.prose{max-width:800px;color:#324238}footer{color:var(--muted);font-size:.82rem;text-align:center}@media(max-width:650px){.hero{padding:52px 0 42px}.section-title{display:block}section{border-radius:12px;padding:18px}.rankings-table th,.rankings-table td{padding:10px 11px;font-size:.86rem}}
"""

CSS += """
.rankings-table th,.rankings-table td{text-align:center}
.rankings-table th:nth-child(2),.rankings-table td:nth-child(2){text-align:left}
.movement.up{color:#15803d;font-weight:800}.movement.down{color:#b91c1c;font-weight:800}.movement.same{color:#111827}
"""


def render_preseason(*, output: Path, title: str, league_name: str, season: str) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    generated = datetime.now(timezone.utc).strftime("%B %d, %Y at %H:%M UTC")
    html = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="Something is coming.">
<title>Stand by.</title><link rel="stylesheet" href="assets/site.css"></head>
<body class="teaser"><main class="teaser-inner"><div class="signal" aria-hidden="true"></div><p class="eyebrow">Transmission pending</p><h1>Something is coming.</h1><p class="teaser-copy">The league will know soon enough.</p><p class="timestamp">Signal checked {generated}</p></main></body></html>'''
    (output / "index.html").write_text(html, encoding="utf-8")
    assets = output / "assets"
    assets.mkdir(exist_ok=True)
    (assets / "site.css").write_text(CSS, encoding="utf-8")
    return output / "index.html"


CSS += """
.teaser{min-height:100vh;background:#080b09;color:#e8eee9;display:grid;place-items:center;overflow:hidden}.teaser:before{content:"";position:fixed;inset:-30%;background:radial-gradient(circle,#1d694933 0,transparent 45%);animation:pulse 7s ease-in-out infinite}.teaser-inner{position:relative;width:min(720px,calc(100% - 40px));padding:72px 20px;text-align:center}.teaser h1{font:800 clamp(3.4rem,12vw,8rem)/.86 Georgia,serif;letter-spacing:-.055em;margin:.2em 0}.teaser-copy{color:#91a097;font-size:clamp(1rem,2.5vw,1.3rem);letter-spacing:.08em}.signal{width:10px;height:10px;margin:0 auto 40px;border-radius:50%;background:#ee5b2b;box-shadow:0 0 30px #ee5b2b;animation:blink 2.4s ease-in-out infinite}.timestamp{margin-top:72px;color:#3d4942;font-size:.68rem;letter-spacing:.12em;text-transform:uppercase}@keyframes pulse{50%{transform:scale(1.18);opacity:.55}}@keyframes blink{50%{opacity:.18;box-shadow:0 0 6px #ee5b2b}}
"""

CSS += """
.report-nav{margin:0 0 24px}.home-button{display:inline-flex;align-items:center;min-height:44px;padding:9px 18px;border:1px solid var(--green);border-radius:10px;color:var(--green);font-weight:700;text-decoration:none;background:var(--card)}
.home-button:hover{background:var(--green);color:white}a:focus-visible{outline:3px solid var(--accent);outline-offset:4px}
.prose p{margin:0 0 1.1em}.prose h3{font:700 clamp(1.25rem,3vw,1.6rem)/1.25 Georgia,serif;margin:1.6em 0 .6em;color:var(--ink)}.prose h3:first-child{margin-top:0}.prose ul,.prose ol{padding-left:1.4em;margin:0 0 1.2em}.prose li{margin:.4em 0}.prose a{color:var(--green)}.prose{overflow-wrap:anywhere}
.section-title{align-items:start;flex-wrap:wrap}.section-title .note{max-width:640px;margin:0 0 20px}.section-title>div{flex:1 0 220px}
.archive-intro{color:var(--muted);margin:0 0 24px}.report-list{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,260px),1fr));gap:14px;list-style:none;padding:0;margin:0}.report-list a,.report-list a:visited{display:flex;align-items:center;justify-content:space-between;gap:16px;padding:20px 22px;min-height:72px;border:1px solid var(--line);border-radius:12px;background:var(--paper);color:var(--ink);font-weight:700;text-decoration:none}.report-list a:hover{border-color:var(--green);background:#eaf0e8}.report-list .arrow{color:var(--green)}
"""
