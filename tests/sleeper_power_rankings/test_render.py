import pandas as pd

from sleeper_rankings.render import CSS, _table


def test_html_tables_center_data_and_color_movement():
    frame = pd.DataFrame(
        {
            "Team": ["Alpha", "Beta", "Gamma"],
            "Power Score": [90, 80, 70],
            "Weekly Change": ["↑ 1", "↓ 1", "—"],
        },
        index=[1, 2, 3],
    )

    html = _table(frame)

    assert '.rankings-table th,.rankings-table td{text-align:center}' in CSS
    assert ".rankings-table th:nth-child(2),.rankings-table td:nth-child(2){text-align:left}" in CSS
    assert '<span class="movement up">↑ 1</span>' in html
    assert '<span class="movement down">↓ 1</span>' in html
    assert '<span class="movement same">—</span>' in html


def test_recap_markdown_and_saved_presentation_are_safe_and_idempotent():
    from bs4 import BeautifulSoup
    from sleeper_rankings.render import render_recap, refresh_saved_presentation
    rendered = render_recap('## Game of the Week\n\n**Great game**\n\n- One\n- Two\n\n<script>alert(1)</script>')
    assert '<h3>Game of the Week</h3>' in rendered
    assert '<strong>Great game</strong>' in rendered
    assert '<ul>' in rendered
    assert '<script>' not in rendered
    old = '<main><table><tr><td>123.4</td></tr></table><section class="recap"><div class="prose">## Game of the Week<br><br>**Great game**</div></section></main>'
    new = refresh_saved_presentation(old)
    soup = BeautifulSoup(new, 'html.parser')
    assert soup.select_one('.home-button')['href'] == '../../../../'
    assert soup.select_one('.prose h3').text == 'Game of the Week'
    assert soup.select_one('td').text == '123.4'
    assert refresh_saved_presentation(new) == new
