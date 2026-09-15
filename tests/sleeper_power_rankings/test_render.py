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
