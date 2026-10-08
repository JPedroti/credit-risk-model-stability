"""Smoke tests: every figure function runs on small inputs and draws what it is asked to draw."""
from datetime import date, timedelta

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src import plots
from src.iv import MISSING_LABEL

MONTHS = [201911, 201912, 202001, 202002, 202003, 202004, 202005, 202006, 202007, 202008]


@pytest.fixture(autouse=True)
def styled_and_closed():
    plots.apply_style()
    yield
    plt.close("all")


def test_month_label():
    assert plots.month_label(201907) == "2019-07"


def test_population_figure_has_volume_above_default_rate():
    volume = pd.DataFrame({"n": [90, 100, 80, 70, 60, 8, 30, 45, 29, 50], "is_oot": [False] * 8 + [True] * 2}, index=MONTHS)
    rate = pd.DataFrame({"default_rate": 0.03, "ci_low": 0.025, "ci_high": 0.035}, index=MONTHS[:8])
    fig = plots.plot_population(volume, rate)
    top, bottom = fig.axes
    assert len(top.patches) == len(MONTHS)  # one bar per month, OOT included
    assert len(bottom.lines[0].get_xdata()) == 8  # the default rate stops before the OOT
    assert "2020-04" in "".join(text.get_text() for text in top.texts)  # the quietest month is labelled


def test_weekly_and_heatmap_figures():
    weekly = pd.DataFrame({"default_rate": 0.03, "ci_low": 0.02, "ci_high": 0.04}, index=range(10))
    assert len(plots.plot_weekly_default_rate(weekly, overall_rate=0.03).axes) == 1

    shares = pd.DataFrame(np.linspace(0, 1, 30).reshape(3, 10), index=["a", "b", "c"], columns=MONTHS)
    axis = plots.plot_share_heatmap(shares, "title").axes[0]
    assert len(axis.texts) == shares.size  # every cell carries its value
    assert [label.get_text() for label in axis.get_xticklabels()][0] == "2019-11"


def test_date_figures():
    counts = pd.Series(range(1, 36), index=range(-20, 15))
    fig = plots.plot_offset_histograms({f"column {index}": counts for index in range(4)}, shift_days=14)
    assert len(fig.axes) == 4

    days = [date(2019, 12, 20) + timedelta(days=offset) for offset in range(37)]
    daily = pd.Series([100] * 37, index=pd.to_datetime(days))
    daily.iloc[12] = 5  # 1 January
    fig = plots.plot_holiday_dips({"first": daily, "second": daily})
    assert len(fig.axes) == 2
    assert all("01/01" in axis.texts[0].get_text() for axis in fig.axes)


def test_feature_figures():
    audit = pd.DataFrame({
        "feature": [f"feature_{index}" for index in range(6)],
        "iv": [0.4, 0.3, 0.2, 0.1, 0.05, 0.01],
        "iv_2019H1": [0.3, 0.3, 0.2, 0.1, 0.05, 0.01],
        "iv_2019H2": [0.5, 0.3, 0.2, 0.1, 0.05, 0.01],
        "decisao": ["manter", "descartar", "manter", "manter", "descartar", "manter"],
    })
    ranking = plots.plot_iv_ranking(audit, top=4, attention_level=0.5).axes[0]
    assert len(ranking.patches) == 4

    by_period = plots.plot_iv_by_period(audit, ["feature_0", "feature_2"], ["2019H1", "2019H2"]).axes[0]
    assert [text.get_text() for text in by_period.get_legend().get_texts()] == ["2019H1", "2019H2"]

    table = pd.DataFrame({"bin": ["(-inf, 0]", "(0, +inf]", MISSING_LABEL], "bad_rate": [0.02, 0.08, 0.03]})
    fig = plots.plot_default_rate_by_bin({"a": table, "b": table, "c": table}, overall_rate=0.03, missing_label=MISSING_LABEL)
    visible = [axis for axis in fig.axes if axis.get_visible()]
    assert len(visible) == 3
    assert [label.get_text() for label in visible[0].get_xticklabels()][-1] == "ausente"
