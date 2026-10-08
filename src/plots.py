"""Shared chart style and the figures of the notebooks, so every chart reads as one system.

Colours follow a validated colour-blind-safe palette: one hue per job (blue for a single series,
blue -> orange -> aqua for categories, a one-hue ramp for magnitude, grey for context). Text always
uses the ink colours, never the series colour; grids are solid hairlines; there is never a second
y-axis (two measures get two panels that share the x-axis).
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

from src.config import OOT, TEST, TRAIN

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
NEUTRAL = "#c3c2b7"  # de-emphasised marks (context, not the story)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SPLIT_COLORS = {TRAIN: BLUE, TEST: ORANGE, OOT: AQUA}
# One hue, light -> dark: magnitude only (heatmaps).
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blue_ramp", ["#f0efec", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)

LINE_WIDTH = 2.0
MARKER_SIZE = 6.0
BAR_WIDTH = 0.62
DARK_CELL = 0.5  # share above which a heatmap cell needs a light label


def apply_style() -> None:
    """Set the matplotlib defaults used by every notebook."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "font.size": 10,
        "text.color": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.labelsize": 9.5,
        "axes.labelcolor": INK_SECONDARY,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "lines.linewidth": LINE_WIDTH,
        "lines.markersize": MARKER_SIZE,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "legend.frameon": False,
        "legend.fontsize": 9,
        "axes.prop_cycle": mpl.cycler(color=[BLUE, ORANGE, AQUA]),
    })


def save_figure(fig: plt.Figure, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)


def month_label(month: int) -> str:
    """201907 -> "2019-07"."""
    return f"{month // 100}-{month % 100:02d}"


def _decimal_comma(text: str) -> str:
    """Brazilian number format for labels: thousands with a dot, decimals with a comma."""
    return text.replace(",", " ").replace(".", ",").replace(" ", ".")


def _percent_axis(axis, decimals: int = 0) -> None:
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: _decimal_comma(f"{value:.{decimals}%}")))


def _count_axis(axis) -> None:
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: _decimal_comma(f"{value:,.0f}")))


# --------------------------------------------------------------------------------------------------
# Notebook 01: population and sources
# --------------------------------------------------------------------------------------------------
def plot_population(volume: pd.DataFrame, default_rate: pd.DataFrame) -> plt.Figure:
    """Cases per month (every split) above the TRAIN default rate per month (development only).

    `volume`: index MONTH, columns `n` and `is_oot`. `default_rate`: index MONTH, columns
    `default_rate`, `ci_low` and `ci_high`.
    """
    months = list(volume.index)
    position = {month: index for index, month in enumerate(months)}
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True, height_ratios=[1, 1.15])

    colors = [SPLIT_COLORS[OOT] if is_oot else BLUE for is_oot in volume["is_oot"]]
    top.bar(range(len(months)), volume["n"], width=BAR_WIDTH, color=colors)
    top.set_title("Casos decididos por mês")
    _count_axis(top)
    top.legend(
        handles=[Patch(color=BLUE, label="Desenvolvimento (TRAIN + TEST)"), Patch(color=SPLIT_COLORS[OOT], label="OOT")],
        loc="upper left", ncols=2,
    )
    # The quietest month gets a label, placed above the bars to its right with a thin leader line.
    smallest = volume.loc[~volume["is_oot"], "n"].idxmin()
    top.annotate(
        f"{month_label(smallest)}: " + _decimal_comma(f"{volume.loc[smallest, 'n']:,}"),
        xy=(position[smallest], volume.loc[smallest, "n"]),
        xytext=(position[smallest] + 0.4, 0.7 * volume["n"].max()),
        arrowprops={"arrowstyle": "-", "color": AXIS, "linewidth": 1}, color=INK_SECONDARY, fontsize=9,
    )

    x = [position[month] for month in default_rate.index]
    bottom.fill_between(x, default_rate["ci_low"], default_rate["ci_high"], color=BLUE, alpha=0.12, linewidth=0)
    bottom.plot(x, default_rate["default_rate"], color=BLUE, marker="o", markeredgecolor=SURFACE, markeredgewidth=1.5)
    bottom.set_title("Taxa de default por mês no TRAIN (faixa: intervalo de 95%)")
    bottom.set_ylim(0, None)
    _percent_axis(bottom, decimals=1)
    oot = [position[month] for month in volume.index[volume["is_oot"]]]
    bottom.axvspan(min(oot) - 0.5, max(oot) + 0.5, color=GRID, alpha=0.6, linewidth=0)
    bottom.text(
        (min(oot) + max(oot)) / 2, 0.5, "OOT\ntarget fechado", transform=bottom.get_xaxis_transform(),
        ha="center", va="center", color=INK_MUTED, fontsize=9,
    )
    ticks = list(range(0, len(months), 3))
    bottom.set_xticks(ticks, [month_label(months[index]) for index in ticks])
    fig.tight_layout()
    return fig


def plot_weekly_default_rate(weekly: pd.DataFrame, overall_rate: float) -> plt.Figure:
    """TRAIN default rate per WEEK_NUM with its 95% interval and the overall rate as reference."""
    fig, axis = plt.subplots(figsize=(10, 3.4))
    axis.fill_between(weekly.index, weekly["ci_low"], weekly["ci_high"], color=BLUE, alpha=0.12, linewidth=0)
    axis.plot(weekly.index, weekly["default_rate"], color=BLUE)
    axis.axhline(overall_rate, color=AXIS, linewidth=1)
    axis.annotate(
        "TRAIN inteiro: " + _decimal_comma(f"{overall_rate:.2%}"), xy=(weekly.index.max(), overall_rate),
        xytext=(0, 5), textcoords="offset points", ha="right", color=INK_SECONDARY, fontsize=9,
    )
    axis.set_title("Taxa de default por semana (WEEK_NUM) no TRAIN (faixa: intervalo de 95%)")
    axis.set_xlabel("WEEK_NUM")
    axis.set_ylim(0, None)
    _percent_axis(axis, decimals=1)
    fig.tight_layout()
    return fig


def plot_share_heatmap(shares: pd.DataFrame, title: str) -> plt.Figure:
    """Heatmap of shares in [0, 1] (rows x months), every cell labelled with its percentage."""
    n_rows, n_columns = shares.shape
    fig, axis = plt.subplots(figsize=(0.52 * n_columns + 2.8, 0.36 * n_rows + 1.4))
    axis.imshow(shares.to_numpy(dtype=float), cmap=SEQUENTIAL, vmin=0, vmax=1, aspect="auto")
    for row in range(n_rows):
        for column in range(n_columns):
            value = shares.iat[row, column]
            axis.text(
                column, row, f"{100 * value:.0f}", ha="center", va="center", fontsize=7.5,
                color=SURFACE if value > DARK_CELL else INK_SECONDARY,
            )
    axis.set_xticks(range(n_columns), [month_label(month) for month in shares.columns], rotation=90)
    axis.set_yticks(range(n_rows), shares.index)
    axis.tick_params(axis="y", labelcolor=INK_SECONDARY)
    axis.grid(False)
    axis.spines["bottom"].set_visible(False)
    axis.set_title(title)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------------------------------
# Notebook 02: dates and features
# --------------------------------------------------------------------------------------------------
def plot_offset_histograms(histograms: dict[str, pd.Series], shift_days: int) -> plt.Figure:
    """Records by `date - date_decision` on the raw calendar, one panel per date column."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 5.6), sharex=True)
    for axis, (title, counts) in zip(axes.flat, histograms.items()):
        axis.bar(counts.index, counts.to_numpy(), width=0.72, color=BLUE)
        for day in (0, shift_days):
            axis.axvline(day, color=AXIS, linewidth=1)
        axis.set_title(title, fontsize=10)
        _count_axis(axis)
    for axis in axes[-1]:
        # The two reference days are named on the axis itself, where no bar can hide them.
        axis.set_xticks([-30, -20, -10, 0, shift_days], ["-30", "-20", "-10", "0\n(decisão)", f"+{shift_days}"])
        axis.set_xlabel("data da coluna - date_decision (dias, como gravado)")
    fig.tight_layout()
    return fig


def plot_holiday_dips(panels: dict[str, pd.Series]) -> plt.Figure:
    """Daily counts around New Year for each series; the quietest day of each one is highlighted."""
    fig, axes = plt.subplots(len(panels), 1, figsize=(10, 2.7 * len(panels)), squeeze=False)
    for axis, (title, counts) in zip(axes.flat, panels.items()):
        quietest = counts.idxmin()
        colors = [BLUE if day == quietest else NEUTRAL for day in counts.index]
        axis.bar(range(len(counts)), counts.to_numpy(), width=BAR_WIDTH, color=colors)
        axis.annotate(
            f"menor dia: {quietest:%d/%m}", xy=(counts.index.get_loc(quietest), counts.min()),
            xytext=(0, 16), textcoords="offset points", ha="center", color=INK, fontsize=9,
        )
        ticks = [index for index, day in enumerate(counts.index) if day.day in (1, 15, 25)]
        axis.set_xticks(ticks, [f"{counts.index[index]:%d/%m/%Y}" for index in ticks])
        axis.set_title(title, fontsize=10)
        _count_axis(axis)
    fig.tight_layout()
    return fig


def plot_iv_ranking(audit: pd.DataFrame, top: int, attention_level: float) -> plt.Figure:
    """The `top` candidates by TRAIN IV; discarded ones are greyed out.

    `audit`: columns `feature`, `iv` and `decisao` ("manter" or "descartar").
    """
    ranked = audit.nlargest(top, "iv").iloc[::-1]
    colors = [BLUE if decision == "manter" else NEUTRAL for decision in ranked["decisao"]]
    fig, axis = plt.subplots(figsize=(8.6, 0.27 * top + 1.2))
    axis.barh(ranked["feature"], ranked["iv"], height=BAR_WIDTH, color=colors)
    axis.axvline(attention_level, color=AXIS, linewidth=1)
    axis.annotate(
        "IV " + _decimal_comma(f"{attention_level:.1f}") + ": investigar", xy=(attention_level, 0),
        xycoords=("data", "axes fraction"), xytext=(-4, 6), textcoords="offset points", ha="right",
        color=INK_SECONDARY, fontsize=8.5,
    )
    axis.set_xlim(0, max(attention_level, ranked["iv"].max()) * 1.08)
    axis.set_title(f"Information Value no TRAIN: as {top} maiores candidatas")
    axis.grid(axis="x")
    axis.grid(axis="y", visible=False)
    axis.tick_params(axis="y", labelcolor=INK_SECONDARY, labelsize=8.5)
    axis.legend(
        handles=[Patch(color=BLUE, label="mantida"), Patch(color=NEUTRAL, label="descartada")],
        loc="lower right", bbox_to_anchor=(1, 0.08),
    )
    fig.tight_layout()
    return fig


def plot_iv_by_period(audit: pd.DataFrame, features: list[str], periods: list[str]) -> plt.Figure:
    """IV of each feature inside each half-year of TRAIN (bins fixed on the whole TRAIN)."""
    table = audit.set_index("feature").loc[features[::-1]]
    columns = [f"iv_{period}" for period in periods]
    fig, axis = plt.subplots(figsize=(8.6, 0.3 * len(features) + 1.4))
    axis.hlines(table.index, table[columns].min(axis=1), table[columns].max(axis=1), color=AXIS, linewidth=1.2, zorder=1)
    for column, period, color in zip(columns, periods, (BLUE, ORANGE, AQUA)):
        axis.scatter(table[column], table.index, s=46, color=color, edgecolor=SURFACE, linewidth=1.3, label=period, zorder=2)
    axis.set_xlim(0, None)
    axis.set_title("O sinal se mantém? IV por semestre do TRAIN")
    axis.grid(axis="x")
    axis.grid(axis="y", visible=False)
    axis.tick_params(axis="y", labelcolor=INK_SECONDARY, labelsize=8.5)
    axis.legend(loc="lower right", ncols=len(periods))
    fig.tight_layout()
    return fig


def plot_default_rate_by_bin(tables: dict[str, pd.DataFrame], overall_rate: float, missing_label: str) -> plt.Figure:
    """Default rate in each bin of a feature (TRAIN); the missing bin is greyed, as it has no order."""
    n_columns = 2
    n_rows = (len(tables) + 1) // n_columns
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(10, 3.6 * n_rows), sharey=True, squeeze=False)
    for axis, (feature, table) in zip(axes.flat, tables.items()):
        colors = [NEUTRAL if label == missing_label else BLUE for label in table["bin"]]
        axis.bar(range(len(table)), table["bad_rate"], width=BAR_WIDTH, color=colors)
        axis.axhline(overall_rate, color=AXIS, linewidth=1)
        labels = ["ausente" if label == missing_label else label for label in table["bin"]]
        axis.set_xticks(range(len(table)), labels, rotation=40, ha="right", fontsize=8)
        axis.set_title(feature, fontsize=10)
        _percent_axis(axis)
    for axis in list(axes.flat)[len(tables):]:
        axis.set_visible(False)
    axes[0, 0].annotate(
        "TRAIN inteiro: " + _decimal_comma(f"{overall_rate:.1%}"), xy=(1, overall_rate),
        xycoords=("axes fraction", "data"), xytext=(-4, 4), textcoords="offset points", ha="right",
        color=INK_SECONDARY, fontsize=8.5,
    )
    fig.suptitle("Taxa de default por faixa da feature (TRAIN)", x=0.01, ha="left", fontsize=11, fontweight="semibold")
    fig.tight_layout()
    return fig
