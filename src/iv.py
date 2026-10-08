"""Weight of Evidence and Information Value, used as a diagnostic (brief, section 7.4).

Conventions, fixed for the whole project:

- "bad" is `target == 1` (default) and "good" is `target == 0`.
- WoE_bin = ln(share of goods in the bin / share of bads in the bin); positive WoE = safer than average.
- IV = sum over bins of (share of goods - share of bads) * WoE.
- Numeric features are cut at quantiles; categorical features use one bin per level, with levels
  below `min_category_share` of the rows pooled into `OTHER_LABEL`. Missing values always get
  their own bin, so the IV also captures "informative missingness". `iv_summary` reports that part
  separately, as the IV of the two-bin split {missing, observed}.
- Shares are smoothed by adding `SMOOTHING` to every cell count, which keeps the WoE finite when
  a bin has no goods or no bads. With bins of thousands of rows the effect is negligible.

Bins are learned by `fit_binning` on TRAIN and only *applied* elsewhere, so no TEST/OOT target
ever takes part in a decision.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl

N_BINS = 10
MIN_CATEGORY_SHARE = 0.005
SMOOTHING = 0.5
MISSING_LABEL = "missing"
OTHER_LABEL = "other"


@dataclass(frozen=True)
class Binning:
    """Bin definition learned on one sample and reusable on any other."""

    kind: str  # "numeric" or "categorical"
    edges: tuple[float, ...] = ()  # numeric: interior cut points, ascending
    categories: tuple[str, ...] = ()  # categorical: levels that keep their own bin

    @property
    def labels(self) -> list[str]:
        if self.kind == "numeric":
            bounds = ["-inf", *(f"{edge:.4g}" for edge in self.edges), "+inf"]
            intervals = [f"({low}, {high}]" for low, high in zip(bounds[:-1], bounds[1:])]
            return [*intervals, MISSING_LABEL]
        return [*self.categories, OTHER_LABEL, MISSING_LABEL]


def _is_numeric(values: pl.Series) -> bool:
    return values.dtype.is_numeric()


def fit_binning(
    values: pl.Series, *, n_bins: int = N_BINS, min_category_share: float = MIN_CATEGORY_SHARE
) -> Binning:
    """Learn quantile cut points (numeric) or the levels kept apart (anything else)."""
    if _is_numeric(values):
        observed = values.drop_nulls().cast(pl.Float64).to_numpy()
        observed = observed[~np.isnan(observed)]
        if observed.size == 0:
            return Binning(kind="numeric")
        quantiles = np.quantile(observed, np.linspace(0, 1, n_bins + 1)[1:-1])
        # Heavily tied features (e.g. mostly zeros) repeat quantiles; a cut at the maximum would
        # leave an empty last bin.
        edges = np.unique(quantiles)
        edges = edges[edges < observed.max()]
        return Binning(kind="numeric", edges=tuple(float(edge) for edge in edges))

    counts = values.drop_nulls().cast(pl.String).value_counts(sort=True)
    name, count = counts.columns
    kept = counts.filter(pl.col(count) >= min_category_share * len(values))[name].to_list()
    return Binning(kind="categorical", categories=tuple(sorted(kept)))


def apply_binning(values: pl.Series, binning: Binning) -> np.ndarray:
    """Map each value to the index of its bin in `binning.labels`."""
    n_labels = len(binning.labels)
    if binning.kind == "numeric":
        numbers = values.cast(pl.Float64).fill_null(np.nan).to_numpy()
        codes = np.searchsorted(np.asarray(binning.edges), numbers, side="left")
        codes[np.isnan(numbers)] = n_labels - 1
        return codes

    position = {category: index for index, category in enumerate(binning.categories)}
    other, missing = n_labels - 2, n_labels - 1
    levels = values.cast(pl.String)
    codes = levels.replace_strict(position, default=other, return_dtype=pl.Int64).to_numpy().copy()
    codes[levels.is_null().to_numpy()] = missing  # `default` also catches nulls, so set them last
    return codes


def _woe_and_iv(good: np.ndarray, bad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Smoothed WoE and IV contribution of each bin, from its counts of goods and bads."""
    n_bins = len(good)
    share_good = (good + SMOOTHING) / (good.sum() + SMOOTHING * n_bins)
    share_bad = (bad + SMOOTHING) / (bad.sum() + SMOOTHING * n_bins)
    woe = np.log(share_good / share_bad)
    return woe, (share_good - share_bad) * woe


def woe_table(codes: np.ndarray, target: np.ndarray, labels: list[str]) -> pd.DataFrame:
    """Per-bin counts, bad rate, WoE and IV contribution. Empty bins are dropped."""
    target = np.asarray(target)
    n_labels = len(labels)
    total = np.bincount(codes, minlength=n_labels)
    bad = np.bincount(codes, weights=target, minlength=n_labels)
    good = total - bad

    table = pd.DataFrame({"bin": labels, "n": total, "bad": bad.astype(int), "good": good.astype(int)})
    table = table[table["n"] > 0].reset_index(drop=True)
    table["share"] = table["n"] / table["n"].sum()
    table["bad_rate"] = table["bad"] / table["n"]
    table["woe"], table["iv"] = _woe_and_iv(table["good"].to_numpy(), table["bad"].to_numpy())
    return table


def missingness_iv(table: pd.DataFrame) -> float:
    """IV of the two-bin split {missing, observed}: the signal of "has a value or not" alone."""
    is_missing = (table["bin"] == MISSING_LABEL).to_numpy()
    if is_missing.all() or not is_missing.any():
        return 0.0
    good = np.array([table.loc[is_missing, "good"].sum(), table.loc[~is_missing, "good"].sum()])
    bad = np.array([table.loc[is_missing, "bad"].sum(), table.loc[~is_missing, "bad"].sum()])
    return float(_woe_and_iv(good, bad)[1].sum())


def information_value(
    values: pl.Series, target: np.ndarray, *, binning: Binning | None = None
) -> tuple[float, pd.DataFrame]:
    """IV of one feature and its WoE table. `binning` defaults to bins learned on `values`."""
    binning = binning or fit_binning(values)
    table = woe_table(apply_binning(values, binning), target, binning.labels)
    return float(table["iv"].sum()), table


def iv_summary(frame: pl.DataFrame, features: list[str], target: np.ndarray) -> pd.DataFrame:
    """One row per feature: IV, number of bins, missing share and the IV of missingness alone."""
    rows = []
    for feature in features:
        iv, table = information_value(frame[feature], target)
        rows.append({
            "feature": feature,
            "iv": iv,
            "n_bins": len(table),
            "missing_share": float(table.loc[table["bin"] == MISSING_LABEL, "share"].sum()),
            "iv_missingness": missingness_iv(table),
        })
    return pd.DataFrame(rows)


def iv_by_period(
    frame: pl.DataFrame, features: list[str], target: np.ndarray, period: np.ndarray
) -> pd.DataFrame:
    """IV of each feature inside each period, with bins fixed on the whole `frame`.

    Answers "does the strength stay similar across periods?" (brief, 7.4, question 5). Long format:
    one row per (feature, period).
    """
    target = np.asarray(target)
    periods, period_index = np.unique(np.asarray(period), return_inverse=True)
    masks = [period_index == index for index in range(len(periods))]
    rows = []
    for feature in features:
        binning = fit_binning(frame[feature])
        codes = apply_binning(frame[feature], binning)
        for value, mask in zip(periods, masks):
            table = woe_table(codes[mask], target[mask], binning.labels)
            rows.append({"feature": feature, "period": value, "iv": float(table["iv"].sum())})
    return pd.DataFrame(rows)


def psi_by_period(frame: pl.DataFrame, features: list[str], period: np.ndarray) -> pd.DataFrame:
    """Population Stability Index of each feature in each period against the whole `frame`.

    PSI = sum over bins of (share in the period - share overall) * ln(ratio of the two), with the
    bins fixed on the whole frame and the missing values in their own bin, so a change in
    availability counts as a shift. Long format: one row per (feature, period).
    """
    periods, period_index = np.unique(np.asarray(period), return_inverse=True)
    rows = []
    for feature in features:
        binning = fit_binning(frame[feature])
        codes = apply_binning(frame[feature], binning)
        n_bins = len(binning.labels)
        overall = np.bincount(codes, minlength=n_bins) + SMOOTHING
        overall = overall / overall.sum()
        for index, value in enumerate(periods):
            inside = np.bincount(codes[period_index == index], minlength=n_bins) + SMOOTHING
            inside = inside / inside.sum()
            rows.append({"feature": feature, "period": value, "psi": float(((inside - overall) * np.log(inside / overall)).sum())})
    return pd.DataFrame(rows)
