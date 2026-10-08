import numpy as np
import polars as pl
import pytest

from src.iv import (
    MISSING_LABEL, N_BINS, OTHER_LABEL, SMOOTHING, apply_binning, fit_binning, information_value,
    iv_by_period, iv_summary, psi_by_period,
)


def test_iv_matches_the_formula_computed_by_hand():
    # Level "a": 80 goods / 20 bads; level "b": 60 goods / 40 bads.
    values = pl.Series(["a"] * 100 + ["b"] * 100)
    target = np.array([0] * 80 + [1] * 20 + [0] * 60 + [1] * 40)

    good, bad = np.array([80, 60]), np.array([20, 40])
    share_good = (good + SMOOTHING) / (good.sum() + 2 * SMOOTHING)
    share_bad = (bad + SMOOTHING) / (bad.sum() + 2 * SMOOTHING)
    expected = float(((share_good - share_bad) * np.log(share_good / share_bad)).sum())

    iv, table = information_value(values, target)
    assert iv == pytest.approx(expected)
    assert table["bin"].tolist() == ["a", "b"]
    assert table["bad_rate"].tolist() == [0.2, 0.4]
    assert table.loc[0, "woe"] > 0 > table.loc[1, "woe"]  # positive WoE = safer than average


def test_noise_has_no_information_and_signal_has():
    rng = np.random.default_rng(1)
    n_rows = 100_000
    target = (rng.random(n_rows) < 0.05).astype(int)
    noise = pl.Series(rng.normal(size=n_rows))
    signal = pl.Series(target + rng.normal(size=n_rows))
    assert information_value(noise, target)[0] < 0.01
    assert information_value(signal, target)[0] > 0.3


def test_perfect_separation_stays_finite():
    target = np.array([0] * 500 + [1] * 500)
    iv, table = information_value(pl.Series(target.astype(float)), target)
    assert np.isfinite(iv) and iv > 5
    assert np.isfinite(table["woe"]).all()


def test_quantile_bins_are_invariant_to_monotone_transformations():
    rng = np.random.default_rng(2)
    values = rng.normal(size=20_000)
    target = (rng.random(20_000) < 1 / (1 + np.exp(-values - 2))).astype(int)
    iv_raw, table = information_value(pl.Series(values), target)
    iv_transformed, _ = information_value(pl.Series(np.exp(values)), target)
    assert iv_raw == pytest.approx(iv_transformed)
    assert len(table) == N_BINS
    assert table["share"].to_numpy() == pytest.approx(1 / N_BINS, abs=1e-3)


def test_missing_values_get_their_own_bin():
    rng = np.random.default_rng(3)
    n_rows = 20_000
    target = (rng.random(n_rows) < 0.10).astype(int)
    # Missingness is informative: defaults are missing three times as often.
    is_missing = rng.random(n_rows) < np.where(target == 1, 0.6, 0.2)
    values = pl.Series(np.where(is_missing, np.nan, rng.normal(size=n_rows))).fill_nan(None)

    frame = pl.DataFrame({"feature": values, "complete": rng.normal(size=n_rows)})
    summary = iv_summary(frame, ["feature", "complete"], target).set_index("feature")
    assert summary.loc["feature", "missing_share"] == pytest.approx(is_missing.mean())
    # The observed values are noise, so nearly all the IV is "has a value or not".
    assert summary.loc["feature", "iv_missingness"] / summary.loc["feature", "iv"] > 0.9
    # Population value of that two-bin IV: P(missing | bad) = 0.6, P(missing | good) = 0.2.
    expected = (0.2 - 0.6) * np.log(0.2 / 0.6) + (0.8 - 0.4) * np.log(0.8 / 0.4)
    assert summary.loc["feature", "iv_missingness"] == pytest.approx(expected, rel=0.1)
    assert summary.loc["complete", "iv_missingness"] == 0 and summary.loc["complete", "missing_share"] == 0


def test_zero_inflated_feature_collapses_repeated_quantiles():
    rng = np.random.default_rng(4)
    values = np.where(rng.random(10_000) < 0.85, 0.0, rng.exponential(size=10_000))
    binning = fit_binning(pl.Series(values))
    assert 0 < len(binning.edges) < N_BINS - 1
    assert len(set(binning.edges)) == len(binning.edges)
    codes = apply_binning(pl.Series(values), binning)
    assert set(np.unique(codes)) == set(range(len(binning.edges) + 1))  # no empty bin, nothing missing


def test_binning_learned_on_train_is_only_applied_elsewhere():
    train = pl.Series(np.arange(1_000, dtype=float))
    binning = fit_binning(train)
    other = pl.Series([-50.0, 5_000.0, None])
    codes = apply_binning(other, binning)
    labels = binning.labels
    assert labels[codes[0]].startswith("(-inf") and labels[codes[1]].endswith("+inf]")
    assert labels[codes[2]] == MISSING_LABEL


def test_rare_and_unseen_categories_are_pooled():
    train = pl.Series(["a"] * 600 + ["b"] * 398 + ["rare"] * 2)
    binning = fit_binning(train)
    assert binning.categories == ("a", "b")
    labels = binning.labels
    codes = apply_binning(pl.Series(["a", "rare", "unseen", None]), binning)
    assert [labels[code] for code in codes] == ["a", OTHER_LABEL, OTHER_LABEL, MISSING_LABEL]


def test_boolean_and_all_missing_features_are_supported():
    target = np.array([0, 0, 1, 1] * 50)
    iv_bool, table = information_value(pl.Series([True, True, False, False] * 50), target)
    assert iv_bool > 1 and table["bin"].tolist() == ["false", "true"]
    iv_empty, table = information_value(pl.Series([None] * 200, dtype=pl.Float64), target)
    assert iv_empty == pytest.approx(0) and table["bin"].tolist() == [MISSING_LABEL]


def test_iv_by_period_uses_bins_fixed_on_the_whole_sample():
    rng = np.random.default_rng(5)
    n_rows = 30_000
    period = np.repeat(["2019H1", "2019H2", "2020H1"], n_rows // 3)
    values = rng.normal(size=n_rows)
    # The signal exists only in the first period.
    risk = np.where(period == "2019H1", 1 / (1 + np.exp(-2 * values - 2)), 0.1)
    target = (rng.random(n_rows) < risk).astype(int)
    frame = pl.DataFrame({"feature": values})

    by_period = iv_by_period(frame, ["feature"], target, period).set_index("period")["iv"]
    # Pure noise still has a small positive IV in 10,000 rows (sampling bias of the estimator).
    assert by_period["2019H1"] > 0.5 and by_period["2019H2"] < 0.05 and by_period["2020H1"] < 0.05

    binning = fit_binning(frame["feature"])
    mask = period == "2019H2"
    expected, _ = information_value(frame["feature"].filter(pl.Series(mask)), target[mask], binning=binning)
    assert by_period["2019H2"] == pytest.approx(expected)


def test_psi_flags_shifts_in_values_and_in_availability():
    rng = np.random.default_rng(6)
    n_rows = 40_000
    period = np.repeat(["Q1", "Q2", "Q3", "Q4"], n_rows // 4)
    first = period == "Q1"
    stable = rng.normal(size=n_rows)
    frame = pl.DataFrame({
        "stable": stable,
        "shifted": stable + np.where(first, 1.5, 0.0),
        "born_later": pl.Series(np.where(first, np.nan, stable)).fill_nan(None),
    })
    psi = psi_by_period(frame, ["stable", "shifted", "born_later"], period).pivot(index="feature", columns="period", values="psi")

    assert (psi.loc["stable"] < 0.01).all()
    assert psi.loc["shifted", "Q1"] > 0.5 > psi.loc["shifted", "Q2"]  # the quarter that moved stands out
    assert psi.loc["born_later", "Q1"] > 1  # a feature that did not exist yet is the largest shift of all
