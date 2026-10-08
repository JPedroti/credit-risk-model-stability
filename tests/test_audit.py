from datetime import date

import numpy as np
import pandas as pd
import polars as pl
import pytest

from src.audit import (
    QUALITY_CHUNK, column_quality, coverage_by_month, daily_counts, date_audit, default_rate_by, feature_diagnostics,
    half_year, offset_histogram, quarter, redundant_pairs, source_inventory, wilson_interval, year_month_offset,
)
from src.config import CASE_ID, DATE_SHIFT_DAYS, MISSING_CATEGORY, MONTH, TARGET
from src.data import SOURCES
from tests.synthetic import raw_date, raw_year_month

HISTORIES = ["applprev_1", "credit_bureau_a_1", "tax_registry_a_1"]


def test_default_rate_comes_with_a_wilson_interval(base):
    table = default_rate_by(base, MONTH, TARGET)
    assert table["n"].sum() == base.height and table["defaults"].sum() == base[TARGET].sum()
    assert table.index.is_monotonic_increasing
    assert ((table["ci_low"] < table["default_rate"]) & (table["default_rate"] < table["ci_high"])).all()

    # Reference value: 10 events in 100 trials has the 95% Wilson interval [0.0552, 0.1744].
    low, high = wilson_interval(pd.Series([10, 0]), pd.Series([100, 50]))
    assert (low[0], high[0]) == pytest.approx((0.0552, 0.1744), abs=1e-4)
    assert low[1] == pytest.approx(0) and 0 < high[1] < 0.1  # zero events still gets an upper bound


def test_inventory_describes_every_source(small_base, raw_dir, raw_tables):
    inventory = source_inventory(small_base, raw_dir).set_index("fonte")
    assert list(inventory.index) == list(SOURCES)
    assert inventory.loc["static_0", "cobertura"] == 1 and inventory.loc["static_0", "registros_por_case_max"] == 1
    assert inventory.loc["applprev_1", "linhas"] == raw_tables["applprev_1"].height
    assert inventory.loc["applprev_1", "cases_unicos"] == raw_tables["applprev_1"][CASE_ID].n_unique()
    assert 0 < inventory.loc["static_cb_0", "cobertura"] < 1
    assert inventory.loc["credit_bureau_a_2", "depth"] == 2


def test_coverage_by_month_is_a_share_of_the_cases_of_the_month(small_base, raw_dir):
    coverage = coverage_by_month(small_base, raw_dir)
    assert list(coverage.columns) == ["n_cases", *SOURCES]
    assert coverage["n_cases"].sum() == small_base.height
    assert (coverage["static_0"] == 1).all() and coverage[list(SOURCES)].max().max() <= 1


def test_column_quality_flags_constant_empty_and_regime_columns():
    n_rows = 400
    cases = pl.DataFrame({"case_id": np.arange(n_rows), MONTH: np.repeat([201901, 201902], n_rows // 2)})
    records = pl.DataFrame({
        "case_id": np.arange(n_rows),
        "fine_1A": np.arange(n_rows, dtype=float),
        "constant_2L": ["same"] * n_rows,
        "almost_empty_3A": [1.0] * 4 + [None] * (n_rows - 4),
        "placeholder_4M": [MISSING_CATEGORY, MISSING_CATEGORY, MISSING_CATEGORY, "P1", MISSING_CATEGORY, MISSING_CATEGORY, MISSING_CATEGORY, "P2"] * (n_rows // 8),
        "born_later_5A": [None] * (n_rows // 2) + [float(value) for value in range(n_rows // 2)],
    })
    # Two rows per case (a history table) plus cases outside the population, which must be ignored.
    outside = records.with_columns(pl.col("case_id") + 10_000, pl.lit("other").alias("constant_2L"))
    quality = column_quality(pl.concat([records, records, outside]).lazy(), cases).set_index("coluna")

    assert quality.loc["fine_1A", "missing"] == 0 and pd.isna(quality.loc["fine_1A", "niveis_categoricos"])
    assert bool(quality.loc["constant_2L", "constante"]) and not bool(quality.loc["fine_1A", "constante"])
    assert bool(quality.loc["almost_empty_3A", "quase_vazia"]) and bool(quality.loc["almost_empty_3A", "constante"])
    assert quality.loc["placeholder_4M", "missing"] == 0.75 and quality.loc["placeholder_4M", "niveis_categoricos"] == 2
    assert not bool(quality.loc["placeholder_4M", "constante"]) and quality.loc["constant_2L", "niveis_categoricos"] == 1
    assert quality.index[quality["muda_de_regime"]].tolist() == ["born_later_5A"]
    assert quality.loc["born_later_5A", ["preenchimento_min_mes", "preenchimento_max_mes"]].tolist() == [0, 1]

    # A table wider than one pass gives the same answer, column by column, in the original order.
    wide = records.with_columns([pl.col("fine_1A").alias(f"copy_{index}A") for index in range(QUALITY_CHUNK)])
    wide_quality = column_quality(wide.lazy(), cases)
    assert wide_quality["coluna"].tolist() == wide.columns[1:]
    assert wide_quality.set_index("coluna").loc[quality.index].equals(column_quality(records.lazy(), cases).set_index("coluna"))


def test_date_audit_separates_the_calendar_offset_from_real_future_records(small_base, raw_dir, raw_dir_with_future):
    clean = date_audit(small_base, HISTORIES, raw_dir).set_index("coluna")
    # Read as stored, recent events look like the future, but never more than the offset...
    assert clean.loc["creationdate_885D", "apos_decisao_bruto"] > 0
    assert clean["maior_avanco_dias"].max() <= DATE_SHIFT_DAYS
    # ...and once the offset is removed nothing is after the decision.
    assert (clean["n_apos_corrigido"] == 0).all()

    with_future = date_audit(small_base, HISTORIES, raw_dir_with_future).set_index("coluna")
    for column in ["creationdate_885D", "dateofcredstart_739D", "recorddate_4527225D"]:
        assert with_future.loc[column, "n_apos_corrigido"] > 0
        assert with_future.loc[column, "maior_avanco_dias"] > DATE_SHIFT_DAYS


def test_offset_histogram_counts_records_by_distance_to_the_decision(small_base, raw_dir, raw_tables):
    histogram = offset_histogram(small_base, "applprev_1", "creationdate_885D", low=-40, high=DATE_SHIFT_DAYS, directory=raw_dir)
    offsets = raw_tables["applprev_1"].join(small_base, on=CASE_ID).select(
        (pl.col("creationdate_885D") - pl.col("date_decision")).dt.total_days()
    ).to_series()
    assert histogram.sum() == offsets.is_between(-40, DATE_SHIFT_DAYS).sum()
    assert histogram.index.max() <= DATE_SHIFT_DAYS


def test_year_month_fields_are_one_month_ahead_of_the_corrected_dates():
    # The same events stored the way the organisers stored them: days +14, (year, month) +1 month.
    events = [date(2019, 1, 5), date(2019, 1, 25), date(2019, 12, 31), date(2020, 2, 18)]
    labels = [raw_year_month(day) for day in events]
    records = pl.DataFrame({
        "when_1D": [raw_date(day) for day in events],
        "year_2T": [year for year, _ in labels],
        "month_3T": [month for _, month in labels],
    }).lazy()

    corrected = year_month_offset(records, "year_2T", "month_3T", "when_1D", shift_days=DATE_SHIFT_DAYS)
    assert corrected.to_dict() == {1: 1.0}
    # Without removing the 14 days, the events of the second half of a month look aligned (offset 0).
    as_stored = year_month_offset(records, "year_2T", "month_3T", "when_1D", shift_days=0)
    assert as_stored.to_dict() == {0: 0.75, 1: 0.25}


def test_daily_counts_fill_days_without_values():
    dates = pl.Series("created", [date(2020, 1, 1), date(2020, 1, 1), date(2020, 1, 3), date(2021, 5, 5)])
    counts = daily_counts(dates, date(2019, 12, 31), date(2020, 1, 3))
    assert counts.tolist() == [0, 2, 0, 1]
    assert [day.isoformat()[:10] for day in counts.index] == ["2019-12-31", "2020-01-01", "2020-01-02", "2020-01-03"]


def test_half_year_and_quarter_labels():
    months = pl.Series(MONTH, [201901, 201903, 201904, 201906, 201907, 202012])
    assert half_year(months).to_list() == ["2019H1", "2019H1", "2019H1", "2019H1", "2019H2", "2020H2"]
    assert quarter(months).to_list() == ["2019Q1", "2019Q1", "2019Q2", "2019Q2", "2019Q3", "2020Q4"]


def test_feature_diagnostics_and_redundancy(small_base):
    rng = np.random.default_rng(0)
    n_rows = small_base.height
    signal = small_base[TARGET].to_numpy() + rng.normal(scale=0.5, size=n_rows)
    train = small_base.with_columns(
        pl.Series("signal", signal),
        pl.Series("signal_copy", signal * 3 + 1),
        pl.Series("noise", rng.normal(size=n_rows)),
        pl.when(pl.col(MONTH) < 201907).then(None).otherwise(pl.lit("level")).alias("born_later"),
    )
    features = ["signal", "signal_copy", "noise", "born_later"]
    diagnostics = feature_diagnostics(train, features, TARGET).set_index("feature")

    assert list(diagnostics.index) == features
    assert diagnostics.loc["signal", "iv"] > 1 > diagnostics.loc["noise", "iv"]
    assert diagnostics.loc["signal", "iv"] == pytest.approx(diagnostics.loc["signal_copy", "iv"])
    assert {"iv_2019H1", "iv_2019H2", "iv_2020H1", "iv_2020H2"} <= set(diagnostics.columns)
    assert diagnostics.loc["born_later", "fill_min_month"] == 0 and diagnostics.loc["born_later", "fill_max_month"] == 1
    assert diagnostics.loc["signal", "fill_min_month"] == 1
    # The feature that appears in 2019H2 is the unstable one; its worst quarter is one without values.
    assert diagnostics.loc["born_later", "psi_max"] > 0.25 > diagnostics.loc["noise", "psi_max"]
    assert diagnostics.loc["born_later", "psi_max_quarter"] in ("2019Q1", "2019Q2")

    pairs = redundant_pairs(train, features, threshold=0.9, seed=0, min_pairs=100)
    assert pairs[["feature_a", "feature_b"]].values.tolist() == [["signal", "signal_copy"]]
    assert pairs.loc[0, "spearman_abs"] == pytest.approx(1)
    # Too few rows observed together: nothing is reported, and the empty table keeps its columns.
    assert redundant_pairs(train, features, threshold=0.9, seed=0).columns.tolist() == ["feature_a", "feature_b", "spearman_abs"]
