"""Diagnostics that support the decisions of the first week: source inventory, column quality,
the audit of every date against `date_decision`, and the per-feature report computed on TRAIN.

Nothing here learns a parameter that is later applied to data. Functions that involve the target
or feature distributions are meant to receive TRAIN rows only; the callers enforce that.
"""
from datetime import date
from pathlib import Path

import pandas as pd
import polars as pl

from src.config import CASE_ID, DATE_DECISION, DATE_SHIFT_DAYS, MISSING_CATEGORY, MONTH, TRAIN_PARQUET_DIR
from src.data import SOURCES, cases_with_records, date_columns, scan_source
from src.iv import iv_by_period, iv_summary, psi_by_period
from src.qa import ALMOST_EMPTY_SHARE

CORRELATION_SAMPLE = 200_000
CORRELATION_MIN_PAIRS = 5_000
QUALITY_CHUNK = 40  # raw columns profiled per pass in `column_quality`


Z_95 = 1.96


# --------------------------------------------------------------------------------------------------
# Population
# --------------------------------------------------------------------------------------------------
def wilson_interval(events: pd.Series, total: pd.Series, z: float = Z_95) -> tuple[pd.Series, pd.Series]:
    """Wilson score interval for a proportion; well behaved for rare events and small groups."""
    rate = events / total
    centre = (rate + z**2 / (2 * total)) / (1 + z**2 / total)
    half_width = z * ((rate * (1 - rate) / total + z**2 / (4 * total**2)) ** 0.5) / (1 + z**2 / total)
    return centre - half_width, centre + half_width


def default_rate_by(cases: pl.DataFrame, by: str, target: str) -> pd.DataFrame:
    """Cases, defaults and default rate (with a 95% interval) per value of `by`."""
    table = (
        cases.group_by(by)
        .agg(pl.len().alias("n"), pl.col(target).sum().alias("defaults"))
        .sort(by)
        .to_pandas()
        .set_index(by)
    )
    table["default_rate"] = table["defaults"] / table["n"]
    table["ci_low"], table["ci_high"] = wilson_interval(table["defaults"], table["n"])
    return table


# --------------------------------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------------------------------
def source_inventory(base: pl.DataFrame, directory: Path = TRAIN_PARQUET_DIR) -> pd.DataFrame:
    """Size and granularity of each source: rows, distinct cases, coverage of the population."""
    rows = []
    for name, source in SOURCES.items():
        cases = cases_with_records(name, directory)
        n_columns = len(scan_source(name, directory).collect_schema())
        rows.append({
            "fonte": name,
            "depth": source.depth,
            "origem": source.origin,
            "linhas": int(cases["n_records"].sum()),
            "cases_unicos": cases.height,
            "cobertura": cases.height / base.height,
            "n_colunas": n_columns,
            "registros_por_case_mediana": float(cases["n_records"].median()),
            "registros_por_case_max": int(cases["n_records"].max()),
        })
    return pd.DataFrame(rows)


def coverage_by_month(cases: pl.DataFrame, directory: Path = TRAIN_PARQUET_DIR) -> pd.DataFrame:
    """Share of the cases of each month that have at least one record in each source."""
    coverage = cases.group_by(MONTH).agg(pl.len().alias("n_cases")).sort(MONTH)
    for name in SOURCES:
        present = cases_with_records(name, directory).select(CASE_ID, pl.lit(True).alias(name))
        by_month = (
            cases.join(present, on=CASE_ID, how="left")
            .group_by(MONTH)
            .agg(pl.col(name).is_not_null().mean())
        )
        coverage = coverage.join(by_month, on=MONTH, how="left")
    return coverage.to_pandas().set_index(MONTH)


def column_quality(records: pl.LazyFrame, cases: pl.DataFrame) -> pd.DataFrame:
    """Per raw column of `records`, for the given `cases` (`case_id` and `MONTH`): missing share
    (overall and its range across months), constant or not, and the levels of the categorical ones.

    The placeholder of masked categories counts as missing. A column whose fill goes to zero in
    some month but not in others changed regime inside the period.

    To keep memory bounded on wide or very long tables, the columns are processed a few at a time
    and each pass first reduces the table to one row per case, which is then joined to the months.
    """
    schema = records.collect_schema()
    columns = [column for column in schema if column not in (CASE_ID, MONTH, DATE_DECISION)]
    population = (
        records.join(cases.lazy().select(CASE_ID), on=CASE_ID, how="semi")
        .with_columns(pl.col(pl.String).replace(MISSING_CATEGORY, None))
    )
    months = cases.select(CASE_ID, MONTH)

    rows = []
    for position in range(0, len(columns), QUALITY_CHUNK):
        chunk = columns[position:position + QUALITY_CHUNK]
        categorical = [column for column in chunk if schema[column] in (pl.String, pl.Boolean)]
        per_case = population.group_by(CASE_ID).agg(
            pl.len().alias("n_rows"),
            *[pl.col(column).count().alias(f"{column}|filled") for column in chunk],
            *[pl.col(column).min().alias(f"{column}|min") for column in chunk],
            *[pl.col(column).max().alias(f"{column}|max") for column in chunk],
        ).collect(engine="streaming")
        monthly = per_case.join(months, on=CASE_ID, how="inner").group_by(MONTH).agg(
            pl.col("n_rows").sum(),
            *[pl.col(f"{column}|filled").sum() for column in chunk],
            *[pl.col(f"{column}|min").min() for column in chunk],
            *[pl.col(f"{column}|max").max() for column in chunk],
        )
        levels = population.select(
            [pl.col(column).drop_nulls().n_unique() for column in categorical]
        ).collect(engine="streaming").row(0, named=True) if categorical else {}

        n_rows = monthly["n_rows"].sum()
        for column in chunk:
            filled = monthly[f"{column}|filled"]
            monthly_fill = filled / monthly["n_rows"]
            rows.append({
                "coluna": column,
                "missing": 1 - filled.sum() / n_rows,
                "niveis_categoricos": levels.get(column),
                "preenchimento_min_mes": monthly_fill.min(),
                "preenchimento_max_mes": monthly_fill.max(),
                # At most one observed value over the whole period.
                "constante": filled.sum() == 0 or monthly[f"{column}|min"].min() == monthly[f"{column}|max"].max(),
            })
    quality = pd.DataFrame(rows)
    quality["quase_vazia"] = quality["missing"] >= ALMOST_EMPTY_SHARE
    quality["muda_de_regime"] = (quality["preenchimento_min_mes"] == 0) & (quality["preenchimento_max_mes"] > 0.05)
    return quality


# --------------------------------------------------------------------------------------------------
# Dates against the point of observation
# --------------------------------------------------------------------------------------------------
def date_audit(cases: pl.DataFrame, sources: list[str], directory: Path = TRAIN_PARQUET_DIR) -> pd.DataFrame:
    """Every date column of `sources` compared with `date_decision`, for the given cases.

    `apos_decisao_bruto` reads the dates as stored; `apos_decisao_corrigido` first removes the
    offset of the raw calendar (`DATE_SHIFT_DAYS`).
    """
    decisions = cases.lazy().select(CASE_ID, DATE_DECISION)
    rows = []
    for name in sources:
        table = scan_source(name, directory)
        columns = date_columns(table)
        if not columns:
            continue
        joined = table.select(CASE_ID, *columns).join(decisions, on=CASE_ID, how="inner")
        aggregations = []
        for column in columns:
            offset = (pl.col(column) - pl.col(DATE_DECISION)).dt.total_days()
            aggregations += [
                pl.col(column).count().alias(f"{column}|n"),
                (offset > 0).sum().alias(f"{column}|after_raw"),
                (offset > DATE_SHIFT_DAYS).sum().alias(f"{column}|after_corrected"),
                offset.max().alias(f"{column}|max_offset"),
            ]
        result = joined.select(aggregations).collect(engine="streaming").row(0, named=True)
        for column in columns:
            n_values = result[f"{column}|n"]
            rows.append({
                "fonte": name,
                "coluna": column,
                "n_valores": n_values,
                "apos_decisao_bruto": result[f"{column}|after_raw"] / n_values if n_values else None,
                "maior_avanco_dias": result[f"{column}|max_offset"],
                "apos_decisao_corrigido": result[f"{column}|after_corrected"] / n_values if n_values else None,
                "n_apos_corrigido": result[f"{column}|after_corrected"],
            })
    return pd.DataFrame(rows)


def offset_histogram(
    cases: pl.DataFrame, source: str, column: str, *, low: int = -30, high: int = 20,
    directory: Path = TRAIN_PARQUET_DIR,
) -> pd.Series:
    """Number of records by `column - date_decision` in days (raw calendar), inside [low, high]."""
    offset = (pl.col(column) - pl.col(DATE_DECISION)).dt.total_days().alias("dias")
    counts = (
        scan_source(source, directory)
        .select(CASE_ID, column)
        .join(cases.lazy().select(CASE_ID, DATE_DECISION), on=CASE_ID, how="inner")
        .select(offset)
        .filter(pl.col("dias").is_between(low, high))
        .group_by("dias")
        .agg(pl.len().alias("registros"))
        .sort("dias")
        .collect(engine="streaming")
    )
    return counts.to_pandas().set_index("dias")["registros"]


def year_month_offset(records: pl.LazyFrame, year: str, month: str, date_column: str, shift_days: int) -> pd.Series:
    """Share of records by ((year, month) field) - (month of `date_column` minus `shift_days`).

    For two fields that describe the same event, this tells how far the (year, month) pairs are
    from the day-level dates once the offset of the latter is removed.
    """
    event_day = pl.col(date_column) - pl.duration(days=shift_days)
    labelled = (pl.col(year) * 12 + pl.col(month)).cast(pl.Int32)
    difference = labelled - (event_day.dt.year() * 12 + event_day.dt.month())
    counts = (
        records.filter(pl.col(year).is_not_null() & pl.col(date_column).is_not_null())
        .select(difference.alias("meses"))
        .group_by("meses")
        .agg(pl.len().alias("n"))
        .sort("meses")
        .collect(engine="streaming")
        .to_pandas()
        .set_index("meses")["n"]
    )
    return counts / counts.sum()


def daily_counts(dates: pl.Series, start: date, end: date) -> pd.Series:
    """Number of values on each calendar day of [start, end], zero-filled."""
    days = pl.DataFrame({"dia": pl.date_range(start, end, "1d", eager=True)})
    counts = dates.filter(dates.is_between(start, end)).value_counts(name="n").rename({dates.name: "dia"})
    return days.join(counts, on="dia", how="left").fill_null(0).to_pandas().set_index("dia")["n"]


# --------------------------------------------------------------------------------------------------
# Candidate features on TRAIN
# --------------------------------------------------------------------------------------------------
def half_year(month: pl.Series) -> pl.Series:
    """`201907` -> `"2019H2"`: the period used to check whether a feature's signal is stable."""
    semester = pl.when(month % 100 <= 6).then(pl.lit("H1")).otherwise(pl.lit("H2"))
    return pl.select((month // 100).cast(pl.String) + semester).to_series().alias("semestre")


def quarter(month: pl.Series) -> pl.Series:
    """`201907` -> `"2019Q3"`: the period used to check whether a feature's distribution is stable."""
    number = ((month % 100 - 1) // 3 + 1).cast(pl.String)
    return pl.select((month // 100).cast(pl.String) + "Q" + number).to_series().alias("trimestre")


def feature_diagnostics(train: pl.DataFrame, features: list[str], target: str) -> pd.DataFrame:
    """Missing, cardinality, fill range across months, IV, IV by half-year and the largest PSI of a
    quarter against the whole sample. TRAIN rows only."""
    labels = train[target].to_numpy()
    by_period = iv_by_period(train, features, labels, half_year(train[MONTH]).to_numpy())
    by_period = by_period.pivot(index="feature", columns="period", values="iv").add_prefix("iv_")
    stability = psi_by_period(train, features, quarter(train[MONTH]).to_numpy())
    worst_quarter = stability.loc[stability.groupby("feature")["psi"].idxmax()].set_index("feature")
    monthly_fill = train.group_by(MONTH).agg([pl.col(feature).is_not_null().mean() for feature in features])
    diagnostics = iv_summary(train, features, labels).set_index("feature").join(by_period)
    diagnostics["psi_max"] = worst_quarter["psi"]
    diagnostics["psi_max_quarter"] = worst_quarter["period"]
    diagnostics["n_distinct"] = [train[feature].n_unique() for feature in features]
    diagnostics["fill_min_month"] = [monthly_fill[feature].min() for feature in features]
    diagnostics["fill_max_month"] = [monthly_fill[feature].max() for feature in features]
    return diagnostics.reset_index()


def redundant_pairs(
    train: pl.DataFrame, features: list[str], threshold: float, seed: int, min_pairs: int = CORRELATION_MIN_PAIRS
) -> pd.DataFrame:
    """Pairs of numeric features with |Spearman| >= threshold, on a random sample of TRAIN.

    A pair observed together in fewer than `min_pairs` rows is not reported.
    """
    numeric = [feature for feature in features if train.schema[feature].is_numeric()]
    sample = train.select(numeric).sample(n=min(CORRELATION_SAMPLE, train.height), seed=seed).to_pandas()
    correlation = sample.corr(method="spearman", min_periods=min_pairs).abs()
    pairs = [
        (first, second, correlation.loc[first, second])
        for position, first in enumerate(numeric)
        for second in numeric[position + 1:]
        if correlation.loc[first, second] >= threshold
    ]
    table = pd.DataFrame(pairs, columns=["feature_a", "feature_b", "spearman_abs"])
    return table.sort_values("spearman_abs", ascending=False, ignore_index=True)
