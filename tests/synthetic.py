"""Synthetic raw tables with the layout of the Kaggle files.

Events are generated on the true calendar (on or before each decision, or deliberately after it)
and then stored the way the organisers stored them: dates 14 days ahead, (year, month) pairs one
month ahead, dates as text. The pipeline therefore has to undo the offset to pass the tests.
"""
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from src.config import CASE_ID, DATE_DECISION, DATE_SHIFT_DAYS, MISSING_CATEGORY, MONTH_SHIFT
from src.data import DATE_FORMAT, SOURCES
from src.features import STATIC_0_COLUMNS, STATIC_CB_0_COLUMNS, TAX_REGISTRIES

TEXT_COLUMNS = {"credtype_322L": ["COL", "CAL", "REL"], "lastst_736L": ["D", "K", "A", "T"],
                "lastrejectreason_759M": [MISSING_CATEGORY, "P99_56_166", "P45_84_106"]}
UNUSED_SOURCES = ["credit_bureau_b_1", "credit_bureau_b_2", "deposit_1", "debitcard_1", "other_1",
                  "applprev_2", "person_2"]
# Sources stored in several files in the real data; the synthetic ones are split in two as well.
SPLIT_SOURCES = ["static_0", "applprev_1", "credit_bureau_a_1", "credit_bureau_a_2"]


def raw_date(day: date | None) -> date | None:
    return None if day is None else day + timedelta(days=DATE_SHIFT_DAYS)


def raw_year_month(day: date) -> tuple[float, float]:
    months = day.year * 12 + (day.month - 1) + MONTH_SHIFT
    return float(months // 12), float(months % 12 + 1)


def _history(base: pl.DataFrame, rng: np.random.Generator, mean_records: float, *, after_decision: bool):
    """Yield (case_id, num_group1, true event date) for a random number of records per case."""
    for case_id, decision in zip(base[CASE_ID].to_list(), base[DATE_DECISION].to_list()):
        for index in range(rng.poisson(mean_records)):
            gap = int(rng.integers(1, 60)) if after_decision else -int(rng.integers(0, 1500))
            yield case_id, index, decision + timedelta(days=gap)


def make_tables(base: pl.DataFrame, seed: int = 0, *, after_decision: bool = False) -> dict[str, pl.DataFrame]:
    """One frame per source. With `after_decision=True` every history record is dated after the
    decision, to be appended to a clean set and prove that such records change nothing."""
    rng = np.random.default_rng(seed)
    n_cases = base.height
    case_ids, decisions = base[CASE_ID].to_list(), base[DATE_DECISION].to_list()
    tables: dict[str, pl.DataFrame] = {}

    static = {CASE_ID: case_ids}
    for column in STATIC_0_COLUMNS:
        if column in TEXT_COLUMNS:
            static[column] = rng.choice(TEXT_COLUMNS[column], size=n_cases).tolist()
        else:
            static[column] = np.round(rng.gamma(2.0, 1000.0, size=n_cases), 1)
    for column in ("lastdelinqdate_224D", "lastrejectdate_50D"):
        static[column] = [
            raw_date(decision - timedelta(days=int(rng.integers(0, 900)))) if rng.random() < 0.5 else None
            for decision in decisions
        ]
    static["downpmt_116A"] = rng.choice([0.0, 500.0], size=n_cases)
    static["price_1097A"] = rng.choice([0.0, 5000.0], size=n_cases)
    tables["static_0"] = pl.DataFrame(static)

    covered = rng.random(n_cases) < 0.9
    tables["static_cb_0"] = pl.DataFrame({
        CASE_ID: case_ids,
        **{column: rng.integers(0, 6, size=n_cases).astype(float) for column in STATIC_CB_0_COLUMNS},
    }).filter(pl.Series(covered))

    persons = []
    for case_id, decision in zip(case_ids, decisions):
        birth = date(decision.year - int(rng.integers(21, 70)), int(rng.integers(1, 13)), 1)
        employed = raw_date(decision - timedelta(days=int(rng.integers(30, 4000)))) if rng.random() < 0.4 else None
        income_type = str(rng.choice(["EMPLOYED", "RETIRED_PENSIONER"]))
        education = str(rng.choice(["P97_36_170", "P33_146_175", MISSING_CATEGORY]))
        persons.append((case_id, 0, float(rng.integers(10, 200)) * 1000, income_type, education, birth, employed))
        for index in range(1, 1 + rng.integers(0, 3)):  # contacts: must never be read as the applicant
            persons.append((case_id, index, None, None, MISSING_CATEGORY, None, None))
    tables["person_1"] = pl.DataFrame(
        persons, orient="row",
        schema={CASE_ID: pl.Int64, "num_group1": pl.Int64, "mainoccupationinc_384A": pl.Float64,
                "incometype_1044T": pl.String, "education_927M": pl.String, "birth_259D": pl.Date,
                "empl_employedfrom_271D": pl.Date},
    )

    applications = []
    for case_id, index, created in _history(base, rng, 2.0, after_decision=after_decision):
        status = str(rng.choice(["D", "K", "A", "T"]))
        contracted = status in ("K", "A")
        applications.append((
            case_id, index, raw_date(created), raw_date(created) if contracted else None, status,
            float(rng.choice([0, 0, 3, 40])) if contracted else None,
            float(rng.integers(0, 50_000)) if contracted else None,
            float(rng.integers(5, 100)) * 1000,
        ))
    tables["applprev_1"] = pl.DataFrame(
        applications, orient="row",
        schema={CASE_ID: pl.Int64, "num_group1": pl.Int64, "creationdate_885D": pl.Date,
                "dateactivated_425D": pl.Date, "status_219L": pl.String, "maxdpdtolerance_577P": pl.Float64,
                "outstandingdebt_522A": pl.Float64, "credamount_590A": pl.Float64},
    )

    contracts, payments = [], []
    for case_id, index, started in _history(base, rng, 2.5, after_decision=after_decision):
        is_active = rng.random() < 0.4
        dpd = float(rng.choice([0, 0, 0, 7, 90]))
        year, month = raw_year_month(started)
        contracts.append((
            case_id, index,
            raw_date(started) if is_active else None, None if is_active else raw_date(started),
            dpd if is_active else None, None if is_active else dpd,
            year if is_active else None, month if is_active else None,
            None if is_active else year, None if is_active else month,
            # The bureau's case-level totals sit on the first row only and have no date of their own.
            float(rng.integers(0, 90_000)) if index == 0 and not after_decision else None,
            float(rng.integers(1, 9)) if index == 0 and not after_decision and rng.random() < 0.5 else None,
            float(rng.integers(100, 9_000)) if is_active else None,
            float(rng.choice([0, 0, 250])) if is_active else None,
        ))
        for position in range(int(rng.integers(1, 6))):  # monthly payments from the start onwards
            paid = date(started.year, started.month, 1) + timedelta(days=31 * position)
            pay_year, pay_month = raw_year_month(paid)
            pay_dpd = float(rng.choice([0, 0, 0, 5, 45]))
            payments.append((
                case_id, index, position,
                pay_year if is_active else None, pay_month if is_active else None, pay_dpd if is_active else None,
                None if is_active else pay_year, None if is_active else pay_month, None if is_active else pay_dpd,
                paid,
            ))
    tables["credit_bureau_a_1"] = pl.DataFrame(
        contracts, orient="row",
        schema={CASE_ID: pl.Int64, "num_group1": pl.Int64, "dateofcredstart_739D": pl.Date,
                "dateofcredstart_181D": pl.Date, "dpdmax_139P": pl.Float64, "dpdmax_757P": pl.Float64,
                "dpdmaxdateyear_596T": pl.Float64, "dpdmaxdatemonth_89T": pl.Float64,
                "dpdmaxdateyear_896T": pl.Float64, "dpdmaxdatemonth_442T": pl.Float64,
                "totaloutstanddebtvalue_39A": pl.Float64, "numberofcontrsvalue_358L": pl.Float64,
                "monthlyinstlamount_332A": pl.Float64,
                "overdueamount_659A": pl.Float64},
    )
    decision_of = dict(zip(case_ids, decisions))
    payment_frame = pl.DataFrame(
        payments, orient="row",
        schema={CASE_ID: pl.Int64, "num_group1": pl.Int64, "num_group2": pl.Int64,
                "pmts_year_1139T": pl.Float64, "pmts_month_158T": pl.Float64, "pmts_dpd_1073P": pl.Float64,
                "pmts_year_507T": pl.Float64, "pmts_month_706T": pl.Float64, "pmts_dpd_303P": pl.Float64,
                "paid": pl.Date},
    )
    # A scheduled month is "reported" (days past due filled) only up to the decision month; the
    # set of records after the decision keeps the opposite half, with the days past due filled.
    decision_month = pl.col(CASE_ID).replace_strict(decision_of, return_dtype=pl.Date).dt.truncate("1mo")
    is_future = pl.col("paid").dt.truncate("1mo") > decision_month
    if after_decision:
        payment_frame = payment_frame.filter(is_future)
    else:
        payment_frame = payment_frame.with_columns(
            pl.when(is_future).then(None).otherwise(pl.col(column)).alias(column)
            for column in ("pmts_dpd_1073P", "pmts_dpd_303P")
        )
    tables["credit_bureau_a_2"] = payment_frame.drop("paid")

    for position, (name, columns) in enumerate(TAX_REGISTRIES.items()):
        records = [
            (case_id, index, float(rng.integers(100, 5_000)), f"employer_{rng.integers(0, 3)}", raw_date(day))
            for case_id, index, day in _history(base, rng, 1.5, after_decision=after_decision)
            if case_id % 3 == position or case_id % 7 == 0  # some cases appear in two providers
        ]
        tables[name] = pl.DataFrame(
            records, orient="row",
            schema={CASE_ID: pl.Int64, "num_group1": pl.Int64, columns["amount"]: pl.Float64,
                    columns["employer"]: pl.String, columns["date"]: pl.Date},
        )

    for name in UNUSED_SOURCES:
        tables[name] = pl.DataFrame({CASE_ID: case_ids[: n_cases // 10], "num_group1": 0})
    return tables


def write_raw(base: pl.DataFrame, tables: dict[str, pl.DataFrame], directory: Path) -> Path:
    """Write `train_base` and the tables as the Kaggle parquet files (dates as text)."""
    directory.mkdir(parents=True, exist_ok=True)
    as_text = pl.col(pl.Date).dt.strftime(DATE_FORMAT)
    base.with_columns(as_text).write_parquet(directory / "train_base.parquet")
    assert set(tables) == set(SOURCES)
    for name, table in tables.items():
        stored = table.with_columns(as_text)
        if name in SPLIT_SOURCES:
            middle = stored[CASE_ID].median()
            stored.filter(pl.col(CASE_ID) <= middle).write_parquet(directory / f"train_{name}_0.parquet")
            stored.filter(pl.col(CASE_ID) > middle).write_parquet(directory / f"train_{name}_1.parquet")
        else:
            stored.write_parquet(directory / f"train_{name}.parquet")
    return directory
