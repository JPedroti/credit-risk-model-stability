"""Synthetic fixtures: small tables with the same shape as the Kaggle files, so no test needs the raw data."""
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from src.config import CASE_ID, DATE_DECISION, MONTH, TARGET, WEEK_NUM
from src.split import assign_split
from tests.synthetic import make_tables, write_raw

FIRST_DAY = date(2019, 1, 1)
LAST_DAY = date(2020, 10, 31)


def make_base(n_cases: int, seed: int) -> pl.DataFrame:
    """A `train_base` look-alike spread over 2019-01..2020-10 with ~10% defaults."""
    rng = np.random.default_rng(seed)
    offsets = rng.integers(0, (LAST_DAY - FIRST_DAY).days + 1, size=n_cases)
    decisions = [FIRST_DAY + timedelta(days=int(offset)) for offset in offsets]
    return pl.DataFrame({
        CASE_ID: rng.permutation(n_cases),
        DATE_DECISION: decisions,
        MONTH: [day.year * 100 + day.month for day in decisions],
        WEEK_NUM: offsets // 7,
        TARGET: (rng.random(n_cases) < 0.10).astype(int),
    })


@pytest.fixture
def base() -> pl.DataFrame:
    return make_base(n_cases=6_000, seed=0)


@pytest.fixture(scope="session")
def small_base() -> pl.DataFrame:
    """The population behind the synthetic raw files, already with its split."""
    return assign_split(make_base(n_cases=600, seed=1))


@pytest.fixture(scope="session")
def raw_tables(small_base) -> dict[str, pl.DataFrame]:
    return make_tables(small_base, seed=2)


@pytest.fixture(scope="session")
def raw_dir(tmp_path_factory, small_base, raw_tables):
    """Raw parquet files in which every history record is on or before the decision."""
    return write_raw(small_base.drop("split"), raw_tables, tmp_path_factory.mktemp("raw"))


@pytest.fixture(scope="session")
def raw_dir_with_future(tmp_path_factory, small_base, raw_tables):
    """The same files plus records dated after the decision in every history table."""
    future = make_tables(small_base, seed=3, after_decision=True)
    histories = ["applprev_1", "credit_bureau_a_1", "credit_bureau_a_2", *[name for name in raw_tables if name.startswith("tax_")]]
    tables = {
        name: pl.concat([table, future[name]]) if name in histories else table
        for name, table in raw_tables.items()
    }
    return write_raw(small_base.drop("split"), tables, tmp_path_factory.mktemp("raw_future"))
