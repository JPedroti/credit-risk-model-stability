"""Access to the raw Kaggle tables (`data/raw/parquet_files/train`).

Every source is read lazily, with its date columns (suffix `D`, stored as text in the files) parsed
to `Date`, so that comparing any event with `date_decision` is a plain date comparison.
"""
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from src.config import CASE_ID, DATE_DECISION, FEATURE_DEFINITIONS_PATH, TRAIN_PARQUET_DIR

DATE_FORMAT = "%Y-%m-%d"
DATE_SUFFIX = "D"


@dataclass(frozen=True)
class Source:
    """One family of tables of the competition (brief, section 3)."""

    name: str
    depth: int  # 0: one row per case; 1: history indexed by num_group1; 2: by num_group1 and num_group2
    origin: str  # "interna" (Home Credit) or "externa" (bureau / tax registry)


SOURCES = {
    source.name: source
    for source in [
        Source("static_0", 0, "interna"),
        Source("static_cb_0", 0, "externa"),
        Source("applprev_1", 1, "interna"),
        Source("person_1", 1, "interna"),
        Source("other_1", 1, "interna"),
        Source("deposit_1", 1, "interna"),
        Source("debitcard_1", 1, "interna"),
        Source("credit_bureau_a_1", 1, "externa"),
        Source("credit_bureau_b_1", 1, "externa"),
        Source("tax_registry_a_1", 1, "externa"),
        Source("tax_registry_b_1", 1, "externa"),
        Source("tax_registry_c_1", 1, "externa"),
        Source("applprev_2", 2, "interna"),
        Source("person_2", 2, "interna"),
        Source("credit_bureau_a_2", 2, "externa"),
        Source("credit_bureau_b_2", 2, "externa"),
    ]
}


def source_files(name: str, directory: Path = TRAIN_PARQUET_DIR) -> list[Path]:
    """Files of one source: a single `train_<name>.parquet` or its `train_<name>_<i>.parquet` parts."""
    single = directory / f"train_{name}.parquet"
    files = [single] if single.exists() else sorted(directory.glob(f"train_{name}_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no file for source '{name}' in {directory}")
    return files


def parse_dates(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Parse the text columns named `*D` (and `date_decision`) to `Date`. Fails on a malformed date."""
    schema = frame.collect_schema()
    date_columns = [
        column for column, dtype in schema.items()
        if dtype == pl.String and (column.endswith(DATE_SUFFIX) or column == DATE_DECISION)
    ]
    return frame.with_columns(pl.col(date_columns).str.to_date(DATE_FORMAT))


def load_base(directory: Path = TRAIN_PARQUET_DIR) -> pl.DataFrame:
    """`train_base`: one row per case, with `date_decision` as a date."""
    return parse_dates(pl.scan_parquet(directory / "train_base.parquet")).collect()


def scan_source(name: str, directory: Path = TRAIN_PARQUET_DIR) -> pl.LazyFrame:
    """Lazy view of every file of one source, with parsed dates."""
    return parse_dates(pl.scan_parquet(source_files(name, directory)))


def date_columns(frame: pl.LazyFrame) -> list[str]:
    return [column for column, dtype in frame.collect_schema().items() if dtype == pl.Date]


def load_feature_definitions(path: Path = FEATURE_DEFINITIONS_PATH) -> dict[str, str]:
    """Data dictionary of the competition: raw column name -> description."""
    definitions = pl.read_csv(path)
    return dict(zip(definitions["Variable"], definitions["Description"]))


def cases_with_records(name: str, directory: Path = TRAIN_PARQUET_DIR) -> pl.DataFrame:
    """Distinct `case_id` present in a source, with how many rows each one has."""
    return (
        pl.scan_parquet(source_files(name, directory))
        .group_by(CASE_ID)
        .agg(pl.len().alias("n_records"))
        .collect(engine="streaming")
    )
