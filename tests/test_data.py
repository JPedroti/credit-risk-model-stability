from datetime import date

import polars as pl
import pytest

from src.config import CASE_ID, DATE_DECISION
from src.data import (
    SOURCES, cases_with_records, date_columns, load_base, load_feature_definitions,
    parse_dates, scan_source, source_files,
)


def test_every_source_of_the_competition_is_registered():
    assert len(SOURCES) == 16
    assert {source.depth for source in SOURCES.values()} == {0, 1, 2}
    assert SOURCES["credit_bureau_a_2"].depth == 2 and SOURCES["static_cb_0"].origin == "externa"


def test_source_files_finds_single_files_and_parts(raw_dir):
    assert [file.name for file in source_files("person_1", raw_dir)] == ["train_person_1.parquet"]
    assert [file.name for file in source_files("applprev_1", raw_dir)] == [
        "train_applprev_1_0.parquet", "train_applprev_1_1.parquet",
    ]
    # `applprev_1` must not pick up `applprev_2`, nor `credit_bureau_a_1` the files of `a_2`.
    assert all("a_1_" in file.name for file in source_files("credit_bureau_a_1", raw_dir))
    with pytest.raises(FileNotFoundError):
        source_files("not_a_source", raw_dir)


def test_dates_are_parsed_and_malformed_dates_fail():
    raw = pl.DataFrame({"case_id": [1], "date_decision": ["2020-03-10"], "created_1D": ["2020-01-31"], "label_2L": ["D"]})
    parsed = parse_dates(raw.lazy()).collect()
    assert parsed.schema["date_decision"] == pl.Date and parsed["created_1D"][0] == date(2020, 1, 31)
    assert parsed.schema["label_2L"] == pl.String  # a text column that merely ends in another letter
    with pytest.raises(pl.exceptions.PolarsError):
        parse_dates(raw.with_columns(pl.lit("31/01/2020").alias("created_1D")).lazy()).collect()


def test_base_and_sources_are_read_with_dates(small_base, raw_dir):
    base = load_base(raw_dir)
    assert base.equals(small_base.drop("split"))
    assert base.schema[DATE_DECISION] == pl.Date

    applprev = scan_source("applprev_1", raw_dir)
    assert date_columns(applprev) == ["creationdate_885D", "dateactivated_425D"]


def test_cases_with_records_counts_rows_per_case(raw_dir, raw_tables):
    counted = cases_with_records("applprev_1", raw_dir).sort(CASE_ID)
    expected = raw_tables["applprev_1"].group_by(CASE_ID).agg(pl.len().alias("n_records")).sort(CASE_ID)
    assert counted.equals(expected)
    assert counted[CASE_ID].n_unique() == counted.height


def test_feature_definitions_are_read_as_a_dictionary(tmp_path):
    path = tmp_path / "feature_definitions.csv"
    path.write_text('Variable,Description\nannuity_780A,Monthly annuity amount.\n"x_1L","Text, with comma"\n', encoding="utf-8")
    assert load_feature_definitions(path) == {"annuity_780A": "Monthly annuity amount.", "x_1L": "Text, with comma"}
