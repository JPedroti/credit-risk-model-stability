from datetime import date, timedelta

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from src.catalog import CANDIDATE_FEATURES, SELECTED_FEATURES
from src.config import CASE_ID, DATE_DECISION, KEY_COLUMNS, MISSING_CATEGORY
from src.data import scan_source
from src.features import (
    CLOSED_DETAIL_MISSING, DAYS_PER_YEAR, NEEDS_CLOSED_DETAIL, STATIC_0_COLUMNS, TAX_REGISTRIES,
    ZERO_WHEN_NO_HISTORY, applicant_features, applprev_features, build_candidate_table, build_master_table,
    bureau_contract_features, bureau_payment_features, cross_source_features, days_before_decision,
    mask_incomplete_bureau, months_before_decision, static_0_features, tax_registry_features,
)
from src.qa import assert_all_passed, check_master_table
from tests.synthetic import raw_date, raw_year_month

DECISION = date(2020, 3, 10)


def decisions(*case_ids: int) -> pl.LazyFrame:
    return pl.DataFrame({CASE_ID: list(case_ids), DATE_DECISION: [DECISION] * len(case_ids)}).lazy()


def stored(days_before: int) -> date:
    """Date found in the raw files for an event `days_before` the decision (negative = after it)."""
    return raw_date(DECISION - timedelta(days=days_before))


def stored_month(months_before: int) -> tuple[float, float]:
    """(year, month) found in the raw files for an event `months_before` the decision month."""
    months = DECISION.year * 12 + (DECISION.month - 1) - months_before
    return raw_year_month(date(months // 12, months % 12 + 1, 1))


def one_row(frame: pl.LazyFrame, case_id: int) -> dict:
    rows = frame.collect().filter(pl.col(CASE_ID) == case_id).to_dicts()
    assert len(rows) == 1
    return rows[0]


def test_elapsed_time_is_measured_on_the_calendar_of_the_decision():
    year, month = zip(stored_month(0), stored_month(1), stored_month(-1))
    frame = pl.DataFrame({
        DATE_DECISION: [DECISION] * 3,
        "event": [stored(0), stored(30), stored(-1)],
        "year": year, "month": month,
    })
    elapsed = frame.select(
        days_before_decision("event").alias("days"), months_before_decision("year", "month").alias("months")
    )
    # Same day / same month is zero, the past is positive and anything after the decision is negative.
    assert elapsed["days"].to_list() == [0, 30, -1]
    assert elapsed["months"].to_list() == [0, 1, -1]


def test_previous_applications_are_aggregated_up_to_the_decision():
    applprev = pl.DataFrame(
        [
            (1, 0, stored(-1), None, "D", None, None, 99_000.0),  # created after the decision: ignored
            (1, 1, stored(0), None, "T", None, None, 10_000.0),
            (1, 2, stored(100), None, "D", None, None, 20_000.0),
            (1, 3, stored(400), stored(400), "K", 12.0, 0.0, 30_000.0),
            (1, 4, stored(1000), stored(1000), "A", 0.0, 20_000.0, 50_000.0),
            (2, 0, stored(-3), stored(-3), "A", 80.0, 5_000.0, 70_000.0),  # only future history
        ],
        orient="row",
        schema={"case_id": pl.Int64, "num_group1": pl.Int64, "creationdate_885D": pl.Date,
                "dateactivated_425D": pl.Date, "status_219L": pl.String, "maxdpdtolerance_577P": pl.Float64,
                "outstandingdebt_522A": pl.Float64, "credamount_590A": pl.Float64},
    ).lazy()
    features = applprev_features(applprev, decisions(1, 2))

    assert one_row(features, 1) == {
        "case_id": 1,
        "prev_n_applications": 4,
        "prev_n_contracts": 2,
        "prev_share_rejected": 0.25,
        "prev_days_since_last_application": 0,
        "prev_years_since_first_application": 1000 / DAYS_PER_YEAR,
        "prev_n_applications_3m": 1,
        "prev_n_applications_12m": 2,
        "prev_n_rejected_12m": 1,
        "prev_max_dpd": 12.0,
        "prev_share_contracts_dpd": 0.5,
        "prev_outstanding_debt": 20_000.0,
        "prev_max_credamount": 50_000.0,
    }
    assert features.collect().filter(pl.col(CASE_ID) == 2).is_empty()


BUREAU_SCHEMA = {
    "case_id": pl.Int64, "num_group1": pl.Int64,
    "dateofcredstart_739D": pl.Date, "dpdmax_139P": pl.Float64, "dpdmaxdateyear_596T": pl.Float64,
    "dpdmaxdatemonth_89T": pl.Float64, "monthlyinstlamount_332A": pl.Float64, "overdueamount_659A": pl.Float64,
    "dateofcredstart_181D": pl.Date, "dpdmax_757P": pl.Float64, "dpdmaxdateyear_896T": pl.Float64,
    "dpdmaxdatemonth_442T": pl.Float64, "totaloutstanddebtvalue_39A": pl.Float64,
    "numberofcontrsvalue_358L": pl.Float64,
}


def test_bureau_contracts_treat_active_and_closed_lists_separately():
    recent_year, recent_month = stored_month(0)
    old_year, old_month = stored_month(7)
    bureau = pl.DataFrame(
        [
            # active contract (start, dpd, dpd year, dpd month, instalment, overdue) | closed contract | totals
            (1, 0, stored(10), 0.0, None, None, 500.0, 0.0, stored(2000), 30.0, old_year, old_month, 9_000.0, 2.0),
            (1, 1, stored(500), 15.0, recent_year, recent_month, 300.0, 120.0, None, None, None, None, None, None),
            (1, 2, stored(-5), 99.0, recent_year, recent_month, 9_999.0, 9_999.0, stored(700), None, None, None, None, None),
            (2, 0, None, None, None, None, None, None, None, None, None, None, None, None),  # bureau hit, no contract
            # The bureau counts three closed contracts but sends none; one more starts after the decision.
            (3, 0, stored(40), 0.0, None, None, 100.0, 0.0, stored(-9), 50.0, None, None, 1_000.0, 3.0),
        ],
        orient="row", schema=BUREAU_SCHEMA,
    ).lazy()
    features = bureau_contract_features(bureau, decisions(1, 2, 3))

    assert one_row(features, 1) == {
        "case_id": 1,
        "cb_n_active_contracts": 2,  # the contract opened after the decision does not count
        "cb_n_closed_contracts": 2,  # as reported by the bureau
        CLOSED_DETAIL_MISSING: False,
        "cb_active_debt": 9_000.0,
        "cb_active_monthly_instalment": 800.0,
        "cb_active_overdue_amount": 120.0,
        "cb_active_max_dpd": 15.0,
        "cb_closed_max_dpd": 30.0,
        "cb_share_contracts_dpd": 2 / 3,  # late in 2 of the 3 contracts with a known maximum
        "cb_years_since_first_credit": 2000 / DAYS_PER_YEAR,
        "cb_days_since_last_credit": 10,
        "cb_n_credits_12m": 1,
        "cb_months_since_max_dpd": 0,
    }
    no_contract = one_row(features, 2)
    assert no_contract["cb_n_active_contracts"] == 0 and no_contract["cb_n_closed_contracts"] == 0
    assert no_contract["cb_active_max_dpd"] is None and no_contract["cb_share_contracts_dpd"] is None
    assert no_contract[CLOSED_DETAIL_MISSING] is False

    incomplete = one_row(features, 3)
    assert incomplete["cb_n_closed_contracts"] == 3 and incomplete[CLOSED_DETAIL_MISSING] is True
    assert incomplete["cb_closed_max_dpd"] is None  # the closed contract after the decision is not used


def test_features_that_need_closed_contracts_are_unknown_when_the_bureau_sends_none():
    table = pl.DataFrame({
        "case_id": [1, 2, 3],
        CLOSED_DETAIL_MISSING: [False, True, None],  # None: no bureau record at all
        **{feature: [1.0, 2.0, None] for feature in NEEDS_CLOSED_DETAIL},
        "cb_active_max_dpd": [5.0, 6.0, None],
    }).lazy()
    masked = mask_incomplete_bureau(table).collect()

    assert CLOSED_DETAIL_MISSING not in masked.columns
    for feature in NEEDS_CLOSED_DETAIL:
        assert masked[feature].to_list() == [1.0, None, None]
    assert masked["cb_active_max_dpd"].to_list() == [5.0, 6.0, None]  # active contracts are always sent


def test_bureau_payments_use_only_reported_months_up_to_the_decision():
    def row(case_id, active, closed):
        active_year, active_month = stored_month(active[0]) if active else (None, None)
        closed_year, closed_month = stored_month(closed[0]) if closed else (None, None)
        return (case_id, active_year, active_month, active[1] if active else None,
                closed_year, closed_month, closed[1] if closed else None)

    payments = pl.DataFrame(
        [
            row(1, (0, 5.0), (1, 0.0)),
            row(1, (2, 0.0), (11, 3.0)),
            row(1, (4, 60.0), None),
            row(1, (13, 200.0), None),  # older than 12 months
            row(1, (-1, None), None),  # scheduled month, nothing reported yet
            row(1, (-2, 77.0), None),  # reported after the decision: must be ignored
            row(2, (20, 0.0), None),
        ],
        orient="row",
        schema={"case_id": pl.Int64, "pmts_year_1139T": pl.Float64, "pmts_month_158T": pl.Float64,
                "pmts_dpd_1073P": pl.Float64, "pmts_year_507T": pl.Float64, "pmts_month_706T": pl.Float64,
                "pmts_dpd_303P": pl.Float64},
    ).lazy()
    features = bureau_payment_features(payments, decisions(1, 2))

    assert one_row(features, 1) == {
        "case_id": 1,
        "cb_pmt_max_dpd_3m": 5.0,
        "cb_pmt_max_dpd_12m": 60.0,
        "cb_pmt_share_late_12m": 3 / 5,
        "cb_pmt_months_since_last_late": 0,
    }
    assert one_row(features, 2) == {
        "case_id": 2, "cb_pmt_max_dpd_3m": None, "cb_pmt_max_dpd_12m": None,
        "cb_pmt_share_late_12m": None, "cb_pmt_months_since_last_late": None,
    }


def test_tax_registries_are_pooled_without_double_counting():
    def registry(name, rows):
        columns = TAX_REGISTRIES[name]
        return pl.DataFrame(
            rows, orient="row",
            schema={"case_id": pl.Int64, columns["amount"]: pl.Float64, columns["employer"]: pl.String, columns["date"]: pl.Date},
        ).lazy()

    registries = {
        "tax_registry_a_1": registry("tax_registry_a_1", [
            (1, 100.0, "x", stored(0)), (1, 300.0, "y", stored(0)), (1, 900.0, "w", stored(-2)),
            (3, 500.0, "x", stored(-1)),  # case 3 only has a record after the decision
        ]),
        "tax_registry_b_1": registry("tax_registry_b_1", [(2, 100.0, "x", stored(30)), (2, 300.0, "y", stored(60))]),
        "tax_registry_c_1": registry("tax_registry_c_1", [
            (1, 200.0, "z", stored(20)), (1, 200.0, "z", stored(50)), (1, 200.0, "z", stored(80)),
        ]),
    }
    features = tax_registry_features(registries, decisions(1, 2, 3)).collect()

    assert features[CASE_ID].sort().to_list() == [1, 2]
    # Case 1 is in two providers: the fuller one (three deductions, one employer) is kept.
    assert one_row(features.lazy(), 1) == {"case_id": 1, "tax_n_deductions": 3, "tax_n_employers": 1, "tax_amount_cv": 0.0}
    second = one_row(features.lazy(), 2)
    assert (second["tax_n_deductions"], second["tax_n_employers"]) == (2, 2)
    assert second["tax_amount_cv"] == pytest.approx(2 ** 0.5 / 2)  # std 141.4 over mean 200


def test_static_features_clean_placeholders_and_future_dates():
    text = {"credtype_322L": ["COL", "CAL"], "lastst_736L": ["D", None],
            "lastrejectreason_759M": [MISSING_CATEGORY, "P99_56_166"]}
    static = pl.DataFrame({
        "case_id": [1, 2],
        **{column: text.get(column, [1.0, 2.0]) for column in STATIC_0_COLUMNS},
        "lastdelinqdate_224D": [stored(20), None],
        "lastrejectdate_50D": [stored(-4), stored(0)],
        "downpmt_116A": [500.0, 500.0],
        "price_1097A": [5_000.0, 0.0],
    }).lazy()
    features = static_0_features(static, decisions(1, 2)).collect().sort(CASE_ID)

    assert features["lastrejectreason_759M"].to_list() == [None, "P99_56_166"]
    assert features["days_since_last_delinquency"].to_list() == [20, None]
    assert features["days_since_last_rejection"].to_list() == [None, 0]  # a date after the decision is not used
    assert features["downpayment_share"].to_list() == [0.1, None]  # no division by a zero price
    assert set(STATIC_0_COLUMNS) <= set(features.columns)


def test_only_the_applicant_row_of_person_1_is_used():
    persons = pl.DataFrame(
        [
            (1, 0, 50_000.0, "EMPLOYED", MISSING_CATEGORY, date(1990, 3, 1), stored(366)),
            (1, 1, 999.0, "OTHER", "P1", date(1950, 1, 1), stored(10)),  # a contact person
        ],
        orient="row",
        schema=["case_id", "num_group1", "mainoccupationinc_384A", "incometype_1044T", "education_927M",
                "birth_259D", "empl_employedfrom_271D"],
    ).lazy()
    applicant = one_row(applicant_features(persons, decisions(1)), 1)

    assert applicant["mainoccupationinc_384A"] == 50_000.0 and applicant["incometype_1044T"] == "EMPLOYED"
    assert applicant["education_927M"] is None
    assert applicant["age_years"] == pytest.approx((DECISION - date(1990, 3, 1)).days / DAYS_PER_YEAR)
    assert applicant["employment_years"] == pytest.approx(366 / DAYS_PER_YEAR)


def test_ratios_are_null_instead_of_infinite():
    table = pl.DataFrame({
        "mainoccupationinc_384A": [50_000.0, 0.0, None],
        "annuity_780A": [5_000.0, 5_000.0, 5_000.0],
        "credamount_770A": [100_000.0, 100_000.0, 100_000.0],
        "cb_active_monthly_instalment": [2_500.0, 2_500.0, None],
        "cb_active_debt": [25_000.0, 25_000.0, 25_000.0],
        "prev_max_credamount": [50_000.0, 0.0, None],
    }).lazy()
    ratios = cross_source_features(table).collect()

    assert ratios["annuity_to_income"].to_list() == [0.1, None, None]
    assert ratios["credit_to_income"].to_list() == [2.0, None, None]
    assert ratios["debt_service_to_income"].to_list() == [0.15, None, None]
    assert ratios["cb_debt_to_income"].to_list() == [0.5, None, None]
    assert ratios["credit_to_prev_max_credit"].to_list() == [2.0, None, None]


def test_candidate_table_keeps_one_row_per_case(small_base, raw_dir):
    candidates = build_candidate_table(small_base, raw_dir)

    assert candidates.height == small_base.height == candidates[CASE_ID].n_unique()
    assert candidates.select(small_base.columns).equals(small_base)  # the spine is untouched
    assert candidates.columns == [*small_base.columns, *CANDIDATE_FEATURES]

    without_history = candidates.join(scan_source("applprev_1", raw_dir).collect(), on=CASE_ID, how="anti")
    assert without_history.height > 0  # the LEFT JOIN keeps clients with no previous application...
    assert without_history["prev_n_applications"].unique().to_list() == [0]  # ...with a count of zero
    assert without_history["prev_share_rejected"].null_count() == without_history.height
    assert candidates.select(pl.col(ZERO_WHEN_NO_HISTORY).null_count()).sum_horizontal().item() == 0

    # Some synthetic cases have closed contracts counted by the bureau but not sent.
    unknown = candidates.filter((pl.col("cb_n_closed_contracts") > 0) & pl.col("cb_closed_max_dpd").is_null())
    assert 0 < unknown.filter(pl.col("cb_share_contracts_dpd").is_null()).height


def test_records_after_the_decision_change_no_feature(small_base, raw_dir, raw_dir_with_future):
    def n_records(directory, source):
        return scan_source(source, directory).select(pl.len()).collect().item()

    for source in ["applprev_1", "credit_bureau_a_1", "credit_bureau_a_2", *TAX_REGISTRIES]:
        assert n_records(raw_dir_with_future, source) > n_records(raw_dir, source)

    clean = build_candidate_table(small_base, raw_dir)
    with_future = build_candidate_table(small_base, raw_dir_with_future)
    assert_frame_equal(clean, with_future, check_exact=False)


def test_master_table_has_the_keys_and_the_selected_features(small_base, raw_dir):
    master = build_master_table(build_candidate_table(small_base, raw_dir), SELECTED_FEATURES)

    assert master.columns == [*KEY_COLUMNS, *SELECTED_FEATURES]
    assert master[CASE_ID].is_sorted()
    assert_all_passed(check_master_table(master, small_base.drop("split")))
