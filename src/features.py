"""Feature engineering: from the relational tables to one row per `case_id`.

Every builder receives lazy tables and returns at most one row per case. Three rules hold throughout:

1. Point of observation. A history is joined to `date_decision` and each dated record is compared
   with it on the decision's calendar (`days_before_decision`, `months_before_decision`). A record
   dated after the decision is dropped before any aggregation.
2. No learned parameter. A feature is a fixed formula of the case's own records, so it is computed
   in exactly the same way for TRAIN, TEST and OOT and cannot carry information between them.
3. Granularity. `build_candidate_table` LEFT JOINs each block to `train_base` and validates the
   join as one-to-one, so no case is lost and none is duplicated.

The business hypothesis and the temporal check of each feature live in `src/catalog.py`.
"""
from pathlib import Path

import polars as pl

from src.config import (
    CASE_ID, DATE_DECISION, DATE_SHIFT_DAYS, KEY_COLUMNS, MISSING_CATEGORY, MONTH_SHIFT, TRAIN_PARQUET_DIR,
)
from src.data import scan_source

DAYS_PER_YEAR = 365.25
APPLICANT_INDEX = 0  # in person tables, num_group1 == 0 is the applicant
REJECTED_STATUS = "D"  # applprev_1.status_219L of a refused application (inferred, see notebook 02)

# --------------------------------------------------------------------------------------------------
# Raw columns kept as they are ("features originais selecionadas")
# --------------------------------------------------------------------------------------------------
STATIC_0_COLUMNS = [
    # the application being decided
    "credamount_770A", "annuity_780A", "pmtnum_254L", "eir_270L", "credtype_322L",
    # obligations the client already has with the lender
    "totaldebt_9A", "annuitynextmonth_57A",
    # repayment behaviour on previous contracts with the lender
    "avgdpdtolclosure24_3658938P", "pctinstlsallpaidlate1d_3546856L", "pctinstlsallpaidlat10d_839L",
    "pctinstlsallpaidearl3d_427L", "maxdpdlast3m_392P", "maxdpdlast12m_727P", "maxdpdlast24m_143P",
    "maxdpdtolerance_374P", "maxdbddpdtollast12m_3658940P", "avgdbddpdlast24m_3658932P",
    "numinstlswithdpd10_728L", "numinstlsallpaid_934L",
    # recent application behaviour
    "numrejects9m_859L", "applications30d_658L", "numcontrs3months_479L", "lastst_736L",
    "lastrejectreason_759M",
    # shared contact data
    "mobilephncnt_593L", "sellerplacecnt_915L",
]
STATIC_CB_0_COLUMNS = ["days30_165L", "days120_123L", "days360_512L"]
PERSON_1_COLUMNS = ["mainoccupationinc_384A", "incometype_1044T", "education_927M"]

# Previous-application counts are zero, not unknown, for a client with no row in applprev_1: the
# table is internal and its absence matches a null `lastapplicationdate_877D` in every TRAIN case.
ZERO_WHEN_NO_HISTORY = [
    "prev_n_applications", "prev_n_contracts", "prev_n_applications_3m", "prev_n_applications_12m",
    "prev_n_rejected_12m",
]

# Features that need the closed-contract records of bureau A. In the first months of the period the
# bureau reports how many closed contracts a client has but sends no record of them; computed on
# the active contracts alone these features would mean something else, so they are left unknown.
CLOSED_DETAIL_MISSING = "cb_closed_detail_missing"
NEEDS_CLOSED_DETAIL = [
    "cb_share_contracts_dpd", "cb_years_since_first_credit", "cb_days_since_last_credit", "cb_n_credits_12m",
    "cb_months_since_max_dpd", "cb_pmt_max_dpd_3m", "cb_pmt_max_dpd_12m", "cb_pmt_share_late_12m",
    "cb_pmt_months_since_last_late",
]

# The three tax registries hold the same kind of record under different column names.
TAX_REGISTRIES = {
    "tax_registry_a_1": {"amount": "amount_4527230A", "employer": "name_4527232M", "date": "recorddate_4527225D"},
    "tax_registry_b_1": {"amount": "amount_4917619A", "employer": "name_4917606M", "date": "deductiondate_4917603D"},
    "tax_registry_c_1": {"amount": "pmtamount_36A", "employer": "employername_160M", "date": "processingdate_168D"},
}


# --------------------------------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------------------------------
def days_before_decision(column: str) -> pl.Expr:
    """Days from an event to the decision, on the decision's calendar. Negative = after the decision."""
    return (pl.col(DATE_DECISION) - pl.col(column)).dt.total_days() + DATE_SHIFT_DAYS


def months_before_decision(year: str, month: str) -> pl.Expr:
    """Whole months from a (year, month) event to the decision month. Negative = after the decision."""
    decision = pl.col(DATE_DECISION).dt.year() * 12 + pl.col(DATE_DECISION).dt.month()
    event = (pl.col(year) * 12 + pl.col(month)).cast(pl.Int32)
    return decision - event + MONTH_SHIFT


def with_decision(frame: pl.LazyFrame, base: pl.LazyFrame) -> pl.LazyFrame:
    """Attach `date_decision` to the records of the cases in `base`."""
    return frame.join(base.select(CASE_ID, DATE_DECISION), on=CASE_ID, how="inner")


def not_after_decision(elapsed: pl.Expr) -> pl.Expr:
    """Keep an elapsed time only when the event is not after the decision (null otherwise)."""
    return pl.when(elapsed >= 0).then(elapsed)


def clean_category(column: str) -> pl.Expr:
    """Masked categories use a fixed code for "missing"; turn it into a real null."""
    return pl.col(column).replace(MISSING_CATEGORY, None)


def safe_ratio(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    """Ratio that is null (not infinite) when the denominator is zero, negative or missing."""
    return pl.when(denominator > 0).then(numerator / denominator)


def share(count: pl.Expr, total: pl.Expr) -> pl.Expr:
    """`count / total` inside an aggregation, null when there is nothing to count."""
    return pl.when(total > 0).then(count / total)


# --------------------------------------------------------------------------------------------------
# Depth 0: one row per case, direct join
# --------------------------------------------------------------------------------------------------
def static_0_features(static_0: pl.LazyFrame, base: pl.LazyFrame) -> pl.LazyFrame:
    """Selected raw columns of static_0, plus recencies and the down-payment share."""
    selected = [
        clean_category(column) if column.endswith("M") else pl.col(column) for column in STATIC_0_COLUMNS
    ]
    return with_decision(static_0, base).select(
        CASE_ID,
        *selected,
        not_after_decision(days_before_decision("lastdelinqdate_224D")).alias("days_since_last_delinquency"),
        not_after_decision(days_before_decision("lastrejectdate_50D")).alias("days_since_last_rejection"),
        safe_ratio(pl.col("downpmt_116A"), pl.col("price_1097A")).alias("downpayment_share"),
    )


def static_cb_0_features(static_cb_0: pl.LazyFrame) -> pl.LazyFrame:
    """Bureau enquiries by window, and how concentrated they are in the last 30 days."""
    return static_cb_0.select(
        CASE_ID,
        *STATIC_CB_0_COLUMNS,
        safe_ratio(pl.col("days30_165L"), pl.col("days360_512L")).alias("cb_queries_30d_share"),
    )


# --------------------------------------------------------------------------------------------------
# Depth 1: history indexed by num_group1
# --------------------------------------------------------------------------------------------------
def applicant_features(person_1: pl.LazyFrame, base: pl.LazyFrame) -> pl.LazyFrame:
    """The applicant's row of person_1 (the other rows are contacts and related persons)."""
    applicant = with_decision(person_1.filter(pl.col("num_group1") == APPLICANT_INDEX), base)
    # Birth dates are truncated to the first day of the month, so the 14-day offset of the other
    # dates cannot be identified in them and is irrelevant at the scale of years.
    age = (pl.col(DATE_DECISION) - pl.col("birth_259D")).dt.total_days() / DAYS_PER_YEAR
    employment = not_after_decision(days_before_decision("empl_employedfrom_271D")) / DAYS_PER_YEAR
    return applicant.select(
        CASE_ID,
        "mainoccupationinc_384A",
        "incometype_1044T",
        clean_category("education_927M"),
        age.alias("age_years"),
        employment.alias("employment_years"),
    )


def applprev_features(applprev_1: pl.LazyFrame, base: pl.LazyFrame) -> pl.LazyFrame:
    """Previous applications to the lender: volume, refusals, recency, delinquency and exposure."""
    history = (
        with_decision(applprev_1, base)
        .with_columns(days_before_decision("creationdate_885D").alias("days_ago"))
        .filter(pl.col("days_ago") >= 0)
    )
    days_ago = pl.col("days_ago")
    is_contract = pl.col("dateactivated_425D").is_not_null()
    is_rejected = pl.col("status_219L") == REJECTED_STATUS
    max_dpd = pl.col("maxdpdtolerance_577P")
    return history.group_by(CASE_ID).agg(
        pl.len().alias("prev_n_applications"),
        is_contract.sum().alias("prev_n_contracts"),
        is_rejected.mean().alias("prev_share_rejected"),
        days_ago.min().alias("prev_days_since_last_application"),
        (days_ago.max() / DAYS_PER_YEAR).alias("prev_years_since_first_application"),
        (days_ago <= 90).sum().alias("prev_n_applications_3m"),
        (days_ago <= 365).sum().alias("prev_n_applications_12m"),
        (is_rejected & (days_ago <= 365)).sum().alias("prev_n_rejected_12m"),
        max_dpd.max().alias("prev_max_dpd"),
        share((max_dpd > 0).sum(), max_dpd.count()).alias("prev_share_contracts_dpd"),
        pl.col("outstandingdebt_522A").sum().alias("prev_outstanding_debt"),
        pl.col("credamount_590A").filter(is_contract).max().alias("prev_max_credamount"),
    )


def bureau_contract_features(credit_bureau_a_1: pl.LazyFrame, base: pl.LazyFrame) -> pl.LazyFrame:
    """Contracts reported by credit bureau A.

    Each row of the table carries two independent lists side by side: the i-th active contract (one
    set of columns) and the i-th closed contract (another set). They are aggregated separately.
    The helper column `CLOSED_DETAIL_MISSING` marks the cases whose closed contracts were counted
    by the bureau but not sent; `mask_incomplete_bureau` uses it and drops it.
    """
    active_start = days_before_decision("dateofcredstart_739D")
    closed_start = days_before_decision("dateofcredstart_181D")
    is_active = (active_start >= 0).fill_null(False)
    is_closed = (closed_start >= 0).fill_null(False)
    active_dpd, closed_dpd = pl.col("dpdmax_139P"), pl.col("dpdmax_757P")
    active_dpd_months = months_before_decision("dpdmaxdateyear_596T", "dpdmaxdatemonth_89T")
    closed_dpd_months = months_before_decision("dpdmaxdateyear_896T", "dpdmaxdatemonth_442T")

    n_with_dpd = (is_active & (active_dpd > 0)).sum() + (is_closed & (closed_dpd > 0)).sum()
    n_dpd_known = (is_active & active_dpd.is_not_null()).sum() + (is_closed & closed_dpd.is_not_null()).sum()
    oldest_start = pl.max_horizontal(active_start.filter(is_active).max(), closed_start.filter(is_closed).max())
    newest_start = pl.min_horizontal(active_start.filter(is_active).min(), closed_start.filter(is_closed).min())
    opened_last_year = (is_active & (active_start <= 365)).sum() + (is_closed & (closed_start <= 365)).sum()
    worst_dpd_recency = pl.min_horizontal(
        active_dpd_months.filter(is_active & (active_dpd > 0) & (active_dpd_months >= 0)).min(),
        closed_dpd_months.filter(is_closed & (closed_dpd > 0) & (closed_dpd_months >= 0)).min(),
    )
    # A row whose only contracts start after the decision is dropped, so it leaves no trace at all;
    # rows without any contract stay (they carry the bureau's totals for the case).
    has_contract = pl.col("dateofcredstart_739D").is_not_null() | pl.col("dateofcredstart_181D").is_not_null()
    known_at_decision = with_decision(credit_bureau_a_1, base).filter(~has_contract | is_active | is_closed)
    # The bureau's own totals for the case are stored once per case (first row), with no date.
    n_closed_reported = pl.col("numberofcontrsvalue_358L").max().fill_null(0)
    return known_at_decision.group_by(CASE_ID).agg(
        is_active.sum().alias("cb_n_active_contracts"),
        n_closed_reported.alias("cb_n_closed_contracts"),
        ((n_closed_reported > 0) & (is_closed.sum() == 0)).alias(CLOSED_DETAIL_MISSING),
        pl.col("totaloutstanddebtvalue_39A").max().alias("cb_active_debt"),
        pl.col("monthlyinstlamount_332A").filter(is_active).sum().alias("cb_active_monthly_instalment"),
        pl.col("overdueamount_659A").filter(is_active).sum().alias("cb_active_overdue_amount"),
        active_dpd.filter(is_active).max().alias("cb_active_max_dpd"),
        closed_dpd.filter(is_closed).max().alias("cb_closed_max_dpd"),
        share(n_with_dpd, n_dpd_known).alias("cb_share_contracts_dpd"),
        (oldest_start / DAYS_PER_YEAR).alias("cb_years_since_first_credit"),
        newest_start.alias("cb_days_since_last_credit"),
        opened_last_year.alias("cb_n_credits_12m"),
        worst_dpd_recency.alias("cb_months_since_max_dpd"),
    )


def tax_registry_features(registries: dict[str, pl.LazyFrame], base: pl.LazyFrame) -> pl.LazyFrame:
    """Tax deductions, pooled over the three providers that take turns along the period.

    Only provider-independent quantities are built (counts and a unit-free dispersion), because the
    amounts of one provider are on a different scale from the other two.
    """
    records = pl.concat([
        frame.select(
            CASE_ID,
            pl.lit(name).alias("provider"),
            pl.col(TAX_REGISTRIES[name]["amount"]).alias("amount"),
            pl.col(TAX_REGISTRIES[name]["employer"]).alias("employer"),
            pl.col(TAX_REGISTRIES[name]["date"]).alias("date"),
        )
        for name, frame in registries.items()
    ])
    amount = pl.col("amount")
    per_provider = (
        with_decision(records, base)
        .filter(days_before_decision("date") >= 0)
        .group_by(CASE_ID, "provider")
        .agg(
            pl.len().alias("tax_n_deductions"),
            pl.col("employer").n_unique().alias("tax_n_employers"),
            safe_ratio(amount.std(), amount.mean()).alias("tax_amount_cv"),
        )
    )
    # In the months when one provider replaces another a case can appear in two of them; keep the
    # fuller record instead of adding up the same deductions twice.
    return (
        per_provider.sort("tax_n_deductions", "provider", descending=[True, False])
        .unique(CASE_ID, keep="first", maintain_order=True)
        .drop("provider")
    )


# --------------------------------------------------------------------------------------------------
# Depth 2: history indexed by num_group1 (contract) and num_group2 (monthly payment)
# --------------------------------------------------------------------------------------------------
def bureau_payment_features(credit_bureau_a_2: pl.LazyFrame, base: pl.LazyFrame) -> pl.LazyFrame:
    """Monthly payment status of the bureau contracts, in windows that look back from the decision.

    As in credit_bureau_a_1, each row carries a payment of an active contract and a payment of a
    closed contract side by side. The table lists every scheduled month, including future ones, but
    the days past due are filled only for the months already reported; only those are used.
    """
    active_dpd, closed_dpd = pl.col("pmts_dpd_1073P"), pl.col("pmts_dpd_303P")
    reported = credit_bureau_a_2.filter(active_dpd.is_not_null() | closed_dpd.is_not_null())
    # Months since each reported payment; null when there is no payment or it is after the decision.
    payments = with_decision(reported, base).select(
        CASE_ID,
        active_dpd.alias("active_dpd"),
        closed_dpd.alias("closed_dpd"),
        not_after_decision(months_before_decision("pmts_year_1139T", "pmts_month_158T")).alias("active_months"),
        not_after_decision(months_before_decision("pmts_year_507T", "pmts_month_706T")).alias("closed_months"),
    )
    blocks = [("active_dpd", "active_months"), ("closed_dpd", "closed_months")]

    def max_dpd(window: int) -> pl.Expr:
        return pl.max_horizontal(pl.col(dpd).filter(pl.col(months) < window).max() for dpd, months in blocks)

    def count(window: int, *, late_only: bool) -> pl.Expr:
        selected = [
            pl.col(dpd).is_not_null() & (pl.col(months) < window) & ((pl.col(dpd) > 0) | (not late_only))
            for dpd, months in blocks
        ]
        return pl.sum_horizontal(condition.sum() for condition in selected)

    last_late = pl.min_horizontal(pl.col(months).filter(pl.col(dpd) > 0).min() for dpd, months in blocks)
    return payments.group_by(CASE_ID).agg(
        max_dpd(3).alias("cb_pmt_max_dpd_3m"),
        max_dpd(12).alias("cb_pmt_max_dpd_12m"),
        share(count(12, late_only=True), count(12, late_only=False)).alias("cb_pmt_share_late_12m"),
        last_late.alias("cb_pmt_months_since_last_late"),
    )


# --------------------------------------------------------------------------------------------------
# Features that combine sources
# --------------------------------------------------------------------------------------------------
def cross_source_features(table: pl.LazyFrame) -> pl.LazyFrame:
    """Affordability and exposure ratios between the application, the income and the debts."""
    income = pl.col("mainoccupationinc_384A")
    annuity, amount = pl.col("annuity_780A"), pl.col("credamount_770A")
    return table.with_columns(
        safe_ratio(annuity, income).alias("annuity_to_income"),
        safe_ratio(amount, income).alias("credit_to_income"),
        safe_ratio(annuity + pl.col("cb_active_monthly_instalment"), income).alias("debt_service_to_income"),
        safe_ratio(pl.col("cb_active_debt"), income).alias("cb_debt_to_income"),
        safe_ratio(amount, pl.col("prev_max_credamount")).alias("credit_to_prev_max_credit"),
    )


def mask_incomplete_bureau(table: pl.LazyFrame) -> pl.LazyFrame:
    """Set the features in `NEEDS_CLOSED_DETAIL` to null where the bureau sent no closed-contract
    record despite reporting closed contracts, then drop the helper flag."""
    incomplete = pl.col(CLOSED_DETAIL_MISSING).fill_null(False)
    masked = [pl.when(incomplete).then(None).otherwise(pl.col(feature)).alias(feature) for feature in NEEDS_CLOSED_DETAIL]
    return table.with_columns(masked).drop(CLOSED_DETAIL_MISSING)


# --------------------------------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------------------------------
def feature_blocks(base: pl.LazyFrame, directory: Path = TRAIN_PARQUET_DIR) -> dict[str, pl.LazyFrame]:
    """One lazy block of features per source family, each with one row per case."""
    def scan(name: str) -> pl.LazyFrame:
        return scan_source(name, directory)

    return {
        "static_0": static_0_features(scan("static_0"), base),
        "static_cb_0": static_cb_0_features(scan("static_cb_0")),
        "person_1": applicant_features(scan("person_1"), base),
        "applprev_1": applprev_features(scan("applprev_1"), base),
        "credit_bureau_a_1": bureau_contract_features(scan("credit_bureau_a_1"), base),
        "credit_bureau_a_2": bureau_payment_features(scan("credit_bureau_a_2"), base),
        "tax_registry": tax_registry_features({name: scan(name) for name in TAX_REGISTRIES}, base),
    }


def build_candidate_table(base: pl.DataFrame, directory: Path = TRAIN_PARQUET_DIR) -> pl.DataFrame:
    """`base` plus every candidate feature, still with one row per `case_id`.

    Each block is LEFT JOINed and the join is validated as one-to-one: a block with a duplicated
    `case_id` raises instead of silently multiplying rows.
    """
    table = base
    for block in feature_blocks(base.lazy(), directory).values():
        features = block.collect(engine="streaming")
        table = table.join(features, on=CASE_ID, how="left", validate="1:1")
    table = table.with_columns(pl.col(ZERO_WHEN_NO_HISTORY).fill_null(0))
    return cross_source_features(mask_incomplete_bureau(table.lazy())).collect()


def build_master_table(candidates: pl.DataFrame, selected: list[str]) -> pl.DataFrame:
    """Key columns of `train_base`, the split and the selected features, ordered by `case_id`."""
    return candidates.select(*KEY_COLUMNS, *selected).sort(CASE_ID)
