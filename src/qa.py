"""Quality checks for the master table (brief, sections 9 and 12)."""
import pandas as pd
import polars as pl

from src.config import (
    CASE_ID, DATE_DECISION, KEY_COLUMNS, MONTH, OOT, OOT_MONTHS, SPLIT, SPLIT_VALUES, TARGET, WEEK_NUM,
)

ALMOST_EMPTY_SHARE = 0.95


def profile_columns(frame: pl.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Missing share, number of distinct values and share of the most frequent value per column."""
    n_rows = frame.height
    rows = []
    for column in columns:
        values = frame[column]
        observed = values.drop_nulls()
        top_count = observed.value_counts(sort=True)["count"][0] if observed.len() else 0
        rows.append({
            "feature": column,
            "dtype": str(values.dtype),
            "missing_share": values.null_count() / n_rows,
            "n_distinct": observed.n_unique(),
            "top_value_share": top_count / n_rows,
        })
    profile = pd.DataFrame(rows)
    profile["is_constant"] = profile["n_distinct"] <= 1
    profile["is_almost_empty"] = profile["missing_share"] >= ALMOST_EMPTY_SHARE
    return profile


def check_master_table(master: pl.DataFrame, base: pl.DataFrame) -> pd.DataFrame:
    """Run every structural check and return one row per check (`check`, `passed`, `detail`).

    `base` is `train_base` as read from disk: the master table must have exactly its cases, in the
    same granularity, with untouched key columns.
    """
    checks: list[dict] = []

    def record(check: str, passed: bool, detail: str) -> None:
        checks.append({"check": check, "passed": bool(passed), "detail": detail})

    n_rows, n_cases = master.height, master[CASE_ID].n_unique()
    record("1 case_id = 1 linha", n_rows == n_cases, f"{n_rows:,} linhas, {n_cases:,} case_id distintos")

    missing_keys = [column for column in KEY_COLUMNS if column not in master.columns]
    record("colunas-chave presentes", not missing_keys, f"ausentes: {missing_keys or 'nenhuma'}")
    if missing_keys:
        return pd.DataFrame(checks)

    lost = base.join(master, on=CASE_ID, how="anti").height
    created = master.join(base, on=CASE_ID, how="anti").height
    record(
        "mesma população da train_base", lost == 0 and created == 0,
        f"{lost:,} casos perdidos, {created:,} casos criados",
    )

    base_keys = [DATE_DECISION, WEEK_NUM, MONTH, TARGET]
    compared = master.select(CASE_ID, *base_keys).join(
        base.select(CASE_ID, *base_keys), on=CASE_ID, how="inner", suffix="_base"
    )
    changed = compared.filter(
        pl.any_horizontal(pl.col(column).ne_missing(pl.col(f"{column}_base")) for column in base_keys)
    ).height
    record("chaves e target idênticos à train_base", changed == 0, f"{changed:,} casos com valor alterado")

    target_values = sorted(master[TARGET].unique().to_list(), key=str)
    record(
        "target binário e sem nulos", master[TARGET].null_count() == 0 and target_values == [0, 1],
        f"valores: {target_values}",
    )

    split_values = set(master[SPLIT].unique().to_list())
    record(
        "split definido para todos os casos", split_values <= set(SPLIT_VALUES) and None not in split_values,
        f"valores: {sorted(map(str, split_values))}",
    )

    is_oot_month = pl.col(MONTH).is_in(OOT_MONTHS)
    mismatched = master.filter(is_oot_month != (pl.col(SPLIT) == OOT)).height
    record("OOT = exatamente os meses do enunciado", mismatched == 0, f"{mismatched:,} casos fora da regra")

    features = [column for column in master.columns if column not in KEY_COLUMNS]
    profile = profile_columns(master, features)
    empty = profile.loc[profile["missing_share"] == 1, "feature"].tolist()
    constant = profile.loc[profile["is_constant"] & (profile["missing_share"] < 1), "feature"].tolist()
    record("nenhuma feature totalmente nula", not empty, f"{empty or 'nenhuma'}")
    record("nenhuma feature constante", not constant, f"{constant or 'nenhuma'}")

    dates = [column for column in features if master.schema[column].is_temporal()]
    record("nenhuma data bruta entre as features", not dates, f"{dates or 'nenhuma'}")

    return pd.DataFrame(checks)


def assert_all_passed(checks: pd.DataFrame) -> None:
    failed = checks.loc[~checks["passed"]]
    if len(failed):
        raise AssertionError("master table QA failed:\n" + failed.to_string(index=False))
