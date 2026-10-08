from datetime import date

import polars as pl
import pytest

from src.config import CASE_ID, MONTH, OOT, SPLIT, TARGET, TRAIN
from src.qa import assert_all_passed, check_master_table, profile_columns
from src.split import assign_split


@pytest.fixture
def master(base) -> pl.DataFrame:
    return assign_split(base).with_columns(
        (pl.col(CASE_ID) % 7).cast(pl.Float64).alias("feature_a"),
        pl.when(pl.col(CASE_ID) % 3 == 0).then(None).otherwise(pl.lit("x")).alias("feature_b"),
        (pl.col(CASE_ID) % 2 == 0).alias("feature_c"),
    )


def failed(checks) -> list[str]:
    return checks.loc[~checks["passed"], "check"].tolist()


def test_valid_master_table_passes_every_check(master, base):
    checks = check_master_table(master, base)
    assert len(checks) == 10
    # feature_b has a single non-null level, which is exactly what "constant" must flag.
    assert failed(checks) == ["nenhuma feature constante"]
    assert_all_passed(check_master_table(master.drop("feature_b"), base))


def test_duplicated_case_breaks_granularity(master, base):
    checks = check_master_table(pl.concat([master, master.head(1)]).drop("feature_b"), base)
    assert "1 case_id = 1 linha" in failed(checks)


def test_lost_case_is_detected(master, base):
    checks = check_master_table(master.drop("feature_b").head(master.height - 5), base)
    assert failed(checks) == ["mesma população da train_base"]
    assert "5 casos perdidos" in checks.set_index("check").loc["mesma população da train_base", "detail"]


def test_changed_target_is_detected(master, base):
    flipped = master.drop("feature_b").with_columns(
        pl.when(pl.col(CASE_ID) == 0).then(1 - pl.col(TARGET)).otherwise(pl.col(TARGET)).alias(TARGET)
    )
    assert failed(check_master_table(flipped, base)) == ["chaves e target idênticos à train_base"]


def test_split_inconsistent_with_the_oot_months_is_detected(master, base):
    wrong = master.drop("feature_b").with_columns(
        pl.when(pl.col(MONTH) == 202007).then(pl.lit(TRAIN)).otherwise(pl.col(SPLIT)).alias(SPLIT)
    )
    assert failed(check_master_table(wrong, base)) == ["OOT = exatamente os meses do enunciado"]
    wrong = master.drop("feature_b").with_columns(
        pl.when(pl.col(MONTH) == 201901).then(pl.lit(OOT)).otherwise(pl.col(SPLIT)).alias(SPLIT)
    )
    assert failed(check_master_table(wrong, base)) == ["OOT = exatamente os meses do enunciado"]


def test_empty_and_date_features_are_detected(master, base):
    broken = master.drop("feature_b").with_columns(
        pl.lit(None, dtype=pl.Float64).alias("empty"), pl.lit(date(2020, 1, 1)).alias("raw_date")
    )
    assert failed(check_master_table(broken, base)) == [
        "nenhuma feature totalmente nula", "nenhuma feature constante", "nenhuma data bruta entre as features",
    ]


def test_missing_key_column_stops_early(master, base):
    checks = check_master_table(master.drop(SPLIT), base)
    assert failed(checks) == ["colunas-chave presentes"]
    with pytest.raises(AssertionError, match="colunas-chave"):
        assert_all_passed(checks)


def test_profile_reports_missing_and_cardinality(master):
    profile = profile_columns(master, ["feature_a", "feature_b", "feature_c"]).set_index("feature")
    assert profile.loc["feature_a", "n_distinct"] == 7 and profile.loc["feature_a", "missing_share"] == 0
    assert profile.loc["feature_b", "missing_share"] == pytest.approx(1 / 3, abs=1e-3)
    assert bool(profile.loc["feature_b", "is_constant"]) and not bool(profile.loc["feature_c", "is_constant"])
