import polars as pl
import pytest

from src.config import CASE_ID, MONTH, OOT, OOT_MONTHS, SPLIT, TARGET, TEST, TEST_SIZE, TRAIN
from src.split import assign_split


def test_oot_is_exactly_the_months_of_the_brief(base):
    split = assign_split(base)
    assert split[SPLIT].null_count() == 0
    assert sorted(split.filter(pl.col(SPLIT) == OOT)[MONTH].unique().to_list()) == list(OOT_MONTHS)
    development = split.filter(pl.col(SPLIT) != OOT)
    assert development[MONTH].max() < min(OOT_MONTHS)
    assert set(development[SPLIT].unique().to_list()) == {TRAIN, TEST}


def test_development_is_split_at_random_and_stratified(base):
    split = assign_split(base)
    train = split.filter(pl.col(SPLIT) == TRAIN)
    test = split.filter(pl.col(SPLIT) == TEST)
    assert test.height / (train.height + test.height) == pytest.approx(TEST_SIZE, abs=1e-3)
    assert train[TARGET].mean() == pytest.approx(test[TARGET].mean(), abs=1e-3)
    assert not set(train[CASE_ID].to_list()) & set(test[CASE_ID].to_list())


def test_split_keeps_rows_and_their_order(base):
    split = assign_split(base)
    assert split.drop(SPLIT).equals(base)


def test_split_does_not_depend_on_the_row_order(base):
    shuffled = base.sample(fraction=1.0, shuffle=True, seed=7)
    expected = assign_split(base).select(CASE_ID, SPLIT).sort(CASE_ID)
    assert assign_split(shuffled).select(CASE_ID, SPLIT).sort(CASE_ID).equals(expected)


def test_split_is_reproducible_and_seed_dependent(base):
    assert assign_split(base).equals(assign_split(base))
    assert not assign_split(base, random_state=1).equals(assign_split(base))


def test_duplicated_case_id_is_rejected(base):
    with pytest.raises(ValueError, match="unique"):
        assign_split(pl.concat([base, base.head(1)]))


def test_month_after_the_first_oot_month_without_a_rule_is_rejected(base):
    late = base.head(1).with_columns(pl.lit(202011).cast(base.schema[MONTH]).alias(MONTH))
    with pytest.raises(ValueError, match="202011"):
        assign_split(pl.concat([base.tail(-1), late]))
