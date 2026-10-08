"""TRAIN / TEST / OOT assignment (brief, section 10).

The split is defined on `train_base` alone, before any feature exists, and is then carried as a
column of the master table. OOT is a fixed set of calendar months; everything decided before it is
divided at random, stratified by target.
"""
import polars as pl
from sklearn.model_selection import train_test_split

from src.config import (
    CASE_ID, MONTH, OOT, OOT_MONTHS, RANDOM_STATE, SPLIT, TARGET, TEST, TEST_SIZE, TRAIN,
)


def assign_split(
    base: pl.DataFrame,
    *,
    oot_months: tuple[int, ...] = OOT_MONTHS,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> pl.DataFrame:
    """Return `base` with a `split` column in {TRAIN, TEST, OOT}, keeping the input row order.

    - OOT: `MONTH` in `oot_months`.
    - Development: `MONTH` before the first OOT month, split at random and stratified by target.
      Cases are sorted by `case_id` first, so the result depends only on (case_id, target, seed)
      and not on the order of the rows in the file.
    - A month after the first OOT month that is not listed in `oot_months` has no rule in the
      brief, so it raises instead of being silently assigned.
    """
    if base[CASE_ID].n_unique() != base.height:
        raise ValueError("case_id must be unique before assigning the split")

    first_oot_month = min(oot_months)
    is_oot = base[MONTH].is_in(oot_months)
    unassigned = base.filter(~is_oot & (pl.col(MONTH) >= first_oot_month))
    if unassigned.height:
        months = sorted(unassigned[MONTH].unique().to_list())
        raise ValueError(f"months {months} are neither development nor OOT")

    development = base.filter(pl.col(MONTH) < first_oot_month).sort(CASE_ID)
    train_ids, test_ids = train_test_split(
        development[CASE_ID].to_numpy(),
        test_size=test_size,
        stratify=development[TARGET].to_numpy(),
        random_state=random_state,
    )
    labels = pl.concat([
        pl.DataFrame({CASE_ID: train_ids, SPLIT: TRAIN}),
        pl.DataFrame({CASE_ID: test_ids, SPLIT: TEST}),
        base.filter(is_oot).select(CASE_ID, pl.lit(OOT).alias(SPLIT)),
    ]).with_columns(pl.col(CASE_ID).cast(base.schema[CASE_ID]))
    return base.join(labels, on=CASE_ID, how="left", maintain_order="left")
