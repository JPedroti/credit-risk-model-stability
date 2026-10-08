"""Project-wide constants: paths, key columns and the TRAIN / TEST / OOT definition.

Everything that defines the experiment lives here so that notebooks, modules and tests share one
source of truth. The temporal definition comes from the brief (section 10) and is fixed before any
feature is built.
"""
from pathlib import Path

# --------------------------------------------------------------------------------------------------
# Paths (relative to the project root, resolved at import time)
# --------------------------------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
TRAIN_PARQUET_DIR = RAW_DIR / "parquet_files" / "train"
FEATURE_DEFINITIONS_PATH = RAW_DIR / "feature_definitions.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MASTER_TABLE_PATH = PROCESSED_DIR / "master_table.parquet"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

SOURCE_INVENTORY_PATH = REPORTS_DIR / "source_inventory.csv"
DATE_AUDIT_PATH = REPORTS_DIR / "date_audit.csv"
FEATURE_CATALOG_PATH = REPORTS_DIR / "feature_catalog.csv"
MASTER_QA_PATH = REPORTS_DIR / "master_table_qa.csv"

# --------------------------------------------------------------------------------------------------
# Key columns of train_base (the spine of the master table)
# --------------------------------------------------------------------------------------------------
CASE_ID = "case_id"
DATE_DECISION = "date_decision"
WEEK_NUM = "WEEK_NUM"
MONTH = "MONTH"
TARGET = "target"
SPLIT = "split"
KEY_COLUMNS = [CASE_ID, DATE_DECISION, WEEK_NUM, MONTH, TARGET, SPLIT]

# --------------------------------------------------------------------------------------------------
# TRAIN / TEST / OOT (brief, section 10)
# --------------------------------------------------------------------------------------------------
# The four months below are the Out-of-Time sample: "the future". They are never used to train, to
# compute selection IV, to choose features or hyperparameters, or to fit any part of the pipeline.
OOT_MONTHS = (202007, 202008, 202009, 202010)
# Every case decided before the first OOT month is split at random, stratified by target.
TEST_SIZE = 0.20
RANDOM_STATE = 42

TRAIN, TEST, OOT = "TRAIN", "TEST", "OOT"
SPLIT_VALUES = (TRAIN, TEST, OOT)

# --------------------------------------------------------------------------------------------------
# Calendar of the raw tables (evidence in notebooks/02_feature_engineering.ipynb, section 1)
# --------------------------------------------------------------------------------------------------
# The organisers transformed every date of the secondary tables. Holidays, month-start effects and
# cross-checks between columns show a constant offset relative to the calendar of `date_decision`:
# day-level dates (suffix D) are 14 days ahead, and (year, month) pairs (suffix T) are 1 month ahead.
# An event happened on or before the decision when `date - 14 days <= date_decision`.
DATE_SHIFT_DAYS = 14
MONTH_SHIFT = 1
# Placeholder the organisers use for a missing value in masked categorical columns (suffix M).
MISSING_CATEGORY = "a55475b1"
