from src.catalog import (
    AUTHORED, CANDIDATE_FEATURES, DISCARD, FEATURES, KEEP, ORIGINAL, SELECTED_FEATURES, SOURCE_NOTES,
    catalog_frame, combined_features, features_of, source_notes_frame,
)
from src.config import KEY_COLUMNS
from src.data import SOURCES
from src.features import PERSON_1_COLUMNS, STATIC_0_COLUMNS, STATIC_CB_0_COLUMNS, build_candidate_table

MIN_AUTHORED_FEATURES = 10  # brief, section 7.3


def test_catalog_describes_exactly_the_features_that_are_built(small_base, raw_dir):
    built = [column for column in build_candidate_table(small_base, raw_dir).columns if column not in KEY_COLUMNS]
    assert CANDIDATE_FEATURES == built
    assert len(set(CANDIDATE_FEATURES)) == len(CANDIDATE_FEATURES)


def test_every_feature_is_documented():
    for spec in FEATURES:
        assert spec.kind in (ORIGINAL, AUTHORED) and spec.decision in (KEEP, DISCARD)
        assert spec.available_at_decision, spec.name
        for text in (spec.source, spec.rule, spec.hypothesis, spec.temporal_check, spec.reason):
            assert text.strip(), spec.name


def test_raw_columns_are_the_ones_marked_as_original():
    originals = {spec.name for spec in FEATURES if spec.kind == ORIGINAL}
    assert originals == set(STATIC_0_COLUMNS + STATIC_CB_0_COLUMNS + PERSON_1_COLUMNS)


def test_selection_keeps_enough_authored_features():
    assert set(SELECTED_FEATURES) < set(CANDIDATE_FEATURES)
    authored_kept = [spec.name for spec in FEATURES if spec.kind == AUTHORED and spec.decision == KEEP]
    assert len(authored_kept) >= MIN_AUTHORED_FEATURES


def test_catalog_and_source_notes_as_tables():
    catalog = catalog_frame()
    assert catalog["feature"].tolist() == CANDIDATE_FEATURES
    assert set(catalog["decisao"]) == {KEEP, DISCARD}
    assert not catalog.isna().any().any()

    assert set(SOURCE_NOTES) == set(SOURCES)
    assert source_notes_frame()["fonte"].tolist() == list(SOURCE_NOTES)

    # Each candidate is listed once: under its source, or among the ones that combine sources.
    by_source = [features_of(source) for source in catalog["fonte"].unique() if " + " not in source]
    assert sum(len(table) for table in by_source) + len(combined_features()) == len(CANDIDATE_FEATURES)
    assert features_of("credit_bureau_a_2")["feature"].str.startswith("cb_pmt_").all()
    assert "annuity_to_income" in combined_features()["feature"].tolist()
