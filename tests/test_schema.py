from src.config import TARGET_COL
from src.data import schema


def test_column_groups_cover_every_feature_exactly_once():
    groups = schema.ORDINAL_COLUMNS + schema.NUMERIC_COLUMNS + schema.CATEGORICAL_COLUMNS
    assert len(groups) == len(set(groups))  # no column in two groups

    expected = (
        set(schema.EXPECTED_COLUMNS)
        - {TARGET_COL, schema.ID_COL}
        - set(schema.CONSTANT_COLUMNS)
    )
    assert set(groups) == expected
