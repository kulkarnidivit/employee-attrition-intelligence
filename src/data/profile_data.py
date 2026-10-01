"""Print a structural profile of the raw dataset (dtype, uniqueness, missing, range)."""

import logging

import pandas as pd

from src.config import TARGET_COL
from src.data.load_data import load_raw_data


def profile_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per column describing its structure."""
    rows = []
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            summary = f"min={s.min()}, max={s.max()}"
        else:
            summary = ", ".join(map(str, s.value_counts().index[:4]))
        rows.append(
            {
                "column": col,
                "dtype": str(s.dtype),
                "n_unique": s.nunique(),
                "n_missing": int(s.isna().sum()),
                "summary": summary,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    df = load_raw_data()

    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 70)

    print("\n=== COLUMN PROFILE ===")
    print(profile_columns(df).to_string(index=False))

    print("\n=== DUPLICATE ROWS ===")
    print(df.duplicated().sum())

    print("\n=== TARGET DISTRIBUTION ===")
    print(df[TARGET_COL].value_counts(dropna=False))
    print(df[TARGET_COL].value_counts(normalize=True, dropna=False).round(3))


if __name__ == "__main__":
    main()
