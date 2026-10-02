"""Validate the raw HR dataset against the expected schema and business rules.

Validation only detects and reports problems. It never modifies the data.
"""

import logging
import sys
from dataclasses import dataclass, field

import pandas as pd

from src.data import schema
from src.data.load_data import load_raw_data

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Issue:
    severity: str  # "error" or "warning"
    check: str
    message: str


@dataclass
class ValidationReport:
    issues: list[Issue] = field(default_factory=list)

    def add(self, severity: str, check: str, message: str) -> None:
        self.issues.append(Issue(severity, check, message))

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def is_valid(self) -> bool:
        return not self.errors


class DataValidationError(Exception):
    """Raised when validation finds one or more errors."""


def _sample_ids(df: pd.DataFrame, mask: pd.Series, n: int = 5) -> str:
    if schema.ID_COL in df.columns:
        return f" (e.g. {schema.ID_COL}: {df.loc[mask, schema.ID_COL].head(n).tolist()})"
    return ""


def _check_columns(df: pd.DataFrame, report: ValidationReport) -> None:
    missing = sorted(set(schema.EXPECTED_COLUMNS) - set(df.columns))
    unexpected = sorted(set(df.columns) - set(schema.EXPECTED_COLUMNS))
    if missing:
        report.add("error", "columns", f"Missing expected columns: {missing}")
    if unexpected:
        report.add("error", "columns", f"Unexpected columns: {unexpected}")


def _check_missing_values(df: pd.DataFrame, report: ValidationReport) -> None:
    nulls = df.isna().sum()
    for col, n in nulls[nulls > 0].items():
        report.add("error", "missing_values", f"'{col}' has {n} missing value(s)")


def _check_categories(df: pd.DataFrame, report: ValidationReport) -> None:
    for col, allowed in schema.CATEGORICAL_ALLOWED.items():
        if col not in df.columns:
            continue
        bad = df[col].notna() & ~df[col].isin(allowed)
        if bad.any():
            values = sorted(df.loc[bad, col].astype(str).unique())[:5]
            report.add(
                "error", "categories",
                f"'{col}' has {int(bad.sum())} row(s) with unexpected values {values}",
            )


def _check_ranges(df: pd.DataFrame, report: ValidationReport) -> None:
    for col, (lo, hi) in schema.NUMERIC_RANGES.items():
        if col not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            report.add("error", "ranges", f"'{col}' should be numeric but is {df[col].dtype}")
            continue
        bad = df[col].notna() & ~df[col].between(lo, hi)
        if bad.any():
            report.add(
                "error", "ranges",
                f"'{col}' has {int(bad.sum())} value(s) outside [{lo}, {hi}]"
                + _sample_ids(df, bad),
            )


def _check_order_rules(df: pd.DataFrame, report: ValidationReport) -> None:
    for smaller, larger in schema.ORDER_RULES:
        if smaller not in df.columns or larger not in df.columns:
            continue
        if not (pd.api.types.is_numeric_dtype(df[smaller]) and pd.api.types.is_numeric_dtype(df[larger])):
            continue
        bad = df[smaller] > df[larger]
        if bad.any():
            report.add(
                "error", "order_rules",
                f"{int(bad.sum())} row(s) where {smaller} > {larger}" + _sample_ids(df, bad),
            )


def _check_id_unique(df: pd.DataFrame, report: ValidationReport) -> None:
    if schema.ID_COL in df.columns and df[schema.ID_COL].duplicated().any():
        n = int(df[schema.ID_COL].duplicated().sum())
        report.add("error", "id_unique", f"'{schema.ID_COL}' has {n} duplicated value(s)")


def _check_known_warnings(df: pd.DataFrame, report: ValidationReport) -> None:
    constant = [c for c in df.columns if df[c].nunique(dropna=False) == 1]
    if constant:
        report.add("warning", "constant_columns", f"{len(constant)} column(s) have a single value: {constant}")
    n_dup = int(df.duplicated().sum())
    if n_dup:
        report.add("warning", "duplicate_rows", f"{n_dup} duplicated row(s)")


def validate_raw_data(df: pd.DataFrame) -> ValidationReport:
    """Run all checks and return a report (does not raise)."""
    report = ValidationReport()
    _check_columns(df, report)
    _check_missing_values(df, report)
    _check_categories(df, report)
    _check_ranges(df, report)
    _check_order_rules(df, report)
    _check_id_unique(df, report)
    _check_known_warnings(df, report)
    logger.info("Validation finished: %d error(s), %d warning(s)", len(report.errors), len(report.warnings))
    return report


def raise_if_invalid(report: ValidationReport) -> None:
    """Raise DataValidationError if the report contains any errors."""
    if not report.is_valid:
        details = "; ".join(f"[{i.check}] {i.message}" for i in report.errors)
        raise DataValidationError(f"{len(report.errors)} validation error(s): {details}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    report = validate_raw_data(load_raw_data())

    print("\n=== VALIDATION REPORT ===")
    for issue in report.issues:
        print(f"[{issue.severity.upper()}] {issue.check}: {issue.message}")
    status = "PASSED" if report.is_valid else "FAILED"
    print(f"\nResult: {status} ({len(report.errors)} errors, {len(report.warnings)} warnings)")
    sys.exit(0 if report.is_valid else 1)


if __name__ == "__main__":
    main()
