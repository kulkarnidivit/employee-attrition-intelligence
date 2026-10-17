"""Error analysis on the out-of-fold predictions (training set only).

Sensitive columns are used here only to CHECK for unequal treatment; the model never sees them.
"""

import logging

import pandas as pd

from src.config import REPORTS_DIR
from src.features.engineering import AttritionFeatureEngineer
from src.models.data import load_train_xy
from src.models.error_analysis import flag_top_share, group_report

MODEL = "logistic_regression"
FLAG_SHARE = 0.16

GROUP_COLUMNS = [
    "OverTime", "BusinessTravel", "is_first_year", "junior_overtime", "work_stress_score",
    "Department", "JobRole",
    "Gender", "MaritalStatus", "age_band",  # evaluation only, not model inputs
]
PROFILE_COLUMNS = [
    "overtime_yes", "is_first_year", "junior_overtime", "work_stress_score",
    "JobLevel", "MonthlyIncome", "TotalWorkingYears", "YearsAtCompany",
]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    X, y = load_train_xy()
    oof = pd.read_csv(REPORTS_DIR / "oof_predictions.csv")
    if not (oof["EmployeeNumber"].to_numpy() == X["EmployeeNumber"].to_numpy()).all():
        raise ValueError("oof_predictions.csv does not line up with the training data; re-run run_evaluation")

    df = AttritionFeatureEngineer().fit_transform(X)
    df["attrition"] = y.to_numpy()
    df["score"] = oof[MODEL].to_numpy()
    df["flagged"] = flag_top_share(df["score"], FLAG_SHARE)
    df["age_band"] = pd.cut(df["Age"], bins=[0, 29, 39, 200], labels=["29 or under", "30-39", "40+"])
    df["overtime_yes"] = (df["OverTime"] == "Yes").astype(int)

    n_flagged = int(df["flagged"].sum())
    caught = int(((df["attrition"] == 1) & df["flagged"]).sum())
    leavers = int(df["attrition"].sum())
    print(f"Model: {MODEL}. Flagging the top {FLAG_SHARE:.0%} by score = {n_flagged} employees")
    print(f"Real leavers: {leavers}. Caught: {caught}. Missed: {leavers - caught}. "
          f"False alarms: {n_flagged - caught}")

    leaver_rows = df[df["attrition"] == 1].copy()
    leaver_rows["group"] = leaver_rows["flagged"].map({True: "caught", False: "missed"})
    print("\n=== CAUGHT vs MISSED LEAVERS (average of each column) ===")
    print(leaver_rows.groupby("group")[PROFILE_COLUMNS].mean().T.round(2).to_string())

    for col in GROUP_COLUMNS:
        print(f"\n=== BY {col} ===")
        print(group_report(df, col).round(2).to_string(index=False))

    df[["EmployeeNumber", "attrition", "score", "flagged"]].to_csv(
        REPORTS_DIR / "error_analysis_flags.csv", index=False
    )


if __name__ == "__main__":
    main()
