"""Demo employee data for the API and frontend: the (synthetic) HR dataset scored by the final model."""

import pandas as pd

from src.config import PROJECT_ROOT
from src.serving.service import AttritionService, linear_contributions, review_areas

DEMO_DATA_PATH = PROJECT_ROOT / "data" / "demo" / "employees.csv"

LIST_FIELDS = ["Department", "JobRole", "JobLevel", "YearsAtCompany", "OverTime"]
PROFILE_FIELDS = [
    "Department", "JobRole", "JobLevel", "YearsAtCompany", "YearsInCurrentRole",
    "YearsWithCurrManager", "OverTime", "BusinessTravel", "MonthlyIncome",
]
LEVELS = ["Low", "Medium", "High"]


def _py(value):
    return value.item() if hasattr(value, "item") else value


class DemoStore:
    def __init__(self, service: AttritionService, data: pd.DataFrame, input_fields: list[str]):
        self.service = service
        self.input_fields = input_fields
        self.data = data.reset_index(drop=True)
        self.features = self.data.drop(columns=["Attrition", "in_training"])
        scores = service.score(self.features)
        self.table = self.data[["EmployeeNumber", *LIST_FIELDS, "in_training"]].join(scores)
        self.table["left"] = (self.data["Attrition"] == "Yes").to_numpy()
        contributions = linear_contributions(service.explainer_pipeline, service.background_mean, self.features)
        self.global_drivers = contributions.abs().mean().sort_values(ascending=False).head(15)

    @classmethod
    def load(cls, service: AttritionService, input_fields: list[str], path=DEMO_DATA_PATH) -> "DemoStore":
        return cls(service, pd.read_csv(path), input_fields)

    def summary(self) -> dict:
        by_level = (
            self.table.groupby("risk_level")
            .agg(employees=("left", "size"), actual_attrition_rate=("left", "mean"))
            .reindex(LEVELS)
            .fillna(0)
        )
        return {
            "employees": int(len(self.table)),
            "actual_attrition_rate": float(self.table["left"].mean()),
            "flagged": int(self.table["flagged"].sum()),
            "by_risk_level": [
                {
                    "risk_level": level,
                    "employees": int(row["employees"]),
                    "actual_attrition_rate": float(row["actual_attrition_rate"]),
                }
                for level, row in by_level.iterrows()
            ],
            "note": "Synthetic demo data. Employees the model trained on are marked in_training; "
                    "their scores look better than they would on new people.",
        }

    def drivers(self) -> list[dict]:
        return [{"feature": name, "importance": float(value)} for name, value in self.global_drivers.items()]

    def list_employees(self, risk_level: str | None, limit: int, offset: int) -> dict:
        view = self.table if risk_level is None else self.table[self.table["risk_level"] == risk_level]
        view = view.sort_values("probability", ascending=False)
        part = view.iloc[offset : offset + limit]
        items = [{k: _py(v) for k, v in record.items()} for record in part.to_dict("records")]
        return {"total": int(len(view)), "items": items}

    def employee(self, employee_number: int) -> dict | None:
        matches = self.data.index[self.data["EmployeeNumber"] == employee_number]
        if len(matches) == 0:
            return None
        idx = matches[0]
        row = self.features.loc[[idx]]
        scored = self.service.score(row).iloc[0]
        drivers = self.service.drivers(row, k=5)[0]
        return {
            "employee_number": int(employee_number),
            "in_training": bool(self.data.loc[idx, "in_training"]),
            "profile": {f: _py(self.features.loc[idx, f]) for f in PROFILE_FIELDS},
            "score": {
                "probability": float(scored["probability"]),
                "risk_level": scored["risk_level"],
                "flagged": bool(scored["flagged"]),
                "cutoff": self.service.threshold,
            },
            "drivers": drivers,
            "review_areas": review_areas(drivers),
            "record": {f: _py(self.features.loc[idx, f]) for f in self.input_fields},
        }
