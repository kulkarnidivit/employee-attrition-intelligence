"""Shared scoring, explanation and what-if logic used by the API and the dashboard."""

import json

import joblib
import pandas as pd

from src.config import EXPLAINER_PATH, METADATA_PATH, MODEL_PATH, RISK_BANDS


def risk_level(probability: float, bands: dict | None = None) -> str:
    """Map a calibrated probability to Low / Medium / High using configurable cutoffs."""
    bands = bands or RISK_BANDS
    if probability >= bands["high"]:
        return "High"
    if probability >= bands["medium"]:
        return "Medium"
    return "Low"


def linear_contributions(explainer_pipeline, background_mean: pd.Series, X: pd.DataFrame) -> pd.DataFrame:
    """Per-feature contribution to the model's log-odds, relative to an average employee.

    For a linear model these are exactly the SHAP values: coefficient x (value - background mean).
    They describe what the model's output depends on, not what causes attrition.
    """
    transformed = explainer_pipeline[:-1].transform(X)
    coef = explainer_pipeline.named_steps["model"].coef_[0]
    values = (transformed.to_numpy() - background_mean[transformed.columns].to_numpy()) * coef
    return pd.DataFrame(values, columns=transformed.columns, index=transformed.index)


class AttritionService:
    def __init__(self, model, explainer_pipeline, background_mean, metadata: dict):
        self.model = model
        self.explainer_pipeline = explainer_pipeline
        self.background_mean = background_mean
        self.threshold = float(metadata["threshold"])
        self.bands = metadata.get("risk_bands", RISK_BANDS)

    @classmethod
    def load(cls) -> "AttritionService":
        explainer = joblib.load(EXPLAINER_PATH)
        metadata = json.loads(METADATA_PATH.read_text())
        return cls(joblib.load(MODEL_PATH), explainer["pipeline"], explainer["background_mean"], metadata)

    def score(self, X: pd.DataFrame) -> pd.DataFrame:
        """Calibrated probability, risk level and flag for each employee row."""
        probability = self.model.predict_proba(X)[:, 1]
        return pd.DataFrame(
            {
                "probability": probability,
                "risk_level": [risk_level(p, self.bands) for p in probability],
                "flagged": probability >= self.threshold,
            },
            index=X.index,
        )

    def drivers(self, X: pd.DataFrame, k: int = 3) -> list[list[dict]]:
        """Top-k features that most move the model's score for each row (raises / lowers)."""
        contributions = linear_contributions(self.explainer_pipeline, self.background_mean, X)
        result = []
        for _, row in contributions.iterrows():
            top = row.reindex(row.abs().sort_values(ascending=False).index).head(k)
            result.append(
                [
                    {
                        "feature": name,
                        "contribution": float(value),
                        "direction": "raises" if value > 0 else "lowers",
                    }
                    for name, value in top.items()
                ]
            )
        return result

    def simulate(self, row: dict, changes: dict) -> dict:
        """Compare the model's output before and after changing some inputs (a scenario, not a causal claim)."""
        before = self.score(pd.DataFrame([row])).iloc[0]
        after = self.score(pd.DataFrame([{**row, **changes}])).iloc[0]
        return {
            "before": {"probability": float(before["probability"]), "risk_level": before["risk_level"]},
            "after": {"probability": float(after["probability"]), "risk_level": after["risk_level"]},
            "change_in_probability": float(after["probability"] - before["probability"]),
            "note": "Model scenario only: this shows how the model's output changes, "
                    "not that changing these factors would change real outcomes.",
        }


REVIEW_AREAS = {
    "OverTime_Yes": "Workload and scheduling review",
    "work_stress_score": "Workload and travel-load review",
    "BusinessTravel_Travel_Frequently": "Travel-load review",
    "junior_overtime": "Workload review for entry-level staff",
    "is_first_year": "Onboarding and first-year support review",
    "JobSatisfaction": "Engagement and manager conversation",
    "EnvironmentSatisfaction": "Work environment review",
    "WorkLifeBalance": "Work-life balance review",
    "JobInvolvement": "Role fit and engagement review",
    "StockOptionLevel": "Total rewards review",
    "job_hopping": "Career path conversation",
    "JobLevel": "Career progression review",
}


def review_areas(drivers: list[dict]) -> list[str]:
    """HR areas worth a look, based on factors that RAISE this employee's model score.

    Decision support only: these are prompts for a human conversation, not automatic decisions.
    """
    areas = []
    for driver in drivers:
        if driver["direction"] == "raises":
            area = REVIEW_AREAS.get(driver["feature"])
            if area and area not in areas:
                areas.append(area)
    return areas
