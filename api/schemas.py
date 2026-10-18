"""Request and response models. Age, gender and marital status are deliberately not accepted."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

JobRoleName = Literal[
    "Sales Executive", "Research Scientist", "Laboratory Technician", "Manufacturing Director",
    "Healthcare Representative", "Manager", "Sales Representative", "Research Director", "Human Resources",
]

EXAMPLE = {
    "BusinessTravel": "Travel_Rarely", "Department": "Sales", "EducationField": "Medical",
    "JobRole": "Sales Executive", "OverTime": "Yes", "DailyRate": 800, "DistanceFromHome": 5,
    "HourlyRate": 60, "MonthlyIncome": 3000, "MonthlyRate": 14000, "NumCompaniesWorked": 2,
    "PercentSalaryHike": 14, "TotalWorkingYears": 4, "TrainingTimesLastYear": 3, "YearsAtCompany": 1,
    "YearsInCurrentRole": 1, "YearsSinceLastPromotion": 0, "YearsWithCurrManager": 1, "Education": 3,
    "EnvironmentSatisfaction": 2, "JobInvolvement": 3, "JobLevel": 1, "JobSatisfaction": 2,
    "PerformanceRating": 3, "RelationshipSatisfaction": 3, "StockOptionLevel": 0, "WorkLifeBalance": 2,
}


class EmployeeInput(BaseModel):
    model_config = {"json_schema_extra": {"example": EXAMPLE}}

    BusinessTravel: Literal["Non-Travel", "Travel_Rarely", "Travel_Frequently"]
    Department: Literal["Sales", "Research & Development", "Human Resources"]
    EducationField: Literal["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"]
    JobRole: JobRoleName
    OverTime: Literal["Yes", "No"]
    DailyRate: int = Field(ge=1)
    DistanceFromHome: int = Field(ge=0, le=100)
    HourlyRate: int = Field(ge=1)
    MonthlyIncome: int = Field(ge=1)
    MonthlyRate: int = Field(ge=1)
    NumCompaniesWorked: int = Field(ge=0, le=30)
    PercentSalaryHike: int = Field(ge=0, le=100)
    TotalWorkingYears: int = Field(ge=0, le=60)
    TrainingTimesLastYear: int = Field(ge=0, le=52)
    YearsAtCompany: int = Field(ge=0, le=60)
    YearsInCurrentRole: int = Field(ge=0, le=60)
    YearsSinceLastPromotion: int = Field(ge=0, le=60)
    YearsWithCurrManager: int = Field(ge=0, le=60)
    Education: int = Field(ge=1, le=5)
    EnvironmentSatisfaction: int = Field(ge=1, le=4)
    JobInvolvement: int = Field(ge=1, le=4)
    JobLevel: int = Field(ge=1, le=5)
    JobSatisfaction: int = Field(ge=1, le=4)
    PerformanceRating: int = Field(ge=1, le=4)
    RelationshipSatisfaction: int = Field(ge=1, le=4)
    StockOptionLevel: int = Field(ge=0, le=3)
    WorkLifeBalance: int = Field(ge=1, le=4)

    @model_validator(mode="after")
    def check_years_are_consistent(self):
        for name in ("YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager"):
            if getattr(self, name) > self.YearsAtCompany:
                raise ValueError(f"{name} cannot exceed YearsAtCompany")
        if self.YearsAtCompany > self.TotalWorkingYears:
            raise ValueError("YearsAtCompany cannot exceed TotalWorkingYears")
        return self


class PredictResponse(BaseModel):
    probability: float
    risk_level: str
    flagged: bool
    cutoff: float


class Driver(BaseModel):
    feature: str
    contribution: float
    direction: str


class ExplainResponse(BaseModel):
    probability: float
    risk_level: str
    drivers: list[Driver]
    review_areas: list[str]
    note: str


class SimulateRequest(BaseModel):
    employee: EmployeeInput
    changes: dict[str, str | int] = Field(min_length=1)

    @field_validator("changes")
    @classmethod
    def only_known_fields(cls, changes):
        unknown = [k for k in changes if k not in EmployeeInput.model_fields]
        if unknown:
            raise ValueError(f"Unknown fields: {unknown}")
        return changes


class ScoreSummary(BaseModel):
    probability: float
    risk_level: str


class SimulateResponse(BaseModel):
    before: ScoreSummary
    after: ScoreSummary
    change_in_probability: float
    note: str
