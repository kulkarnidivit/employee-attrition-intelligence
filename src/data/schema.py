"""Expected structure of the raw HR dataset.

Validation rules live here (data), not in the checking code (logic).
Ranges are plausibility bounds, deliberately not copied from observed min/max.
"""

ID_COL = "EmployeeNumber"

EXPECTED_COLUMNS = [
    "Age", "Attrition", "BusinessTravel", "DailyRate", "Department",
    "DistanceFromHome", "Education", "EducationField", "EmployeeCount",
    "EmployeeNumber", "EnvironmentSatisfaction", "Gender", "HourlyRate",
    "JobInvolvement", "JobLevel", "JobRole", "JobSatisfaction", "MaritalStatus",
    "MonthlyIncome", "MonthlyRate", "NumCompaniesWorked", "Over18", "OverTime",
    "PercentSalaryHike", "PerformanceRating", "RelationshipSatisfaction",
    "StandardHours", "StockOptionLevel", "TotalWorkingYears",
    "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany",
    "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager",
]

CATEGORICAL_ALLOWED = {
    "Attrition": {"Yes", "No"},
    "BusinessTravel": {"Non-Travel", "Travel_Rarely", "Travel_Frequently"},
    "Department": {"Sales", "Research & Development", "Human Resources"},
    "EducationField": {
        "Life Sciences", "Medical", "Marketing",
        "Technical Degree", "Human Resources", "Other",
    },
    "Gender": {"Male", "Female"},
    "JobRole": {
        "Sales Executive", "Research Scientist", "Laboratory Technician",
        "Manufacturing Director", "Healthcare Representative", "Manager",
        "Sales Representative", "Research Director", "Human Resources",
    },
    "MaritalStatus": {"Single", "Married", "Divorced"},
    "Over18": {"Y"},
    "OverTime": {"Yes", "No"},
}

# (min, max), inclusive
NUMERIC_RANGES = {
    "Age": (18, 70),
    "DailyRate": (1, 1_000_000),
    "DistanceFromHome": (0, 100),
    "Education": (1, 5),
    "EnvironmentSatisfaction": (1, 4),
    "HourlyRate": (1, 1_000_000),
    "JobInvolvement": (1, 4),
    "JobLevel": (1, 5),
    "JobSatisfaction": (1, 4),
    "MonthlyIncome": (1, 1_000_000),
    "MonthlyRate": (1, 1_000_000),
    "NumCompaniesWorked": (0, 30),
    "PercentSalaryHike": (0, 100),
    "PerformanceRating": (1, 4),
    "RelationshipSatisfaction": (1, 4),
    "StockOptionLevel": (0, 3),
    "TotalWorkingYears": (0, 60),
    "TrainingTimesLastYear": (0, 52),
    "WorkLifeBalance": (1, 4),
    "YearsAtCompany": (0, 60),
    "YearsInCurrentRole": (0, 60),
    "YearsSinceLastPromotion": (0, 60),
    "YearsWithCurrManager": (0, 60),
}

# (smaller, larger): the first column must never exceed the second
ORDER_RULES = [
    ("YearsInCurrentRole", "YearsAtCompany"),
    ("YearsSinceLastPromotion", "YearsAtCompany"),
    ("YearsWithCurrManager", "YearsAtCompany"),
    ("YearsAtCompany", "TotalWorkingYears"),
]
