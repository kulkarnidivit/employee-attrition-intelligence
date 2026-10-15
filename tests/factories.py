import pandas as pd

from tests.helpers import make_row


def make_dataset(n: int = 120) -> pd.DataFrame:
    """Synthetic cleaned dataset with variety in every column and 25% leavers."""
    rows = []
    for i in range(n):
        years = i % 10
        rows.append(
            make_row(i + 1)
            | {
                "Attrition": "Yes" if i % 4 == 0 else "No",
                "Age": 22 + i % 30,
                "BusinessTravel": ["Non-Travel", "Travel_Rarely", "Travel_Frequently"][i % 3],
                "Department": ["Sales", "Research & Development", "Human Resources"][i % 3],
                "EducationField": [
                    "Life Sciences", "Medical", "Marketing",
                    "Technical Degree", "Human Resources", "Other",
                ][i % 6],
                "Gender": ["Male", "Female"][i % 2],
                "JobLevel": 1 + i % 3,
                "JobRole": ["Sales Executive", "Research Scientist", "Laboratory Technician"][i % 3],
                "MaritalStatus": ["Single", "Married", "Divorced"][i % 3],
                "MonthlyIncome": 3000 + 150 * i,
                "OverTime": ["No", "Yes"][i % 2],
                "YearsAtCompany": years,
                "TotalWorkingYears": years + 3,
                "YearsInCurrentRole": min(years, 2),
                "YearsWithCurrManager": min(years, 1),
                "YearsSinceLastPromotion": min(years, 1),
            }
        )
    return pd.DataFrame(rows)
