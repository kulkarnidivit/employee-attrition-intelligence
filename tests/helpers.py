def make_row(employee_number: int) -> dict:
    """One valid synthetic employee record, used across tests."""
    return {
        "Age": 35, "Attrition": "No", "BusinessTravel": "Travel_Rarely", "DailyRate": 800,
        "Department": "Sales", "DistanceFromHome": 5, "Education": 3, "EducationField": "Medical",
        "EmployeeCount": 1, "EmployeeNumber": employee_number, "EnvironmentSatisfaction": 3,
        "Gender": "Male", "HourlyRate": 60, "JobInvolvement": 3, "JobLevel": 2,
        "JobRole": "Sales Executive", "JobSatisfaction": 3, "MaritalStatus": "Single",
        "MonthlyIncome": 5000, "MonthlyRate": 14000, "NumCompaniesWorked": 2, "Over18": "Y",
        "OverTime": "No", "PercentSalaryHike": 14, "PerformanceRating": 3,
        "RelationshipSatisfaction": 3, "StandardHours": 80, "StockOptionLevel": 1,
        "TotalWorkingYears": 10, "TrainingTimesLastYear": 3, "WorkLifeBalance": 3,
        "YearsAtCompany": 5, "YearsInCurrentRole": 3, "YearsSinceLastPromotion": 1,
        "YearsWithCurrManager": 2,
    }
