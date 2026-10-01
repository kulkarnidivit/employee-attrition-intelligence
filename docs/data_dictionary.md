# Data Dictionary

Dataset: IBM HR Analytics Employee Attrition & Performance (1,470 rows, 35 columns).
Coded-value labels follow IBM's commonly cited description of the dataset; the CSV
itself ships no documentation, so they are treated as unverified assumptions.

Roles: TARGET, ID, CONSTANT (drop), FEATURE, SENSITIVE (decision pending),
REVIEW (check signal in EDA).

| Column | Type | Meaning | Role / notes |
|---|---|---|---|
| Age | int | Age in years (18-60) | SENSITIVE |
| Attrition | text | Whether the employee left: Yes / No | TARGET (Yes = left, 16.1%) |
| BusinessTravel | text | Non-Travel / Travel_Rarely / Travel_Frequently | FEATURE (ordered); work-stress candidate |
| DailyRate | int | Rate value (102-1499); meaning not documented | REVIEW, likely low signal |
| Department | text | Sales / Research & Development / Human Resources | FEATURE |
| DistanceFromHome | int | Distance from home to work (1-29); unit not documented | FEATURE; work-stress candidate |
| Education | int | 1 Below College, 2 College, 3 Bachelor, 4 Master, 5 Doctor | FEATURE (ordinal) |
| EducationField | text | Field of study (6 categories) | FEATURE |
| EmployeeCount | int | Always 1 | CONSTANT, drop |
| EmployeeNumber | int | Unique employee identifier | ID, not a feature; keep for risk table / Employee 360 |
| EnvironmentSatisfaction | int | 1 Low, 2 Medium, 3 High, 4 Very High | FEATURE (ordinal) |
| Gender | text | Male / Female | SENSITIVE |
| HourlyRate | int | Rate value (30-100); meaning not documented | REVIEW |
| JobInvolvement | int | 1 Low ... 4 Very High | FEATURE (ordinal) |
| JobLevel | int | Seniority level 1-5 | FEATURE; promotion-delay candidate |
| JobRole | text | 9 job roles | FEATURE |
| JobSatisfaction | int | 1 Low ... 4 Very High | FEATURE (ordinal) |
| MaritalStatus | text | Single / Married / Divorced | SENSITIVE |
| MonthlyIncome | int | Monthly income (1009-19999) | FEATURE |
| MonthlyRate | int | Rate value (2094-26999); meaning not documented | REVIEW |
| NumCompaniesWorked | int | Number of previous companies (0-9) | FEATURE; career-mobility candidate |
| Over18 | text | Always Y | CONSTANT, drop |
| OverTime | text | Works overtime: Yes / No | FEATURE; work-stress candidate |
| PercentSalaryHike | int | Last salary hike percent (11-25) | FEATURE |
| PerformanceRating | int | Observed values 3 and 4 only | FEATURE; very low variation |
| RelationshipSatisfaction | int | 1 Low ... 4 Very High | FEATURE (ordinal) |
| StandardHours | int | Always 80 | CONSTANT, drop |
| StockOptionLevel | int | 0-3 | FEATURE (ordinal) |
| TotalWorkingYears | int | Total career years (0-40) | FEATURE |
| TrainingTimesLastYear | int | Trainings attended last year (0-6) | FEATURE |
| WorkLifeBalance | int | 1 Bad, 2 Good, 3 Better, 4 Best | FEATURE (ordinal); work-stress candidate |
| YearsAtCompany | int | Years at current company (0-40) | FEATURE; tenure candidate |
| YearsInCurrentRole | int | Years in current role (0-18) | FEATURE; role-stability candidate |
| YearsSinceLastPromotion | int | Years since last promotion (0-15) | FEATURE; promotion-delay candidate |
| YearsWithCurrManager | int | Years with current manager (0-17) | FEATURE; manager-stability candidate |

## Initial quality findings
- No missing values; no duplicate rows
- Target imbalance: 1,233 No / 237 Yes
- 3 constant columns to drop; 1 ID column to exclude from model inputs
- 3 "rate" columns with undocumented meaning, to be checked in EDA
- Text columns load as pandas `str` dtype (not `object`)
- Consistency checks planned for Phase 2: tenure/role/manager/promotion years
  should not exceed YearsAtCompany; YearsAtCompany should not exceed TotalWorkingYears
