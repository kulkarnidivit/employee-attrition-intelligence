# EDA Findings (Phase 3)

All figures come from the **training set only** (1,176 employees, 190 leavers, about 16% attrition).
The test set was split off before EDA and has not been used. Findings are **associations in a
synthetic dataset, not causes**, and do not describe real companies.

## Method
- Attrition rate per group with 95% Wilson intervals and group sizes (wide interval = weak evidence)
- Spearman correlation between features and with the target
- Two-way tables for specific HR questions
- Cut-offs (for example "<= 1 year") were chosen after looking at the training data, so they are
  hypotheses to validate with cross-validation, not conclusions.

## Data quality
- No missing values, no duplicate rows; real data passes all validation rules
- 3 constant columns dropped (EmployeeCount, Over18, StandardHours); EmployeeNumber kept only as an ID
- Long tails in income and tenure look genuine; outliers were not removed

## Target
- About 16% of employees left: imbalanced, so use stratified cross-validation and PR-AUC / recall

## Categorical features
- OverTime: about 28% (Yes) vs 11% (No), the clearest categorical signal
- JobRole: spread from about 1% to 41%; highest Sales Representative (n=60) and Laboratory Technician
- MaritalStatus: Single about 25% vs about 11-12% for Married / Divorced
- BusinessTravel: steady step, Non-Travel about 7%, Travel_Rarely about 15%, Travel_Frequently about 25%
- Department: Sales about 21% vs R&D about 13%; HR (n=48) inconclusive
- EducationField: weak; Gender: no clear difference

## Ordinal and numeric features
- Strong: YearsWithCurrManager <= 1 (about 31%), Age <= 29 (about 29%), JobLevel 1 (about 27%), StockOptionLevel 0 (about 24%)
- Steady decline with higher JobSatisfaction and JobInvolvement; low EnvironmentSatisfaction and WorkLifeBalance (level 1) stand out (small groups)
- Modest: DistanceFromHome above 9; weak: DailyRate
- Little or none: PerformanceRating, Education, RelationshipSatisfaction, raw YearsSinceLastPromotion
- HourlyRate and MonthlyRate look like noise

## Feature overlap
- Age, TotalWorkingYears, JobLevel, MonthlyIncome and the Years* tenure columns form one correlated block
  (up to Spearman 0.92 for MonthlyIncome / JobLevel)
- Strongest numeric correlations with attrition are only about -0.2; no single numeric feature dominates

## Combinations (HR questions)
- New hires (<= 1 year): about 39% attrition (n=166), about a third of all leavers
- OverTime and BusinessTravel stack: about 3% (no overtime, non-travel) to about 39% (overtime, frequent travel)
- Junior level (JobLevel 1) + overtime: about 52% (n=116); overtime gap is much larger at level 1
- Junior level + age <= 29: about 37% (n=175); age effect mainly visible at level 1
- Pay below the level median: higher attrition only at level 1 (34% vs 20%); likely overlaps with young / junior

## Implications for feature engineering (Phase 4)
Candidates to build and test (evidence strength in brackets):
1. First-year / tenure-band feature (strong)
2. Work Stress score from OverTime, BusinessTravel, possibly WorkLifeBalance and DistanceFromHome (strong)
3. Overtime x junior-level interaction, needed explicitly for Logistic Regression (strong)
4. Manager-change indicator, YearsWithCurrManager below YearsAtCompany (weak)
5. Pay relative to job level, medians fit on training data only inside the pipeline (promising, overlaps with age)
6. Relative promotion delay (raw column showed no pattern; test before using it for HR insights)

Likely drop candidates: HourlyRate, MonthlyRate, PerformanceRating (decide with cross-validation).

## Modeling notes
- Use stratified cross-validation, PR-AUC, recall and precision, not accuracy
- Regularize Logistic Regression because of correlated features; expect SHAP credit to be shared among them
- Expect moderate performance: no single feature dominates, signal comes from combinations

## Sensitive attributes
Gender, MaritalStatus and Age are sensitive. Default plan: exclude them from model inputs, then measure in
Phase 8 how performance changes when included, and check fairness at evaluation. Final decision in Phase 4/5.
HR suggestions will never be based on these attributes.

## Limitations
- Synthetic dataset: findings describe this data only
- Small groups give wide intervals; some cells have n < 100
- Many comparisons were made, so some patterns will be chance
- Cut-offs were chosen after viewing the training data
- Associations only: no causal claims
