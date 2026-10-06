# Design Decisions

## D1. Hold-out test set created before EDA
- Decision: stratified 80/20 split before any exploration; EDA uses the training set only.
- Why: keeps the final evaluation honest; exploration cannot leak into test metrics.

## D2. Sensitive attributes (Age, Gender, MaritalStatus)
- Decision: excluded from model inputs by default (schema.SENSITIVE_COLUMNS).
- Plan: in Phase 8, compare performance with and without them, and run a fairness check
  across groups. HR suggestions are never based on these attributes.
- Trade-off: possible small loss of accuracy. JobLevel and TotalWorkingYears can act as proxies
  for age, so exclusion does not remove all influence; this is stated as a limitation.

## D3. Feature engineering as a scikit-learn transformer
- Decision: AttritionFeatureEngineer with fit/transform inside the pipeline.
- Why: pay_vs_level_median needs medians learned on training data only and re-fit in each
  cross-validation fold; the same object is reused by the API and dashboard.
- Rejected: pre-computing features and saving a CSV (would leak test-set statistics).

## D4. Build all ten candidate features, then prune by ablation
- Decision: build every candidate now; in Phase 8 train with and without each feature group
  and keep only what improves cross-validated PR-AUC / recall.
- Note: thresholds (e.g. first year <= 1) came from looking at training data, so they are
  hypotheses to be validated, not conclusions.

## D5. Metrics
- Decision: optimize and report PR-AUC, recall, precision, F1 and ROC-AUC; never accuracy alone,
  because only about 16% of employees leave.
