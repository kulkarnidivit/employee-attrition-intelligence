# Project Scope

## Goal
Predict employee attrition risk from a snapshot of HR data, explain the model's
predictions, and provide HR decision support.

## In scope
- Attrition probability per employee
- Configurable risk levels (Low / Medium / High)
- SHAP explanations (global and per employee)
- Dashboard, Employee 360 view, what-if simulator
- FastAPI service, tests, Docker

## Out of scope
- Causal claims about why employees leave
- Automated HR decisions (output is decision support only)
- Monitoring on real live data (drift will be simulated)
- PostgreSQL / cloud infrastructure until they add real value

## Success criteria
- Model beats a naive baseline on PR-AUC and recall (not accuracy alone)
- No data leakage: split first, fit preprocessing/resampling on training data only
- All explanations and simulations are worded as model behavior, not causation
- Reproducible: a new developer can clone the repo and rerun everything
- Tests pass; API runs in Docker

## Dataset
IBM HR Analytics Employee Attrition & Performance (Kaggle).
1,470 rows, 35 columns, target `Attrition` (Yes/No).

### Known limitations
- Synthetic data: findings describe this dataset, not real companies
- Small sample (~237 leavers): metrics are noisy, cross-validation is essential
- No manager IDs or team structure
- No dates or time dimension: drift/monitoring/retraining will be simulated
- Several constant or low-signal columns (verify in EDA)
- Sensitive attributes (Age, Gender, MaritalStatus): default plan is to exclude
  Gender and MaritalStatus from model inputs; to be revisited in Phase 4/5
