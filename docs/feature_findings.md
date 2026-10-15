# Feature Engineering Findings (Phase 4)

Computed on the **training set only** (1,176 employees, 190 leavers); the test set is untouched.
Associations only, not causes. Several features come from patterns seen while exploring the
training data, so these numbers are optimistic; the locked test set is the final check.

## Method
- Transformer: src/features/engineering.py (AttritionFeatureEngineer); the stateful feature
  (pay_vs_level_median) learns medians on training data only
- Validation: no NaN/inf, ranges and flags checked (find_feature_problems), 11 unit tests
- Check 1: attrition rate and Spearman correlation per feature (all training rows)
- Check 2: the same among established employees only (is_first_year = 0; 1,010 rows, 126 leavers)
  to see whether a feature adds anything beyond "new hire"
- With about 1,000 rows, correlations within roughly +/-0.06 are indistinguishable from zero

## Results (Spearman with attrition)
| Feature | All rows | Established only | Status |
|---|---|---|---|
| junior_overtime | +0.32 | +0.27 | Keep (about 52% attrition vs 12%, n=116) |
| is_first_year | +0.25 | n/a | Keep (39% vs 12%, n=166) |
| work_stress_score | +0.23 | +0.21 | Keep (3% -> 11% -> 24% -> 39% across scores 0 to 3) |
| job_hopping | +0.23 | +0.16 | Keep; may proxy for early career / age |
| promotion_delay_ratio | -0.03 | +0.07 | Weak; test in ablation |
| recent_manager_change | +0.02 | +0.06 | Weak; test in ablation |
| role_stability | -0.15 | -0.02 | Mostly repeats first-year effect; drop candidate |
| manager_stability | -0.14 | 0.00 | Mostly repeats first-year effect; drop candidate |
| tenure_ratio | -0.11 | -0.03 | Mostly repeats first-year effect; drop candidate |
| pay_vs_level_median | -0.08 | -0.01 | Mostly repeats first-year effect; drop candidate |

No pair of engineered features has |Spearman| >= 0.6.

## Conclusions
- Real signal beyond "new hire": junior_overtime, work_stress_score, job_hopping (plus is_first_year itself)
- Tenure, role and manager stability ratios did not show independent signal in this dataset
- Promotion delay and manager change are weak and borderline; keep for ablation only
- Nothing is deleted yet: Spearman only measures steady trends, so the cross-validated ablation in
  Phase 8 makes the final keep/drop decision
- HR suggestions will only reference drivers the final model actually uses; promotion-delay
  advice is not supported by this dataset so far

## Caveats
- Thresholds (first year <= 1, junior = level 1) were chosen after viewing training data
- job_hopping may carry age-like information even though Age is excluded; checked in the Phase 8 fairness analysis
- Synthetic dataset: findings describe this data only
