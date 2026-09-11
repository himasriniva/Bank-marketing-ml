# Bank Marketing Machine Learning Project
## Key Findings

- Benchmark: UCI Bank Marketing dataset (bank-additional-full), 41,176 rows after cleaning
- Target is heavily imbalanced: 11.2% subscribed to a term deposit
- `duration` (call length) leaks the outcome - including it inflates ROC-AUC from
  0.78 to 0.95. All realistic modeling excludes it.
- Realistic, deployable model (Logistic Regression, no duration): ROC-AUC 0.80, F1 0.46
- Strongest legitimate predictors: [fill in once you've seen the chart]
- `job` and `age` show strong non-linear relationships with subscribing -
  students and retirees convert far better than working-age professionals