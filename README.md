# Bank Marketing Machine Learning Project

Predicts whether a bank customer will subscribe to a term deposit, based on
the UCI Bank Marketing dataset (direct marketing phone campaigns run by a
Portuguese bank). The project walks through cleaning the raw data, exploring
it, training and comparing classifiers, and using the saved model to score
new rows.

## Dataset

- Source: UCI Bank Marketing dataset (`bank-additional-full`)
- 41,176 rows after cleaning, one row per client contacted in a campaign
- Target `y`: whether the client subscribed to a term deposit (yes/no)
- Heavily imbalanced: ~11.2% positive class

## Pipeline

The scripts in `src/` run in order, each reading the previous stage's output:

1. **`01_load_and_inspect.py`** - inspects the raw CSV (shape, dtypes, missing
   values, class balance) and prints the results; no output file.
2. **`02_clean_data.py`** - cleans the raw data (dedupes, keeps `unknown` as
   its own category, adds a `was_contacted_before` flag) ->
   `data/processed/bank_marketing_clean.csv`
3. **`03_eda.py`** - generates exploratory charts (class balance, duration
   leakage, job/age patterns, correlation heatmap) -> `results/figures/`
4. **`04_train_models.py`** - trains and compares Random Forest (with/without
   the leaky `duration` feature) and Logistic Regression, then saves the
   realistic (no-duration) Logistic Regression model, scaler, and feature
   columns -> `models/`
5. **`05_predict.py`** - loads the saved model/scaler/feature columns and
   predicts the subscription probability for a sample row.

## Setup

```bash
python -m venv venv
venv\Scripts\activate      # on Windows
pip install -r requirements.txt
```

## Running

From the project root, run the stages in order:

```bash
python src/01_load_and_inspect.py
python src/02_clean_data.py
python src/03_eda.py
python src/04_train_models.py
python src/05_predict.py
```

## Key Findings

- Benchmark: UCI Bank Marketing dataset (bank-additional-full), 41,176 rows after cleaning
- Target is heavily imbalanced: 11.2% subscribed to a term deposit
- `duration` (call length) leaks the outcome - including it inflates ROC-AUC from
  0.78 to 0.95. All realistic modeling excludes it.
- Realistic, deployable model (Logistic Regression, no duration): ROC-AUC 0.80, F1 0.46
- Strongest legitimate predictors: [fill in once you've seen the chart]
- `job` and `age` show strong non-linear relationships with subscribing -
  students and retirees convert far better than working-age professionals
