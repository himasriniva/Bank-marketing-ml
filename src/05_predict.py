"""
Predict the subscription probability for a single sample row using the
saved no-duration Logistic Regression model.

Input:  data/processed/bank_marketing_clean.csv
        models/logistic_regression_no_duration.joblib
        models/scaler_no_duration.joblib
        models/feature_columns_no_duration.joblib
Output: none (prints the predicted probability to stdout)
"""
import joblib
import pandas as pd

model = joblib.load("models/logistic_regression_no_duration.joblib")
scaler = joblib.load("models/scaler_no_duration.joblib")
feature_columns = joblib.load("models/feature_columns_no_duration.joblib")

df = pd.read_csv("data/processed/bank_marketing_clean.csv")
sample = df.sample(n=1, random_state=42).drop(columns=["y", "duration", "pdays"])

# One-hot encode the same way as training, then align to the saved column
# order (missing dummy columns - categories not present in this row - become 0)
sample_encoded = pd.get_dummies(sample, columns=sample.select_dtypes(include="object").columns)
sample_aligned = sample_encoded.reindex(columns=feature_columns, fill_value=0)

sample_scaled = scaler.transform(sample_aligned)
probability = model.predict_proba(sample_scaled)[0, 1]

print(f"Predicted probability of subscribing: {probability:.4f}")
