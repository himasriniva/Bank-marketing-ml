"""
Train and compare classifiers on the cleaned Bank Marketing data: a Random
Forest with and without the leaky 'duration' feature, then a Logistic
Regression on the realistic (no-duration) version, plus a Random Forest
feature-importance chart for that same version.

Input:  data/processed/bank_marketing_clean.csv
Output: results/figures/06_feature_importance_realistic.png
        models/logistic_regression_no_duration.joblib
        models/scaler_no_duration.joblib
        models/feature_columns_no_duration.joblib
        (metrics printed to stdout)
"""
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, classification_report)

df = pd.read_csv("data/processed/bank_marketing_clean.csv")

# Drop pdays - it's redundant with was_contacted_before (near -1 correlation, Chart 1)
df = df.drop(columns=["pdays"])

# One-hot encode all categorical columns
df_encoded = pd.get_dummies(df, columns=df.select_dtypes(include="object").columns.drop("y"))
y = df_encoded["y"].map({"yes": 1, "no": 0})
X_full = df_encoded.drop(columns=["y"])

def train_and_evaluate(X, y, label):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)
    proba = model.predict_proba(X_test_s)[:, 1]

    print(f"\n=== {label} ===")
    print(f"Accuracy:  {accuracy_score(y_test, preds):.4f}")
    print(f"Precision: {precision_score(y_test, preds):.4f}")
    print(f"Recall:    {recall_score(y_test, preds):.4f}")
    print(f"F1:        {f1_score(y_test, preds):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, proba):.4f}")
    print(classification_report(y_test, preds, target_names=["no", "yes"]))

# Version 1: WITH duration (matches most published benchmarks — but leaks the outcome)
train_and_evaluate(X_full, y, "WITH duration (leaky, benchmark-style)")

# Version 2: WITHOUT duration (the realistic, deployable version)
X_no_duration = X_full.drop(columns=["duration"])
train_and_evaluate(X_no_duration, y, "WITHOUT duration (realistic, deployable)")
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Compare Random Forest vs Logistic Regression on the REALISTIC (no-duration) version
X_train, X_test, y_train, y_test = train_test_split(
    X_no_duration, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

log_reg = LogisticRegression(max_iter=5000, random_state=42, class_weight="balanced")
log_reg.fit(X_train_s, y_train)
proba_lr = log_reg.predict_proba(X_test_s)[:, 1]
preds_lr = log_reg.predict(X_test_s)
print(f"\n=== Logistic Regression (no duration) ===")
print(f"F1: {f1_score(y_test, preds_lr):.4f} | ROC-AUC: {roc_auc_score(y_test, proba_lr):.4f}")

# Persist the fitted model, scaler, and column order so new data can be preprocessed identically
os.makedirs("models", exist_ok=True)
joblib.dump(log_reg, "models/logistic_regression_no_duration.joblib")
joblib.dump(scaler, "models/scaler_no_duration.joblib")
joblib.dump(list(X_no_duration.columns), "models/feature_columns_no_duration.joblib")
print("Saved model, scaler, and feature columns -> models/")

# Feature importance from the Random Forest (realistic version) - which features actually matter?
rf_final = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
rf_final.fit(X_train_s, y_train)
importances = pd.Series(rf_final.feature_importances_, index=X_no_duration.columns).sort_values(ascending=False).head(12)

plt.figure(figsize=(8, 6))
sns.barplot(x=importances.values, y=importances.index, hue=importances.index, legend=False, palette="viridis")
plt.title("Top 12 features - realistic model (no duration)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("results/figures/06_feature_importance_realistic.png", dpi=130)
plt.close()
print("\nSaved feature importance chart -> results/figures/06_feature_importance_realistic.png")