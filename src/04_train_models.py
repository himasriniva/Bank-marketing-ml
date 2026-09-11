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