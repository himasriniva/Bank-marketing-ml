"""
06_full_benchmark.py
Proper ColumnTransformer + Pipeline preprocessing, benchmarked across
6 classification algorithms, on the REALISTIC (no duration) feature set.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, ConfusionMatrixDisplay,
                              RocCurveDisplay)

FIG = "results/figures"
sns.set_theme(style="whitegrid")

df = pd.read_csv("data/processed/bank_marketing_clean.csv")
df = df.drop(columns=["pdays", "duration"])  # pdays redundant, duration leaks

df["target"] = df["y"].map({"no": 0, "yes": 1})
df = df.drop(columns=["y"])

X = df.drop(columns=["target"])
y = df["target"]

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns
print("Numeric features:", list(numeric_features))
print("Categorical features:", list(categorical_features))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
])

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced"),
    "SVM": SVC(probability=True, random_state=42, class_weight="balanced"),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}

def evaluate_model(pipeline, X_test, y_test):
    preds = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)[:, 1]
    return {
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds),
        "Recall": recall_score(y_test, preds),
        "F1": f1_score(y_test, preds),
        "ROC-AUC": roc_auc_score(y_test, proba),
    }

fitted_pipelines = {}
results = []
for name, model in models.items():
    print(f"\nTraining {name}..." + (" (this one is slow - be patient)" if name == "SVM" else ""))
    pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    fitted_pipelines[name] = pipe
    metrics = evaluate_model(pipe, X_test, y_test)
    metrics["Model"] = name
    results.append(metrics)
    print(f"  {name}: F1={metrics['F1']:.4f}  ROC-AUC={metrics['ROC-AUC']:.4f}")

results_df = pd.DataFrame(results).sort_values("F1", ascending=False)
results_df.to_csv("results/model_comparison_full.csv", index=False)
print("\n=== Full comparison, ranked by F1 ===")
print(results_df.round(4).to_string(index=False))

best_name = results_df.iloc[0]["Model"]
best_pipeline = fitted_pipelines[best_name]
print(f"\nBest model: {best_name}")

# Confusion matrix for the best model
plt.figure(figsize=(5, 4.5))
ConfusionMatrixDisplay.from_estimator(best_pipeline, X_test, y_test, cmap="Blues")
plt.title(f"Confusion Matrix - {best_name}")
plt.tight_layout()
plt.savefig(f"{FIG}/07_confusion_matrix_best.png", dpi=130)
plt.close()

# ROC curves, all 6 models overlaid
plt.figure(figsize=(7, 6))
ax = plt.gca()
for name, pipe in fitted_pipelines.items():
    RocCurveDisplay.from_estimator(pipe, X_test, y_test, name=name, ax=ax)
plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
plt.title("ROC Curves - all 6 models")
plt.tight_layout()
plt.savefig(f"{FIG}/08_roc_curves_all6.png", dpi=130)
plt.close()

# Feature importance from Random Forest
rf_pipe = fitted_pipelines["Random Forest"]
feature_names = rf_pipe.named_steps["preprocessor"].get_feature_names_out()
importances = pd.Series(
    rf_pipe.named_steps["model"].feature_importances_, index=feature_names
).sort_values(ascending=False).head(12)

plt.figure(figsize=(8, 6))
sns.barplot(x=importances.values, y=importances.index, hue=importances.index, legend=False, palette="viridis")
plt.title("Top 12 features - Random Forest (ColumnTransformer pipeline)")
plt.tight_layout()
plt.savefig(f"{FIG}/09_feature_importance_pipeline.png", dpi=130)
plt.close()

print("\nSaved: confusion matrix, ROC curves, feature importance -> results/figures/")