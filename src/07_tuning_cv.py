import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("data/processed/bank_marketing_clean.csv")
df = df.drop(columns=["pdays", "duration"])
df["target"] = df["y"].map({"no": 0, "yes": 1})
df = df.drop(columns=["y"])

X = df.drop(columns=["target"])
y = df["target"]
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
])

rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(random_state=42, class_weight="balanced")),
])

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_split": [2, 5],
}

print("Running GridSearchCV (12 combinations x 5 folds = 60 fits)... this will take a minute or two.")
grid_search = GridSearchCV(rf_pipeline, param_grid, cv=5, scoring="f1", n_jobs=-1)
grid_search.fit(X_train, y_train)

print("\nBest parameters:", grid_search.best_params_)
print("Best cross-validated F1:", grid_search.best_score_)

best_model = grid_search.best_estimator_
test_preds = best_model.predict(X_test)
from sklearn.metrics import f1_score
print("F1 on held-out test set:", f1_score(y_test, test_preds))

from sklearn.model_selection import StratifiedKFold

# Cross-validation on the ORIGINAL (untuned) Random Forest, for comparison
print("\nCross-validating the original untuned Random Forest (5-fold, SHUFFLED)...")
baseline_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")),
])
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(baseline_pipeline, X, y, cv=cv_strategy, scoring="f1")
print("Fold F1 scores:", cv_scores.round(4))
print("Mean F1:", cv_scores.mean().round(4), "| Std:", cv_scores.std().round(4))