import pandas as pd
import numpy as np

df = pd.read_csv("data/raw/bank_marketing_raw.csv", sep=";")
print(f"STEP 0 | Loaded raw: {df.shape[0]} rows, {df.shape[1]} columns")

# STEP 1 — duplicates
n_before = len(df)
df = df.drop_duplicates()
print(f"STEP 1 | Duplicates removed: {n_before - len(df)} ({n_before} -> {len(df)})")

# STEP 2 — "unknown" categories: keep as their own valid category
# (default has 20.9% unknown - too much to drop; treated as a real answer, not missing)
categorical_cols = df.select_dtypes(include="object").columns
print("STEP 2 | 'unknown' kept as its own category in all categorical columns "
      "(not imputed, not dropped - see notes for why).")

# STEP 3 — pdays: create a flag before the 999 sentinel misleads any model
df["was_contacted_before"] = (df["pdays"] != 999).astype(int)
print(f"STEP 3 | Created 'was_contacted_before' flag. "
      f"Previously contacted: {df['was_contacted_before'].sum()} "
      f"({df['was_contacted_before'].mean()*100:.1f}%)")

# STEP 4 — outlier check on key numeric columns (report only, cap conservatively)
for col in ["duration", "campaign", "age"]:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = ((df[col] < lower) | (df[col] > upper)).sum()
    print(f"STEP 4 | '{col}': {n_out} outliers outside [{lower:.1f}, {upper:.1f}]")

# STEP 5 — final check
print(f"\nSTEP 5 | Final shape: {df.shape}")
print(f"   Duplicates left: {df.duplicated().sum()}")

df.to_csv("data/processed/bank_marketing_clean.csv", index=False)
print("Saved -> data/processed/bank_marketing_clean.csv")