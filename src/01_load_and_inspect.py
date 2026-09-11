import pandas as pd

# UCI's file is semicolon-separated, not comma-separated
df = pd.read_csv("data/raw/bank_marketing_raw.csv", sep=";")

print("Shape:", df.shape)
print("\nColumn dtypes:\n", df.dtypes)
print("\nFirst 5 rows:\n", df.head())
print("\nSummary statistics:\n", df.describe())
print("\nTarget class balance:\n", df["y"].value_counts())
print("\nMissing values per column:\n", df.isnull().sum())
print("\n'unknown' counts per categorical column:")
categorical_cols = df.select_dtypes(include="object").columns
for col in categorical_cols:
    unknown_count = (df[col] == "unknown").sum()
    if unknown_count > 0:
        print(f"  {col}: {unknown_count}")

print("\n'pdays' = 999 (never contacted before) count:", (df["pdays"] == 999).sum())