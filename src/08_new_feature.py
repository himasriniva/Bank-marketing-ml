"""
Add a seasonal feature to the cleaned Bank Marketing data: whether the
client was contacted in a spring/summer month (Mar-Aug) vs autumn/winter
(Sep-Feb).

Input:  data/processed/bank_marketing_clean.csv
Output: data/processed/bank_marketing_clean.csv (overwritten, with the new
        'is_spring_summer' column added)
"""
import pandas as pd

df = pd.read_csv("data/processed/bank_marketing_clean.csv")

SPRING_SUMMER_MONTHS = {"mar", "apr", "may", "jun", "jul", "aug"}
df["is_spring_summer"] = df["month"].isin(SPRING_SUMMER_MONTHS).astype(int)

print(f"Spring/summer contacts: {df['is_spring_summer'].sum()} "
      f"({df['is_spring_summer'].mean()*100:.1f}%)")

df.to_csv("data/processed/bank_marketing_clean.csv", index=False)
print("Saved -> data/processed/bank_marketing_clean.csv")
