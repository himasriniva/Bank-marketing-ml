import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="Set2")
FIG = "results/figures"

df = pd.read_csv("data/processed/bank_marketing_clean.csv")
print("Loaded:", df.shape)

# 1. Class balance
plt.figure(figsize=(5, 4))
ax = sns.countplot(data=df, x="y", hue="y", legend=False)
ax.set_title("Subscribed to term deposit? (target balance)")
for c in ax.containers:
    ax.bar_label(c)
plt.tight_layout()
plt.savefig(f"{FIG}/01_class_balance.png", dpi=130)
plt.close()

# 2. The leakage warning, visualized: duration vs target
plt.figure(figsize=(6, 4.5))
sns.boxplot(data=df, x="y", y="duration", hue="y", legend=False)
plt.title("Call duration by outcome (see why this leaks the answer)")
plt.tight_layout()
plt.savefig(f"{FIG}/02_duration_vs_target.png", dpi=130)
plt.close()

# 3. Subscription rate by job
job_rate = df.groupby("job")["y"].apply(lambda x: (x == "yes").mean()).sort_values(ascending=False)
plt.figure(figsize=(8, 5))
job_rate.plot(kind="barh", color="#5fd8c8")
plt.xlabel("Subscription rate")
plt.title("Subscription rate by job type")
plt.tight_layout()
plt.savefig(f"{FIG}/03_subscription_by_job.png", dpi=130)
plt.close()

# 4. Age distribution by outcome
plt.figure(figsize=(7, 4.5))
sns.histplot(data=df, x="age", hue="y", kde=True, alpha=0.55)
plt.title("Age distribution by outcome")
plt.tight_layout()
plt.savefig(f"{FIG}/04_age_distribution.png", dpi=130)
plt.close()

# 5. Correlation heatmap (numeric columns only)
plt.figure(figsize=(9, 7))
numeric_df = df.select_dtypes(include="number")
sns.heatmap(numeric_df.corr(), cmap="coolwarm", center=0, square=True, linewidths=0.2)
plt.title("Correlation heatmap (numeric features)")
plt.tight_layout()
plt.savefig(f"{FIG}/05_correlation_heatmap.png", dpi=130)
plt.close()

print("Saved 5 figures to results/figures/")