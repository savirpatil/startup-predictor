import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("data/raw/big_startup_secsees_dataset.csv")
df = df[df["status"].isin(["closed", "acquired", "ipo"])].copy()
df["funding_total_usd"] = df["funding_total_usd"].replace("-", np.nan)

missing_rates = df.groupby("status").apply(lambda g: pd.Series({
    "Founding Date": g["founded_at"].isnull().mean() * 100,
    "Country": g["country_code"].isnull().mean() * 100,
    "Funding Amount": g["funding_total_usd"].isnull().mean() * 100,
}))

missing_rates = missing_rates.reindex(["acquired", "ipo", "closed"])

fig, ax = plt.subplots(figsize=(8, 5))
missing_rates.plot(kind="bar", ax=ax, color=["#aed7ff", "#66aaff", "#2b6fff"])

ax.set_ylabel("Missing Data Rate (%)")
ax.set_xlabel("Outcome")
ax.set_title("Missing Data Rate by Startup Outcome")
ax.set_xticklabels(missing_rates.index, rotation=0)
ax.legend(title="Field")

plt.tight_layout()
plt.savefig("viz3_missing_data_bias.png", dpi=200)
plt.show()