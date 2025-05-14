import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_parquet("./data/subset.parquet")

# --- 1. Pick just the INDEX_ columns ----------------------------------------
index_cols = [c for c in df.columns if c.startswith("INDEX_")]
if not index_cols:
    raise ValueError("No columns start with 'INDEX_'")

# --- 2. Correlation matrix ---------------------------------------------------
corr = df[index_cols].corr()

# --- 3. Heatmap --------------------------------------------------------------
plt.figure(figsize=(len(index_cols)*0.6 + 3, len(index_cols)*0.6 + 3))
sns.heatmap(
    corr,
    annot=False,              # show the numbers
    fmt=".2f",               # two decimals
    cmap="coolwarm",         # diverging palette
    center=0,                # white at zero correlation
    linewidths=0.5,
    cbar_kws={"label": "Correlation"}
)
plt.title("Correlation Heatmap – columns starting with 'INDEX_'")
plt.tight_layout()
plt.savefig("heatmap.png", dpi=200)

