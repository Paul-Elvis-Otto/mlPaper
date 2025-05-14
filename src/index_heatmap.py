import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

df = pd.read_parquet("./data/full.parquet")

# 1. Identify just your INDEX_ columns
index_cols = [col for col in df.columns if col.startswith('INDEX_')]

# 2. Subset your df to only those columns
df_indices = df[index_cols]
# 3. Compute the pairwise correlation
corr = df_indices.corr()

# 4. Plot the heatmap
plt.figure(figsize=(22, 18))
sns.heatmap(
    corr,
    fmt=".2f",
    cmap='coolwarm',
)
plt.title('Correlation Among INDEX_ Variables')
plt.tight_layout()
plt.savefig('./writeup/plots/indices_heatmap.png', dpi=300)
print("created './writeup/plots/indices_heatmap.png' ")
