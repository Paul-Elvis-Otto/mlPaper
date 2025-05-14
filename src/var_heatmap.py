import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

df = pd.read_parquet("./data/full.parquet")

# 1. Identify just your INDEX_ columns
index_cols = [col for col in df.columns if col.startswith('INDEX_')]

# 2. Subset your df to only those columns
df_var = df.drop(columns=index_cols)
df_var_numeric = df_var.select_dtypes(include=[np.number])


corr = df_var_numeric.corr()

# 4. Plot the heatmap
plt.figure(figsize=(22, 18))
sns.heatmap(
    corr,
    fmt=".2f",
    cmap='coolwarm',
)
plt.title('Correlation Among INDEX_ Variables')
plt.tight_layout()
plt.savefig('./writeup/plots/var_heatmap.png', dpi=300)
print("created './writeup/plots/var_heatmap.png' ")
