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
target = 'INDEX_v2x_corr'
target_corr = corr[target].drop(labels=target)  # drop self-correlation

top_pos = target_corr.nlargest(40)
top_neg = target_corr.nsmallest(40)

# 6. Combine them into one Series (negatives first, then positives)
selected = pd.concat([top_neg, top_pos])

# 7. Plot
plt.figure(figsize=(16, 6))
selected.sort_values().plot(kind='bar', color=sns.color_palette("coolwarm", len(selected)))
plt.ylabel(f'Correlation with {target}')
plt.title(f'Top 20 Negative & Top 20 Positive Correlations with {target}')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('./writeup/plots/index_corr_with_target_top80.png', dpi=300)
print("created './writeup/plots/index_corr_with_target_top80.png'")

