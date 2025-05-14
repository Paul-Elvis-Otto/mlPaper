import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

df = pd.read_parquet("./data/full.parquet")

# 1. Identify all INDEX_ columns except the target
index_cols = [
    col for col in df.columns 
    if col.startswith('INDEX_') and col != 'INDEX_v2x_corr'
]

# 2. Drop those columns (this keeps INDEX_v2x_corr)
df_var = df.drop(columns=index_cols)

# 3. Restrict to numeric columns
df_var_numeric = df_var.select_dtypes(include=[np.number])

# 4. Compute full corr matrix, then target correlations
corr = df_var_numeric.corr()
target = 'INDEX_v2x_corr'
target_corr = corr[target].drop(labels=target)

# 5. Grab top 20 positive and top 20 negative
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
plt.savefig('./writeup/plots/var_corr_with_target_top80.png', dpi=300)
print("created './writeup/plots/var_corr_with_target_top80.png'")

