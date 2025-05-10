# Plot correlation heatmap for indices
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_parquet("data/subset.parquet")

cols = ['year', 'country_text_id'] + [
    col for col in df.columns
    if col.startswith('INDEX_')
]

df = df[cols]

df['year'] = pd.to_datetime(df['year'], format='%Y').dt.year

# 1. Select numeric columns and fill missing values with the column means
numeric_df = df.select_dtypes(include=['number']).fillna(df.select_dtypes(include=['number']).mean())

# 2. Compute the correlation matrix
corr = numeric_df.corr()

# 3. Plot the heatmap
plt.figure(figsize=(16, 14))
sns.heatmap(
    corr,
    fmt=".2f",         # two decimal places
    cmap='coolwarm',   # diverging map from blue to red
    linewidths=0.1,    # lines between cells
    square=True        # square cells
)
plt.title('Correlation Matrix of Numeric Variables')
plt.tight_layout()
plt.savefig('./writeup/plots/correlation_heatmap_indices.png', dpi=300)
