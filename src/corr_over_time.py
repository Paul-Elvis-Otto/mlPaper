# src/corr_over_time.py
# Generate Plot for avg corruption over time
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_parquet("./data/subset.parquet")
cols = ['year', 'country_text_id'] + [
    col for col in df.columns
    if col.startswith('INDEX_')
]
df = df[cols]
df['year'] = pd.to_datetime(df['year'], format='%Y').dt.year

# 2. Aggregate
agg = (
    df
    .groupby('year')['INDEX_v2x_corr']
    .mean()
    .reset_index()
)
aggexe = (
    df
    .groupby('year')['INDEX_v2x_execorr']
    .mean()
    .reset_index()
)
aggpub = (
    df
    .groupby('year')['INDEX_v2x_pubcorr']
    .mean()
    .reset_index()
)

# 3. Plot both on the same figure
plt.figure(figsize=(10, 6))

plt.plot(
    agg['year'], 
    agg['INDEX_v2x_corr'], 
    marker='x', 
    label='Average INDEX_v2x_corr'
)
plt.plot(
    aggexe['year'], 
    aggexe['INDEX_v2x_execorr'], 
    marker='o', 
    label='Average INDEX_v2x_execorr'
)

plt.plot(
    aggpub['year'], 
    aggpub['INDEX_v2x_pubcorr'], 
    marker='o', 
    label='Average INDEX_v2x_pubcorr'
)
# 4. Axes limits
plt.xlim(left=1970)
plt.ylim(bottom=0)

# 5. Labels, title, legend, grid
plt.xlabel('Year')
plt.ylabel('Average Index Value')
plt.title('Average v2x_corr & v2x_execorr & v2x_pubcorr Over Time (All Countries)')
plt.legend(title='Metric')
plt.grid(True)
plt.tight_layout()
plt.savefig("./writeup/plots/corruption_over_time.png", dpi=300)
