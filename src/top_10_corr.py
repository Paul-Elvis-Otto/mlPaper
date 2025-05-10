#get the top 10 correlations in the df with v2x_corr
import pandas as pd

# 0. Load
df = pd.read_parquet("./data/subset_numeric.parquet")

# 1. Identify all INDEX_ columns, but plan to keep just your target
keep_index = "INDEX_v2x_corr"
all_index_cols = [c for c in df.columns if c.startswith("INDEX_")]
drop_index_cols = [c for c in all_index_cols if c != keep_index]

#. 2. Drop the unwanted INDEX_ columns
df = df.drop(columns=drop_index_cols)

# 3. Now recompute your data columns (everything except the two identifiers)
data_cols = [c for c in df.columns if c not in ("year", "country_text_id")]

# 4. Force them to numeric, coercing errors → NaN
df[data_cols] = df[data_cols].apply(pd.to_numeric, errors="coerce")

# 5. Fill those NaNs with each column’s mean
df[data_cols] = df[data_cols].fillna(df[data_cols].mean())

# 6. (Optional) clean up year
df["year"] = pd.to_datetime(df["year"], format="%Y", errors="coerce").dt.year

# 7. Compute one-vs-all correlations
corr_with_pubcorr = df[data_cols].corr()[keep_index].drop(labels=[keep_index])

# 8. Extract top-10 positive and top-10 negative correlations
top_pos = corr_with_pubcorr.nlargest(10)
top_neg = corr_with_pubcorr.nsmallest(10)

# 9. Convert to DataFrame for easy viewing/output
df_pos_top_10 = top_pos.reset_index().rename(columns={"index": "variable", keep_index: "correlation"})
df_neg_top_10 = top_neg.reset_index().rename(columns={"index": "variable", keep_index: "correlation"})

# 10. (Optional) display or save
print("Top 10 Positive Correlations:")
print(df_pos_top_10)
print("\nTop 10 Negative Correlations:")
print(df_neg_top_10)

# If you want to save them:
df_pos_top_10.to_csv("data/pos_top10_correlations.csv", index=False)
df_neg_top_10.to_csv("data/neg_top10_correlations.csv", index=False)

