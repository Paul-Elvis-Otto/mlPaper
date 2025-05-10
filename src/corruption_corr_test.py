import pandas as pd
import numpy as np  # For NaN checking and potential type checks
import os

# --- Configuration ---
# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path relative to the script directory
VDEM_PARQUET_PATH = os.path.join(
    script_dir, "../data/vdem_cleaned_final.parquet"
)  # Input file
OUTPUT_CORR_CSV_PATH = os.path.join(
    script_dir, "../data/vdem_correlations_with_corruption_pandas.csv"
)  # Optional output path

TARGET_COLUMN = "v2x_corr"  # V-Dem's Political Corruption Index - CHANGE IF NEEDED

# --- Check if input file exists ---
if not os.path.exists(VDEM_PARQUET_PATH):
    print(f"Error: Input Parquet file not found at '{VDEM_PARQUET_PATH}'")
    print(
        "Please ensure the preprocessing script ran successfully and the path is correct."
    )
    exit()

# --- Load the Preprocessed Dataset using Pandas ---
try:
    print(f"Loading preprocessed V-Dem data from: {VDEM_PARQUET_PATH}")
    df = pd.read_parquet(VDEM_PARQUET_PATH)
    print(f"Loaded data shape: {df.shape}")

    # --- Validate Target Column ---
    if TARGET_COLUMN not in df.columns:
        print(f"Error: Target column '{TARGET_COLUMN}' not found in the dataset.")
        print(f"Available columns are: {list(df.columns)}")
        exit()

    # Ensure target column is numeric (attempt conversion if needed)
    if not pd.api.types.is_numeric_dtype(df[TARGET_COLUMN]):
        print(
            f"Warning: Target column '{TARGET_COLUMN}' is not a numeric type (it's {df[TARGET_COLUMN].dtype}). Attempting conversion."
        )
        df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
        # Check if conversion resulted in all NaNs
        if df[TARGET_COLUMN].isnull().all():
            print(
                f"Error: Target column '{TARGET_COLUMN}' could not be converted to numeric or became all nulls."
            )
            exit()
        print(
            f"Target column '{TARGET_COLUMN}' converted to {df[TARGET_COLUMN].dtype}."
        )

    # Drop rows where the target variable is missing, as they can't be used for correlation
    initial_rows = len(df)
    df.dropna(subset=[TARGET_COLUMN], inplace=True)
    if len(df) < initial_rows:
        print(
            f"Dropped {initial_rows - len(df)} rows due to missing values in target column '{TARGET_COLUMN}'."
        )

    if df.empty:
        print("Error: No rows remaining after dropping missing target values.")
        exit()

    # --- Identify Numeric Columns (Excluding the Target) ---
    numeric_predictors = []
    potentially_convertible = []

    for col_name in df.columns:
        if col_name == TARGET_COLUMN:
            continue  # Skip the target column itself

        if pd.api.types.is_numeric_dtype(df[col_name]):
            numeric_predictors.append(col_name)
        # Check if object/string columns might contain mostly numbers
        elif df[col_name].dtype == "object":
            # Attempt conversion on a sample or check if most non-null values look numeric
            # A simple check: try converting and see if null count increases drastically
            original_nulls = df[col_name].isnull().sum()
            converted_nulls = (
                pd.to_numeric(df[col_name], errors="coerce").isnull().sum()
            )
            # If conversion doesn't add too many *new* nulls relative to non-nulls, consider it
            if (converted_nulls - original_nulls) < (
                len(df) - original_nulls
            ) * 0.5:  # Heuristic: <50% conversion failure
                print(
                    f"Info: Column '{col_name}' (object) seems potentially convertible to numeric."
                )
                potentially_convertible.append(col_name)

    if not numeric_predictors and not potentially_convertible:
        print("Error: No numeric or potentially convertible predictor columns found.")
        exit()

    print(f"Found {len(numeric_predictors)} definite numeric predictors.")
    if potentially_convertible:
        print(
            f"Found {len(potentially_convertible)} potentially convertible (object/string) predictors."
        )

    # --- Calculate Correlations ---
    print(f"\nCalculating correlations with target variable '{TARGET_COLUMN}'...")
    correlations = []
    target_series = df[
        TARGET_COLUMN
    ]  # Already checked/converted to numeric and NaNs dropped

    # Combine definite numeric and potentially convertible for the loop
    all_predictors_to_check = numeric_predictors + potentially_convertible

    for predictor in all_predictors_to_check:
        try:
            predictor_series = df[predictor]

            # Attempt conversion to numeric if it's not already, coercing errors
            if not pd.api.types.is_numeric_dtype(predictor_series):
                predictor_series_numeric = pd.to_numeric(
                    predictor_series, errors="coerce"
                )
                # Skip if conversion failed entirely
                if predictor_series_numeric.isnull().all():
                    # print(f"  - Skipping '{predictor}': Failed to convert to numeric (all nulls).")
                    continue
            else:
                predictor_series_numeric = predictor_series

            # Check for sufficient variance (Pandas .corr handles this, but explicit check is clearer)
            valid_data = predictor_series_numeric.dropna()
            if len(valid_data) < 2 or valid_data.var() == 0:
                # print(f"  - Skipping '{predictor}': Insufficient data or zero variance after handling NaNs.")
                continue

            # Calculate Pearson correlation using pandas Series method
            # Handles NaN values using pairwise deletion by default
            corr_value = target_series.corr(predictor_series_numeric, method="pearson")

            # Check if correlation is NaN (can happen with insufficient overlapping data)
            if pd.notna(corr_value):
                correlations.append({"variable": predictor, "correlation": corr_value})
            # else:
            #    print(f"  - Skipping '{predictor}' due to NaN correlation result (likely insufficient pairwise data).")

        except Exception as e:
            print(
                f"Warning: Could not calculate correlation for '{predictor}'. Error: {e}"
            )

    # --- Format and Display Results ---
    if not correlations:
        print("No valid correlations could be calculated.")
    else:
        corr_df = pd.DataFrame(correlations)
        corr_df["abs_correlation"] = corr_df["correlation"].abs()
        corr_df_sorted = corr_df.sort_values(
            by="abs_correlation", ascending=False
        ).reset_index(drop=True)

        print("\n--- Top 20 Correlated Variables with", TARGET_COLUMN, "---")
        print(corr_df_sorted.head(20))

        print("\n--- Bottom 20 Correlated Variables with", TARGET_COLUMN, "---")
        # Ensure there are at least 20 rows before trying to show the tail
        print(corr_df_sorted.tail(min(20, len(corr_df_sorted))))

        # Save the full correlation results to a CSV file
        try:
            output_dir = os.path.dirname(OUTPUT_CORR_CSV_PATH)
            if not os.path.exists(output_dir):
                print(f"Creating output directory: {output_dir}")
                os.makedirs(output_dir)
            corr_df_sorted.to_csv(OUTPUT_CORR_CSV_PATH, index=False)
            print(f"\nFull correlation results saved to {OUTPUT_CORR_CSV_PATH}")
        except Exception as e:
            print(f"\nError saving correlation results to CSV: {e}")

except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")
