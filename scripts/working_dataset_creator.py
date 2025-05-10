import polars as pl
import polars.exceptions
import os
import re

# --- Configuration ---
# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path relative to the script directory
VDEM_CSV_PATH = os.path.join(script_dir, "../vdemData/V-Dem-CY-Full+Others-v15.csv")
OUTPUT_PARQUET_PATH = os.path.join(
    script_dir, "../data/vdem_subset_1970_original_vars.parquet"
)  # Final output file name reflecting content
START_YEAR = 1970
MAX_LOAD_RETRIES = 20  # Limit retries to prevent infinite loops

# --- Define Countries ---
eu_members_iso3 = [
    "AUT",
    "BEL",
    "BGR",
    "HRV",
    "CYP",
    "CZE",
    "DNK",
    "EST",
    "FIN",
    "FRA",
    "DEU",
    "GRC",
    "HUN",
    "IRL",
    "ITA",
    "LVA",
    "LTU",
    "LUX",
    "MLT",
    "NLD",
    "POL",
    "PRT",
    "ROU",
    "SVK",
    "SVN",
    "ESP",
    "SWE",
]
other_countries_iso3 = ["USA", "GBR"]
target_countries_iso3 = list(set(eu_members_iso3 + other_countries_iso3))

print(
    f"Target countries ({len(target_countries_iso3)}): {', '.join(target_countries_iso3)}"
)
print(f"Target start year: {START_YEAR}")

# --- Iterative Loading with Error Handling ---
schema_overrides = {}
loaded_successfully = False
load_attempts = 0
vdem_subset_df = None

while load_attempts < MAX_LOAD_RETRIES:
    load_attempts += 1
    print(f"\n--- Load Attempt {load_attempts}/{MAX_LOAD_RETRIES} ---")
    if schema_overrides:
        print(f"Applying schema overrides: {schema_overrides}")
    else:
        print("No schema overrides applied yet.")

    try:
        print(f"Scanning V-Dem data from: {VDEM_CSV_PATH}")
        vdem_lf = pl.scan_csv(
            VDEM_CSV_PATH,
            schema_overrides=schema_overrides,
            infer_schema_length=10000,
        )

        print("Applying filters (Countries and Year)...")
        vdem_filtered_lf = vdem_lf.filter(
            (pl.col("country_text_id").is_in(target_countries_iso3))
            & (pl.col("year") >= START_YEAR)
        )

        print("Collecting filtered data...")
        vdem_subset_df = vdem_filtered_lf.collect()

        print("Data loaded and filtered successfully!")
        loaded_successfully = True
        break

    except pl.ComputeError as e:
        error_message = str(e)
        print(f"Encountered compute error during load/filter: {error_message[:500]}...")
        match = re.search(r"column '([^']*)'", error_message)
        if match:
            column_name = match.group(1)
            if (
                column_name not in schema_overrides
                or schema_overrides[column_name] != pl.Utf8
            ):
                print(
                    f"Identified problematic column: '{column_name}'. Forcing to Utf8 (String) for next attempt."
                )
                schema_overrides[column_name] = pl.Utf8
            else:
                print(
                    f"Error persisted for column '{column_name}' even after forcing to Utf8. Stopping."
                )
                loaded_successfully = False
                break
        else:
            print(
                "Could not automatically identify problematic column from error message. Stopping."
            )
            loaded_successfully = False
            break
    except FileNotFoundError:
        print(
            f"Error: V-Dem CSV file not found at '{VDEM_CSV_PATH}'. Please check the path."
        )
        loaded_successfully = False
        break
    except pl.exceptions.NoDataError:
        print(
            f"Error: No data could be read from '{VDEM_CSV_PATH}'. Is the file empty or corrupted?"
        )
        loaded_successfully = False
        break
    except Exception as e:
        print(f"An unexpected error occurred during loading/filtering: {e}")
        loaded_successfully = False
        break

# --- Post-Loading Processing (only if loaded successfully) ---
if loaded_successfully and vdem_subset_df is not None:
    if vdem_subset_df.height == 0:
        print(
            "\nWarning: No data found for the specified countries and year range after filtering."
        )
        print("Please check the VDEM_CSV_PATH, country codes, and START_YEAR.")
    else:
        print(
            f"\nFiltered dataset shape before final column cleaning: {vdem_subset_df.shape}"
        )
        original_columns = vdem_subset_df.columns
        # Use a set for efficient lookup of base column names
        all_columns_set = set(original_columns)

        # --- Identify and Drop DERIVED Columns (where base name also exists) ---
        cols_to_drop_derived = []
        for col in original_columns:
            if "_" in col:
                # Split at the last underscore to get potential base name and suffix
                parts = col.rsplit("_", 1)
                base_name = parts[0]
                # Check if the part before the last underscore is ALSO a column name
                if base_name in all_columns_set:
                    cols_to_drop_derived.append(col)

        if not cols_to_drop_derived:
            print("\nNo derived columns (where base name also exists) found to drop.")
            vdem_final_df = vdem_subset_df  # Keep the current dataframe
        else:
            print(
                f"\nIdentified {len(cols_to_drop_derived)} derived columns to remove."
            )
            # print("Columns to be dropped:", cols_to_drop_derived) # Uncomment to see the full list

            # --- Drop the identified columns ---
            vdem_final_df = vdem_subset_df.drop(columns=cols_to_drop_derived)
            print(
                f"Final dataset shape after dropping derived columns: {vdem_final_df.shape}"
            )

            # Verify columns were dropped
            dropped_count = len(original_columns) - len(vdem_final_df.columns)
            if dropped_count != len(cols_to_drop_derived):
                print("Warning: Discrepancy in expected vs actual dropped columns!")
            else:
                print(f"Successfully dropped {dropped_count} derived columns.")

        # --- Save Output to Parquet ---
        output_dir = os.path.dirname(OUTPUT_PARQUET_PATH)
        if not os.path.exists(output_dir):
            print(f"Creating output directory: {output_dir}")
            os.makedirs(output_dir)

        print(
            f"\nSaving final dataset with original variables only to: {OUTPUT_PARQUET_PATH}"
        )
        vdem_final_df.write_parquet(OUTPUT_PARQUET_PATH)
        print("Script finished successfully!")

elif load_attempts >= MAX_LOAD_RETRIES:
    print(
        f"\nFailed to load the CSV after {MAX_LOAD_RETRIES} attempts due to persistent parsing errors."
    )
else:
    print("\nScript stopped due to non-parsing error during loading or filtering.")
