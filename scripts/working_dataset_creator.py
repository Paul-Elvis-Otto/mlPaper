import polars as pl
import polars.exceptions
import os
import re  # Import regular expressions for error parsing

# --- Configuration ---
# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path relative to the script directory
VDEM_CSV_PATH = os.path.join(script_dir, "../vdemData/V-Dem-CY-Full+Others-v15.csv")
OUTPUT_PARQUET_PATH = os.path.join(
    script_dir, "../data/vdem_subset_1970_nomean_strfix.parquet"
)  # New output name
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
schema_overrides = {}  # Start with no overrides
loaded_successfully = False
load_attempts = 0

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
            infer_schema_length=10000,  # Keep a reasonably high inference length
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
        break  # Exit the loop on success

    except pl.ComputeError as e:
        error_message = str(e)
        print(
            f"Encountered compute error: {error_message[:500]}..."
        )  # Print start of error

        # Try to extract the problematic column name using regex
        match = re.search(r"column '([^']*)'", error_message)

        if match:
            column_name = match.group(1)
            if (
                column_name not in schema_overrides
            ):  # Avoid infinite loops if override didn't fix it
                print(
                    f"Identified problematic column: '{column_name}'. Forcing to Utf8 (String)."
                )
                schema_overrides[column_name] = pl.Utf8  # Force to String type
            else:
                print(
                    f"Error persisted for column '{column_name}' even after setting to Utf8. Stopping."
                )
                raise e  # Re-raise the error if forcing to string didn't help
        else:
            print(
                "Could not automatically identify problematic column from error message. Stopping."
            )
            raise e  # Re-raise the original error

    except FileNotFoundError:
        print(
            f"Error: V-Dem CSV file not found at '{VDEM_CSV_PATH}'. Please check the path."
        )
        break  # Exit loop, file not found
    except pl.exceptions.NoDataError:
        print(
            f"Error: No data could be read from '{VDEM_CSV_PATH}'. Is the file empty or corrupted?"
        )
        break  # Exit loop, no data
    except Exception as e:
        print(f"An unexpected error occurred during loading/filtering: {e}")
        raise e  # Re-raise other unexpected errors

# --- Post-Loading Processing (only if loaded successfully) ---
if loaded_successfully:
    if vdem_subset_df.height == 0:
        print(
            "Warning: No data found for the specified countries and year range after filtering."
        )
        print("Please check the VDEM_CSV_PATH, country codes, and START_YEAR.")
    else:
        print(
            f"\nFiltered dataset shape before dropping columns: {vdem_subset_df.shape}"
        )

        # --- Identify and Drop Columns ending with '_mean' ---
        cols_to_drop = [col for col in vdem_subset_df.columns if col.endswith("_mean")]

        if cols_to_drop:
            print(f"Dropping {len(cols_to_drop)} columns ending with '_mean'...")
            vdem_final_df = vdem_subset_df.drop(cols_to_drop)
            print(f"Dataset shape after dropping columns: {vdem_final_df.shape}")
        else:
            print("No columns ending with '_mean' found to drop.")
            vdem_final_df = vdem_subset_df

        # --- Save Output to Parquet ---
        output_dir = os.path.dirname(OUTPUT_PARQUET_PATH)
        if not os.path.exists(output_dir):
            print(f"Creating output directory: {output_dir}")
            os.makedirs(output_dir)

        print(f"Saving final subset dataset to: {OUTPUT_PARQUET_PATH}")
        vdem_final_df.write_parquet(OUTPUT_PARQUET_PATH)
        print("Script finished successfully!")

elif load_attempts >= MAX_LOAD_RETRIES:
    print(
        f"\nFailed to load the CSV after {MAX_LOAD_RETRIES} attempts due to persistent parsing errors."
    )
else:
    print("\nScript stopped due to non-parsing error during loading.")
