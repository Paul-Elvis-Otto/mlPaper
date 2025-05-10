#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt

# ─── Configuration ─────────────────────────────────────────────────────────────
INPUT_FILE = "./temp/df_imputed.csv"  # CSV with at least 'year' and 'v2x_corr' columns
START_DATE = "1970"  # Keep records on or after this year (YYYY)
OUTPUT_FILE = "./writeup/plots/avg_v2x_corr.png"  # Where to save the plot
# ─── End configuration ─────────────────────────────────────────────────────────


def main():
    # 1. Load data
    df = pd.read_csv(INPUT_FILE)
    if "v2x_corr" not in df.columns or "year" not in df.columns:
        raise KeyError(
            f"Input file {INPUT_FILE!r} must contain both 'year' and 'v2x_corr' columns"
        )

    # 2. Parse year, filter by START_DATE
    #    Treat 'year' as a datetime at Jan 1 of that year
    df["year"] = pd.to_datetime(df["year"], format="%Y")
    cutoff = pd.to_datetime(START_DATE, format="%Y")
    df = df[df["year"] >= cutoff]
    if df.empty:
        raise ValueError(f"No data for year ≥ {START_DATE!r}")

    # 3. Compute annual averages
    #    Group by calendar year and take the mean of v2x_corr
    df["yr"] = df["year"].dt.year
    annual_avg = df.groupby("yr")["v2x_corr"].mean().reset_index()

    # 4. Plot
    plt.figure(figsize=(10, 5))
    plt.plot(annual_avg["yr"], annual_avg["v2x_corr"], marker="o")
    plt.xlabel("Year")
    plt.ylabel("Average v2x_corr")
    plt.title(f"Average v2x_corr Since {START_DATE}")
    plt.xticks(annual_avg["yr"], rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=300)
    print(f"Plot saved to {OUTPUT_FILE!r}")


if __name__ == "__main__":
    main()
