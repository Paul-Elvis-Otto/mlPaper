# install / load packages in one go
library(dplyr)
library(vdemdata)
library(nanoparquet)
# define what we need
eu_iso3 <- c(
  "AUT",
  "BEL",
  "BGR",
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
  "SWE"
)
vartypes <- c("A*", "A", "B", "C")
year_cut <- 1970

# pull the list of tags for our chosen vartypes
list_of_vars <- vdemdata::codebook %>%
  filter(vartype %in% vartypes) %>%
  pull(tag)
list_of_vars

list_of_indices <- vdemdata::codebook %>%
  filter(vartype == "D") %>%
  pull(tag)
list_of_indices

main_df <- vdemdata::vdem %>%
  filter(country_text_id %in% eu_iso3, year >= year_cut) %>%
  select(
    country_text_id,
    year,
    any_of(list_of_vars),
    any_of(list_of_indices)
  ) %>%
  # prefix all index columns with "INDEX_"
  rename_with(~ paste0("INDEX_", .), any_of(list_of_indices))

glimpse(main_df)

num_main_df <- main_df %>%
  select(country_text_id, where(is.numeric))
glimpse(num_main_df)

print("writing subset to file")
nanoparquet::write_parquet(main_df, "data/subset.parquet")
print("success")
print("writing numeric subset to file")
nanoparquet::write_parquet(num_main_df, "data/subset_numeric.parquet")
