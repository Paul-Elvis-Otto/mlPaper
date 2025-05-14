library(dplyr)
library(vdemdata)
library(nanoparquet)

# define what we need
vartypes <- c("A*", "A", "B", "C")
year_cut <- 1970

# core vars and indices
list_of_vars <- vdemdata::codebook %>%
  filter(vartype %in% vartypes) %>%
  pull(tag)

list_of_indices <- vdemdata::codebook %>%
  filter(vartype == "D") %>%
  pull(tag)

# the ones we want to drop:
drop_vars <- c(
  "v2lgcrrpt",
  "v2jucorrdc",
  "v2exbribe",
  "v2exembez",
  "v2excrptps",
  "v2exthftps"
)

# build the main df
main_df <- vdemdata::vdem %>%
  filter(year >= year_cut) %>%
  select(
    country_text_id,
    year,
    any_of(list_of_vars),
    any_of(list_of_indices)
  ) %>%
  # drop those unwanted columns
  select(-all_of(drop_vars)) %>%
  # now prefix the indices that remain
  rename_with(~ paste0("INDEX_", .), any_of(list_of_indices))

glimpse(main_df)

# numeric‐only subset (country_text_id + all numeric cols)
num_main_df <- main_df %>%
  select(country_text_id, where(is.numeric))

glimpse(num_main_df)

# write to disk
message("writing subset to file")
nanoparquet::write_parquet(main_df, "data/full.parquet")
message("writing numeric subset to file")
nanoparquet::write_parquet(num_main_df, "data/full_numeric.parquet")
message("done!")
