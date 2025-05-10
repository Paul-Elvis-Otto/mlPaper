# install / load packages in one go
library(dplyr)
library(vdemdata)
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

# now build the final tibble in one go
main_df <- vdemdata::vdem %>%
  filter(country_text_id %in% eu_iso3, year >= year_cut) %>%
  select(country_text_id, year, any_of(list_of_vars))

# quick sanity check
glimpse(main_df)
