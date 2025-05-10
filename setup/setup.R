library(dplyr)
library(pak)
library(vdemdata)

# Load vdem dataset into a df
df <- vdemdata::vdem


# Set the countries we want to keep in the dataset
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

df_countries <- df %>%
  filter(
    country_text_id %in% eu_iso3
  )

# print for debug
#df_countries

# subset the df for data only after 1970

df_countries_years <- df_countries %>%
  filter(
    year >= 1970
  )

into <- vdemdata::var_info(
  "v2pscomprg_sd"
)
into

var_to_use <- c("A*", "A", "B", "C")

# Get all variables of type "C" from the V-Dem codebook
c_vars <- vdemdata::codebook %>%
  filter(vartype %in% var_to_use) %>%
  arrange(tag)

# Inspect the first few
head(c_vars)

# get all the core variable names, for further search
all_core_vars <- c_vars %>%
  select(name, tag, vartype)
all_core_vars

list_of_vars <- all_core_vars %>%
  select(tag)
list_of_vars
typeof(list_of_vars)

main_df <- df_countries_years %>%
  select(all_of(list_of_vars))
main_df
