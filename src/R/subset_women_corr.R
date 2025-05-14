library(dplyr)
library(vdemdata)

# 1. define what we need ---------------------------------------------------
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
year_cut <- 1950

# all A*, A, B, C vars
vartypes <- c("A*", "A", "B", "C")
list_of_vars <- codebook %>% filter(vartype %in% vartypes) %>% pull(tag)

# all D indices except the ones we want to drop
drop_vars <- c(
  "v2lgcrrpt",
  "v2jucorrdc",
  "v2exbribe",
  "v2exembez",
  "v2excrptps",
  "v2exthftps"
)
list_of_indices <- codebook %>%
  filter(vartype == "D", !tag %in% drop_vars) %>%
  pull(tag)

# your social predictors (no duplicates)
preds <- c(
  "v2clacjstw",
  "v2clprptyw",
  "v2clslavef",
  "v2cldmovew",
  "v2csgender",
  "v2pepwrgen",
  "v2pepwrsoc",
  "v2clacjstw_osp",
  "v2cldiscw",
  "v2pepwrses",
  "v2clacjust",
  "v2peapsecon",
  "v2peasjsoecon",
  "v2peasbecon",
  "v2clgencl",
  "v2peapsgen",
  "v2peasjgen",
  "v2peasbgen",
  "v2mefemjrn",
  "v2lgfemleg",
  "e_gdp"
)

# 2. subset & select -------------------------------------------------------
df_mod <- vdemdata::vdem %>%
  # only EU, only years ≥ 1970
  filter(country_text_id %in% eu_iso3, year >= year_cut) %>%
  # pick id‐vars + outcome + social preds
  select(country_text_id, year, v2x_corr, all_of(preds)) %>%
  # drop any rows with NA in any column
  na.omit()

# quick check
glimpse(df_mod)

# 3. fit OLS ---------------------------------------------------------------
model <- lm(
  v2x_corr ~
    v2clacjstw +
      v2clprptyw +
      v2clslavef +
      v2cldmovew +
      v2csgender +
      v2pepwrgen +
      v2pepwrsoc +
      v2clacjstw_osp +
      v2cldiscw +
      v2pepwrses +
      v2clacjust +
      v2peapsecon +
      v2peasjsoecon +
      v2peasbecon +
      v2clgencl +
      v2peapsgen +
      v2peasjgen +
      v2peasbgen +
      v2mefemjrn +
      v2lgfemleg +
      egdp,
  data = df_mod
)

summary(model)
par(mfrow = c(2, 2))
plot(model)
