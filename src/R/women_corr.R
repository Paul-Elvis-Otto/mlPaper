library(dplyr)
library(tidyverse)
library(vdemdata)

df <- vdemdata::vdem

#df <- df %>%
#  tibble::rownames_to_column("obs") %>% # if rownames are e.g. "USA_2020"
#  separate(obs, into = c("country", "year"), sep = "_") %>%
#  mutate(year = as.integer(year))

# 2. Select vars & drop NAs -----------------------------------------------
#
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
  "v2lgfemleg"
  # note: you listed some twice (e.g. v2pepwrgen, v2clslavef, etc.) — include each only once
)

# create analysis dataframe
df_mod <- df %>%
  select(v2x_corr, all_of(preds)) %>%
  na.omit()

# 3. Fit the OLS ---------------------------------------------------------
model <- lm(v2x_corr ~ ., data = df_mod)

# 4. Summarize & diagnose ------------------------------------------------
summary(model) # coef estimates, R², p-values
par(mfrow = c(2, 2))
plot(model) # residual vs fitted, QQ, Cook’s D, etc.


vdemdata::find_var("v2peapsgen")
vdemdata::var_info("v2peapsgen")
