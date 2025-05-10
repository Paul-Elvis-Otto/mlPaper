# TODO: Is broken needs fixing
#
#–– Load libs ––#
library(readr) # read/write CSV
library(dplyr) # data-manipulation
library(vdemdata) # var_info()
library(tools) # file_path_sans_ext()
library(purrr)

#–– Specify your input CSV paths here ––#
csv_files <- c(
  "./data/neg_top10_correlations.csv",
  "./data/pos_top10_correlations.csv"
)

#for (csv_file in csv_files) {
#  # 1. Read original CSV
#  df <- read_csv(csv_file, show_col_types = FALSE)
#
#  # 2. Grab each unique tag
#  vars <- unique(df$variable)
#
#  # 3. Pull back the raw list from var_info()
#  defs_list <- vdemdata::var_info(vars)
#
#  # 4. Turn that list of lists into a two‐column tibble
#  info_df <- map_dfr(defs_list, function(x) {
#    tibble(
#      variable = x[["tag"]], # tag field
#      description = x[["definition"]] # definition text
#    )
#  })
#
#  # 5. Left‐join the lookup onto your data
#  df_out <- df %>%
#    left_join(info_df, by = "variable")
#
#  # 6. Write out with “_definition” suffix
#  out_file <- paste0(
#    file_path_sans_ext(csv_file),
#    "_definition.csv"
#  )
#  write_csv(df_out, out_file)
#  message("Wrote: ", out_file)
#}
#

