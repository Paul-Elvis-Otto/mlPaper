# This generates the plot between political corruption
# and liberal democracy index
library(ggplot2)
library(dplyr)
library(vdemdata)
library(tidyverse)
library(broom)
library(knitr)


data <- vdemdata::vdem

head(data)

df <- data %>%
  select(country_name, year, v2x_libdem, v2x_corr)

# 2. Fit the linear model
model <- lm(v2x_corr ~ v2x_libdem, data = df)

# 3. Print model summary to console
cat("\n--- MODEL SUMMARY ---\n")
print(summary(model))

# 4. Plot: corruption vs democracy with regression line + R² subtitle
p <- ggplot(df, aes(x = v2x_libdem, y = v2x_corr)) +
  geom_point(alpha = 0.4) +
  geom_smooth(method = "lm", se = TRUE) +
  labs(
    title = "How Well Liberal Democracy Predicts Corruption",
    subtitle = paste0(
      "n = ",
      nrow(df),
      "  •  R² = ",
      round(summary(model)$r.squared, 3)
    ),
    x = "Liberal Democracy Index (v2x_libdem)",
    y = "Corruption Index (v2x_corr)"
  ) +
  theme_minimal()

ggsave("./writeup/plots/libdem_vs_corr.png")
print(p)

# 5. Extract and write short performance table as Markdown
perf <- glance(model) %>%
  select(r.squared, adj.r.squared, sigma, statistic, p.value) %>%
  rename(
    `R-squared` = r.squared,
    `Adj. R-squared` = adj.r.squared,
    `Residual Std. Error` = sigma,
    `F-statistic` = statistic,
    `Model p-value` = p.value
  )

md_table <- kable(perf, format = "markdown", digits = 3)
perf_path <- "./writeup/tables/model_performance.md"
dir.create(dirname(perf_path), showWarnings = FALSE, recursive = TRUE)
writeLines(c("# Model Performance Summary", "", md_table), perf_path)
message("→ Wrote: ", perf_path)

## 6. Capture and write full summary() as Markdown
#full_sum <- capture.output(summary(model))
#full_path <- "./writeup/full_model_summary.md"
#dir.create(dirname(full_path), showWarnings = FALSE, recursive = TRUE)
#
#md_full <- c(
#  "# Full Regression Model Summary",
#  "",
#  "```r",
#  full_sum,
#  "```"
#)
#writeLines(md_full, full_path)
#message("→ Wrote: ", full_path)
