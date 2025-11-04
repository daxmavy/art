#!/usr/bin/env Rscript

library(BradleyTerry2)
options(show.signif.stars = FALSE)

# Get command line arguments
args <- commandArgs(trailingOnly = TRUE)

# Check if correct number of arguments provided
if (length(args) != 2) {
  stop("Usage: Rscript fit_bt_model.R <input_csv> <output_txt>", call. = FALSE)
}

input_file <- args[1]
output_file <- args[2]

# Read the data
data <- read.csv(input_file)

# Get all unique player values
all_levels <- unique(c(as.character(data$real_artwork), as.character(data$ai_artwork)))

# Create player-specific data frames with the is_ai covariate
data$real_artwork <- data.frame(
  player = factor(data$real_artwork, levels = all_levels),
  is_ai = as.numeric(data$real_artwork %in% unique(data$ai_artwork))
)

data$ai_artwork <- data.frame(
  player = factor(data$ai_artwork, levels = all_levels),
  is_ai = as.numeric(data$ai_artwork %in% unique(data$ai_artwork))
)

# Fit the model
model <- BTm(
  outcome = cbind(real_win, ai_win),
  player1 = real_artwork,
  player2 = ai_artwork,
  formula = ~ is_ai,
  id = "player",
  data = data
)

# Get confidence intervals and coefficient for is_ai
ci <- confint(model, parm = "is_ai")
coef_estimate <- coef(model)["is_ai"]

model_summary <- summary(model)
p_value <- model_summary$coefficients["is_ai", "Pr(>|z|)"]

# Create result: (lower, estimate, upper, p_value)
result <- c(ci[1], coef_estimate, ci[2], p_value)

# Save to file
writeLines(paste0(paste(result, collapse = ", ")), output_file)