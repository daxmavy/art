#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(BradleyTerry2)
})

options(show.signif.stars = FALSE)

# ---- Args ----
args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 2) {
  stop("Usage: Rscript model_fit_batch.R <folder_name> <N>", call. = FALSE)
}

folder_name <- args[1]
N <- as.integer(args[2])
if (is.na(N) || N < 1) stop("N must be a positive integer.", call. = FALSE)

# Paths that mirror your Python pattern:
# input:  data/<folder_name>/sims/{i}.csv
# output: data/<folder_name>/analysis/results_{i}.csv
input_dir  <- file.path("data", folder_name, "sims")
output_dir <- file.path("data", folder_name, "analysis")

if (!dir.exists(input_dir)) {
  stop(sprintf("Input directory does not exist: %s", input_dir), call. = FALSE)
}
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
}

# ---- Helper to fit one file ----
fit_one <- function(input_file, output_file) {
  # Read
  data <- read.csv(input_file)

  # Get all unique player values
  all_levels <- unique(c(as.character(data$real_artwork), as.character(data$ai_artwork)))

  # Create player-specific data frames with the is_ai covariate
  data$real_artwork <- data.frame(
    player = factor(data$real_artwork, levels = all_levels),
    is_ai  = as.numeric(data$real_artwork %in% unique(data$ai_artwork))
  )

  data$ai_artwork <- data.frame(
    player = factor(data$ai_artwork, levels = all_levels),
    is_ai  = as.numeric(data$ai_artwork %in% unique(data$ai_artwork))
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

  # Extract CI, coef, p-value for is_ai
  ci <- suppressMessages(confint(model, parm = "is_ai"))
  coef_estimate <- coef(model)["is_ai"]

  model_summary <- summary(model)
  p_value <- model_summary$coefficients["is_ai", "Pr(>|z|)"]

  # (lower, estimate, upper, p_value)
  result <- c(ci[1], coef_estimate, ci[2], p_value)

  # Save a single line "lower, estimate, upper, p_value"
  writeLines(paste(result, collapse = ", "), con = output_file)

  invisible(result)
}

# ---- Batch loop ----
for (i in 0:(N - 1)) {
  input_file  <- file.path(input_dir,  sprintf("%d.csv", i))
  output_file <- file.path(output_dir, sprintf("results_%d.csv", i))

  if (!file.exists(input_file)) {
    warning(sprintf("Missing input file: %s (skipping)", input_file), call. = FALSE)
    next
  }

  # Run each fit safely so one failure doesn't abort the whole batch
  tryCatch(
    {
      fit_one(input_file, output_file)
      # message(sprintf("✓ Wrote %s", output_file))
    },
    error = function(e) {
      warning(sprintf("Failed on %s: %s", input_file, conditionMessage(e)), call. = FALSE)
    }
  )
}

warnings()
