#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(BradleyTerry2)
})

options(show.signif.stars = FALSE)

# ---- Args ----
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) {
  stop("Usage: Rscript model_fit_batch.R <input_dir> <output_dir> <N>", call. = FALSE)
}

input_dir  <- args[1]
output_dir <- args[2]
N <- as.integer(args[3])

fit_one <- function(data) {
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

  # ---- Fixed-effects model ----
  model_fixed <- BTm(
    outcome = cbind(real_win, ai_win),
    player1 = real_artwork,
    player2 = ai_artwork,
    formula = ~ is_ai,
    id = "player",
    data = data
  )

#   # ---- Random-effects model ----
#   model_random <- BTm(
#     outcome = cbind(real_win, ai_win),
#     player1 = real_artwork,
#     player2 = ai_artwork,
#     formula = ~ is_ai + player,
#     id = "player",
#     data = data,
#     br=TRUE,
#     subset = real_win > 0 & ai_win > 0
#   )

  # ---- Extract results for is_ai ----
  extract_results <- function(model) {
    ci <- suppressMessages(confint(model, parm = "is_ai"))
    est <- coef(model)["is_ai"]
    pval <- summary(model)$coefficients["is_ai", "Pr(>|z|)"]
    data.frame(lower = ci[1], estimate = est, upper = ci[2], p_value = pval)
  }

  list(
    fixed = extract_results(model_fixed)
    # random = extract_results(model_random)
  )
}

fixed_all  <- data.frame()
# random_all <- data.frame()

for (i in 0:(N - 1)) {
  input_file <- file.path(input_dir, sprintf("%d.csv", i))

  if (!file.exists(input_file)) {
    warning(sprintf("Missing input file: %s (skipping)", input_file), call. = FALSE)
    next
  }

  tryCatch({
    data <- read.csv(input_file)
    res <- fit_one(data)

    fixed_all  <- rbind(fixed_all,  cbind(run = i, res$fixed,  status = "ok"))
    # random_all <- rbind(random_all, cbind(run = i, res$random, status = "ok"))

  }, error = function(e) {
    warning(sprintf("Failed on %s: %s", input_file, conditionMessage(e)), call. = FALSE)

    fixed_all  <- rbind(fixed_all,  data.frame(run = i, lower = NA, estimate = NA, upper = NA, p_value = NA, status = "failed"))
    # random_all <- rbind(random_all, data.frame(run = i, lower = NA, estimate = NA, upper = NA, p_value = NA, status = "failed"))
  })
}

write.csv(fixed_all,  file = file.path(output_dir, "results_fixed.csv"),  row.names = FALSE)
# write.csv(random_all, file = file.path(output_dir, "results_random.csv"), row.names = FALSE)

warnings()
