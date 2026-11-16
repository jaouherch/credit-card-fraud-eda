# Credit Card Fraud: reproducible exploratory data analysis
# Author: Jaouher Chouchane | Tunis Business School
# Usage: Rscript analysis/analysis.R data/creditcard.csv outputs
# This script performs data analysis, not supervised fraud prediction.

args <- commandArgs(trailingOnly = TRUE)
input_path <- if (length(args) >= 1) args[[1]] else "data/creditcard.csv"
output_dir <- if (length(args) >= 2) args[[2]] else "outputs"

for (package in c("data.table", "ggplot2")) {
  if (!requireNamespace(package, quietly = TRUE)) {
    stop(sprintf("Install the missing R package: %s", package))
  }
}

if (!file.exists(input_path)) {
  stop(paste0(
    "Dataset not found: ", input_path, "\n",
    "Download 'creditcard.csv' from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud ",
    "and place it under data/ (the dataset is not stored in this repository)."
  ))
}

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
figure_dir <- file.path(output_dir, "figures")
dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)

data <- as.data.frame(data.table::fread(input_path, showProgress = FALSE))
feature_cols <- paste0("V", 1:28)
required_cols <- c("Time", "Amount", "Class", feature_cols)
missing_cols <- setdiff(required_cols, names(data))
if (length(missing_cols) > 0) {
  stop(paste("Missing required columns:", paste(missing_cols, collapse = ", ")))
}

if (anyNA(data[, required_cols])) {
  stop("Missing values detected in columns required by the analysis.")
}
if (any(!is.finite(as.matrix(data[, c("Time", "Amount", feature_cols)])))) {
  stop("Non-finite numeric values detected in the dataset.")
}
if (!all(data$Class %in% c(0, 1))) {
  stop("Expected Class to contain only 0 (legitimate) and 1 (fraud).")
}
if (any(data$Amount < 0)) {
  stop("Transaction Amount must not be negative for the chosen plots.")
}

data$Class <- factor(data$Class, levels = c(0, 1), labels = c("Legitimate", "Fraud"))
data$elapsed_hour_bucket <- floor(data$Time / 3600) %% 24
data$AmountScaled <- as.numeric(scale(data$Amount))

# Time is relative to the first transaction, NOT a verified time of day.
# The hour bucket above is elapsed-time modulo 24 hours, not a clock-hour feature.

class_levels <- levels(data$Class)
summary_by_class <- do.call(rbind, lapply(class_levels, function(group) {
  amounts <- data$Amount[data$Class == group]
  data.frame(
    Class = group,
    Count = length(amounts),
    Share = length(amounts) / nrow(data),
    Mean_Amount = mean(amounts),
    Median_Amount = median(amounts),
    SD_Amount = sd(amounts),
    Q1_Amount = as.numeric(quantile(amounts, 0.25)),
    Q3_Amount = as.numeric(quantile(amounts, 0.75))
  )
}))

if (any(summary_by_class$Count < 2)) {
  stop("At least two transactions per class are needed for statistical comparison.")
}

write.csv(summary_by_class, file.path(output_dir, "class_summary.csv"), row.names = FALSE)

# Welch's t-test: difference in mean Amount, allowing unequal group variances.
# Transaction Amount is heavily skewed; this test alone is not a fraud detector.
amount_test <- t.test(Amount ~ Class, data = data, var.equal = FALSE)
test_results <- data.frame(
  Method = "Welch two-sample t-test",
  Comparison = "Legitimate minus Fraud",
  T_Statistic = unname(amount_test$statistic),
  Degrees_of_Freedom = unname(amount_test$parameter),
  P_Value = amount_test$p.value,
  CI_Lower = unname(amount_test$conf.int[[1]]),
  CI_Upper = unname(amount_test$conf.int[[2]])
)
write.csv(test_results, file.path(output_dir, "amount_test.csv"), row.names = FALSE)

# A reproducible stratified 70/30 split for methodological demonstration.
# No predictive model is fitted and this split is not used to estimate accuracy.
set.seed(42)
train_ids <- unlist(lapply(class_levels, function(group) {
  ids <- which(data$Class == group)
  sample(ids, size = floor(0.70 * length(ids)))
}), use.names = FALSE)
train_mask <- rep(FALSE, nrow(data))
train_mask[train_ids] <- TRUE
split_summary <- do.call(rbind, lapply(class_levels, function(group) {
  ids <- which(data$Class == group)
  data.frame(
    Class = group,
    Training = sum(train_mask[ids]),
    Testing = sum(!train_mask[ids])
  )
}))
write.csv(split_summary, file.path(output_dir, "stratified_split.csv"), row.names = FALSE)

# Plot every fraudulent transaction but sample the majority class to reduce overplotting.
legitimate_ids <- which(data$Class == "Legitimate")
fraud_ids <- which(data$Class == "Fraud")
plot_ids <- sort(c(sample(legitimate_ids, min(length(legitimate_ids), 10000L)), fraud_ids))
plot_data <- data[plot_ids, , drop = FALSE]

theme_fraud <- ggplot2::theme_minimal(base_size = 12) +
  ggplot2::theme(legend.position = "bottom", plot.title.position = "plot")
class_colors <- c(Legitimate = "#334155", Fraud = "#e05a67")

class_plot <- ggplot2::ggplot(
  summary_by_class, ggplot2::aes(x = Class, y = Count, fill = Class)
) +
  ggplot2::geom_col(width = 0.6, show.legend = FALSE) +
  ggplot2::scale_fill_manual(values = class_colors) +
  ggplot2::scale_y_log10() +
  ggplot2::labs(
    title = "Severe Class Imbalance",
    subtitle = "Logarithmic count scale; calculated from the full dataset",
    x = NULL, y = "Transaction count (log10)"
  ) + theme_fraud
ggplot2::ggsave(file.path(figure_dir, "01_class_balance.png"), class_plot,
                width = 8, height = 5, dpi = 160)

v_plot <- ggplot2::ggplot(
  plot_data, ggplot2::aes(x = V1, y = V2, color = Class)
) +
  ggplot2::geom_point(size = 0.8, alpha = 0.55) +
  ggplot2::scale_color_manual(values = class_colors) +
  ggplot2::labs(
    title = "Provided PCA Features: V1 vs V2",
    subtitle = "All fraud cases and up to 10,000 legitimate cases for legibility"
  ) + theme_fraud
ggplot2::ggsave(file.path(figure_dir, "02_v1_v2.png"), v_plot,
                width = 8, height = 5, dpi = 160)

# The dataset's V1-V28 are already PCA-transformed and anonymized.
# A second PCA is included solely as an exploratory projection; it is NOT
# a recovery of confidential (unavailable) raw transaction variables.
secondary_pca <- prcomp(data[, feature_cols, drop = FALSE],
                       center = TRUE, scale. = TRUE)
pca_variance <- secondary_pca$sdev^2 / sum(secondary_pca$sdev^2)
write.csv(
  data.frame(Component = paste0("PC", seq_along(pca_variance)),
             Variance_Explained = pca_variance),
  file.path(output_dir, "secondary_pca_variance.csv"), row.names = FALSE
)
pca_plot_data <- data.frame(
  PC1 = secondary_pca$x[plot_ids, 1],
  PC2 = secondary_pca$x[plot_ids, 2],
  Class = data$Class[plot_ids]
)
pca_plot <- ggplot2::ggplot(
  pca_plot_data, ggplot2::aes(x = PC1, y = PC2, color = Class)
) +
  ggplot2::geom_point(size = 0.8, alpha = 0.55) +
  ggplot2::scale_color_manual(values = class_colors) +
  ggplot2::labs(
    title = "Secondary PCA Projection",
    subtitle = "Demonstration only: applied to already PCA-transformed features"
  ) + theme_fraud
ggplot2::ggsave(file.path(figure_dir, "03_secondary_pca.png"), pca_plot,
                width = 8, height = 5, dpi = 160)

amount_plot <- ggplot2::ggplot(
  plot_data, ggplot2::aes(x = Class, y = log1p(Amount), fill = Class)
) +
  ggplot2::geom_boxplot(outlier.shape = NA, show.legend = FALSE) +
  ggplot2::scale_fill_manual(values = class_colors) +
  ggplot2::labs(
    title = "Transaction Amount Distribution",
    subtitle = "Log-transformed amounts; exploratory comparison only",
    x = NULL, y = "log(1 + Amount)"
  ) + theme_fraud
ggplot2::ggsave(file.path(figure_dir, "04_amount_distribution.png"), amount_plot,
                width = 8, height = 5, dpi = 160)

cat("Analysis complete. Results written to:", normalizePath(output_dir), "\n")
print(summary_by_class)
print(test_results)
cat("Secondary PCA variance explained by PC1 and PC2:",
    round(sum(pca_variance[1:2]) * 100, 2), "%\n")
