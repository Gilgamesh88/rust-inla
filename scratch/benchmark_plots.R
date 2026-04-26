local_rustyinla_lib <- Sys.getenv("RUSTYINLA_LIB", "")
if (nzchar(local_rustyinla_lib)) {
    .libPaths(c(
        normalizePath(local_rustyinla_lib, winslash = "/", mustWork = TRUE),
        .libPaths()
    ))
}

suppressPackageStartupMessages({
    library(CASdatasets)
    library(INLA)
    library(rustyINLA)
})

output_dir <- file.path(
    "C:/Users/Antonio/Documents/rustyINLA/rustyINLA",
    "scratch",
    "plots"
)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

performance_results <- data.frame(
    case_id = c(
        "poisson_iid_offset",
        "poisson_multi_iid_offset",
        "gamma_rw1",
        "poisson_ar1_offset",
        "zip_iid_offset",
        "tweedie_rw1_offset"
    ),
    label = c(
        "Poisson + iid(VehBrand)",
        "Poisson + iid(VehBrand) + iid(Region)",
        "Gamma + rw1(AgeGroup)",
        "Poisson + ar1(AgeIndex)",
        "ZIP + iid(VehBrand)",
        "Tweedie + rw1(AgeGroup)"
    ),
    rusty_time = c(22.67, 39.30, 1.17, 31.53, 55.45, 70.59),
    inla_time = c(22.39, 37.99, 2.32, 35.08, 96.42, 107.30),
    rusty_mem = c(431.1, 852.4, 831.8, 998.0, 1160.9, 1349.9),
    inla_mem = c(1013.8, 1497.0, 982.5, 1502.0, 1795.0, 2092.8),
    passed = c(FALSE, FALSE, TRUE, FALSE, FALSE, FALSE),
    stringsAsFactors = FALSE
)

plot_performance <- function(results, file_path) {
    old_par <- par(no.readonly = TRUE)
    on.exit(par(old_par), add = TRUE)

    png(file_path, width = 2200, height = 1200, res = 170)
    on.exit(dev.off(), add = TRUE)

    par(mfrow = c(1, 2), mar = c(12, 5, 4, 1) + 0.1, oma = c(0, 0, 2, 0))

    bar_cols <- c("#1B5E20", "#8E1B1B")
    case_labels <- results$label
    time_mat <- rbind(results$rusty_time, results$inla_time)
    mem_mat <- rbind(results$rusty_mem, results$inla_mem)

    time_bp <- barplot(
        time_mat,
        beside = TRUE,
        col = bar_cols,
        las = 2,
        names.arg = case_labels,
        ylab = "Seconds",
        main = "Runtime Comparison",
        cex.names = 0.85,
        ylim = c(0, max(time_mat) * 1.20)
    )
    legend(
        "topleft",
        legend = c("rustyINLA", "R-INLA"),
        fill = bar_cols,
        bty = "n",
        cex = 0.95
    )
    text(
        x = colMeans(time_bp),
        y = apply(time_mat, 2, max) + max(time_mat) * 0.05,
        labels = ifelse(results$passed, "PASS", "FAIL"),
        cex = 0.9,
        font = 2,
        col = ifelse(results$passed, "#1B5E20", "#8E1B1B")
    )

    mem_bp <- barplot(
        mem_mat,
        beside = TRUE,
        col = bar_cols,
        las = 2,
        names.arg = case_labels,
        ylab = "Peak Memory (MB)",
        main = "Memory Comparison",
        cex.names = 0.85,
        ylim = c(0, max(mem_mat) * 1.20)
    )
    legend(
        "topleft",
        legend = c("rustyINLA", "R-INLA"),
        fill = bar_cols,
        bty = "n",
        cex = 0.95
    )
    text(
        x = colMeans(mem_bp),
        y = apply(mem_mat, 2, max) + max(mem_mat) * 0.05,
        labels = sprintf("%.1fx", results$inla_mem / results$rusty_mem),
        cex = 0.85,
        font = 2,
        col = "#444444"
    )

    mtext("freMTPL2 Benchmark Performance", outer = TRUE, cex = 1.4, font = 2)
}

build_frequency_sample <- function() {
    data(freMTPL2freq)
    freq_data <- freMTPL2freq
    freq_data$Exposure <- pmin(freq_data$Exposure, 1)
    freq_data$ClaimNb <- pmin(freq_data$ClaimNb, 4)

    set.seed(42)
    sample_idx <- sample(seq_len(nrow(freq_data)), 10000)
    freq_sample <- freq_data[sample_idx, ]

    freq_sample$VehPower <- as.factor(freq_sample$VehPower)
    freq_sample$VehBrand <- as.factor(freq_sample$VehBrand)
    freq_sample$Region <- as.factor(freq_sample$Region)
    freq_sample$Area <- as.factor(freq_sample$Area)
    freq_sample$log_exposure <- log(freq_sample$Exposure)

    freq_sample
}

collect_fixed_coefficients <- function() {
    freq_sample <- build_frequency_sample()
    formula_freq <- ClaimNb ~ 1 + VehPower + VehAge + DrivAge + BonusMalus +
        f(Area, model = "iid") + offset(log_exposure)

    rusty_fit <- rusty_inla(
        formula_freq,
        data = freq_sample,
        family = "poisson"
    )
    inla_fit <- suppressWarnings(suppressMessages(
        inla(
            formula_freq,
            family = "poisson",
            data = freq_sample,
            control.compute = list(config = FALSE),
            control.predictor = list(compute = TRUE),
            num.threads = 1
        )
    ))

    rusty_fixed <- rusty_fit$summary.fixed
    inla_fixed <- inla_fit$summary.fixed
    shared <- intersect(rownames(rusty_fixed), rownames(inla_fixed))

    coeff_table <- data.frame(
        term = shared,
        rusty_mean = rusty_fixed[shared, "mean"],
        rusty_sd = rusty_fixed[shared, "sd"],
        inla_mean = inla_fixed[shared, "mean"],
        inla_sd = inla_fixed[shared, "sd"],
        stringsAsFactors = FALSE
    )
    coeff_table$abs_diff <- abs(coeff_table$rusty_mean - coeff_table$inla_mean)
    coeff_table
}

plot_coefficients <- function(coeff_table, file_path) {
    old_par <- par(no.readonly = TRUE)
    on.exit(par(old_par), add = TRUE)

    png(file_path, width = 2200, height = 1400, res = 170)
    on.exit(dev.off(), add = TRUE)

    par(mfrow = c(1, 2), mar = c(7, 12, 4, 2) + 0.1, oma = c(0, 0, 2, 0))

    coeff_table <- coeff_table[order(coeff_table$inla_mean), ]
    y_pos <- seq_len(nrow(coeff_table))
    x_min <- min(
        coeff_table$rusty_mean - 1.96 * coeff_table$rusty_sd,
        coeff_table$inla_mean - 1.96 * coeff_table$inla_sd
    )
    x_max <- max(
        coeff_table$rusty_mean + 1.96 * coeff_table$rusty_sd,
        coeff_table$inla_mean + 1.96 * coeff_table$inla_sd
    )

    plot(
        NA,
        xlim = c(x_min, x_max),
        ylim = c(0.5, nrow(coeff_table) + 0.5),
        yaxt = "n",
        ylab = "",
        xlab = "Coefficient Mean",
        main = "Fixed Coefficient Comparison"
    )
    axis(2, at = y_pos, labels = coeff_table$term, las = 2, cex.axis = 0.9)
    abline(v = 0, lty = 3, col = "#999999")

    rusty_y <- y_pos + 0.16
    inla_y <- y_pos - 0.16
    segments(
        coeff_table$rusty_mean - 1.96 * coeff_table$rusty_sd,
        rusty_y,
        coeff_table$rusty_mean + 1.96 * coeff_table$rusty_sd,
        rusty_y,
        col = "#1B5E20",
        lwd = 3
    )
    segments(
        coeff_table$inla_mean - 1.96 * coeff_table$inla_sd,
        inla_y,
        coeff_table$inla_mean + 1.96 * coeff_table$inla_sd,
        inla_y,
        col = "#8E1B1B",
        lwd = 3
    )
    points(coeff_table$rusty_mean, rusty_y, pch = 16, col = "#1B5E20", cex = 1.1)
    points(coeff_table$inla_mean, inla_y, pch = 17, col = "#8E1B1B", cex = 1.1)
    legend(
        "bottomright",
        legend = c("rustyINLA", "R-INLA"),
        col = c("#1B5E20", "#8E1B1B"),
        pch = c(16, 17),
        lwd = 3,
        bty = "n"
    )

    barplot(
        coeff_table$abs_diff,
        horiz = TRUE,
        names.arg = coeff_table$term,
        las = 1,
        col = "#355C7D",
        border = NA,
        xlab = "Absolute Difference",
        main = "Absolute Difference by Coefficient"
    )

    mtext(
        "freMTPL2 Poisson Fixed Effects: rustyINLA vs R-INLA",
        outer = TRUE,
        cex = 1.4,
        font = 2
    )
}

performance_plot_path <- file.path(output_dir, "benchmark_performance.png")
coefficients_plot_path <- file.path(output_dir, "benchmark_fixed_coefficients.png")
coefficients_csv_path <- file.path(output_dir, "benchmark_fixed_coefficients.csv")

plot_performance(performance_results, performance_plot_path)
coeff_table <- collect_fixed_coefficients()
write.csv(coeff_table, coefficients_csv_path, row.names = FALSE)
plot_coefficients(coeff_table, coefficients_plot_path)

cat("performance_plot=", performance_plot_path, "\n", sep = "")
cat("coefficients_plot=", coefficients_plot_path, "\n", sep = "")
cat("coefficients_csv=", coefficients_csv_path, "\n", sep = "")
