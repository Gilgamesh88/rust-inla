# Phase 8 rolling update walkthrough: two iid random effects.
#
# This is intentionally written as a readable package-use example rather than a
# harness. Edit the columns/periods below, run the script, and inspect the CSVs.

repo_root <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
source(file.path(repo_root, "tools", "load_worktree_package.R"), local = TRUE)
load_rustyinla_for_benchmarks(repo_root)

if (!requireNamespace("data.table", quietly = TRUE)) {
    stop("Package 'data.table' is required for this walkthrough.", call. = FALSE)
}

data_path <- Sys.getenv(
    "RUSTYINLA_REALDATA_PATH",
    file.path(repo_root, "data", "real", "inla_sbi_test.RDS")
)
if (!file.exists(data_path) && file.exists(sub("\\.RDS$", ".rds", data_path))) {
    data_path <- sub("\\.RDS$", ".rds", data_path)
}
if (!file.exists(data_path)) {
    stop(sprintf("Data file not found: %s", data_path), call. = FALSE)
}

report_dir <- Sys.getenv(
    "RUSTYINLA_WALKTHROUGH_REPORT_DIR",
    file.path(repo_root, "scratch", "phase8_reports", "walkthrough_two_iid_2021_to_2022_1t")
)
dir.create(report_dir, recursive = TRUE, showWarnings = FALSE)

period_col <- "periodo"
response_col <- "pt"
exposure_col <- "expuesto"
base_period <- "2021_y"
update_period <- "2022_1t"
fixed_effects <- c("modeloc", "medio_emit")
iid_cols <- c("desc_edo_circula", "desc_armadora")
family <- "poisson"
fixed_reference_levels <- list(
    modeloc = Sys.getenv("RUSTYINLA_WALKTHROUGH_REF_MODELOC", ""),
    medio_emit = Sys.getenv("RUSTYINLA_WALKTHROUGH_REF_MEDIO_EMIT", "")
)
plot_top_n <- as.integer(Sys.getenv("RUSTYINLA_WALKTHROUGH_PLOT_TOP_N", "30"))
if (!is.finite(plot_top_n) || plot_top_n < 1L) {
    plot_top_n <- 30L
}

formula_name <- function(x) {
    if (grepl("^[.A-Za-z][._A-Za-z0-9]*$", x)) {
        return(x)
    }
    paste0("`", gsub("`", "\\\\`", x), "`")
}

model_formula <- stats::as.formula(sprintf(
    "%s ~ %s",
    formula_name(response_col),
    paste(c(
        "1",
        vapply(fixed_effects, formula_name, character(1)),
        sprintf("offset(log(%s))", formula_name(exposure_col)),
        sprintf("f(%s, model = \"iid\")", vapply(iid_cols, formula_name, character(1)))
    ), collapse = " + ")
))

as_number <- function(x) {
    if (is.numeric(x)) {
        return(as.numeric(x))
    }
    suppressWarnings(as.numeric(as.character(x)))
}

timed_fit <- function(expr) {
    value <- NULL
    elapsed <- system.time({
        value <- eval.parent(substitute(expr))
    })[["elapsed"]]
    list(value = value, elapsed = as.numeric(elapsed))
}

write_report <- function(x, name) {
    path <- file.path(report_dir, name)
    utils::write.csv(x, path, row.names = FALSE, na = "")
    path
}

theta_comparison <- function(update_fit, joint_fit) {
    theta_names <- intersect(
        rownames(update_fit$summary.hyperpar.internal),
        rownames(joint_fit$summary.hyperpar.internal)
    )
    update_internal <- update_fit$summary.hyperpar.internal[theta_names, , drop = FALSE]
    joint_internal <- joint_fit$summary.hyperpar.internal[theta_names, , drop = FALSE]
    update_precision <- update_fit$summary.hyperpar[theta_names, , drop = FALSE]
    joint_precision <- joint_fit$summary.hyperpar[theta_names, , drop = FALSE]
    update_block_sd <- exp(-0.5 * as.numeric(update_internal$mean))
    joint_block_sd <- exp(-0.5 * as.numeric(joint_internal$mean))

    data.frame(
        theta_index = seq_along(theta_names),
        theta_name = theta_names,
        iid_block = sub("^Precision for ", "", theta_names),
        update_log_precision_mean = as.numeric(update_internal$mean),
        joint_log_precision_mean = as.numeric(joint_internal$mean),
        log_precision_mean_abs_diff = abs(as.numeric(update_internal$mean) - as.numeric(joint_internal$mean)),
        update_log_precision_mode = as.numeric(update_internal$mode),
        joint_log_precision_mode = as.numeric(joint_internal$mode),
        log_precision_mode_abs_diff = abs(as.numeric(update_internal$mode) - as.numeric(joint_internal$mode)),
        update_precision_mean = as.numeric(update_precision$mean),
        joint_precision_mean = as.numeric(joint_precision$mean),
        precision_rel_diff = abs(as.numeric(update_precision$mean) - as.numeric(joint_precision$mean)) /
            pmax(abs(as.numeric(joint_precision$mean)), 1e-12),
        update_block_sd_from_log_mean = update_block_sd,
        joint_block_sd_from_log_mean = joint_block_sd,
        block_sd_rel_diff = abs(update_block_sd - joint_block_sd) / pmax(abs(joint_block_sd), 1e-12),
        check.names = FALSE
    )
}

effect_comparison <- function(update_fit, joint_fit, current_data) {
    fixed_names <- union(rownames(update_fit$summary.fixed), rownames(joint_fit$summary.fixed))
    fixed <- data.frame(
        effect_type = "fixed",
        effect_group = "fixed",
        parameter = fixed_names,
        current_exposure = NA_real_,
        update_mean = as.numeric(update_fit$summary.fixed[fixed_names, "mean"]),
        joint_mean = as.numeric(joint_fit$summary.fixed[fixed_names, "mean"]),
        update_sd = as.numeric(update_fit$summary.fixed[fixed_names, "sd"]),
        joint_sd = as.numeric(joint_fit$summary.fixed[fixed_names, "sd"]),
        update_q025 = as.numeric(update_fit$summary.fixed[fixed_names, "0.025quant"]),
        joint_q025 = as.numeric(joint_fit$summary.fixed[fixed_names, "0.025quant"]),
        update_q975 = as.numeric(update_fit$summary.fixed[fixed_names, "0.975quant"]),
        joint_q975 = as.numeric(joint_fit$summary.fixed[fixed_names, "0.975quant"]),
        check.names = FALSE
    )

    random <- do.call(rbind, lapply(iid_cols, function(col) {
        update_random <- update_fit$summary.random[[col]]
        joint_random <- joint_fit$summary.random[[col]]
        level_names <- union(rownames(update_random), rownames(joint_random))
        exposure_by_level <- tapply(current_data[[exposure_col]], as.character(current_data[[col]]), sum)
        data.frame(
            effect_type = "random",
            effect_group = col,
            parameter = level_names,
            current_exposure = as.numeric(exposure_by_level[level_names]),
            update_mean = as.numeric(update_random[level_names, "mean"]),
            joint_mean = as.numeric(joint_random[level_names, "mean"]),
            update_sd = as.numeric(update_random[level_names, "sd"]),
            joint_sd = as.numeric(joint_random[level_names, "sd"]),
            update_q025 = as.numeric(update_random[level_names, "0.025quant"]),
            joint_q025 = as.numeric(joint_random[level_names, "0.025quant"]),
            update_q975 = as.numeric(update_random[level_names, "0.975quant"]),
            joint_q975 = as.numeric(joint_random[level_names, "0.975quant"]),
            check.names = FALSE
        )
    }))
    out <- rbind(fixed, random)
    out$current_exposure[is.na(out$current_exposure) & out$effect_type == "random"] <- 0
    out$mean_abs_diff <- abs(out$update_mean - out$joint_mean)
    out$sd_ratio <- out$update_sd / pmax(out$joint_sd, 1e-12)
    out$update_relativity <- exp(out$update_mean)
    out$joint_relativity <- exp(out$joint_mean)
    out$relativity_abs_diff <- abs(out$update_relativity - out$joint_relativity)
    out$update_abs_mean <- abs(out$update_mean)
    out$joint_abs_mean <- abs(out$joint_mean)
    out$update_sign <- ifelse(out$update_mean > 0, "positive", ifelse(out$update_mean < 0, "negative", "zero"))
    out$joint_sign <- ifelse(out$joint_mean > 0, "positive", ifelse(out$joint_mean < 0, "negative", "zero"))
    out$sign_changed <- out$update_sign != out$joint_sign
    out$update_interval_width <- out$update_q975 - out$update_q025
    out$joint_interval_width <- out$joint_q975 - out$joint_q025
    out$interval_width_ratio <- out$update_interval_width / pmax(out$joint_interval_width, 1e-12)
    out[order(-out$mean_abs_diff), , drop = FALSE]
}

random_effect_stats <- function(effects) {
    random <- effects[effects$effect_type == "random", , drop = FALSE]
    if (nrow(random) == 0L) {
        return(data.frame())
    }
    rows <- lapply(split(random, random$effect_group), function(df) {
        data.frame(
            effect_group = df$effect_group[[1L]],
            n_levels = nrow(df),
            active_levels = sum(df$current_exposure > 0, na.rm = TRUE),
            total_current_exposure = sum(df$current_exposure, na.rm = TRUE),
            mean_abs_diff = mean(df$mean_abs_diff, na.rm = TRUE),
            median_abs_diff = stats::median(df$mean_abs_diff, na.rm = TRUE),
            q95_abs_diff = as.numeric(stats::quantile(df$mean_abs_diff, 0.95, names = FALSE, na.rm = TRUE)),
            max_abs_diff = max(df$mean_abs_diff, na.rm = TRUE),
            min_sd_ratio = min(df$sd_ratio, na.rm = TRUE),
            median_sd_ratio = stats::median(df$sd_ratio, na.rm = TRUE),
            max_sd_ratio = max(df$sd_ratio, na.rm = TRUE),
            positive_update_levels = sum(df$update_mean > 0, na.rm = TRUE),
            negative_update_levels = sum(df$update_mean < 0, na.rm = TRUE),
            zero_update_levels = sum(df$update_mean == 0, na.rm = TRUE),
            sign_changed_levels = sum(df$sign_changed, na.rm = TRUE),
            max_abs_update_log_effect = max(abs(df$update_mean), na.rm = TRUE),
            max_abs_joint_log_effect = max(abs(df$joint_mean), na.rm = TRUE),
            min_update_relativity = min(df$update_relativity, na.rm = TRUE),
            max_update_relativity = max(df$update_relativity, na.rm = TRUE),
            check.names = FALSE
        )
    })
    do.call(rbind, rows)
}

random_top_abs_by_block <- function(effects, top_n) {
    random <- effects[effects$effect_type == "random", , drop = FALSE]
    if (nrow(random) == 0L) {
        return(data.frame())
    }
    rows <- lapply(split(random, random$effect_group), function(df) {
        df <- df[order(-df$update_abs_mean, df$parameter), , drop = FALSE]
        utils::head(df, top_n)
    })
    do.call(rbind, rows)
}

level_values_for_grid <- function(data, col) {
    values <- data[[col]]
    if (is.factor(values)) {
        return(levels(values))
    }
    if (is.character(values) || is.logical(values)) {
        return(sort(unique(as.character(values))))
    }
    values <- as.numeric(values)
    values <- values[is.finite(values)]
    unique_values <- sort(unique(values))
    if (length(unique_values) <= 20L) {
        return(unique_values)
    }
    as.numeric(stats::quantile(values, probs = c(0.1, 0.5, 0.9), names = FALSE, na.rm = TRUE))
}

build_level_grid <- function(model_data, fixed_effects, iid_cols) {
    grid_inputs <- lapply(c(fixed_effects, iid_cols), function(col) level_values_for_grid(model_data, col))
    names(grid_inputs) <- c(fixed_effects, iid_cols)
    grid <- do.call(expand.grid, c(grid_inputs, list(KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE)))
    for (col in c(fixed_effects, iid_cols)) {
        if (is.factor(model_data[[col]])) {
            grid[[col]] <- factor(as.character(grid[[col]]), levels = levels(model_data[[col]]))
        }
    }
    grid[[exposure_col]] <- 1.0
    grid
}

observed_period_summary <- function(model_data) {
    dt <- data.table::as.data.table(model_data)
    out <- dt[, list(
        claims = sum(get(response_col)),
        exposure = sum(get(exposure_col)),
        cells = .N
    ), by = period_col]
    out$frequency <- out$claims / out$exposure
    out <- out[order(get(period_col))]
    as.data.frame(out)
}

observed_cell_summary <- function(model_data, fixed_effects, iid_cols) {
    key_cols <- c(fixed_effects, iid_cols)
    dt <- data.table::as.data.table(model_data)
    out <- dt[, list(
        observed_claims = sum(get(response_col)),
        observed_exposure = sum(get(exposure_col)),
        observed_periods = data.table::uniqueN(get(period_col))
    ), by = key_cols]
    out$observed_frequency <- out$observed_claims / out$observed_exposure
    as.data.frame(out)
}

predict_level_grid <- function(fit, grid, fixed_effects, iid_cols) {
    fixed_formula <- stats::as.formula(sprintf(
        "~ %s",
        paste(vapply(fixed_effects, formula_name, character(1)), collapse = " + ")
    ))
    x <- stats::model.matrix(fixed_formula, data = grid)
    beta <- stats::setNames(as.numeric(fit$summary.fixed$mean), rownames(fit$summary.fixed))
    common <- intersect(names(beta), colnames(x))
    x_common <- NULL
    if (length(common) == 0L) {
        fixed_eta <- rep(0.0, nrow(grid))
        fixed_var <- rep(0.0, nrow(grid))
    } else {
        fixed_eta <- as.numeric(x[, common, drop = FALSE] %*% beta[common])
        fixed_cov <- fit$internal.gaussian$fixed_cov
        if (is.null(fixed_cov) || length(fixed_cov) == 0L) {
            fixed_cov <- fit$internal.gaussian$fixed_cov_theta_opt
        }
        if (!is.null(fixed_cov) &&
            all(common %in% rownames(fixed_cov)) &&
            all(common %in% colnames(fixed_cov))) {
            x_common <- x[, common, drop = FALSE]
            cov_common <- fixed_cov[common, common, drop = FALSE]
            fixed_var <- rowSums((x_common %*% cov_common) * x_common)
        } else {
            fixed_var <- rep(NA_real_, nrow(grid))
        }
    }

    random_eta <- rep(0.0, nrow(grid))
    random_var <- rep(0.0, nrow(grid))
    latent_keys_by_col <- list()
    for (col in iid_cols) {
        random_table <- fit$summary.random[[col]]
        random_mean <- stats::setNames(as.numeric(random_table$mean), rownames(random_table))
        random_sd <- stats::setNames(as.numeric(random_table$sd), rownames(random_table))
        values <- as.character(grid[[col]])
        latent_keys_by_col[[col]] <- paste(col, values, sep = ":")
        contribution <- random_mean[values]
        contribution[is.na(contribution)] <- 0.0
        random_eta <- random_eta + as.numeric(contribution)
        contribution_sd <- random_sd[values]
        contribution_sd[is.na(contribution_sd)] <- 0.0
        random_var <- random_var + as.numeric(contribution_sd)^2
    }

    fixed_random_cov <- rep(0.0, nrow(grid))
    latent_fixed_cov <- fit$internal.gaussian$latent_fixed_cov
    if (!is.null(x_common) &&
        !is.null(latent_fixed_cov) &&
        length(common) > 0L &&
        all(common %in% colnames(latent_fixed_cov))) {
        for (col in iid_cols) {
            latent_keys <- latent_keys_by_col[[col]]
            latent_idx <- match(latent_keys, rownames(latent_fixed_cov))
            valid <- which(!is.na(latent_idx))
            if (length(valid) == 0L) {
                next
            }
            cov_selected <- matrix(0.0, nrow = nrow(grid), ncol = length(common))
            cov_selected[valid, ] <- latent_fixed_cov[latent_idx[valid], common, drop = FALSE]
            fixed_random_cov <- fixed_random_cov + rowSums(x_common * cov_selected)
        }
    }

    random_pair_cov <- rep(0.0, nrow(grid))
    latent_pair_cov <- fit$internal.gaussian$latent_pair_cov
    if (!is.null(latent_pair_cov) && nrow(latent_pair_cov) > 0L && length(iid_cols) > 1L) {
        pair_lookup <- c(
            stats::setNames(latent_pair_cov$covariance, paste(latent_pair_cov$latent_i, latent_pair_cov$latent_j, sep = "\001")),
            stats::setNames(latent_pair_cov$covariance, paste(latent_pair_cov$latent_j, latent_pair_cov$latent_i, sep = "\001"))
        )
        for (left in seq_len(length(iid_cols) - 1L)) {
            for (right in (left + 1L):length(iid_cols)) {
                lookup_key <- paste(
                    latent_keys_by_col[[iid_cols[[left]]]],
                    latent_keys_by_col[[iid_cols[[right]]]],
                    sep = "\001"
                )
                contribution <- as.numeric(pair_lookup[lookup_key])
                contribution[is.na(contribution)] <- 0.0
                random_pair_cov <- random_pair_cov + contribution
            }
        }
    }

    eta <- fixed_eta + random_eta
    eta_var_diag <- pmax(fixed_var + random_var, 0.0)
    eta_sd_diag <- sqrt(eta_var_diag)
    eta_q025_diag <- eta - 1.96 * eta_sd_diag
    eta_q975_diag <- eta + 1.96 * eta_sd_diag
    eta_cross_adjustment <- 2.0 * fixed_random_cov + 2.0 * random_pair_cov
    eta_var <- pmax(fixed_var + random_var + eta_cross_adjustment, 0.0)
    eta_sd <- sqrt(eta_var)
    eta_q025 <- eta - 1.96 * eta_sd
    eta_q975 <- eta + 1.96 * eta_sd
    list(
        fixed_eta = fixed_eta,
        fixed_eta_sd = sqrt(pmax(fixed_var, 0.0)),
        random_eta = random_eta,
        random_eta_sd = sqrt(pmax(random_var, 0.0)),
        fixed_random_cov = fixed_random_cov,
        random_pair_cov = random_pair_cov,
        eta_cross_adjustment = eta_cross_adjustment,
        eta = eta,
        eta_sd_diag = eta_sd_diag,
        eta_q025_diag = eta_q025_diag,
        eta_q975_diag = eta_q975_diag,
        eta_sd = eta_sd,
        eta_q025 = eta_q025,
        eta_q975 = eta_q975,
        frequency_per_exposure = exp(eta),
        frequency_q025_diag_per_exposure = exp(eta_q025_diag),
        frequency_q975_diag_per_exposure = exp(eta_q975_diag),
        frequency_q025_per_exposure = exp(eta_q025),
        frequency_q975_per_exposure = exp(eta_q975)
    )
}

level_prediction_table <- function(update_fit, joint_fit, model_data, fixed_effects, iid_cols) {
    grid <- build_level_grid(model_data, fixed_effects, iid_cols)
    update_pred <- predict_level_grid(update_fit, grid, fixed_effects, iid_cols)
    joint_pred <- predict_level_grid(joint_fit, grid, fixed_effects, iid_cols)
    key_cols <- c(fixed_effects, iid_cols)
    out <- cbind(
        grid[, key_cols, drop = FALSE],
        data.frame(
            update_eta = update_pred$eta,
            joint_eta = joint_pred$eta,
            eta_abs_diff = abs(update_pred$eta - joint_pred$eta),
            update_eta_sd_diag_approx = update_pred$eta_sd_diag,
            update_eta_sd_approx = update_pred$eta_sd,
            update_eta_q025_diag_approx = update_pred$eta_q025_diag,
            update_eta_q975_diag_approx = update_pred$eta_q975_diag,
            update_eta_q025_approx = update_pred$eta_q025,
            update_eta_q975_approx = update_pred$eta_q975,
            joint_eta_sd_diag_approx = joint_pred$eta_sd_diag,
            joint_eta_sd_approx = joint_pred$eta_sd,
            joint_eta_q025_diag_approx = joint_pred$eta_q025_diag,
            joint_eta_q975_diag_approx = joint_pred$eta_q975_diag,
            joint_eta_q025_approx = joint_pred$eta_q025,
            joint_eta_q975_approx = joint_pred$eta_q975,
            update_frequency_per_exposure = update_pred$frequency_per_exposure,
            update_frequency_q025_diag_per_exposure_approx = update_pred$frequency_q025_diag_per_exposure,
            update_frequency_q975_diag_per_exposure_approx = update_pred$frequency_q975_diag_per_exposure,
            update_frequency_q025_per_exposure_approx = update_pred$frequency_q025_per_exposure,
            update_frequency_q975_per_exposure_approx = update_pred$frequency_q975_per_exposure,
            joint_frequency_per_exposure = joint_pred$frequency_per_exposure,
            joint_frequency_q025_diag_per_exposure_approx = joint_pred$frequency_q025_diag_per_exposure,
            joint_frequency_q975_diag_per_exposure_approx = joint_pred$frequency_q975_diag_per_exposure,
            joint_frequency_q025_per_exposure_approx = joint_pred$frequency_q025_per_exposure,
            joint_frequency_q975_per_exposure_approx = joint_pred$frequency_q975_per_exposure,
            frequency_abs_diff = abs(update_pred$frequency_per_exposure - joint_pred$frequency_per_exposure),
            frequency_rel_diff = abs(update_pred$frequency_per_exposure - joint_pred$frequency_per_exposure) /
                pmax(abs(joint_pred$frequency_per_exposure), 1e-12),
            update_fixed_eta = update_pred$fixed_eta,
            update_fixed_eta_sd_approx = update_pred$fixed_eta_sd,
            update_random_eta = update_pred$random_eta,
            update_random_eta_sd_approx = update_pred$random_eta_sd,
            update_fixed_random_cov_approx = update_pred$fixed_random_cov,
            update_random_pair_cov_approx = update_pred$random_pair_cov,
            update_eta_var_cross_adjustment_approx = update_pred$eta_cross_adjustment,
            joint_fixed_eta = joint_pred$fixed_eta,
            joint_fixed_eta_sd_approx = joint_pred$fixed_eta_sd,
            joint_random_eta = joint_pred$random_eta,
            joint_random_eta_sd_approx = joint_pred$random_eta_sd,
            joint_fixed_random_cov_approx = joint_pred$fixed_random_cov,
            joint_random_pair_cov_approx = joint_pred$random_pair_cov,
            joint_eta_var_cross_adjustment_approx = joint_pred$eta_cross_adjustment,
            check.names = FALSE
        )
    )
    out$.prediction_row <- seq_len(nrow(out))
    out <- merge(
        out,
        observed_cell_summary(model_data, fixed_effects, iid_cols),
        by = key_cols,
        all.x = TRUE,
        sort = FALSE
    )
    out <- out[order(out$.prediction_row), , drop = FALSE]
    out$.prediction_row <- NULL
    out$observed_cell <- is.finite(out$observed_exposure) & out$observed_exposure > 0
    out$observed_claims[is.na(out$observed_claims)] <- 0.0
    out$observed_exposure[is.na(out$observed_exposure)] <- 0.0
    out$observed_periods[is.na(out$observed_periods)] <- 0L
    out[order(-out$update_frequency_per_exposure), , drop = FALSE]
}

prediction_frequency_summary <- function(level_predictions, observed_periods) {
    q <- function(x, p) as.numeric(stats::quantile(x, p, names = FALSE, na.rm = TRUE))
    data.frame(
        prediction_rows = nrow(level_predictions),
        observed_prediction_cells = sum(level_predictions$observed_cell),
        observed_period_frequency_min = min(observed_periods$frequency, na.rm = TRUE),
        observed_period_frequency_max = max(observed_periods$frequency, na.rm = TRUE),
        update_frequency_min = min(level_predictions$update_frequency_per_exposure, na.rm = TRUE),
        update_frequency_median = stats::median(level_predictions$update_frequency_per_exposure, na.rm = TRUE),
        update_frequency_q95 = q(level_predictions$update_frequency_per_exposure, 0.95),
        update_frequency_max = max(level_predictions$update_frequency_per_exposure, na.rm = TRUE),
        joint_frequency_min = min(level_predictions$joint_frequency_per_exposure, na.rm = TRUE),
        joint_frequency_median = stats::median(level_predictions$joint_frequency_per_exposure, na.rm = TRUE),
        joint_frequency_q95 = q(level_predictions$joint_frequency_per_exposure, 0.95),
        joint_frequency_max = max(level_predictions$joint_frequency_per_exposure, na.rm = TRUE),
        max_update_to_observed_period_max = max(level_predictions$update_frequency_per_exposure, na.rm = TRUE) /
            max(observed_periods$frequency, na.rm = TRUE),
        check.names = FALSE
    )
}

fitted_summary <- function(update_fit, joint_fit, joint_rows) {
    lhs <- as.numeric(update_fit$summary.fitted.values$mean)
    rhs <- as.numeric(joint_fit$summary.fitted.values$mean[joint_rows])
    n <- min(length(lhs), length(rhs))
    abs_diff <- abs(lhs[seq_len(n)] - rhs[seq_len(n)])
    data.frame(
        total_update = sum(lhs[seq_len(n)]),
        total_joint = sum(rhs[seq_len(n)]),
        total_abs_diff = abs(sum(lhs[seq_len(n)]) - sum(rhs[seq_len(n)])),
        mean_abs_diff = mean(abs_diff),
        q95_abs_diff = as.numeric(stats::quantile(abs_diff, 0.95, names = FALSE)),
        max_abs_diff = max(abs_diff),
        max_rel_diff = max(abs_diff / pmax(1.0, abs(rhs[seq_len(n)]))),
        check.names = FALSE
    )
}

write_plot <- function(name, expr, width = 1200L, height = 800L) {
    path <- file.path(report_dir, name)
    grDevices::png(path, width = width, height = height, res = 120)
    on.exit(grDevices::dev.off(), add = TRUE)
    eval.parent(substitute(expr))
    path
}

short_label <- function(x, max_chars = 42L) {
    x <- as.character(x)
    ifelse(nchar(x) > max_chars, paste0(substr(x, 1L, max_chars - 3L), "..."), x)
}

file_label <- function(x) {
    gsub("[^A-Za-z0-9_]+", "_", x)
}

apply_reference_level <- function(values, col, reference_levels) {
    levels <- sort(unique(as.character(values)))
    ref <- reference_levels[[col]]
    if (!is.null(ref) && nzchar(ref)) {
        if (!(ref %in% levels)) {
            warning(
                sprintf("Requested reference level '%s' was not found for %s; using default.", ref, col),
                call. = FALSE
            )
        } else {
            levels <- c(ref, setdiff(levels, ref))
        }
    }
    factor(as.character(values), levels = levels)
}

plot_top_drift <- function(effects, top_n) {
    top <- utils::head(effects[order(-effects$mean_abs_diff), , drop = FALSE], top_n)
    top <- top[rev(seq_len(nrow(top))), , drop = FALSE]
    labels <- short_label(paste(top$effect_group, top$parameter, sep = ": "), 54L)
    label_cex <- if (nrow(top) > 25L) 0.72 else 0.85
    graphics::par(mar = c(5, 16, 5, 2), xpd = FALSE)
    graphics::barplot(
        top$mean_abs_diff,
        names.arg = labels,
        horiz = TRUE,
        las = 1,
        cex.names = label_cex,
        col = ifelse(top$effect_type == "fixed", "#4C78A8", "#F58518"),
        xlab = "absolute mean difference on log-rate scale",
        main = sprintf("Top %d Effect Drifts: Update vs Joint", nrow(top))
    )
    graphics::legend(
        "bottomright",
        legend = c("fixed", "random"),
        fill = c("#4C78A8", "#F58518"),
        bty = "n"
    )
}

plot_update_vs_joint <- function(effects, effect_type, title) {
    df <- effects[effects$effect_type == effect_type, , drop = FALSE]
    lim <- range(c(df$update_mean, df$joint_mean), finite = TRUE)
    graphics::par(mar = c(5, 5, 4, 2))
    graphics::plot(
        df$joint_mean,
        df$update_mean,
        pch = 19,
        col = if (identical(effect_type, "fixed")) "#4C78A8" else "#F58518",
        xlab = "joint refit mean",
        ylab = "rolling update mean",
        main = title,
        xlim = lim,
        ylim = lim
    )
    graphics::abline(0, 1, col = "gray40", lwd = 2)
}

plot_fixed_effect_estimates <- function(fixed_effects_table) {
    df <- fixed_effects_table[order(fixed_effects_table$update_mean), , drop = FALSE]
    y <- seq_len(nrow(df))
    xlim <- range(c(df$update_q025, df$update_q975, df$joint_mean, 0), finite = TRUE)
    colors <- ifelse(df$update_mean >= 0, "#D62728", "#1F77B4")
    graphics::par(mar = c(5, 14, 5, 2), xpd = FALSE)
    graphics::plot(
        df$update_mean,
        y,
        xlim = xlim,
        ylim = c(0.5, nrow(df) + 0.5),
        yaxt = "n",
        ylab = "",
        xlab = "fixed-effect estimate on log-rate scale",
        main = "Fixed Effects: Signed Estimates With 95% Intervals",
        pch = 19,
        col = colors
    )
    graphics::segments(df$update_q025, y, df$update_q975, y, col = colors, lwd = 2)
    graphics::points(df$joint_mean, y, pch = 1, col = "black", cex = 1.1)
    graphics::abline(v = 0, col = "gray40", lwd = 2)
    graphics::axis(2, at = y, labels = short_label(df$parameter, 54L), las = 1, cex.axis = 0.85)
    graphics::legend(
        "bottomright",
        legend = c("update positive", "update negative", "joint mean"),
        col = c("#D62728", "#1F77B4", "black"),
        pch = c(19, 19, 1),
        bty = "n"
    )
}

plot_random_abs_effect <- function(effects, effect_group, top_n) {
    df <- effects[effects$effect_type == "random" & effects$effect_group == effect_group, , drop = FALSE]
    df <- utils::head(df[order(-df$update_abs_mean, df$parameter), , drop = FALSE], top_n)
    df <- df[rev(seq_len(nrow(df))), , drop = FALSE]
    labels <- short_label(df$parameter, 54L)
    label_cex <- if (nrow(df) > 25L) 0.72 else 0.85
    colors <- ifelse(df$update_mean >= 0, "#D62728", "#1F77B4")
    graphics::par(mar = c(5, 14, 5, 2), xpd = FALSE)
    graphics::barplot(
        df$update_mean,
        names.arg = labels,
        horiz = TRUE,
        las = 1,
        cex.names = label_cex,
        col = colors,
        xlab = "rolling update log-rate effect",
        main = sprintf("Top %d Absolute %s Effects", nrow(df), effect_group)
    )
    graphics::abline(v = 0, col = "gray40", lwd = 2)
    graphics::legend(
        "bottomright",
        legend = c("positive surcharge", "negative discount"),
        fill = c("#D62728", "#1F77B4"),
        bty = "n"
    )
}

plot_random_sd_ratio <- function(effects, top_n) {
    df <- effects[effects$effect_type == "random", , drop = FALSE]
    df$sd_ratio_drift <- abs(df$sd_ratio - 1.0)
    top <- utils::head(df[order(-df$sd_ratio_drift), , drop = FALSE], top_n)
    top <- top[rev(seq_len(nrow(top))), , drop = FALSE]
    labels <- short_label(paste(top$effect_group, top$parameter, sep = ": "), 54L)
    label_cex <- if (nrow(top) > 25L) 0.72 else 0.85
    graphics::par(mar = c(5, 16, 5, 2), xpd = FALSE)
    graphics::barplot(
        top$sd_ratio,
        names.arg = labels,
        horiz = TRUE,
        las = 1,
        cex.names = label_cex,
        col = "#54A24B",
        xlab = "update SD / joint SD",
        main = sprintf("Top %d Random-Effect SD Ratio Deviations", nrow(top))
    )
    graphics::abline(v = 1, col = "gray40", lwd = 2)
}

plot_level_prediction_top <- function(predictions, fixed_effects, iid_cols, top_n) {
    top <- utils::head(predictions[order(-predictions$update_frequency_per_exposure), , drop = FALSE], top_n)
    top <- top[rev(seq_len(nrow(top))), , drop = FALSE]
    labels <- short_label(apply(top[, c(fixed_effects, iid_cols), drop = FALSE], 1L, paste, collapse = " | "), 70L)
    label_cex <- if (nrow(top) > 25L) 0.64 else 0.78
    xlim <- range(c(
        0,
        top$update_frequency_q025_per_exposure_approx,
        top$update_frequency_q975_per_exposure_approx
    ), finite = TRUE)
    graphics::par(mar = c(5, 18, 5, 2), xpd = FALSE)
    mids <- graphics::barplot(
        top$update_frequency_per_exposure,
        names.arg = labels,
        horiz = TRUE,
        las = 1,
        cex.names = label_cex,
        col = "#D62728",
        xlim = xlim,
        xlab = "predicted frequency per unit exposure",
        main = sprintf("Top %d Predicted Level Combinations", nrow(top))
    )
    graphics::segments(
        top$update_frequency_q025_per_exposure_approx,
        mids,
        top$update_frequency_q975_per_exposure_approx,
        mids,
        col = "black",
        lwd = 1.5
    )
}

raw <- data.table::as.data.table(readRDS(data_path))
needed <- c(period_col, response_col, exposure_col, fixed_effects, iid_cols)
missing <- setdiff(needed, names(raw))
if (length(missing) > 0L) {
    stop(sprintf("Missing column(s): %s", paste(missing, collapse = ", ")), call. = FALSE)
}

raw <- raw[get(period_col) %in% c(base_period, update_period)]
raw[, (response_col) := as_number(get(response_col))]
raw[, (exposure_col) := as_number(get(exposure_col))]
for (col in c(period_col, fixed_effects, iid_cols)) {
    raw[, (col) := as.character(get(col))]
}
model_rows <- is.finite(raw[[response_col]]) &
    raw[[response_col]] >= 0 &
    is.finite(raw[[exposure_col]]) &
    raw[[exposure_col]] >= 0
for (col in c(period_col, fixed_effects, iid_cols)) {
    model_rows <- model_rows & !is.na(raw[[col]]) & nzchar(raw[[col]])
}
raw <- raw[model_rows]

group_cols <- c(period_col, fixed_effects, iid_cols)
model_data <- raw[, list(
    pt = sum(get(response_col)),
    expuesto = sum(get(exposure_col)),
    original_rows = .N
), by = group_cols]
model_data <- model_data[is.finite(get(exposure_col)) & get(exposure_col) > 0]
model_data <- as.data.frame(model_data)

for (col in fixed_effects) {
    model_data[[col]] <- apply_reference_level(model_data[[col]], col, fixed_reference_levels)
}
for (col in iid_cols) {
    model_data[[col]] <- factor(model_data[[col]], levels = sort(unique(model_data[[col]])))
}

base_data <- model_data[model_data[[period_col]] == base_period, , drop = FALSE]
update_data <- model_data[model_data[[period_col]] == update_period, , drop = FALSE]
joint_data <- rbind(base_data, update_data)
joint_rows <- seq.int(nrow(base_data) + 1L, nrow(joint_data))
period_summary <- observed_period_summary(model_data)

cat(sprintf("Formula: %s\n", paste(deparse(model_formula), collapse = " ")))
cat(sprintf("Base rows: %d; update rows: %d; joint rows: %d\n", nrow(base_data), nrow(update_data), nrow(joint_data)))
reference_table <- data.frame(
    fixed_effect = fixed_effects,
    reference_level = vapply(fixed_effects, function(col) levels(model_data[[col]])[[1L]], character(1)),
    check.names = FALSE
)
cat("Fixed-effect reference levels:\n")
print(reference_table)
cat("Observed period frequencies:\n")
print(period_summary[period_summary[[period_col]] %in% c(base_period, update_period), , drop = FALSE])

base_run <- timed_fit(rusty_inla(model_formula, data = base_data, family = family, output_profile = "benchmark"))
state <- rusty_update_state(base_run$value)
update_run <- timed_fit(rusty_inla(
    model_formula,
    data = update_data,
    family = family,
    output_profile = "benchmark",
    control.update = list(
        state = state,
        mode = "fixed_iid_cross_theta_evidence"
    )
))
joint_run <- timed_fit(rusty_inla(model_formula, data = joint_data, family = family, output_profile = "benchmark"))

theta <- theta_comparison(update_run$value, joint_run$value)
effects <- effect_comparison(update_run$value, joint_run$value, update_data)
fixed_effect_table <- effects[effects$effect_type == "fixed", , drop = FALSE]
random_effect_table <- effects[effects$effect_type == "random", , drop = FALSE]
random_stats <- random_effect_stats(effects)
random_top_abs <- random_top_abs_by_block(effects, plot_top_n)
level_predictions <- level_prediction_table(update_run$value, joint_run$value, model_data, fixed_effects, iid_cols)
level_prediction_top <- utils::head(level_predictions[order(-level_predictions$update_frequency_per_exposure), , drop = FALSE], plot_top_n)
prediction_frequency <- prediction_frequency_summary(
    level_predictions,
    period_summary[period_summary[[period_col]] %in% c(base_period, update_period), , drop = FALSE]
)
fitted <- fitted_summary(update_run$value, joint_run$value, joint_rows)
times <- data.frame(
    base_time_sec = base_run$elapsed,
    update_time_sec = update_run$elapsed,
    joint_time_sec = joint_run$elapsed,
    base_object_mb = as.numeric(utils::object.size(base_run$value)) / 1024^2,
    update_object_mb = as.numeric(utils::object.size(update_run$value)) / 1024^2,
    joint_object_mb = as.numeric(utils::object.size(joint_run$value)) / 1024^2,
    check.names = FALSE
)

cat("Times:\n")
print(times)
cat("Theta comparison:\n")
print(theta)
cat("Fitted comparison:\n")
print(fitted)
cat("Prediction frequency sanity:\n")
print(prediction_frequency)
cat("Largest effect drifts:\n")
print(utils::head(effects, 15L))
cat("Random-effect statistics by block:\n")
print(random_stats)
cat(sprintf("Top %d absolute random effects by block:\n", plot_top_n))
print(random_top_abs[, c(
    "effect_group",
    "parameter",
    "current_exposure",
    "update_mean",
    "update_sign",
    "update_relativity",
    "joint_mean",
    "mean_abs_diff"
), drop = FALSE])
cat(sprintf("Top %d predicted level combinations by update frequency:\n", plot_top_n))
print(level_prediction_top)

bar_plot_height <- max(800L, as.integer(280L + 28L * plot_top_n))
plot_paths <- c(
    write_plot(
        "walkthrough_top_effect_drifts.png",
        plot_top_drift(effects, plot_top_n),
        height = bar_plot_height
    ),
    write_plot("walkthrough_fixed_update_vs_joint.png", plot_update_vs_joint(
        fixed_effect_table,
        "fixed",
        "Fixed Effects: Rolling Update vs Joint Refit"
    )),
    write_plot("walkthrough_fixed_effect_estimates_signed.png", plot_fixed_effect_estimates(
        fixed_effect_table
    )),
    write_plot("walkthrough_random_update_vs_joint.png", plot_update_vs_joint(
        random_effect_table,
        "random",
        "Random Effects: Rolling Update vs Joint Refit"
    )),
    write_plot(
        "walkthrough_random_sd_ratio_top.png",
        plot_random_sd_ratio(effects, plot_top_n),
        height = bar_plot_height
    ),
    write_plot(
        "walkthrough_level_predictions_top.png",
        plot_level_prediction_top(level_predictions, fixed_effects, iid_cols, plot_top_n),
        height = bar_plot_height
    )
)
random_abs_plot_paths <- vapply(iid_cols, function(col) {
    write_plot(
        sprintf("walkthrough_random_%s_top_abs_signed.png", file_label(col)),
        plot_random_abs_effect(effects, col, plot_top_n),
        height = bar_plot_height
    )
}, character(1))
plot_paths <- c(plot_paths, random_abs_plot_paths)

cat("Reports:\n")
cat(sprintf("  %s\n", write_report(reference_table, "walkthrough_fixed_reference_levels.csv")))
cat(sprintf("  %s\n", write_report(period_summary, "walkthrough_observed_period_frequencies.csv")))
cat(sprintf("  %s\n", write_report(times, "walkthrough_times.csv")))
cat(sprintf("  %s\n", write_report(theta, "walkthrough_theta_proxy_vs_joint.csv")))
cat(sprintf("  %s\n", write_report(effects, "walkthrough_effects_proxy_vs_joint.csv")))
cat(sprintf("  %s\n", write_report(fixed_effect_table, "walkthrough_fixed_effects_proxy_vs_joint.csv")))
cat(sprintf("  %s\n", write_report(random_effect_table, "walkthrough_random_effects_proxy_vs_joint.csv")))
cat(sprintf("  %s\n", write_report(random_stats, "walkthrough_random_effect_stats_by_block.csv")))
cat(sprintf("  %s\n", write_report(random_top_abs, "walkthrough_random_top_abs_by_block.csv")))
cat(sprintf("  %s\n", write_report(level_predictions, "walkthrough_level_predictions.csv")))
cat(sprintf("  %s\n", write_report(level_prediction_top, "walkthrough_level_predictions_top.csv")))
cat(sprintf("  %s\n", write_report(prediction_frequency, "walkthrough_prediction_frequency_summary.csv")))
cat(sprintf("  %s\n", write_report(fitted, "walkthrough_fitted_summary.csv")))
cat("Plots:\n")
cat(sprintf("  %s\n", plot_paths))
