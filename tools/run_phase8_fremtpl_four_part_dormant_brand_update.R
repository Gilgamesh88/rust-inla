source(file.path(getwd(), "tools", "load_worktree_package.R"), local = TRUE)
load_rustyinla_for_benchmarks(getwd())

if (!requireNamespace("CASdatasets", quietly = TRUE)) {
    stop("Package 'CASdatasets' is required for the freMTPL dormant-brand experiment.", call. = FALSE)
}

timed_fit <- function(expr) {
    elapsed <- system.time(value <- eval.parent(substitute(expr)))
    list(value = value, elapsed = unname(elapsed[["elapsed"]]))
}

named_column <- function(df, column) {
    if (is.null(df) || nrow(df) == 0L || !(column %in% names(df))) {
        return(stats::setNames(numeric(), character()))
    }
    stats::setNames(as.numeric(df[[column]]), rownames(df))
}

max_abs_named_diff <- function(lhs, rhs) {
    shared <- intersect(names(lhs), names(rhs))
    if (length(shared) == 0L) {
        return(NA_real_)
    }
    max(abs(lhs[shared] - rhs[shared]))
}

min_named_ratio <- function(lhs, rhs) {
    shared <- intersect(names(lhs), names(rhs))
    if (length(shared) == 0L) {
        return(NA_real_)
    }
    min(lhs[shared] / pmax(rhs[shared], 1e-12))
}

max_fitted_rel_diff <- function(fit, joint_fit, joint_rows) {
    lhs <- as.numeric(fit$summary.fitted.values$mean)
    rhs <- as.numeric(joint_fit$summary.fitted.values$mean[joint_rows])
    n <- min(length(lhs), length(rhs))
    if (n == 0L) {
        return(NA_real_)
    }
    max(abs(lhs[seq_len(n)] - rhs[seq_len(n)]) / pmax(1.0, abs(rhs[seq_len(n)])))
}

fit_metrics <- function(label, fit, elapsed, joint_fit, joint_rows, random_name = "VehBrand") {
    theta <- as.numeric(fit$mode$theta[[1L]])
    theta_joint <- as.numeric(joint_fit$mode$theta[[1L]])
    fixed <- named_column(fit$summary.fixed, "mean")
    fixed_joint <- named_column(joint_fit$summary.fixed, "mean")
    fixed_sd <- named_column(fit$summary.fixed, "sd")
    fixed_joint_sd <- named_column(joint_fit$summary.fixed, "sd")
    random <- named_column(fit$summary.random[[random_name]], "mean")
    random_joint <- named_column(joint_fit$summary.random[[random_name]], "mean")
    random_sd <- named_column(fit$summary.random[[random_name]], "sd")
    random_joint_sd <- named_column(joint_fit$summary.random[[random_name]], "sd")

    data.frame(
        fit = label,
        time_sec = elapsed,
        theta_log_precision = theta,
        abs_theta_to_joint = abs(theta - theta_joint),
        fixed_mean_max_abs_to_joint = max_abs_named_diff(fixed, fixed_joint),
        random_mean_max_abs_to_joint = max_abs_named_diff(random, random_joint),
        fitted_new_max_rel_to_joint = max_fitted_rel_diff(fit, joint_fit, joint_rows),
        fixed_sd_min_ratio_to_joint = min_named_ratio(fixed_sd, fixed_joint_sd),
        random_sd_min_ratio_to_joint = min_named_ratio(random_sd, random_joint_sd),
        check.names = FALSE
    )
}

anchor_metrics <- function(label, fit, elapsed) {
    data.frame(
        fit = label,
        time_sec = elapsed,
        theta_log_precision = as.numeric(fit$mode$theta[[1L]]),
        abs_theta_to_joint = 0.0,
        fixed_mean_max_abs_to_joint = 0.0,
        random_mean_max_abs_to_joint = 0.0,
        fitted_new_max_rel_to_joint = 0.0,
        fixed_sd_min_ratio_to_joint = 1.0,
        random_sd_min_ratio_to_joint = 1.0,
        check.names = FALSE
    )
}

slice_rows <- function(df, first, last) {
    if (last < first) {
        return(df[0L, , drop = FALSE])
    }
    df[seq.int(first, last), , drop = FALSE]
}

random_mean <- function(fit, level, random_name = "VehBrand") {
    named_column(fit$summary.random[[random_name]], "mean")[[level]]
}

random_sd <- function(fit, level, random_name = "VehBrand") {
    named_column(fit$summary.random[[random_name]], "sd")[[level]]
}

summary_value <- function(df, row_names, column) {
    if (is.null(df) || nrow(df) == 0L || !(column %in% names(df))) {
        return(rep(NA_real_, length(row_names)))
    }
    as.numeric(df[row_names, column])
}

intercept_mean <- function(fit) {
    named_column(fit$summary.fixed, "mean")[["(Intercept)"]]
}

current_random_exposure <- function(data, levels, random_name = "VehBrand") {
    exposure <- tapply(data$Exposure, as.character(data[[random_name]]), sum)
    rows <- tapply(rep.int(1L, nrow(data)), as.character(data[[random_name]]), sum)
    data.frame(
        parameter = levels,
        current_exposure = as.numeric(exposure[levels]),
        current_rows = as.integer(rows[levels]),
        check.names = FALSE
    )
}

effect_comparison_table <- function(stage, proxy_fit, joint_fit, current_data, random_name = "VehBrand") {
    fixed_proxy <- proxy_fit$summary.fixed
    fixed_joint <- joint_fit$summary.fixed
    fixed_names <- union(rownames(fixed_proxy), rownames(fixed_joint))
    fixed_table <- data.frame(
        stage = stage,
        effect_type = "fixed",
        parameter = fixed_names,
        current_exposure = NA_real_,
        current_rows = NA_integer_,
        proxy_mean = as.numeric(fixed_proxy[fixed_names, "mean"]),
        true_refit_mean = as.numeric(fixed_joint[fixed_names, "mean"]),
        proxy_sd = as.numeric(fixed_proxy[fixed_names, "sd"]),
        true_refit_sd = as.numeric(fixed_joint[fixed_names, "sd"]),
        proxy_q025 = summary_value(fixed_proxy, fixed_names, "0.025quant"),
        true_refit_q025 = summary_value(fixed_joint, fixed_names, "0.025quant"),
        proxy_q500 = summary_value(fixed_proxy, fixed_names, "0.5quant"),
        true_refit_q500 = summary_value(fixed_joint, fixed_names, "0.5quant"),
        proxy_q975 = summary_value(fixed_proxy, fixed_names, "0.975quant"),
        true_refit_q975 = summary_value(fixed_joint, fixed_names, "0.975quant"),
        check.names = FALSE
    )

    random_proxy <- proxy_fit$summary.random[[random_name]]
    random_joint <- joint_fit$summary.random[[random_name]]
    random_names <- union(rownames(random_proxy), rownames(random_joint))
    exposure_table <- current_random_exposure(current_data, random_names, random_name)
    random_table <- data.frame(
        stage = stage,
        effect_type = "random",
        parameter = random_names,
        current_exposure = exposure_table$current_exposure,
        current_rows = exposure_table$current_rows,
        proxy_mean = as.numeric(random_proxy[random_names, "mean"]),
        true_refit_mean = as.numeric(random_joint[random_names, "mean"]),
        proxy_sd = as.numeric(random_proxy[random_names, "sd"]),
        true_refit_sd = as.numeric(random_joint[random_names, "sd"]),
        proxy_q025 = summary_value(random_proxy, random_names, "0.025quant"),
        true_refit_q025 = summary_value(random_joint, random_names, "0.025quant"),
        proxy_q500 = summary_value(random_proxy, random_names, "0.5quant"),
        true_refit_q500 = summary_value(random_joint, random_names, "0.5quant"),
        proxy_q975 = summary_value(random_proxy, random_names, "0.975quant"),
        true_refit_q975 = summary_value(random_joint, random_names, "0.975quant"),
        check.names = FALSE
    )

    out <- rbind(fixed_table, random_table)
    random_missing_exposure <- is.na(out$current_exposure) & out$effect_type == "random"
    random_missing_rows <- is.na(out$current_rows) & out$effect_type == "random"
    out$current_exposure[random_missing_exposure] <- 0.0
    out$current_rows[random_missing_rows] <- 0L
    out$mean_abs_diff <- abs(out$proxy_mean - out$true_refit_mean)
    out$sd_ratio <- out$proxy_sd / pmax(out$true_refit_sd, 1e-12)
    out$q025_abs_diff <- abs(out$proxy_q025 - out$true_refit_q025)
    out$q500_abs_diff <- abs(out$proxy_q500 - out$true_refit_q500)
    out$q975_abs_diff <- abs(out$proxy_q975 - out$true_refit_q975)
    out$proxy_interval_width <- out$proxy_q975 - out$proxy_q025
    out$true_refit_interval_width <- out$true_refit_q975 - out$true_refit_q025
    out$interval_width_ratio <- out$proxy_interval_width / pmax(out$true_refit_interval_width, 1e-12)
    out$interval_overlap_width <- pmax(
        0.0,
        pmin(out$proxy_q975, out$true_refit_q975) - pmax(out$proxy_q025, out$true_refit_q025)
    )
    out$interval_overlap_true_fraction <- out$interval_overlap_width /
        pmax(out$true_refit_interval_width, 1e-12)
    out
}

interval_summary_table <- function(detailed_table) {
    split_rows <- split(detailed_table, list(detailed_table$stage, detailed_table$effect_type), drop = TRUE)
    do.call(rbind, lapply(names(split_rows), function(key) {
        rows <- split_rows[[key]]
        data.frame(
            stage = rows$stage[[1L]],
            effect_type = rows$effect_type[[1L]],
            n_parameters = nrow(rows),
            max_mean_abs_diff = max(rows$mean_abs_diff, na.rm = TRUE),
            max_q025_abs_diff = max(rows$q025_abs_diff, na.rm = TRUE),
            max_q500_abs_diff = max(rows$q500_abs_diff, na.rm = TRUE),
            max_q975_abs_diff = max(rows$q975_abs_diff, na.rm = TRUE),
            min_sd_ratio = min(rows$sd_ratio, na.rm = TRUE),
            max_sd_ratio = max(rows$sd_ratio, na.rm = TRUE),
            min_interval_width_ratio = min(rows$interval_width_ratio, na.rm = TRUE),
            max_interval_width_ratio = max(rows$interval_width_ratio, na.rm = TRUE),
            min_interval_overlap_true_fraction = min(rows$interval_overlap_true_fraction, na.rm = TRUE),
            check.names = FALSE
        )
    }))
}

theta_marginal_table <- function(stage, proxy_fit, joint_fit) {
    proxy <- proxy_fit$summary.hyperpar
    joint <- joint_fit$summary.hyperpar
    theta_names <- union(rownames(proxy), rownames(joint))
    out <- data.frame(
        stage = stage,
        theta_parameter = theta_names,
        proxy_mean = summary_value(proxy, theta_names, "mean"),
        true_refit_mean = summary_value(joint, theta_names, "mean"),
        proxy_sd = summary_value(proxy, theta_names, "sd"),
        true_refit_sd = summary_value(joint, theta_names, "sd"),
        proxy_q025 = summary_value(proxy, theta_names, "0.025quant"),
        true_refit_q025 = summary_value(joint, theta_names, "0.025quant"),
        proxy_q500 = summary_value(proxy, theta_names, "0.5quant"),
        true_refit_q500 = summary_value(joint, theta_names, "0.5quant"),
        proxy_q975 = summary_value(proxy, theta_names, "0.975quant"),
        true_refit_q975 = summary_value(joint, theta_names, "0.975quant"),
        proxy_mode = summary_value(proxy, theta_names, "mode"),
        true_refit_mode = summary_value(joint, theta_names, "mode"),
        check.names = FALSE
    )
    out$mean_abs_diff <- abs(out$proxy_mean - out$true_refit_mean)
    out$sd_ratio <- out$proxy_sd / pmax(out$true_refit_sd, 1e-12)
    out$q025_abs_diff <- abs(out$proxy_q025 - out$true_refit_q025)
    out$q500_abs_diff <- abs(out$proxy_q500 - out$true_refit_q500)
    out$q975_abs_diff <- abs(out$proxy_q975 - out$true_refit_q975)
    out$mode_abs_diff <- abs(out$proxy_mode - out$true_refit_mode)
    out$mean_rel_diff <- out$mean_abs_diff / pmax(1.0, abs(out$true_refit_mean))
    out$q025_rel_diff <- out$q025_abs_diff / pmax(1.0, abs(out$true_refit_q025))
    out$q500_rel_diff <- out$q500_abs_diff / pmax(1.0, abs(out$true_refit_q500))
    out$q975_rel_diff <- out$q975_abs_diff / pmax(1.0, abs(out$true_refit_q975))
    out$mode_rel_diff <- out$mode_abs_diff / pmax(1.0, abs(out$true_refit_mode))
    out$interval_width_ratio <- (out$proxy_q975 - out$proxy_q025) /
        pmax(out$true_refit_q975 - out$true_refit_q025, 1e-12)
    out
}

trapz <- function(x, y) {
    if (length(x) < 2L) {
        return(NA_real_)
    }
    sum(diff(x) * (head(y, -1L) + tail(y, -1L)) / 2.0)
}

marginal_grid_distance <- function(proxy_grid, joint_grid, n = 400L) {
    if (is.null(proxy_grid) || is.null(joint_grid)) {
        return(c(l1_density = NA_real_, cdf_ks = NA_real_))
    }
    lo <- max(min(proxy_grid[, "x"]), min(joint_grid[, "x"]))
    hi <- min(max(proxy_grid[, "x"]), max(joint_grid[, "x"]))
    if (!is.finite(lo) || !is.finite(hi) || lo >= hi) {
        return(c(l1_density = NA_real_, cdf_ks = NA_real_))
    }
    x <- seq(lo, hi, length.out = n)
    proxy_y <- approx(proxy_grid[, "x"], proxy_grid[, "y"], xout = x, rule = 2)$y
    joint_y <- approx(joint_grid[, "x"], joint_grid[, "y"], xout = x, rule = 2)$y
    proxy_area <- trapz(x, proxy_y)
    joint_area <- trapz(x, joint_y)
    if (!is.finite(proxy_area) || proxy_area <= 0.0 ||
        !is.finite(joint_area) || joint_area <= 0.0) {
        return(c(l1_density = NA_real_, cdf_ks = NA_real_))
    }
    proxy_y <- proxy_y / proxy_area
    joint_y <- joint_y / joint_area
    l1_density <- trapz(x, abs(proxy_y - joint_y))
    proxy_cdf <- cumsum(c(0.0, diff(x) * (head(proxy_y, -1L) + tail(proxy_y, -1L)) / 2.0))
    joint_cdf <- cumsum(c(0.0, diff(x) * (head(joint_y, -1L) + tail(joint_y, -1L)) / 2.0))
    c(l1_density = l1_density, cdf_ks = max(abs(proxy_cdf - joint_cdf)))
}

theta_grid_distance_table <- function(stage, proxy_fit, joint_fit) {
    proxy_marginals <- proxy_fit$marginals.hyperpar
    joint_marginals <- joint_fit$marginals.hyperpar
    theta_names <- union(names(proxy_marginals), names(joint_marginals))
    if (length(theta_names) == 0L) {
        return(data.frame(
            stage = character(),
            theta_parameter = character(),
            l1_density = numeric(),
            cdf_ks = numeric(),
            check.names = FALSE
        ))
    }
    do.call(rbind, lapply(theta_names, function(theta_name) {
        distances <- marginal_grid_distance(proxy_marginals[[theta_name]], joint_marginals[[theta_name]])
        data.frame(
            stage = stage,
            theta_parameter = theta_name,
            l1_density = unname(distances[["l1_density"]]),
            cdf_ks = unname(distances[["cdf_ks"]]),
            check.names = FALSE
        )
    }))
}

data("freMTPL2freq", package = "CASdatasets")

df <- get("freMTPL2freq")
df <- df[is.finite(df$ClaimNb) & is.finite(df$Exposure) & df$Exposure > 0, ]
df <- df[order(df$IDpol), ]
df$VehBrand <- factor(df$VehBrand)

exposure_by_brand <- tapply(df$Exposure, df$VehBrand, sum)
born_brand <- names(which.min(exposure_by_brand))
born_rows <- df$VehBrand == born_brand
born_data <- df[born_rows, ]
born_data <- born_data[order(born_data$IDpol), ]
rest_data <- df[!born_rows, ]
rest_data <- rest_data[order(rest_data$IDpol), ]

rest_breaks <- floor(seq(0, nrow(rest_data), length.out = 5L))
part1 <- slice_rows(rest_data, rest_breaks[[1L]] + 1L, rest_breaks[[2L]])
part2_rest <- slice_rows(rest_data, rest_breaks[[2L]] + 1L, rest_breaks[[3L]])
part3 <- slice_rows(rest_data, rest_breaks[[3L]] + 1L, rest_breaks[[4L]])
part4_rest <- slice_rows(rest_data, rest_breaks[[4L]] + 1L, rest_breaks[[5L]])

born_cut <- floor(nrow(born_data) / 2.0)
born_part2 <- slice_rows(born_data, 1L, born_cut)
born_part4 <- slice_rows(born_data, born_cut + 1L, nrow(born_data))

part2 <- rbind(part2_rest, born_part2)
part4 <- rbind(part4_rest, born_part4)
part2 <- part2[order(part2$IDpol), ]
part4 <- part4[order(part4$IDpol), ]

part1$VehBrand <- droplevels(part1$VehBrand)
part2$VehBrand <- droplevels(part2$VehBrand)
part3$VehBrand <- droplevels(part3$VehBrand)
part4$VehBrand <- droplevels(part4$VehBrand)
joint12_data <- rbind(part1, part2)
joint12_data$VehBrand <- droplevels(joint12_data$VehBrand)
joint123_data <- rbind(joint12_data, part3)
joint123_data$VehBrand <- droplevels(joint123_data$VehBrand)
joint1234_data <- rbind(joint123_data, part4)
joint1234_data$VehBrand <- droplevels(joint1234_data$VehBrand)

formula <- ClaimNb ~ 1 + offset(log(Exposure)) + f(VehBrand, model = "iid")
fit_output_profile <- "benchmark"

fit_model <- function(data, control.update = NULL) {
    if (is.null(control.update)) {
        return(rusty_inla(
            formula,
            data = data,
            family = "poisson",
            output_profile = fit_output_profile
        ))
    }
    rusty_inla(
        formula,
        data = data,
        family = "poisson",
        output_profile = fit_output_profile,
        control.update = control.update
    )
}

cat("Phase 8 freMTPL four-part dormant-brand posterior-state experiment\n")
cat("Split strategy: lowest-exposure VehBrand is absent in period 1, split into periods 2 and 4, and dormant in period 3.\n")
cat("Caveat: freMTPL2freq has no explicit time variable here, so this is a constructed pseudo-period split.\n")
cat(sprintf(
    "Tracked brand: %s; total exposure %.6f; total rows %d; period2 rows %d; period4 rows %d\n",
    born_brand,
    unname(exposure_by_brand[[born_brand]]),
    nrow(born_data),
    nrow(born_part2),
    nrow(born_part4)
))
cat(sprintf(
    "Rows: p1 %d, p2 %d, p3 %d, p4 %d, joint12 %d, joint123 %d, joint1234 %d\n",
    nrow(part1),
    nrow(part2),
    nrow(part3),
    nrow(part4),
    nrow(joint12_data),
    nrow(joint123_data),
    nrow(joint1234_data)
))
cat(sprintf(
    "Tracked brand present? p1=%s, p2=%s, p3=%s, p4=%s\n",
    born_brand %in% as.character(part1$VehBrand),
    born_brand %in% as.character(part2$VehBrand),
    born_brand %in% as.character(part3$VehBrand),
    born_brand %in% as.character(part4$VehBrand)
))

fit_part1 <- timed_fit(fit_model(part1))
state_part1 <- rusty_update_state(fit_part1$value)

fit_part2_default <- timed_fit(fit_model(part2))
fit_part2_update <- timed_fit(fit_model(
    part2,
    control.update = list(
        state = state_part1,
        mode = "fixed_iid_cross_theta_evidence"
    )
))
fit_joint12 <- timed_fit(fit_model(joint12_data))
state_part12 <- rusty_compose_update_state(state_part1, fit_part2_update$value)

fit_part3_update <- timed_fit(fit_model(
    part3,
    control.update = list(
        state = state_part12,
        mode = "fixed_iid_cross_theta_evidence"
    )
))
fit_joint123 <- timed_fit(fit_model(joint123_data))
state_part123 <- rusty_compose_update_state(state_part12, fit_part3_update$value)

fit_part4_default <- timed_fit(fit_model(part4))
fit_part4_update <- timed_fit(fit_model(
    part4,
    control.update = list(
        state = state_part123,
        mode = "fixed_iid_cross_theta_evidence"
    )
))
fit_joint1234 <- timed_fit(fit_model(joint1234_data))

joint12_part2_rows <- seq.int(nrow(part1) + 1L, nrow(joint12_data))
joint123_part3_rows <- seq.int(nrow(joint12_data) + 1L, nrow(joint123_data))
joint1234_part4_rows <- seq.int(nrow(joint123_data) + 1L, nrow(joint1234_data))

metrics_12 <- rbind(
    fit_metrics("period2_default_with_born_brand", fit_part2_default$value, fit_part2_default$elapsed, fit_joint12$value, joint12_part2_rows),
    fit_metrics("period2_theta_evidence_with_born_brand", fit_part2_update$value, fit_part2_update$elapsed, fit_joint12$value, joint12_part2_rows),
    anchor_metrics("joint12_anchor", fit_joint12$value, fit_joint12$elapsed)
)
metrics_123 <- rbind(
    fit_metrics("period3_theta_evidence_dormant_brand", fit_part3_update$value, fit_part3_update$elapsed, fit_joint123$value, joint123_part3_rows),
    anchor_metrics("joint123_anchor", fit_joint123$value, fit_joint123$elapsed)
)
metrics_1234 <- rbind(
    fit_metrics("period4_default_reentry_brand", fit_part4_default$value, fit_part4_default$elapsed, fit_joint1234$value, joint1234_part4_rows),
    fit_metrics("period4_theta_evidence_reentry_brand", fit_part4_update$value, fit_part4_update$elapsed, fit_joint1234$value, joint1234_part4_rows),
    anchor_metrics("joint1234_anchor", fit_joint1234$value, fit_joint1234$elapsed)
)

summary_table <- data.frame(
    stage = c(
        "period2_default",
        "period2_update",
        "joint12",
        "period3_update_dormant",
        "joint123",
        "period4_default",
        "period4_update_reentry",
        "joint1234"
    ),
    time_sec = c(
        fit_part2_default$elapsed,
        fit_part2_update$elapsed,
        fit_joint12$elapsed,
        fit_part3_update$elapsed,
        fit_joint123$elapsed,
        fit_part4_default$elapsed,
        fit_part4_update$elapsed,
        fit_joint1234$elapsed
    ),
    theta_log_precision = c(
        as.numeric(fit_part2_default$value$mode$theta[[1L]]),
        as.numeric(fit_part2_update$value$mode$theta[[1L]]),
        as.numeric(fit_joint12$value$mode$theta[[1L]]),
        as.numeric(fit_part3_update$value$mode$theta[[1L]]),
        as.numeric(fit_joint123$value$mode$theta[[1L]]),
        as.numeric(fit_part4_default$value$mode$theta[[1L]]),
        as.numeric(fit_part4_update$value$mode$theta[[1L]]),
        as.numeric(fit_joint1234$value$mode$theta[[1L]])
    ),
    intercept_mean = c(
        intercept_mean(fit_part2_default$value),
        intercept_mean(fit_part2_update$value),
        intercept_mean(fit_joint12$value),
        intercept_mean(fit_part3_update$value),
        intercept_mean(fit_joint123$value),
        intercept_mean(fit_part4_default$value),
        intercept_mean(fit_part4_update$value),
        intercept_mean(fit_joint1234$value)
    ),
    tracked_brand_mean = c(
        random_mean(fit_part2_default$value, born_brand),
        random_mean(fit_part2_update$value, born_brand),
        random_mean(fit_joint12$value, born_brand),
        random_mean(fit_part3_update$value, born_brand),
        random_mean(fit_joint123$value, born_brand),
        random_mean(fit_part4_default$value, born_brand),
        random_mean(fit_part4_update$value, born_brand),
        random_mean(fit_joint1234$value, born_brand)
    ),
    tracked_brand_sd = c(
        random_sd(fit_part2_default$value, born_brand),
        random_sd(fit_part2_update$value, born_brand),
        random_sd(fit_joint12$value, born_brand),
        random_sd(fit_part3_update$value, born_brand),
        random_sd(fit_joint123$value, born_brand),
        random_sd(fit_part4_default$value, born_brand),
        random_sd(fit_part4_update$value, born_brand),
        random_sd(fit_joint1234$value, born_brand)
    ),
    check.names = FALSE
)

summary_table$theta_abs_to_anchor <- c(
    abs(summary_table$theta_log_precision[[1L]] - summary_table$theta_log_precision[[3L]]),
    abs(summary_table$theta_log_precision[[2L]] - summary_table$theta_log_precision[[3L]]),
    0.0,
    abs(summary_table$theta_log_precision[[4L]] - summary_table$theta_log_precision[[5L]]),
    0.0,
    abs(summary_table$theta_log_precision[[6L]] - summary_table$theta_log_precision[[8L]]),
    abs(summary_table$theta_log_precision[[7L]] - summary_table$theta_log_precision[[8L]]),
    0.0
)
summary_table$tracked_brand_abs_to_anchor <- c(
    abs(summary_table$tracked_brand_mean[[1L]] - summary_table$tracked_brand_mean[[3L]]),
    abs(summary_table$tracked_brand_mean[[2L]] - summary_table$tracked_brand_mean[[3L]]),
    0.0,
    abs(summary_table$tracked_brand_mean[[4L]] - summary_table$tracked_brand_mean[[5L]]),
    0.0,
    abs(summary_table$tracked_brand_mean[[6L]] - summary_table$tracked_brand_mean[[8L]]),
    abs(summary_table$tracked_brand_mean[[7L]] - summary_table$tracked_brand_mean[[8L]]),
    0.0
)

report_dir <- file.path(getwd(), "scratch", "phase8_reports")
dir.create(report_dir, recursive = TRUE, showWarnings = FALSE)
csv_path <- file.path(report_dir, "four_part_dormant_brand_parameter_drift.csv")
png_path <- file.path(report_dir, "four_part_dormant_brand_parameter_drift.png")
detailed_csv_path <- file.path(report_dir, "four_part_all_effects_proxy_vs_joint.csv")
interval_summary_csv_path <- file.path(report_dir, "four_part_interval_summary.csv")
theta_summary_csv_path <- file.path(report_dir, "four_part_theta_marginal_proxy_vs_joint.csv")
theta_grid_csv_path <- file.path(report_dir, "four_part_theta_grid_distance.csv")
utils::write.csv(summary_table, csv_path, row.names = FALSE)

detailed_table <- rbind(
    effect_comparison_table(
        stage = "period2_born",
        proxy_fit = fit_part2_update$value,
        joint_fit = fit_joint12$value,
        current_data = part2
    ),
    effect_comparison_table(
        stage = "period3_dormant",
        proxy_fit = fit_part3_update$value,
        joint_fit = fit_joint123$value,
        current_data = part3
    ),
    effect_comparison_table(
        stage = "period4_reentry",
        proxy_fit = fit_part4_update$value,
        joint_fit = fit_joint1234$value,
        current_data = part4
    )
)
utils::write.csv(detailed_table, detailed_csv_path, row.names = FALSE)

interval_summary <- interval_summary_table(detailed_table)
utils::write.csv(interval_summary, interval_summary_csv_path, row.names = FALSE)

theta_summary <- rbind(
    theta_marginal_table(
        stage = "period2_born",
        proxy_fit = fit_part2_update$value,
        joint_fit = fit_joint12$value
    ),
    theta_marginal_table(
        stage = "period3_dormant",
        proxy_fit = fit_part3_update$value,
        joint_fit = fit_joint123$value
    ),
    theta_marginal_table(
        stage = "period4_reentry",
        proxy_fit = fit_part4_update$value,
        joint_fit = fit_joint1234$value
    )
)
utils::write.csv(theta_summary, theta_summary_csv_path, row.names = FALSE)

theta_grid_distance <- rbind(
    theta_grid_distance_table(
        stage = "period2_born",
        proxy_fit = fit_part2_update$value,
        joint_fit = fit_joint12$value
    ),
    theta_grid_distance_table(
        stage = "period3_dormant",
        proxy_fit = fit_part3_update$value,
        joint_fit = fit_joint123$value
    ),
    theta_grid_distance_table(
        stage = "period4_reentry",
        proxy_fit = fit_part4_update$value,
        joint_fit = fit_joint1234$value
    )
)
utils::write.csv(theta_grid_distance, theta_grid_csv_path, row.names = FALSE)

png(filename = png_path, width = 1100, height = 720)
old_par <- par(no.readonly = TRUE)
on.exit(par(old_par), add = TRUE)
par(mfrow = c(1L, 2L), mar = c(5, 5, 4, 2))
period_x <- c(2, 3, 4)
joint_brand <- c(
    random_mean(fit_joint12$value, born_brand),
    random_mean(fit_joint123$value, born_brand),
    random_mean(fit_joint1234$value, born_brand)
)
update_brand <- c(
    random_mean(fit_part2_update$value, born_brand),
    random_mean(fit_part3_update$value, born_brand),
    random_mean(fit_part4_update$value, born_brand)
)
default_brand <- c(
    random_mean(fit_part2_default$value, born_brand),
    NA_real_,
    random_mean(fit_part4_default$value, born_brand)
)
ylim_brand <- range(c(joint_brand, update_brand, default_brand), finite = TRUE)
plot(period_x, joint_brand, type = "b", pch = 19, col = "#1b4f72",
     xaxt = "n", xlab = "Pseudo-period", ylab = sprintf("%s iid mean", born_brand),
     ylim = ylim_brand, main = "Tracked Brand Parameter")
axis(1, at = period_x, labels = c("2 born", "3 dormant", "4 re-entry"))
lines(period_x, update_brand, type = "b", pch = 17, col = "#c0392b")
lines(period_x, default_brand, type = "b", pch = 15, col = "#7d3c98")
legend("topleft", legend = c("joint refit", "rolling update", "new-data only"),
       col = c("#1b4f72", "#c0392b", "#7d3c98"), pch = c(19, 17, 15), bty = "n")

joint_theta <- c(
    as.numeric(fit_joint12$value$mode$theta[[1L]]),
    as.numeric(fit_joint123$value$mode$theta[[1L]]),
    as.numeric(fit_joint1234$value$mode$theta[[1L]])
)
update_theta <- c(
    as.numeric(fit_part2_update$value$mode$theta[[1L]]),
    as.numeric(fit_part3_update$value$mode$theta[[1L]]),
    as.numeric(fit_part4_update$value$mode$theta[[1L]])
)
default_theta <- c(
    as.numeric(fit_part2_default$value$mode$theta[[1L]]),
    NA_real_,
    as.numeric(fit_part4_default$value$mode$theta[[1L]])
)
ylim_theta <- range(c(joint_theta, update_theta, default_theta), finite = TRUE)
plot(period_x, joint_theta, type = "b", pch = 19, col = "#1b4f72",
     xaxt = "n", xlab = "Pseudo-period", ylab = "iid log-precision",
     ylim = ylim_theta, main = "Hyperparameter Path")
axis(1, at = period_x, labels = c("2 born", "3 dormant", "4 re-entry"))
lines(period_x, update_theta, type = "b", pch = 17, col = "#c0392b")
lines(period_x, default_theta, type = "b", pch = 15, col = "#7d3c98")
legend("topleft", legend = c("joint refit", "rolling update", "new-data only"),
       col = c("#1b4f72", "#c0392b", "#7d3c98"), pch = c(19, 17, 15), bty = "n")
dev.off()

cat(sprintf("\nInitial period1 fit time: %.2fs\n", fit_part1$elapsed))
cat("\nPeriod 1 -> Period 2 with born brand, compared with joint(1,2):\n")
print(metrics_12, row.names = FALSE, digits = 6)
cat("\nPeriod 3 dormant-brand carry, compared with joint(1,2,3):\n")
print(metrics_123, row.names = FALSE, digits = 6)
cat("\nPeriod 4 re-entry after dormant carry, compared with joint(1,2,3,4):\n")
print(metrics_1234, row.names = FALSE, digits = 6)
cat("\nTracked-brand parameter table:\n")
print(summary_table, row.names = FALSE, digits = 6)
cat("\nAll fixed and random effects, rolling proxy vs true joint refit:\n")
print(detailed_table, row.names = FALSE, digits = 6)
cat("\nInterval validation summary, rolling proxy vs true joint refit:\n")
print(interval_summary, row.names = FALSE, digits = 6)
cat("\nTheta marginal summary, rolling proxy vs true joint refit:\n")
print(theta_summary, row.names = FALSE, digits = 6)
cat("\nTheta marginal grid distances, rolling proxy vs true joint refit:\n")
print(theta_grid_distance, row.names = FALSE, digits = 6)
cat(sprintf("\nReport CSV: %s\n", csv_path))
cat(sprintf("Detailed effects CSV: %s\n", detailed_csv_path))
cat(sprintf("Interval summary CSV: %s\n", interval_summary_csv_path))
cat(sprintf("Theta marginal summary CSV: %s\n", theta_summary_csv_path))
cat(sprintf("Theta marginal grid distance CSV: %s\n", theta_grid_csv_path))
cat(sprintf("Report PNG: %s\n", png_path))

period2_metadata <- fit_part2_update$value$posterior_state_used
period3_metadata <- fit_part3_update$value$posterior_state_used
period4_metadata <- fit_part4_update$value$posterior_state_used
cat(sprintf(
    "\nMetadata: period2 born=%s; period3 dormant=%s carried=%s; period4 active=%s dormant=%s\n",
    paste(period2_metadata$born_iid_levels, collapse = ","),
    paste(period3_metadata$dormant_iid_levels, collapse = ","),
    paste(period3_metadata$carried_iid_levels, collapse = ","),
    paste(period4_metadata$active_iid_levels, collapse = ","),
    paste(period4_metadata$dormant_iid_levels, collapse = ",")
))

if (!(born_brand %in% period2_metadata$born_iid_levels)) {
    stop("Tracked brand was not recorded as a born level in period 2.", call. = FALSE)
}
if (!(born_brand %in% period3_metadata$dormant_iid_levels)) {
    stop("Tracked brand was not recorded as dormant in period 3.", call. = FALSE)
}
if (!(born_brand %in% rownames(fit_part3_update$value$summary.random$VehBrand))) {
    stop("Dormant tracked brand is missing from the period 3 random-effect table.", call. = FALSE)
}
if (!(born_brand %in% period4_metadata$active_iid_levels)) {
    stop("Tracked brand was not recorded as active again in period 4.", call. = FALSE)
}
if (!is.finite(summary_table$tracked_brand_abs_to_anchor[[4L]]) ||
    summary_table$tracked_brand_abs_to_anchor[[4L]] > 0.35) {
    stop("Dormant period tracked-brand drift is larger than expected.", call. = FALSE)
}
if (!is.finite(summary_table$tracked_brand_abs_to_anchor[[7L]]) ||
    summary_table$tracked_brand_abs_to_anchor[[7L]] > 0.35) {
    stop("Re-entry period tracked-brand drift is larger than expected.", call. = FALSE)
}
if (min(interval_summary$min_interval_width_ratio, na.rm = TRUE) < 0.95) {
    stop("At least one proxy interval is materially narrower than the joint-refit interval.", call. = FALSE)
}
if (max(interval_summary$max_q025_abs_diff, interval_summary$max_q975_abs_diff, na.rm = TRUE) > 0.05) {
    stop("At least one proxy interval endpoint drift is larger than expected.", call. = FALSE)
}
if (max(theta_summary$q025_rel_diff, theta_summary$q975_rel_diff, na.rm = TRUE) > 0.08) {
    stop("Theta marginal endpoint drift is larger than expected.", call. = FALSE)
}
if (max(theta_grid_distance$cdf_ks, na.rm = TRUE) > 0.05) {
    stop("Theta marginal CDF distance is larger than expected.", call. = FALSE)
}

cat("four_part_dormant_brand_update: PASS\n")

invisible(list(
    part12 = metrics_12,
    part123 = metrics_123,
    part1234 = metrics_1234,
    summary = summary_table,
    detailed = detailed_table,
    interval_summary = interval_summary,
    theta_summary = theta_summary,
    theta_grid_distance = theta_grid_distance,
    born_brand = born_brand,
    csv = csv_path,
    detailed_csv = detailed_csv_path,
    interval_summary_csv = interval_summary_csv_path,
    theta_summary_csv = theta_summary_csv_path,
    theta_grid_csv = theta_grid_csv_path,
    png = png_path
))
