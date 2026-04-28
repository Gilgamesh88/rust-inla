source(file.path(getwd(), "tools", "load_worktree_package.R"), local = TRUE)
load_rustyinla_for_benchmarks(getwd())

if (!requireNamespace("CASdatasets", quietly = TRUE)) {
    stop("Package 'CASdatasets' is required for the freMTPL born-brand experiment.", call. = FALSE)
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

data("freMTPL2freq", package = "CASdatasets")

df <- get("freMTPL2freq")
df <- df[is.finite(df$ClaimNb) & is.finite(df$Exposure) & df$Exposure > 0, ]
df <- df[order(df$IDpol), ]
df$VehBrand <- factor(df$VehBrand)

exposure_by_brand <- tapply(df$Exposure, df$VehBrand, sum)
born_brand <- names(which.min(exposure_by_brand))
born_rows <- df$VehBrand == born_brand
born_data <- df[born_rows, ]
rest_data <- df[!born_rows, ]
rest_data <- rest_data[order(rest_data$IDpol), ]

split_1 <- floor(nrow(rest_data) / 3.0)
split_2 <- floor(2.0 * nrow(rest_data) / 3.0)
part1 <- rest_data[seq_len(split_1), ]
part2_rest <- rest_data[(split_1 + 1L):split_2, ]
part3 <- rest_data[(split_2 + 1L):nrow(rest_data), ]
part2 <- rbind(part2_rest, born_data)
part2 <- part2[order(part2$IDpol), ]

part1$VehBrand <- droplevels(part1$VehBrand)
part2$VehBrand <- droplevels(part2$VehBrand)
part3$VehBrand <- droplevels(part3$VehBrand)
joint12_data <- rbind(part1, part2)
joint12_data$VehBrand <- droplevels(joint12_data$VehBrand)
joint123_data <- rbind(joint12_data, part3)
joint123_data$VehBrand <- droplevels(joint123_data$VehBrand)

formula <- ClaimNb ~ 1 + offset(log(Exposure)) + f(VehBrand, model = "iid")

cat("Phase 8 freMTPL born-brand posterior-state experiment\n")
cat("Split strategy: lowest-exposure VehBrand is absent from period 1 and inserted into period 2.\n")
cat("Caveat: freMTPL2freq has no explicit time variable here, so this is a constructed pseudo-period split.\n")
cat(sprintf(
    "Born brand: %s; exposure %.6f; rows %d\n",
    born_brand,
    unname(exposure_by_brand[[born_brand]]),
    nrow(born_data)
))
cat(sprintf(
    "Rows: part1 %d, part2 %d, part3 %d, joint12 %d, joint123 %d\n",
    nrow(part1),
    nrow(part2),
    nrow(part3),
    nrow(joint12_data),
    nrow(joint123_data)
))
cat(sprintf(
    "Brand present? part1=%s, part2=%s, part3=%s\n",
    born_brand %in% levels(part1$VehBrand),
    born_brand %in% levels(part2$VehBrand),
    born_brand %in% levels(part3$VehBrand)
))

fit_part1 <- timed_fit(rusty_inla(formula, data = part1, family = "poisson"))
state_part1 <- rusty_update_state(fit_part1$value)

fit_part2_default <- timed_fit(rusty_inla(formula, data = part2, family = "poisson"))
fit_part2_theta <- timed_fit(rusty_inla(
    formula,
    data = part2,
    family = "poisson",
    control.update = list(
        state = state_part1,
        mode = "fixed_iid_cross_theta_evidence"
    )
))
fit_joint12 <- timed_fit(rusty_inla(formula, data = joint12_data, family = "poisson"))

state_reextract12 <- rusty_update_state(fit_part2_theta$value)
state_composed12 <- rusty_compose_update_state(state_part1, fit_part2_theta$value)

fit_part3_reextract <- timed_fit(rusty_inla(
    formula,
    data = part3,
    family = "poisson",
    control.update = list(
        state = state_reextract12,
        mode = "fixed_iid_cross_theta_evidence"
    )
))
fit_part3_composed <- timed_fit(rusty_inla(
    formula,
    data = part3,
    family = "poisson",
    control.update = list(
        state = state_composed12,
        mode = "fixed_iid_cross_theta_evidence"
    )
))
fit_joint123 <- timed_fit(rusty_inla(formula, data = joint123_data, family = "poisson"))

joint12_part2_rows <- seq.int(nrow(part1) + 1L, nrow(joint12_data))
joint123_part3_rows <- seq.int(nrow(joint12_data) + 1L, nrow(joint123_data))

metrics_12 <- rbind(
    fit_metrics("part2_default_with_born_brand", fit_part2_default$value, fit_part2_default$elapsed, fit_joint12$value, joint12_part2_rows),
    fit_metrics("part2_theta_evidence_with_born_brand", fit_part2_theta$value, fit_part2_theta$elapsed, fit_joint12$value, joint12_part2_rows),
    data.frame(
        fit = "joint12_anchor",
        time_sec = fit_joint12$elapsed,
        theta_log_precision = as.numeric(fit_joint12$value$mode$theta[[1L]]),
        abs_theta_to_joint = 0.0,
        fixed_mean_max_abs_to_joint = 0.0,
        random_mean_max_abs_to_joint = 0.0,
        fitted_new_max_rel_to_joint = 0.0,
        fixed_sd_min_ratio_to_joint = 1.0,
        random_sd_min_ratio_to_joint = 1.0,
        check.names = FALSE
    )
)

metrics_123 <- rbind(
    fit_metrics("part3_theta_from_reextract12", fit_part3_reextract$value, fit_part3_reextract$elapsed, fit_joint123$value, joint123_part3_rows),
    fit_metrics("part3_theta_from_composed12", fit_part3_composed$value, fit_part3_composed$elapsed, fit_joint123$value, joint123_part3_rows),
    data.frame(
        fit = "joint123_anchor",
        time_sec = fit_joint123$elapsed,
        theta_log_precision = as.numeric(fit_joint123$value$mode$theta[[1L]]),
        abs_theta_to_joint = 0.0,
        fixed_mean_max_abs_to_joint = 0.0,
        random_mean_max_abs_to_joint = 0.0,
        fitted_new_max_rel_to_joint = 0.0,
        fixed_sd_min_ratio_to_joint = 1.0,
        random_sd_min_ratio_to_joint = 1.0,
        check.names = FALSE
    )
)

born_metadata <- fit_part2_theta$value$posterior_state_used$born_iid_levels
born_random_update <- named_column(fit_part2_theta$value$summary.random$VehBrand, "mean")[[born_brand]]
born_random_joint12 <- named_column(fit_joint12$value$summary.random$VehBrand, "mean")[[born_brand]]
born_precision_composed <- state_composed12$iid$evidence_precision_diag[[born_brand]]
born_linear_composed <- state_composed12$iid$evidence_linear[[born_brand]]

cat("\nPart 1 -> Part 2 update with born brand, compared with joint(1,2):\n")
print(metrics_12, row.names = FALSE, digits = 6)
cat(sprintf(
    "\nBorn-level metadata: %s\n",
    if (length(born_metadata) == 0L) "none" else paste(born_metadata, collapse = ", ")
))
cat(sprintf(
    "Born brand random mean: update %.6f vs joint12 %.6f; abs diff %.6f\n",
    born_random_update,
    born_random_joint12,
    abs(born_random_update - born_random_joint12)
))
cat(sprintf(
    "Composed state born evidence: precision %.6f, linear %.6f\n",
    born_precision_composed,
    born_linear_composed
))

cat("\nPart 3 update after born-brand period, compared with joint(1,2,3):\n")
print(metrics_123, row.names = FALSE, digits = 6)

if (!(born_brand %in% born_metadata)) {
    stop("Born brand was not recorded as a zero-old-evidence level.", call. = FALSE)
}
if (!(born_brand %in% state_composed12$iid$levels)) {
    stop("Composed state did not retain the born brand after period 2.", call. = FALSE)
}
if (!is.finite(born_precision_composed) || born_precision_composed <= 0.0) {
    stop("Composed state did not add current-period evidence for the born brand.", call. = FALSE)
}

cat(sprintf(
    paste(
        "\nTheta drift:",
        "part2 default %.6f -> theta evidence %.6f;",
        "part3 reextract %.6f -> composed %.6f\n"
    ),
    metrics_12$abs_theta_to_joint[[1L]],
    metrics_12$abs_theta_to_joint[[2L]],
    metrics_123$abs_theta_to_joint[[1L]],
    metrics_123$abs_theta_to_joint[[2L]]
))
cat(sprintf(
    paste(
        "Fitted drift:",
        "part2 default %.6f -> theta evidence %.6f;",
        "part3 reextract %.6f -> composed %.6f\n"
    ),
    metrics_12$fitted_new_max_rel_to_joint[[1L]],
    metrics_12$fitted_new_max_rel_to_joint[[2L]],
    metrics_123$fitted_new_max_rel_to_joint[[1L]],
    metrics_123$fitted_new_max_rel_to_joint[[2L]]
))

invisible(list(part12 = metrics_12, part123 = metrics_123, born_brand = born_brand))
