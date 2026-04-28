source(file.path(getwd(), "tools", "load_worktree_package.R"), local = TRUE)
load_rustyinla_for_benchmarks(getwd())

if (!requireNamespace("CASdatasets", quietly = TRUE)) {
    stop("Package 'CASdatasets' is required for the freMTPL three-part experiment.", call. = FALSE)
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

data("freMTPL2freq", package = "CASdatasets")

df <- get("freMTPL2freq")
df <- df[is.finite(df$ClaimNb) & is.finite(df$Exposure) & df$Exposure > 0, ]
df <- df[order(df$IDpol), ]
df$VehBrand <- factor(df$VehBrand)

split_1 <- floor(nrow(df) / 3.0)
split_2 <- floor(2.0 * nrow(df) / 3.0)

part1 <- df[seq_len(split_1), ]
part2 <- df[(split_1 + 1L):split_2, ]
part3 <- df[(split_2 + 1L):nrow(df), ]
for (name in c("part1", "part2", "part3")) {
    value <- get(name)
    value$VehBrand <- factor(value$VehBrand, levels = levels(df$VehBrand))
    assign(name, value)
}

joint12_data <- rbind(part1, part2)
joint123_data <- rbind(joint12_data, part3)
joint12_data$VehBrand <- factor(joint12_data$VehBrand, levels = levels(df$VehBrand))
joint123_data$VehBrand <- factor(joint123_data$VehBrand, levels = levels(df$VehBrand))

born_12 <- setdiff(sort(unique(as.character(part2$VehBrand))), sort(unique(as.character(part1$VehBrand))))
born_123 <- setdiff(sort(unique(as.character(part3$VehBrand))), sort(unique(as.character(joint12_data$VehBrand))))

formula <- ClaimNb ~ 1 + offset(log(Exposure)) + f(VehBrand, model = "iid")

cat("Phase 8 freMTPL three-part posterior-state experiment\n")
cat("Split strategy: ordered by IDpol into thirds.\n")
cat("Caveat: freMTPL2freq has no explicit time variable here, so this is a pseudo-period split.\n")
cat(sprintf(
    "Rows: part1 %d, part2 %d, part3 %d, joint12 %d, joint123 %d; VehBrand levels %d\n",
    nrow(part1),
    nrow(part2),
    nrow(part3),
    nrow(joint12_data),
    nrow(joint123_data),
    nlevels(df$VehBrand)
))
cat(sprintf("Born levels part2 vs part1: %s\n", if (length(born_12) == 0L) "none" else paste(born_12, collapse = ", ")))
cat(sprintf("Born levels part3 vs joint12: %s\n", if (length(born_123) == 0L) "none" else paste(born_123, collapse = ", ")))

fit_part1 <- timed_fit(rusty_inla(formula, data = part1, family = "poisson"))
state_part1 <- rusty_update_state(fit_part1$value)

fit_part2_default <- timed_fit(rusty_inla(formula, data = part2, family = "poisson"))
fit_part2_cross <- timed_fit(rusty_inla(
    formula,
    data = part2,
    family = "poisson",
    control.update = list(
        state = state_part1,
        mode = "fixed_iid_cross_gaussian_evidence"
    )
))
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
state_joint12 <- rusty_update_state(fit_joint12$value)

fit_part3_default <- timed_fit(rusty_inla(formula, data = part3, family = "poisson"))
fit_part3_cross_from_refit12 <- timed_fit(rusty_inla(
    formula,
    data = part3,
    family = "poisson",
    control.update = list(
        state = state_joint12,
        mode = "fixed_iid_cross_gaussian_evidence"
    )
))
fit_part3_theta_from_refit12 <- timed_fit(rusty_inla(
    formula,
    data = part3,
    family = "poisson",
    control.update = list(
        state = state_joint12,
        mode = "fixed_iid_cross_theta_evidence"
    )
))

state_rolling_reextract <- rusty_update_state(fit_part2_theta$value)
fit_part3_theta_from_reextract <- timed_fit(rusty_inla(
    formula,
    data = part3,
    family = "poisson",
    control.update = list(
        state = state_rolling_reextract,
        mode = "fixed_iid_cross_theta_evidence"
    )
))
state_rolling_composed <- rusty_compose_update_state(state_part1, fit_part2_theta$value)
fit_part3_theta_from_composed <- timed_fit(rusty_inla(
    formula,
    data = part3,
    family = "poisson",
    control.update = list(
        state = state_rolling_composed,
        mode = "fixed_iid_cross_theta_evidence"
    )
))

fit_joint123 <- timed_fit(rusty_inla(formula, data = joint123_data, family = "poisson"))

joint12_part2_rows <- seq.int(nrow(part1) + 1L, nrow(joint12_data))
joint123_part3_rows <- seq.int(nrow(joint12_data) + 1L, nrow(joint123_data))

metrics_12 <- rbind(
    fit_metrics("part2_default", fit_part2_default$value, fit_part2_default$elapsed, fit_joint12$value, joint12_part2_rows),
    fit_metrics("part2_source_cross_from_part1", fit_part2_cross$value, fit_part2_cross$elapsed, fit_joint12$value, joint12_part2_rows),
    fit_metrics("part2_theta_evidence_from_part1", fit_part2_theta$value, fit_part2_theta$elapsed, fit_joint12$value, joint12_part2_rows),
    anchor_metrics("joint12_anchor", fit_joint12$value, fit_joint12$elapsed)
)

metrics_123 <- rbind(
    fit_metrics("part3_default", fit_part3_default$value, fit_part3_default$elapsed, fit_joint123$value, joint123_part3_rows),
    fit_metrics("part3_source_cross_from_refit12", fit_part3_cross_from_refit12$value, fit_part3_cross_from_refit12$elapsed, fit_joint123$value, joint123_part3_rows),
    fit_metrics("part3_theta_evidence_from_refit12", fit_part3_theta_from_refit12$value, fit_part3_theta_from_refit12$elapsed, fit_joint123$value, joint123_part3_rows),
    fit_metrics("part3_theta_from_rolling_reextract_diagnostic", fit_part3_theta_from_reextract$value, fit_part3_theta_from_reextract$elapsed, fit_joint123$value, joint123_part3_rows),
    fit_metrics("part3_theta_from_composed_state", fit_part3_theta_from_composed$value, fit_part3_theta_from_composed$elapsed, fit_joint123$value, joint123_part3_rows),
    anchor_metrics("joint123_anchor", fit_joint123$value, fit_joint123$elapsed)
)

cat(sprintf("\nInitial part1 fit time: %.2fs\n", fit_part1$elapsed))

cat("\nPart 1 -> Part 2 update, compared with joint(1,2):\n")
print(metrics_12, row.names = FALSE, digits = 6)

cat("\nRefit joint(1,2) -> Part 3 update, compared with joint(1,2,3):\n")
print(metrics_123, row.names = FALSE, digits = 6)

cat(sprintf(
    paste(
        "\nPart2 theta drift:",
        "default %.6f -> source-cross %.6f -> theta-evidence %.6f\n"
    ),
    metrics_12$abs_theta_to_joint[[1L]],
    metrics_12$abs_theta_to_joint[[2L]],
    metrics_12$abs_theta_to_joint[[3L]]
))
cat(sprintf(
    paste(
        "Part3 theta drift:",
        "default %.6f -> source-cross refit12 %.6f -> theta-evidence refit12 %.6f -> rolling-reextract diagnostic %.6f -> composed-state %.6f\n"
    ),
    metrics_123$abs_theta_to_joint[[1L]],
    metrics_123$abs_theta_to_joint[[2L]],
    metrics_123$abs_theta_to_joint[[3L]],
    metrics_123$abs_theta_to_joint[[4L]],
    metrics_123$abs_theta_to_joint[[5L]]
))
cat(sprintf(
    paste(
        "Part3 fitted max rel drift:",
        "default %.6f -> source-cross refit12 %.6f -> theta-evidence refit12 %.6f -> rolling-reextract diagnostic %.6f -> composed-state %.6f\n"
    ),
    metrics_123$fitted_new_max_rel_to_joint[[1L]],
    metrics_123$fitted_new_max_rel_to_joint[[2L]],
    metrics_123$fitted_new_max_rel_to_joint[[3L]],
    metrics_123$fitted_new_max_rel_to_joint[[4L]],
    metrics_123$fitted_new_max_rel_to_joint[[5L]]
))
cat(
    paste(
        "rolling-reextract diagnostic is intentionally not treated as supported composition:",
        "rusty_update_state() currently extracts likelihood evidence from the current fit data,",
        "not a composed state that carries prior compressed evidence forward.",
        "The composed-state row uses rusty_compose_update_state() to carry that previous evidence forward.\n"
    )
)

invisible(list(part12 = metrics_12, part123 = metrics_123))
