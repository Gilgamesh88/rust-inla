source(file.path(getwd(), "tools", "load_worktree_package.R"), local = TRUE)
load_rustyinla_for_benchmarks(getwd())

if (!requireNamespace("CASdatasets", quietly = TRUE)) {
    stop("Package 'CASdatasets' is required for the VehBrand posterior-state experiment.", call. = FALSE)
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

data("freMTPL2freq", package = "CASdatasets")

df <- get("freMTPL2freq")
df <- df[is.finite(df$ClaimNb) & is.finite(df$Exposure) & df$Exposure > 0, ]
df <- df[order(df$IDpol), ]
df$VehBrand <- factor(df$VehBrand)

split_idx <- floor(2.0 * nrow(df) / 3.0)
old_data <- df[seq_len(split_idx), ]
new_data <- df[(split_idx + 1L):nrow(df), ]
old_data$VehBrand <- factor(old_data$VehBrand, levels = levels(df$VehBrand))
new_data$VehBrand <- factor(new_data$VehBrand, levels = levels(df$VehBrand))
joint_data <- rbind(old_data, new_data)
joint_data$VehBrand <- factor(joint_data$VehBrand, levels = levels(df$VehBrand))
born_levels <- setdiff(
    sort(unique(as.character(new_data$VehBrand))),
    sort(unique(as.character(old_data$VehBrand)))
)

formula <- ClaimNb ~ 1 + offset(log(Exposure)) + f(VehBrand, model = "iid")

cat("Phase 8 VehBrand posterior-state experiment\n")
cat("Split strategy: first 2/3 by IDpol as old data, final 1/3 as new data.\n")
cat("Caveat: freMTPL2freq has no explicit time variable here, so this is a pseudo-period split.\n")
cat(sprintf(
    "Rows: old %d, new %d, joint %d; VehBrand levels %d\n",
    nrow(old_data),
    nrow(new_data),
    nrow(joint_data),
    nlevels(df$VehBrand)
))
cat(sprintf(
    "Born VehBrand levels in this split: %s\n",
    if (length(born_levels) == 0L) "none" else paste(born_levels, collapse = ", ")
))

old <- timed_fit(rusty_inla(formula, data = old_data, family = "poisson"))
state <- rusty_posterior_state(old$value)
evidence_state <- rusty_update_state(old$value)
new_default <- timed_fit(rusty_inla(formula, data = new_data, family = "poisson"))
new_updated <- timed_fit(rusty_inla(
    formula,
    data = new_data,
    family = "poisson",
    control.update = list(
        posterior_state = state,
        mode = "iid_hyper_gaussian"
    )
))
new_evidence_diag <- timed_fit(rusty_inla(
    formula,
    data = new_data,
    family = "poisson",
    control.update = list(
        state = evidence_state,
        mode = "fixed_iid_gaussian_evidence"
    )
))
new_evidence_cross <- timed_fit(rusty_inla(
    formula,
    data = new_data,
    family = "poisson",
    control.update = list(
        state = evidence_state,
        mode = "fixed_iid_cross_gaussian_evidence"
    )
))

old_fixed <- named_column(old$value$summary.fixed, "mean")
old_random <- named_column(old$value$summary.random$VehBrand, "mean")
old_intercept <- unname(old_fixed[["(Intercept)"]])
if (!is.finite(old_intercept)) {
    stop("Old fit did not produce a finite intercept.", call. = FALSE)
}
old_random_offset <- old_random[as.character(new_data$VehBrand)]
old_random_offset[!is.finite(old_random_offset)] <- 0.0

new_data_mean_offset <- new_data
new_data_mean_offset$prior_fixed_offset <- log(new_data_mean_offset$Exposure) +
    old_intercept
new_data_mean_offset$prior_random_offset <- log(new_data_mean_offset$Exposure) +
    unname(old_random_offset)
new_data_mean_offset$prior_eta_offset <- log(new_data_mean_offset$Exposure) +
    old_intercept + unname(old_random_offset)
formula_fixed_mean_offset <- ClaimNb ~ 0 + offset(prior_fixed_offset) + f(VehBrand, model = "iid")
formula_random_mean_offset <- ClaimNb ~ 1 + offset(prior_random_offset) + f(VehBrand, model = "iid")
formula_full_mean_offset <- ClaimNb ~ 0 + offset(prior_eta_offset) + f(VehBrand, model = "iid")

new_fixed_mean_offset <- timed_fit(rusty_inla(
    formula_fixed_mean_offset,
    data = new_data_mean_offset,
    family = "poisson"
))
new_random_mean_offset <- timed_fit(rusty_inla(
    formula_random_mean_offset,
    data = new_data_mean_offset,
    family = "poisson"
))
new_state_random_mean_offset <- timed_fit(rusty_inla(
    formula_random_mean_offset,
    data = new_data_mean_offset,
    family = "poisson",
    control.update = list(
        posterior_state = state,
        mode = "iid_hyper_gaussian"
    )
))
new_full_mean_offset <- timed_fit(rusty_inla(
    formula_full_mean_offset,
    data = new_data_mean_offset,
    family = "poisson"
))
joint <- timed_fit(rusty_inla(formula, data = joint_data, family = "poisson"))

theta_state <- state$theta_prior_mean[[1L]]
theta_state_sd <- 1.0 / sqrt(state$theta_prior_precision[[1L]])
theta_old <- old$value$mode$theta[[1L]]
theta_default <- new_default$value$mode$theta[[1L]]
theta_updated <- new_updated$value$mode$theta[[1L]]
theta_evidence_diag <- new_evidence_diag$value$mode$theta[[1L]]
theta_evidence_cross <- new_evidence_cross$value$mode$theta[[1L]]
theta_fixed_mean_offset <- new_fixed_mean_offset$value$mode$theta[[1L]]
theta_random_mean_offset <- new_random_mean_offset$value$mode$theta[[1L]]
theta_state_random_mean_offset <- new_state_random_mean_offset$value$mode$theta[[1L]]
theta_full_mean_offset <- new_full_mean_offset$value$mode$theta[[1L]]
theta_joint <- joint$value$mode$theta[[1L]]

joint_new_rows <- seq.int(nrow(old_data) + 1L, nrow(joint_data))
fixed_joint <- named_column(joint$value$summary.fixed, "mean")
random_joint <- named_column(joint$value$summary.random$VehBrand, "mean")
fixed_joint_sd <- named_column(joint$value$summary.fixed, "sd")
random_joint_sd <- named_column(joint$value$summary.random$VehBrand, "sd")

effective_intercept <- function(fit, base_intercept = 0.0) {
    fit_fixed <- named_column(fit$summary.fixed, "mean")
    residual_intercept <- if ("(Intercept)" %in% names(fit_fixed)) {
        unname(fit_fixed[["(Intercept)"]])
    } else {
        0.0
    }
    base_intercept + residual_intercept
}

effective_random <- function(fit, base_random = NULL) {
    fit_random <- named_column(fit$summary.random$VehBrand, "mean")
    if (is.null(base_random)) {
        return(fit_random)
    }
    all_names <- union(names(base_random), names(fit_random))
    base <- stats::setNames(rep(0.0, length(all_names)), all_names)
    resid <- stats::setNames(rep(0.0, length(all_names)), all_names)
    base[names(base_random)] <- base_random
    resid[names(fit_random)] <- fit_random
    base + resid
}

joint_intercept <- unname(fixed_joint[["(Intercept)"]])

metrics <- data.frame(
    fit = c(
        "old",
        "new_default",
        "new_updated",
        "new_fixed_iid_diag_evidence",
        "new_fixed_iid_cross_evidence",
        "new_fixed_mean_offset",
        "new_random_mean_offset",
        "new_state_random_mean_offset",
        "new_full_mean_offset",
        "joint"
    ),
    time_sec = c(
        old$elapsed,
        new_default$elapsed,
        new_updated$elapsed,
        new_evidence_diag$elapsed,
        new_evidence_cross$elapsed,
        new_fixed_mean_offset$elapsed,
        new_random_mean_offset$elapsed,
        new_state_random_mean_offset$elapsed,
        new_full_mean_offset$elapsed,
        joint$elapsed
    ),
    theta_log_precision = c(
        theta_old,
        theta_default,
        theta_updated,
        theta_evidence_diag,
        theta_evidence_cross,
        theta_fixed_mean_offset,
        theta_random_mean_offset,
        theta_state_random_mean_offset,
        theta_full_mean_offset,
        theta_joint
    ),
    tau_precision = exp(c(
        theta_old,
        theta_default,
        theta_updated,
        theta_evidence_diag,
        theta_evidence_cross,
        theta_fixed_mean_offset,
        theta_random_mean_offset,
        theta_state_random_mean_offset,
        theta_full_mean_offset,
        theta_joint
    )),
    abs_theta_to_joint = c(
        abs(theta_old - theta_joint),
        abs(theta_default - theta_joint),
        abs(theta_updated - theta_joint),
        abs(theta_evidence_diag - theta_joint),
        abs(theta_evidence_cross - theta_joint),
        abs(theta_fixed_mean_offset - theta_joint),
        abs(theta_random_mean_offset - theta_joint),
        abs(theta_state_random_mean_offset - theta_joint),
        abs(theta_full_mean_offset - theta_joint),
        0.0
    ),
    effective_intercept_abs_to_joint = c(
        abs(effective_intercept(old$value) - joint_intercept),
        abs(effective_intercept(new_default$value) - joint_intercept),
        abs(effective_intercept(new_updated$value) - joint_intercept),
        abs(effective_intercept(new_evidence_diag$value) - joint_intercept),
        abs(effective_intercept(new_evidence_cross$value) - joint_intercept),
        abs(effective_intercept(new_fixed_mean_offset$value, old_intercept) - joint_intercept),
        abs(effective_intercept(new_random_mean_offset$value) - joint_intercept),
        abs(effective_intercept(new_state_random_mean_offset$value) - joint_intercept),
        abs(effective_intercept(new_full_mean_offset$value, old_intercept) - joint_intercept),
        0.0
    ),
    effective_random_mean_max_abs_to_joint = c(
        max_abs_named_diff(effective_random(old$value), random_joint),
        max_abs_named_diff(effective_random(new_default$value), random_joint),
        max_abs_named_diff(effective_random(new_updated$value), random_joint),
        max_abs_named_diff(effective_random(new_evidence_diag$value), random_joint),
        max_abs_named_diff(effective_random(new_evidence_cross$value), random_joint),
        max_abs_named_diff(effective_random(new_fixed_mean_offset$value), random_joint),
        max_abs_named_diff(effective_random(new_random_mean_offset$value, old_random), random_joint),
        max_abs_named_diff(effective_random(new_state_random_mean_offset$value, old_random), random_joint),
        max_abs_named_diff(effective_random(new_full_mean_offset$value, old_random), random_joint),
        0.0
    ),
    fixed_sd_min_ratio_to_joint = c(
        min_named_ratio(named_column(old$value$summary.fixed, "sd"), fixed_joint_sd),
        min_named_ratio(named_column(new_default$value$summary.fixed, "sd"), fixed_joint_sd),
        min_named_ratio(named_column(new_updated$value$summary.fixed, "sd"), fixed_joint_sd),
        min_named_ratio(named_column(new_evidence_diag$value$summary.fixed, "sd"), fixed_joint_sd),
        min_named_ratio(named_column(new_evidence_cross$value$summary.fixed, "sd"), fixed_joint_sd),
        min_named_ratio(named_column(new_fixed_mean_offset$value$summary.fixed, "sd"), fixed_joint_sd),
        min_named_ratio(named_column(new_random_mean_offset$value$summary.fixed, "sd"), fixed_joint_sd),
        min_named_ratio(named_column(new_state_random_mean_offset$value$summary.fixed, "sd"), fixed_joint_sd),
        min_named_ratio(named_column(new_full_mean_offset$value$summary.fixed, "sd"), fixed_joint_sd),
        1.0
    ),
    random_sd_min_ratio_to_joint = c(
        min_named_ratio(named_column(old$value$summary.random$VehBrand, "sd"), random_joint_sd),
        min_named_ratio(named_column(new_default$value$summary.random$VehBrand, "sd"), random_joint_sd),
        min_named_ratio(named_column(new_updated$value$summary.random$VehBrand, "sd"), random_joint_sd),
        min_named_ratio(named_column(new_evidence_diag$value$summary.random$VehBrand, "sd"), random_joint_sd),
        min_named_ratio(named_column(new_evidence_cross$value$summary.random$VehBrand, "sd"), random_joint_sd),
        min_named_ratio(named_column(new_fixed_mean_offset$value$summary.random$VehBrand, "sd"), random_joint_sd),
        min_named_ratio(named_column(new_random_mean_offset$value$summary.random$VehBrand, "sd"), random_joint_sd),
        min_named_ratio(named_column(new_state_random_mean_offset$value$summary.random$VehBrand, "sd"), random_joint_sd),
        min_named_ratio(named_column(new_full_mean_offset$value$summary.random$VehBrand, "sd"), random_joint_sd),
        1.0
    ),
    fitted_new_max_rel_to_joint = c(
        NA_real_,
        max_fitted_rel_diff(new_default$value, joint$value, joint_new_rows),
        max_fitted_rel_diff(new_updated$value, joint$value, joint_new_rows),
        max_fitted_rel_diff(new_evidence_diag$value, joint$value, joint_new_rows),
        max_fitted_rel_diff(new_evidence_cross$value, joint$value, joint_new_rows),
        max_fitted_rel_diff(new_fixed_mean_offset$value, joint$value, joint_new_rows),
        max_fitted_rel_diff(new_random_mean_offset$value, joint$value, joint_new_rows),
        max_fitted_rel_diff(new_state_random_mean_offset$value, joint$value, joint_new_rows),
        max_fitted_rel_diff(new_full_mean_offset$value, joint$value, joint_new_rows),
        0.0
    ),
    check.names = FALSE
)

cat(sprintf(
    "Posterior-state iid log-precision prior: mean %.6f, sd %.6f\n",
    theta_state,
    theta_state_sd
))
print(metrics, row.names = FALSE, digits = 6)

cat(sprintf(
    paste(
        "Theta improvement vs joint:",
        "default %.6f -> updated %.6f\n"
    ),
    abs(theta_default - theta_joint),
    abs(theta_updated - theta_joint)
))
cat(sprintf(
    paste(
        "Fitted new-row max rel diff vs joint:",
        "default %.6f -> theta-updated %.6f -> diag-evidence %.6f -> cross-evidence %.6f -> fixed-mean %.6f ->",
        "random-mean %.6f -> state+random-mean %.6f -> full-mean %.6f\n"
    ),
    metrics$fitted_new_max_rel_to_joint[[2L]],
    metrics$fitted_new_max_rel_to_joint[[3L]],
    metrics$fitted_new_max_rel_to_joint[[4L]],
    metrics$fitted_new_max_rel_to_joint[[5L]],
    metrics$fitted_new_max_rel_to_joint[[6L]],
    metrics$fitted_new_max_rel_to_joint[[7L]],
    metrics$fitted_new_max_rel_to_joint[[8L]],
    metrics$fitted_new_max_rel_to_joint[[9L]]
))
cat(
    paste(
        "fixed+iid-cross-evidence is the reusable Gaussian evidence experiment.",
        "Mean-offset rows are diagnostics only.",
        "fixed-mean carries the old intercept as a point offset;",
        "random-mean carries old VehBrand means as point offsets;",
        "full-mean carries both and estimates residual iid corrections.",
        "None of these rows is yet a full Bayesian latent-state prior with uncertainty.\n"
    )
)

invisible(metrics)
