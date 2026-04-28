source(file.path(getwd(), "tools", "load_worktree_package.R"), local = TRUE)
load_rustyinla_for_benchmarks(getwd())

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
    max(abs(lhs[seq_len(n)] - rhs[seq_len(n)]) / pmax(1.0, abs(rhs[seq_len(n)])))
}

make_batch <- function(groups, levels, n_per_group, seed, shift = 0.0) {
    set.seed(seed)
    group <- rep(groups, each = n_per_group)
    x <- stats::rnorm(length(group))
    effects <- c("1" = -0.75, "2" = 0.35, "3" = 0.8, "4" = -0.25, "5" = 1.2)
    eta <- -0.55 + shift + 0.45 * x + unname(effects[as.character(group)])
    data.frame(
        y = stats::rpois(length(group), lambda = exp(eta)),
        x = x,
        group = factor(group, levels = levels)
    )
}

old_data <- make_batch(groups = 1:4, levels = 1:4, n_per_group = 90L, seed = 9201L)
new_data <- make_batch(groups = 1:5, levels = 1:5, n_per_group = 24L, seed = 9202L, shift = 0.08)
joint_data <- rbind(
    transform(old_data, group = factor(as.character(group), levels = 1:5)),
    new_data
)
formula <- y ~ 1 + x + f(group, model = "iid")

old <- timed_fit(rusty_inla(formula, data = old_data, family = "poisson"))
state <- rusty_update_state(old$value)
new_default <- timed_fit(rusty_inla(formula, data = new_data, family = "poisson"))
new_cross <- timed_fit(rusty_inla(
    formula,
    data = new_data,
    family = "poisson",
    control.update = list(
        state = state,
        mode = "fixed_iid_cross_gaussian_evidence"
    )
))
new_theta <- timed_fit(rusty_inla(
    formula,
    data = new_data,
    family = "poisson",
    control.update = list(
        state = state,
        mode = "fixed_iid_cross_theta_evidence"
    )
))
joint <- timed_fit(rusty_inla(formula, data = joint_data, family = "poisson"))

joint_new_rows <- seq.int(nrow(old_data) + 1L, nrow(joint_data))
theta_joint <- as.numeric(joint$value$summary.hyperpar$mean[[1L]])
theta_default <- as.numeric(new_default$value$summary.hyperpar$mean[[1L]])
theta_cross <- as.numeric(new_cross$value$summary.hyperpar$mean[[1L]])
theta_dynamic <- as.numeric(new_theta$value$summary.hyperpar$mean[[1L]])

fixed_dynamic <- named_column(new_theta$value$summary.fixed, "mean")
fixed_joint <- named_column(joint$value$summary.fixed, "mean")
random_dynamic <- named_column(new_theta$value$summary.random$group, "mean")
random_joint <- named_column(joint$value$summary.random$group, "mean")
fixed_sd_dynamic <- named_column(new_theta$value$summary.fixed, "sd")
fixed_sd_joint <- named_column(joint$value$summary.fixed, "sd")
random_sd_dynamic <- named_column(new_theta$value$summary.random$group, "sd")
random_sd_joint <- named_column(joint$value$summary.random$group, "sd")

theta_default_diff <- abs(theta_default - theta_joint)
theta_cross_diff <- abs(theta_cross - theta_joint)
theta_dynamic_diff <- abs(theta_dynamic - theta_joint)
fixed_dynamic_diff <- max_abs_named_diff(fixed_dynamic, fixed_joint)
random_dynamic_diff <- max_abs_named_diff(random_dynamic, random_joint)
fitted_dynamic_diff <- max_fitted_rel_diff(new_theta$value, joint$value, joint_new_rows)
fitted_default_diff <- max_fitted_rel_diff(new_default$value, joint$value, joint_new_rows)
fixed_sd_dynamic_min_ratio <- min_named_ratio(fixed_sd_dynamic, fixed_sd_joint)
random_sd_dynamic_min_ratio <- min_named_ratio(random_sd_dynamic, random_sd_joint)

cat(sprintf(
    paste(
        "posterior_state_theta_objective:",
        "old %.3fs, new_default %.3fs, source_cross %.3fs, theta_dynamic %.3fs, joint %.3fs\n"
    ),
    old$elapsed,
    new_default$elapsed,
    new_cross$elapsed,
    new_theta$elapsed,
    joint$elapsed
))
cat(sprintf(
    paste(
        "theta drift vs joint:",
        "default %.6f, source_cross %.6f, theta_dynamic %.6f\n"
    ),
    theta_default_diff,
    theta_cross_diff,
    theta_dynamic_diff
))
cat(sprintf(
    paste(
        "theta_dynamic accuracy vs joint:",
        "fixed %.6f, random %.6f, fitted %.6f;",
        "sd ratios fixed %.6f, random %.6f\n"
    ),
    fixed_dynamic_diff,
    random_dynamic_diff,
    fitted_dynamic_diff,
    fixed_sd_dynamic_min_ratio,
    random_sd_dynamic_min_ratio
))

metadata <- new_theta$value$posterior_state_used
if (is.null(metadata) ||
    !identical(metadata$mode, "fixed_iid_cross_theta_evidence") ||
    !identical(metadata$theta_evidence_solver_status, "linear_1d_integrated")) {
    stop("Theta-dependent evidence fit did not record integrated theta evidence metadata.", call. = FALSE)
}
if (as.integer(metadata$theta_evidence_support_points) <= 1L) {
    stop("Theta-dependent evidence fit did not receive multi-point theta evidence.", call. = FALSE)
}
if (!("5" %in% metadata$born_iid_levels)) {
    stop("Born iid level was not preserved by theta-dependent evidence expansion.", call. = FALSE)
}
if (theta_dynamic_diff >= theta_default_diff) {
    stop("Theta-dependent evidence did not improve theta drift versus new-data-only.", call. = FALSE)
}
if (fitted_dynamic_diff >= fitted_default_diff) {
    stop("Theta-dependent evidence did not improve fitted values versus new-data-only.", call. = FALSE)
}
if (fixed_sd_dynamic_min_ratio < 0.25 || random_sd_dynamic_min_ratio < 0.25) {
    stop("Theta-dependent evidence narrowed posterior SDs too aggressively versus joint refit.", call. = FALSE)
}

cat("posterior_state_theta_objective: PASS\n")
