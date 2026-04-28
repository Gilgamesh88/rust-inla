source(file.path(getwd(), "tools", "load_worktree_package.R"), local = TRUE)
load_rustyinla_for_benchmarks(getwd())

make_data <- function(groups, levels, n_per_group, seed) {
    set.seed(seed)
    group <- rep(groups, each = n_per_group)
    x <- stats::rnorm(length(group))
    effects <- c("A" = -0.35, "B" = 0.25, "C" = 0.75)
    eta <- -0.15 + 0.4 * x + unname(effects[as.character(group)])
    data.frame(
        y = stats::rpois(length(group), lambda = exp(eta)),
        x = x,
        group = factor(group, levels = levels)
    )
}

named_column <- function(df, column) {
    stats::setNames(as.numeric(df[[column]]), rownames(df))
}

assert_close <- function(lhs, rhs, tolerance, label) {
    if (!isTRUE(all.equal(as.numeric(lhs), as.numeric(rhs), tolerance = tolerance))) {
        stop(sprintf("%s did not match within tolerance %.3g.", label, tolerance), call. = FALSE)
    }
}

old_data <- make_data(
    groups = c("A", "B"),
    levels = c("A", "B"),
    n_per_group = 80L,
    seed = 9401L
)
new_data <- make_data(
    groups = c("A", "B", "C"),
    levels = c("A", "B", "C"),
    n_per_group = 25L,
    seed = 9402L
)

formula <- y ~ 1 + x + f(group, model = "iid")
old_fit <- rusty_inla(formula, data = old_data, family = "poisson")
state <- rusty_update_state(old_fit)
theta_evidence <- state$theta_evidence

if (is.null(theta_evidence)) {
    stop("rusty_update_state() did not include theta_evidence.", call. = FALSE)
}
if (!identical(theta_evidence$version, 2L)) {
    stop("Unexpected theta_evidence version.", call. = FALSE)
}
if (!identical(theta_evidence$strategy, "ccd_support_modes")) {
    stop("Unexpected theta_evidence strategy.", call. = FALSE)
}
if (!identical(theta_evidence$solver_status, "not_integrated")) {
    stop("Theta evidence should be extraction-only in Phase 8C.", call. = FALSE)
}
if (as.integer(theta_evidence$n_support) <= 1L) {
    stop("Phase 8C should expose more than one CCD support point when CCD is available.", call. = FALSE)
}
if (!isTRUE(all.equal(sum(theta_evidence$weights), 1.0, tolerance = 1e-12))) {
    stop("Theta evidence weights must be normalized.", call. = FALSE)
}
if (!all(is.finite(theta_evidence$log_constants))) {
    stop("Theta evidence log constant should be finite for Poisson.", call. = FALSE)
}

n_fixed <- length(state$fixed$names)
n_iid <- length(state$iid$levels)
n_support <- as.integer(theta_evidence$n_support)
if (!identical(dim(theta_evidence$theta), c(n_support, length(state$theta_mode)))) {
    stop("Theta evidence support matrix has unexpected dimensions.", call. = FALSE)
}
if (!isTRUE(all.equal(as.numeric(theta_evidence$theta[1L, ]), as.numeric(state$theta_mode), tolerance = 1e-12))) {
    stop("First theta evidence support point should be the source theta mode.", call. = FALSE)
}
if (!identical(dim(theta_evidence$H_beta_beta), c(n_fixed, n_fixed, n_support))) {
    stop("Theta evidence fixed precision array has unexpected dimensions.", call. = FALSE)
}
if (!identical(dim(theta_evidence$h_beta), c(n_support, n_fixed))) {
    stop("Theta evidence fixed linear matrix has unexpected dimensions.", call. = FALSE)
}
if (!identical(dim(theta_evidence$H_u_u_diag), c(n_support, n_iid))) {
    stop("Theta evidence iid precision matrix has unexpected dimensions.", call. = FALSE)
}
if (!identical(dim(theta_evidence$h_u), c(n_support, n_iid))) {
    stop("Theta evidence iid linear matrix has unexpected dimensions.", call. = FALSE)
}
if (!identical(dim(theta_evidence$H_u_beta), c(n_iid, n_fixed, n_support))) {
    stop("Theta evidence cross precision array has unexpected dimensions.", call. = FALSE)
}

assert_close(theta_evidence$H_beta_beta[, , 1L], state$fixed$evidence_precision, 1e-5, "H_beta_beta")
assert_close(theta_evidence$h_beta[1L, ], state$fixed$evidence_linear, 1e-5, "h_beta")
assert_close(theta_evidence$H_u_u_diag[1L, ], state$iid$evidence_precision_diag, 1e-5, "H_u_u_diag")
assert_close(theta_evidence$h_u[1L, ], state$iid$evidence_linear, 1e-5, "h_u")
assert_close(theta_evidence$H_u_beta[, , 1L], state$iid_fixed_cross_precision, 1e-5, "H_u_beta")

if (length(unique(round(as.numeric(theta_evidence$theta[, 1L]), 10))) <= 1L) {
    stop("Theta evidence support should contain non-identical theta values.", call. = FALSE)
}

fit_with_theta_evidence <- rusty_inla(
    formula,
    data = new_data,
    family = "poisson",
    control.update = list(
        state = state,
        mode = "fixed_iid_cross_gaussian_evidence"
    )
)

legacy_state <- state
legacy_state$theta_evidence <- NULL
legacy_state$semantics$theta_evidence_policy <- NULL
fit_without_theta_evidence <- rusty_inla(
    formula,
    data = new_data,
    family = "poisson",
    control.update = list(
        state = legacy_state,
        mode = "fixed_iid_cross_gaussian_evidence"
    )
)

assert_close(
    named_column(fit_with_theta_evidence$summary.fixed, "mean"),
    named_column(fit_without_theta_evidence$summary.fixed, "mean"),
    1e-10,
    "fixed means"
)
assert_close(
    named_column(fit_with_theta_evidence$summary.random$group, "mean"),
    named_column(fit_without_theta_evidence$summary.random$group, "mean"),
    1e-10,
    "random means"
)
assert_close(
    fit_with_theta_evidence$summary.fitted.values$mean,
    fit_without_theta_evidence$summary.fitted.values$mean,
    1e-10,
    "fitted means"
)

metadata <- fit_with_theta_evidence$posterior_state_used
if (!identical(metadata$theta_evidence_strategy, "ccd_support_modes") ||
    as.integer(metadata$theta_evidence_support_points) <= 1L ||
    !identical(metadata$theta_evidence_solver_status, "not_integrated")) {
    stop("Fit metadata did not preserve theta evidence extraction status.", call. = FALSE)
}

cat("posterior_state_theta_evidence_shape: PASS\n")
