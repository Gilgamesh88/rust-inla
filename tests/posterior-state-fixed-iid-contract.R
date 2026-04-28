source(file.path(getwd(), "tools", "load_worktree_package.R"), local = TRUE)
load_rustyinla_for_benchmarks(getwd())

expect_error <- function(expr, pattern, label) {
    message <- tryCatch(
        {
            force(expr)
            NULL
        },
        error = function(e) conditionMessage(e)
    )
    if (is.null(message)) {
        stop(sprintf("%s: expected an error.", label), call. = FALSE)
    }
    if (!grepl(pattern, message, fixed = TRUE)) {
        stop(
            sprintf(
                "%s: expected error containing '%s', got '%s'.",
                label,
                pattern,
                message
            ),
            call. = FALSE
        )
    }
}

make_contract_data <- function(groups, levels, n_per_group, seed) {
    set.seed(seed)
    group <- rep(groups, each = n_per_group)
    x <- stats::rnorm(length(group))
    effects <- c("A" = -0.4, "B" = 0.35, "C" = 0.9)
    eta <- -0.25 + 0.3 * x + unname(effects[as.character(group)])
    data.frame(
        y = stats::rpois(length(group), lambda = exp(eta)),
        x = x,
        z = x + 0.1 * stats::rnorm(length(group)),
        group = factor(group, levels = levels),
        group2 = factor(group, levels = levels)
    )
}

old_data <- make_contract_data(
    groups = c("A", "B"),
    levels = c("A", "B"),
    n_per_group = 70L,
    seed = 9301L
)
new_data <- make_contract_data(
    groups = c("A", "B", "C"),
    levels = c("A", "B", "C"),
    n_per_group = 18L,
    seed = 9302L
)

formula <- y ~ 1 + x + f(group, model = "iid")
old_fit <- rusty_inla(formula, data = old_data, family = "poisson")
state <- rusty_update_state(old_fit)

if (!inherits(state, "rusty_update_state")) {
    stop("rusty_update_state() did not return the expected class.", call. = FALSE)
}
if (!identical(state$scope, "fixed_iid_gaussian")) {
    stop("Unexpected update-state scope.", call. = FALSE)
}
if (!identical(state$approximation, "fixed_iid_cross_gaussian_evidence")) {
    stop("Unexpected update-state approximation.", call. = FALSE)
}
if (!identical(state$version, 3L)) {
    stop("Unexpected update-state version.", call. = FALSE)
}
if (!identical(state$semantics$kind, "old_data_likelihood_evidence")) {
    stop("Update state must declare old-data likelihood evidence semantics.", call. = FALSE)
}
if (!identical(state$semantics$prior_policy, "original_model_priors_remain_active_in_update")) {
    stop("Update state must declare that original priors remain active.", call. = FALSE)
}
if (!identical(state$semantics$posterior_reuse, "not_posterior_as_prior")) {
    stop("Update state must not be labeled as posterior-as-prior reuse.", call. = FALSE)
}
if (!identical(state$semantics$theta_evidence_policy, "ccd_support_modes_not_integrated")) {
    stop("Update state must declare the current CCD-support theta evidence policy.", call. = FALSE)
}
if (!identical(state$source$n_obs, as.integer(nrow(old_data)))) {
    stop("Update state source metadata did not record the old observation count.", call. = FALSE)
}
if (is.null(state$theta_evidence) ||
    !identical(state$theta_evidence$strategy, "ccd_support_modes") ||
    as.integer(state$theta_evidence$n_support) <= 1L) {
    stop("Update state is missing the Phase 8C CCD-support theta evidence container.", call. = FALSE)
}
if (!isTRUE(all.equal(as.numeric(state$theta_evidence$theta[1L, ]), as.numeric(state$theta_mode), tolerance = 1e-12))) {
    stop("Theta evidence support point does not match the source theta mode.", call. = FALSE)
}
if (!isTRUE(all.equal(state$theta_evidence$H_beta_beta[, , 1L], state$fixed$evidence_precision, tolerance = 1e-5))) {
    stop("Theta evidence fixed precision does not reproduce the source-mode fixed evidence.", call. = FALSE)
}
if (!isTRUE(all.equal(as.numeric(state$theta_evidence$h_beta[1L, ]), as.numeric(state$fixed$evidence_linear), tolerance = 1e-5))) {
    stop("Theta evidence fixed linear term does not reproduce the source-mode fixed evidence.", call. = FALSE)
}
if (!isTRUE(all.equal(as.numeric(state$theta_evidence$H_u_u_diag[1L, ]), as.numeric(state$iid$evidence_precision_diag), tolerance = 1e-5))) {
    stop("Theta evidence iid precision does not reproduce the source-mode iid evidence.", call. = FALSE)
}
if (!isTRUE(all.equal(as.numeric(state$theta_evidence$h_u[1L, ]), as.numeric(state$iid$evidence_linear), tolerance = 1e-5))) {
    stop("Theta evidence iid linear term does not reproduce the source-mode iid evidence.", call. = FALSE)
}
if (!isTRUE(all.equal(state$theta_evidence$H_u_beta[, , 1L], state$iid_fixed_cross_precision, tolerance = 1e-5))) {
    stop("Theta evidence cross precision does not reproduce the source-mode cross evidence.", call. = FALSE)
}
if (!all(is.finite(state$theta_evidence$log_constants))) {
    stop("Theta evidence log constants must be finite for the supported Poisson contract case.", call. = FALSE)
}
if (is.null(state$iid_fixed_cross_precision)) {
    stop("Cross evidence state is missing iid_fixed_cross_precision.", call. = FALSE)
}
if (nrow(state$iid_fixed_cross_precision) != length(state$iid$levels) ||
    ncol(state$iid_fixed_cross_precision) != length(state$fixed$names)) {
    stop("Cross evidence state dimensions do not match fixed/iid names.", call. = FALSE)
}

cross_fit <- rusty_inla(
    formula,
    data = new_data,
    family = "poisson",
    control.update = list(
        state = state,
        mode = "fixed_iid_cross_gaussian_evidence"
    )
)
metadata <- cross_fit$posterior_state_used
if (is.null(metadata)) {
    stop("Cross evidence fit did not record posterior_state_used metadata.", call. = FALSE)
}
if (!identical(metadata$mode, "fixed_iid_cross_gaussian_evidence")) {
    stop("Cross evidence metadata recorded the wrong mode.", call. = FALSE)
}
if (!identical(metadata$approximation, "fixed_iid_cross_gaussian_evidence")) {
    stop("Cross evidence metadata recorded the wrong approximation.", call. = FALSE)
}
if (!identical(metadata$state_version, 3L)) {
    stop("Cross evidence metadata recorded the wrong state version.", call. = FALSE)
}
if (!identical(metadata$evidence_semantics, "old_data_likelihood_evidence")) {
    stop("Cross evidence metadata did not preserve old-data evidence semantics.", call. = FALSE)
}
if (!identical(metadata$prior_policy, "original_model_priors_remain_active_in_update")) {
    stop("Cross evidence metadata did not preserve the prior policy.", call. = FALSE)
}
if (!identical(metadata$posterior_reuse, "not_posterior_as_prior")) {
    stop("Cross evidence metadata should explicitly reject posterior-as-prior semantics.", call. = FALSE)
}
if (!identical(metadata$source_n_obs, as.integer(nrow(old_data)))) {
    stop("Cross evidence metadata did not preserve the source observation count.", call. = FALSE)
}
if (!identical(metadata$theta_evidence_policy, "ccd_support_modes_not_integrated") ||
    !identical(metadata$theta_evidence_strategy, "ccd_support_modes") ||
    as.integer(metadata$theta_evidence_support_points) <= 1L ||
    !identical(metadata$theta_evidence_solver_status, "not_integrated")) {
    stop("Cross evidence metadata did not preserve Phase 8C theta evidence extraction metadata.", call. = FALSE)
}
if (!("C" %in% metadata$born_iid_levels)) {
    stop("Born iid level C was not recorded as zero old evidence.", call. = FALSE)
}
if (!grepl("cross block", metadata$caveat, fixed = TRUE)) {
    stop("Cross evidence metadata caveat does not mention the cross block.", call. = FALSE)
}

diag_fit <- rusty_inla(
    formula,
    data = new_data,
    family = "poisson",
    control.update = list(
        state = state,
        mode = "fixed_iid_gaussian_evidence"
    )
)
diag_metadata <- diag_fit$posterior_state_used
if (!grepl("omits the fixed-iid cross block", diag_metadata$caveat, fixed = TRUE)) {
    stop("Diagonal evidence metadata should remain explicitly diagnostic.", call. = FALSE)
}

bad_state <- state
bad_state$iid_fixed_cross_precision <- NULL
expect_error(
    rusty_inla(
        formula,
        data = new_data,
        family = "poisson",
        control.update = list(
            state = bad_state,
            mode = "fixed_iid_cross_gaussian_evidence"
        )
    ),
    "requires a state with iid_fixed_cross_precision",
    "missing cross evidence"
)

bad_state <- state
bad_state$semantics <- NULL
expect_error(
    rusty_inla(
        formula,
        data = new_data,
        family = "poisson",
        control.update = list(
            state = bad_state,
            mode = "fixed_iid_cross_gaussian_evidence"
        )
    ),
    "missing old-data evidence semantics",
    "missing state semantics"
)

bad_state <- state
bad_state$semantics$prior_policy <- "posterior_as_prior"
expect_error(
    rusty_inla(
        formula,
        data = new_data,
        family = "poisson",
        control.update = list(
            state = bad_state,
            mode = "fixed_iid_cross_gaussian_evidence"
        )
    ),
    "Unsupported update-state prior policy",
    "bad prior policy"
)

expect_error(
    rusty_inla(
        formula,
        data = new_data,
        family = "gaussian",
        control.update = list(
            state = state,
            mode = "fixed_iid_cross_gaussian_evidence"
        )
    ),
    "family mismatch",
    "changed family"
)

expect_error(
    rusty_inla(
        y ~ 1 + z + f(group, model = "iid"),
        data = new_data,
        family = "poisson",
        control.update = list(
            state = state,
            mode = "fixed_iid_cross_gaussian_evidence"
        )
    ),
    "fixed-effect columns mismatch",
    "changed fixed design"
)

expect_error(
    rusty_inla(
        y ~ 1 + x + f(group2, model = "iid"),
        data = new_data,
        family = "poisson",
        control.update = list(
            state = state,
            mode = "fixed_iid_cross_gaussian_evidence"
        )
    ),
    "iid covariate mismatch",
    "changed iid covariate"
)

expect_error(
    rusty_inla(
        y ~ 1 + x + f(group, model = "rw1"),
        data = new_data,
        family = "poisson",
        control.update = list(
            state = state,
            mode = "fixed_iid_cross_gaussian_evidence"
        )
    ),
    "latent model mismatch",
    "changed latent model"
)

cat("posterior_state_fixed_iid_contract: PASS\n")
