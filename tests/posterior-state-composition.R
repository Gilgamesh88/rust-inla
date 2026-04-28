source(file.path(getwd(), "tools", "load_worktree_package.R"), local = TRUE)
load_rustyinla_for_benchmarks(getwd())

named_column <- function(df, column) {
    if (is.null(df) || nrow(df) == 0L || !(column %in% names(df))) {
        return(stats::setNames(numeric(), character()))
    }
    stats::setNames(as.numeric(df[[column]]), rownames(df))
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

part1 <- make_batch(groups = 1:4, levels = 1:5, n_per_group = 75L, seed = 9301L)
part2 <- make_batch(groups = 1:5, levels = 1:5, n_per_group = 32L, seed = 9302L, shift = 0.04)
part3 <- make_batch(groups = 1:5, levels = 1:5, n_per_group = 24L, seed = 9303L, shift = 0.08)
joint12 <- rbind(part1, part2)
joint123 <- rbind(joint12, part3)
joint12$group <- factor(joint12$group, levels = 1:5)
joint123$group <- factor(joint123$group, levels = 1:5)

formula <- y ~ 1 + x + f(group, model = "iid")

fit1 <- rusty_inla(formula, data = part1, family = "poisson")
state1 <- rusty_update_state(fit1)
fit2 <- rusty_inla(
    formula,
    data = part2,
    family = "poisson",
    control.update = list(
        state = state1,
        mode = "fixed_iid_cross_theta_evidence"
    )
)
state_reextract <- rusty_update_state(fit2)
state_composed <- rusty_compose_update_state(state1, fit2)

fit3_reextract <- rusty_inla(
    formula,
    data = part3,
    family = "poisson",
    control.update = list(
        state = state_reextract,
        mode = "fixed_iid_cross_theta_evidence"
    )
)
fit3_composed <- rusty_inla(
    formula,
    data = part3,
    family = "poisson",
    control.update = list(
        state = state_composed,
        mode = "fixed_iid_cross_theta_evidence"
    )
)
fit_joint123 <- rusty_inla(formula, data = joint123, family = "poisson")

joint123_part3_rows <- seq.int(nrow(joint12) + 1L, nrow(joint123))
theta_joint <- as.numeric(fit_joint123$summary.hyperpar$mean[[1L]])
theta_reextract <- as.numeric(fit3_reextract$summary.hyperpar$mean[[1L]])
theta_composed <- as.numeric(fit3_composed$summary.hyperpar$mean[[1L]])
theta_reextract_diff <- abs(theta_reextract - theta_joint)
theta_composed_diff <- abs(theta_composed - theta_joint)
fitted_reextract_diff <- max_fitted_rel_diff(fit3_reextract, fit_joint123, joint123_part3_rows)
fitted_composed_diff <- max_fitted_rel_diff(fit3_composed, fit_joint123, joint123_part3_rows)

cat(sprintf(
    paste(
        "posterior_state_composition:",
        "theta reextract %.6f -> composed %.6f;",
        "fitted reextract %.6f -> composed %.6f\n"
    ),
    theta_reextract_diff,
    theta_composed_diff,
    fitted_reextract_diff,
    fitted_composed_diff
))

if (!inherits(state_composed, "rusty_update_state") || !identical(as.integer(state_composed$version), 4L)) {
    stop("Composed state did not return a version-4 rusty_update_state object.", call. = FALSE)
}
if (!identical(state_composed$semantics$composition, "previous_compressed_evidence_plus_current_likelihood_evidence")) {
    stop("Composed state did not record composition semantics.", call. = FALSE)
}
if (as.integer(state_composed$source$n_obs) != nrow(part1) + nrow(part2)) {
    stop("Composed state did not preserve cumulative source observation count.", call. = FALSE)
}
if (theta_composed_diff >= theta_reextract_diff) {
    stop("Composed update state did not improve theta drift versus rolling re-extraction.", call. = FALSE)
}
if (fitted_composed_diff >= fitted_reextract_diff) {
    stop("Composed update state did not improve fitted drift versus rolling re-extraction.", call. = FALSE)
}

cat("posterior_state_composition: PASS\n")
