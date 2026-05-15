source(file.path(getwd(), "tools", "load_worktree_package.R"), local = TRUE)
load_rustyinla_for_benchmarks(getwd())

named_random <- function(fit, block) {
    df <- fit$summary.random[[block]]
    stats::setNames(as.numeric(df$mean), rownames(df))
}

max_fitted_rel_diff <- function(fit, joint_fit, joint_rows) {
    lhs <- as.numeric(fit$summary.fitted.values$mean)
    rhs <- as.numeric(joint_fit$summary.fitted.values$mean[joint_rows])
    max(abs(lhs - rhs) / pmax(1.0, abs(rhs)))
}

timed_fit <- function(expr) {
    value <- NULL
    elapsed <- system.time({
        value <- eval.parent(substitute(expr))
    })[["elapsed"]]
    list(value = value, elapsed = as.numeric(elapsed))
}

make_two_iid_batch <- function(n, g1_levels, g2_levels, all_g1, all_g2, seed, shift = 0.0) {
    set.seed(seed)
    g1 <- sample(g1_levels, n, replace = TRUE)
    g2 <- sample(g2_levels, n, replace = TRUE)
    x <- stats::rnorm(n)
    expo <- stats::runif(n, 0.2, 2.0)
    g1_eff <- c(A = -0.35, B = 0.15, C = 0.55)
    g2_eff <- c(U = -0.25, V = 0.2, W = 0.45, X = -0.5)
    eta <- -1.15 + shift + 0.35 * x + unname(g1_eff[g1]) + unname(g2_eff[g2])
    data.frame(
        y = stats::rpois(n, lambda = expo * exp(eta)),
        x = x,
        expo = expo,
        g1 = factor(g1, levels = all_g1),
        g2 = factor(g2, levels = all_g2)
    )
}

all_g1 <- c("A", "B", "C")
all_g2 <- c("U", "V", "W", "X")
old_data <- make_two_iid_batch(
    n = 260L,
    g1_levels = c("A", "B"),
    g2_levels = c("U", "V", "W"),
    all_g1 = c("A", "B"),
    all_g2 = c("U", "V", "W"),
    seed = 81401L
)
new_data <- make_two_iid_batch(
    n = 120L,
    g1_levels = all_g1,
    g2_levels = all_g2,
    all_g1 = all_g1,
    all_g2 = all_g2,
    seed = 81402L,
    shift = 0.04
)
joint_data <- rbind(old_data, new_data)
joint_data$g1 <- factor(joint_data$g1, levels = all_g1)
joint_data$g2 <- factor(joint_data$g2, levels = all_g2)

formula <- y ~ 1 + x + offset(log(expo)) + f(g1, model = "iid") + f(g2, model = "iid")

old_run <- timed_fit(rusty_inla(formula, data = old_data, family = "poisson"))
old_fit <- old_run$value
state_ccd_run <- timed_fit(rusty_update_state(old_fit, theta_support_expansion = "none"))
state_ccd <- state_ccd_run$value
state_run <- timed_fit(rusty_update_state(old_fit))
state <- state_run$value
if (length(state$iid_blocks) != 2L) {
    stop("Expected a two-iid update state.", call. = FALSE)
}
if (is.null(state$theta_evidence$support_expansion) ||
    as.integer(state$theta_evidence$support_expansion$added) <= 0L ||
    as.integer(state$theta_evidence$n_support) <= as.integer(state_ccd$theta_evidence$n_support)) {
    stop("Expected multi-iid update state to add expanded theta guard support.", call. = FALSE)
}
if (length(state$latent_pair_precision$x) == 0L) {
    stop("Two-iid update state should carry sparse iid-iid cross evidence.", call. = FALSE)
}
if (is.null(state$theta_evidence$H_u_u_sparse) ||
    length(state$theta_evidence$H_u_u_sparse$i) == 0L ||
    nrow(state$theta_evidence$H_u_u_sparse$x) != as.integer(state$theta_evidence$n_support)) {
    stop("Two-iid update state should carry theta-dependent sparse iid-iid evidence.", call. = FALSE)
}

updated_run <- timed_fit(rusty_inla(
    formula,
    data = new_data,
    family = "poisson",
    control.update = list(
        state = state,
        mode = "fixed_iid_cross_gaussian_evidence"
    )
))
updated_fit <- updated_run$value
theta_run <- timed_fit(rusty_inla(
    formula,
    data = new_data,
    family = "poisson",
    control.update = list(
        state = state,
        mode = "fixed_iid_cross_theta_evidence"
    )
))
theta_fit <- theta_run$value
theta_ccd_run <- timed_fit(rusty_inla(
    formula,
    data = new_data,
    family = "poisson",
    control.update = list(
        state = state_ccd,
        mode = "fixed_iid_cross_theta_evidence"
    )
))
theta_ccd_fit <- theta_ccd_run$value
joint_run <- timed_fit(rusty_inla(formula, data = joint_data, family = "poisson"))
joint_fit <- joint_run$value

metadata <- updated_fit$posterior_state_used
if (!identical(as.character(metadata$source_iid_covariate_names), c("g1", "g2"))) {
    stop("Multi-iid metadata did not preserve iid covariate names.", call. = FALSE)
}
if (!("C" %in% metadata$born_iid_levels_by_block$g1) ||
    !("X" %in% metadata$born_iid_levels_by_block$g2)) {
    stop("Born iid levels were not recorded by block.", call. = FALSE)
}
if (!grepl("sparse iid-iid cross edges", metadata$caveat, fixed = TRUE)) {
    stop("Multi-iid metadata caveat should mention sparse iid-iid cross edges.", call. = FALSE)
}
if (!identical(theta_fit$posterior_state_used$theta_evidence_solver_status, "guarded_shepard_nd_integrated")) {
    stop("Multi-iid theta evidence should report the guarded multidimensional interpolation path.", call. = FALSE)
}

joint_new_rows <- seq.int(nrow(old_data) + 1L, nrow(joint_data))
fitted_rel_diff <- max_fitted_rel_diff(updated_fit, joint_fit, joint_new_rows)
fitted_theta_rel_diff <- max_fitted_rel_diff(theta_fit, joint_fit, joint_new_rows)
theta_diff <- max(abs(
    as.numeric(updated_fit$summary.hyperpar$mean[seq_len(2L)]) -
        as.numeric(joint_fit$summary.hyperpar$mean[seq_len(2L)])
))
theta_dynamic_diff <- max(abs(
    as.numeric(theta_fit$summary.hyperpar$mean[seq_len(2L)]) -
        as.numeric(joint_fit$summary.hyperpar$mean[seq_len(2L)])
))
theta_ccd_diff <- max(abs(
    as.numeric(theta_ccd_fit$summary.hyperpar$mean[seq_len(2L)]) -
        as.numeric(joint_fit$summary.hyperpar$mean[seq_len(2L)])
))
fixed_diff <- max(abs(
    as.numeric(updated_fit$summary.fixed$mean) -
        as.numeric(joint_fit$summary.fixed$mean)
))
fixed_theta_diff <- max(abs(
    as.numeric(theta_fit$summary.fixed$mean) -
        as.numeric(joint_fit$summary.fixed$mean)
))
g1_common <- intersect(names(named_random(updated_fit, "g1")), names(named_random(joint_fit, "g1")))
g2_common <- intersect(names(named_random(updated_fit, "g2")), names(named_random(joint_fit, "g2")))
random_diff <- max(
    abs(named_random(updated_fit, "g1")[g1_common] - named_random(joint_fit, "g1")[g1_common]),
    abs(named_random(updated_fit, "g2")[g2_common] - named_random(joint_fit, "g2")[g2_common])
)
random_theta_diff <- max(
    abs(named_random(theta_fit, "g1")[g1_common] - named_random(joint_fit, "g1")[g1_common]),
    abs(named_random(theta_fit, "g2")[g2_common] - named_random(joint_fit, "g2")[g2_common])
)

cat(sprintf(
    paste(
        "posterior_state_multi_iid_evidence:",
        "old %.3fs; state_ccd %.3fs; state_guard %.3fs;",
        "support %d -> %d;",
        "source %.3fs; ccd_dynamic %.3fs; guard_dynamic %.3fs; joint %.3fs;",
        "source theta %.6f; ccd theta %.6f; guard theta %.6f;",
        "source fixed %.6f; dynamic fixed %.6f;",
        "source random %.6f; dynamic random %.6f;",
        "source fitted_rel %.6f; dynamic fitted_rel %.6f\n"
    ),
    old_run$elapsed,
    state_ccd_run$elapsed,
    state_run$elapsed,
    as.integer(state_ccd$theta_evidence$n_support),
    as.integer(state$theta_evidence$n_support),
    updated_run$elapsed,
    theta_ccd_run$elapsed,
    theta_run$elapsed,
    joint_run$elapsed,
    theta_diff,
    theta_ccd_diff,
    theta_dynamic_diff,
    fixed_diff,
    fixed_theta_diff,
    random_diff,
    random_theta_diff,
    fitted_rel_diff,
    fitted_theta_rel_diff
))

if (!is.finite(theta_diff) || !is.finite(theta_dynamic_diff)) {
    stop("Multi-iid update produced non-finite theta diagnostics.", call. = FALSE)
}
if (fitted_rel_diff > 0.08 || fixed_diff > 0.12 || random_diff > 0.20) {
    stop("Multi-iid update drift exceeded the focused synthetic tolerance.", call. = FALSE)
}
if (theta_dynamic_diff >= theta_ccd_diff || theta_dynamic_diff >= theta_diff) {
    stop("Expanded multi-iid theta evidence did not improve theta drift versus source/CCD evidence.", call. = FALSE)
}
if (fitted_theta_rel_diff > 0.08 || fixed_theta_diff > 0.12 || random_theta_diff > 0.20) {
    stop("Multi-iid theta evidence drift exceeded the focused synthetic tolerance.", call. = FALSE)
}

cat("posterior_state_multi_iid_evidence: PASS\n")
