source(file.path(getwd(), "tools", "load_worktree_package.R"), local = TRUE)
load_rustyinla_for_benchmarks(getwd())

timed_fit <- function(expr) {
    elapsed <- system.time(value <- eval.parent(substitute(expr)))
    list(value = value, elapsed = unname(elapsed[["elapsed"]]))
}

make_iid_batch <- function(group_effects, n_per_group, seed) {
    set.seed(seed)
    n_group <- length(group_effects)
    group_id <- rep(seq_len(n_group), each = n_per_group)
    n <- length(group_id)
    x <- stats::rnorm(n)
    eta <- -0.35 + 0.25 * x + group_effects[group_id]
    data.frame(
        y = stats::rpois(n, lambda = exp(eta)),
        x = x,
        group = factor(group_id, levels = seq_len(n_group))
    )
}

set.seed(8801L)
n_group <- 8L
group_effects <- stats::rnorm(n_group, sd = 1 / sqrt(exp(0.75)))
old_data <- make_iid_batch(group_effects, n_per_group = 35L, seed = 8802L)
new_data <- make_iid_batch(group_effects, n_per_group = 6L, seed = 8803L)
joint_data <- rbind(old_data, new_data)
formula <- y ~ 1 + x + f(group, model = "iid")

old <- timed_fit(rusty_inla(formula, data = old_data, family = "poisson"))
state <- rusty_posterior_state(old$value)
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
joint <- timed_fit(rusty_inla(formula, data = joint_data, family = "poisson"))

theta_prior_mean <- state$theta_prior_mean[[1L]]
theta_default <- new_default$value$mode$theta[[1L]]
theta_updated <- new_updated$value$mode$theta[[1L]]
theta_joint <- joint$value$mode$theta[[1L]]

dist_default_to_state <- abs(theta_default - theta_prior_mean)
dist_updated_to_state <- abs(theta_updated - theta_prior_mean)
dist_default_to_joint <- abs(theta_default - theta_joint)
dist_updated_to_joint <- abs(theta_updated - theta_joint)

cat(sprintf(
    paste(
        "posterior_state_iid_experimental:",
        "old %.3fs, new_default %.3fs, new_updated %.3fs, joint %.3fs\n"
    ),
    old$elapsed,
    new_default$elapsed,
    new_updated$elapsed,
    joint$elapsed
))
cat(sprintf(
    paste(
        "theta log-precision:",
        "state_mean %.6f, new_default %.6f, new_updated %.6f, joint %.6f\n"
    ),
    theta_prior_mean,
    theta_default,
    theta_updated,
    theta_joint
))
cat(sprintf(
    paste(
        "theta distances:",
        "default_to_state %.6f, updated_to_state %.6f,",
        "default_to_joint %.6f, updated_to_joint %.6f\n"
    ),
    dist_default_to_state,
    dist_updated_to_state,
    dist_default_to_joint,
    dist_updated_to_joint
))

if (!inherits(state, "rusty_posterior_state")) {
    stop("rusty_posterior_state() did not return the expected S3 class.", call. = FALSE)
}
if (is.null(new_updated$value$posterior_state_used)) {
    stop("Updated fit did not record posterior_state_used metadata.", call. = FALSE)
}
if (!is.finite(theta_prior_mean) || !is.finite(state$theta_prior_precision[[1L]])) {
    stop("Posterior state contains non-finite iid prior parameters.", call. = FALSE)
}
if (state$theta_prior_precision[[1L]] <= 0.0) {
    stop("Posterior state iid prior precision must be positive.", call. = FALSE)
}
if (abs(theta_updated - theta_default) <= 1e-5) {
    stop("Experimental posterior-state prior did not affect the iid hyperparameter mode.", call. = FALSE)
}
if (dist_updated_to_state >= dist_default_to_state) {
    stop("Experimental iid posterior prior did not pull the update toward the previous state.", call. = FALSE)
}

make_born_level_data <- function() {
    set.seed(8810L)
    old_group <- rep(c("A", "B"), each = 120L)
    old_eta <- -0.8 + ifelse(old_group == "B", 0.9, -0.9)
    old <- data.frame(
        y = stats::rpois(length(old_group), lambda = exp(old_eta)),
        group = factor(old_group, levels = c("A", "B", "C"))
    )

    new_group <- rep(c("A", "B", "C"), each = 90L)
    new_eta <- -0.8 + ifelse(new_group == "C", 1.8, ifelse(new_group == "B", 0.9, -0.9))
    new <- data.frame(
        y = stats::rpois(length(new_group), lambda = exp(new_eta)),
        group = factor(new_group, levels = c("A", "B", "C"))
    )

    list(old = old, new = new)
}

born_data <- make_born_level_data()
born_formula <- y ~ 1 + f(group, model = "iid")
born_old <- rusty_inla(born_formula, data = born_data$old, family = "poisson")
born_state <- rusty_posterior_state(born_old)
born_updated <- rusty_inla(
    born_formula,
    data = born_data$new,
    family = "poisson",
    control.update = list(
        posterior_state = born_state,
        mode = "iid_hyper_gaussian"
    )
)
born_random <- born_updated$summary.random$group
born_c_mean <- born_random["C", "mean"]

cat(sprintf(
    "born iid level diagnostic: level C mean %.6f, theta %.6f\n",
    born_c_mean,
    born_updated$mode$theta[[1L]]
))

if (!is.finite(born_c_mean)) {
    stop("Born iid level did not produce a finite posterior mean.", call. = FALSE)
}
if (abs(born_c_mean) <= 0.05) {
    stop("Born iid level remained effectively pinned at zero instead of updating from data.", call. = FALSE)
}

cat("posterior_state_iid_experimental: PASS\n")
