source(file.path(getwd(), "tools", "load_worktree_package.R"), local = TRUE)
load_rustyinla_for_benchmarks(getwd())

set.seed(9001L)
n_group <- 5L
n_per_group <- 8L
group <- factor(rep(seq_len(n_group), each = n_per_group))
x <- stats::rnorm(n_group * n_per_group)
eta <- -0.2 + 0.35 * x + stats::rnorm(n_group, sd = 0.3)[as.integer(group)]
data <- data.frame(
    y = stats::rpois(length(eta), lambda = exp(eta)),
    x = x,
    group = group
)
formula <- y ~ 1 + x + f(group, model = "iid")

fit <- rusty_inla(formula, data = data, family = "poisson")

stopifnot(is.list(fit$control.used))
stopifnot(is.list(fit$prior.used))
stopifnot(is.list(fit$theta.internal))
stopifnot(isTRUE(fit$theta.internal$internal_scale))
stopifnot(is.data.frame(fit$summary.hyperpar.internal))
stopifnot(nrow(fit$summary.hyperpar.internal) == fit$nhyper)
stopifnot(is.list(fit$accuracy.diagnostics))
stopifnot(identical(fit$accuracy.diagnostics$purpose, "pre_sequential_validation_accuracy_gate"))

theta <- fit$theta.internal$mode
replay <- rusty_inla(
    formula,
    data = data,
    family = "poisson",
    control.mode = list(theta = theta, restart = FALSE)
)

stopifnot(isTRUE(replay$control.used$mode$fixed_theta_replay))
stopifnot(isTRUE(replay$control.used$engine$skip_ccd))
stopifnot(identical(replay$control.used$engine$optimizer_max_evals, 0L))
stopifnot(isTRUE(replay$accuracy.diagnostics$flags$fixed_theta_replay))
stopifnot(isTRUE(replay$accuracy.diagnostics$flags$ccd_disabled))
stopifnot(as.integer(replay$diagnostics$optimizer_outer_iterations) == 0L)
stopifnot(max(abs(as.numeric(replay$theta.internal$mode) - as.numeric(theta))) < 1e-10)

bad <- try(
    rusty_inla(
        formula,
        data = data,
        family = "poisson",
        control.mode = list(restart = FALSE)
    ),
    silent = TRUE
)
stopifnot(inherits(bad, "try-error"))

cat("accuracy_gate_controls: ok\n")
