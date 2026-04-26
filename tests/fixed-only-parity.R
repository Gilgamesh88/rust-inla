source(file.path(getwd(), "tools", "load_worktree_package.R"), local = TRUE)
load_rustyinla_for_benchmarks(getwd())

if (!requireNamespace("INLA", quietly = TRUE)) {
    stop("Package 'INLA' is required for fixed-only parity tests.", call. = FALSE)
}

named_column <- function(df, column) {
    if (is.null(df) || nrow(df) == 0L || !(column %in% names(df))) {
        return(stats::setNames(numeric(), character()))
    }
    stats::setNames(as.numeric(df[[column]]), rownames(df))
}

max_fixed_mean_diff <- function(rusty_fit, inla_fit) {
    rusty_fixed <- named_column(rusty_fit$summary.fixed, "mean")
    inla_fixed <- named_column(inla_fit$summary.fixed, "mean")
    shared <- intersect(names(rusty_fixed), names(inla_fixed))
    if (length(shared) == 0L) {
        return(Inf)
    }
    max(abs(rusty_fixed[shared] - inla_fixed[shared]))
}

max_fitted_mean_rel_diff <- function(rusty_fit, inla_fit) {
    rusty_fitted <- rusty_fit$summary.fitted.values
    inla_fitted <- inla_fit$summary.fitted.values
    n_shared <- min(nrow(rusty_fitted), nrow(inla_fitted))
    if (n_shared == 0L) {
        return(Inf)
    }
    rusty_mean <- as.numeric(rusty_fitted$mean[seq_len(n_shared)])
    inla_mean <- as.numeric(inla_fitted$mean[seq_len(n_shared)])
    max(abs(rusty_mean - inla_mean) / pmax(1.0, abs(inla_mean)))
}

fit_pair <- function(formula, data, family) {
    rusty_fit <- rusty_inla(
        formula,
        data = data,
        family = family,
        output_profile = "thin"
    )
    inla_fit <- suppressWarnings(suppressMessages(
        INLA::inla(
            formula,
            data = data,
            family = family,
            control.compute = list(config = FALSE),
            control.predictor = list(compute = TRUE),
            num.threads = 1
        )
    ))
    list(rusty = rusty_fit, inla = inla_fit)
}

make_gaussian_case <- function(seed = 4101L) {
    set.seed(seed)
    n <- 90L
    x1 <- stats::rnorm(n)
    promo <- factor(sample(c("base", "promo"), n, replace = TRUE))
    eta <- 0.8 - 0.45 * x1 + 0.35 * (promo == "promo")
    data.frame(
        y = eta + stats::rnorm(n, sd = 0.18),
        x1 = x1,
        promo = promo
    )
}

make_poisson_case <- function(seed = 4102L) {
    set.seed(seed)
    n <- 160L
    x1 <- stats::rnorm(n)
    x2 <- stats::runif(n, -1.0, 1.0)
    promo <- factor(sample(c("base", "promo"), n, replace = TRUE))
    exposure <- stats::runif(n, 0.5, 2.0)
    log_exposure <- log(exposure)
    eta <- -0.30 + 0.35 * x1 - 0.20 * x2 + 0.30 * (promo == "promo") +
        0.15 * x1 * (promo == "promo") + log_exposure
    data.frame(
        y = stats::rpois(n, lambda = exp(eta)),
        x1 = x1,
        x2 = x2,
        promo = promo,
        log_exposure = log_exposure
    )
}

make_gamma_case <- function(seed = 4103L) {
    set.seed(seed)
    n <- 140L
    x1 <- stats::rnorm(n)
    promo <- factor(sample(c("base", "promo"), n, replace = TRUE))
    mu <- exp(1.1 + 0.25 * x1 + 0.20 * (promo == "promo"))
    shape <- 12.0
    data.frame(
        y = stats::rgamma(n, shape = shape, rate = shape / mu),
        x1 = x1,
        promo = promo
    )
}

cases <- list(
    list(
        id = "fixed_only_gaussian",
        family = "gaussian",
        formula = y ~ 1 + x1 + promo,
        data = make_gaussian_case(),
        fixed_tol = 0.08,
        fitted_tol = 0.08
    ),
    list(
        id = "fixed_only_poisson_offset_interaction",
        family = "poisson",
        formula = y ~ 1 + x1 * promo + x2 + offset(log_exposure),
        data = make_poisson_case(),
        fixed_tol = 0.20,
        fitted_tol = 0.20
    ),
    list(
        id = "fixed_only_gamma",
        family = "gamma",
        formula = y ~ 1 + x1 + promo,
        data = make_gamma_case(),
        fixed_tol = 0.25,
        fitted_tol = 0.20
    )
)

results <- lapply(cases, function(case) {
    fits <- fit_pair(case$formula, case$data, case$family)
    fixed_diff <- max_fixed_mean_diff(fits$rusty, fits$inla)
    fitted_diff <- max_fitted_mean_rel_diff(fits$rusty, fits$inla)

    if (length(fits$rusty$summary.random) != 0L) {
        stop(sprintf("Case %s unexpectedly returned random effects.", case$id), call. = FALSE)
    }
    if (!isTRUE(fixed_diff <= case$fixed_tol)) {
        stop(
            sprintf(
                "Case %s fixed mean diff %.6f exceeds tolerance %.6f.",
                case$id,
                fixed_diff,
                case$fixed_tol
            ),
            call. = FALSE
        )
    }
    if (!isTRUE(fitted_diff <= case$fitted_tol)) {
        stop(
            sprintf(
                "Case %s fitted mean rel diff %.6f exceeds tolerance %.6f.",
                case$id,
                fitted_diff,
                case$fitted_tol
            ),
            call. = FALSE
        )
    }

    data.frame(
        case_id = case$id,
        family = case$family,
        fixed_mean_max_abs = fixed_diff,
        fitted_mean_max_rel = fitted_diff,
        stringsAsFactors = FALSE
    )
})

results <- do.call(rbind, results)
print(results, row.names = FALSE)
cat(sprintf("\nFixed-only parity cases passed: %d/%d\n", nrow(results), length(cases)))
