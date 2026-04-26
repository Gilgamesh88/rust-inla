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

fixed_prior_precision <- 0.001

max_fixed_mean_diff <- function(rusty_fit, inla_fit) {
    rusty_fixed <- named_column(rusty_fit$summary.fixed, "mean")
    inla_fixed <- named_column(inla_fit$summary.fixed, "mean")
    shared <- intersect(names(rusty_fixed), names(inla_fixed))
    if (length(shared) == 0L) {
        return(Inf)
    }
    max(abs(rusty_fixed[shared] - inla_fixed[shared]))
}

max_fixed_sd_diff <- function(rusty_fit, inla_fit) {
    rusty_fixed <- named_column(rusty_fit$summary.fixed, "sd")
    inla_fixed <- named_column(inla_fit$summary.fixed, "sd")
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

glm_family <- function(family) {
    switch(
        family,
        gaussian = stats::gaussian(),
        poisson = stats::poisson(),
        gamma = stats::Gamma(link = "log"),
        stop(sprintf("No base glm() comparator for family '%s'.", family), call. = FALSE)
    )
}

max_glm_fixed_mean_diff <- function(rusty_fit, formula, data, family) {
    glm_fit <- suppressWarnings(stats::glm(formula, data = data, family = glm_family(family)))
    glm_fixed <- stats::coef(glm_fit)
    rusty_fixed <- named_column(rusty_fit$summary.fixed, "mean")
    shared <- intersect(names(rusty_fixed), names(glm_fixed))
    if (length(shared) == 0L) {
        return(Inf)
    }
    max(abs(rusty_fixed[shared] - glm_fixed[shared]))
}

build_model_frame <- function(formula, data) {
    mf <- stats::model.frame(formula, data = data, na.action = stats::na.pass)
    y <- as.numeric(stats::model.response(mf))
    X <- stats::model.matrix(formula, data = mf)
    offset <- stats::model.offset(mf)
    if (is.null(offset)) {
        offset <- rep(0.0, length(y))
    }
    list(y = y, X = X, offset = as.numeric(offset))
}

poisson_map_reference <- function(formula, data, prior_precision = fixed_prior_precision) {
    design <- build_model_frame(formula, data)
    y <- design$y
    X <- design$X
    offset <- design$offset
    k <- ncol(X)

    glm_fit <- suppressWarnings(stats::glm(formula, data = data, family = stats::poisson()))
    beta0 <- stats::coef(glm_fit)
    if (length(beta0) != k || any(!is.finite(beta0))) {
        beta0 <- rep(0.0, k)
    }

    objective <- function(beta) {
        eta <- as.numeric(X %*% beta) + offset
        -sum(y * eta - exp(eta) - lgamma(y + 1.0)) +
            0.5 * prior_precision * sum(beta^2)
    }
    gradient <- function(beta) {
        eta <- as.numeric(X %*% beta) + offset
        -as.numeric(crossprod(X, y - exp(eta))) + prior_precision * beta
    }

    opt <- stats::optim(
        par = beta0,
        fn = objective,
        gr = gradient,
        method = "BFGS",
        control = list(reltol = 1e-14, maxit = 10000)
    )
    if (opt$convergence != 0L) {
        stop(sprintf("Poisson MAP comparator did not converge: %s", opt$convergence), call. = FALSE)
    }

    beta <- opt$par
    eta <- as.numeric(X %*% beta) + offset
    w <- exp(eta)
    hessian <- crossprod(X * sqrt(w)) + diag(prior_precision, k)
    covariance <- solve(hessian)
    names(beta) <- colnames(X)
    sd <- sqrt(pmax(0.0, diag(covariance)))
    names(sd) <- colnames(X)

    list(mean = beta, sd = sd)
}

max_poisson_map_diff <- function(rusty_fit, formula, data, column) {
    ref <- poisson_map_reference(formula, data)
    rusty_fixed <- named_column(rusty_fit$summary.fixed, column)
    ref_values <- ref[[if (identical(column, "mean")) "mean" else "sd"]]
    shared <- intersect(names(rusty_fixed), names(ref_values))
    if (length(shared) == 0L) {
        return(Inf)
    }
    max(abs(rusty_fixed[shared] - ref_values[shared]))
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
        fixed_sd_tol = 0.08,
        fitted_tol = 0.08,
        glm_fixed_tol = 0.02,
        map_fixed_tol = NA_real_,
        map_fixed_sd_tol = NA_real_
    ),
    list(
        id = "fixed_only_poisson_offset_interaction",
        family = "poisson",
        formula = y ~ 1 + x1 * promo + x2 + offset(log_exposure),
        data = make_poisson_case(),
        fixed_tol = 0.20,
        fixed_sd_tol = 0.08,
        fitted_tol = 0.20,
        glm_fixed_tol = 0.02,
        map_fixed_tol = 1e-5,
        map_fixed_sd_tol = 1e-5
    ),
    list(
        id = "fixed_only_gamma",
        family = "gamma",
        formula = y ~ 1 + x1 + promo,
        data = make_gamma_case(),
        fixed_tol = 0.25,
        fixed_sd_tol = 0.10,
        fitted_tol = 0.20,
        glm_fixed_tol = 0.05,
        map_fixed_tol = NA_real_,
        map_fixed_sd_tol = NA_real_
    )
)

results <- lapply(cases, function(case) {
    fits <- fit_pair(case$formula, case$data, case$family)
    fixed_diff <- max_fixed_mean_diff(fits$rusty, fits$inla)
    fixed_sd_diff <- max_fixed_sd_diff(fits$rusty, fits$inla)
    fitted_diff <- max_fitted_mean_rel_diff(fits$rusty, fits$inla)
    glm_fixed_diff <- max_glm_fixed_mean_diff(
        fits$rusty,
        case$formula,
        case$data,
        case$family
    )
    map_fixed_diff <- NA_real_
    map_fixed_sd_diff <- NA_real_
    if (identical(case$family, "poisson")) {
        map_fixed_diff <- max_poisson_map_diff(fits$rusty, case$formula, case$data, "mean")
        map_fixed_sd_diff <- max_poisson_map_diff(fits$rusty, case$formula, case$data, "sd")
    }

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
    if (!isTRUE(fixed_sd_diff <= case$fixed_sd_tol)) {
        stop(
            sprintf(
                "Case %s fixed sd diff %.6f exceeds tolerance %.6f.",
                case$id,
                fixed_sd_diff,
                case$fixed_sd_tol
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
    if (!isTRUE(glm_fixed_diff <= case$glm_fixed_tol)) {
        stop(
            sprintf(
                "Case %s glm fixed mean diff %.6f exceeds tolerance %.6f.",
                case$id,
                glm_fixed_diff,
                case$glm_fixed_tol
            ),
            call. = FALSE
        )
    }
    if (is.finite(case$map_fixed_tol) && !isTRUE(map_fixed_diff <= case$map_fixed_tol)) {
        stop(
            sprintf(
                "Case %s MAP fixed mean diff %.10f exceeds tolerance %.10f.",
                case$id,
                map_fixed_diff,
                case$map_fixed_tol
            ),
            call. = FALSE
        )
    }
    if (is.finite(case$map_fixed_sd_tol) && !isTRUE(map_fixed_sd_diff <= case$map_fixed_sd_tol)) {
        stop(
            sprintf(
                "Case %s MAP fixed sd diff %.10f exceeds tolerance %.10f.",
                case$id,
                map_fixed_sd_diff,
                case$map_fixed_sd_tol
            ),
            call. = FALSE
        )
    }

    data.frame(
        case_id = case$id,
        family = case$family,
        fixed_mean_max_abs = fixed_diff,
        fixed_sd_max_abs = fixed_sd_diff,
        fitted_mean_max_rel = fitted_diff,
        glm_fixed_mean_max_abs = glm_fixed_diff,
        map_fixed_mean_max_abs = map_fixed_diff,
        map_fixed_sd_max_abs = map_fixed_sd_diff,
        stringsAsFactors = FALSE
    )
})

results <- do.call(rbind, results)
print(results, row.names = FALSE)
cat(sprintf("\nFixed-only parity cases passed: %d/%d\n", nrow(results), length(cases)))
