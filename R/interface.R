# Internal helper to assemble the backend specification consumed by Rust.
build_backend_spec <- function(
    formula,
    data,
    family,
    offset = NULL,
    offset_expr = substitute(offset),
    offset_env = parent.frame(),
    offset_provided = !missing(offset)
) {
    # 1. Parse formula to extract response
    tf <- terms(formula, specials = "f")
    resp_idx <- attr(tf, "response")
    if (resp_idx == 0) stop("Formula requires a response variable.")

    y_var <- as.character(attr(tf, "variables")[[resp_idx + 1]])
    y <- as.numeric(data[[y_var]])

    # 2. Extract fixed terms design matrix
    t_labels <- attr(tf, "term.labels")
    f_term_idx <- grep("^f\\(", t_labels)
    if (length(f_term_idx) > 0) {
        tf_fixed <- drop.terms(tf, f_term_idx, keep.response = FALSE)
    } else {
        tf_fixed <- delete.response(tf)
    }

    mf_fixed <- model.frame(tf_fixed, data = data, na.action = na.pass)
    X_fixed <- model.matrix(tf_fixed, mf_fixed)
    if (ncol(X_fixed) > 0L) {
        qr_fixed <- qr(X_fixed)
        if (qr_fixed$rank < ncol(X_fixed)) {
            aliased_idx <- qr_fixed$pivot[seq.int(qr_fixed$rank + 1L, ncol(X_fixed))]
            aliased_names <- colnames(X_fixed)[aliased_idx]
            stop(
                sprintf(
                    paste(
                        "Fixed-effects design matrix is rank-deficient.",
                        "Remove or reparameterize aliased columns: %s"
                    ),
                    paste(aliased_names, collapse = ", ")
                ),
                call. = FALSE
            )
        }
    }
    formula_offset <- model.offset(mf_fixed)
    if (!is.null(formula_offset)) {
        formula_offset <- as.numeric(formula_offset)
        if (length(formula_offset) != nrow(data)) {
            stop("Formula offset length does not match the number of observations.")
        }
    }
    user_offset <- NULL
    if (isTRUE(offset_provided)) {
        user_offset <- tryCatch(
            eval(offset_expr, envir = data, enclos = offset_env),
            error = function(e) {
                stop(
                    sprintf(
                        "Could not evaluate offset in the data/caller environment: %s",
                        e$message
                    ),
                    call. = FALSE
                )
            }
        )
        if (!is.null(user_offset) && !is.numeric(user_offset)) {
            stop("offset must evaluate to a numeric vector.")
        }
        if (!is.null(user_offset)) {
            user_offset <- as.numeric(user_offset)
        }
    }
    if (!is.null(user_offset) && length(user_offset) != nrow(data)) {
        stop("offset length does not match the number of observations.")
    }
    offset_parts <- Filter(Negate(is.null), list(formula_offset, user_offset))
    offset_vec <- if (length(offset_parts) == 0) {
        NULL
    } else {
        Reduce(`+`, offset_parts)
    }

    n_fixed <- ncol(X_fixed)
    x_matrix_flat <- as.numeric(X_fixed)

    # 3. Setup A Matrix Triplets for Random Effects
    A_i <- integer()
    A_j <- integer()
    A_x <- numeric()

    n_latent_total <- 0
    latent_blocks <- list()

    f_idx <- attr(tf, "specials")$f

    C_rows <- list()

    if (!is.null(f_idx)) {
        eval_env <- new.env(parent = emptyenv())
        eval_env$f <- f
        for (idx_f in f_idx) {
            f_call <- attr(tf, "variables")[[idx_f + 1]]
            f_res <- eval(f_call, envir = eval_env)
            c_name <- f_res$covariate_name
            m_type <- f_res$model
            constr <- isTRUE(f_res$constr)

            cov_data <- data[[c_name]]

            if (is.factor(cov_data)) {
                c_idx <- as.numeric(cov_data)
                level_values <- levels(cov_data)
            } else {
                cov_factor <- as.factor(cov_data)
                c_idx <- as.numeric(cov_factor)
                level_values <- type.convert(levels(cov_factor), as.is = TRUE)
            }

            n_latent_cov <- max(c_idx, na.rm=TRUE)
            if (identical(m_type, "rw2") && n_latent_cov < 3L) {
                stop("rw2 requires at least 3 unique levels.", call. = FALSE)
            }
            structure_values <- NULL
            if (identical(m_type, "rw2")) {
                structure_values <- suppressWarnings(as.numeric(level_values))
                if (length(structure_values) != n_latent_cov ||
                    anyNA(structure_values) ||
                    !all(is.finite(structure_values))) {
                    stop(
                        "rw2 requires numeric covariate values or factor levels that convert cleanly to numeric.",
                        call. = FALSE
                    )
                }
                if (any(diff(structure_values) <= 0)) {
                    stop("rw2 requires strictly increasing covariate values.", call. = FALSE)
                }
            }

            # Map into Trips (N_A rows)
            # R arrays are 1-based. A_i, A_j must be 0-based for Rust!
            valid_rows <- which(!is.na(c_idx))
            A_i <- c(A_i, valid_rows - 1)
            A_j <- c(A_j, (c_idx[valid_rows] - 1) + n_latent_total)
            A_x <- c(A_x, rep(1.0, length(valid_rows)))

            latent_blocks[[length(latent_blocks) + 1]] <- list(
                covariate_name = c_name,
                model = m_type,
                n_levels = as.integer(n_latent_cov),
                start = as.integer(n_latent_total),
                level_values = level_values,
                structure_values = structure_values
            )
            if (constr) {
                C_rows[[length(C_rows) + 1]] <- list(
                    start = n_latent_total,
                    len = n_latent_cov,
                    weights = rep(1.0, n_latent_cov)
                )
                if (identical(m_type, "rw2")) {
                    # RW2 has a two-dimensional null space: constant and linear trends.
                    C_rows[[length(C_rows) + 1]] <- list(
                        start = n_latent_total,
                        len = n_latent_cov,
                        weights = structure_values - mean(structure_values)
                    )
                }
            }

            n_latent_total <- n_latent_total + n_latent_cov
        }
    }

    n_constr <- length(C_rows)
    C_matrix_flat <- numeric()
    if (n_constr > 0) {
        C_matrix <- matrix(0.0, nrow = n_constr, ncol = n_latent_total)
        for (k in seq_along(C_rows)) {
            start <- C_rows[[k]]$start + 1
            len <- C_rows[[k]]$len
            C_matrix[k, start:(start + len - 1)] <- C_rows[[k]]$weights
        }
        C_matrix_flat <- as.numeric(t(C_matrix)) # Flatten row-major
    }

    list(
        y = y,
        likelihood = as.character(family),
        fixed_matrix = if (n_fixed > 0) x_matrix_flat else NULL,
        fixed_names = colnames(X_fixed),
        offset_user = if (is.null(user_offset)) NULL else user_offset,
        offset_arg_provided = isTRUE(offset_provided),
        n_fixed = as.integer(n_fixed),
        n_latent = as.integer(n_latent_total),
        a_i = if (length(A_i) > 0) as.integer(A_i) else NULL,
        a_j = if (length(A_j) > 0) as.integer(A_j) else NULL,
        a_x = if (length(A_x) > 0) as.numeric(A_x) else NULL,
        offset = if (is.null(offset_vec)) NULL else offset_vec,
        extr_constr = if (n_constr > 0) as.numeric(C_matrix_flat) else NULL,
        n_constr = as.integer(n_constr),
        latent_blocks = latent_blocks
    )
}

gaussian_marginal_grid <- function(mean, sd, n = 75L, sds = 4.0) {
    sd <- max(as.numeric(sd), 1e-10)
    x <- seq(mean - sds * sd, mean + sds * sd, length.out = n)
    y <- stats::dnorm(x, mean = mean, sd = sd)
    area <- sum(diff(x) * (head(y, -1) + tail(y, -1)) / 2)
    cbind(x = x, y = y / area)
}

weighted_kernel_marginal_grid <- function(samples, weights, n = 75L, sds = 4.0) {
    samples <- as.numeric(samples)
    weights <- as.numeric(weights)

    keep <- is.finite(samples) & is.finite(weights) & weights > 0
    samples <- samples[keep]
    weights <- weights[keep]

    if (length(samples) == 0) {
        return(NULL)
    }

    weights <- weights / sum(weights)
    weighted_mean <- sum(weights * samples)
    weighted_var <- sum(weights * (samples - weighted_mean)^2)
    weighted_sd <- sqrt(max(weighted_var, 0))

    unique_samples <- sort(unique(samples))
    min_spacing <- if (length(unique_samples) >= 2) {
        min(diff(unique_samples))
    } else {
        NA_real_
    }

    n_eff <- 1.0 / sum(weights^2)
    bw <- 1.06 * max(weighted_sd, 1e-6) * n_eff^(-1 / 5)
    if (is.finite(min_spacing)) {
        bw <- max(bw, 0.5 * min_spacing)
    }
    bw <- max(bw, 1e-6)

    span <- max(weighted_sd, bw)
    x_lo <- min(samples) - sds * span
    x_hi <- max(samples) + sds * span
    if (!is.finite(x_lo) || !is.finite(x_hi) || x_lo >= x_hi) {
        x_lo <- weighted_mean - sds * bw
        x_hi <- weighted_mean + sds * bw
    }

    x <- seq(x_lo, x_hi, length.out = n)
    y <- vapply(x, function(xi) {
        sum(weights * stats::dnorm(xi, mean = samples, sd = bw))
    }, numeric(1))
    area <- sum(diff(x) * (head(y, -1) + tail(y, -1)) / 2)
    cbind(x = x, y = y / area)
}

safe_elapsed_time <- function() {
    unname(proc.time()[["elapsed"]])
}

rusty_package_version <- local({
    cached_version <- NULL

    function() {
        if (!is.null(cached_version)) {
            return(cached_version)
        }

        cached_version <<- tryCatch(
            as.character(utils::packageVersion("rustyINLA")),
            error = function(e) NULL
        )
        if (!is.null(cached_version)) {
            return(cached_version)
        }

        description_path <- file.path(getwd(), "DESCRIPTION")
        if (file.exists(description_path)) {
            dcf <- tryCatch(read.dcf(description_path), error = function(e) NULL)
            if (!is.null(dcf) && "Version" %in% colnames(dcf)) {
                cached_version <<- as.character(dcf[1, "Version"])
                return(cached_version)
            }
        }

        cached_version <<- "unknown"
        cached_version
    }
})

build_size_random <- function(latent_blocks) {
    if (length(latent_blocks) == 0) {
        return(integer())
    }

    sizes <- vapply(latent_blocks, function(block) as.integer(block$n_levels), integer(1))
    names(sizes) <- vapply(latent_blocks, function(block) block$covariate_name, character(1))
    sizes
}

build_cpu_profile <- function(pre_time, running_time, post_time) {
    stats::setNames(
        pmax(c(pre_time, running_time, post_time, pre_time + running_time + post_time), 0),
        c("Pre", "Running", "Post", "Total")
    )
}

fallback_hyperparameter_specs <- function(n_theta) {
    lapply(seq_len(n_theta), function(idx) {
        list(
            name = paste("theta", idx),
            transform = identity
        )
    })
}

build_hyperparameter_specs <- function(backend_spec, family) {
    specs <- list()
    add_spec <- function(name, transform) {
        specs[[length(specs) + 1L]] <<- list(
            name = name,
            transform = transform
        )
    }

    for (block in backend_spec$latent_blocks) {
        cov_name <- block$covariate_name
        if (block$model %in% c("iid", "rw1", "rw2")) {
            add_spec(sprintf("Precision for %s", cov_name), exp)
            next
        }

        if (identical(block$model, "ar1")) {
            add_spec(sprintf("Precision for %s", cov_name), exp)
            add_spec(sprintf("Rho for %s", cov_name), function(theta) tanh(theta / 2.0))
        }

        if (identical(block$model, "ar2")) {
            add_spec(sprintf("Precision for %s", cov_name), exp)
            add_spec(sprintf("PACF1 for %s", cov_name), function(theta) tanh(theta / 2.0))
            add_spec(sprintf("PACF2 for %s", cov_name), function(theta) tanh(theta / 2.0))
        }
    }

    family_name <- as.character(family)[[1]]
    switch(
        family_name,
        gaussian = add_spec("Precision for the Gaussian observations", exp),
        poisson = NULL,
        gamma = add_spec("Shape for the Gamma observations", exp),
        zeroinflatedpoisson1 = add_spec(
            "Zero-inflation probability for the ZIP observations",
            stats::plogis
        ),
        tweedie = {
            add_spec("Dispersion for the Tweedie observations", exp)
            add_spec("Power for the Tweedie observations", function(theta) 1.0 + stats::plogis(theta))
        },
        NULL
    )

    specs
}

resolve_hyperparameter_specs <- function(backend_spec, family, n_theta) {
    specs <- build_hyperparameter_specs(backend_spec, family)
    if (length(specs) != n_theta) {
        warning(
            sprintf(
                "Hyperparameter spec count %d does not match theta length %d; using internal theta fallback.",
                length(specs),
                n_theta
            ),
            call. = FALSE
        )
        specs <- fallback_hyperparameter_specs(n_theta)
    }
    specs
}

transform_hyperparameter_values <- function(values, specs) {
    out <- numeric(length(specs))
    for (idx in seq_along(specs)) {
        out[[idx]] <- specs[[idx]]$transform(values[[idx]])
    }
    out
}

transform_hyperparameter_matrix <- function(theta_matrix, specs) {
    out <- theta_matrix
    for (idx in seq_along(specs)) {
        out[, idx] <- specs[[idx]]$transform(theta_matrix[, idx])
    }
    out
}

point_hyperparameter_summary <- function(values, row_names) {
    data.frame(
        row.names = row_names,
        mean = values,
        sd = NA_real_,
        `0.025quant` = NA_real_,
        `0.5quant` = values,
        `0.975quant` = NA_real_,
        mode = values,
        check.names = FALSE
    )
}

weighted_quantile <- function(values, weights, probs) {
    if (length(values) == 0 || length(weights) == 0) {
        return(rep(NA_real_, length(probs)))
    }

    ord <- order(values)
    values <- values[ord]
    weights <- weights[ord]
    weights <- weights / sum(weights)
    cdf <- cumsum(weights)

    vapply(probs, function(prob) {
        values[[which(cdf >= prob)[1L]]]
    }, numeric(1))
}

build_hyperparameter_summary <- function(res, backend_spec, family) {
    theta_opt <- if (is.null(res$theta_opt)) numeric() else as.numeric(res$theta_opt)
    n_theta <- length(theta_opt)
    if (n_theta == 0) {
        return(data.frame())
    }

    specs <- resolve_hyperparameter_specs(backend_spec, family, n_theta)
    theta_names <- vapply(specs, `[[`, character(1), "name")
    theta_mode <- transform_hyperparameter_values(theta_opt, specs)

    ccd_weights <- if (is.null(res$ccd_weights)) numeric() else as.numeric(res$ccd_weights)
    ccd_thetas <- if (is.null(res$ccd_thetas)) numeric() else as.numeric(res$ccd_thetas)

    if (length(ccd_weights) == 0 || length(ccd_thetas) == 0) {
        return(point_hyperparameter_summary(theta_mode, theta_names))
    }

    if (length(ccd_thetas) %% n_theta != 0) {
        warning(
            "CCD theta grid does not align with theta_opt length; using point hyperparameter fallback.",
            call. = FALSE
        )
        return(point_hyperparameter_summary(theta_mode, theta_names))
    }

    theta_matrix <- matrix(ccd_thetas, ncol = n_theta, byrow = TRUE)
    if (nrow(theta_matrix) != length(ccd_weights)) {
        warning(
            "CCD theta matrix row count does not match CCD weights; using point hyperparameter fallback.",
            call. = FALSE
        )
        return(point_hyperparameter_summary(theta_mode, theta_names))
    }

    theta_matrix <- transform_hyperparameter_matrix(theta_matrix, specs)

    means <- numeric(n_theta)
    sds <- numeric(n_theta)
    q025 <- numeric(n_theta)
    q500 <- numeric(n_theta)
    q975 <- numeric(n_theta)

    for (idx in seq_len(n_theta)) {
        vals <- theta_matrix[, idx]
        keep <- is.finite(vals) & is.finite(ccd_weights) & (ccd_weights > 0)
        if (!any(keep)) {
            means[[idx]] <- theta_mode[[idx]]
            sds[[idx]] <- NA_real_
            q025[[idx]] <- NA_real_
            q500[[idx]] <- theta_mode[[idx]]
            q975[[idx]] <- NA_real_
            next
        }

        vals <- vals[keep]
        weights <- ccd_weights[keep]
        weights <- weights / sum(weights)
        means[[idx]] <- sum(weights * vals)
        sds[[idx]] <- sqrt(max(sum(weights * (vals - means[[idx]])^2), 0))
        qs <- weighted_quantile(vals, weights, probs = c(0.025, 0.5, 0.975))
        q025[[idx]] <- qs[[1]]
        q500[[idx]] <- qs[[2]]
        q975[[idx]] <- qs[[3]]
    }

    data.frame(
        row.names = theta_names,
        mean = means,
        sd = sds,
        `0.025quant` = q025,
        `0.5quant` = q500,
        `0.975quant` = q975,
        mode = theta_mode,
        check.names = FALSE
    )
}

build_hyperparameter_marginals <- function(res, backend_spec, family, n = 75L, sds = 4.0) {
    theta_opt <- if (is.null(res$theta_opt)) numeric() else as.numeric(res$theta_opt)
    n_theta <- length(theta_opt)
    if (n_theta == 0) {
        return(NULL)
    }

    specs <- resolve_hyperparameter_specs(backend_spec, family, n_theta)
    theta_names <- vapply(specs, `[[`, character(1), "name")
    theta_mode <- transform_hyperparameter_values(theta_opt, specs)

    ccd_weights <- if (is.null(res$ccd_weights)) numeric() else as.numeric(res$ccd_weights)
    ccd_thetas <- if (is.null(res$ccd_thetas)) numeric() else as.numeric(res$ccd_thetas)

    if (length(ccd_weights) == 0 || length(ccd_thetas) == 0) {
        return(setNames(
            lapply(seq_len(n_theta), function(idx) {
                gaussian_marginal_grid(theta_mode[[idx]], 1e-3, n = n, sds = sds)
            }),
            theta_names
        ))
    }

    if (length(ccd_thetas) %% n_theta != 0) {
        warning(
            "CCD theta grid does not align with theta_opt length; using Gaussian hyperparameter fallback.",
            call. = FALSE
        )
        return(setNames(
            lapply(seq_len(n_theta), function(idx) {
                gaussian_marginal_grid(theta_mode[[idx]], 1e-3, n = n, sds = sds)
            }),
            theta_names
        ))
    }

    theta_matrix <- matrix(ccd_thetas, ncol = n_theta, byrow = TRUE)
    if (nrow(theta_matrix) != length(ccd_weights)) {
        warning(
            "CCD theta matrix row count does not match CCD weights; using Gaussian hyperparameter fallback.",
            call. = FALSE
        )
        return(setNames(
            lapply(seq_len(n_theta), function(idx) {
                gaussian_marginal_grid(theta_mode[[idx]], 1e-3, n = n, sds = sds)
            }),
            theta_names
        ))
    }

    theta_matrix <- transform_hyperparameter_matrix(theta_matrix, specs)

    setNames(
        lapply(seq_len(n_theta), function(idx) {
            weighted_kernel_marginal_grid(theta_matrix[, idx], ccd_weights, n = n, sds = sds)
        }),
        theta_names
    )
}

build_benchmark_args <- function(formula, family, backend_spec, output_profile) {
    list(
        formula = formula,
        family = family,
        offset = backend_spec$offset,
        offset_arg_provided = backend_spec$offset_arg_provided,
        control.compute = list(config = FALSE),
        control.predictor = list(compute = TRUE),
        output_profile = output_profile,
        n_fixed = backend_spec$n_fixed,
        n_latent = backend_spec$n_latent,
        latent_blocks = backend_spec$latent_blocks
    )
}

append_benchmark_outputs <- function(fit, res, backend_spec, data, formula, family, output_profile) {
    if (!is.null(res$eta_mean)) {
        n_eta <- min(length(res$eta_mean), nrow(data))
        eta_sd <- sqrt(pmax(res$eta_var[seq_len(n_eta)], 0))
        fit$summary.linear.predictor <- data.frame(
            mean = res$eta_mean[seq_len(n_eta)],
            sd = eta_sd,
            `0.025quant` = res$eta_q025[seq_len(n_eta)],
            `0.5quant` = res$eta_q500[seq_len(n_eta)],
            `0.975quant` = res$eta_q975[seq_len(n_eta)],
            mode = res$eta_q500[seq_len(n_eta)],
            check.names = FALSE
        )
    }

    if (!is.null(backend_spec$fixed_matrix) && backend_spec$n_fixed > 0) {
        fit$model.matrix <- matrix(
            backend_spec$fixed_matrix,
            nrow = nrow(data),
            ncol = backend_spec$n_fixed,
            dimnames = list(rownames(data), backend_spec$fixed_names)
        )
    }

    if (nrow(fit$summary.fixed) > 0) {
        fit$marginals.fixed <- setNames(
            lapply(seq_len(nrow(fit$summary.fixed)), function(idx) {
                gaussian_marginal_grid(
                    fit$summary.fixed$mean[[idx]],
                    fit$summary.fixed$sd[[idx]]
                )
            }),
            rownames(fit$summary.fixed)
        )
    }

    if (length(fit$summary.random) > 0) {
        fit$marginals.random <- lapply(fit$summary.random, function(rnd_df) {
            lapply(seq_len(nrow(rnd_df)), function(idx) {
                gaussian_marginal_grid(
                    rnd_df$mean[[idx]],
                    rnd_df$sd[[idx]]
                )
            })
        })
    }

    if (nrow(fit$summary.hyperpar) > 0) {
        fit$marginals.hyperpar <- build_hyperparameter_marginals(
            res = res,
            backend_spec = backend_spec,
            family = family
        )
    }

    fit$.args <- build_benchmark_args(
        formula = formula,
        family = family,
        backend_spec = backend_spec,
        output_profile = output_profile
    )

    fit
}

#' Native R Formula Interface for Rusty-INLA
#'
#' @param formula A robust R formula `y ~ 1 + f(cov, model="iid")`.
#' @param data A data.frame containing the variables.
#' @param family The likelihood family.
#' @param offset Optional offset supplied either as a numeric vector or as an
#'   expression evaluated in `data`, for example `offset = log(exposure)`.
#'   Formula-based offsets through `offset(...)` are also supported and are
#'   added to this argument when both are present.
#' @param output_profile Output payload profile. Use `"thin"` for the current
#'   lightweight default or `"benchmark"` to add parity-oriented outputs such
#'   as marginal curves and linear-predictor summaries for fairer memory
#'   comparisons against `R-INLA`.
#' @export
rusty_inla <- function(
    formula,
    data,
    family,
    offset = NULL,
    output_profile = c("thin", "benchmark")
) {
    fit_start_time <- safe_elapsed_time()
    output_profile <- match.arg(output_profile)
    backend_spec <- build_backend_spec(
        formula,
        data,
        family,
        offset = offset,
        offset_expr = substitute(offset),
        offset_env = parent.frame(),
        offset_provided = !missing(offset)
    )
    after_spec_time <- safe_elapsed_time()

    # 4. Invoke the Rust Core
    res <- rust_inla_run(backend_spec)
    after_run_time <- safe_elapsed_time()

    # Error handling from backend
    if (is.character(res)) { stop(res) }

    # 5. Build Standard Output Structure matching R-INLA expectations
    fit <- list(
        call = match.call(),
        formula = formula,
        data = data,
        family = family,
        offset = backend_spec$offset_user,
        offset_arg_provided = backend_spec$offset_arg_provided,
        mlik = res$log_mlik,
        summary.fixed = data.frame(
            row.names = backend_spec$fixed_names,
            mean = res$fixed_means,
            sd = res$fixed_sds,
            `0.025quant` = res$fixed_means - 1.96 * res$fixed_sds,
            `0.5quant` = res$fixed_means,
            `0.975quant` = res$fixed_means + 1.96 * res$fixed_sds,
            mode = res$fixed_means,
            check.names = FALSE
        ),
        summary.random = list(),
        output_profile = output_profile
    )

    # NEW: Extract Bayesian Marginal Fitted Values from the backend
    if (!is.null(res$fitted_mean)) {
        # INLA natively attaches structural marginals to the first latent indices
        # In simple models without custom A matrix mapping, these match the data rows exactly.
        n_fitted <- min(length(res$fitted_mean), nrow(data))
        fit$summary.fitted.values <- data.frame(
            mean = res$fitted_mean[1:n_fitted],
            sd   = rep(NA, n_fitted), # SD on response scale is complex mathematically
            `0.025quant` = res$fitted_q025[1:n_fitted],
            `0.5quant`   = res$fitted_q500[1:n_fitted],
            `0.975quant` = res$fitted_q975[1:n_fitted],
            mode = res$fitted_mode[1:n_fitted],
            check.names = FALSE
        )
    }

    # Format Hyperparameters securely if present
    if (length(res$theta_opt) > 0) {
        fit$summary.hyperpar <- build_hyperparameter_summary(
            res = res,
            backend_spec = backend_spec,
            family = family
        )
    } else {
        fit$summary.hyperpar <- data.frame()
    }

    # Populate Random Effects (Latent margins)
    if (length(backend_spec$latent_blocks) > 0) {
        start_idx <- 1
        for (block in backend_spec$latent_blocks) {
            c_name <- block$covariate_name
            nl <- as.integer(block$n_levels)
            end_idx <- start_idx + nl - 1
            level_values <- if (!is.null(block$level_values) && length(block$level_values) == nl) {
                block$level_values
            } else {
                seq_len(nl)
            }

            rnd_mean <- res$marg_means[start_idx:end_idx]
            rnd_var <- res$marg_vars[start_idx:end_idx]
            rnd_sd <- sqrt(rnd_var)

            rnd_df <- data.frame(
                ID = level_values,
                mean = rnd_mean,
                sd = rnd_sd,
                `0.025quant` = rnd_mean - 1.96 * rnd_sd,
                `0.5quant` = rnd_mean,
                `0.975quant` = rnd_mean + 1.96 * rnd_sd,
                mode = rnd_mean,
                check.names = FALSE
            )
            rownames(rnd_df) <- as.character(level_values)
            fit$summary.random[[c_name]] <- rnd_df
            start_idx <- end_idx + 1
        }
    }

    if (!is.null(res$diagnostics)) {
        fit$diagnostics <- res$diagnostics
    }
    if (!is.null(res$theta_init_used)) {
        fit$theta_init_used <- res$theta_init_used
    }
    if (!is.null(res$laplace_terms)) {
        fit$laplace_terms <- res$laplace_terms
    }
    if (!is.null(res$mode_x) || !is.null(res$mode_beta) || !is.null(res$mode_eta)) {
        fit$mode <- list(
            theta = res$theta_opt,
            x = res$mode_x,
            beta = res$mode_beta,
            eta = res$mode_eta,
            grad = res$mode_grad,
            curvature_raw = res$mode_curvature_raw,
            curvature = res$mode_curvature
        )
    }

    if (identical(output_profile, "benchmark")) {
        fit <- append_benchmark_outputs(
            fit = fit,
            res = res,
            backend_spec = backend_spec,
            data = data,
            formula = formula,
            family = family,
            output_profile = output_profile
        )
    }

    fit$names.fixed <- rownames(fit$summary.fixed)
    fit$size.random <- build_size_random(backend_spec$latent_blocks)
    fit$size.linear.predictor <- as.integer(nrow(data))
    fit$nhyper <- as.integer(length(res$theta_opt))
    fit$ok <- TRUE
    fit$version <- c(
        package = "rustyINLA",
        version = rusty_package_version()
    )
    after_post_time <- safe_elapsed_time()
    cpu_profile <- build_cpu_profile(
        pre_time = after_spec_time - fit_start_time,
        running_time = after_run_time - after_spec_time,
        post_time = after_post_time - after_run_time
    )
    fit$cpu.used <- cpu_profile
    fit$cpu.intern <- cpu_profile

    class(fit) <- "rusty_inla"
    return(fit)
}

#' @export
print.rusty_inla <- function(x, ...) {
    cat("Call:\n")
    print(x$call)
    cat(sprintf("\nLog Marginal-Likelihood: %f\n", x$mlik))
    cat("\nFixed effects:\n")
    print(round(x$summary.fixed, 4))
    invisible(x)
}

#' @export
summary.rusty_inla <- function(object, ...) {
    print(object)

    if (length(object$summary.random) > 0) {
        cat("\nRandom effects:\n")
        for (rnd_name in names(object$summary.random)) {
            cat(sprintf("  Name '%s' with %d levels\n", rnd_name, nrow(object$summary.random[[rnd_name]])))
        }
    }

    if (nrow(object$summary.hyperpar) > 0) {
        cat("\nModel hyperparameters:\n")
        print(round(object$summary.hyperpar, 4))
    }

    invisible(object)
}

#' Bayesian Prediction bypass for rusty-INLA
#'
#' Automatically appends new data, generates NA targets, forces the NA-trick
#' through the Rust backend, and extracts the posterior marginal quantiles.
#'
#' @export
predict.rusty_inla <- function(object, newdata, ...) {
    if (missing(newdata)) stop("Please provide newdata for predictions.")
    if (isTRUE(object$offset_arg_provided)) {
        stop("Prediction with an explicit offset vector is not supported yet. Put the offset transformation inside the formula using offset(...).")
    }

    tf <- terms(object$formula, specials = "f")
    resp_idx <- attr(tf, "response")
    y_var <- as.character(attr(tf, "variables")[[resp_idx + 1]])

    # 1. Structure the new data to match the training data
    # Bind targets as NA explicitly so Rust invokes the NA-Trick (skip logll, zero gradients)
    newdata[[y_var]] <- NA

    # Save lengths for extracting
    n_train <- nrow(object$data)
    n_test  <- nrow(newdata)

    # Concatenate Datasets
    combined_data <- rbind(object$data, newdata)

    # 2. Re-run rusty_inla invisibly on the combined dataset
    cat(sprintf("Running NA-Trick bypass for %d predictions...\n", n_test))
    fit_pred <- suppressWarnings(suppressMessages(
        rusty_inla(
            object$formula,
            data = combined_data,
            family = object$family,
            output_profile = if (is.null(object$output_profile)) "thin" else object$output_profile
        )
    ))

    # 3. Extract purely the predicted marginal quantiles!
    # Because they were appended sequentially, the predictions lie at the end.
    fitted_vals <- fit_pred$summary.fitted.values
    if (is.null(fitted_vals) || nrow(fitted_vals) < (n_train + n_test)) {
        stop("Backend did not return correctly formatted structural marginals for predictions.")
    }

    predictions <- fitted_vals[(n_train + 1):(n_train + n_test), ]
    rownames(predictions) <- 1:n_test

    return(predictions)
}
