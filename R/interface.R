formula_expr_label <- function(expr) {
    paste(deparse(expr, width.cutoff = 500L), collapse = "")
}

is_bare_formula_name <- function(expr) {
    is.name(expr) && !identical(as.character(expr), ".")
}

formula_variable_expr <- function(variables, idx) {
    variables[[idx + 1L]]
}

validate_supported_f_call <- function(f_call) {
    if (!is.call(f_call) || !identical(as.character(f_call[[1L]]), "f")) {
        stop("Latent terms must use f(...).", call. = FALSE)
    }

    args <- as.list(f_call[-1L])
    if (length(args) == 0L) {
        stop("f() requires a covariate and model.", call. = FALSE)
    }

    arg_names <- names(args)
    if (is.null(arg_names)) {
        arg_names <- rep("", length(args))
    } else {
        arg_names[is.na(arg_names)] <- ""
    }

    unsupported_args <- unique(arg_names[
        nzchar(arg_names) & !(arg_names %in% c("model", "constr"))
    ])
    if (length(unsupported_args) > 0L) {
        stop(
            sprintf(
                paste(
                    "Unsupported f() argument(s): %s.",
                    "The current rustyINLA formula subset supports only",
                    "f(covariate, model = ..., constr = ...)."
                ),
                paste(unsupported_args, collapse = ", ")
            ),
            call. = FALSE
        )
    }

    unnamed_count <- sum(!nzchar(arg_names))
    if (unnamed_count > 2L) {
        stop("f() supports only covariate and optional model as positional arguments.", call. = FALSE)
    }

    if (!is_bare_formula_name(args[[1L]])) {
        stop("f() covariate must be a single untransformed data column name.", call. = FALSE)
    }

    model_named <- which(arg_names == "model")
    if (length(model_named) > 1L) {
        stop("f() model was supplied more than once.", call. = FALSE)
    }
    if (length(model_named) == 1L && unnamed_count >= 2L) {
        stop("f() model was supplied both positionally and by name.", call. = FALSE)
    }

    model_expr <- if (length(model_named) == 1L) {
        args[[model_named[[1L]]]]
    } else if (length(args) >= 2L && !nzchar(arg_names[[2L]])) {
        args[[2L]]
    } else {
        NULL
    }
    if (!is.character(model_expr) || length(model_expr) != 1L || is.na(model_expr)) {
        stop("f() model must be a literal supported model string.", call. = FALSE)
    }
    if (!(model_expr %in% c("iid", "rw1", "rw2", "ar1", "ar2"))) {
        stop(sprintf("Unsupported latent model '%s'.", model_expr), call. = FALSE)
    }

    constr_named <- which(arg_names == "constr")
    if (length(constr_named) > 1L) {
        stop("f() constr was supplied more than once.", call. = FALSE)
    }
    if (length(constr_named) == 1L) {
        constr_expr <- args[[constr_named[[1L]]]]
        if (!is.logical(constr_expr) || length(constr_expr) != 1L || is.na(constr_expr)) {
            stop("f() constr must be a literal single TRUE/FALSE value.", call. = FALSE)
        }
    }

    invisible(TRUE)
}

validate_fixed_formula_variable <- function(expr, data) {
    if (!is_bare_formula_name(expr)) {
        stop(
            sprintf(
                paste(
                    "Unsupported fixed-effect term '%s'.",
                    "Create transformed variables in data first;",
                    "the current subset supports bare columns and simple interactions among them."
                ),
                formula_expr_label(expr)
            ),
            call. = FALSE
        )
    }

    var_name <- as.character(expr)
    if (!(var_name %in% names(data))) {
        stop(sprintf("Formula variable '%s' was not found in data.", var_name), call. = FALSE)
    }

    value <- data[[var_name]]
    if (is.character(value)) {
        stop(
            sprintf("Fixed-effect column '%s' is character; convert it to a factor explicitly.", var_name),
            call. = FALSE
        )
    }
    if (!(is.numeric(value) || is.integer(value) || is.logical(value) || is.factor(value))) {
        stop(
            sprintf(
                "Fixed-effect column '%s' must be numeric, integer, logical, or factor.",
                var_name
            ),
            call. = FALSE
        )
    }

    invisible(TRUE)
}

validate_supported_formula_subset <- function(formula, data, tf) {
    if (!is.data.frame(data)) {
        stop("data must be a data.frame.", call. = FALSE)
    }

    resp_idx <- attr(tf, "response")
    if (resp_idx == 0L) {
        stop("Formula requires a response variable.", call. = FALSE)
    }

    variables <- attr(tf, "variables")
    response_expr <- formula_variable_expr(variables, resp_idx)
    if (!is_bare_formula_name(response_expr)) {
        stop("Response must be a single data column name.", call. = FALSE)
    }

    response_name <- as.character(response_expr)
    if (!(response_name %in% names(data))) {
        stop(sprintf("Response variable '%s' was not found in data.", response_name), call. = FALSE)
    }
    response_value <- data[[response_name]]
    if (!(is.numeric(response_value) || is.integer(response_value) || is.logical(response_value))) {
        stop("Response must be numeric, integer, or logical.", call. = FALSE)
    }
    y <- as.numeric(response_value)
    if (any(is.infinite(y))) {
        stop("Response contains infinite values.", call. = FALSE)
    }

    f_idx <- attr(tf, "specials")$f
    if (is.null(f_idx)) {
        f_idx <- integer()
    }
    for (idx in f_idx) {
        validate_supported_f_call(formula_variable_expr(variables, idx))
    }

    factors <- attr(tf, "factors")
    term_labels <- attr(tf, "term.labels")
    f_term_idx <- integer()
    fixed_variable_idx <- integer()

    has_factor_terms <- !is.null(factors) && is.matrix(factors) && ncol(factors) > 0L

    if (has_factor_terms && any(factors[resp_idx, ] != 0)) {
        stop("Response variable cannot also appear on the right-hand side.", call. = FALSE)
    }

    if (has_factor_terms) {
        for (term_idx in seq_len(ncol(factors))) {
            variable_rows <- which(factors[, term_idx] != 0)
            f_rows <- intersect(variable_rows, f_idx)

            if (length(f_rows) > 0L) {
                standalone_f <- length(variable_rows) == 1L &&
                    length(f_rows) == 1L &&
                    factors[f_rows[[1L]], term_idx] == 1
                if (!standalone_f) {
                    stop(
                        sprintf(
                            paste(
                                "Latent f() terms must be standalone additive terms;",
                                "unsupported term: %s"
                            ),
                            term_labels[[term_idx]]
                        ),
                        call. = FALSE
                    )
                }
                f_term_idx <- c(f_term_idx, term_idx)
            } else {
                fixed_variable_idx <- union(fixed_variable_idx, variable_rows)
            }
        }
    }

    fixed_variable_idx <- setdiff(fixed_variable_idx, resp_idx)
    for (idx in fixed_variable_idx) {
        validate_fixed_formula_variable(formula_variable_expr(variables, idx), data)
    }

    list(
        response_name = response_name,
        f_term_idx = f_term_idx
    )
}

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
    formula_info <- validate_supported_formula_subset(formula, data, tf)

    y_var <- formula_info$response_name
    y <- as.numeric(data[[y_var]])

    # 2. Extract fixed terms design matrix
    f_term_idx <- formula_info$f_term_idx
    if (length(f_term_idx) > 0) {
        tf_fixed <- drop.terms(tf, f_term_idx, keep.response = FALSE)
    } else {
        tf_fixed <- delete.response(tf)
    }

    mf_fixed <- tryCatch(
        model.frame(tf_fixed, data = data, na.action = na.pass),
        error = function(e) {
            stop(
                sprintf("Could not build fixed-effects model frame: %s", e$message),
                call. = FALSE
            )
        }
    )
    X_fixed <- tryCatch(
        model.matrix(tf_fixed, mf_fixed),
        error = function(e) {
            stop(
                sprintf("Could not build fixed-effects design matrix: %s", e$message),
                call. = FALSE
            )
        }
    )
    if (length(X_fixed) > 0L && any(!is.finite(X_fixed))) {
        stop(
            paste(
                "Fixed-effects design matrix contains non-finite values.",
                "Check fixed-effect covariates and factor levels."
            ),
            call. = FALSE
        )
    }
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
        if (any(!is.finite(formula_offset))) {
            stop("Formula offset contains non-finite values.", call. = FALSE)
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
            if (any(!is.finite(user_offset))) {
                stop("offset contains non-finite values.", call. = FALSE)
            }
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
    if (n_fixed == 0L && length(f_term_idx) == 0L) {
        stop(
            "Formula must include at least one fixed-effect column or standalone f(...) latent term.",
            call. = FALSE
        )
    }
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
                n_latent_cov <- length(level_values)
            } else {
                cov_factor <- as.factor(cov_data)
                c_idx <- as.numeric(cov_factor)
                level_values <- type.convert(levels(cov_factor), as.is = TRUE)
                n_latent_cov <- max(c_idx, na.rm=TRUE)
            }

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

build_backend_signature <- function(backend_spec, family) {
    list(
        family = as.character(family)[[1L]],
        fixed_names = backend_spec$fixed_names,
        n_fixed = as.integer(backend_spec$n_fixed),
        n_latent = as.integer(backend_spec$n_latent),
        latent_blocks = lapply(backend_spec$latent_blocks, function(block) {
            list(
                covariate_name = block$covariate_name,
                model = block$model,
                n_levels = as.integer(block$n_levels),
                level_values = block$level_values,
                structure_values = block$structure_values
            )
        })
    )
}

build_internal_design <- function(backend_spec, n_data) {
    list(
        n_data = as.integer(n_data),
        fixed_matrix = backend_spec$fixed_matrix,
        fixed_names = backend_spec$fixed_names,
        n_fixed = as.integer(backend_spec$n_fixed),
        n_latent = as.integer(backend_spec$n_latent),
        a_i = backend_spec$a_i,
        a_j = backend_spec$a_j,
        a_x = backend_spec$a_x,
        offset = if (is.null(backend_spec$offset)) rep(0.0, n_data) else backend_spec$offset,
        latent_blocks = backend_spec$latent_blocks
    )
}

validate_iid_posterior_state_scope <- function(signature, n_theta) {
    blocks <- signature$latent_blocks
    if (length(blocks) != 1L || !identical(blocks[[1L]]$model, "iid")) {
        stop(
            paste(
                "Experimental posterior-state updates currently support only",
                "one iid latent block."
            ),
            call. = FALSE
        )
    }
    if (n_theta < 1L) {
        stop("Posterior-state update requires at least one iid hyperparameter.", call. = FALSE)
    }
    invisible(TRUE)
}

sum_by_index <- function(values, index, n) {
    out <- numeric(n)
    if (length(values) == 0L) {
        return(out)
    }
    grouped <- rowsum(as.numeric(values), group = as.integer(index), reorder = FALSE)
    out[as.integer(rownames(grouped))] <- as.numeric(grouped[, 1L])
    out
}

response_values_from_fit <- function(fit) {
    tf <- terms(fit$formula, specials = "f")
    resp_idx <- attr(tf, "response")
    if (is.null(resp_idx) || resp_idx == 0L) {
        stop("fit formula does not contain a response for likelihood evidence extraction.", call. = FALSE)
    }
    y_var <- as.character(attr(tf, "variables")[[resp_idx + 1L]])
    if (!(y_var %in% names(fit$data))) {
        stop("fit data does not contain the response needed for likelihood evidence extraction.", call. = FALSE)
    }
    as.numeric(fit$data[[y_var]])
}

fixed_iid_likelihood_theta <- function(fit) {
    theta <- as.numeric(fit$internal.hyperpar$theta_mode)
    if (length(theta) <= 1L) {
        return(numeric())
    }
    theta[-1L]
}

fixed_iid_mode_loglik_values <- function(fit, eta) {
    y <- response_values_from_fit(fit)
    if (length(y) != length(eta)) {
        stop("fit response length does not match mode eta for likelihood evidence extraction.", call. = FALSE)
    }
    family_name <- as.character(fit$family)[[1L]]
    theta_lik <- fixed_iid_likelihood_theta(fit)
    safe_eta <- pmax(pmin(as.numeric(eta), 50.0), -50.0)
    out <- numeric(length(y))
    observed <- !is.na(y)

    switch(
        family_name,
        gaussian = {
            if (length(theta_lik) < 1L) {
                stop("Gaussian likelihood evidence extraction requires an observation precision theta.", call. = FALSE)
            }
            tau <- exp(theta_lik[[1L]])
            resid <- y[observed] - eta[observed]
            out[observed] <- 0.5 * theta_lik[[1L]] - 0.5 * tau * resid * resid - 0.5 * log(2.0 * pi)
        },
        poisson = {
            out[observed] <- y[observed] * safe_eta[observed] -
                exp(safe_eta[observed]) -
                lgamma(y[observed] + 1.0)
        },
        gamma = {
            if (length(theta_lik) < 1L) {
                stop("Gamma likelihood evidence extraction requires a shape theta.", call. = FALSE)
            }
            phi <- exp(theta_lik[[1L]])
            y_obs <- y[observed]
            eta_obs <- safe_eta[observed]
            out[observed] <- phi * theta_lik[[1L]] -
                phi * eta_obs +
                (phi - 1.0) * log(y_obs) -
                phi * y_obs / exp(eta_obs) -
                lgamma(phi)
        },
        zeroinflatedpoisson1 = {
            if (length(theta_lik) < 1L) {
                stop("ZIP likelihood evidence extraction requires a zero-inflation theta.", call. = FALSE)
            }
            p <- stats::plogis(theta_lik[[1L]])
            y_obs <- y[observed]
            eta_obs <- safe_eta[observed]
            mu <- exp(eta_obs)
            zero <- y_obs == 0.0
            values <- numeric(length(y_obs))
            values[zero] <- log(p + (1.0 - p) * exp(-mu[zero]))
            values[!zero] <- log(1.0 - p) +
                y_obs[!zero] * eta_obs[!zero] -
                mu[!zero] -
                lgamma(y_obs[!zero] + 1.0)
            out[observed] <- values
        },
        tweedie = {
            if (length(theta_lik) < 2L) {
                stop("Tweedie likelihood evidence extraction requires dispersion and power theta values.", call. = FALSE)
            }
            phi <- exp(theta_lik[[1L]])
            p_power <- 1.0 + stats::plogis(theta_lik[[2L]])
            y_obs <- y[observed]
            eta_obs <- safe_eta[observed]
            mu <- exp(eta_obs)
            zero <- y_obs == 0.0
            values <- numeric(length(y_obs))
            values[zero] <- -(mu[zero]^(2.0 - p_power)) / (phi * (2.0 - p_power))
            if (any(!zero)) {
                y_pos <- y_obs[!zero]
                mu_pos <- mu[!zero]
                d <- 2.0 * (
                    (y_pos^(2.0 - p_power)) / ((1.0 - p_power) * (2.0 - p_power)) -
                        (y_pos * mu_pos^(1.0 - p_power)) / (1.0 - p_power) +
                        (mu_pos^(2.0 - p_power)) / (2.0 - p_power)
                )
                values[!zero] <- -0.5 * log(2.0 * pi * phi * y_pos^p_power) - d / (2.0 * phi)
            }
            out[observed] <- values
        },
        stop(sprintf(
            "Likelihood evidence log constants are not implemented for family '%s'.",
            family_name
        ), call. = FALSE)
    )

    if (any(!is.finite(out[observed]))) {
        stop("Likelihood evidence log constants contain non-finite values.", call. = FALSE)
    }
    out
}

fixed_iid_likelihood_evidence <- function(fit) {
    design <- fit$internal.design
    if (is.null(design)) {
        stop("fit does not contain the internal design needed for update-state extraction.", call. = FALSE)
    }

    n_data <- as.integer(design$n_data)
    n_fixed <- as.integer(design$n_fixed)
    n_latent <- as.integer(design$n_latent)
    if (n_data <= 0L || n_fixed <= 0L || n_latent <= 0L) {
        stop("fixed+iid update-state extraction requires positive data, fixed, and latent dimensions.", call. = FALSE)
    }
    if (is.null(design$fixed_matrix) || length(design$fixed_matrix) != n_data * n_fixed) {
        stop("fit fixed-effect design is not available for update-state extraction.", call. = FALSE)
    }
    if (is.null(design$a_i) || is.null(design$a_j) || is.null(design$a_x)) {
        stop("fit latent mapping is not available for update-state extraction.", call. = FALSE)
    }
    if (length(design$a_i) != length(design$a_j) || length(design$a_i) != length(design$a_x)) {
        stop("fit latent mapping is internally inconsistent.", call. = FALSE)
    }

    eta <- as.numeric(fit$mode$eta)
    grad <- as.numeric(fit$mode$grad)
    curvature <- as.numeric(fit$mode$curvature)
    offset <- as.numeric(design$offset)
    if (length(eta) != n_data || length(grad) != n_data ||
        length(curvature) != n_data || length(offset) != n_data) {
        stop("fit mode quantities do not match the stored design.", call. = FALSE)
    }
    if (any(!is.finite(eta)) || any(!is.finite(grad)) ||
        any(!is.finite(curvature)) || any(!is.finite(offset))) {
        stop("fit mode quantities contain non-finite values.", call. = FALSE)
    }

    curvature <- pmax(curvature, sqrt(.Machine$double.eps))
    X <- matrix(
        as.numeric(design$fixed_matrix),
        nrow = n_data,
        ncol = n_fixed,
        dimnames = list(NULL, design$fixed_names)
    )
    centered_mode <- eta - offset
    weighted_pseudo <- curvature * centered_mode + grad
    mode_loglik <- fixed_iid_mode_loglik_values(fit, eta)
    log_constant <- sum(mode_loglik) -
        sum(grad * centered_mode) -
        0.5 * sum(curvature * centered_mode * centered_mode)

    fixed_precision <- crossprod(X, X * curvature)
    fixed_precision <- (fixed_precision + t(fixed_precision)) / 2.0
    fixed_linear <- as.numeric(crossprod(X, weighted_pseudo))
    names(fixed_linear) <- design$fixed_names
    dimnames(fixed_precision) <- list(design$fixed_names, design$fixed_names)

    a_rows <- as.integer(design$a_i) + 1L
    a_cols <- as.integer(design$a_j) + 1L
    a_x <- as.numeric(design$a_x)
    if (any(a_rows < 1L | a_rows > n_data) || any(a_cols < 1L | a_cols > n_latent) ||
        any(!is.finite(a_x))) {
        stop("fit latent mapping contains invalid entries.", call. = FALSE)
    }

    row_curvature <- curvature[a_rows]
    latent_precision <- sum_by_index(a_x * row_curvature * a_x, a_cols, n_latent)
    latent_linear <- sum_by_index(a_x * weighted_pseudo[a_rows], a_cols, n_latent)
    latent_fixed_precision <- matrix(
        0.0,
        nrow = n_latent,
        ncol = n_fixed,
        dimnames = list(NULL, design$fixed_names)
    )
    for (j in seq_len(n_fixed)) {
        latent_fixed_precision[, j] <- sum_by_index(
            a_x * row_curvature * X[a_rows, j],
            a_cols,
            n_latent
        )
    }

    list(
        fixed_precision = fixed_precision,
        fixed_linear = fixed_linear,
        latent_precision = latent_precision,
        latent_linear = latent_linear,
        latent_fixed_precision = latent_fixed_precision,
        log_constant = as.numeric(log_constant)
    )
}

build_source_mode_theta_evidence <- function(
    fit,
    theta_mode,
    fixed_names,
    iid_levels,
    fixed_precision,
    fixed_linear,
    iid_precision_diag,
    iid_linear,
    iid_fixed_cross_precision,
    log_constant
) {
    support_names <- "source_mode"
    theta_names <- fit$internal.hyperpar$theta_names
    if (is.null(theta_names) || length(theta_names) != length(theta_mode)) {
        theta_names <- paste0("theta", seq_along(theta_mode))
    }
    theta <- matrix(
        as.numeric(theta_mode),
        nrow = 1L,
        dimnames = list(support_names, theta_names)
    )
    H_beta_beta <- array(
        as.numeric(fixed_precision),
        dim = c(length(fixed_names), length(fixed_names), 1L),
        dimnames = list(fixed_names, fixed_names, support_names)
    )
    h_beta <- matrix(
        as.numeric(fixed_linear),
        nrow = 1L,
        dimnames = list(support_names, fixed_names)
    )
    H_u_u_diag <- matrix(
        as.numeric(iid_precision_diag),
        nrow = 1L,
        dimnames = list(support_names, iid_levels)
    )
    h_u <- matrix(
        as.numeric(iid_linear),
        nrow = 1L,
        dimnames = list(support_names, iid_levels)
    )
    H_u_beta <- array(
        as.numeric(iid_fixed_cross_precision),
        dim = c(length(iid_levels), length(fixed_names), 1L),
        dimnames = list(iid_levels, fixed_names, support_names)
    )

    list(
        version = 1L,
        strategy = "source_mode_single_point",
        solver_status = "not_integrated",
        n_support = 1L,
        theta_names = theta_names,
        theta = theta,
        weights = stats::setNames(1.0, support_names),
        log_weights = stats::setNames(0.0, support_names),
        log_constants = stats::setNames(as.numeric(log_constant), support_names),
        block_format = "dense_fixed_diag_iid_cross_by_support",
        H_beta_beta = H_beta_beta,
        h_beta = h_beta,
        H_u_u_diag = H_u_u_diag,
        h_u = h_u,
        H_u_beta = H_u_beta
    )
}

build_internal_theta_evidence <- function(
    res,
    backend_spec,
    theta_matrix,
    ccd_weights,
    ccd_log_mlik,
    ccd_log_weight,
    theta_names
) {
    n_support <- if (is.null(theta_matrix)) 0L else nrow(theta_matrix)
    n_fixed <- as.integer(backend_spec$n_fixed)
    n_latent <- as.integer(backend_spec$n_latent)
    if (n_support <= 0L || n_fixed <= 0L || n_latent <= 0L) {
        return(NULL)
    }

    fixed_precision <- if (is.null(res$theta_evidence_fixed_precision)) {
        numeric()
    } else {
        as.numeric(res$theta_evidence_fixed_precision)
    }
    fixed_linear <- if (is.null(res$theta_evidence_fixed_linear)) {
        numeric()
    } else {
        as.numeric(res$theta_evidence_fixed_linear)
    }
    latent_precision <- if (is.null(res$theta_evidence_latent_precision_diag)) {
        numeric()
    } else {
        as.numeric(res$theta_evidence_latent_precision_diag)
    }
    latent_linear <- if (is.null(res$theta_evidence_latent_linear)) {
        numeric()
    } else {
        as.numeric(res$theta_evidence_latent_linear)
    }
    latent_fixed <- if (is.null(res$theta_evidence_latent_fixed_precision)) {
        numeric()
    } else {
        as.numeric(res$theta_evidence_latent_fixed_precision)
    }
    log_constants <- if (is.null(res$theta_evidence_log_constant)) {
        numeric()
    } else {
        as.numeric(res$theta_evidence_log_constant)
    }

    expected_fixed_precision <- n_support * n_fixed * n_fixed
    expected_fixed_linear <- n_support * n_fixed
    expected_latent <- n_support * n_latent
    expected_latent_fixed <- n_support * n_latent * n_fixed
    if (length(fixed_precision) != expected_fixed_precision ||
        length(fixed_linear) != expected_fixed_linear ||
        length(latent_precision) != expected_latent ||
        length(latent_linear) != expected_latent ||
        length(latent_fixed) != expected_latent_fixed ||
        length(log_constants) != n_support) {
        return(NULL)
    }

    support_names <- paste0("theta_", seq_len(n_support))
    support_names[[1L]] <- "source_mode"
    fixed_names <- as.character(backend_spec$fixed_names)
    latent_names <- as.character(seq_len(n_latent))
    H_beta_beta <- array(
        0.0,
        dim = c(n_fixed, n_fixed, n_support),
        dimnames = list(fixed_names, fixed_names, support_names)
    )
    H_u_beta <- array(
        0.0,
        dim = c(n_latent, n_fixed, n_support),
        dimnames = list(latent_names, fixed_names, support_names)
    )
    for (s in seq_len(n_support)) {
        fixed_start <- (s - 1L) * n_fixed * n_fixed + 1L
        fixed_end <- fixed_start + n_fixed * n_fixed - 1L
        H_beta_beta[, , s] <- matrix(
            fixed_precision[fixed_start:fixed_end],
            nrow = n_fixed,
            ncol = n_fixed,
            byrow = TRUE,
            dimnames = list(fixed_names, fixed_names)
        )

        cross_start <- (s - 1L) * n_latent * n_fixed + 1L
        cross_end <- cross_start + n_latent * n_fixed - 1L
        H_u_beta[, , s] <- matrix(
            latent_fixed[cross_start:cross_end],
            nrow = n_latent,
            ncol = n_fixed,
            dimnames = list(latent_names, fixed_names)
        )
    }

    weights <- as.numeric(ccd_weights)
    if (length(weights) != n_support || any(!is.finite(weights)) || sum(weights) <= 0.0) {
        weights <- rep(1.0 / n_support, n_support)
    } else {
        weights <- weights / sum(weights)
    }

    dimnames(theta_matrix) <- list(support_names, theta_names)
    h_beta <- matrix(
        fixed_linear,
        nrow = n_support,
        byrow = TRUE,
        dimnames = list(support_names, fixed_names)
    )
    H_u_u_diag <- matrix(
        latent_precision,
        nrow = n_support,
        byrow = TRUE,
        dimnames = list(support_names, latent_names)
    )
    h_u <- matrix(
        latent_linear,
        nrow = n_support,
        byrow = TRUE,
        dimnames = list(support_names, latent_names)
    )

    list(
        version = 2L,
        strategy = if (n_support > 1L) "ccd_support_modes" else "source_mode_single_point",
        solver_status = "not_integrated",
        n_support = as.integer(n_support),
        theta_names = theta_names,
        theta = theta_matrix,
        weights = stats::setNames(weights, support_names),
        log_weights = stats::setNames(log(pmax(weights, 1e-300)), support_names),
        log_unnormalized_weights = stats::setNames(as.numeric(ccd_log_weight), support_names),
        log_mlik = stats::setNames(as.numeric(ccd_log_mlik), support_names),
        log_constants = stats::setNames(log_constants, support_names),
        block_format = "dense_fixed_diag_iid_cross_by_support",
        H_beta_beta = H_beta_beta,
        h_beta = h_beta,
        H_u_u_diag = H_u_u_diag,
        h_u = h_u,
        H_u_beta = H_u_beta
    )
}

weighted_mean_var <- function(values, weights) {
    keep <- is.finite(values) & is.finite(weights) & weights > 0
    values <- as.numeric(values[keep])
    weights <- as.numeric(weights[keep])
    if (length(values) == 0L) {
        return(list(mean = NA_real_, var = NA_real_))
    }
    weights <- weights / sum(weights)
    mean <- sum(weights * values)
    var <- sum(weights * (values - mean)^2)
    list(mean = mean, var = max(var, 0.0))
}

build_internal_hyperparameter_state <- function(res, backend_spec, family) {
    theta_opt <- if (is.null(res$theta_opt)) numeric() else as.numeric(res$theta_opt)
    n_theta <- length(theta_opt)
    if (n_theta == 0L) {
        return(NULL)
    }

    specs <- resolve_hyperparameter_specs(backend_spec, family, n_theta)
    theta_names <- vapply(specs, `[[`, character(1), "name")
    ccd_weights <- if (is.null(res$ccd_weights)) numeric() else as.numeric(res$ccd_weights)
    ccd_thetas <- if (is.null(res$ccd_thetas)) numeric() else as.numeric(res$ccd_thetas)

    theta_matrix <- NULL
    if (length(ccd_weights) > 0L &&
        length(ccd_thetas) > 0L &&
        length(ccd_thetas) %% n_theta == 0L) {
        candidate <- matrix(ccd_thetas, ncol = n_theta, byrow = TRUE)
        if (nrow(candidate) == length(ccd_weights)) {
            theta_matrix <- candidate
        }
    }
    ccd_log_mlik <- if (is.null(res$ccd_log_mlik)) numeric() else as.numeric(res$ccd_log_mlik)
    ccd_log_weight <- if (is.null(res$ccd_log_weight)) numeric() else as.numeric(res$ccd_log_weight)
    theta_evidence <- build_internal_theta_evidence(
        res = res,
        backend_spec = backend_spec,
        theta_matrix = theta_matrix,
        ccd_weights = ccd_weights,
        ccd_log_mlik = ccd_log_mlik,
        ccd_log_weight = ccd_log_weight,
        theta_names = theta_names
    )

    list(
        theta_names = theta_names,
        theta_mode = theta_opt,
        ccd_thetas = theta_matrix,
        ccd_weights = ccd_weights,
        ccd_base_weights = if (is.null(res$ccd_base_weights)) numeric() else as.numeric(res$ccd_base_weights),
        ccd_log_mlik = ccd_log_mlik,
        ccd_log_weight = ccd_log_weight,
        ccd_hessian_eigenvalues = if (is.null(res$ccd_hessian_eigenvalues)) numeric() else as.numeric(res$ccd_hessian_eigenvalues),
        theta_evidence = theta_evidence,
        internal_scale = TRUE
    )
}

same_fixed_signature <- function(lhs, rhs) {
    identical(as.character(lhs$fixed_names), as.character(rhs$fixed_names))
}

format_signature_values <- function(values) {
    values <- as.character(values)
    if (length(values) == 0L) {
        return("<none>")
    }
    paste(values, collapse = ", ")
}

iid_update_signature_mismatch <- function(state_signature, backend_signature) {
    if (!identical(state_signature$family, backend_signature$family)) {
        return(sprintf(
            "family mismatch: state uses '%s', update uses '%s'",
            state_signature$family,
            backend_signature$family
        ))
    }
    if (!same_fixed_signature(state_signature, backend_signature)) {
        return(sprintf(
            "fixed-effect columns mismatch: state has [%s], update has [%s]",
            format_signature_values(state_signature$fixed_names),
            format_signature_values(backend_signature$fixed_names)
        ))
    }
    if (length(state_signature$latent_blocks) != 1L) {
        return(sprintf(
            "state latent-block count is %d; expected exactly one iid block",
            length(state_signature$latent_blocks)
        ))
    }
    if (length(backend_signature$latent_blocks) != 1L) {
        return(sprintf(
            "update latent-block count is %d; expected exactly one iid block",
            length(backend_signature$latent_blocks)
        ))
    }
    old_block <- state_signature$latent_blocks[[1L]]
    new_block <- backend_signature$latent_blocks[[1L]]
    if (!identical(old_block$model, "iid") || !identical(new_block$model, "iid")) {
        return(sprintf(
            "latent model mismatch: state uses '%s', update uses '%s'; only iid is supported",
            old_block$model,
            new_block$model
        ))
    }
    if (!identical(old_block$covariate_name, new_block$covariate_name)) {
        return(sprintf(
            "iid covariate mismatch: state uses '%s', update uses '%s'",
            old_block$covariate_name,
            new_block$covariate_name
        ))
    }
    NULL
}

same_iid_update_signature <- function(state_signature, backend_signature) {
    is.null(iid_update_signature_mismatch(state_signature, backend_signature))
}

relabel_theta_evidence_for_update_state <- function(theta_evidence, fixed_names, iid_levels) {
    if (is.null(theta_evidence) || !is.list(theta_evidence)) {
        return(NULL)
    }
    n_support <- as.integer(theta_evidence$n_support)
    if (!is.finite(n_support) || n_support <= 0L) {
        return(NULL)
    }
    n_fixed <- length(fixed_names)
    n_iid <- length(iid_levels)
    if (!identical(dim(theta_evidence$H_beta_beta), c(n_fixed, n_fixed, n_support)) ||
        !identical(dim(theta_evidence$h_beta), c(n_support, n_fixed)) ||
        !identical(dim(theta_evidence$H_u_u_diag), c(n_support, n_iid)) ||
        !identical(dim(theta_evidence$h_u), c(n_support, n_iid)) ||
        !identical(dim(theta_evidence$H_u_beta), c(n_iid, n_fixed, n_support))) {
        return(NULL)
    }

    support_names <- rownames(theta_evidence$theta)
    if (is.null(support_names) || length(support_names) != n_support) {
        support_names <- paste0("theta_", seq_len(n_support))
        support_names[[1L]] <- "source_mode"
    }
    theta_evidence$theta <- as.matrix(theta_evidence$theta)
    rownames(theta_evidence$theta) <- support_names
    dimnames(theta_evidence$H_beta_beta) <- list(fixed_names, fixed_names, support_names)
    dimnames(theta_evidence$h_beta) <- list(support_names, fixed_names)
    dimnames(theta_evidence$H_u_u_diag) <- list(support_names, iid_levels)
    dimnames(theta_evidence$h_u) <- list(support_names, iid_levels)
    dimnames(theta_evidence$H_u_beta) <- list(iid_levels, fixed_names, support_names)
    names(theta_evidence$weights) <- support_names
    names(theta_evidence$log_weights) <- support_names
    names(theta_evidence$log_constants) <- support_names
    if (!is.null(theta_evidence$log_mlik)) {
        names(theta_evidence$log_mlik) <- support_names
    }
    if (!is.null(theta_evidence$log_unnormalized_weights)) {
        names(theta_evidence$log_unnormalized_weights) <- support_names
    }
    theta_evidence
}

fixed_iid_update_state_semantics <- function(theta_evidence_policy = "single_support_point_source_mode") {
    list(
        kind = "old_data_likelihood_evidence",
        approximation_family = "local_gaussian_taylor_at_source_mode",
        prior_policy = "original_model_priors_remain_active_in_update",
        posterior_reuse = "not_posterior_as_prior",
        theta_policy = if (identical(theta_evidence_policy, "ccd_support_modes_not_integrated")) {
            "ccd_support_extracted_solver_uses_source_mode"
        } else {
            "source_theta_mode_only"
        },
        theta_evidence_policy = theta_evidence_policy,
        compatible_update_modes = c(
            "fixed_iid_gaussian_evidence",
            "fixed_iid_cross_gaussian_evidence",
            "fixed_iid_cross_theta_evidence"
        )
    )
}

validate_fixed_iid_update_state_semantics <- function(state, mode) {
    if (is.null(state$version) || length(state$version) != 1L ||
        !is.finite(as.numeric(state$version)) || as.integer(state$version) < 2L) {
        stop(
            "rusty_update_state object must be version 2 or newer; recreate it with rusty_update_state().",
            call. = FALSE
        )
    }
    semantics <- state$semantics
    if (is.null(semantics) || !is.list(semantics)) {
        stop(
            "rusty_update_state object is missing old-data evidence semantics; recreate it with rusty_update_state().",
            call. = FALSE
        )
    }
    if (!identical(semantics$kind, "old_data_likelihood_evidence")) {
        stop("rusty_update_state semantics must be old_data_likelihood_evidence.", call. = FALSE)
    }
    if (!identical(semantics$prior_policy, "original_model_priors_remain_active_in_update")) {
        stop(
            "Unsupported update-state prior policy; fixed/iid evidence states must keep original model priors active in the update.",
            call. = FALSE
        )
    }
    if (!identical(semantics$posterior_reuse, "not_posterior_as_prior")) {
        stop("rusty_update_state must store old-data evidence, not a posterior-as-prior object.", call. = FALSE)
    }
    supported_theta_evidence_policies <- c(
        "single_support_point_source_mode",
        "ccd_support_modes_not_integrated"
    )
    if (!is.null(semantics$theta_evidence_policy) &&
        !(semantics$theta_evidence_policy %in% supported_theta_evidence_policies)) {
        stop("Unsupported update-state theta evidence policy for fixed/iid evidence reuse.", call. = FALSE)
    }
    if (!(mode %in% as.character(semantics$compatible_update_modes))) {
        stop(sprintf("Update-state mode '%s' is not compatible with this state.", mode), call. = FALSE)
    }
}

flatten_support_fixed_precision <- function(arr) {
    n_support <- dim(arr)[[3L]]
    as.numeric(unlist(
        lapply(seq_len(n_support), function(s) {
            as.numeric(t(arr[, , s]))
        }),
        use.names = FALSE
    ))
}

flatten_support_rows <- function(mat) {
    as.numeric(t(mat))
}

flatten_support_latent_fixed <- function(arr) {
    n_support <- dim(arr)[[3L]]
    as.numeric(unlist(
        lapply(seq_len(n_support), function(s) {
            as.numeric(arr[, , s])
        }),
        use.names = FALSE
    ))
}

expand_theta_evidence_for_new_iid_levels <- function(theta_evidence, new_levels, level_match) {
    n_support <- as.integer(theta_evidence$n_support)
    n_fixed <- dim(theta_evidence$H_beta_beta)[[1L]]
    n_new <- length(new_levels)
    matched <- which(!is.na(level_match))

    latent_precision <- matrix(0.0, nrow = n_support, ncol = n_new)
    latent_linear <- matrix(0.0, nrow = n_support, ncol = n_new)
    latent_fixed <- array(0.0, dim = c(n_new, n_fixed, n_support))

    if (length(matched) > 0L) {
        latent_precision[, matched] <- theta_evidence$H_u_u_diag[, level_match[matched], drop = FALSE]
        latent_linear[, matched] <- theta_evidence$h_u[, level_match[matched], drop = FALSE]
        latent_fixed[matched, , ] <- theta_evidence$H_u_beta[level_match[matched], , , drop = FALSE]
    }

    list(
        latent_precision = latent_precision,
        latent_linear = latent_linear,
        latent_fixed = latent_fixed,
        n_support = n_support
    )
}

apply_fixed_iid_update_state_to_backend_spec <- function(backend_spec, family, control.update, mode) {
    state <- control.update$state
    if (is.null(state)) {
        state <- control.update$posterior_state
    }
    if (is.null(state)) {
        stop("control.update requires state for fixed_iid_gaussian_evidence.", call. = FALSE)
    }
    if (!inherits(state, "rusty_update_state")) {
        stop("control.update state must come from rusty_update_state().", call. = FALSE)
    }
    supported_approximations <- c(
        "fixed_iid_gaussian_evidence",
        "fixed_iid_cross_gaussian_evidence",
        "fixed_iid_cross_theta_evidence_composed"
    )
    if (!identical(state$scope, "fixed_iid_gaussian") ||
        !(state$approximation %in% supported_approximations)) {
        stop("Only fixed_iid Gaussian evidence update states are supported.", call. = FALSE)
    }
    include_cross <- mode %in% c("fixed_iid_cross_gaussian_evidence", "fixed_iid_cross_theta_evidence")
    use_theta_evidence <- identical(mode, "fixed_iid_cross_theta_evidence")
    validate_fixed_iid_update_state_semantics(state, mode)

    backend_signature <- build_backend_signature(backend_spec, family)
    mismatch <- iid_update_signature_mismatch(state$signature, backend_signature)
    if (!is.null(mismatch)) {
        stop(
            paste(
                sprintf("Fixed/iid update-state signature mismatch: %s.", mismatch),
                "Required contract: same family, same fixed-effect columns,",
                "same iid covariate name, and exactly one iid latent model.",
                "New iid levels are allowed and receive zero old evidence."
            ),
            call. = FALSE
        )
    }
    validate_iid_posterior_state_scope(backend_signature, length(state$theta_mode))

    fixed_precision <- as.matrix(state$fixed$evidence_precision)
    fixed_linear <- as.numeric(state$fixed$evidence_linear)
    if (nrow(fixed_precision) != backend_spec$n_fixed ||
        ncol(fixed_precision) != backend_spec$n_fixed ||
        length(fixed_linear) != backend_spec$n_fixed) {
        stop("Fixed update-state evidence dimensions do not match the new fixed-effect design.", call. = FALSE)
    }

    new_block <- backend_signature$latent_blocks[[1L]]
    new_levels <- as.character(new_block$level_values)
    if (length(new_levels) != as.integer(new_block$n_levels)) {
        new_levels <- as.character(seq_len(as.integer(new_block$n_levels)))
    }
    old_levels <- as.character(state$iid$levels)
    level_match <- match(new_levels, old_levels)
    latent_precision <- rep(0.0, length(new_levels))
    latent_linear <- rep(0.0, length(new_levels))
    matched <- which(!is.na(level_match))
    if (length(matched) > 0L) {
        latent_precision[matched] <- as.numeric(state$iid$evidence_precision_diag[level_match[matched]])
        latent_linear[matched] <- as.numeric(state$iid$evidence_linear[level_match[matched]])
    }
    latent_fixed_cross <- NULL
    if (include_cross) {
        if (is.null(state$iid_fixed_cross_precision)) {
            stop("Fixed/iid cross evidence mode requires a state with iid_fixed_cross_precision.", call. = FALSE)
        }
        cross_precision <- as.matrix(state$iid_fixed_cross_precision)
        if (nrow(cross_precision) != length(old_levels) ||
            ncol(cross_precision) != backend_spec$n_fixed) {
            stop("Fixed/iid cross evidence dimensions do not match the update-state signature.", call. = FALSE)
        }
        latent_fixed_cross <- matrix(0.0, nrow = length(new_levels), ncol = backend_spec$n_fixed)
        if (length(matched) > 0L) {
            latent_fixed_cross[matched, ] <- cross_precision[level_match[matched], , drop = FALSE]
        }
    }
    born_levels <- new_levels[is.na(level_match)]

    if (use_theta_evidence) {
        theta_evidence <- state$theta_evidence
        if (is.null(theta_evidence) || !is.list(theta_evidence) ||
            as.integer(theta_evidence$n_support) <= 1L) {
            stop(
                "fixed_iid_cross_theta_evidence requires a rusty_update_state() object with multi-point theta_evidence.",
                call. = FALSE
            )
        }
        if (length(state$theta_mode) != 1L || ncol(theta_evidence$theta) != 1L) {
            stop(
                "fixed_iid_cross_theta_evidence currently supports exactly one theta dimension.",
                call. = FALSE
            )
        }
        expanded_theta <- expand_theta_evidence_for_new_iid_levels(
            theta_evidence = theta_evidence,
            new_levels = new_levels,
            level_match = level_match
        )
        backend_spec$theta_state_n_support <- as.integer(expanded_theta$n_support)
        backend_spec$theta_state_support <- flatten_support_rows(theta_evidence$theta)
        backend_spec$theta_state_fixed_precision <- flatten_support_fixed_precision(theta_evidence$H_beta_beta)
        backend_spec$theta_state_fixed_linear <- flatten_support_rows(theta_evidence$h_beta)
        backend_spec$theta_state_latent_precision_diag <- flatten_support_rows(expanded_theta$latent_precision)
        backend_spec$theta_state_latent_linear <- flatten_support_rows(expanded_theta$latent_linear)
        backend_spec$theta_state_latent_fixed_precision <- flatten_support_latent_fixed(expanded_theta$latent_fixed)
        backend_spec$theta_state_log_constant <- as.numeric(theta_evidence$log_constants)
    } else {
        backend_spec$fixed_state_precision <- as.numeric(t(fixed_precision))
        backend_spec$fixed_state_linear <- fixed_linear
        backend_spec$latent_state_precision_diag <- latent_precision
        backend_spec$latent_state_linear <- latent_linear
        if (include_cross) {
            backend_spec$latent_fixed_state_precision <- as.numeric(latent_fixed_cross)
        }
    }
    backend_spec$posterior_update_metadata <- list(
        mode = mode,
        state_version = as.integer(state$version),
        scope = state$scope,
        approximation = state$approximation,
        evidence_semantics = state$semantics$kind,
        evidence_approximation = state$semantics$approximation_family,
        prior_policy = state$semantics$prior_policy,
        posterior_reuse = state$semantics$posterior_reuse,
        theta_policy = state$semantics$theta_policy,
        theta_evidence_policy = state$semantics$theta_evidence_policy,
        theta_evidence_strategy = if (is.null(state$theta_evidence)) NA_character_ else state$theta_evidence$strategy,
        theta_evidence_support_points = if (is.null(state$theta_evidence)) NA_integer_ else as.integer(state$theta_evidence$n_support),
        theta_evidence_solver_status = if (use_theta_evidence) {
            "linear_1d_integrated"
        } else if (is.null(state$theta_evidence)) {
            NA_character_
        } else {
            state$theta_evidence$solver_status
        },
        source_family = state$signature$family,
        source_n_obs = state$source$n_obs,
        source_fixed_names = state$fixed$names,
        source_iid_covariate_name = state$iid$covariate_name,
        born_iid_levels = born_levels,
        caveat = paste(
            "Experimental fixed+iid Gaussian old-data evidence update;",
            if (use_theta_evidence) {
                "uses linear interpolation across CCD theta evidence support with the fixed-iid cross block."
            } else if (include_cross) {
                "uses dense fixed evidence, diagonal iid evidence, and the fixed-iid cross block."
            } else {
                "uses dense fixed evidence and diagonal iid evidence, but omits the fixed-iid cross block."
            }
        )
    )
    backend_spec
}

apply_control_update_to_backend_spec <- function(backend_spec, family, control.update) {
    if (is.null(control.update)) {
        return(backend_spec)
    }
    if (!is.list(control.update)) {
        stop("control.update must be a list.", call. = FALSE)
    }

    mode <- control.update$mode
    if (is.null(mode)) {
        mode <- "iid_hyper_gaussian"
    }
    if (mode %in% c(
        "fixed_iid_gaussian_evidence",
        "fixed_iid_cross_gaussian_evidence",
        "fixed_iid_cross_theta_evidence"
    )) {
        return(apply_fixed_iid_update_state_to_backend_spec(
            backend_spec = backend_spec,
            family = family,
            control.update = control.update,
            mode = mode
        ))
    }
    if (!identical(mode, "iid_hyper_gaussian")) {
        stop(
            "control.update mode must be 'iid_hyper_gaussian', 'fixed_iid_gaussian_evidence', 'fixed_iid_cross_gaussian_evidence', or 'fixed_iid_cross_theta_evidence' for the current experiments.",
            call. = FALSE
        )
    }

    state <- control.update$posterior_state
    if (is.null(state)) {
        stop("control.update requires posterior_state.", call. = FALSE)
    }
    if (!inherits(state, "rusty_posterior_state")) {
        stop("control.update$posterior_state must come from rusty_posterior_state().", call. = FALSE)
    }
    if (!identical(state$scope, "iid_hyper") ||
        !identical(state$approximation, "gaussian_internal_theta")) {
        stop("Only iid_hyper Gaussian internal-theta posterior states are supported.", call. = FALSE)
    }

    backend_signature <- build_backend_signature(backend_spec, family)
    validate_iid_posterior_state_scope(backend_signature, length(state$theta_prior_mean))
    mismatch <- iid_update_signature_mismatch(state$signature, backend_signature)
    if (!is.null(mismatch)) {
        stop(
            paste(
                sprintf("Posterior-state update signature mismatch: %s.", mismatch),
                "Required contract: same family, same fixed-effect columns,",
                "same iid covariate name, and exactly one iid latent model."
            ),
            call. = FALSE
        )
    }

    mean <- as.numeric(state$theta_prior_mean)
    precision <- as.numeric(state$theta_prior_precision)
    mask <- as.integer(state$theta_prior_mask)
    if (length(mean) != length(precision) || length(mean) != length(mask)) {
        stop("Posterior-state prior vectors have inconsistent lengths.", call. = FALSE)
    }
    if (length(mask) == 0L || mask[[1L]] != 1L || any(mask[-1L] != 0L)) {
        stop("The iid experiment can override only the first model hyperparameter.", call. = FALSE)
    }
    if (any(!is.finite(mean)) || any(!is.finite(precision)) || any(precision[mask == 1L] <= 0)) {
        stop("Posterior-state prior mean and precision must be finite and positive.", call. = FALSE)
    }

    backend_spec$theta_prior_mean <- mean
    backend_spec$theta_prior_precision <- precision
    backend_spec$theta_prior_mask <- mask
    backend_spec$posterior_update_metadata <- list(
        mode = mode,
        scope = state$scope,
        approximation = state$approximation,
        source_family = state$signature$family,
        source_theta_names = state$theta_names,
        source_theta_prior_mean = mean,
        source_theta_prior_precision = precision,
        caveat = paste(
            "Experimental iid hyperparameter posterior-as-prior update;",
            "does not reuse fixed-effect covariance or latent random-effect state."
        )
    )
    backend_spec
}

#' Extract an experimental posterior-state object for sequential updates
#'
#' This first experimental state supports only one `iid` latent block. It
#' approximates the previous fit's internal log-precision posterior with a
#' Gaussian prior for a later `control.update` call.
#'
#' @param fit A `rusty_inla` fit.
#' @param scope Currently only `"iid_hyper"`.
#' @param min_variance Lower bound for the internal-theta posterior variance.
#' @export
rusty_posterior_state <- function(fit, scope = c("iid_hyper"), min_variance = 1e-8) {
    scope <- match.arg(scope)
    if (!inherits(fit, "rusty_inla")) {
        stop("fit must be a rusty_inla object.", call. = FALSE)
    }
    if (is.null(fit$backend_signature) || is.null(fit$internal.hyperpar)) {
        stop("fit does not contain the internal metadata needed for posterior-state extraction.", call. = FALSE)
    }

    internal <- fit$internal.hyperpar
    theta_mode <- as.numeric(internal$theta_mode)
    validate_iid_posterior_state_scope(fit$backend_signature, length(theta_mode))

    theta_matrix <- internal$ccd_thetas
    weights <- as.numeric(internal$ccd_weights)
    if (is.null(theta_matrix) || nrow(theta_matrix) == 0L || length(weights) != nrow(theta_matrix)) {
        stop(
            "fit does not contain a usable CCD theta grid for posterior-state extraction.",
            call. = FALSE
        )
    }

    first_theta <- as.numeric(theta_matrix[, 1L])
    moments <- weighted_mean_var(first_theta, weights)
    if (!is.finite(moments$mean) || !is.finite(moments$var)) {
        stop("Could not compute a finite iid hyperparameter posterior approximation.", call. = FALSE)
    }

    variance <- max(moments$var, as.numeric(min_variance))
    n_theta <- length(theta_mode)
    prior_mean <- theta_mode
    prior_precision <- rep(0.0, n_theta)
    prior_mask <- rep(0L, n_theta)
    prior_mean[[1L]] <- moments$mean
    prior_precision[[1L]] <- 1.0 / variance
    prior_mask[[1L]] <- 1L

    state <- list(
        scope = scope,
        approximation = "gaussian_internal_theta",
        signature = fit$backend_signature,
        theta_names = internal$theta_names,
        theta_mode = theta_mode,
        theta_prior_mean = prior_mean,
        theta_prior_precision = prior_precision,
        theta_prior_mask = prior_mask,
        ccd_hessian_eigenvalues = internal$ccd_hessian_eigenvalues,
        summary = list(
            iid_log_precision_mean = prior_mean[[1L]],
            iid_log_precision_sd = sqrt(variance),
            iid_log_precision_precision = prior_precision[[1L]]
        ),
        caveats = c(
            "Experimental iid hyperparameter posterior-as-prior approximation.",
            "Uses a one-dimensional Gaussian approximation on the internal log-precision scale.",
            "Does not reuse latent random effects, fixed-effect covariance, or full joint posterior state."
        )
    )
    class(state) <- "rusty_posterior_state"
    state
}

invert_spd_matrix <- function(cov, label, jitter_scale = 1e-10) {
    cov <- (as.matrix(cov) + t(as.matrix(cov))) / 2.0
    n <- nrow(cov)
    if (n == 0L) {
        return(cov)
    }
    scale <- max(abs(diag(cov)), 1.0, na.rm = TRUE)
    for (attempt in 0:8) {
        jitter <- if (attempt == 0L) 0.0 else jitter_scale * (10.0 ^ (attempt - 1L)) * scale
        candidate <- cov
        if (jitter > 0.0) {
            diag(candidate) <- diag(candidate) + jitter
        }
        chol_candidate <- tryCatch(chol(candidate), error = function(e) NULL)
        if (!is.null(chol_candidate)) {
            return(chol2inv(chol_candidate))
        }
    }
    stop(sprintf("Could not invert %s covariance for update-state extraction.", label), call. = FALSE)
}

project_psd_matrix <- function(mat, label, tolerance = 1e-7) {
    mat <- (as.matrix(mat) + t(as.matrix(mat))) / 2.0
    if (nrow(mat) == 0L) {
        return(mat)
    }
    eig <- eigen(mat, symmetric = TRUE)
    scale <- max(abs(eig$values), 1.0, na.rm = TRUE)
    min_value <- min(eig$values)
    if (min_value < -tolerance * scale) {
        stop(
            sprintf(
                "%s evidence precision is not positive semidefinite; minimum eigenvalue %.6g.",
                label,
                min_value
            ),
            call. = FALSE
        )
    }
    values <- pmax(eig$values, 0.0)
    projected <- eig$vectors %*% (values * t(eig$vectors))
    (projected + t(projected)) / 2.0
}

#' Extract an experimental fixed+iid update-state object
#'
#' This state stores old-data evidence, not a posterior-as-prior object. It
#' compresses the old likelihood into Gaussian evidence terms: a dense
#' fixed-effect block, a diagonal `iid` block, and the fixed-iid cross block.
#' The evidence is a local Taylor approximation at the old posterior mode; the
#' original model priors remain active in the later update fit.
#'
#' @param fit A `rusty_inla` fit with exactly one `iid` latent block.
#' @param scope Currently only `"fixed_iid_gaussian"`.
#' @param min_variance Lower bound for conditional variances.
#' @export
rusty_update_state <- function(fit, scope = c("fixed_iid_gaussian"), min_variance = 1e-10) {
    scope <- match.arg(scope)
    if (!inherits(fit, "rusty_inla")) {
        stop("fit must be a rusty_inla object.", call. = FALSE)
    }
    if (is.null(fit$backend_signature) || is.null(fit$internal.gaussian) ||
        is.null(fit$internal.hyperpar) || is.null(fit$internal.design) ||
        is.null(fit$mode)) {
        stop("fit does not contain the internal metadata needed for update-state extraction.", call. = FALSE)
    }
    validate_iid_posterior_state_scope(fit$backend_signature, length(fit$internal.hyperpar$theta_mode))
    if (length(fit$backend_signature$fixed_names) == 0L) {
        stop("fixed_iid_gaussian update states require at least one fixed-effect column.", call. = FALSE)
    }

    block <- fit$backend_signature$latent_blocks[[1L]]
    cov_name <- block$covariate_name
    random_df <- fit$summary.random[[cov_name]]
    if (is.null(random_df) || nrow(random_df) != as.integer(block$n_levels)) {
        stop("fit random-effect summary does not match the iid backend signature.", call. = FALSE)
    }

    evidence <- fixed_iid_likelihood_evidence(fit)

    fixed_names <- as.character(fit$backend_signature$fixed_names)
    fixed_mode <- as.numeric(fit$internal.gaussian$fixed_mode)
    fixed_cov <- as.matrix(fit$internal.gaussian$fixed_cov_theta_opt)
    if (length(fixed_mode) != length(fixed_names) ||
        nrow(fixed_cov) != length(fixed_names) ||
        ncol(fixed_cov) != length(fixed_names)) {
        stop("fit fixed-effect conditional covariance is not available for update-state extraction.", call. = FALSE)
    }
    names(fixed_mode) <- fixed_names
    dimnames(fixed_cov) <- list(fixed_names, fixed_names)

    fixed_prior_precision <- fit$internal.gaussian$fixed_prior_precision
    if (is.null(fixed_prior_precision)) {
        fixed_prior_precision <- 0.001
    }
    fixed_prior_precision <- as.numeric(fixed_prior_precision)
    fixed_evidence_precision <- as.matrix(evidence$fixed_precision)
    dimnames(fixed_evidence_precision) <- list(fixed_names, fixed_names)
    fixed_evidence_linear <- as.numeric(evidence$fixed_linear)
    names(fixed_evidence_linear) <- fixed_names

    theta_mode <- as.numeric(fit$internal.hyperpar$theta_mode)
    tau_old <- exp(theta_mode[[1L]])
    latent_mode <- as.numeric(fit$internal.gaussian$latent_mode)
    latent_var <- pmax(as.numeric(fit$internal.gaussian$latent_var_theta_opt), as.numeric(min_variance))
    if (length(latent_mode) != nrow(random_df) || length(latent_var) != nrow(random_df)) {
        stop("fit iid conditional state is not available for update-state extraction.", call. = FALSE)
    }
    iid_levels <- rownames(random_df)
    if (is.null(iid_levels)) {
        iid_levels <- as.character(random_df$ID)
    }
    iid_evidence_precision <- pmax(as.numeric(evidence$latent_precision), 0.0)
    iid_evidence_linear <- as.numeric(evidence$latent_linear)
    iid_fixed_cross_precision <- as.matrix(evidence$latent_fixed_precision)
    if (length(iid_evidence_precision) != length(iid_levels) ||
        length(iid_evidence_linear) != length(iid_levels) ||
        nrow(iid_fixed_cross_precision) != length(iid_levels) ||
        ncol(iid_fixed_cross_precision) != length(fixed_names)) {
        stop("Local likelihood evidence dimensions do not match the fit signature.", call. = FALSE)
    }
    names(iid_evidence_precision) <- iid_levels
    names(iid_evidence_linear) <- iid_levels
    dimnames(iid_fixed_cross_precision) <- list(iid_levels, fixed_names)

    theta_evidence <- relabel_theta_evidence_for_update_state(
        fit$internal.hyperpar$theta_evidence,
        fixed_names = fixed_names,
        iid_levels = iid_levels
    )
    if (is.null(theta_evidence)) {
        theta_evidence <- build_source_mode_theta_evidence(
            fit = fit,
            theta_mode = theta_mode,
            fixed_names = fixed_names,
            iid_levels = iid_levels,
            fixed_precision = fixed_evidence_precision,
            fixed_linear = fixed_evidence_linear,
            iid_precision_diag = iid_evidence_precision,
            iid_linear = iid_evidence_linear,
            iid_fixed_cross_precision = iid_fixed_cross_precision,
            log_constant = evidence$log_constant
        )
    }
    theta_evidence_policy <- if (as.integer(theta_evidence$n_support) > 1L) {
        "ccd_support_modes_not_integrated"
    } else {
        "single_support_point_source_mode"
    }

    state <- list(
        version = 3L,
        scope = scope,
        approximation = "fixed_iid_cross_gaussian_evidence",
        semantics = fixed_iid_update_state_semantics(theta_evidence_policy),
        signature = fit$backend_signature,
        source = list(
            n_obs = as.integer(fit$internal.design$n_data),
            family = fit$backend_signature$family,
            fixed_names = fixed_names,
            iid_covariate_name = cov_name,
            iid_model = block$model,
            theta_mode = theta_mode
        ),
        theta_mode = theta_mode,
        theta_evidence = theta_evidence,
        fixed = list(
            names = fixed_names,
            mode = fixed_mode,
            covariance_theta_opt = fixed_cov,
            prior_precision = fixed_prior_precision,
            evidence_precision = fixed_evidence_precision,
            evidence_linear = fixed_evidence_linear
        ),
        iid = list(
            covariate_name = cov_name,
            levels = iid_levels,
            mode = latent_mode,
            variance_theta_opt = latent_var,
            tau_at_source_theta = tau_old,
            evidence_precision_diag = iid_evidence_precision,
            evidence_linear = iid_evidence_linear,
            dropped_negative_precision = 0L
        ),
        iid_fixed_cross_precision = iid_fixed_cross_precision,
        caveats = c(
            "Experimental fixed+iid Gaussian old-data likelihood evidence approximation.",
            "Uses dense fixed-effect evidence, diagonal iid evidence, and the fixed-iid cross block.",
            if (identical(theta_evidence_policy, "ccd_support_modes_not_integrated")) {
                "Stores a Phase 8C CCD-support theta-evidence container extracted at old theta support points."
            } else {
                "Stores a Phase 8C single-point theta-evidence container at the old source theta."
            },
            "The active solver path still uses the source-mode evidence block, so hyperparameter uncertainty is still simplified.",
            "New iid levels receive zero old evidence and start from the ordinary zero-mean iid prior."
        )
    )
    class(state) <- "rusty_update_state"
    state
}

theta_evidence_support_blend <- function(theta_evidence, theta) {
    n_support <- as.integer(theta_evidence$n_support)
    theta_matrix <- as.matrix(theta_evidence$theta)
    if (n_support <= 0L || nrow(theta_matrix) != n_support) {
        stop("theta_evidence has inconsistent support dimensions.", call. = FALSE)
    }
    if (ncol(theta_matrix) != 1L || length(theta) != 1L) {
        stop("Composed update states currently support exactly one theta dimension.", call. = FALSE)
    }
    if (n_support == 1L) {
        return(list(left = 1L, right = 1L, right_weight = 0.0))
    }

    support <- as.numeric(theta_matrix[, 1L])
    order_idx <- order(support)
    theta_value <- as.numeric(theta[[1L]])
    if (theta_value <= support[order_idx[[1L]]]) {
        idx <- order_idx[[1L]]
        return(list(left = idx, right = idx, right_weight = 0.0))
    }
    if (theta_value >= support[order_idx[[n_support]]]) {
        idx <- order_idx[[n_support]]
        return(list(left = idx, right = idx, right_weight = 0.0))
    }

    for (pos in seq_len(n_support - 1L)) {
        left <- order_idx[[pos]]
        right <- order_idx[[pos + 1L]]
        theta_left <- support[[left]]
        theta_right <- support[[right]]
        if (theta_value >= theta_left && theta_value <= theta_right) {
            denom <- theta_right - theta_left
            right_weight <- if (abs(denom) <= .Machine$double.eps) {
                0.0
            } else {
                min(max((theta_value - theta_left) / denom, 0.0), 1.0)
            }
            return(list(left = left, right = right, right_weight = right_weight))
        }
    }

    distances <- abs(support - theta_value)
    idx <- which.min(distances)
    list(left = idx, right = idx, right_weight = 0.0)
}

blend_theta_array <- function(arr, blend) {
    left <- blend$left
    right <- blend$right
    right_weight <- blend$right_weight
    if (length(dim(arr)) == 3L) {
        if (identical(left, right) || right_weight <= 0.0) {
            return(arr[, , left, drop = FALSE][, , 1L])
        }
        (1.0 - right_weight) * arr[, , left] + right_weight * arr[, , right]
    } else {
        if (identical(left, right) || right_weight <= 0.0) {
            return(arr[left, ])
        }
        (1.0 - right_weight) * arr[left, ] + right_weight * arr[right, ]
    }
}

theta_evidence_block_at <- function(theta_evidence, theta, fixed_names, iid_levels) {
    blend <- theta_evidence_support_blend(theta_evidence, theta)
    old_fixed_names <- dimnames(theta_evidence$H_beta_beta)[[1L]]
    old_levels <- dimnames(theta_evidence$H_u_beta)[[1L]]
    if (!identical(as.character(old_fixed_names), as.character(fixed_names))) {
        stop("Cannot compose update states with changed fixed-effect columns.", call. = FALSE)
    }
    if (is.null(old_levels)) {
        old_levels <- seq_len(dim(theta_evidence$H_u_beta)[[1L]])
    }
    old_levels <- as.character(old_levels)
    iid_levels <- as.character(iid_levels)
    level_match <- match(iid_levels, old_levels)
    matched <- which(!is.na(level_match))
    n_iid <- length(iid_levels)
    n_fixed <- length(fixed_names)

    H_u_old <- as.numeric(blend_theta_array(theta_evidence$H_u_u_diag, blend))
    h_u_old <- as.numeric(blend_theta_array(theta_evidence$h_u, blend))
    H_u_beta_old <- as.matrix(blend_theta_array(theta_evidence$H_u_beta, blend))

    H_u <- rep(0.0, n_iid)
    h_u <- rep(0.0, n_iid)
    H_u_beta <- matrix(0.0, nrow = n_iid, ncol = n_fixed)
    if (length(matched) > 0L) {
        H_u[matched] <- H_u_old[level_match[matched]]
        h_u[matched] <- h_u_old[level_match[matched]]
        H_u_beta[matched, ] <- H_u_beta_old[level_match[matched], , drop = FALSE]
    }
    names(H_u) <- iid_levels
    names(h_u) <- iid_levels
    dimnames(H_u_beta) <- list(iid_levels, fixed_names)

    H_beta_beta <- as.matrix(blend_theta_array(theta_evidence$H_beta_beta, blend))
    h_beta <- as.numeric(blend_theta_array(theta_evidence$h_beta, blend))
    dimnames(H_beta_beta) <- list(fixed_names, fixed_names)
    names(h_beta) <- fixed_names
    log_constant <- if (identical(blend$left, blend$right) || blend$right_weight <= 0.0) {
        as.numeric(theta_evidence$log_constants[[blend$left]])
    } else {
        (1.0 - blend$right_weight) * as.numeric(theta_evidence$log_constants[[blend$left]]) +
            blend$right_weight * as.numeric(theta_evidence$log_constants[[blend$right]])
    }

    list(
        H_beta_beta = H_beta_beta,
        h_beta = h_beta,
        H_u_u_diag = H_u,
        h_u = h_u,
        H_u_beta = H_u_beta,
        log_constant = log_constant
    )
}

compose_theta_evidence <- function(previous_state, incremental_state) {
    theta_evidence <- incremental_state$theta_evidence
    previous_theta_evidence <- previous_state$theta_evidence
    if (is.null(theta_evidence) || is.null(previous_theta_evidence)) {
        stop("Composed update states require theta_evidence in both states.", call. = FALSE)
    }
    if (ncol(as.matrix(theta_evidence$theta)) != 1L ||
        ncol(as.matrix(previous_theta_evidence$theta)) != 1L) {
        stop("Composed update states currently support exactly one theta dimension.", call. = FALSE)
    }

    fixed_names <- incremental_state$fixed$names
    iid_levels <- incremental_state$iid$levels
    n_support <- as.integer(theta_evidence$n_support)
    composed <- theta_evidence
    composed$version <- 3L
    composed$strategy <- "composed_ccd_support_modes"
    composed$solver_status <- "not_integrated"

    for (support_idx in seq_len(n_support)) {
        old_block <- theta_evidence_block_at(
            theta_evidence = previous_theta_evidence,
            theta = theta_evidence$theta[support_idx, ],
            fixed_names = fixed_names,
            iid_levels = iid_levels
        )
        composed$H_beta_beta[, , support_idx] <-
            theta_evidence$H_beta_beta[, , support_idx] + old_block$H_beta_beta
        composed$h_beta[support_idx, ] <-
            theta_evidence$h_beta[support_idx, ] + old_block$h_beta
        composed$H_u_u_diag[support_idx, ] <-
            theta_evidence$H_u_u_diag[support_idx, ] + old_block$H_u_u_diag
        composed$h_u[support_idx, ] <-
            theta_evidence$h_u[support_idx, ] + old_block$h_u
        composed$H_u_beta[, , support_idx] <-
            theta_evidence$H_u_beta[, , support_idx] + old_block$H_u_beta
        composed$log_constants[[support_idx]] <-
            theta_evidence$log_constants[[support_idx]] + old_block$log_constant
    }

    composed
}

compose_source_mode_evidence <- function(previous_state, incremental_state, theta_evidence) {
    fixed_names <- incremental_state$fixed$names
    iid_levels <- incremental_state$iid$levels
    source_block <- theta_evidence_block_at(
        theta_evidence = theta_evidence,
        theta = incremental_state$theta_mode,
        fixed_names = fixed_names,
        iid_levels = iid_levels
    )
    source_block
}

validate_update_state_composition_inputs <- function(previous_state, fit, incremental_state) {
    if (!inherits(previous_state, "rusty_update_state")) {
        stop("previous_state must be a rusty_update_state object.", call. = FALSE)
    }
    if (!inherits(fit, "rusty_inla")) {
        stop("fit must be a rusty_inla object.", call. = FALSE)
    }
    if (!identical(previous_state$signature$family, incremental_state$signature$family)) {
        stop("Cannot compose update states with changed family.", call. = FALSE)
    }
    if (!identical(
        as.character(previous_state$signature$fixed_names),
        as.character(incremental_state$signature$fixed_names)
    )) {
        stop("Cannot compose update states with changed fixed-effect columns.", call. = FALSE)
    }
    if (!identical(previous_state$iid$covariate_name, incremental_state$iid$covariate_name)) {
        stop("Cannot compose update states with changed iid covariate name.", call. = FALSE)
    }
    if (length(previous_state$theta_mode) != 1L || length(incremental_state$theta_mode) != 1L) {
        stop("Composed update states currently support exactly one theta dimension.", call. = FALSE)
    }
}

#' Compose an existing fixed+iid update state with a new fitted period
#'
#' This experimental Phase 8E helper carries compressed old-data evidence
#' forward for rolling updates. It extracts likelihood evidence from `fit$data`
#' and adds it to `previous_state`, producing a new state that represents the
#' previous compressed evidence plus the current period's likelihood evidence.
#'
#' @param previous_state A `rusty_update_state` object from an earlier period.
#' @param fit A `rusty_inla` fit for the next period, usually fitted with
#'   `control.update = list(state = previous_state,
#'   mode = "fixed_iid_cross_theta_evidence")`.
#' @param min_variance Lower bound passed to `rusty_update_state()`.
#' @export
rusty_compose_update_state <- function(previous_state, fit, min_variance = 1e-10) {
    incremental_state <- rusty_update_state(fit, min_variance = min_variance)
    validate_update_state_composition_inputs(previous_state, fit, incremental_state)

    theta_evidence <- compose_theta_evidence(previous_state, incremental_state)
    source_block <- compose_source_mode_evidence(previous_state, incremental_state, theta_evidence)

    composed <- incremental_state
    composed$version <- 4L
    composed$approximation <- "fixed_iid_cross_theta_evidence_composed"
    composed$semantics <- fixed_iid_update_state_semantics("ccd_support_modes_not_integrated")
    composed$semantics$theta_policy <- "composed_ccd_support_linear_1d"
    composed$semantics$composition <- "previous_compressed_evidence_plus_current_likelihood_evidence"
    composed$signature <- incremental_state$signature
    composed$source$n_obs <- as.integer(previous_state$source$n_obs + incremental_state$source$n_obs)
    composed$source$composed_from <- list(
        previous_n_obs = as.integer(previous_state$source$n_obs),
        incremental_n_obs = as.integer(incremental_state$source$n_obs)
    )
    composed$theta_evidence <- theta_evidence
    composed$fixed$evidence_precision <- source_block$H_beta_beta
    composed$fixed$evidence_linear <- source_block$h_beta
    composed$iid$evidence_precision_diag <- source_block$H_u_u_diag
    composed$iid$evidence_linear <- source_block$h_u
    composed$iid_fixed_cross_precision <- source_block$H_u_beta
    composed$caveats <- c(
        "Experimental composed fixed+iid Gaussian old-data likelihood evidence approximation.",
        "Adds previous compressed evidence to the current fit's extracted likelihood evidence.",
        "Uses one-dimensional linear interpolation over the previous theta-evidence support.",
        "Intended for rolling diagnostics; validate against joint refits before production use."
    )
    class(composed) <- "rusty_update_state"
    composed
}

#' @export
print.rusty_posterior_state <- function(x, ...) {
    cat("Experimental rustyINLA posterior state\n")
    cat(sprintf("Scope: %s\n", x$scope))
    cat(sprintf("Approximation: %s\n", x$approximation))
    if (!is.null(x$summary)) {
        cat(sprintf(
            "iid log-precision prior: mean = %.6f, sd = %.6f\n",
            x$summary$iid_log_precision_mean,
            x$summary$iid_log_precision_sd
        ))
    }
    invisible(x)
}

#' @export
print.rusty_update_state <- function(x, ...) {
    cat("Experimental rustyINLA update state\n")
    cat(sprintf("Version: %s\n", x$version))
    cat(sprintf("Scope: %s\n", x$scope))
    cat(sprintf("Approximation: %s\n", x$approximation))
    if (!is.null(x$semantics$kind)) {
        cat(sprintf("Semantics: %s\n", x$semantics$kind))
    }
    if (!is.null(x$semantics$prior_policy)) {
        cat(sprintf("Prior policy: %s\n", x$semantics$prior_policy))
    }
    cat(sprintf("Fixed effects: %d\n", length(x$fixed$names)))
    cat(sprintf("iid block: %s with %d levels\n", x$iid$covariate_name, length(x$iid$levels)))
    if (!is.null(x$theta_evidence)) {
        cat(sprintf(
            "Theta evidence: %d support point(s), strategy = %s\n",
            as.integer(x$theta_evidence$n_support),
            x$theta_evidence$strategy
        ))
    }
    cat("Caveat: local Gaussian old-data evidence; hyperparameter uncertainty is still simplified.\n")
    invisible(x)
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

prepare_control_update_data <- function(data, control.update) {
    prepared <- list(
        data = data,
        dormant_iid_levels = character(),
        active_iid_levels = character(),
        carried_iid_levels = character()
    )
    if (is.null(control.update) || !is.list(control.update)) {
        return(prepared)
    }

    mode <- control.update$mode
    if (is.null(mode)) {
        mode <- "iid_hyper_gaussian"
    }
    if (!(mode %in% c(
        "fixed_iid_gaussian_evidence",
        "fixed_iid_cross_gaussian_evidence",
        "fixed_iid_cross_theta_evidence"
    ))) {
        return(prepared)
    }

    state <- control.update$state
    if (is.null(state)) {
        state <- control.update$posterior_state
    }
    if (!inherits(state, "rusty_update_state") || is.null(state$iid)) {
        return(prepared)
    }

    cov_name <- state$iid$covariate_name
    if (is.null(cov_name) || !(cov_name %in% names(data))) {
        return(prepared)
    }

    old_levels <- as.character(state$iid$levels)
    old_levels <- old_levels[!is.na(old_levels)]
    if (length(old_levels) == 0L) {
        return(prepared)
    }

    cov_data <- data[[cov_name]]
    observed_levels <- unique(as.character(cov_data[!is.na(cov_data)]))
    dormant_levels <- setdiff(old_levels, observed_levels)
    active_levels <- intersect(old_levels, observed_levels)

    if (!is.factor(cov_data)) {
        if (length(dormant_levels) > 0L) {
            stop(
                sprintf(
                    paste(
                        "Dormant iid level carry for '%s' requires a factor covariate.",
                        "Convert the iid covariate to factor before fitting rolling update states."
                    ),
                    cov_name
                ),
                call. = FALSE
            )
        }
        return(prepared)
    }

    current_levels <- levels(cov_data)
    union_levels <- unique(c(old_levels, current_levels))
    carried_levels <- setdiff(old_levels, current_levels)
    if (!identical(current_levels, union_levels)) {
        prepared$data[[cov_name]] <- factor(as.character(cov_data), levels = union_levels)
    }

    prepared$dormant_iid_levels <- dormant_levels
    prepared$active_iid_levels <- active_levels
    prepared$carried_iid_levels <- carried_levels
    prepared
}

#' Native R Formula Interface for Rusty-INLA
#'
#' @param formula A robust R formula such as
#'   `y ~ 1 + x1 * factor_col + offset(log_exposure) + f(cov, model="iid")`.
#'   The current fixed-effects subset supports bare data columns and simple
#'   interactions among them, with or without standalone latent `f(...)` terms;
#'   create transformed columns in `data` first.
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
#' @param control.update Experimental posterior-state update control. The
#'   current Phase 8 experiments support
#'   `list(posterior_state = rusty_posterior_state(fit), mode = "iid_hyper_gaussian")`
#'   and `list(state = rusty_update_state(fit), mode = "fixed_iid_cross_gaussian_evidence")`.
#'   The opt-in Phase 8D path
#'   `mode = "fixed_iid_cross_theta_evidence"` linearly interpolates the
#'   extracted CCD theta-evidence blocks for one-dimensional iid updates.
#' @export
rusty_inla <- function(
    formula,
    data,
    family,
    offset = NULL,
    output_profile = c("thin", "benchmark"),
    control.update = NULL
) {
    fit_start_time <- safe_elapsed_time()
    output_profile <- match.arg(output_profile)
    update_data <- prepare_control_update_data(data, control.update)
    data <- update_data$data
    backend_spec <- build_backend_spec(
        formula,
        data,
        family,
        offset = offset,
        offset_expr = substitute(offset),
        offset_env = parent.frame(),
        offset_provided = !missing(offset)
    )
    backend_spec <- apply_control_update_to_backend_spec(
        backend_spec = backend_spec,
        family = family,
        control.update = control.update
    )
    if (!is.null(backend_spec$posterior_update_metadata)) {
        backend_spec$posterior_update_metadata$dormant_iid_levels <- update_data$dormant_iid_levels
        backend_spec$posterior_update_metadata$active_iid_levels <- update_data$active_iid_levels
        backend_spec$posterior_update_metadata$carried_iid_levels <- update_data$carried_iid_levels
    }
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
        output_profile = output_profile,
        backend_signature = build_backend_signature(backend_spec, family),
        internal.hyperpar = build_internal_hyperparameter_state(
            res = res,
            backend_spec = backend_spec,
            family = family
        ),
        internal.design = build_internal_design(backend_spec, nrow(data))
    )
    if (!is.null(backend_spec$posterior_update_metadata)) {
        fit$posterior_state_used <- backend_spec$posterior_update_metadata
    }

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
    fit$internal.gaussian <- list(
        fixed_mode = if (is.null(res$mode_beta)) numeric() else as.numeric(res$mode_beta),
        fixed_cov_theta_opt = matrix(
            if (is.null(res$fixed_cov_theta_opt)) numeric() else as.numeric(res$fixed_cov_theta_opt),
            nrow = backend_spec$n_fixed,
            ncol = backend_spec$n_fixed,
            byrow = TRUE,
            dimnames = list(backend_spec$fixed_names, backend_spec$fixed_names)
        ),
        fixed_prior_precision = 0.001,
        latent_mode = if (is.null(res$mode_x)) numeric() else as.numeric(res$mode_x),
        latent_var_theta_opt = if (is.null(res$latent_var_theta_opt)) numeric() else as.numeric(res$latent_var_theta_opt)
    )

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
