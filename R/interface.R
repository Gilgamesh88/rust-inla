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

extract_formula_offset <- function(tf, data) {
    offset_idx <- attr(tf, "offset")
    if (is.null(offset_idx) || length(offset_idx) == 0L) {
        return(NULL)
    }

    variables <- attr(tf, "variables")
    formula_env <- attr(tf, ".Environment")
    if (is.null(formula_env)) {
        formula_env <- parent.frame()
    }

    offsets <- lapply(offset_idx, function(idx) {
        expr <- formula_variable_expr(variables, idx)
        if (!is.call(expr) || !identical(expr[[1L]], as.name("offset")) || length(expr) != 2L) {
            stop("Formula offset must be an offset(...) call.", call. = FALSE)
        }
        value <- tryCatch(
            eval(expr[[2L]], envir = data, enclos = formula_env),
            error = function(e) {
                stop(
                    sprintf("Could not evaluate formula offset: %s", e$message),
                    call. = FALSE
                )
            }
        )
        if (!is.numeric(value)) {
            stop("Formula offset must evaluate to a numeric vector.", call. = FALSE)
        }
        value <- as.numeric(value)
        if (length(value) != nrow(data)) {
            stop("Formula offset length does not match the number of observations.")
        }
        if (any(!is.finite(value))) {
            stop("Formula offset contains non-finite values.", call. = FALSE)
        }
        value
    })

    Reduce(`+`, offsets)
}

drop_latent_terms_for_fixed_design <- function(tf, f_term_idx) {
    if (length(f_term_idx) == 0L) {
        return(delete.response(tf))
    }

    term_labels <- attr(tf, "term.labels")
    keep_idx <- setdiff(seq_along(term_labels), f_term_idx)
    if (length(keep_idx) > 0L) {
        return(drop.terms(tf, f_term_idx, keep.response = FALSE))
    }

    fixed_formula <- if (identical(attr(tf, "intercept"), 0L)) {
        stats::as.formula("~ 0")
    } else {
        stats::as.formula("~ 1")
    }
    attr(fixed_formula, ".Environment") <- attr(tf, ".Environment")
    terms(fixed_formula)
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
    tf_fixed <- drop_latent_terms_for_fixed_design(tf, f_term_idx)

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
    formula_offset <- extract_formula_offset(tf, data)
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
                constr = constr,
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
                constr = isTRUE(block$constr),
                n_levels = as.integer(block$n_levels),
                start = as.integer(block$start),
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

build_latent_level_metadata <- function(backend_spec) {
    if (length(backend_spec$latent_blocks) == 0L) {
        return(data.frame(
            index = integer(),
            covariate_name = character(),
            level = character(),
            latent_name = character(),
            check.names = FALSE
        ))
    }

    rows <- lapply(backend_spec$latent_blocks, function(block) {
        nl <- as.integer(block$n_levels)
        levels <- if (!is.null(block$level_values) && length(block$level_values) == nl) {
            as.character(block$level_values)
        } else {
            as.character(seq_len(nl))
        }
        data.frame(
            index = as.integer(block$start) + seq_len(nl),
            covariate_name = block$covariate_name,
            level = levels,
            latent_name = paste(block$covariate_name, levels, sep = ":"),
            check.names = FALSE
        )
    })
    do.call(rbind, rows)
}

build_latent_pair_covariance <- function(res, latent_names) {
    pair_i <- if (is.null(res$latent_pair_cov_i)) integer() else as.integer(res$latent_pair_cov_i)
    pair_j <- if (is.null(res$latent_pair_cov_j)) integer() else as.integer(res$latent_pair_cov_j)
    pair_cov <- if (is.null(res$latent_pair_cov)) numeric() else as.numeric(res$latent_pair_cov)
    n_pair <- min(length(pair_i), length(pair_j), length(pair_cov))
    if (n_pair == 0L) {
        return(data.frame(
            i = integer(),
            j = integer(),
            latent_i = character(),
            latent_j = character(),
            covariance = numeric(),
            check.names = FALSE
        ))
    }
    pair_i <- pair_i[seq_len(n_pair)] + 1L
    pair_j <- pair_j[seq_len(n_pair)] + 1L
    data.frame(
        i = pair_i,
        j = pair_j,
        latent_i = latent_names[pair_i],
        latent_j = latent_names[pair_j],
        covariance = pair_cov[seq_len(n_pair)],
        check.names = FALSE
    )
}

likelihood_theta_count <- function(family) {
    switch(
        as.character(family)[[1L]],
        gaussian = 1L,
        poisson = 0L,
        gamma = 1L,
        zeroinflatedpoisson1 = 1L,
        tweedie = 2L,
        stop(sprintf("Unsupported family '%s'.", as.character(family)[[1L]]), call. = FALSE)
    )
}

latent_block_theta_count <- function(model) {
    switch(
        model,
        iid = 1L,
        rw1 = 1L,
        rw2 = 1L,
        ar1 = 2L,
        ar2 = 3L,
        stop(sprintf("Unsupported latent model '%s'.", model), call. = FALSE)
    )
}

expected_theta_count <- function(backend_spec, family) {
    latent_count <- sum(vapply(
        backend_spec$latent_blocks,
        function(block) latent_block_theta_count(block$model),
        integer(1)
    ))
    as.integer(latent_count + likelihood_theta_count(family))
}

theta_names_for_backend <- function(backend_spec, family) {
    n_theta <- expected_theta_count(backend_spec, family)
    if (n_theta == 0L) {
        return(character())
    }
    specs <- resolve_hyperparameter_specs(backend_spec, family, n_theta)
    vapply(specs, `[[`, character(1), "name")
}

logical_control_value <- function(value, field, default = NULL) {
    if (is.null(value)) {
        return(default)
    }
    if (!is.logical(value) || length(value) != 1L || is.na(value)) {
        stop(sprintf("%s must be a single TRUE/FALSE value.", field), call. = FALSE)
    }
    value
}

nonnegative_int_control_value <- function(value, field, default = NULL) {
    if (is.null(value)) {
        return(default)
    }
    if (!is.numeric(value) || length(value) != 1L || is.na(value) ||
        !is.finite(value) || value < 0 || value != floor(value)) {
        stop(sprintf("%s must be a single non-negative integer.", field), call. = FALSE)
    }
    as.integer(value)
}

normalize_control_compute <- function(control.compute) {
    defaults <- list(
        config = FALSE,
        diagnostics = TRUE,
        internal = TRUE,
        theta.internal = TRUE,
        evidence = TRUE,
        skip.ccd = NULL,
        optimizer.max.evals = NULL,
        ignored = character()
    )
    if (is.null(control.compute)) {
        return(defaults)
    }
    if (!is.list(control.compute)) {
        stop("control.compute must be a list.", call. = FALSE)
    }

    known <- c(
        "config",
        "diagnostics",
        "internal",
        "theta.internal",
        "evidence",
        "skip.ccd",
        "optimizer.max.evals"
    )
    out <- defaults
    out$config <- logical_control_value(control.compute$config, "control.compute$config", out$config)
    out$diagnostics <- logical_control_value(
        control.compute$diagnostics,
        "control.compute$diagnostics",
        out$diagnostics
    )
    out$internal <- logical_control_value(control.compute$internal, "control.compute$internal", out$internal)
    out$theta.internal <- logical_control_value(
        control.compute$theta.internal,
        "control.compute$theta.internal",
        out$theta.internal
    )
    out$evidence <- logical_control_value(control.compute$evidence, "control.compute$evidence", out$evidence)
    out$skip.ccd <- logical_control_value(control.compute$skip.ccd, "control.compute$skip.ccd")
    out$optimizer.max.evals <- nonnegative_int_control_value(
        control.compute$optimizer.max.evals,
        "control.compute$optimizer.max.evals"
    )
    out$ignored <- setdiff(names(control.compute), known)
    out
}

extract_control_theta <- function(theta, n_theta, theta_names, field) {
    if (is.null(theta)) {
        return(NULL)
    }
    if (inherits(theta, "rusty_inla")) {
        theta <- theta$mode$theta
    } else if (is.list(theta) && !is.null(theta$theta)) {
        theta <- theta$theta
    }
    if (!is.numeric(theta)) {
        stop(sprintf("%s must be a numeric internal-theta vector.", field), call. = FALSE)
    }
    theta <- as.numeric(theta)
    if (length(theta) != n_theta) {
        stop(
            sprintf("%s length %d does not match expected theta length %d.", field, length(theta), n_theta),
            call. = FALSE
        )
    }
    if (any(!is.finite(theta))) {
        stop(sprintf("%s must contain only finite values.", field), call. = FALSE)
    }
    stats::setNames(theta, theta_names)
}

normalize_control_mode <- function(control.mode, n_theta, theta_names) {
    defaults <- list(
        theta = numeric(),
        theta_provided = FALSE,
        restart = TRUE,
        fixed = FALSE,
        skip.ccd = NULL,
        optimizer.max.evals = NULL
    )
    if (is.null(control.mode)) {
        return(defaults)
    }
    if (!is.list(control.mode)) {
        stop("control.mode must be a list.", call. = FALSE)
    }

    known <- c("theta", "restart", "fixed", "skip.ccd", "optimizer.max.evals")
    unknown <- setdiff(names(control.mode), known)
    if (length(unknown) > 0L) {
        stop(
            sprintf("Unsupported control.mode field(s): %s.", paste(unknown, collapse = ", ")),
            call. = FALSE
        )
    }

    out <- defaults
    theta_value <- extract_control_theta(
        theta = control.mode$theta,
        n_theta = n_theta,
        theta_names = theta_names,
        field = "control.mode$theta"
    )
    out[["theta"]] <- if (is.null(theta_value)) numeric() else theta_value
    out[["theta_provided"]] <- !is.null(theta_value)
    out[["restart"]] <- logical_control_value(control.mode$restart, "control.mode$restart", out[["restart"]])
    out[["fixed"]] <- logical_control_value(control.mode$fixed, "control.mode$fixed", out[["fixed"]])
    out[["skip.ccd"]] <- logical_control_value(control.mode$skip.ccd, "control.mode$skip.ccd")
    out[["optimizer.max.evals"]] <- nonnegative_int_control_value(
        control.mode$optimizer.max.evals,
        "control.mode$optimizer.max.evals"
    )
    if (isTRUE(out[["fixed"]])) {
        out[["restart"]] <- FALSE
    }
    if (!isTRUE(out[["restart"]]) && !out[["theta_provided"]]) {
        stop("control.mode with restart = FALSE requires control.mode$theta.", call. = FALSE)
    }
    out
}

apply_engine_controls_to_backend_spec <- function(
    backend_spec,
    family,
    output_profile,
    control.mode,
    control.compute,
    control.update
) {
    n_theta <- expected_theta_count(backend_spec, family)
    theta_names <- theta_names_for_backend(backend_spec, family)
    compute <- normalize_control_compute(control.compute)
    mode <- normalize_control_mode(control.mode, n_theta, theta_names)

    if (!is.null(compute$skip.ccd)) {
        backend_spec$skip_ccd <- compute$skip.ccd
    }
    if (!is.null(compute$optimizer.max.evals)) {
        backend_spec$optimizer_max_evals <- compute$optimizer.max.evals
    }
    if (mode$theta_provided) {
        backend_spec$theta_init <- as.numeric(mode$theta)
    }
    if (!is.null(mode$skip.ccd)) {
        backend_spec$skip_ccd <- mode$skip.ccd
    }
    if (!is.null(mode$optimizer.max.evals)) {
        backend_spec$optimizer_max_evals <- mode$optimizer.max.evals
    }

    fixed_theta_replay <- !isTRUE(mode$restart)
    if (fixed_theta_replay) {
        backend_spec$optimizer_max_evals <- 0L
        if (is.null(mode$skip.ccd) && is.null(compute$skip.ccd)) {
            backend_spec$skip_ccd <- TRUE
        }
    }

    backend_spec$control_metadata <- list(
        output_profile = output_profile,
        update_mode = if (is.null(control.update) || is.null(control.update$mode)) {
            if (is.null(control.update)) NULL else "iid_hyper_gaussian"
        } else {
            control.update$mode
        },
        mode = list(
            theta_provided = mode$theta_provided,
            theta = mode$theta,
            restart = mode$restart,
            fixed_theta_replay = fixed_theta_replay,
            optimizer_max_evals = backend_spec$optimizer_max_evals,
            skip_ccd = isTRUE(backend_spec$skip_ccd),
            theta_names = theta_names
        ),
        compute = compute,
        engine = list(
            optimizer_max_evals = backend_spec$optimizer_max_evals,
            skip_ccd = isTRUE(backend_spec$skip_ccd)
        )
    )
    backend_spec
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

build_internal_hyperparameter_summary <- function(res, backend_spec, family) {
    theta_opt <- if (is.null(res$theta_opt)) numeric() else as.numeric(res$theta_opt)
    n_theta <- length(theta_opt)
    if (n_theta == 0) {
        return(data.frame())
    }

    specs <- resolve_hyperparameter_specs(backend_spec, family, n_theta)
    theta_names <- vapply(specs, `[[`, character(1), "name")
    ccd_weights <- if (is.null(res$ccd_weights)) numeric() else as.numeric(res$ccd_weights)
    ccd_thetas <- if (is.null(res$ccd_thetas)) numeric() else as.numeric(res$ccd_thetas)

    if (length(ccd_weights) == 0 || length(ccd_thetas) == 0 ||
        length(ccd_thetas) %% n_theta != 0L) {
        out <- point_hyperparameter_summary(theta_opt, theta_names)
        attr(out, "internal_scale") <- TRUE
        return(out)
    }

    theta_matrix <- matrix(ccd_thetas, ncol = n_theta, byrow = TRUE)
    if (nrow(theta_matrix) != length(ccd_weights)) {
        out <- point_hyperparameter_summary(theta_opt, theta_names)
        attr(out, "internal_scale") <- TRUE
        return(out)
    }

    means <- numeric(n_theta)
    sds <- numeric(n_theta)
    q025 <- numeric(n_theta)
    q500 <- numeric(n_theta)
    q975 <- numeric(n_theta)

    for (idx in seq_len(n_theta)) {
        vals <- theta_matrix[, idx]
        keep <- is.finite(vals) & is.finite(ccd_weights) & (ccd_weights > 0)
        if (!any(keep)) {
            means[[idx]] <- theta_opt[[idx]]
            sds[[idx]] <- NA_real_
            q025[[idx]] <- NA_real_
            q500[[idx]] <- theta_opt[[idx]]
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

    out <- data.frame(
        row.names = theta_names,
        mean = means,
        sd = sds,
        `0.025quant` = q025,
        `0.5quant` = q500,
        `0.975quant` = q975,
        mode = theta_opt,
        check.names = FALSE
    )
    attr(out, "internal_scale") <- TRUE
    out
}

default_theta_prior_labels <- function(backend_spec, family) {
    labels <- character()
    for (block in backend_spec$latent_blocks) {
        if (block$model %in% c("iid", "rw1", "rw2")) {
            labels <- c(labels, "loggamma(shape = 1, rate = 5e-5) on log precision")
        } else if (identical(block$model, "ar1")) {
            labels <- c(
                labels,
                "loggamma(shape = 1, rate = 5e-5) on log precision",
                "gaussian(mean = 0, precision = 0.15) on internal correlation"
            )
        } else if (identical(block$model, "ar2")) {
            labels <- c(
                labels,
                "loggamma(shape = 1, rate = 5e-5) on log precision",
                "pc_cor0(u = 0.5, alpha = 0.5) on internal PACF1",
                "pc_cor0(u = 0.5, alpha = 0.4) on internal PACF2"
            )
        }
    }

    labels <- c(labels, switch(
        as.character(family)[[1L]],
        gaussian = "loggamma(shape = 1, rate = 5e-5) on observation log precision",
        poisson = character(),
        gamma = "loggamma(shape = 1, rate = 0.01) on log shape",
        zeroinflatedpoisson1 = "gaussian(mean = -1, precision = 0.2) on zero-inflation logit",
        tweedie = c(
            "loggamma(shape = 1, rate = 5e-5) on log dispersion",
            "loggamma(shape = 1, rate = 5e-5) on internal power"
        ),
        character()
    ))
    labels
}

build_prior_metadata <- function(backend_spec, family) {
    theta_names <- theta_names_for_backend(backend_spec, family)
    n_theta <- length(theta_names)
    prior_labels <- default_theta_prior_labels(backend_spec, family)
    if (length(prior_labels) != n_theta) {
        prior_labels <- rep("backend default prior on internal theta", n_theta)
    }

    mask <- if (is.null(backend_spec$theta_prior_mask)) {
        integer(n_theta)
    } else {
        as.integer(backend_spec$theta_prior_mask)
    }
    override_mean <- rep(NA_real_, n_theta)
    override_precision <- rep(NA_real_, n_theta)
    if (!is.null(backend_spec$theta_prior_mean) && length(backend_spec$theta_prior_mean) == n_theta) {
        override_mean <- as.numeric(backend_spec$theta_prior_mean)
    }
    if (!is.null(backend_spec$theta_prior_precision) &&
        length(backend_spec$theta_prior_precision) == n_theta) {
        override_precision <- as.numeric(backend_spec$theta_prior_precision)
    }

    theta <- data.frame(
        row.names = theta_names,
        name = theta_names,
        internal_scale = rep(TRUE, n_theta),
        default_prior = prior_labels,
        override = mask == 1L,
        override_mean = override_mean,
        override_precision = override_precision,
        check.names = FALSE
    )

    list(
        fixed = list(
            names = backend_spec$fixed_names,
            mean = rep(0.0, length(backend_spec$fixed_names)),
            precision = rep(0.001, length(backend_spec$fixed_names)),
            source = "backend fixed-effect Gaussian prior"
        ),
        latent = lapply(backend_spec$latent_blocks, function(block) {
            list(
                covariate_name = block$covariate_name,
                model = block$model,
                constr = isTRUE(block$constr),
                n_levels = as.integer(block$n_levels),
                level_values = block$level_values,
                structure_values = block$structure_values
            )
        }),
        theta = theta,
        update = backend_spec$posterior_update_metadata
    )
}

build_theta_internal_output <- function(internal, res, control_metadata) {
    if (is.null(internal)) {
        return(NULL)
    }
    theta_names <- as.character(internal$theta_names)
    theta_mode <- stats::setNames(as.numeric(internal$theta_mode), theta_names)
    theta_initial <- if (is.null(res$theta_init_used)) {
        rep(NA_real_, length(theta_mode))
    } else {
        as.numeric(res$theta_init_used)
    }
    names(theta_initial) <- theta_names

    ccd_thetas <- internal$ccd_thetas
    support_names <- character()
    if (!is.null(ccd_thetas) && length(ccd_thetas) > 0L) {
        support_names <- rownames(ccd_thetas)
        if (is.null(support_names)) {
            support_names <- paste0("theta_", seq_len(nrow(ccd_thetas)))
            support_names[[1L]] <- "source_mode"
            rownames(ccd_thetas) <- support_names
        }
        colnames(ccd_thetas) <- theta_names
    }

    weights <- as.numeric(internal$ccd_weights)
    if (length(weights) > 0L && length(weights) == length(support_names)) {
        names(weights) <- support_names
    }

    list(
        names = theta_names,
        mode = theta_mode,
        initial = theta_initial,
        ccd_thetas = ccd_thetas,
        ccd_weights = weights,
        ccd_log_mlik = internal$ccd_log_mlik,
        ccd_log_weight = internal$ccd_log_weight,
        hessian_eigenvalues = internal$ccd_hessian_eigenvalues,
        theta_evidence = internal$theta_evidence,
        internal_scale = TRUE,
        control = control_metadata$mode
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
        control.compute = backend_spec$control_metadata$compute,
        control.mode = backend_spec$control_metadata$mode,
        control.predictor = list(compute = TRUE),
        output_profile = output_profile,
        n_fixed = backend_spec$n_fixed,
        n_latent = backend_spec$n_latent,
        latent_blocks = backend_spec$latent_blocks
    )
}

effective_support_size <- function(weights) {
    weights <- as.numeric(weights)
    keep <- is.finite(weights) & weights > 0
    if (!any(keep)) {
        return(NA_real_)
    }
    weights <- weights[keep] / sum(weights[keep])
    1.0 / sum(weights^2)
}

build_accuracy_diagnostics <- function(fit, res, backend_spec) {
    diagnostics <- if (is.null(fit$diagnostics)) list() else fit$diagnostics
    internal <- fit$internal.hyperpar
    theta_weights <- if (is.null(internal)) numeric() else as.numeric(internal$ccd_weights)
    theta_support <- if (is.null(internal) || is.null(internal$ccd_thetas)) {
        0L
    } else {
        nrow(internal$ccd_thetas)
    }
    theta_evidence <- if (is.null(internal)) NULL else internal$theta_evidence
    control <- backend_spec$control_metadata

    list(
        version = 1L,
        purpose = "pre_sequential_validation_accuracy_gate",
        theta = list(
            n_theta = as.integer(fit$nhyper),
            names = if (is.null(fit$theta.internal)) character() else fit$theta.internal$names,
            mode = if (is.null(fit$theta.internal)) numeric() else fit$theta.internal$mode,
            initial = if (is.null(fit$theta.internal)) numeric() else fit$theta.internal$initial,
            fixed_theta_replay = isTRUE(control$mode$fixed_theta_replay),
            optimizer_max_evals = control$engine$optimizer_max_evals,
            skip_ccd = isTRUE(control$engine$skip_ccd),
            ccd_support_points = as.integer(theta_support),
            ccd_weight_sum = if (length(theta_weights) == 0L) NA_real_ else sum(theta_weights),
            ccd_effective_support = effective_support_size(theta_weights),
            hessian_eigenvalues = if (is.null(internal)) numeric() else internal$ccd_hessian_eigenvalues
        ),
        solver = list(
            optimizer_outer_iterations = diagnostics$optimizer_outer_iterations,
            laplace_eval_calls_total = diagnostics$laplace_eval_calls_total,
            laplace_eval_calls_optimizer = diagnostics$laplace_eval_calls_optimizer,
            laplace_eval_calls_ccd = diagnostics$laplace_eval_calls_ccd,
            latent_mode_solve_calls = diagnostics$latent_mode_solve_calls,
            latent_mode_iterations_total = diagnostics$latent_mode_iterations_total,
            latent_mode_max_iter_hits = diagnostics$latent_mode_max_iter_hits,
            latent_mode_restarts = diagnostics$latent_mode_restarts,
            latent_mode_step_factor_min = diagnostics$latent_mode_step_factor_min,
            factorization_count = diagnostics$factorization_count,
            selected_inverse_count = diagnostics$selected_inverse_count
        ),
        timing = list(
            optimizer_time_sec = diagnostics$optimizer_time_sec,
            ccd_time_sec = diagnostics$ccd_time_sec,
            latent_mode_solve_time_sec = diagnostics$latent_mode_solve_time_sec,
            likelihood_assembly_time_sec = diagnostics$likelihood_assembly_time_sec,
            sparse_factorization_time_sec = diagnostics$sparse_factorization_time_sec,
            selected_inverse_time_sec = diagnostics$selected_inverse_time_sec
        ),
        evidence = list(
            posterior_update = backend_spec$posterior_update_metadata,
            theta_evidence_strategy = if (is.null(theta_evidence)) NA_character_ else theta_evidence$strategy,
            theta_evidence_solver_status = if (is.null(theta_evidence)) NA_character_ else theta_evidence$solver_status,
            theta_evidence_support_points = if (is.null(theta_evidence)) 0L else as.integer(theta_evidence$n_support),
            laplace_terms = fit$laplace_terms
        ),
        flags = list(
            optimizer_hit_eval_budget = isTRUE(!is.null(control$engine$optimizer_max_evals) &&
                !is.null(diagnostics$optimizer_outer_iterations) &&
                diagnostics$optimizer_outer_iterations >= control$engine$optimizer_max_evals &&
                control$engine$optimizer_max_evals > 0L),
            latent_mode_hit_max_iter = isTRUE(!is.null(diagnostics$latent_mode_max_iter_hits) &&
                diagnostics$latent_mode_max_iter_hits > 0L),
            ccd_disabled = isTRUE(control$engine$skip_ccd),
            fixed_theta_replay = isTRUE(control$mode$fixed_theta_replay)
        )
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
        carried_iid_levels = character(),
        dormant_iid_levels_by_block = list(),
        active_iid_levels_by_block = list(),
        carried_iid_levels_by_block = list()
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
    if (!inherits(state, "rusty_update_state") || length(state_iid_blocks(state)) == 0L) {
        return(prepared)
    }

    for (block in state_iid_blocks(state)) {
        cov_name <- block$covariate_name
        if (is.null(cov_name) || !(cov_name %in% names(prepared$data))) {
            next
        }

        old_levels <- as.character(block$levels)
        old_levels <- old_levels[!is.na(old_levels)]
        if (length(old_levels) == 0L) {
            next
        }

        cov_data <- prepared$data[[cov_name]]
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
            next
        }

        current_levels <- levels(cov_data)
        union_levels <- unique(c(old_levels, current_levels))
        carried_levels <- setdiff(old_levels, current_levels)
        if (!identical(current_levels, union_levels)) {
            prepared$data[[cov_name]] <- factor(as.character(cov_data), levels = union_levels)
        }

        prepared$dormant_iid_levels_by_block[[cov_name]] <- dormant_levels
        prepared$active_iid_levels_by_block[[cov_name]] <- active_levels
        prepared$carried_iid_levels_by_block[[cov_name]] <- carried_levels
        prepared$dormant_iid_levels <- unique(c(prepared$dormant_iid_levels, dormant_levels))
        prepared$active_iid_levels <- unique(c(prepared$active_iid_levels, active_levels))
        prepared$carried_iid_levels <- unique(c(prepared$carried_iid_levels, carried_levels))
    }
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
#'   extracted CCD theta-evidence blocks for one-dimensional iid updates and
#'   uses guarded local interpolation with nearest-support fallback for
#'   multi-iid theta evidence.
#' @param control.mode Optional internal-theta starting or replay control.
#'   Use `list(theta = fit$theta.internal$mode, restart = FALSE)` to evaluate
#'   the same model at a fixed internal theta before sequential diagnostics.
#' @param control.compute Optional diagnostic/output controls. The current
#'   supported fields are `diagnostics`, `internal`, `theta.internal`,
#'   `evidence`, `skip.ccd`, and `optimizer.max.evals`.
#' @export
rusty_inla <- function(
    formula,
    data,
    family,
    offset = NULL,
    output_profile = c("thin", "benchmark"),
    control.update = NULL,
    control.mode = NULL,
    control.compute = NULL
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
        backend_spec$posterior_update_metadata$dormant_iid_levels_by_block <- update_data$dormant_iid_levels_by_block
        backend_spec$posterior_update_metadata$active_iid_levels_by_block <- update_data$active_iid_levels_by_block
        backend_spec$posterior_update_metadata$carried_iid_levels_by_block <- update_data$carried_iid_levels_by_block
    }
    backend_spec <- apply_engine_controls_to_backend_spec(
        backend_spec = backend_spec,
        family = family,
        output_profile = output_profile,
        control.mode = control.mode,
        control.compute = control.compute,
        control.update = control.update
    )
    after_spec_time <- safe_elapsed_time()

    # 4. Invoke the Rust Core
    res <- rust_inla_run(backend_spec)
    after_run_time <- safe_elapsed_time()

    # Error handling from backend
    if (is.character(res)) { stop(res) }

    # 5. Build Standard Output Structure matching R-INLA expectations
    latent_metadata <- build_latent_level_metadata(backend_spec)
    latent_names <- latent_metadata$latent_name
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
        control.used = backend_spec$control_metadata,
        prior.used = build_prior_metadata(backend_spec, family),
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
        fit$summary.hyperpar.internal <- build_internal_hyperparameter_summary(
            res = res,
            backend_spec = backend_spec,
            family = family
        )
    } else {
        fit$summary.hyperpar <- data.frame()
        fit$summary.hyperpar.internal <- data.frame()
    }
    fit$theta.internal <- build_theta_internal_output(
        internal = fit$internal.hyperpar,
        res = res,
        control_metadata = backend_spec$control_metadata
    )

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
        fixed_cov = matrix(
            if (is.null(res$fixed_cov)) numeric() else as.numeric(res$fixed_cov),
            nrow = backend_spec$n_fixed,
            ncol = backend_spec$n_fixed,
            byrow = TRUE,
            dimnames = list(backend_spec$fixed_names, backend_spec$fixed_names)
        ),
        fixed_prior_precision = 0.001,
        latent_mode = if (is.null(res$mode_x)) numeric() else as.numeric(res$mode_x),
        latent_var_theta_opt = if (is.null(res$latent_var_theta_opt)) numeric() else as.numeric(res$latent_var_theta_opt),
        latent_fixed_cov = matrix(
            if (is.null(res$latent_fixed_cov)) numeric() else as.numeric(res$latent_fixed_cov),
            nrow = backend_spec$n_latent,
            ncol = backend_spec$n_fixed,
            dimnames = list(latent_names, backend_spec$fixed_names)
        ),
        latent_pair_cov = build_latent_pair_covariance(res, latent_names)
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
    fit$accuracy.diagnostics <- build_accuracy_diagnostics(
        fit = fit,
        res = res,
        backend_spec = backend_spec
    )
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
