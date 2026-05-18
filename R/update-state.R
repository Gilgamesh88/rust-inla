# Phase 8 posterior-state and update-state helpers.
#
# Kept separate from the public rusty_inla() interface so the sequential
# update machinery can evolve without making R/interface.R carry every
# backend state and theta-evidence helper.

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

validate_fixed_iid_evidence_state_scope <- function(signature, n_theta) {
    blocks <- signature$latent_blocks
    if (length(blocks) < 1L || !all(vapply(
        blocks,
        function(block) identical(block$model, "iid"),
        logical(1)
    ))) {
        stop(
            paste(
                "Fixed/iid evidence updates currently support one or more",
                "iid latent blocks and no other latent model types."
            ),
            call. = FALSE
        )
    }
    if (n_theta < length(blocks)) {
        stop("Fixed/iid evidence updates require one iid hyperparameter per iid block.", call. = FALSE)
    }
    invisible(TRUE)
}

latent_block_ranges <- function(signature) {
    blocks <- signature$latent_blocks
    next_start <- 0L
    lapply(blocks, function(block) {
        n_levels <- as.integer(block$n_levels)
        start <- if (!is.null(block$start) && length(block$start) == 1L && !is.na(block$start)) {
            as.integer(block$start)
        } else {
            next_start
        }
        next_start <<- start + n_levels
        list(
            start0 = start,
            end0 = start + n_levels - 1L,
            index1 = seq.int(start + 1L, length.out = n_levels)
        )
    })
}

state_iid_blocks <- function(state) {
    if (!is.null(state$iid_blocks)) {
        return(state$iid_blocks)
    }
    if (!is.null(state$iid)) {
        return(list(state$iid))
    }
    list()
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

latent_pair_precision_from_mapping <- function(a_rows, a_cols, a_x, curvature, n_latent) {
    diag_extra <- numeric(n_latent)
    if (length(a_rows) <= 1L) {
        return(list(i = integer(), j = integer(), x = numeric(), diag_extra = diag_extra))
    }

    by_row <- split(seq_along(a_rows), a_rows)
    pair_values <- new.env(parent = emptyenv())
    for (idx in by_row) {
        n_entry <- length(idx)
        if (n_entry <= 1L) {
            next
        }
        row_weight <- curvature[[a_rows[[idx[[1L]]]]]]
        for (lhs_pos in seq_len(n_entry - 1L)) {
            lhs <- idx[[lhs_pos]]
            lhs_col <- a_cols[[lhs]]
            lhs_x <- a_x[[lhs]]
            for (rhs_pos in (lhs_pos + 1L):n_entry) {
                rhs <- idx[[rhs_pos]]
                rhs_col <- a_cols[[rhs]]
                value <- lhs_x * row_weight * a_x[[rhs]]
                if (lhs_col == rhs_col) {
                    diag_extra[[lhs_col]] <- diag_extra[[lhs_col]] + 2.0 * value
                    next
                }
                lo <- min(lhs_col, rhs_col)
                hi <- max(lhs_col, rhs_col)
                key <- paste0(lo, ":", hi)
                old <- pair_values[[key]]
                pair_values[[key]] <- if (is.null(old)) value else old + value
            }
        }
    }

    keys <- ls(pair_values, all.names = TRUE)
    if (length(keys) == 0L) {
        return(list(i = integer(), j = integer(), x = numeric(), diag_extra = diag_extra))
    }
    values <- vapply(keys, function(key) pair_values[[key]], numeric(1))
    keep <- is.finite(values) & values != 0.0
    keys <- keys[keep]
    values <- values[keep]
    if (length(keys) == 0L) {
        return(list(i = integer(), j = integer(), x = numeric(), diag_extra = diag_extra))
    }
    parts <- strsplit(keys, ":", fixed = TRUE)
    i <- vapply(parts, function(part) as.integer(part[[1L]]) - 1L, integer(1))
    j <- vapply(parts, function(part) as.integer(part[[2L]]) - 1L, integer(1))
    ord <- order(i, j)
    list(
        i = i[ord],
        j = j[ord],
        x = as.numeric(values[ord]),
        diag_extra = diag_extra
    )
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
    latent_pair_precision <- latent_pair_precision_from_mapping(
        a_rows = a_rows,
        a_cols = a_cols,
        a_x = a_x,
        curvature = curvature,
        n_latent = n_latent
    )
    latent_precision <- latent_precision + latent_pair_precision$diag_extra
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
        latent_pair_precision = latent_pair_precision[c("i", "j", "x")],
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
        H_u_u_sparse = list(i = integer(), j = integer(), x = matrix(numeric(), nrow = 1L)),
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
    latent_sparse_i <- if (is.null(res$theta_evidence_latent_precision_i)) {
        integer()
    } else {
        as.integer(res$theta_evidence_latent_precision_i)
    }
    latent_sparse_j <- if (is.null(res$theta_evidence_latent_precision_j)) {
        integer()
    } else {
        as.integer(res$theta_evidence_latent_precision_j)
    }
    latent_sparse_x <- if (is.null(res$theta_evidence_latent_precision_x)) {
        numeric()
    } else {
        as.numeric(res$theta_evidence_latent_precision_x)
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
    n_sparse <- length(latent_sparse_i)
    if (length(latent_sparse_j) != n_sparse ||
        length(latent_sparse_x) != n_support * n_sparse) {
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
    H_u_u_sparse <- list(
        i = latent_sparse_i,
        j = latent_sparse_j,
        x = matrix(
            latent_sparse_x,
            nrow = n_support,
            byrow = TRUE,
            dimnames = list(support_names, if (n_sparse > 0L) paste0("edge_", seq_len(n_sparse)) else character())
        )
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
        H_u_u_sparse = H_u_u_sparse,
        h_u = h_u,
        H_u_beta = H_u_beta
    )
}

theta_support_has_point <- function(theta_matrix, theta, tolerance = 1e-7) {
    theta_matrix <- as.matrix(theta_matrix)
    theta <- as.numeric(theta)
    if (nrow(theta_matrix) == 0L || ncol(theta_matrix) != length(theta)) {
        return(FALSE)
    }
    any(apply(theta_matrix, 1L, function(row) max(abs(row - theta)) <= tolerance))
}

theta_support_guard_candidates <- function(theta_evidence, guard_factor = 2.0) {
    theta_matrix <- as.matrix(theta_evidence$theta)
    if (nrow(theta_matrix) <= 1L || ncol(theta_matrix) == 0L) {
        return(matrix(numeric(), nrow = 0L, ncol = ncol(theta_matrix)))
    }
    center <- as.numeric(theta_matrix[1L, ])
    theta_min <- apply(theta_matrix, 2L, min)
    theta_max <- apply(theta_matrix, 2L, max)
    width <- theta_max - theta_min
    fallback_width <- pmax(abs(center) * 0.25, 1.0)
    width[!is.finite(width) | width <= 1e-8] <- fallback_width[!is.finite(width) | width <= 1e-8]

    candidates <- list()
    for (theta_idx in seq_along(center)) {
        lower <- center
        lower[[theta_idx]] <- theta_min[[theta_idx]] - guard_factor * width[[theta_idx]]
        upper <- center
        upper[[theta_idx]] <- theta_max[[theta_idx]] + guard_factor * width[[theta_idx]]
        candidates[[length(candidates) + 1L]] <- lower
        candidates[[length(candidates) + 1L]] <- upper
    }
    candidate_matrix <- do.call(rbind, candidates)
    colnames(candidate_matrix) <- colnames(theta_matrix)
    keep <- apply(candidate_matrix, 1L, function(theta) {
        all(is.finite(theta)) && !theta_support_has_point(theta_matrix, theta)
    })
    candidate_matrix[keep, , drop = FALSE]
}

append_theta_evidence_support <- function(theta_evidence, extra_evidence, support_name) {
    if (is.null(extra_evidence) || !is.list(extra_evidence) ||
        as.integer(extra_evidence$n_support) < 1L) {
        return(theta_evidence)
    }
    if (!identical(colnames(theta_evidence$theta), colnames(extra_evidence$theta)) ||
        !identical(dim(theta_evidence$H_beta_beta)[1:2], dim(extra_evidence$H_beta_beta)[1:2]) ||
        !identical(dim(theta_evidence$H_u_beta)[1:2], dim(extra_evidence$H_u_beta)[1:2]) ||
        !identical(ncol(theta_evidence$H_u_u_diag), ncol(extra_evidence$H_u_u_diag))) {
        return(theta_evidence)
    }

    base_sparse <- theta_evidence$H_u_u_sparse
    extra_sparse <- extra_evidence$H_u_u_sparse
    if (!is.null(base_sparse) || !is.null(extra_sparse)) {
        if (is.null(base_sparse) || is.null(extra_sparse) ||
            !identical(as.integer(base_sparse$i), as.integer(extra_sparse$i)) ||
            !identical(as.integer(base_sparse$j), as.integer(extra_sparse$j))) {
            return(theta_evidence)
        }
    }

    old_n <- as.integer(theta_evidence$n_support)
    new_n <- old_n + 1L
    support_names <- c(rownames(theta_evidence$theta), support_name)

    theta_evidence$theta <- rbind(theta_evidence$theta, extra_evidence$theta[1L, , drop = FALSE])
    rownames(theta_evidence$theta) <- support_names

    n_fixed <- dim(theta_evidence$H_beta_beta)[[1L]]
    n_latent <- dim(theta_evidence$H_u_beta)[[1L]]
    fixed_names <- dimnames(theta_evidence$H_beta_beta)[[1L]]
    latent_names <- dimnames(theta_evidence$H_u_beta)[[1L]]

    H_beta_beta <- array(
        0.0,
        dim = c(n_fixed, n_fixed, new_n),
        dimnames = list(fixed_names, fixed_names, support_names)
    )
    H_beta_beta[, , seq_len(old_n)] <- theta_evidence$H_beta_beta
    H_beta_beta[, , new_n] <- extra_evidence$H_beta_beta[, , 1L]
    theta_evidence$H_beta_beta <- H_beta_beta

    H_u_beta <- array(
        0.0,
        dim = c(n_latent, n_fixed, new_n),
        dimnames = list(latent_names, fixed_names, support_names)
    )
    H_u_beta[, , seq_len(old_n)] <- theta_evidence$H_u_beta
    H_u_beta[, , new_n] <- extra_evidence$H_u_beta[, , 1L]
    theta_evidence$H_u_beta <- H_u_beta

    theta_evidence$h_beta <- rbind(theta_evidence$h_beta, extra_evidence$h_beta[1L, , drop = FALSE])
    rownames(theta_evidence$h_beta) <- support_names
    theta_evidence$H_u_u_diag <- rbind(
        theta_evidence$H_u_u_diag,
        extra_evidence$H_u_u_diag[1L, , drop = FALSE]
    )
    rownames(theta_evidence$H_u_u_diag) <- support_names
    theta_evidence$h_u <- rbind(theta_evidence$h_u, extra_evidence$h_u[1L, , drop = FALSE])
    rownames(theta_evidence$h_u) <- support_names

    if (!is.null(base_sparse)) {
        theta_evidence$H_u_u_sparse$x <- rbind(
            theta_evidence$H_u_u_sparse$x,
            extra_evidence$H_u_u_sparse$x[1L, , drop = FALSE]
        )
        rownames(theta_evidence$H_u_u_sparse$x) <- support_names
    }

    theta_evidence$weights <- stats::setNames(c(as.numeric(theta_evidence$weights), 0.0), support_names)
    theta_evidence$log_weights <- stats::setNames(log(pmax(theta_evidence$weights, 1e-300)), support_names)
    theta_evidence$log_unnormalized_weights <- stats::setNames(
        c(as.numeric(theta_evidence$log_unnormalized_weights), NA_real_),
        support_names
    )
    theta_evidence$log_mlik <- stats::setNames(c(as.numeric(theta_evidence$log_mlik), NA_real_), support_names)
    theta_evidence$log_constants <- stats::setNames(
        c(as.numeric(theta_evidence$log_constants), as.numeric(extra_evidence$log_constants[[1L]])),
        support_names
    )
    theta_evidence$n_support <- as.integer(new_n)
    theta_evidence
}

evaluate_theta_evidence_at <- function(fit, theta) {
    args <- list(
        formula = fit$formula,
        data = fit$data,
        family = fit$family,
        output_profile = "thin",
        control.mode = list(
            theta = as.numeric(theta),
            restart = FALSE,
            skip.ccd = TRUE
        ),
        control.compute = list(
            skip.ccd = TRUE,
            internal = TRUE,
            theta.internal = TRUE,
            evidence = TRUE
        )
    )
    if (isTRUE(fit$offset_arg_provided)) {
        args$offset <- fit$offset
    }
    replay_fit <- tryCatch(
        do.call(rusty_inla, args),
        error = function(e) NULL
    )
    if (is.null(replay_fit) || is.null(replay_fit$internal.hyperpar$theta_evidence)) {
        return(NULL)
    }
    replay_fit$internal.hyperpar$theta_evidence
}

expand_multi_iid_theta_support <- function(
    fit,
    theta_evidence,
    guard_factor = 2.0,
    max_extra_points = 12L
) {
    if (is.null(theta_evidence) || !is.list(theta_evidence) ||
        as.integer(theta_evidence$n_support) <= 1L ||
        ncol(as.matrix(theta_evidence$theta)) <= 1L) {
        return(theta_evidence)
    }

    candidates <- theta_support_guard_candidates(theta_evidence, guard_factor = guard_factor)
    if (nrow(candidates) == 0L) {
        return(theta_evidence)
    }
    if (nrow(candidates) > max_extra_points) {
        candidates <- candidates[seq_len(max_extra_points), , drop = FALSE]
    }

    attempted <- nrow(candidates)
    added <- 0L
    for (candidate_idx in seq_len(nrow(candidates))) {
        theta <- as.numeric(candidates[candidate_idx, ])
        if (theta_support_has_point(theta_evidence$theta, theta)) {
            next
        }
        extra <- evaluate_theta_evidence_at(fit, theta)
        if (is.null(extra)) {
            next
        }
        before <- as.integer(theta_evidence$n_support)
        theta_evidence <- append_theta_evidence_support(
            theta_evidence = theta_evidence,
            extra_evidence = extra,
            support_name = paste0("guard_", candidate_idx)
        )
        if (as.integer(theta_evidence$n_support) > before) {
            added <- added + 1L
        }
    }
    if (added > 0L) {
        theta_evidence$strategy <- "expanded_guard_ccd_support_modes"
        theta_evidence$support_expansion <- list(
            strategy = "axial_fixed_theta_replay",
            guard_factor = as.numeric(guard_factor),
            attempted = as.integer(attempted),
            added = as.integer(added),
            max_extra_points = as.integer(max_extra_points)
        )
    }
    theta_evidence
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
    names(theta_opt) <- theta_names
    ccd_weights <- if (is.null(res$ccd_weights)) numeric() else as.numeric(res$ccd_weights)
    ccd_thetas <- if (is.null(res$ccd_thetas)) numeric() else as.numeric(res$ccd_thetas)

    theta_matrix <- NULL
    if (length(ccd_weights) > 0L &&
        length(ccd_thetas) > 0L &&
        length(ccd_thetas) %% n_theta == 0L) {
        candidate <- matrix(ccd_thetas, ncol = n_theta, byrow = TRUE)
        if (nrow(candidate) == length(ccd_weights)) {
            theta_matrix <- candidate
            support_names <- paste0("theta_", seq_len(nrow(theta_matrix)))
            support_names[[1L]] <- "source_mode"
            dimnames(theta_matrix) <- list(support_names, theta_names)
            names(ccd_weights) <- support_names
        }
    }
    ccd_log_mlik <- if (is.null(res$ccd_log_mlik)) numeric() else as.numeric(res$ccd_log_mlik)
    ccd_log_weight <- if (is.null(res$ccd_log_weight)) numeric() else as.numeric(res$ccd_log_weight)
    if (!is.null(theta_matrix)) {
        support_names <- rownames(theta_matrix)
        if (length(ccd_log_mlik) == length(support_names)) {
            names(ccd_log_mlik) <- support_names
        }
        if (length(ccd_log_weight) == length(support_names)) {
            names(ccd_log_weight) <- support_names
        }
    }
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

iid_update_signature_mismatch <- function(state_signature, backend_signature, allow_multiple = FALSE) {
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
    if (isTRUE(allow_multiple)) {
        old_blocks <- state_signature$latent_blocks
        new_blocks <- backend_signature$latent_blocks
        if (length(old_blocks) < 1L || length(new_blocks) < 1L) {
            return("fixed/iid evidence states require at least one iid latent block")
        }
        if (length(old_blocks) != length(new_blocks)) {
            return(sprintf(
                "latent-block count mismatch: state has %d, update has %d",
                length(old_blocks),
                length(new_blocks)
            ))
        }
        for (idx in seq_along(old_blocks)) {
            old_block <- old_blocks[[idx]]
            new_block <- new_blocks[[idx]]
            if (!identical(old_block$model, "iid") || !identical(new_block$model, "iid")) {
                return(sprintf(
                    "latent model mismatch in block %d: state uses '%s', update uses '%s'; only iid is supported",
                    idx,
                    old_block$model,
                    new_block$model
                ))
            }
            if (!identical(old_block$covariate_name, new_block$covariate_name)) {
                return(sprintf(
                    "iid covariate mismatch in block %d: state uses '%s', update uses '%s'",
                    idx,
                    old_block$covariate_name,
                    new_block$covariate_name
                ))
            }
        }
        return(NULL)
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

expand_theta_evidence_for_new_iid_blocks <- function(theta_evidence, state, backend_signature) {
    n_support <- as.integer(theta_evidence$n_support)
    n_fixed <- dim(theta_evidence$H_beta_beta)[[1L]]
    n_new <- as.integer(backend_signature$n_latent)
    new_blocks <- backend_signature$latent_blocks
    state_blocks <- state_iid_blocks(state)
    old_ranges <- latent_block_ranges(state$signature)
    new_ranges <- latent_block_ranges(backend_signature)

    latent_precision <- matrix(0.0, nrow = n_support, ncol = n_new)
    latent_linear <- matrix(0.0, nrow = n_support, ncol = n_new)
    latent_fixed <- array(0.0, dim = c(n_new, n_fixed, n_support))
    old_to_new <- rep(NA_integer_, as.integer(state$signature$n_latent))

    for (idx in seq_along(new_blocks)) {
        old_levels <- as.character(state_blocks[[idx]]$levels)
        new_levels <- iid_levels_from_signature_block(new_blocks[[idx]])
        level_match <- match(new_levels, old_levels)
        matched <- which(!is.na(level_match))
        if (length(matched) == 0L) {
            next
        }
        old_index <- old_ranges[[idx]]$index1
        new_index <- new_ranges[[idx]]$index1
        old_pos <- level_match[matched]
        old_latent_idx <- old_index[old_pos]
        new_latent_idx <- new_index[matched]

        latent_precision[, new_latent_idx] <-
            theta_evidence$H_u_u_diag[, old_latent_idx, drop = FALSE]
        latent_linear[, new_latent_idx] <-
            theta_evidence$h_u[, old_latent_idx, drop = FALSE]
        latent_fixed[new_latent_idx, , ] <-
            theta_evidence$H_u_beta[old_latent_idx, , , drop = FALSE]
        old_to_new[old_latent_idx] <- new_latent_idx
    }

    sparse <- list(i = integer(), j = integer(), x = matrix(numeric(), nrow = n_support))
    theta_sparse <- theta_evidence$H_u_u_sparse
    if (!is.null(theta_sparse) && length(theta_sparse$x) > 0L) {
        old_i <- as.integer(theta_sparse$i)
        old_j <- as.integer(theta_sparse$j)
        x_matrix <- as.matrix(theta_sparse$x)
        if (length(old_i) > 0L && nrow(x_matrix) == n_support && ncol(x_matrix) == length(old_i)) {
            new_i <- old_to_new[old_i + 1L]
            new_j <- old_to_new[old_j + 1L]
            keep <- !is.na(new_i) & !is.na(new_j)
            if (any(keep)) {
                new_i0 <- new_i[keep] - 1L
                new_j0 <- new_j[keep] - 1L
                x_keep <- x_matrix[, keep, drop = FALSE]
                swapped <- new_i0 > new_j0
                lo <- ifelse(swapped, new_j0, new_i0)
                hi <- ifelse(swapped, new_i0, new_j0)
                sparse <- list(i = as.integer(lo), j = as.integer(hi), x = x_keep)
            }
        }
    }

    list(
        latent_precision = latent_precision,
        latent_linear = latent_linear,
        latent_fixed = latent_fixed,
        latent_sparse = sparse,
        n_support = n_support
    )
}

iid_levels_from_signature_block <- function(block) {
    levels <- as.character(block$level_values)
    if (length(levels) != as.integer(block$n_levels)) {
        levels <- as.character(seq_len(as.integer(block$n_levels)))
    }
    levels
}

aggregate_sparse_pairs_0 <- function(i, j, x) {
    if (length(x) == 0L) {
        return(list(i = integer(), j = integer(), x = numeric()))
    }
    pair_values <- new.env(parent = emptyenv())
    for (idx in seq_along(x)) {
        value <- as.numeric(x[[idx]])
        if (!is.finite(value) || value == 0.0) {
            next
        }
        lo <- min(as.integer(i[[idx]]), as.integer(j[[idx]]))
        hi <- max(as.integer(i[[idx]]), as.integer(j[[idx]]))
        if (lo == hi) {
            next
        }
        key <- paste0(lo, ":", hi)
        old <- pair_values[[key]]
        pair_values[[key]] <- if (is.null(old)) value else old + value
    }
    keys <- ls(pair_values, all.names = TRUE)
    if (length(keys) == 0L) {
        return(list(i = integer(), j = integer(), x = numeric()))
    }
    values <- vapply(keys, function(key) pair_values[[key]], numeric(1))
    keep <- is.finite(values) & values != 0.0
    keys <- keys[keep]
    values <- values[keep]
    if (length(keys) == 0L) {
        return(list(i = integer(), j = integer(), x = numeric()))
    }
    parts <- strsplit(keys, ":", fixed = TRUE)
    out_i <- vapply(parts, function(part) as.integer(part[[1L]]), integer(1))
    out_j <- vapply(parts, function(part) as.integer(part[[2L]]), integer(1))
    ord <- order(out_i, out_j)
    list(i = out_i[ord], j = out_j[ord], x = as.numeric(values[ord]))
}

aggregate_sparse_pair_matrix_0 <- function(i, j, x) {
    x <- as.matrix(x)
    n_support <- nrow(x)
    if (length(i) == 0L || length(j) == 0L || ncol(x) == 0L) {
        return(list(i = integer(), j = integer(), x = matrix(numeric(), nrow = n_support)))
    }
    if (length(i) != length(j) || ncol(x) != length(i)) {
        stop("Sparse support edge matrix has inconsistent dimensions.", call. = FALSE)
    }

    pair_values <- new.env(parent = emptyenv())
    for (idx in seq_along(i)) {
        lo <- min(as.integer(i[[idx]]), as.integer(j[[idx]]))
        hi <- max(as.integer(i[[idx]]), as.integer(j[[idx]]))
        if (lo == hi) {
            next
        }
        values <- as.numeric(x[, idx])
        values[!is.finite(values)] <- 0.0
        if (!any(values != 0.0)) {
            next
        }
        key <- paste0(lo, ":", hi)
        old <- pair_values[[key]]
        pair_values[[key]] <- if (is.null(old)) values else old + values
    }

    keys <- ls(pair_values, all.names = TRUE)
    if (length(keys) == 0L) {
        return(list(i = integer(), j = integer(), x = matrix(numeric(), nrow = n_support)))
    }
    parts <- strsplit(keys, ":", fixed = TRUE)
    out_i <- vapply(parts, function(part) as.integer(part[[1L]]), integer(1))
    out_j <- vapply(parts, function(part) as.integer(part[[2L]]), integer(1))
    ord <- order(out_i, out_j)
    out_x <- do.call(cbind, lapply(keys[ord], function(key) pair_values[[key]]))
    if (is.null(dim(out_x))) {
        out_x <- matrix(out_x, ncol = 1L)
    }
    colnames(out_x) <- paste0("edge_", seq_len(ncol(out_x)))
    list(i = out_i[ord], j = out_j[ord], x = out_x)
}

state_latent_pair_precision <- function(state) {
    if (!is.null(state$latent_pair_precision)) {
        return(state$latent_pair_precision)
    }
    if (!is.null(state$graph$latent_pair_precision)) {
        return(state$graph$latent_pair_precision)
    }
    list(i = integer(), j = integer(), x = numeric())
}

align_update_state_graph_to_backend <- function(state, backend_signature, include_cross) {
    state_blocks <- state_iid_blocks(state)
    new_blocks <- backend_signature$latent_blocks
    old_ranges <- latent_block_ranges(state$signature)
    new_ranges <- latent_block_ranges(backend_signature)

    n_latent <- as.integer(backend_signature$n_latent)
    n_fixed <- as.integer(backend_signature$n_fixed)
    latent_precision <- numeric(n_latent)
    latent_linear <- numeric(n_latent)
    latent_fixed_cross <- matrix(0.0, nrow = n_latent, ncol = n_fixed)
    flat_old_to_new <- rep(NA_integer_, as.integer(state$signature$n_latent))
    born_levels_by_block <- list()

    for (idx in seq_along(new_blocks)) {
        state_block <- state_blocks[[idx]]
        new_block <- new_blocks[[idx]]
        old_levels <- as.character(state_block$levels)
        new_levels <- iid_levels_from_signature_block(new_block)
        level_match <- match(new_levels, old_levels)
        matched <- which(!is.na(level_match))
        new_index <- new_ranges[[idx]]$index1
        old_index <- old_ranges[[idx]]$index1

        if (length(matched) > 0L) {
            old_pos <- level_match[matched]
            latent_precision[new_index[matched]] <-
                as.numeric(state_block$evidence_precision_diag[old_pos])
            latent_linear[new_index[matched]] <-
                as.numeric(state_block$evidence_linear[old_pos])
            flat_old_to_new[old_index[old_pos]] <- new_index[matched]
            if (isTRUE(include_cross)) {
                cross_precision <- if (!is.null(state_block$fixed_cross_precision)) {
                    as.matrix(state_block$fixed_cross_precision)
                } else if (idx == 1L && !is.null(state$iid_fixed_cross_precision)) {
                    as.matrix(state$iid_fixed_cross_precision)
                } else {
                    NULL
                }
                if (is.null(cross_precision)) {
                    stop("Fixed/iid cross evidence mode requires a state with fixed cross precision.", call. = FALSE)
                }
                if (nrow(cross_precision) != length(old_levels) ||
                    ncol(cross_precision) != n_fixed) {
                    stop("Fixed/iid cross evidence dimensions do not match the update-state signature.", call. = FALSE)
                }
                latent_fixed_cross[new_index[matched], ] <-
                    cross_precision[old_pos, , drop = FALSE]
            }
        }
        born_levels_by_block[[as.character(new_block$covariate_name)]] <- new_levels[is.na(level_match)]
    }

    pair_precision <- list(i = integer(), j = integer(), x = numeric())
    if (isTRUE(include_cross)) {
        old_pairs <- state_latent_pair_precision(state)
        old_i <- as.integer(old_pairs$i)
        old_j <- as.integer(old_pairs$j)
        old_x <- as.numeric(old_pairs$x)
        if (length(old_x) > 0L) {
            i_new <- flat_old_to_new[old_i + 1L]
            j_new <- flat_old_to_new[old_j + 1L]
            keep <- !is.na(i_new) & !is.na(j_new)
            pair_precision <- aggregate_sparse_pairs_0(
                i = i_new[keep] - 1L,
                j = j_new[keep] - 1L,
                x = old_x[keep]
            )
        }
    }

    list(
        latent_precision = latent_precision,
        latent_linear = latent_linear,
        latent_fixed_cross = latent_fixed_cross,
        latent_pair_precision = pair_precision,
        born_levels_by_block = born_levels_by_block
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
    mismatch <- iid_update_signature_mismatch(
        state$signature,
        backend_signature,
        allow_multiple = TRUE
    )
    if (!is.null(mismatch)) {
        stop(
            paste(
                sprintf("Fixed/iid update-state signature mismatch: %s.", mismatch),
                "Required contract: same family, same fixed-effect columns,",
                "same iid covariate names in the same order, and only iid latent models.",
                "New iid levels are allowed and receive zero old evidence."
            ),
            call. = FALSE
        )
    }
    validate_fixed_iid_evidence_state_scope(backend_signature, length(state$theta_mode))

    fixed_precision <- as.matrix(state$fixed$evidence_precision)
    fixed_linear <- as.numeric(state$fixed$evidence_linear)
    if (nrow(fixed_precision) != backend_spec$n_fixed ||
        ncol(fixed_precision) != backend_spec$n_fixed ||
        length(fixed_linear) != backend_spec$n_fixed) {
        stop("Fixed update-state evidence dimensions do not match the new fixed-effect design.", call. = FALSE)
    }

    aligned_state <- align_update_state_graph_to_backend(
        state = state,
        backend_signature = backend_signature,
        include_cross = include_cross
    )
    first_new_block <- backend_signature$latent_blocks[[1L]]
    new_levels <- iid_levels_from_signature_block(first_new_block)
    old_levels <- as.character(state_iid_blocks(state)[[1L]]$levels)
    level_match <- match(new_levels, old_levels)
    born_levels_by_block <- aligned_state$born_levels_by_block
    born_levels <- unlist(born_levels_by_block, use.names = FALSE)

    if (use_theta_evidence) {
        theta_evidence <- state$theta_evidence
        if (is.null(theta_evidence) || !is.list(theta_evidence) ||
            as.integer(theta_evidence$n_support) <= 1L) {
            stop(
                "fixed_iid_cross_theta_evidence requires a rusty_update_state() object with multi-point theta_evidence.",
                call. = FALSE
            )
        }
        if (length(state$theta_mode) != ncol(theta_evidence$theta)) {
            stop(
                "fixed_iid_cross_theta_evidence theta dimensions do not match the update state.",
                call. = FALSE
            )
        }
        expanded_theta <- expand_theta_evidence_for_new_iid_blocks(
            theta_evidence = theta_evidence,
            state = state,
            backend_signature = backend_signature
        )
        backend_spec$theta_state_n_support <- as.integer(expanded_theta$n_support)
        backend_spec$theta_state_support <- flatten_support_rows(theta_evidence$theta)
        backend_spec$theta_state_fixed_precision <- flatten_support_fixed_precision(theta_evidence$H_beta_beta)
        backend_spec$theta_state_fixed_linear <- flatten_support_rows(theta_evidence$h_beta)
        backend_spec$theta_state_latent_precision_diag <- flatten_support_rows(expanded_theta$latent_precision)
        backend_spec$theta_state_latent_linear <- flatten_support_rows(expanded_theta$latent_linear)
        if (length(expanded_theta$latent_sparse$i) > 0L) {
            backend_spec$theta_state_latent_precision_i <- as.integer(expanded_theta$latent_sparse$i)
            backend_spec$theta_state_latent_precision_j <- as.integer(expanded_theta$latent_sparse$j)
            backend_spec$theta_state_latent_precision_x <- flatten_support_rows(expanded_theta$latent_sparse$x)
        }
        backend_spec$theta_state_latent_fixed_precision <- flatten_support_latent_fixed(expanded_theta$latent_fixed)
        backend_spec$theta_state_log_constant <- as.numeric(theta_evidence$log_constants)
    } else {
        backend_spec$fixed_state_precision <- as.numeric(t(fixed_precision))
        backend_spec$fixed_state_linear <- fixed_linear
        backend_spec$latent_state_precision_diag <- aligned_state$latent_precision
        backend_spec$latent_state_linear <- aligned_state$latent_linear
        if (include_cross) {
            backend_spec$latent_fixed_state_precision <- as.numeric(aligned_state$latent_fixed_cross)
            if (length(aligned_state$latent_pair_precision$x) > 0L) {
                backend_spec$latent_state_precision_i <- as.integer(aligned_state$latent_pair_precision$i)
                backend_spec$latent_state_precision_j <- as.integer(aligned_state$latent_pair_precision$j)
                backend_spec$latent_state_precision_x <- as.numeric(aligned_state$latent_pair_precision$x)
            }
        }
    }
    source_iid_names <- vapply(
        state_iid_blocks(state),
        function(block) as.character(block$covariate_name),
        character(1)
    )
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
        theta_evidence_support_expansion = if (is.null(state$theta_evidence$support_expansion)) {
            NA_character_
        } else {
            state$theta_evidence$support_expansion$strategy
        },
        theta_evidence_guard_points_added = if (is.null(state$theta_evidence$support_expansion)) {
            NA_integer_
        } else {
            as.integer(state$theta_evidence$support_expansion$added)
        },
        theta_evidence_solver_status = if (use_theta_evidence) {
            if (ncol(as.matrix(state$theta_evidence$theta)) == 1L) {
                "linear_1d_integrated"
            } else {
                "guarded_shepard_nd_integrated"
            }
        } else if (is.null(state$theta_evidence)) {
            NA_character_
        } else {
            state$theta_evidence$solver_status
        },
        source_family = state$signature$family,
        source_n_obs = state$source$n_obs,
        source_fixed_names = state$fixed$names,
        source_iid_covariate_name = source_iid_names[[1L]],
        source_iid_covariate_names = source_iid_names,
        born_iid_levels = born_levels,
        born_iid_levels_by_block = born_levels_by_block,
        caveat = paste(
            "Experimental fixed+iid Gaussian old-data evidence update;",
            if (use_theta_evidence) {
                "uses guarded interpolation across theta evidence support with fixed-iid and iid-iid cross blocks."
            } else if (include_cross) {
                "uses dense fixed evidence, diagonal iid evidence, fixed-iid cross blocks, and sparse iid-iid cross edges."
            } else {
                "uses dense fixed evidence and diagonal iid evidence, but omits the fixed-iid cross block and iid-iid cross blocks."
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
#' fixed-effect block, diagonal `iid` blocks, fixed-iid cross blocks, and
#' sparse iid-iid cross edges when several iid blocks are present.
#' The evidence is a local Taylor approximation at the old posterior mode; the
#' original model priors remain active in the later update fit.
#'
#' @param fit A `rusty_inla` fit with one or more `iid` latent blocks.
#' @param scope Currently only `"fixed_iid_gaussian"`.
#' @param min_variance Lower bound for conditional variances.
#' @param theta_support_expansion Use `"auto"` to add fixed-theta guard
#'   evidence points for multi-`iid` states, or `"none"` to keep only the
#'   original CCD support.
#' @param theta_support_guard_factor Distance multiplier used for axial guard
#'   points beyond the original CCD support range.
#' @export
rusty_update_state <- function(
    fit,
    scope = c("fixed_iid_gaussian"),
    min_variance = 1e-10,
    theta_support_expansion = c("auto", "none"),
    theta_support_guard_factor = 2.0
) {
    scope <- match.arg(scope)
    theta_support_expansion <- match.arg(theta_support_expansion)
    if (!inherits(fit, "rusty_inla")) {
        stop("fit must be a rusty_inla object.", call. = FALSE)
    }
    if (!is.numeric(theta_support_guard_factor) || length(theta_support_guard_factor) != 1L ||
        !is.finite(theta_support_guard_factor) || theta_support_guard_factor <= 0.0) {
        stop("theta_support_guard_factor must be a positive finite number.", call. = FALSE)
    }
    if (is.null(fit$backend_signature) || is.null(fit$internal.gaussian) ||
        is.null(fit$internal.hyperpar) || is.null(fit$internal.design) ||
        is.null(fit$mode)) {
        stop("fit does not contain the internal metadata needed for update-state extraction.", call. = FALSE)
    }
    validate_fixed_iid_evidence_state_scope(
        fit$backend_signature,
        length(fit$internal.hyperpar$theta_mode)
    )
    if (length(fit$backend_signature$fixed_names) == 0L) {
        stop("fixed_iid_gaussian update states require at least one fixed-effect column.", call. = FALSE)
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
    latent_mode <- as.numeric(fit$internal.gaussian$latent_mode)
    latent_var <- pmax(as.numeric(fit$internal.gaussian$latent_var_theta_opt), as.numeric(min_variance))
    if (length(latent_mode) != as.integer(fit$backend_signature$n_latent) ||
        length(latent_var) != as.integer(fit$backend_signature$n_latent)) {
        stop("fit iid conditional state is not available for update-state extraction.", call. = FALSE)
    }

    block_ranges <- latent_block_ranges(fit$backend_signature)
    iid_blocks <- vector("list", length(fit$backend_signature$latent_blocks))
    for (block_idx in seq_along(fit$backend_signature$latent_blocks)) {
        block <- fit$backend_signature$latent_blocks[[block_idx]]
        cov_name <- block$covariate_name
        random_df <- fit$summary.random[[cov_name]]
        if (is.null(random_df) || nrow(random_df) != as.integer(block$n_levels)) {
            stop("fit random-effect summary does not match the iid backend signature.", call. = FALSE)
        }
        iid_levels <- rownames(random_df)
        if (is.null(iid_levels)) {
            iid_levels <- as.character(random_df$ID)
        }
        idx <- block_ranges[[block_idx]]$index1
        iid_evidence_precision <- pmax(as.numeric(evidence$latent_precision[idx]), 0.0)
        iid_evidence_linear <- as.numeric(evidence$latent_linear[idx])
        iid_fixed_cross_precision <- as.matrix(evidence$latent_fixed_precision[idx, , drop = FALSE])
        if (length(iid_evidence_precision) != length(iid_levels) ||
            length(iid_evidence_linear) != length(iid_levels) ||
            nrow(iid_fixed_cross_precision) != length(iid_levels) ||
            ncol(iid_fixed_cross_precision) != length(fixed_names)) {
            stop("Local likelihood evidence dimensions do not match the fit signature.", call. = FALSE)
        }
        names(iid_evidence_precision) <- iid_levels
        names(iid_evidence_linear) <- iid_levels
        dimnames(iid_fixed_cross_precision) <- list(iid_levels, fixed_names)
        iid_blocks[[block_idx]] <- list(
            covariate_name = cov_name,
            model = block$model,
            levels = iid_levels,
            start = as.integer(block_ranges[[block_idx]]$start0),
            mode = latent_mode[idx],
            variance_theta_opt = latent_var[idx],
            tau_at_source_theta = exp(theta_mode[[block_idx]]),
            evidence_precision_diag = iid_evidence_precision,
            evidence_linear = iid_evidence_linear,
            fixed_cross_precision = iid_fixed_cross_precision,
            dropped_negative_precision = 0L
        )
    }
    names(iid_blocks) <- vapply(iid_blocks, function(block) block$covariate_name, character(1))
    first_iid <- iid_blocks[[1L]]
    iid_levels <- first_iid$levels
    iid_evidence_precision <- first_iid$evidence_precision_diag
    iid_evidence_linear <- first_iid$evidence_linear
    iid_fixed_cross_precision <- first_iid$fixed_cross_precision

    if (length(evidence$latent_precision) != as.integer(fit$backend_signature$n_latent) ||
        length(evidence$latent_linear) != as.integer(fit$backend_signature$n_latent) ||
        nrow(evidence$latent_fixed_precision) != as.integer(fit$backend_signature$n_latent) ||
        ncol(evidence$latent_fixed_precision) != length(fixed_names)) {
        stop("Local likelihood evidence dimensions do not match the fit signature.", call. = FALSE)
    }

    theta_evidence <- NULL
    if (length(iid_blocks) == 1L) {
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
    } else {
        theta_evidence <- fit$internal.hyperpar$theta_evidence
        if (!is.null(theta_evidence) && !is.list(theta_evidence)) {
            theta_evidence <- NULL
        }
        if (!is.null(theta_evidence)) {
            theta_evidence$block_format <- "dense_fixed_diag_iid_sparse_iid_cross_by_support"
            if (identical(theta_support_expansion, "auto")) {
                theta_evidence <- expand_multi_iid_theta_support(
                    fit = fit,
                    theta_evidence = theta_evidence,
                    guard_factor = theta_support_guard_factor
                )
            }
        }
    }
    theta_evidence_policy <- if (!is.null(theta_evidence) && as.integer(theta_evidence$n_support) > 1L) {
        "ccd_support_modes_not_integrated"
    } else {
        "single_support_point_source_mode"
    }
    semantics <- fixed_iid_update_state_semantics(theta_evidence_policy)
    if (length(iid_blocks) > 1L &&
        (is.null(theta_evidence) || as.integer(theta_evidence$n_support) <= 1L)) {
        semantics$compatible_update_modes <- setdiff(
            semantics$compatible_update_modes,
            "fixed_iid_cross_theta_evidence"
        )
    }

    state <- list(
        version = 3L,
        scope = scope,
        approximation = "fixed_iid_cross_gaussian_evidence",
        semantics = semantics,
        signature = fit$backend_signature,
        source = list(
            n_obs = as.integer(fit$internal.design$n_data),
            family = fit$backend_signature$family,
            fixed_names = fixed_names,
            iid_covariate_name = first_iid$covariate_name,
            iid_covariate_names = names(iid_blocks),
            iid_model = first_iid$model,
            iid_models = vapply(iid_blocks, function(block) block$model, character(1)),
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
            covariate_name = first_iid$covariate_name,
            levels = iid_levels,
            mode = first_iid$mode,
            variance_theta_opt = first_iid$variance_theta_opt,
            tau_at_source_theta = first_iid$tau_at_source_theta,
            evidence_precision_diag = iid_evidence_precision,
            evidence_linear = iid_evidence_linear,
            dropped_negative_precision = 0L
        ),
        iid_blocks = iid_blocks,
        iid_fixed_cross_precision = iid_fixed_cross_precision,
        latent_pair_precision = evidence$latent_pair_precision,
        graph = list(
            index_base = 0L,
            block_names = names(iid_blocks),
            latent_pair_precision = evidence$latent_pair_precision
        ),
        caveats = c(
            "Experimental fixed+iid Gaussian old-data likelihood evidence approximation.",
            if (length(iid_blocks) == 1L) {
                "Uses dense fixed-effect evidence, diagonal iid evidence, and the fixed-iid cross block."
            } else {
                "Uses dense fixed-effect evidence, diagonal iid evidence blocks, fixed-iid cross blocks, and sparse iid-iid cross edges."
            },
            if (identical(theta_evidence_policy, "ccd_support_modes_not_integrated")) {
                "Stores a Phase 8C CCD-support theta-evidence container extracted at old theta support points."
            } else if (length(iid_blocks) > 1L) {
                "Multi-iid states currently store source-mode evidence only; theta-mixture evidence remains one-iid."
            } else {
                "Stores a Phase 8C single-point theta-evidence container at the old source theta."
            },
            "Source-mode updates use one frozen evidence block; theta-evidence updates still approximate old data by local Gaussian support blocks.",
            if (!is.null(theta_evidence$support_expansion) &&
                as.integer(theta_evidence$support_expansion$added) > 0L) {
                "Multi-iid theta evidence includes fixed-theta guard support points beyond the original CCD cloud."
            },
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

theta_evidence_support_weights <- function(theta_evidence, theta) {
    n_support <- as.integer(theta_evidence$n_support)
    theta_matrix <- as.matrix(theta_evidence$theta)
    theta <- as.numeric(theta)
    if (n_support <= 0L || nrow(theta_matrix) != n_support) {
        stop("theta_evidence has inconsistent support dimensions.", call. = FALSE)
    }
    if (ncol(theta_matrix) != length(theta)) {
        stop("theta_evidence theta dimensions do not match the requested theta.", call. = FALSE)
    }
    if (n_support == 1L) {
        return(list(index = 1L, weight = 1.0))
    }
    if (length(theta) == 1L) {
        blend <- theta_evidence_support_blend(theta_evidence, theta)
        if (identical(blend$left, blend$right) || blend$right_weight <= 0.0) {
            return(list(index = blend$left, weight = 1.0))
        }
        return(list(
            index = c(blend$left, blend$right),
            weight = c(1.0 - blend$right_weight, blend$right_weight)
        ))
    }

    n_theta <- length(theta)
    theta_min <- apply(theta_matrix, 2L, min)
    theta_max <- apply(theta_matrix, 2L, max)
    scale <- theta_max - theta_min
    scale[!is.finite(scale) | scale <= 1e-12] <- 1.0
    scaled <- sweep(theta_matrix, 2L, theta, FUN = "-")
    scaled <- sweep(scaled, 2L, scale, FUN = "/")
    distances <- rowSums(scaled * scaled)
    order_idx <- order(distances)
    if (distances[[order_idx[[1L]]]] <= 1e-12) {
        return(list(index = order_idx[[1L]], weight = 1.0))
    }

    outside_support <- any(theta < theta_min - 1e-10 | theta > theta_max + 1e-10)
    if (outside_support) {
        return(list(index = order_idx[[1L]], weight = 1.0))
    }

    target_count <- max(min(2L * n_theta + 1L, n_support), 1L)
    radius_limit <- max(distances[[order_idx[[1L]]]] * 4.0, 1e-10)
    selected <- integer()
    for (support_idx in order_idx) {
        dist <- distances[[support_idx]]
        if (length(selected) < target_count || dist <= radius_limit) {
            selected <- c(selected, support_idx)
        }
        if (length(selected) >= target_count && dist > radius_limit) {
            break
        }
    }

    raw_weights <- 1.0 / pmax(distances[selected], 1e-12)
    weight_sum <- sum(raw_weights)
    if (!is.finite(weight_sum) || weight_sum <= 0.0) {
        return(list(index = order_idx[[1L]], weight = 1.0))
    }
    list(index = selected, weight = raw_weights / weight_sum)
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

blend_theta_array_weighted <- function(arr, weights) {
    index <- as.integer(weights$index)
    weight <- as.numeric(weights$weight)
    if (length(index) != length(weight) || length(index) == 0L) {
        stop("Invalid theta support weights.", call. = FALSE)
    }

    if (length(dim(arr)) == 3L) {
        out <- array(0.0, dim = dim(arr)[1:2], dimnames = dimnames(arr)[1:2])
        for (idx in seq_along(index)) {
            out <- out + weight[[idx]] * arr[, , index[[idx]]]
        }
        return(out)
    }

    out <- rep(0.0, ncol(as.matrix(arr)))
    for (idx in seq_along(index)) {
        out <- out + weight[[idx]] * as.numeric(as.matrix(arr)[index[[idx]], ])
    }
    out
}

blend_theta_log_constant <- function(theta_evidence, weights) {
    values <- as.numeric(theta_evidence$log_constants)
    sum(as.numeric(weights$weight) * values[as.integer(weights$index)])
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

theta_evidence_block_at_signature <- function(theta_evidence, theta, state, backend_signature) {
    weights <- theta_evidence_support_weights(theta_evidence, theta)
    fixed_names <- as.character(backend_signature$fixed_names)
    old_fixed_names <- dimnames(theta_evidence$H_beta_beta)[[1L]]
    if (!identical(as.character(old_fixed_names), fixed_names)) {
        stop("Cannot compose update states with changed fixed-effect columns.", call. = FALSE)
    }

    expanded <- expand_theta_evidence_for_new_iid_blocks(
        theta_evidence = theta_evidence,
        state = state,
        backend_signature = backend_signature
    )
    n_latent <- as.integer(backend_signature$n_latent)
    latent_names <- as.character(seq_len(n_latent))

    H_beta_beta <- as.matrix(blend_theta_array_weighted(theta_evidence$H_beta_beta, weights))
    h_beta <- as.numeric(blend_theta_array_weighted(theta_evidence$h_beta, weights))
    H_u_u_diag <- as.numeric(blend_theta_array_weighted(expanded$latent_precision, weights))
    h_u <- as.numeric(blend_theta_array_weighted(expanded$latent_linear, weights))
    H_u_beta <- as.matrix(blend_theta_array_weighted(expanded$latent_fixed, weights))

    dimnames(H_beta_beta) <- list(fixed_names, fixed_names)
    names(h_beta) <- fixed_names
    names(H_u_u_diag) <- latent_names
    names(h_u) <- latent_names
    dimnames(H_u_beta) <- list(latent_names, fixed_names)

    sparse <- list(i = integer(), j = integer(), x = numeric())
    if (!is.null(expanded$latent_sparse) &&
        length(expanded$latent_sparse$i) > 0L &&
        ncol(as.matrix(expanded$latent_sparse$x)) == length(expanded$latent_sparse$i)) {
        sparse <- list(
            i = as.integer(expanded$latent_sparse$i),
            j = as.integer(expanded$latent_sparse$j),
            x = as.numeric(blend_theta_array_weighted(expanded$latent_sparse$x, weights))
        )
    }

    list(
        H_beta_beta = H_beta_beta,
        h_beta = h_beta,
        H_u_u_diag = H_u_u_diag,
        h_u = h_u,
        H_u_beta = H_u_beta,
        H_u_u_sparse = sparse,
        log_constant = blend_theta_log_constant(theta_evidence, weights)
    )
}

theta_sparse_matrix_or_empty <- function(theta_evidence) {
    n_support <- as.integer(theta_evidence$n_support)
    sparse <- theta_evidence$H_u_u_sparse
    if (is.null(sparse) || length(sparse$i) == 0L) {
        return(list(i = integer(), j = integer(), x = matrix(numeric(), nrow = n_support)))
    }
    x <- as.matrix(sparse$x)
    if (nrow(x) != n_support || ncol(x) != length(sparse$i)) {
        stop("theta_evidence sparse iid-iid support dimensions are inconsistent.", call. = FALSE)
    }
    list(i = as.integer(sparse$i), j = as.integer(sparse$j), x = x)
}

compose_theta_evidence <- function(previous_state, incremental_state) {
    theta_evidence <- incremental_state$theta_evidence
    previous_theta_evidence <- previous_state$theta_evidence
    if (is.null(theta_evidence) || is.null(previous_theta_evidence)) {
        stop("Composed update states require theta_evidence in both states.", call. = FALSE)
    }
    if (ncol(as.matrix(theta_evidence$theta)) != ncol(as.matrix(previous_theta_evidence$theta))) {
        stop("Composed update states require matching theta dimensions.", call. = FALSE)
    }

    fixed_names <- incremental_state$fixed$names
    use_flat_signature <- length(state_iid_blocks(incremental_state)) > 1L ||
        ncol(as.matrix(theta_evidence$theta)) > 1L
    iid_levels <- if (use_flat_signature) {
        as.character(seq_len(as.integer(incremental_state$signature$n_latent)))
    } else {
        incremental_state$iid$levels
    }
    n_support <- as.integer(theta_evidence$n_support)
    composed <- theta_evidence
    composed$version <- 3L
    composed$strategy <- if (use_flat_signature) {
        "composed_guarded_nd_support_modes"
    } else {
        "composed_ccd_support_modes"
    }
    composed$solver_status <- "not_integrated"

    for (support_idx in seq_len(n_support)) {
        old_block <- if (use_flat_signature) {
            theta_evidence_block_at_signature(
                theta_evidence = previous_theta_evidence,
                theta = theta_evidence$theta[support_idx, ],
                state = previous_state,
                backend_signature = incremental_state$signature
            )
        } else {
            theta_evidence_block_at(
                theta_evidence = previous_theta_evidence,
                theta = theta_evidence$theta[support_idx, ],
                fixed_names = fixed_names,
                iid_levels = iid_levels
            )
        }
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

    if (use_flat_signature) {
        previous_expanded <- expand_theta_evidence_for_new_iid_blocks(
            theta_evidence = previous_theta_evidence,
            state = previous_state,
            backend_signature = incremental_state$signature
        )
        previous_sparse <- previous_expanded$latent_sparse
        previous_sparse_x <- matrix(numeric(), nrow = n_support)
        previous_sparse_i <- integer()
        previous_sparse_j <- integer()
        if (!is.null(previous_sparse) &&
            length(previous_sparse$i) > 0L &&
            ncol(as.matrix(previous_sparse$x)) == length(previous_sparse$i)) {
            previous_sparse_i <- as.integer(previous_sparse$i)
            previous_sparse_j <- as.integer(previous_sparse$j)
            previous_sparse_x <- matrix(0.0, nrow = n_support, ncol = length(previous_sparse_i))
            for (support_idx in seq_len(n_support)) {
                weights <- theta_evidence_support_weights(
                    previous_theta_evidence,
                    theta_evidence$theta[support_idx, ]
                )
                previous_sparse_x[support_idx, ] <-
                    as.numeric(blend_theta_array_weighted(previous_sparse$x, weights))
            }
        }
        incremental_sparse <- theta_sparse_matrix_or_empty(theta_evidence)
        composed$H_u_u_sparse <- aggregate_sparse_pair_matrix_0(
            i = c(as.integer(incremental_sparse$i), previous_sparse_i),
            j = c(as.integer(incremental_sparse$j), previous_sparse_j),
            x = cbind(incremental_sparse$x, previous_sparse_x)
        )
        rownames(composed$H_u_u_sparse$x) <- rownames(theta_evidence$theta)
    }

    composed
}

compose_source_mode_evidence <- function(previous_state, incremental_state, theta_evidence) {
    fixed_names <- incremental_state$fixed$names
    if (length(state_iid_blocks(incremental_state)) > 1L ||
        ncol(as.matrix(theta_evidence$theta)) > 1L) {
        source_block <- theta_evidence_block_at_signature(
            theta_evidence = theta_evidence,
            theta = incremental_state$theta_mode,
            state = incremental_state,
            backend_signature = incremental_state$signature
        )
    } else {
        iid_levels <- incremental_state$iid$levels
        source_block <- theta_evidence_block_at(
            theta_evidence = theta_evidence,
            theta = incremental_state$theta_mode,
            fixed_names = fixed_names,
            iid_levels = iid_levels
        )
    }
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
    previous_iid_names <- vapply(state_iid_blocks(previous_state), function(block) block$covariate_name, character(1))
    incremental_iid_names <- vapply(state_iid_blocks(incremental_state), function(block) block$covariate_name, character(1))
    if (!identical(previous_iid_names, incremental_iid_names)) {
        stop("Cannot compose update states with changed iid covariate names.", call. = FALSE)
    }
}

compose_source_mode_update_state <- function(previous_state, incremental_state) {
    aligned_previous <- align_update_state_graph_to_backend(
        state = previous_state,
        backend_signature = incremental_state$signature,
        include_cross = TRUE
    )
    composed <- incremental_state
    composed$version <- 4L
    composed$semantics <- fixed_iid_update_state_semantics("single_support_point_source_mode")
    composed$semantics$theta_policy <- "composed_source_mode"
    composed$semantics$composition <- "previous_compressed_evidence_plus_current_likelihood_evidence"
    composed$source$n_obs <- as.integer(previous_state$source$n_obs + incremental_state$source$n_obs)
    composed$source$composed_from <- list(
        previous_n_obs = as.integer(previous_state$source$n_obs),
        incremental_n_obs = as.integer(incremental_state$source$n_obs)
    )
    composed$theta_evidence <- NULL
    composed$fixed$evidence_precision <-
        incremental_state$fixed$evidence_precision + previous_state$fixed$evidence_precision
    composed$fixed$evidence_linear <-
        incremental_state$fixed$evidence_linear + previous_state$fixed$evidence_linear

    ranges <- latent_block_ranges(incremental_state$signature)
    for (idx in seq_along(composed$iid_blocks)) {
        latent_idx <- ranges[[idx]]$index1
        block <- composed$iid_blocks[[idx]]
        block$evidence_precision_diag <-
            as.numeric(block$evidence_precision_diag) + aligned_previous$latent_precision[latent_idx]
        block$evidence_linear <-
            as.numeric(block$evidence_linear) + aligned_previous$latent_linear[latent_idx]
        block$fixed_cross_precision <-
            as.matrix(block$fixed_cross_precision) +
            aligned_previous$latent_fixed_cross[latent_idx, , drop = FALSE]
        names(block$evidence_precision_diag) <- block$levels
        names(block$evidence_linear) <- block$levels
        dimnames(block$fixed_cross_precision) <- list(block$levels, composed$fixed$names)
        composed$iid_blocks[[idx]] <- block
    }

    incremental_pairs <- state_latent_pair_precision(incremental_state)
    previous_pairs <- aligned_previous$latent_pair_precision
    composed$latent_pair_precision <- aggregate_sparse_pairs_0(
        i = c(as.integer(incremental_pairs$i), as.integer(previous_pairs$i)),
        j = c(as.integer(incremental_pairs$j), as.integer(previous_pairs$j)),
        x = c(as.numeric(incremental_pairs$x), as.numeric(previous_pairs$x))
    )
    composed$graph$latent_pair_precision <- composed$latent_pair_precision

    first_iid <- composed$iid_blocks[[1L]]
    composed$iid <- list(
        covariate_name = first_iid$covariate_name,
        levels = first_iid$levels,
        mode = first_iid$mode,
        variance_theta_opt = first_iid$variance_theta_opt,
        tau_at_source_theta = first_iid$tau_at_source_theta,
        evidence_precision_diag = first_iid$evidence_precision_diag,
        evidence_linear = first_iid$evidence_linear,
        dropped_negative_precision = 0L
    )
    composed$iid_fixed_cross_precision <- first_iid$fixed_cross_precision
    composed$caveats <- c(
        "Experimental composed fixed+iid Gaussian old-data likelihood evidence approximation.",
        "Adds previous compressed evidence to the current fit's extracted likelihood evidence.",
        "Uses source-mode evidence composition for one or more iid blocks.",
        "Intended for rolling diagnostics; validate against joint refits before production use."
    )
    class(composed) <- "rusty_update_state"
    composed
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

    previous_n_blocks <- length(state_iid_blocks(previous_state))
    incremental_n_blocks <- length(state_iid_blocks(incremental_state))
    previous_theta_cols <- if (is.null(previous_state$theta_evidence)) {
        0L
    } else {
        ncol(as.matrix(previous_state$theta_evidence$theta))
    }
    incremental_theta_cols <- if (is.null(incremental_state$theta_evidence)) {
        0L
    } else {
        ncol(as.matrix(incremental_state$theta_evidence$theta))
    }
    can_use_theta_composition <- !is.null(previous_state$theta_evidence) &&
        !is.null(incremental_state$theta_evidence) &&
        previous_theta_cols == incremental_theta_cols &&
        length(previous_state$theta_mode) == previous_theta_cols &&
        length(incremental_state$theta_mode) == incremental_theta_cols &&
        previous_n_blocks == incremental_n_blocks &&
        previous_n_blocks >= 1L &&
        previous_theta_cols == previous_n_blocks
    if (!isTRUE(can_use_theta_composition)) {
        return(compose_source_mode_update_state(previous_state, incremental_state))
    }

    theta_evidence <- compose_theta_evidence(previous_state, incremental_state)
    source_block <- compose_source_mode_evidence(previous_state, incremental_state, theta_evidence)

    composed <- incremental_state
    composed$version <- 4L
    composed$approximation <- "fixed_iid_cross_theta_evidence_composed"
    composed$semantics <- fixed_iid_update_state_semantics("ccd_support_modes_not_integrated")
    composed$semantics$theta_policy <- if (previous_theta_cols == 1L) {
        "composed_ccd_support_linear_1d"
    } else {
        "composed_guarded_shepard_nd"
    }
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

    ranges <- latent_block_ranges(composed$signature)
    for (idx in seq_along(composed$iid_blocks)) {
        latent_idx <- ranges[[idx]]$index1
        block <- composed$iid_blocks[[idx]]
        block$evidence_precision_diag <- as.numeric(source_block$H_u_u_diag[latent_idx])
        block$evidence_linear <- as.numeric(source_block$h_u[latent_idx])
        block$fixed_cross_precision <- as.matrix(source_block$H_u_beta[latent_idx, , drop = FALSE])
        names(block$evidence_precision_diag) <- block$levels
        names(block$evidence_linear) <- block$levels
        dimnames(block$fixed_cross_precision) <- list(block$levels, composed$fixed$names)
        composed$iid_blocks[[idx]] <- block
    }

    first_iid <- composed$iid_blocks[[1L]]
    composed$iid <- list(
        covariate_name = first_iid$covariate_name,
        levels = first_iid$levels,
        mode = first_iid$mode,
        variance_theta_opt = first_iid$variance_theta_opt,
        tau_at_source_theta = first_iid$tau_at_source_theta,
        evidence_precision_diag = first_iid$evidence_precision_diag,
        evidence_linear = first_iid$evidence_linear,
        dropped_negative_precision = 0L
    )
    composed$iid_fixed_cross_precision <- first_iid$fixed_cross_precision
    composed$latent_pair_precision <- if (!is.null(source_block$H_u_u_sparse) &&
        length(source_block$H_u_u_sparse$x) > 0L) {
        aggregate_sparse_pairs_0(
            i = as.integer(source_block$H_u_u_sparse$i),
            j = as.integer(source_block$H_u_u_sparse$j),
            x = as.numeric(source_block$H_u_u_sparse$x)
        )
    } else {
        list(i = integer(), j = integer(), x = numeric())
    }
    composed$graph$latent_pair_precision <- composed$latent_pair_precision
    composed$caveats <- c(
        "Experimental composed fixed+iid Gaussian old-data likelihood evidence approximation.",
        "Adds previous compressed evidence to the current fit's extracted likelihood evidence.",
        if (previous_theta_cols == 1L) {
            "Uses one-dimensional linear interpolation over the previous theta-evidence support."
        } else {
            "Uses guarded multidimensional interpolation over a composed multi-iid theta-evidence support."
        },
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
    blocks <- state_iid_blocks(x)
    if (length(blocks) == 1L) {
        cat(sprintf("iid block: %s with %d levels\n", blocks[[1L]]$covariate_name, length(blocks[[1L]]$levels)))
    } else {
        cat(sprintf("iid blocks: %d\n", length(blocks)))
        for (block in blocks) {
            cat(sprintf("  %s with %d levels\n", block$covariate_name, length(block$levels)))
        }
    }
    if (!is.null(x$theta_evidence)) {
        cat(sprintf(
            "Theta evidence: %d support point(s), strategy = %s\n",
            as.integer(x$theta_evidence$n_support),
            x$theta_evidence$strategy
        ))
        if (!is.null(x$theta_evidence$support_expansion)) {
            cat(sprintf(
                "Theta support expansion: %s, added %d guard point(s)\n",
                x$theta_evidence$support_expansion$strategy,
                as.integer(x$theta_evidence$support_expansion$added)
            ))
        }
    }
    cat("Caveat: local Gaussian old-data evidence; hyperparameter uncertainty is still simplified.\n")
    invisible(x)
}

