source(file.path(getwd(), "tools", "load_worktree_package.R"), local = TRUE)
load_rustyinla_for_benchmarks(getwd())

timed_fit <- function(expr) {
    value <- NULL
    elapsed <- system.time({
        value <- eval.parent(substitute(expr))
    })[["elapsed"]]
    list(value = value, elapsed = as.numeric(elapsed))
}

named_random <- function(fit, block) {
    df <- fit$summary.random[[block]]
    stats::setNames(as.numeric(df$mean), rownames(df))
}

max_fitted_rel_diff <- function(fit, joint_fit, joint_rows) {
    lhs <- as.numeric(fit$summary.fitted.values$mean)
    rhs <- as.numeric(joint_fit$summary.fitted.values$mean[joint_rows])
    max(abs(lhs - rhs) / pmax(1.0, abs(rhs)))
}

max_random_diff <- function(fit, joint_fit, blocks) {
    max(vapply(blocks, function(block) {
        lhs <- named_random(fit, block)
        rhs <- named_random(joint_fit, block)
        shared <- intersect(names(lhs), names(rhs))
        max(abs(lhs[shared] - rhs[shared]))
    }, numeric(1)))
}

theta_scale_diagnostics <- function(fit, joint_fit, k) {
    theta_names <- rownames(fit$summary.hyperpar.internal)[seq_len(k)]
    internal_fit <- fit$summary.hyperpar.internal[theta_names, , drop = FALSE]
    internal_joint <- joint_fit$summary.hyperpar.internal[theta_names, , drop = FALSE]
    precision_fit <- fit$summary.hyperpar[theta_names, , drop = FALSE]
    precision_joint <- joint_fit$summary.hyperpar[theta_names, , drop = FALSE]

    log_mean_diff <- abs(as.numeric(internal_fit$mean) - as.numeric(internal_joint$mean))
    log_mode_diff <- abs(as.numeric(internal_fit$mode) - as.numeric(internal_joint$mode))
    precision_rel_diff <- abs(as.numeric(precision_fit$mean) - as.numeric(precision_joint$mean)) /
        pmax(abs(as.numeric(precision_joint$mean)), 1e-12)
    fit_block_sd <- exp(-0.5 * as.numeric(internal_fit$mean))
    joint_block_sd <- exp(-0.5 * as.numeric(internal_joint$mean))
    block_sd_rel_diff <- abs(fit_block_sd - joint_block_sd) / pmax(abs(joint_block_sd), 1e-12)
    z_diff <- log_mean_diff / pmax(
        sqrt(as.numeric(internal_fit$sd)^2 + as.numeric(internal_joint$sd)^2),
        1e-12
    )

    list(
        max_log_mean = max(log_mean_diff),
        max_log_mode = max(log_mode_diff),
        max_precision_rel = max(precision_rel_diff),
        max_block_sd_rel = max(block_sd_rel_diff),
        max_log_z = max(z_diff)
    )
}

make_k_iid_batch <- function(n, k, level_specs, seed, shift = 0.0, old = FALSE) {
    set.seed(seed)
    x <- stats::rnorm(n)
    expo <- stats::runif(n, 0.4, 2.2)
    eta <- -1.05 + shift + 0.28 * x
    out <- data.frame(y = numeric(n), x = x, expo = expo)

    for (idx in seq_len(k)) {
        spec <- level_specs[[idx]]
        active_levels <- if (old) spec$old_levels else spec$all_levels
        values <- sample(active_levels, n, replace = TRUE)
        eta <- eta + unname(spec$effects[values])
        out[[spec$name]] <- factor(values, levels = active_levels)
    }

    out$y <- stats::rpois(n, lambda = expo * exp(eta))
    out
}

build_formula <- function(blocks) {
    rhs <- c(
        "1",
        "x",
        "offset(log(expo))",
        sprintf("f(%s, model = \"iid\")", blocks)
    )
    stats::as.formula(sprintf("y ~ %s", paste(rhs, collapse = " + ")))
}

run_k_iid_case <- function(k) {
    level_specs <- list(
        list(
            name = "g1",
            old_levels = c("A", "B"),
            all_levels = c("A", "B", "C"),
            effects = c(A = -0.25, B = 0.10, C = 0.35),
            born = "C"
        ),
        list(
            name = "g2",
            old_levels = c("U", "V", "W"),
            all_levels = c("U", "V", "W", "X"),
            effects = c(U = -0.20, V = 0.18, W = 0.38, X = -0.30),
            born = "X"
        ),
        list(
            name = "g3",
            old_levels = c("K", "L", "M"),
            all_levels = c("K", "L", "M", "N"),
            effects = c(K = -0.15, L = 0.12, M = 0.30, N = -0.25),
            born = "N"
        ),
        list(
            name = "g4",
            old_levels = c("R", "S", "T"),
            all_levels = c("R", "S", "T", "Q"),
            effects = c(R = -0.18, S = 0.08, T = 0.26, Q = -0.22),
            born = "Q"
        )
    )[seq_len(k)]
    blocks <- vapply(level_specs, `[[`, character(1), "name")

    old_data <- make_k_iid_batch(
        n = 220L + 60L * k,
        k = k,
        level_specs = level_specs,
        seed = 91000L + k,
        old = TRUE
    )
    new_data <- make_k_iid_batch(
        n = 120L + 30L * k,
        k = k,
        level_specs = level_specs,
        seed = 92000L + k,
        shift = 0.03,
        old = FALSE
    )
    joint_data <- rbind(old_data, new_data)
    for (spec in level_specs) {
        joint_data[[spec$name]] <- factor(joint_data[[spec$name]], levels = spec$all_levels)
    }

    formula <- build_formula(blocks)
    old_run <- timed_fit(rusty_inla(formula, data = old_data, family = "poisson"))
    old_fit <- old_run$value
    state_ccd_run <- timed_fit(rusty_update_state(old_fit, theta_support_expansion = "none"))
    state_ccd <- state_ccd_run$value
    state_run <- timed_fit(rusty_update_state(old_fit))
    state <- state_run$value

    if (length(state$iid_blocks) != k) {
        stop(sprintf("Expected a %d-iid update state.", k), call. = FALSE)
    }
    state_block_names <- unname(vapply(state$iid_blocks, `[[`, character(1), "covariate_name"))
    if (!identical(state_block_names, blocks)) {
        stop(sprintf("%d-iid state did not preserve iid block ordering.", k), call. = FALSE)
    }
    if (k == 1L && length(state$latent_pair_precision$x) != 0L) {
        stop("One-iid state should not carry iid-iid cross edges.", call. = FALSE)
    }
    if (k > 1L && length(state$latent_pair_precision$x) == 0L) {
        stop(sprintf("%d-iid state should carry sparse iid-iid cross edges.", k), call. = FALSE)
    }
    if (k > 1L && (
        is.null(state$theta_evidence$support_expansion) ||
            as.integer(state$theta_evidence$support_expansion$added) <= 0L ||
            as.integer(state$theta_evidence$n_support) <= as.integer(state_ccd$theta_evidence$n_support)
    )) {
        stop(sprintf("%d-iid state should add expanded theta guard support.", k), call. = FALSE)
    }

    source_run <- timed_fit(rusty_inla(
        formula,
        data = new_data,
        family = "poisson",
        control.update = list(
            state = state,
            mode = "fixed_iid_cross_gaussian_evidence"
        )
    ))
    source_fit <- source_run$value
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
    joint_run <- timed_fit(rusty_inla(formula, data = joint_data, family = "poisson"))
    joint_fit <- joint_run$value

    metadata <- source_fit$posterior_state_used
    if (!identical(as.character(metadata$source_iid_covariate_names), blocks)) {
        stop(sprintf("%d-iid metadata did not preserve iid covariate names.", k), call. = FALSE)
    }
    for (spec in level_specs) {
        born <- metadata$born_iid_levels_by_block[[spec$name]]
        if (!(spec$born %in% born)) {
            stop(sprintf("Born level was not recorded for block %s.", spec$name), call. = FALSE)
        }
    }
    if (k > 1L &&
        !identical(theta_fit$posterior_state_used$theta_evidence_solver_status, "guarded_shepard_nd_integrated")) {
        stop(sprintf("%d-iid theta evidence should use the guarded multidimensional path.", k), call. = FALSE)
    }

    joint_new_rows <- seq.int(nrow(old_data) + 1L, nrow(joint_data))
    source_fitted <- max_fitted_rel_diff(source_fit, joint_fit, joint_new_rows)
    theta_fitted <- max_fitted_rel_diff(theta_fit, joint_fit, joint_new_rows)
    source_theta <- max(abs(
        as.numeric(source_fit$summary.hyperpar$mean[seq_len(k)]) -
            as.numeric(joint_fit$summary.hyperpar$mean[seq_len(k)])
    ))
    theta_dynamic <- max(abs(
        as.numeric(theta_fit$summary.hyperpar$mean[seq_len(k)]) -
            as.numeric(joint_fit$summary.hyperpar$mean[seq_len(k)])
    ))
    source_fixed <- max(abs(
        as.numeric(source_fit$summary.fixed$mean) -
            as.numeric(joint_fit$summary.fixed$mean)
    ))
    theta_fixed <- max(abs(
        as.numeric(theta_fit$summary.fixed$mean) -
            as.numeric(joint_fit$summary.fixed$mean)
    ))
    source_random <- max_random_diff(source_fit, joint_fit, blocks)
    theta_random <- max_random_diff(theta_fit, joint_fit, blocks)
    source_theta_scale <- theta_scale_diagnostics(source_fit, joint_fit, k)
    theta_scale <- theta_scale_diagnostics(theta_fit, joint_fit, k)

    cat(sprintf(
        paste(
            "posterior_state_k_iid_evidence[K=%d]:",
            "old %.3fs; state_ccd %.3fs; state_guard %.3fs;",
            "support %d -> %d;",
            "source %.3fs; theta_dynamic %.3fs; joint %.3fs;",
            "source theta %.6f; dynamic theta %.6f;",
            "source log_theta %.6f; dynamic log_theta %.6f;",
            "source sd_rel %.6f; dynamic sd_rel %.6f;",
            "source fixed %.6f; dynamic fixed %.6f;",
            "source random %.6f; dynamic random %.6f;",
            "source fitted_rel %.6f; dynamic fitted_rel %.6f\n"
        ),
        k,
        old_run$elapsed,
        state_ccd_run$elapsed,
        state_run$elapsed,
        as.integer(state_ccd$theta_evidence$n_support),
        as.integer(state$theta_evidence$n_support),
        source_run$elapsed,
        theta_run$elapsed,
        joint_run$elapsed,
        source_theta,
        theta_dynamic,
        source_theta_scale$max_log_mean,
        theta_scale$max_log_mean,
        source_theta_scale$max_block_sd_rel,
        theta_scale$max_block_sd_rel,
        source_fixed,
        theta_fixed,
        source_random,
        theta_random,
        source_fitted,
        theta_fitted
    ))

    diagnostics <- c(
        source_theta,
        theta_dynamic,
        source_fixed,
        theta_fixed,
        source_random,
        theta_random,
        source_fitted,
        theta_fitted,
        source_theta_scale$max_log_mean,
        theta_scale$max_log_mean,
        source_theta_scale$max_block_sd_rel,
        theta_scale$max_block_sd_rel,
        source_theta_scale$max_log_z,
        theta_scale$max_log_z
    )
    if (any(!is.finite(diagnostics))) {
        stop(sprintf("%d-iid update produced non-finite diagnostics.", k), call. = FALSE)
    }
    if (source_fitted > 0.18 || source_fixed > 0.18 || source_random > 0.30) {
        stop(sprintf("%d-iid source update drift exceeded the synthetic tolerance.", k), call. = FALSE)
    }
    if (theta_fitted > 0.18 || theta_fixed > 0.18 || theta_random > 0.30) {
        stop(sprintf("%d-iid theta update drift exceeded the synthetic tolerance.", k), call. = FALSE)
    }

    invisible(NULL)
}

for (k in 1:4) {
    run_k_iid_case(k)
}

cat("posterior_state_k_iid_evidence: PASS\n")
