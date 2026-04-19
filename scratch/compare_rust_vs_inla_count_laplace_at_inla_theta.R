local_rustyinla_lib <- Sys.getenv(
    "RUSTYINLA_LIB",
    "C:/Users/Antonio/Documents/rustyINLA/rustyINLA/scratch/rlib"
)
if (nzchar(local_rustyinla_lib)) {
    .libPaths(c(
        normalizePath(local_rustyinla_lib, winslash = "/", mustWork = TRUE),
        .libPaths()
    ))
}

suppressPackageStartupMessages({
    library(INLA)
    library(CASdatasets)
    library(rustyINLA)
})

options(digits = 15)

data(freMTPL2freq)

df <- freMTPL2freq
df$VehBrand <- as.factor(df$VehBrand)
df$Region <- as.factor(df$Region)

ctrl_compute <- list(config = TRUE, dic = FALSE, waic = FALSE, cpo = FALSE)
ctrl_predictor <- list(compute = TRUE)
ctrl_inla <- list(strategy = "auto", int.strategy = "auto")

build_rust_eta <- function(spec, raw) {
    n_data <- length(spec$y)
    eta <- numeric(n_data)

    if (!is.null(spec$fixed_matrix) && spec$n_fixed > 0L) {
        x_mat <- matrix(spec$fixed_matrix, nrow = n_data, ncol = spec$n_fixed)
        eta <- eta + as.vector(x_mat %*% raw$fixed_means)
    }

    if (!is.null(spec$a_i) && !is.null(spec$a_j) && !is.null(spec$a_x)) {
        for (k in seq_along(spec$a_i)) {
            eta[spec$a_i[[k]] + 1L] <- eta[spec$a_i[[k]] + 1L] +
                spec$a_x[[k]] * raw$prior_mean[[spec$a_j[[k]] + 1L]]
        }
    } else if (spec$n_latent > 0L) {
        n_shared <- min(n_data, spec$n_latent)
        eta[seq_len(n_shared)] <- eta[seq_len(n_shared)] + raw$prior_mean[seq_len(n_shared)]
    }

    if (!is.null(spec$offset)) {
        eta <- eta + spec$offset
    }

    eta
}

plugin_loglik <- function(family, y, eta, theta_internal) {
    mu <- exp(eta)

    if (identical(family, "poisson")) {
        return(sum(dpois(y, lambda = mu, log = TRUE)))
    }

    if (identical(family, "zeroinflatedpoisson1")) {
        zero_prob <- plogis(theta_internal[[length(theta_internal)]])
        loglik <- ifelse(
            y == 0,
            log(zero_prob + (1.0 - zero_prob) * exp(-mu)),
            log1p(-zero_prob) + dpois(y, lambda = mu, log = TRUE)
        )
        return(sum(loglik))
    }

    stop(sprintf("Unsupported family: %s", family))
}

fit_rust_case <- function(formula, family, theta_override = NULL, max_evals = NULL) {
    spec <- rustyINLA:::build_backend_spec(formula, data = df, family = family)
    if (!is.null(theta_override)) {
        spec$theta_init <- as.numeric(theta_override)
    }
    if (!is.null(max_evals)) {
        spec$optimizer_max_evals <- as.integer(max_evals)
    }

    elapsed <- system.time(raw <- rust_inla_run(spec))
    if (is.character(raw)) {
        stop(raw)
    }

    list(
        spec = spec,
        raw = raw,
        elapsed_sec = unname(elapsed["elapsed"])
    )
}

fit_inla_case <- function(formula, family) {
    elapsed <- system.time(fit <- inla(
        formula,
        family = family,
        data = df,
        control.compute = ctrl_compute,
        control.predictor = ctrl_predictor,
        control.inla = ctrl_inla,
        num.threads = 1,
        verbose = FALSE
    ))

    list(
        fit = fit,
        elapsed_sec = unname(elapsed["elapsed"])
    )
}

extract_inla_theta <- function(case_name, inla_fit) {
    theta_values <- as.numeric(inla_fit$mode$theta)
    theta_names <- names(inla_fit$mode$theta)

    pick_one <- function(pattern) {
        hit <- grep(pattern, theta_names, ignore.case = TRUE)
        if (length(hit) != 1L) {
            stop(sprintf(
                "Could not uniquely match INLA mode theta '%s' for case '%s'",
                pattern,
                case_name
            ))
        }
        theta_values[[hit]]
    }

    if (identical(case_name, "zip_iid")) {
        return(c(
            pick_one("VehBrand"),
            pick_one("zero-probability")
        ))
    }

    if (identical(case_name, "poisson_iid")) {
        return(c(pick_one("VehBrand")))
    }

    if (identical(case_name, "poisson_multi_iid")) {
        return(c(
            pick_one("VehBrand"),
            pick_one("Region")
        ))
    }

    stop(sprintf("Unknown case_name: %s", case_name))
}

extract_inla_theta_order <- function(case_name, inla_fit) {
    theta_names <- names(inla_fit$mode$theta)

    pick_one <- function(pattern) {
        hit <- grep(pattern, theta_names, ignore.case = TRUE)
        if (length(hit) != 1L) {
            stop(sprintf(
                "Could not uniquely match INLA config theta '%s' for case '%s'",
                pattern,
                case_name
            ))
        }
        hit[[1]]
    }

    if (identical(case_name, "zip_iid")) {
        return(c(
            pick_one("VehBrand"),
            pick_one("zero-probability")
        ))
    }

    if (identical(case_name, "poisson_iid")) {
        return(c(pick_one("VehBrand")))
    }

    if (identical(case_name, "poisson_multi_iid")) {
        return(c(
            pick_one("VehBrand"),
            pick_one("Region")
        ))
    }

    stop(sprintf("Unknown case_name: %s", case_name))
}

evaluate_inla_prior_density <- function(prior, param, theta) {
    prior_name <- as.character(prior[[1]])

    if (identical(prior_name, "loggamma")) {
        return(dgamma(exp(theta), shape = param[[1]], rate = param[[2]], log = TRUE) + theta)
    }

    if (identical(prior_name, "gaussian") || identical(prior_name, "normal")) {
        if (param[[2]] == 0) {
            return(0.0)
        }
        return(dnorm(theta, mean = param[[1]], sd = sqrt(1 / param[[2]]), log = TRUE))
    }

    if (identical(prior_name, "none")) {
        return(0.0)
    }

    stop(sprintf("Unsupported INLA prior '%s' in mode-theta comparison", prior_name))
}

extract_inla_prior_terms <- function(inla_fit) {
    theta_values <- as.numeric(inla_fit$mode$theta)
    theta_names <- names(inla_fit$mode$theta)
    h <- inla_fit$all.hyper
    rows <- list()
    theta_index <- 0L

    add_term <- function(section, component, hyper) {
        if (isTRUE(hyper$fixed)) {
            return(invisible(NULL))
        }

        theta_index <<- theta_index + 1L
        rows[[length(rows) + 1L]] <<- data.frame(
            theta_index = theta_index,
            theta_name = theta_names[[theta_index]],
            section = section,
            component = component,
            hyper_name = hyper$name,
            prior = as.character(hyper$prior[[1]]),
            param_1 = if (length(hyper$param) >= 1L) hyper$param[[1]] else NA_real_,
            param_2 = if (length(hyper$param) >= 2L) hyper$param[[2]] else NA_real_,
            theta_value = theta_values[[theta_index]],
            log_prior = evaluate_inla_prior_density(
                hyper$prior,
                hyper$param,
                theta_values[[theta_index]]
            ),
            stringsAsFactors = FALSE
        )
    }

    for (nm in names(h)) {
        h2 <- h[[nm]]
        if (nm %in% c("predictor", "fixed", "linear")) {
            next
        }

        for (i in seq_along(h2)) {
            h3 <- h2[[i]]
            component_label <- if (!is.null(h3$label)) {
                as.character(h3$label[[1]])
            } else if (!is.null(h3$hyperid)) {
                as.character(h3$hyperid[[1]])
            } else {
                sprintf("%s_%d", nm, i)
            }

            if (identical(nm, "lp.scale")) {
                add_term(nm, component_label, h3)
            } else {
                for (j in seq_along(h3$hyper)) {
                    add_term(nm, component_label, h3$hyper[[j]])
                }
            }
        }
    }

    do.call(rbind, rows)
}

extract_inla_config_logpost <- function(case_name, inla_fit) {
    configs <- inla_fit$misc$configs$config
    if (length(configs) == 0L) {
        stop("INLA fit does not contain config objects.")
    }

    theta_order <- extract_inla_theta_order(case_name, inla_fit)
    theta_dim <- length(as.numeric(configs[[1]]$theta))
    theta_mat <- matrix(
        unlist(lapply(configs, function(cfg) as.numeric(cfg$theta))),
        ncol = theta_dim,
        byrow = TRUE
    )[, theta_order, drop = FALSE]
    log_post <- vapply(configs, function(cfg) as.numeric(cfg$log.posterior), numeric(1))

    list(
        theta = theta_mat,
        log_post = log_post,
        centered_log_post = log_post - max(log_post)
    )
}

fit_rust_at_theta_exact <- function(
    formula,
    family,
    theta_target,
    latent_init = NULL,
    fixed_init = NULL
) {
    spec <- rustyINLA:::build_backend_spec(formula, data = df, family = family)
    spec$theta_init <- as.numeric(theta_target)
    spec$optimizer_max_evals <- 0L
    spec$skip_ccd <- TRUE
    if (!is.null(latent_init)) {
        spec$latent_init <- as.numeric(latent_init)
    }
    if (!is.null(fixed_init)) {
        spec$fixed_init <- as.numeric(fixed_init)
    }

    elapsed <- system.time(raw <- rust_inla_run(spec))
    if (is.character(raw)) {
        stop(raw)
    }

    list(
        raw = raw,
        elapsed_sec = unname(elapsed["elapsed"])
    )
}

fit_inla_user_design <- function(formula, family, theta_design_inla_order, mode_result) {
    design <- cbind(theta_design_inla_order, rep(1.0, nrow(theta_design_inla_order)))
    control_inla_user <- utils::modifyList(
        ctrl_inla,
        list(
            int.strategy = "user",
            int.design = design
        )
    )

    elapsed <- system.time(fit <- inla(
        formula,
        family = family,
        data = df,
        control.compute = ctrl_compute,
        control.predictor = ctrl_predictor,
        control.inla = control_inla_user,
        control.mode = list(result = mode_result, restart = FALSE),
        num.threads = 1,
        verbose = FALSE
    ))

    list(
        fit = fit,
        elapsed_sec = unname(elapsed["elapsed"])
    )
}

match_theta_rows <- function(target_theta, candidate_theta) {
    n_target <- nrow(target_theta)
    n_candidate <- nrow(candidate_theta)
    used <- rep(FALSE, n_candidate)
    match_idx <- integer(n_target)
    match_dist <- numeric(n_target)

    dist_mat <- matrix(NA_real_, nrow = n_target, ncol = n_candidate)
    for (i in seq_len(n_target)) {
        dist_mat[i, ] <- sqrt(rowSums((candidate_theta - matrix(
            target_theta[i, ],
            nrow = n_candidate,
            ncol = ncol(target_theta),
            byrow = TRUE
        ))^2))
    }

    for (i in seq_len(n_target)) {
        dists <- dist_mat[i, ]
        dists[used] <- Inf
        idx <- which.min(dists)
        match_idx[[i]] <- idx
        match_dist[[i]] <- dists[[idx]]
        used[[idx]] <- TRUE
    }

    list(index = match_idx, distance = match_dist)
}

softmax_log_weights <- function(log_w) {
    log_w <- as.numeric(log_w)
    shifted <- log_w - max(log_w)
    weight <- exp(shifted)
    weight / sum(weight)
}

summarize_ccd_alignment <- function(case_name, rust_raw, inla_fit) {
    theta_dim <- length(as.numeric(rust_raw$theta_opt))
    rust_theta_mat <- matrix(
        as.numeric(rust_raw$ccd_thetas),
        ncol = theta_dim,
        byrow = TRUE
    )
    rust_log_mlik <- as.numeric(rust_raw$ccd_log_mlik)
    rust_centered <- rust_log_mlik - max(rust_log_mlik)

    inla_cfg <- extract_inla_config_logpost(case_name, inla_fit)
    nearest_idx <- integer(nrow(rust_theta_mat))
    nearest_dist <- numeric(nrow(rust_theta_mat))
    nearest_centered <- numeric(nrow(rust_theta_mat))

    for (i in seq_len(nrow(rust_theta_mat))) {
        dists <- sqrt(rowSums((inla_cfg$theta - matrix(
            rust_theta_mat[i, ],
            nrow = nrow(inla_cfg$theta),
            ncol = theta_dim,
            byrow = TRUE
        ))^2))
        nearest_idx[[i]] <- which.min(dists)
        nearest_dist[[i]] <- dists[[nearest_idx[[i]]]]
        nearest_centered[[i]] <- inla_cfg$centered_log_post[[nearest_idx[[i]]]]
    }

    data.frame(
        case = case_name,
        rust_ccd_points = nrow(rust_theta_mat),
        inla_config_points = nrow(inla_cfg$theta),
        nearest_theta_dist_max = max(nearest_dist),
        nearest_theta_dist_avg = mean(nearest_dist),
        centered_logpost_rmse = sqrt(mean((rust_centered - nearest_centered)^2)),
        centered_logpost_max_abs = max(abs(rust_centered - nearest_centered)),
        stringsAsFactors = FALSE
    )
}

summarize_ccd_common_support <- function(case_name, formula, family, rust_fit, inla_fit) {
    theta_dim <- length(as.numeric(rust_fit$raw$theta_opt))
    rust_theta_mat <- matrix(
        as.numeric(rust_fit$raw$ccd_thetas),
        ncol = theta_dim,
        byrow = TRUE
    )
    theta_order <- extract_inla_theta_order(case_name, inla_fit)
    inverse_order <- match(seq_len(theta_dim), theta_order)
    theta_inla_order_mat <- rust_theta_mat[, inverse_order, drop = FALSE]

    inla_user_fit <- fit_inla_user_design(
        formula,
        family,
        theta_inla_order_mat,
        inla_fit
    )
    inla_user_cfg <- extract_inla_config_logpost(case_name, inla_user_fit$fit)
    cfg_match <- match_theta_rows(rust_theta_mat, inla_user_cfg$theta)

    exact_rows <- vector("list", nrow(rust_theta_mat))
    for (i in seq_len(nrow(rust_theta_mat))) {
        exact_fit <- fit_rust_at_theta_exact(
            formula,
            family,
            rust_theta_mat[i, , drop = TRUE],
            latent_init = rust_fit$raw$mode_x,
            fixed_init = rust_fit$raw$mode_beta
        )
        exact_rows[[i]] <- data.frame(
            config_rank = i,
            exact_log_mlik = as.numeric(exact_fit$raw$log_mlik[[1]]),
            exact_theta_replay_max_abs = max(abs(
                as.numeric(exact_fit$raw$theta_opt) - rust_theta_mat[i, , drop = TRUE]
            )),
            exact_elapsed_sec = exact_fit$elapsed_sec,
            stringsAsFactors = FALSE
        )
    }
    exact_df <- do.call(rbind, exact_rows)

    rust_builtin_log_mlik <- as.numeric(rust_fit$raw$ccd_log_mlik)
    rust_builtin_log_weight <- if (!is.null(rust_fit$raw$ccd_log_weight)) {
        as.numeric(rust_fit$raw$ccd_log_weight)
    } else {
        rust_builtin_log_mlik
    }
    rust_base_weights <- if (!is.null(rust_fit$raw$ccd_base_weights)) {
        as.numeric(rust_fit$raw$ccd_base_weights)
    } else {
        rep(1.0, nrow(rust_theta_mat))
    }
    rust_builtin_weights <- as.numeric(rust_fit$raw$ccd_weights)
    rust_exact_log_weight <- log(pmax(rust_base_weights, 1e-300)) + exact_df$exact_log_mlik
    rust_exact_weights <- softmax_log_weights(log(pmax(rust_base_weights, 1e-300)) + exact_df$exact_log_mlik)
    rust_uniform_weights <- softmax_log_weights(exact_df$exact_log_mlik)
    inla_log_post <- inla_user_cfg$log_post[cfg_match$index]
    inla_weights <- softmax_log_weights(inla_log_post)

    data.frame(
        case = case_name,
        rust_ccd_points = nrow(rust_theta_mat),
        inla_user_elapsed_sec = inla_user_fit$elapsed_sec,
        support_match_max_dist = max(cfg_match$distance),
        support_match_avg_dist = mean(cfg_match$distance),
        exact_theta_replay_max_abs = max(exact_df$exact_theta_replay_max_abs),
        builtin_log_mlik_rmse_vs_exact = sqrt(mean((rust_builtin_log_mlik - exact_df$exact_log_mlik)^2)),
        builtin_log_weight_rmse_vs_exact = sqrt(mean((rust_builtin_log_weight - rust_exact_log_weight)^2)),
        builtin_weight_rmse_vs_exact = sqrt(mean((rust_builtin_weights - rust_exact_weights)^2)),
        exact_centered_rmse_vs_inla = sqrt(mean(((
            exact_df$exact_log_mlik - max(exact_df$exact_log_mlik)
        ) - (
            inla_log_post - max(inla_log_post)
        ))^2)),
        exact_weight_rmse_vs_inla = sqrt(mean((rust_exact_weights - inla_weights)^2)),
        exact_weight_max_abs_vs_inla = max(abs(rust_exact_weights - inla_weights)),
        uniform_weight_rmse_vs_inla = sqrt(mean((rust_uniform_weights - inla_weights)^2)),
        uniform_weight_max_abs_vs_inla = max(abs(rust_uniform_weights - inla_weights)),
        center_base_weight = rust_base_weights[[1]],
        noncenter_base_weight = if (length(rust_base_weights) > 1L) {
            rust_base_weights[[2]]
        } else {
            NA_real_
        },
        stringsAsFactors = FALSE
    )
}

summarize_rust_fit <- function(label, fit, family) {
    eta <- build_rust_eta(fit$spec, fit$raw)
    plugin <- plugin_loglik(family, df$ClaimNb, eta, as.numeric(fit$raw$theta_opt))
    terms <- fit$raw$laplace_terms

    data.frame(
        run = label,
        rust_elapsed_sec = fit$elapsed_sec,
        log_mlik = as.numeric(fit$raw$log_mlik[[1]]),
        plugin_loglik = plugin,
        correction_gap = as.numeric(fit$raw$log_mlik[[1]]) - plugin,
        det_adjustment = 0.5 * (
            as.numeric(terms$final_log_det_q) -
                as.numeric(terms$final_log_det_aug)
        ),
        quadratic_penalty = -0.5 * as.numeric(terms$final_q_form),
        log_prior = as.numeric(terms$log_prior),
        theta_laplace_correction = if (!is.null(fit$raw$theta_laplace_correction)) {
            as.numeric(fit$raw$theta_laplace_correction[[1]])
        } else {
            NA_real_
        },
        stringsAsFactors = FALSE
    )
}

run_case <- function(case_name, formula, family) {
    cat("\n============================================================\n")
    cat("Running case:", case_name, "\n")
    cat("Formula:", deparse(formula, width.cutoff = 500L), "\n")
    flush.console()

    inla_fit <- fit_inla_case(formula, family)
    inla_theta <- extract_inla_theta(case_name, inla_fit$fit)
    inla_prior_terms <- extract_inla_prior_terms(inla_fit$fit)
    rust_default <- fit_rust_case(formula, family)
    rust_at_inla_theta <- fit_rust_case(
        formula,
        family,
        theta_override = inla_theta,
        max_evals = 0L
    )

    theta_df <- data.frame(
        case = case_name,
        theta_index = seq_along(inla_theta),
        rust_theta_opt = as.numeric(rust_default$raw$theta_opt),
        rust_theta_at_inla = as.numeric(rust_at_inla_theta$raw$theta_opt),
        inla_theta = inla_theta,
        rust_minus_inla = as.numeric(rust_default$raw$theta_opt) - inla_theta,
        stringsAsFactors = FALSE
    )

    prior_df <- data.frame(
        case = case_name,
        inla_log_prior_at_mode = sum(inla_prior_terms$log_prior),
        rust_log_prior_default = as.numeric(rust_default$raw$laplace_terms$log_prior),
        rust_log_prior_at_inla_theta = as.numeric(rust_at_inla_theta$raw$laplace_terms$log_prior),
        rust_minus_inla_default = as.numeric(rust_default$raw$laplace_terms$log_prior) -
            sum(inla_prior_terms$log_prior),
        rust_minus_inla_at_inla_theta = as.numeric(rust_at_inla_theta$raw$laplace_terms$log_prior) -
            sum(inla_prior_terms$log_prior),
        stringsAsFactors = FALSE
    )

    summary_df <- rbind(
        cbind(case = case_name, summarize_rust_fit("rust_default", rust_default, family)),
        cbind(case = case_name, summarize_rust_fit("rust_at_inla_theta", rust_at_inla_theta, family))
    )

    summary_df$inla_mlik <- as.numeric(inla_fit$fit$mlik[1, 1])
    summary_df$inla_plugin_loglik <- plugin_loglik(
        family,
        df$ClaimNb,
        as.numeric(inla_fit$fit$summary.linear.predictor$mean[seq_len(nrow(df))]),
        inla_theta
    )
    summary_df$inla_correction_gap <- summary_df$inla_mlik - summary_df$inla_plugin_loglik
    summary_df$correction_gap_diff_vs_inla <- summary_df$correction_gap - summary_df$inla_correction_gap

    ccd_df <- summarize_ccd_alignment(case_name, rust_default$raw, inla_fit$fit)
    ccd_common_df <- summarize_ccd_common_support(
        case_name,
        formula,
        family,
        rust_default,
        inla_fit$fit
    )

    list(theta = theta_df, summary = summary_df, prior = prior_df, ccd = ccd_df, ccd_common = ccd_common_df)
}

case_results <- list(
    run_case(
        case_name = "zip_iid",
        formula = ClaimNb ~ 1 + offset(log(Exposure)) + f(VehBrand, model = "iid"),
        family = "zeroinflatedpoisson1"
    ),
    run_case(
        case_name = "poisson_iid",
        formula = ClaimNb ~ 1 + offset(log(Exposure)) + f(VehBrand, model = "iid"),
        family = "poisson"
    ),
    run_case(
        case_name = "poisson_multi_iid",
        formula = ClaimNb ~ 1 + offset(log(Exposure)) +
            f(VehBrand, model = "iid") +
            f(Region, model = "iid"),
        family = "poisson"
    )
)

theta_df <- do.call(rbind, lapply(case_results, `[[`, "theta"))
summary_df <- do.call(rbind, lapply(case_results, `[[`, "summary"))
prior_df <- do.call(rbind, lapply(case_results, `[[`, "prior"))
ccd_df <- do.call(rbind, lapply(case_results, `[[`, "ccd"))
ccd_common_df <- do.call(rbind, lapply(case_results, `[[`, "ccd_common"))

cat("\nTHETA COMPARISON\n")
print(theta_df, row.names = FALSE)

cat("\nPOINTWISE LAPLACE COMPARISON\n")
print(summary_df, row.names = FALSE)

cat("\nHYPERPRIOR COMPARISON\n")
print(prior_df, row.names = FALSE)

cat("\nCCD LOG-POSTERIOR ALIGNMENT\n")
print(ccd_df, row.names = FALSE)

cat("\nCCD COMMON-SUPPORT COMPARISON\n")
print(ccd_common_df, row.names = FALSE)
