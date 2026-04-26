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
df$AgeGroup <- cut(
    df$DrivAge,
    breaks = c(17, 25, 40, 60, 80, 150),
    labels = c("18-25", "26-40", "41-60", "61-80", "81+"),
    ordered_result = TRUE
)
df$AgeIndex <- as.integer(df$AgeGroup)
df$VehBrand <- as.factor(df$VehBrand)
df$Region <- as.factor(df$Region)

ctrl_compute <- list(config = TRUE, dic = FALSE, waic = FALSE, cpo = FALSE)
ctrl_predictor <- list(compute = TRUE)
ctrl_inla <- list(strategy = "auto", int.strategy = "auto")

build_rust_eta <- function(spec, raw) {
    n_data <- length(spec$y)
    eta <- numeric(n_data)

    if (!is.null(spec$fixed_matrix) && spec$n_fixed > 0L) {
        x_mat <- matrix(
            spec$fixed_matrix,
            nrow = n_data,
            ncol = spec$n_fixed
        )
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

    stop(sprintf("Unsupported family for plug-in log-likelihood: %s", family))
}

fit_rust_case <- function(formula, family) {
    spec <- rustyINLA:::build_backend_spec(formula, data = df, family = family)
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

extract_inla_latent_config_decomposition <- function(rust_spec, inla_fit) {
    configs <- inla_fit$misc$configs$config
    if (length(configs) == 0L) {
        stop("INLA fit does not contain config objects.")
    }

    target_theta <- as.numeric(inla_fit$mode$theta)
    theta_distance <- vapply(
        configs,
        function(cfg) sqrt(sum((as.numeric(cfg$theta) - target_theta)^2)),
        numeric(1)
    )
    mode_cfg_idx <- which.min(theta_distance)
    config_logpost <- vapply(configs, function(cfg) as.numeric(cfg$log.posterior), numeric(1))
    config_weights <- exp(config_logpost - max(config_logpost))
    config_weights <- config_weights / sum(config_weights)

    extract_one <- function(cfg) {
        cfg_mean <- if (!is.null(cfg$improved.mean)) {
            as.numeric(cfg$improved.mean)
        } else {
            as.numeric(cfg$mean)
        }
        cfg_diag <- pmax(as.numeric(diag(cfg$Qinv)), 0.0)
        if (length(cfg_mean) < rust_spec$n_latent || length(cfg_diag) < rust_spec$n_latent) {
            stop("INLA config mean/Qinv dimension is smaller than the Rust latent dimension.")
        }
        list(
            latent_mean = cfg_mean[seq_len(rust_spec$n_latent)],
            latent_var = cfg_diag[seq_len(rust_spec$n_latent)]
        )
    }

    mode_cfg <- extract_one(configs[[mode_cfg_idx]])
    mix_mean <- numeric(rust_spec$n_latent)
    mix_within_var <- numeric(rust_spec$n_latent)
    mix_mean_sq <- numeric(rust_spec$n_latent)

    for (idx in seq_along(configs)) {
        cfg_parts <- extract_one(configs[[idx]])
        weight <- config_weights[[idx]]
        mix_mean <- mix_mean + weight * cfg_parts$latent_mean
        mix_within_var <- mix_within_var + weight * cfg_parts$latent_var
        mix_mean_sq <- mix_mean_sq + weight * cfg_parts$latent_mean^2
    }

    mix_between_var <- pmax(mix_mean_sq - mix_mean^2, 0.0)

    list(
        mode_config_index = mode_cfg_idx,
        mode_theta_distance = theta_distance[[mode_cfg_idx]],
        config_weights = config_weights,
        mode_latent_var = mode_cfg$latent_var,
        mix_within_var = mix_within_var,
        mix_between_var = mix_between_var,
        mix_final_var = mix_within_var + mix_between_var
    )
}

summarize_latent_blocks <- function(case_name, rust_spec, rust_raw, inla_fit, inla_cfg) {
    if (length(rust_spec$latent_blocks) == 0L) {
        return(data.frame())
    }

    rows <- vector("list", length(rust_spec$latent_blocks))
    for (idx in seq_along(rust_spec$latent_blocks)) {
        block <- rust_spec$latent_blocks[[idx]]
        block_index <- seq.int(block$start + 1L, length.out = block$n_levels)

        rust_mean <- as.numeric(rust_raw$prior_mean[block_index])
        rust_theta_opt_sd <- sqrt(pmax(as.numeric(rust_raw$latent_var_theta_opt[block_index]), 0.0))
        rust_within_var <- pmax(as.numeric(rust_raw$latent_var_within_theta[block_index]), 0.0)
        rust_between_var <- pmax(as.numeric(rust_raw$latent_var_between_theta[block_index]), 0.0)
        rust_sd <- sqrt(pmax(as.numeric(rust_raw$marg_vars[block_index]), 0.0))
        inla_df <- inla_fit$summary.random[[block$covariate_name]]
        inla_mean <- as.numeric(inla_df$mean)
        inla_sd <- as.numeric(inla_df$sd)
        inla_mode_cfg_sd <- sqrt(pmax(inla_cfg$mode_latent_var[block_index], 0.0))
        inla_mix_within_var <- pmax(inla_cfg$mix_within_var[block_index], 0.0)
        inla_mix_between_var <- pmax(inla_cfg$mix_between_var[block_index], 0.0)
        inla_mix_final_sd <- sqrt(pmax(inla_cfg$mix_final_var[block_index], 0.0))

        rust_within_var_avg <- mean(rust_within_var)
        rust_between_var_avg <- mean(rust_between_var)
        inla_mix_within_var_avg <- mean(inla_mix_within_var)
        inla_mix_between_var_avg <- mean(inla_mix_between_var)

        rows[[idx]] <- data.frame(
            case = case_name,
            block = block$covariate_name,
            model = block$model,
            n_levels = block$n_levels,
            mean_rmse = sqrt(mean((rust_mean - inla_mean)^2)),
            rust_theta_opt_sd_avg = mean(rust_theta_opt_sd),
            inla_mode_cfg_sd_avg = mean(inla_mode_cfg_sd),
            conditional_sd_ratio = mean(rust_theta_opt_sd) / mean(inla_mode_cfg_sd),
            conditional_sd_rmse = sqrt(mean((rust_theta_opt_sd - inla_mode_cfg_sd)^2)),
            rust_within_var_avg = rust_within_var_avg,
            rust_between_var_avg = rust_between_var_avg,
            rust_between_share = rust_between_var_avg / (rust_within_var_avg + rust_between_var_avg),
            rust_sd_avg = mean(rust_sd),
            inla_sd_avg = mean(inla_sd),
            sd_ratio = mean(rust_sd) / mean(inla_sd),
            sd_rmse = sqrt(mean((rust_sd - inla_sd)^2)),
            sd_max_abs = max(abs(rust_sd - inla_sd)),
            inla_mix_within_var_avg = inla_mix_within_var_avg,
            inla_mix_between_var_avg = inla_mix_between_var_avg,
            inla_mix_between_share = inla_mix_between_var_avg /
                (inla_mix_within_var_avg + inla_mix_between_var_avg),
            inla_mix_final_sd_avg = mean(inla_mix_final_sd),
            inla_mix_final_sd_rmse_vs_summary = sqrt(mean((inla_mix_final_sd - inla_sd)^2)),
            stringsAsFactors = FALSE
        )
    }

    do.call(rbind, rows)
}

summarize_correction_gap <- function(case_name, family, rust_fit, inla_fit, inla_cfg) {
    rust_theta <- as.numeric(rust_fit$raw$theta_opt)
    inla_theta <- extract_inla_theta(case_name, inla_fit$fit)
    rust_eta <- build_rust_eta(rust_fit$spec, rust_fit$raw)
    inla_eta <- as.numeric(inla_fit$fit$summary.linear.predictor$mean[seq_len(nrow(df))])
    rust_plugin <- plugin_loglik(family, df$ClaimNb, rust_eta, rust_theta)
    inla_plugin <- plugin_loglik(family, df$ClaimNb, inla_eta, inla_theta)
    rust_terms <- rust_fit$raw$laplace_terms
    ccd_weights <- as.numeric(rust_fit$raw$ccd_weights)
    ccd_log_mlik <- if (!is.null(rust_fit$raw$ccd_log_mlik)) {
        as.numeric(rust_fit$raw$ccd_log_mlik)
    } else {
        rep(as.numeric(rust_fit$raw$log_mlik[[1]]), length(ccd_weights))
    }
    ccd_hessian_eigenvalues <- if (!is.null(rust_fit$raw$ccd_hessian_eigenvalues)) {
        as.numeric(rust_fit$raw$ccd_hessian_eigenvalues)
    } else {
        numeric(0)
    }
    rust_mlik_theta_opt <- if (!is.null(rust_fit$raw$log_mlik_theta_opt)) {
        as.numeric(rust_fit$raw$log_mlik_theta_opt[[1]])
    } else {
        as.numeric(rust_fit$raw$log_mlik[[1]])
    }
    rust_theta_laplace_correction <- if (!is.null(rust_fit$raw$theta_laplace_correction)) {
        as.numeric(rust_fit$raw$theta_laplace_correction[[1]])
    } else {
        NA_real_
    }
    rust_mlik_theta_laplace <- if (!is.null(rust_fit$raw$log_mlik_theta_laplace)) {
        as.numeric(rust_fit$raw$log_mlik_theta_laplace[[1]])
    } else {
        NA_real_
    }
    inla_mlik <- as.numeric(inla_fit$fit$mlik[1, 1])
    inla_correction_gap <- inla_mlik - inla_plugin

    data.frame(
        case = case_name,
        rust_elapsed_sec = rust_fit$elapsed_sec,
        inla_elapsed_sec = inla_fit$elapsed_sec,
        rust_mlik = rust_mlik_theta_opt,
        rust_mlik_theta_opt = rust_mlik_theta_opt,
        rust_mlik_theta_laplace = rust_mlik_theta_laplace,
        rust_theta_laplace_correction = rust_theta_laplace_correction,
        inla_mlik = inla_mlik,
        rust_plugin_loglik = rust_plugin,
        inla_plugin_loglik = inla_plugin,
        plugin_gap_rust_minus_inla = rust_plugin - inla_plugin,
        rust_correction_gap = rust_mlik_theta_opt - rust_plugin,
        rust_correction_gap_theta_opt = rust_mlik_theta_opt - rust_plugin,
        rust_correction_gap_theta_laplace = rust_mlik_theta_laplace - rust_plugin,
        inla_correction_gap = inla_correction_gap,
        correction_gap_diff = (rust_mlik_theta_opt - rust_plugin) - inla_correction_gap,
        correction_gap_theta_laplace_diff = (rust_mlik_theta_laplace - rust_plugin) - inla_correction_gap,
        rust_det_adjustment = 0.5 * (
            as.numeric(rust_terms$final_log_det_q) -
                as.numeric(rust_terms$final_log_det_aug)
        ),
        rust_quadratic_penalty = -0.5 * as.numeric(rust_terms$final_q_form),
        rust_log_prior = as.numeric(rust_terms$log_prior),
        rust_ccd_points = length(ccd_weights),
        rust_ccd_top_weight = max(ccd_weights),
        rust_ccd_effective_n = 1.0 / sum(ccd_weights^2),
        rust_ccd_log_mlik_span = max(ccd_log_mlik) - min(ccd_log_mlik),
        rust_theta_hessian_logdet = sum(log(pmax(ccd_hessian_eigenvalues, 1e-12))),
        inla_config_points = length(inla_cfg$config_weights),
        inla_config_top_weight = max(inla_cfg$config_weights),
        inla_config_effective_n = 1.0 / sum(inla_cfg$config_weights^2),
        inla_mode_config_index = inla_cfg$mode_config_index,
        inla_mode_theta_distance = inla_cfg$mode_theta_distance,
        stringsAsFactors = FALSE
    )
}

run_case <- function(case_name, formula, family) {
    cat("\n============================================================\n")
    cat("Running case:", case_name, "\n")
    cat("Formula:", deparse(formula, width.cutoff = 500L), "\n")
    flush.console()

    rust_fit <- fit_rust_case(formula, family)
    inla_fit <- fit_inla_case(formula, family)
    inla_cfg <- extract_inla_latent_config_decomposition(rust_fit$spec, inla_fit$fit)

    list(
        correction = summarize_correction_gap(case_name, family, rust_fit, inla_fit, inla_cfg),
        latent = summarize_latent_blocks(
            case_name,
            rust_fit$spec,
            rust_fit$raw,
            inla_fit$fit,
            inla_cfg
        )
    )
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

correction_df <- do.call(rbind, lapply(case_results, `[[`, "correction"))
latent_df <- do.call(rbind, lapply(case_results, `[[`, "latent"))

cat("\nCORRECTION GAP SUMMARY\n")
print(correction_df, row.names = FALSE)

cat("\nLATENT SD SUMMARY\n")
print(latent_df, row.names = FALSE)
