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
data(freMTPL2sev)

df_freq <- freMTPL2freq
df_freq$AgeGroup <- cut(
    df_freq$DrivAge,
    breaks = c(17, 25, 40, 60, 80, 150),
    labels = c("18-25", "26-40", "41-60", "61-80", "81+"),
    ordered_result = TRUE
)

df <- merge(
    freMTPL2sev,
    df_freq[, c("IDpol", "AgeGroup")],
    by = "IDpol",
    all.x = FALSE,
    all.y = FALSE
)
df$AgeGroup <- factor(
    df$AgeGroup,
    levels = levels(df_freq$AgeGroup),
    ordered = is.ordered(df_freq$AgeGroup)
)

sev_cutoff <- as.numeric(stats::quantile(
    df$ClaimAmount[df$ClaimAmount > 0],
    probs = 0.90,
    na.rm = TRUE
))
df <- df[df$ClaimAmount > 0 & df$ClaimAmount <= sev_cutoff, ]

ctrl_compute <- list(config = TRUE, dic = FALSE, waic = FALSE, cpo = FALSE)
ctrl_predictor <- list(compute = TRUE)
ctrl_inla <- list(strategy = "auto", int.strategy = "auto")

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

fitted_rel_summary <- function(rust_fit, inla_fit) {
    rust_mean <- as.numeric(rust_fit$raw$fitted_mean)
    inla_mean <- as.numeric(inla_fit$fit$summary.fitted.values$mean[seq_len(length(rust_mean))])
    rel <- abs(rust_mean - inla_mean) / pmax(1.0, abs(inla_mean))
    data.frame(
        fitted_mean_max_rel = max(rel),
        fitted_mean_rmse = sqrt(mean((rust_mean - inla_mean)^2)),
        rust_fitted_mean_max = max(rust_mean),
        inla_fitted_mean_max = max(inla_mean),
        stringsAsFactors = FALSE
    )
}

summarize_replay_centering <- function(formula, family, rust_fit, block_index) {
    theta_dim <- length(as.numeric(rust_fit$raw$theta_opt))
    theta_mat <- matrix(
        as.numeric(rust_fit$raw$ccd_thetas),
        ncol = theta_dim,
        byrow = TRUE
    )
    weights <- as.numeric(rust_fit$raw$ccd_weights)

    replay_rows <- vector("list", nrow(theta_mat))
    latent_mean_mat <- matrix(NA_real_, nrow = nrow(theta_mat), ncol = length(block_index))
    latent_var_mat <- matrix(NA_real_, nrow = nrow(theta_mat), ncol = length(block_index))
    centered_mean_mat <- matrix(NA_real_, nrow = nrow(theta_mat), ncol = length(block_index))
    level_shift <- numeric(nrow(theta_mat))
    fixed_intercept <- numeric(nrow(theta_mat))

    for (i in seq_len(nrow(theta_mat))) {
        replay_fit <- fit_rust_at_theta_exact(
            formula,
            family,
            theta_mat[i, , drop = TRUE],
            latent_init = rust_fit$raw$mode_x,
            fixed_init = rust_fit$raw$mode_beta
        )
        latent_mean <- as.numeric(replay_fit$raw$prior_mean[block_index])
        latent_var <- pmax(as.numeric(replay_fit$raw$latent_var_theta_opt[block_index]), 0.0)
        level_shift[[i]] <- mean(latent_mean)
        centered_mean <- latent_mean - level_shift[[i]]
        fixed_intercept[[i]] <- if (length(replay_fit$raw$fixed_means) > 0L) {
            as.numeric(replay_fit$raw$fixed_means[[1]])
        } else {
            NA_real_
        }

        latent_mean_mat[i, ] <- latent_mean
        latent_var_mat[i, ] <- latent_var
        centered_mean_mat[i, ] <- centered_mean

        replay_rows[[i]] <- data.frame(
            ccd_rank = i,
            theta_replay_max_abs = max(abs(
                as.numeric(replay_fit$raw$theta_opt) - theta_mat[i, , drop = TRUE]
            )),
            stringsAsFactors = FALSE
        )
    }

    replay_df <- do.call(rbind, replay_rows)
    weight_mat <- matrix(weights, nrow = nrow(latent_mean_mat), ncol = ncol(latent_mean_mat))

    within_var <- colSums(weight_mat * latent_var_mat)

    unc_mean <- colSums(weight_mat * latent_mean_mat)
    unc_mean_sq <- colSums(weight_mat * latent_mean_mat^2)
    unc_between_var <- pmax(unc_mean_sq - unc_mean^2, 0.0)

    ctr_mean <- colSums(weight_mat * centered_mean_mat)
    ctr_mean_sq <- colSums(weight_mat * centered_mean_mat^2)
    ctr_between_var <- pmax(ctr_mean_sq - ctr_mean^2, 0.0)

    data.frame(
        replay_theta_max_abs = max(replay_df$theta_replay_max_abs),
        replay_within_var_avg = mean(within_var),
        replay_between_var_avg_uncentered = mean(unc_between_var),
        replay_between_var_avg_centered = mean(ctr_between_var),
        replay_sd_avg_uncentered = mean(sqrt(within_var + unc_between_var)),
        replay_sd_avg_centered = mean(sqrt(within_var + ctr_between_var)),
        replay_level_shift_sd = stats::sd(level_shift),
        replay_level_shift_range = diff(range(level_shift)),
        replay_intercept_sd = stats::sd(fixed_intercept),
        replay_location_sd = stats::sd(level_shift + fixed_intercept),
        replay_shift_intercept_cor = stats::cor(level_shift, fixed_intercept),
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

    block <- rust_fit$spec$latent_blocks[[1]]
    block_index <- seq.int(block$start + 1L, length.out = block$n_levels)

    rust_mean <- as.numeric(rust_fit$raw$marg_means[block_index])
    rust_sd <- sqrt(pmax(as.numeric(rust_fit$raw$marg_vars[block_index]), 0.0))
    inla_df <- inla_fit$fit$summary.random[[block$covariate_name]]
    inla_mean <- as.numeric(inla_df$mean)
    inla_sd <- as.numeric(inla_df$sd)

    replay_centering <- summarize_replay_centering(formula, family, rust_fit, block_index)
    fitted_summary <- fitted_rel_summary(rust_fit, inla_fit)

    cbind(
        data.frame(
            case = case_name,
            model = block$model,
            rust_elapsed_sec = rust_fit$elapsed_sec,
            inla_elapsed_sec = inla_fit$elapsed_sec,
            rust_theta_opt = paste(format(as.numeric(rust_fit$raw$theta_opt), digits = 8), collapse = ", "),
            inla_theta_mode = paste(format(as.numeric(inla_fit$fit$mode$theta), digits = 8), collapse = ", "),
            mean_rmse = sqrt(mean((rust_mean - inla_mean)^2)),
            rust_sd_avg = mean(rust_sd),
            inla_sd_avg = mean(inla_sd),
            rust_sd_max = max(rust_sd),
            inla_sd_max = max(inla_sd),
            rust_within_var_avg = mean(pmax(as.numeric(rust_fit$raw$latent_var_within_theta[block_index]), 0.0)),
            rust_between_var_avg = mean(pmax(as.numeric(rust_fit$raw$latent_var_between_theta[block_index]), 0.0)),
            inla_within_var_avg = mean(pmax(inla_cfg$mix_within_var[block_index], 0.0)),
            inla_between_var_avg = mean(pmax(inla_cfg$mix_between_var[block_index], 0.0)),
            stringsAsFactors = FALSE
        ),
        replay_centering,
        fitted_summary,
        stringsAsFactors = FALSE
    )
}

case_results <- rbind(
    run_case(
        case_name = "gamma_rw1_agegroup",
        formula = ClaimAmount ~ 1 + f(AgeGroup, model = "rw1"),
        family = "gamma"
    ),
    run_case(
        case_name = "gamma_iid_agegroup",
        formula = ClaimAmount ~ 1 + f(AgeGroup, model = "iid"),
        family = "gamma"
    )
)

cat("\nGAMMA UNCERTAINTY DIAGNOSIS\n")
print(case_results, row.names = FALSE)
