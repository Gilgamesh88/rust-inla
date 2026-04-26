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
        return(c(
            pick_one("VehBrand")
        ))
    }

    if (identical(case_name, "poisson_multi_iid")) {
        return(c(
            pick_one("VehBrand"),
            pick_one("Region")
        ))
    }

    stop(sprintf("Unknown case_name: %s", case_name))
}

extract_inla_configs <- function(case_name, inla_fit) {
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
    order_idx <- order(log_post, decreasing = TRUE)

    list(
        theta = theta_mat[order_idx, , drop = FALSE],
        theta_names = names(inla_fit$mode$theta)[theta_order],
        log_post = log_post[order_idx],
        centered_log_post = log_post[order_idx] - max(log_post),
        original_config_id = order_idx
    )
}

fit_rust_at_theta <- function(formula, family, theta_target, latent_init = NULL, fixed_init = NULL) {
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

    eta <- build_rust_eta(spec, raw)
    plugin <- plugin_loglik(family, df$ClaimNb, eta, as.numeric(raw$theta_opt))
    terms <- raw$laplace_terms

    data.frame(
        rust_log_mlik = as.numeric(raw$log_mlik[[1]]),
        rust_plugin_loglik = plugin,
        rust_correction_gap = as.numeric(raw$log_mlik[[1]]) - plugin,
        rust_sum_loglik = as.numeric(terms$sum_loglik),
        rust_log_prior = as.numeric(terms$log_prior),
        rust_det_adjustment = 0.5 * (
            as.numeric(terms$final_log_det_q) -
                as.numeric(terms$final_log_det_aug)
        ),
        rust_quadratic_penalty = -0.5 * as.numeric(terms$final_q_form),
        theta_replay_max_abs = max(abs(as.numeric(raw$theta_opt) - as.numeric(theta_target))),
        n_evals = as.integer(raw$n_evals[[1]]),
        elapsed_sec = unname(elapsed["elapsed"]),
        stringsAsFactors = FALSE
    )
}

summarize_case_surface <- function(case_name, formula, family) {
    cat("\n============================================================\n")
    cat("Running case:", case_name, "\n")
    cat("Formula:", deparse(formula, width.cutoff = 500L), "\n")
    flush.console()

    inla_fit <- fit_inla_case(formula, family)
    inla_cfg <- extract_inla_configs(case_name, inla_fit$fit)
    base_spec <- rustyINLA:::build_backend_spec(formula, data = df, family = family)
    base_raw <- rust_inla_run(base_spec)
    if (is.character(base_raw)) {
        stop(base_raw)
    }
    theta_dim <- ncol(inla_cfg$theta)
    point_rows <- vector("list", nrow(inla_cfg$theta))

    for (i in seq_len(nrow(inla_cfg$theta))) {
        if (i == 1L || i %% 5L == 0L || i == nrow(inla_cfg$theta)) {
            cat(sprintf("  replaying config %d / %d\n", i, nrow(inla_cfg$theta)))
            flush.console()
        }

        theta_target <- inla_cfg$theta[i, , drop = TRUE]
        theta_df <- as.data.frame(as.list(stats::setNames(
            as.numeric(theta_target),
            paste0("theta_", seq_len(theta_dim))
        )))
        for (theta_name in paste0("theta_", seq_len(3L))) {
            if (is.null(theta_df[[theta_name]])) {
                theta_df[[theta_name]] <- NA_real_
            }
        }
        rust_eval <- tryCatch(
            fit_rust_at_theta(
                formula,
                family,
                theta_target,
                latent_init = base_raw$mode_x,
                fixed_init = base_raw$mode_beta
            ),
            error = function(err) {
                data.frame(
                    rust_log_mlik = NA_real_,
                    rust_plugin_loglik = NA_real_,
                    rust_correction_gap = NA_real_,
                    rust_sum_loglik = NA_real_,
                    rust_log_prior = NA_real_,
                    rust_det_adjustment = NA_real_,
                    rust_quadratic_penalty = NA_real_,
                    theta_replay_max_abs = NA_real_,
                    n_evals = NA_integer_,
                    elapsed_sec = NA_real_,
                    rust_error = conditionMessage(err),
                    stringsAsFactors = FALSE
                )
            }
        )

        if (is.null(rust_eval$rust_error)) {
            rust_eval$rust_error <- NA_character_
        }

        point_rows[[i]] <- cbind(
            data.frame(
                case = case_name,
                config_rank = i,
                original_config_id = inla_cfg$original_config_id[[i]],
                inla_log_posterior = inla_cfg$log_post[[i]],
                stringsAsFactors = FALSE
            ),
            rust_eval,
            theta_df
        )
    }

    point_df <- do.call(rbind, point_rows)
    point_df$success <- is.finite(point_df$rust_log_mlik)
    point_df$inla_centered <- point_df$inla_log_posterior - max(point_df$inla_log_posterior)
    point_df$rust_centered <- NA_real_
    point_df$rust_centered[point_df$success] <- point_df$rust_log_mlik[point_df$success] -
        max(point_df$rust_log_mlik[point_df$success])
    point_df$centered_diff <- point_df$rust_centered - point_df$inla_centered
    point_df$level_diff <- point_df$rust_log_mlik - point_df$inla_log_posterior

    near_mode <- point_df$config_rank <= min(5L, nrow(point_df))
    success_rows <- point_df$success
    safe_mean <- function(x) if (length(x) > 0L) mean(x) else NA_real_
    safe_sd <- function(x) if (length(x) > 1L) stats::sd(x) else NA_real_
    safe_max <- function(x) if (length(x) > 0L) max(x) else NA_real_
    safe_rmse <- function(x) if (length(x) > 0L) sqrt(mean(x^2)) else NA_real_
    safe_range_diff <- function(x) if (length(x) > 0L) diff(range(x)) else NA_real_

    summary_df <- data.frame(
        case = case_name,
        theta_names = paste(inla_cfg$theta_names, collapse = " | "),
        theta_dim = theta_dim,
        n_configs = nrow(point_df),
        n_success = sum(success_rows),
        n_fail = sum(!success_rows),
        inla_elapsed_sec = inla_fit$elapsed_sec,
        rust_elapsed_sec_avg = safe_mean(point_df$elapsed_sec[success_rows]),
        theta_replay_max_abs = safe_max(point_df$theta_replay_max_abs[success_rows]),
        centered_rmse = safe_rmse(point_df$centered_diff[success_rows]),
        centered_max_abs = safe_max(abs(point_df$centered_diff[success_rows])),
        level_diff_mean = safe_mean(point_df$level_diff[success_rows]),
        level_diff_sd = safe_sd(point_df$level_diff[success_rows]),
        level_diff_range = safe_range_diff(point_df$level_diff[success_rows]),
        near_mode_centered_rmse = safe_rmse(point_df$centered_diff[near_mode & success_rows]),
        near_mode_level_diff_range = safe_range_diff(point_df$level_diff[near_mode & success_rows]),
        stringsAsFactors = FALSE
    )

    worst_df <- point_df[success_rows, ]
    worst_df <- worst_df[order(-abs(worst_df$centered_diff), worst_df$config_rank), ]
    worst_df <- worst_df[seq_len(min(5L, nrow(worst_df))), c(
        "case",
        "config_rank",
        "original_config_id",
        "inla_log_posterior",
        "rust_log_mlik",
        "inla_centered",
        "rust_centered",
        "centered_diff",
        "level_diff",
        "rust_det_adjustment",
        "rust_quadratic_penalty",
        "rust_log_prior",
        "theta_1",
        "theta_2",
        "theta_3"
    )]

    list(summary = summary_df, worst = worst_df, all = point_df)
}

case_results <- list(
    summarize_case_surface(
        case_name = "zip_iid",
        formula = ClaimNb ~ 1 + offset(log(Exposure)) + f(VehBrand, model = "iid"),
        family = "zeroinflatedpoisson1"
    ),
    summarize_case_surface(
        case_name = "poisson_iid",
        formula = ClaimNb ~ 1 + offset(log(Exposure)) + f(VehBrand, model = "iid"),
        family = "poisson"
    ),
    summarize_case_surface(
        case_name = "poisson_multi_iid",
        formula = ClaimNb ~ 1 + offset(log(Exposure)) +
            f(VehBrand, model = "iid") +
            f(Region, model = "iid"),
        family = "poisson"
    )
)

summary_df <- do.call(rbind, lapply(case_results, `[[`, "summary"))
worst_df <- do.call(rbind, lapply(case_results, `[[`, "worst"))

cat("\nPOINTWISE CONFIG-THETA SURFACE SUMMARY\n")
print(summary_df, row.names = FALSE)

cat("\nWORST EXACT-THETA CENTERED DIFFERENCES\n")
print(worst_df, row.names = FALSE)
