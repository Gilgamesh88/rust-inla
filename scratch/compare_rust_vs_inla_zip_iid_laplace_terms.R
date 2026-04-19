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

models_info <- inla.models()
ctrl_compute <- list(config = TRUE, dic = FALSE, waic = FALSE, cpo = FALSE)
ctrl_predictor <- list(compute = TRUE)
ctrl_inla <- list(strategy = "auto", int.strategy = "auto")

collapse_fun <- function(fun) {
    paste(deparse(fun), collapse = " ")
}

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

expected_response_mean <- function(family, eta, theta_internal) {
    mu <- exp(eta)

    if (identical(family, "poisson")) {
        return(mu)
    }

    if (identical(family, "zeroinflatedpoisson1")) {
        zero_prob <- plogis(theta_internal[[length(theta_internal)]])
        return((1.0 - zero_prob) * mu)
    }

    stop(sprintf("Unsupported family for expected response mean: %s", family))
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

inla_hyper_audit <- function() {
    data.frame(
        label = c(
            "latent iid precision",
            "latent ar1 precision",
            "latent ar1 correlation",
            "likelihood zip1 zero probability"
        ),
        initial = c(
            models_info$latent$iid$hyper$theta$initial,
            models_info$latent$ar1$hyper$theta1$initial,
            models_info$latent$ar1$hyper$theta2$initial,
            models_info$likelihood$zeroinflatedpoisson1$hyper$theta$initial
        ),
        prior = c(
            as.character(models_info$latent$iid$hyper$theta$prior)[1],
            as.character(models_info$latent$ar1$hyper$theta1$prior)[1],
            as.character(models_info$latent$ar1$hyper$theta2$prior)[1],
            as.character(models_info$likelihood$zeroinflatedpoisson1$hyper$theta$prior)[1]
        ),
        param_1 = c(
            as.numeric(models_info$latent$iid$hyper$theta$param)[1],
            as.numeric(models_info$latent$ar1$hyper$theta1$param)[1],
            as.numeric(models_info$latent$ar1$hyper$theta2$param)[1],
            as.numeric(models_info$likelihood$zeroinflatedpoisson1$hyper$theta$param)[1]
        ),
        param_2 = c(
            as.numeric(models_info$latent$iid$hyper$theta$param)[2],
            as.numeric(models_info$latent$ar1$hyper$theta1$param)[2],
            as.numeric(models_info$latent$ar1$hyper$theta2$param)[2],
            as.numeric(models_info$likelihood$zeroinflatedpoisson1$hyper$theta$param)[2]
        ),
        to_theta = c(
            collapse_fun(models_info$latent$iid$hyper$theta$to.theta),
            collapse_fun(models_info$latent$ar1$hyper$theta1$to.theta),
            collapse_fun(models_info$latent$ar1$hyper$theta2$to.theta),
            collapse_fun(models_info$likelihood$zeroinflatedpoisson1$hyper$theta$to.theta)
        ),
        from_theta = c(
            collapse_fun(models_info$latent$iid$hyper$theta$from.theta),
            collapse_fun(models_info$latent$ar1$hyper$theta1$from.theta),
            collapse_fun(models_info$latent$ar1$hyper$theta2$from.theta),
            collapse_fun(models_info$likelihood$zeroinflatedpoisson1$hyper$theta$from.theta)
        ),
        stringsAsFactors = FALSE
    )
}

case_hyper_specs <- function(case_name) {
    if (identical(case_name, "zip_iid")) {
        return(list(
            list(
                label = "prec(VehBrand)",
                initial = models_info$latent$iid$hyper$theta$initial,
                from_theta = function(x) exp(x)
            ),
            list(
                label = "zero_prob",
                initial = models_info$likelihood$zeroinflatedpoisson1$hyper$theta$initial,
                from_theta = models_info$likelihood$zeroinflatedpoisson1$hyper$theta$from.theta
            )
        ))
    }

    if (identical(case_name, "poisson_iid")) {
        return(list(
            list(
                label = "prec(VehBrand)",
                initial = models_info$latent$iid$hyper$theta$initial,
                from_theta = function(x) exp(x)
            )
        ))
    }

    stop(sprintf("Unknown case_name: %s", case_name))
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
        return(c(
            pick_one("VehBrand")
        ))
    }

    stop(sprintf("Unknown case_name: %s", case_name))
}

summarize_starts <- function(case_name, rust_raw) {
    hyper_specs <- case_hyper_specs(case_name)
    inla_initial <- vapply(hyper_specs, `[[`, numeric(1), "initial")

    data.frame(
        case = case_name,
        theta_index = seq_along(hyper_specs),
        label = vapply(hyper_specs, `[[`, character(1), "label"),
        rust_start = as.numeric(rust_raw$theta_init_used),
        inla_initial = inla_initial,
        start_diff = as.numeric(rust_raw$theta_init_used) - inla_initial,
        stringsAsFactors = FALSE
    )
}

summarize_fixed <- function(case_name, rust_raw, inla_fit) {
    inla_fixed <- inla_fit$summary.fixed
    n_fixed <- length(rust_raw$fixed_means)

    data.frame(
        case = case_name,
        coefficient = rownames(inla_fixed)[seq_len(n_fixed)],
        rust_mean = as.numeric(rust_raw$fixed_means[seq_len(n_fixed)]),
        inla_mean = as.numeric(inla_fixed[seq_len(n_fixed), "mean"]),
        diff = as.numeric(rust_raw$fixed_means[seq_len(n_fixed)]) -
            as.numeric(inla_fixed[seq_len(n_fixed), "mean"]),
        stringsAsFactors = FALSE
    )
}

summarize_hypers <- function(case_name, rust_theta, inla_theta) {
    hyper_specs <- case_hyper_specs(case_name)
    rows <- vector("list", length(hyper_specs))

    for (idx in seq_along(hyper_specs)) {
        spec <- hyper_specs[[idx]]
        rows[[idx]] <- data.frame(
            case = case_name,
            theta_index = idx,
            label = spec$label,
            rust_internal = rust_theta[[idx]],
            inla_internal = inla_theta[[idx]],
            internal_diff = rust_theta[[idx]] - inla_theta[[idx]],
            rust_external = spec$from_theta(rust_theta[[idx]]),
            inla_external = spec$from_theta(inla_theta[[idx]]),
            stringsAsFactors = FALSE
        )
    }

    do.call(rbind, rows)
}

summarize_latent <- function(case_name, rust_spec, rust_raw, inla_fit) {
    block <- rust_spec$latent_blocks[[1]]
    block_index <- seq.int(block$start + 1L, length.out = block$n_levels)

    rust_mean <- as.numeric(rust_raw$prior_mean[block_index])
    rust_sd <- sqrt(pmax(as.numeric(rust_raw$marg_vars[block_index]), 0.0))
    inla_df <- inla_fit$summary.random[[block$covariate_name]]
    inla_mean <- as.numeric(inla_df$mean)
    inla_sd <- as.numeric(inla_df$sd)

    data.frame(
        case = case_name,
        block = block$covariate_name,
        model = block$model,
        n_levels = block$n_levels,
        rust_mean_avg = mean(rust_mean),
        inla_mean_avg = mean(inla_mean),
        mean_rmse = sqrt(mean((rust_mean - inla_mean)^2)),
        mean_max_abs = max(abs(rust_mean - inla_mean)),
        rust_sd_avg = mean(rust_sd),
        inla_sd_avg = mean(inla_sd),
        sd_rmse = sqrt(mean((rust_sd - inla_sd)^2)),
        sd_max_abs = max(abs(rust_sd - inla_sd)),
        stringsAsFactors = FALSE
    )
}

summarize_predictor <- function(case_name, family, y, rust_eta, inla_eta, rust_theta, inla_theta) {
    rust_mu <- expected_response_mean(family, rust_eta, rust_theta)
    inla_mu <- expected_response_mean(family, inla_eta, inla_theta)

    data.frame(
        case = case_name,
        rust_eta_avg = mean(rust_eta),
        inla_eta_avg = mean(inla_eta),
        eta_rmse = sqrt(mean((rust_eta - inla_eta)^2)),
        eta_max_abs = max(abs(rust_eta - inla_eta)),
        rust_mu_avg = mean(rust_mu),
        inla_mu_avg = mean(inla_mu),
        mu_rmse = sqrt(mean((rust_mu - inla_mu)^2)),
        mu_max_abs = max(abs(rust_mu - inla_mu)),
        plugin_loglik_rust = plugin_loglik(family, y, rust_eta, rust_theta),
        plugin_loglik_inla = plugin_loglik(family, y, inla_eta, inla_theta),
        stringsAsFactors = FALSE
    )
}

summarize_laplace <- function(case_name, raw, inla_fit, family, y, rust_eta, inla_eta, rust_theta, inla_theta) {
    terms <- raw$laplace_terms

    data.frame(
        case = case_name,
        rust_log_mlik = as.numeric(raw$log_mlik[[1]]),
        inla_log_mlik = as.numeric(inla_fit$mlik[1, 1]),
        rust_sum_loglik = as.numeric(terms$sum_loglik),
        rust_log_prior_model = as.numeric(terms$log_prior_model),
        rust_log_prior_likelihood = as.numeric(terms$log_prior_likelihood),
        rust_log_prior = as.numeric(terms$log_prior),
        rust_det_adjustment = 0.5 * (
            as.numeric(terms$final_log_det_q) - as.numeric(terms$final_log_det_aug)
        ),
        rust_quadratic_penalty = -0.5 * as.numeric(terms$final_q_form),
        rust_fixed_q_penalty = -0.5 * as.numeric(terms$fixed_q_form),
        rust_plugin_loglik = plugin_loglik(family, y, rust_eta, rust_theta),
        inla_plugin_loglik = plugin_loglik(family, y, inla_eta, inla_theta),
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

    rust_theta <- as.numeric(rust_fit$raw$theta_opt)
    inla_theta <- extract_inla_theta(case_name, inla_fit$fit)
    rust_eta <- build_rust_eta(rust_fit$spec, rust_fit$raw)
    inla_eta <- as.numeric(inla_fit$fit$summary.linear.predictor$mean[seq_len(nrow(df))])

    list(
        headline = data.frame(
            case = case_name,
            rust_elapsed_sec = rust_fit$elapsed_sec,
            inla_elapsed_sec = inla_fit$elapsed_sec,
            rust_mlik = as.numeric(rust_fit$raw$log_mlik[[1]]),
            inla_mlik = as.numeric(inla_fit$fit$mlik[1, 1]),
            intercept_rust = as.numeric(rust_fit$raw$fixed_means[1]),
            intercept_inla = as.numeric(inla_fit$fit$summary.fixed["(Intercept)", "mean"]),
            stringsAsFactors = FALSE
        ),
        starts = summarize_starts(case_name, rust_fit$raw),
        fixed = summarize_fixed(case_name, rust_fit$raw, inla_fit$fit),
        hyper = summarize_hypers(case_name, rust_theta, inla_theta),
        latent = summarize_latent(case_name, rust_fit$spec, rust_fit$raw, inla_fit$fit),
        predictor = summarize_predictor(case_name, family, df$ClaimNb, rust_eta, inla_eta, rust_theta, inla_theta),
        laplace = summarize_laplace(
            case_name,
            rust_fit$raw,
            inla_fit$fit,
            family,
            df$ClaimNb,
            rust_eta,
            inla_eta,
            rust_theta,
            inla_theta
        )
    )
}

cat("INLA HYPER DEFAULT AUDIT\n")
print(inla_hyper_audit(), row.names = FALSE)
cat("\nPoisson likelihood hyperparameter count:", length(models_info$likelihood$poisson$hyper), "\n")

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
    )
)

headline_df <- do.call(rbind, lapply(case_results, `[[`, "headline"))
starts_df <- do.call(rbind, lapply(case_results, `[[`, "starts"))
fixed_df <- do.call(rbind, lapply(case_results, `[[`, "fixed"))
hyper_df <- do.call(rbind, lapply(case_results, `[[`, "hyper"))
latent_df <- do.call(rbind, lapply(case_results, `[[`, "latent"))
predictor_df <- do.call(rbind, lapply(case_results, `[[`, "predictor"))
laplace_df <- do.call(rbind, lapply(case_results, `[[`, "laplace"))

cat("\nHEADLINE SUMMARY\n")
print(headline_df, row.names = FALSE)

cat("\nSTART VALUE COMPARISON\n")
print(starts_df, row.names = FALSE)

cat("\nFIXED EFFECT COMPARISON\n")
print(fixed_df, row.names = FALSE)

cat("\nHYPERPARAMETER COMPARISON\n")
print(hyper_df, row.names = FALSE)

cat("\nLATENT BLOCK COMPARISON\n")
print(latent_df, row.names = FALSE)

cat("\nPREDICTOR / PLUG-IN LIKELIHOOD COMPARISON\n")
print(predictor_df, row.names = FALSE)

cat("\nRUST LAPLACE DECOMPOSITION SUMMARY\n")
print(laplace_df, row.names = FALSE)
