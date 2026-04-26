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

PRIOR_PREC_BETA <- 0.001
ctrl_compute <- list(config = TRUE, dic = FALSE, waic = FALSE, cpo = FALSE)
ctrl_predictor <- list(compute = TRUE)
ctrl_inla <- list(strategy = "auto", int.strategy = "auto")
models_info <- inla.models()

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

extract_inla_mode_theta <- function(case_name, inla_fit) {
    theta_values <- as.numeric(inla_fit[["mode"]][["theta"]])
    theta_names <- names(inla_fit[["mode"]][["theta"]])

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

extract_inla_mode_components <- function(case_name, inla_fit, latent_tag) {
    mode_x <- as.numeric(inla_fit[["mode"]][["x"]])
    contents <- inla_fit[["misc"]][["configs"]][["contents"]]

    get_block <- function(tag_name) {
        hit <- match(tag_name, contents$tag)
        if (is.na(hit)) {
            stop(sprintf("Could not find mode block '%s' in INLA contents", tag_name))
        }
        start <- contents$start[[hit]]
        len <- contents$length[[hit]]
        mode_x[seq.int(start, length.out = len)]
    }

    list(
        theta = extract_inla_mode_theta(case_name, inla_fit),
        eta = get_block("Predictor"),
        x = get_block(latent_tag),
        beta = get_block("(Intercept)")
    )
}

reconstruct_mode_inputs <- function(family, eta, y, theta_internal) {
    safe_eta <- pmin(pmax(eta, -50.0), 50.0)
    mu <- exp(safe_eta)

    if (identical(family, "poisson")) {
        grad <- ifelse(is.na(y), 0.0, y - mu)
        curv_raw <- ifelse(is.na(y), 0.0, mu)
        return(list(
            grad = grad,
            curvature_raw = curv_raw,
            curvature = pmax(curv_raw, 1e-6)
        ))
    }

    if (identical(family, "zeroinflatedpoisson1")) {
        p_zero <- plogis(theta_internal[[length(theta_internal)]])
        grad <- numeric(length(y))
        curv_raw <- numeric(length(y))

        missing_idx <- is.na(y)
        zero_idx <- (!missing_idx) & (y == 0)
        pos_idx <- (!missing_idx) & (y > 0)

        if (any(zero_idx)) {
            p0_pois <- exp(-mu[zero_idx])
            l0 <- p_zero + (1.0 - p_zero) * p0_pois
            w <- (1.0 - p_zero) * p0_pois / l0
            grad[zero_idx] <- -mu[zero_idx] * w
            curv_raw[zero_idx] <- w * mu[zero_idx] * (1.0 - mu[zero_idx] * (1.0 - w))
        }

        if (any(pos_idx)) {
            grad[pos_idx] <- y[pos_idx] - mu[pos_idx]
            curv_raw[pos_idx] <- mu[pos_idx]
        }

        grad[missing_idx] <- 0.0
        curv_raw[missing_idx] <- 0.0

        return(list(
            grad = grad,
            curvature_raw = curv_raw,
            curvature = pmax(curv_raw, 1e-6)
        ))
    }

    stop(sprintf("Unsupported family: %s", family))
}

rmse <- function(x, y) {
    sqrt(mean((x - y)^2))
}

group_score_residuals <- function(group_index, grad, x, theta_prec_internal) {
    tau <- exp(theta_prec_internal[[1]])
    if (length(group_index) != length(grad)) {
        stop(sprintf(
            "group_index length (%d) does not match grad length (%d)",
            length(group_index),
            length(grad)
        ))
    }
    grouped <- rowsum(
        matrix(grad, ncol = 1L),
        factor(group_index, levels = seq_along(x)),
        reorder = FALSE
    )
    score_by_group <- as.numeric(grouped[, 1])
    score_by_group - tau * x
}

summarize_starts <- function(case_name, rust_raw, inla_initial) {
    data.frame(
        case = case_name,
        theta_index = seq_along(inla_initial),
        rust_start = as.numeric(rust_raw$theta_init_used),
        inla_initial = as.numeric(inla_initial),
        start_diff = as.numeric(rust_raw$theta_init_used) - as.numeric(inla_initial),
        stringsAsFactors = FALSE
    )
}

summarize_inputs <- function(case_name, family, rust_raw, inla_mode) {
    rust_eta <- as.numeric(rust_raw$mode_eta)
    rust_theta <- as.numeric(rust_raw$theta_opt)
    rust_grad <- as.numeric(rust_raw$mode_grad)
    rust_curv_raw <- as.numeric(rust_raw$mode_curvature_raw)
    rust_curv <- as.numeric(rust_raw$mode_curvature)

    rust_reconstructed <- reconstruct_mode_inputs(family, rust_eta, df$ClaimNb, rust_theta)
    inla_reconstructed <- reconstruct_mode_inputs(
        family,
        as.numeric(inla_mode$eta),
        df$ClaimNb,
        as.numeric(inla_mode$theta)
    )
    mixed_theta_reconstructed <- reconstruct_mode_inputs(
        family,
        rust_eta,
        df$ClaimNb,
        as.numeric(inla_mode$theta)
    )

    data.frame(
        case = case_name,
        eta_rmse = rmse(rust_eta, as.numeric(inla_mode$eta)),
        rust_grad_selfcheck_rmse = rmse(rust_grad, rust_reconstructed$grad),
        rust_curv_raw_selfcheck_rmse = rmse(rust_curv_raw, rust_reconstructed$curvature_raw),
        rust_curv_selfcheck_rmse = rmse(rust_curv, rust_reconstructed$curvature),
        grad_rmse_vs_inla = rmse(rust_grad, inla_reconstructed$grad),
        grad_rmse_with_inla_theta_only = rmse(mixed_theta_reconstructed$grad, inla_reconstructed$grad),
        curv_raw_rmse_vs_inla = rmse(rust_curv_raw, inla_reconstructed$curvature_raw),
        curv_raw_rmse_with_inla_theta_only = rmse(
            mixed_theta_reconstructed$curvature_raw,
            inla_reconstructed$curvature_raw
        ),
        curv_rmse_vs_inla = rmse(rust_curv, inla_reconstructed$curvature),
        curv_rmse_with_inla_theta_only = rmse(
            mixed_theta_reconstructed$curvature,
            inla_reconstructed$curvature
        ),
        rust_clamped_count = sum(rust_curv_raw <= 1e-6),
        rust_clamped_share = mean(rust_curv_raw <= 1e-6),
        inla_nonpos_curv_count = sum(inla_reconstructed$curvature_raw <= 0.0),
        inla_nonpos_curv_share = mean(inla_reconstructed$curvature_raw <= 0.0),
        stringsAsFactors = FALSE
    )
}

summarize_score_residuals <- function(case_name, family, rust_raw, inla_mode, group_index) {
    rust_theta <- as.numeric(rust_raw$theta_opt)
    rust_grad <- as.numeric(rust_raw$mode_grad)
    inla_reconstructed <- reconstruct_mode_inputs(
        family,
        as.numeric(inla_mode$eta),
        df$ClaimNb,
        as.numeric(inla_mode$theta)
    )

    rust_fixed_resid <- sum(rust_grad) - PRIOR_PREC_BETA * as.numeric(rust_raw$mode_beta[[1]])
    inla_fixed_resid <- sum(inla_reconstructed$grad) - PRIOR_PREC_BETA * as.numeric(inla_mode$beta[[1]])

    rust_latent_resid <- group_score_residuals(
        group_index,
        rust_grad,
        as.numeric(rust_raw$mode_x),
        rust_theta[seq_len(1)]
    )
    inla_latent_resid <- group_score_residuals(
        group_index,
        inla_reconstructed$grad,
        as.numeric(inla_mode$x),
        as.numeric(inla_mode$theta)[seq_len(1)]
    )

    data.frame(
        case = case_name,
        rust_fixed_score_resid = rust_fixed_resid,
        inla_fixed_score_resid = inla_fixed_resid,
        rust_latent_resid_rmse = sqrt(mean(rust_latent_resid^2)),
        rust_latent_resid_max_abs = max(abs(rust_latent_resid)),
        inla_latent_resid_rmse = sqrt(mean(inla_latent_resid^2)),
        inla_latent_resid_max_abs = max(abs(inla_latent_resid)),
        stringsAsFactors = FALSE
    )
}

run_case <- function(case_name, formula, family, latent_tag) {
    cat("\n============================================================\n")
    cat("Running case:", case_name, "\n")
    cat("Formula:", deparse(formula, width.cutoff = 500L), "\n")
    flush.console()

    rust_fit <- fit_rust_case(formula, family)
    inla_fit <- fit_inla_case(formula, family)
    inla_mode <- extract_inla_mode_components(case_name, inla_fit$fit, latent_tag)
    inla_initial <- if (identical(case_name, "zip_iid")) {
        c(
            models_info$latent$iid$hyper$theta$initial,
            models_info$likelihood$zeroinflatedpoisson1$hyper$theta$initial
        )
    } else {
        c(models_info$latent$iid$hyper$theta$initial)
    }
    group_index <- as.integer(df[[latent_tag]])

    list(
        headline = data.frame(
            case = case_name,
            rust_elapsed_sec = rust_fit$elapsed_sec,
            inla_elapsed_sec = inla_fit$elapsed_sec,
            rust_mlik = as.numeric(rust_fit$raw$log_mlik[[1]]),
            inla_mlik = as.numeric(inla_fit$fit$mlik[1, 1]),
            rust_mode_solve_calls = as.numeric(rust_fit$raw$diagnostics$latent_mode_solve_calls),
            rust_mode_iterations_total = as.numeric(
                rust_fit$raw$diagnostics$latent_mode_iterations_total
            ),
            rust_mode_max_iter_hits = as.numeric(
                rust_fit$raw$diagnostics$latent_mode_max_iter_hits
            ),
            rust_mode_avg_iterations = as.numeric(
                rust_fit$raw$diagnostics$latent_mode_iterations_total
            ) / as.numeric(rust_fit$raw$diagnostics$latent_mode_solve_calls),
            stringsAsFactors = FALSE
        ),
        starts = summarize_starts(case_name, rust_fit$raw, inla_initial),
        inputs = summarize_inputs(case_name, family, rust_fit$raw, inla_mode),
        residuals = summarize_score_residuals(case_name, family, rust_fit$raw, inla_mode, group_index)
    )
}

case_results <- list(
    run_case(
        case_name = "zip_iid",
        formula = ClaimNb ~ 1 + offset(log(Exposure)) + f(VehBrand, model = "iid"),
        family = "zeroinflatedpoisson1",
        latent_tag = "VehBrand"
    ),
    run_case(
        case_name = "poisson_iid",
        formula = ClaimNb ~ 1 + offset(log(Exposure)) + f(VehBrand, model = "iid"),
        family = "poisson",
        latent_tag = "VehBrand"
    )
)

headline_df <- do.call(rbind, lapply(case_results, `[[`, "headline"))
starts_df <- do.call(rbind, lapply(case_results, `[[`, "starts"))
inputs_df <- do.call(rbind, lapply(case_results, `[[`, "inputs"))
residuals_df <- do.call(rbind, lapply(case_results, `[[`, "residuals"))

cat("\nHEADLINE SUMMARY\n")
print(headline_df, row.names = FALSE)

cat("\nSTART VALUE COMPARISON\n")
print(starts_df, row.names = FALSE)

cat("\nMODE INPUT COMPARISON\n")
print(inputs_df, row.names = FALSE)

cat("\nIID SCORE RESIDUAL COMPARISON\n")
print(residuals_df, row.names = FALSE)
