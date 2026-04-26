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

PRIOR_PREC_BETA <- 0.001

data(freMTPL2freq)

df <- freMTPL2freq
df$VehBrand <- as.factor(df$VehBrand)
df$Region <- as.factor(df$Region)

ctrl_compute <- list(config = TRUE, dic = FALSE, waic = FALSE, cpo = FALSE)
ctrl_predictor <- list(compute = TRUE)
ctrl_inla <- list(strategy = "auto", int.strategy = "auto")

formula_case <- ClaimNb ~ 1 + offset(log(Exposure)) +
    f(VehBrand, model = "iid") +
    f(Region, model = "iid")

fit_rust_case <- function() {
    spec <- rustyINLA:::build_backend_spec(formula_case, data = df, family = "poisson")
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

fit_inla_case <- function() {
    elapsed <- system.time(fit <- inla(
        formula_case,
        family = "poisson",
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

select_inla_mode_config <- function(inla_fit) {
    configs <- inla_fit$misc$configs$config
    target_theta <- as.numeric(inla_fit$mode$theta)
    theta_distance <- vapply(
        configs,
        function(cfg) sqrt(sum((as.numeric(cfg$theta) - target_theta)^2)),
        numeric(1)
    )
    configs[[which.min(theta_distance)]]
}

build_rust_joint_precision <- function(spec, raw) {
    n_latent <- spec$n_latent
    n_fixed <- spec$n_fixed
    if (n_fixed != 1L) {
        stop(sprintf("Expected one fixed effect, got %d", n_fixed))
    }

    theta <- as.numeric(raw$theta_opt)
    n_brand <- spec$latent_blocks[[1]]$n_levels
    n_region <- spec$latent_blocks[[2]]$n_levels
    tau_brand <- exp(theta[[1]])
    tau_region <- exp(theta[[2]])
    w <- as.numeric(raw$mode_curvature)

    q_latent <- matrix(0.0, nrow = n_latent, ncol = n_latent)
    diag(q_latent)[seq_len(n_brand)] <- tau_brand
    diag(q_latent)[n_brand + seq_len(n_region)] <- tau_region

    atwa <- matrix(0.0, nrow = n_latent, ncol = n_latent)
    c_vec <- numeric(n_latent)
    xtwx <- PRIOR_PREC_BETA

    brand_idx <- as.integer(df$VehBrand)
    region_idx <- n_brand + as.integer(df$Region)

    for (i in seq_len(nrow(df))) {
        wi <- w[[i]]
        b <- brand_idx[[i]]
        r <- region_idx[[i]]

        atwa[b, b] <- atwa[b, b] + wi
        atwa[r, r] <- atwa[r, r] + wi
        atwa[b, r] <- atwa[b, r] + wi
        atwa[r, b] <- atwa[r, b] + wi

        c_vec[b] <- c_vec[b] + wi
        c_vec[r] <- c_vec[r] + wi
        xtwx <- xtwx + wi
    }

    q_aug <- q_latent + atwa
    joint_q <- rbind(
        cbind(q_aug, matrix(c_vec, ncol = 1L)),
        cbind(matrix(c_vec, nrow = 1L), matrix(xtwx, nrow = 1L))
    )

    list(
        q_latent = q_latent,
        atwa = atwa,
        q_aug = q_aug,
        c_vec = c_vec,
        joint_q = joint_q
    )
}

extract_inla_mode_blocks <- function(inla_fit, mode_cfg, rust_spec) {
    n_latent <- rust_spec$n_latent
    cfg_mean <- as.numeric(mode_cfg$mean)
    cfg_q <- as.matrix(mode_cfg$Q)
    cfg_qinv <- as.matrix(mode_cfg$Qinv)

    list(
        mean_latent = cfg_mean[seq_len(n_latent)],
        mean_fixed = cfg_mean[n_latent + seq_len(rust_spec$n_fixed)],
        q_joint = cfg_q,
        qinv_joint = cfg_qinv
    )
}

compare_joint_matrices <- function(rust_joint, rust_raw, inla_fit, inla_mode) {
    n_latent <- nrow(rust_joint$joint_q) - 1L
    rust_joint_q <- rust_joint$joint_q
    rust_joint_qinv <- solve(rust_joint_q)

    rust_latent_conditional_var <- diag(rust_joint_qinv)[seq_len(n_latent)]
    rust_bridge_latent_var <- as.numeric(rust_raw$latent_var_theta_opt[seq_len(n_latent)])
    inla_latent_conditional_var <- diag(inla_mode$qinv_joint)[seq_len(n_latent)]

    data.frame(
        metric = c(
            "joint_Q_rmse",
            "joint_Q_max_abs",
            "joint_Qinv_rmse",
            "joint_Qinv_max_abs",
            "rust_dense_vs_bridge_latent_var_rmse",
            "rust_dense_vs_bridge_latent_var_max_abs",
            "rust_dense_vs_inla_latent_var_rmse",
            "rust_dense_vs_inla_latent_var_max_abs",
            "rust_bridge_vs_inla_latent_var_rmse",
            "rust_bridge_vs_inla_latent_var_max_abs"
        ),
        value = c(
            sqrt(mean((rust_joint_q - inla_mode$q_joint)^2)),
            max(abs(rust_joint_q - inla_mode$q_joint)),
            sqrt(mean((rust_joint_qinv - inla_mode$qinv_joint)^2)),
            max(abs(rust_joint_qinv - inla_mode$qinv_joint)),
            sqrt(mean((rust_latent_conditional_var - rust_bridge_latent_var)^2)),
            max(abs(rust_latent_conditional_var - rust_bridge_latent_var)),
            sqrt(mean((rust_latent_conditional_var - inla_latent_conditional_var)^2)),
            max(abs(rust_latent_conditional_var - inla_latent_conditional_var)),
            sqrt(mean((rust_bridge_latent_var - inla_latent_conditional_var)^2)),
            max(abs(rust_bridge_latent_var - inla_latent_conditional_var))
        ),
        stringsAsFactors = FALSE
    )
}

compare_block_diagonals <- function(rust_spec, rust_joint, rust_raw, inla_mode, inla_fit) {
    n_latent <- rust_spec$n_latent
    rust_dense_qinv_diag <- diag(solve(rust_joint$joint_q))[seq_len(n_latent)]
    rust_bridge_diag <- as.numeric(rust_raw$latent_var_theta_opt[seq_len(n_latent)])
    inla_diag <- diag(inla_mode$qinv_joint)[seq_len(n_latent)]

    rows <- vector("list", length(rust_spec$latent_blocks))
    for (idx in seq_along(rust_spec$latent_blocks)) {
        block <- rust_spec$latent_blocks[[idx]]
        block_index <- seq.int(block$start + 1L, length.out = block$n_levels)
        inla_summary_sd <- as.numeric(inla_fit$summary.random[[block$covariate_name]]$sd)

        rows[[idx]] <- data.frame(
            block = block$covariate_name,
            rust_dense_conditional_sd_avg = mean(sqrt(pmax(rust_dense_qinv_diag[block_index], 0.0))),
            rust_bridge_conditional_sd_avg = mean(sqrt(pmax(rust_bridge_diag[block_index], 0.0))),
            inla_mode_conditional_sd_avg = mean(sqrt(pmax(inla_diag[block_index], 0.0))),
            inla_final_sd_avg = mean(inla_summary_sd),
            rust_dense_vs_inla_conditional_rmse = sqrt(mean((
                sqrt(pmax(rust_dense_qinv_diag[block_index], 0.0)) -
                    sqrt(pmax(inla_diag[block_index], 0.0))
            )^2)),
            rust_bridge_vs_inla_conditional_rmse = sqrt(mean((
                sqrt(pmax(rust_bridge_diag[block_index], 0.0)) -
                    sqrt(pmax(inla_diag[block_index], 0.0))
            )^2)),
            stringsAsFactors = FALSE
        )
    }

    do.call(rbind, rows)
}

cat("Running Rust multi-iid fit\n")
rust_fit <- fit_rust_case()

cat("Running INLA multi-iid fit\n")
inla_fit <- fit_inla_case()

inla_mode_cfg <- select_inla_mode_config(inla_fit$fit)
inla_mode <- extract_inla_mode_blocks(inla_fit$fit, inla_mode_cfg, rust_fit$spec)
rust_joint <- build_rust_joint_precision(rust_fit$spec, rust_fit$raw)

mode_order_check <- data.frame(
    target = c("VehBrand_mean", "Region_mean", "Intercept_mean"),
    rmse = c(
        sqrt(mean((
            inla_mode$mean_latent[seq_len(rust_fit$spec$latent_blocks[[1]]$n_levels)] -
                as.numeric(inla_fit$fit$summary.random$VehBrand$mean)
        )^2)),
        sqrt(mean((
            inla_mode$mean_latent[
                rust_fit$spec$latent_blocks[[2]]$start + seq_len(rust_fit$spec$latent_blocks[[2]]$n_levels)
            ] - as.numeric(inla_fit$fit$summary.random$Region$mean)
        )^2)),
        abs(inla_mode$mean_fixed[[1]] - as.numeric(inla_fit$fit$summary.fixed["(Intercept)", "mean"]))
    ),
    stringsAsFactors = FALSE
)

matrix_summary <- compare_joint_matrices(rust_joint, rust_fit$raw, inla_fit$fit, inla_mode)
block_diag_summary <- compare_block_diagonals(
    rust_fit$spec,
    rust_joint,
    rust_fit$raw,
    inla_mode,
    inla_fit$fit
)

cat("\nMODE ORDER CHECK\n")
print(mode_order_check, row.names = FALSE)

cat("\nJOINT PRECISION / COVARIANCE COMPARISON\n")
print(matrix_summary, row.names = FALSE)

cat("\nBLOCK CONDITIONAL SD COMPARISON\n")
print(block_diag_summary, row.names = FALSE)
