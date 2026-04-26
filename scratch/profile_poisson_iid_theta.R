local_rustyinla_lib <- Sys.getenv("RUSTYINLA_LIB", "")
if (nzchar(local_rustyinla_lib)) {
    .libPaths(c(
        normalizePath(local_rustyinla_lib, winslash = "/", mustWork = TRUE),
        .libPaths()
    ))
}

suppressPackageStartupMessages({
    library(CASdatasets)
    library(rustyINLA)
})

options(digits = 15)

data(freMTPL2freq)

df <- freMTPL2freq
df$VehBrand <- as.factor(df$VehBrand)

spec_base <- rustyINLA:::build_backend_spec(
    ClaimNb ~ 1 + offset(log(Exposure)) + f(VehBrand, model = "iid"),
    data = df,
    family = "poisson"
)

extract_diagnostics <- function(raw) {
    diag <- raw$diagnostics
    data.frame(
        outer_iters = as.integer(diag$optimizer_outer_iterations[1]),
        laplace_total = as.integer(diag$laplace_eval_calls_total[1]),
        laplace_opt = as.integer(diag$laplace_eval_calls_optimizer[1]),
        laplace_ccd = as.integer(diag$laplace_eval_calls_ccd[1]),
        line_trials = as.integer(diag$line_search_trial_evals[1]),
        coord_evals = as.integer(diag$coordinate_probe_evals[1]),
        factorizations = as.integer(diag$factorization_count[1]),
        selected_inverse = as.integer(diag$selected_inverse_count[1]),
        optimizer_sec = as.numeric(diag$optimizer_time_sec[1]),
        ccd_sec = as.numeric(diag$ccd_time_sec[1]),
        latent_mode_sec = as.numeric(diag$latent_mode_solve_time_sec[1]),
        assembly_sec = as.numeric(diag$likelihood_assembly_time_sec[1]),
        factorization_sec = as.numeric(diag$sparse_factorization_time_sec[1]),
        selected_inverse_sec = as.numeric(diag$selected_inverse_time_sec[1]),
        stringsAsFactors = FALSE
    )
}

eval_at_theta <- function(theta_value) {
    spec <- spec_base
    spec$theta_init <- as.numeric(theta_value)
    spec$optimizer_max_evals <- 0L
    raw <- rust_inla_run(spec)
    cbind(data.frame(
        theta_init = theta_value,
        precision = exp(theta_value),
        log_mlik = as.numeric(raw$log_mlik[1]),
        stringsAsFactors = FALSE
    ), extract_diagnostics(raw))
}

optimize_from_theta <- function(theta_value) {
    spec <- spec_base
    spec$theta_init <- as.numeric(theta_value)
    raw <- rust_inla_run(spec)
    cbind(data.frame(
        theta_init = theta_value,
        theta_opt = as.numeric(raw$theta_opt[1]),
        precision_opt = exp(as.numeric(raw$theta_opt[1])),
        log_mlik = as.numeric(raw$log_mlik[1]),
        n_evals = as.integer(raw$n_evals[1]),
        intercept = as.numeric(raw$fixed_means[1]),
        stringsAsFactors = FALSE
    ), extract_diagnostics(raw))
}

theta_grid <- c(2, 3, 4, 5, 6)
profile_results <- do.call(rbind, lapply(theta_grid, eval_at_theta))
opt_results <- do.call(rbind, lapply(theta_grid, optimize_from_theta))

cat("Fixed-theta profile (optimizer_max_evals = 0)\n")
print(profile_results, row.names = FALSE)
cat("\nOptimization from multiple starts\n")
print(opt_results, row.names = FALSE)
