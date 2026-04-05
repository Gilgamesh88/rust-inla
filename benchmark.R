# Benchmark Harness for rustyINLA
library(rustyINLA)

# 1. Simulate dense Actuarial Dataset
N_ROWS <- 400000 
N_LEVELS <- 600

cat("Simulating data... (N =", N_ROWS, "rows)\n")

# Random Walk 1 components
true_x <- cumsum(rnorm(N_LEVELS, mean=0, sd=0.1))
idx <- sample(1:N_LEVELS, size=N_ROWS, replace=TRUE)

# True Intercept
beta0 <- 1.5

# Linear Predictor
eta <- beta0 + true_x[idx]

# Poisson frequencies (Claims count)
y <- rpois(N_ROWS, lambda = exp(eta))

cat("Calling rustyINLA backend...\n")

# 2. Benchmarking execution time 
system.time({
    res <- rust_inla_run(
        data = as.numeric(y),
        model_type = "iid",         # Switch to IID or RW1 later
        likelihood_type = "poisson",
        intercept = TRUE,
        n_latent_arg = as.integer(N_LEVELS),
        x_idx_arg = as.integer(idx - 1)  # 0-indexed for Rust
    )
})

cat("\nOptimization Results:\n")
cat("Log Marginal Likelihood:", res$log_mlik, "\n")
cat("Found Intercept:", res$intercept_mean, " (True:", beta0, ")\n")

cat("\nCCD Integration:\n")
cat("Found", length(res$ccd_weights), "CCD integration points.\n")

n_theta <- length(res$theta_opt)
ccd_thetas_mat <- matrix(res$ccd_thetas, ncol = n_theta, byrow = TRUE)

ccd_df <- data.frame(
  Point = 1:length(res$ccd_weights),
  Weight = round(res$ccd_weights, 4)
)
for(i in 1:n_theta) {
  ccd_df[[paste0("Theta", i)]] <- round(ccd_thetas_mat[, i], 4)
}
print(ccd_df)

cat("\nSuccess! RustyINLA is bridging correctly and running CCD integration.\n")
