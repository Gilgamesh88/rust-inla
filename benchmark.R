# Benchmark Harness for rustyINLA
# Compare Rusty-INLA with standard R-INLA for Actuarial Likelihoods

# Load all latest code directly
library(INLA)
library(CASdatasets)
library(rustyINLA)

# -----------------------------------------------------------------------------
# Data Preparation & Variable Transformation (cut)
# -----------------------------------------------------------------------------
cat("Loading datasets...\n")
data(freMTPL2freq)
data(freMTPL2sev)

cat("Discretizing continuous covariates into factors using 'cut'...\n")
# Transforming DrivAge (continuous) into a factor using cut()
freMTPL2freq$AgeGroup <- cut(
    freMTPL2freq$DrivAge, 
    breaks = c(17, 25, 40, 60, 80, 150), 
    labels = c("18-25", "26-40", "41-60", "61-80", "81+")
)

# Transforming VehPower (continuous/pseudo-continuous) into groups
freMTPL2freq$VehPowerGroup <- cut(
    freMTPL2freq$VehPower, 
    breaks = c(0, 5, 8, 12, 100), 
    labels = c("Low(1-5)", "Med(6-8)", "High(9-12)", "VeryHigh(13+)")
)

# Latent Indices preparation
df_freq <- freMTPL2freq
df_freq$AgeGroupIdx <- as.numeric(df_freq$AgeGroup)
N_LEVELS_AGE <- max(df_freq$AgeGroupIdx, na.rm=TRUE)

# Transforming VehBrand into numeric index for iid
df_freq$VehBrand <- as.factor(df_freq$VehBrand)
df_freq$VehBrandIdx <- as.numeric(df_freq$VehBrand)
N_LEVELS_BRAND <- max(df_freq$VehBrandIdx, na.rm=TRUE)

N_ROWS <- nrow(df_freq)

# Performance measurement helper
track_perf <- function(expr_sub) {
    gc(reset=TRUE)
    t0 <- proc.time()
    res <- eval(expr_sub)
    t1 <- proc.time()
    g <- gc()
    max_mem_mb <- sum(g[, 6]) # column 6 is (max used in Mb)
    return(list(res=res, time=(t1-t0)["elapsed"], mem=max_mem_mb))
}

# -----------------------------------------------------------------------------
# 1. Zero-Inflated Poisson (ZIP) Example & Benchmark
# -----------------------------------------------------------------------------
cat("\n==================================================\n")
cat("1. Zero-Inflated Poisson (ZIP) Benchmark\n")
cat("==================================================\n")
cat(sprintf("Using freMTPL2freq: %d rows.\n", N_ROWS))
cat("Response: ClaimNb  | Latent Effect: IID(VehBrand)\n")

# Setup expressions
rusty_zip_expr <- substitute({
    rust_inla_run(
        data = as.numeric(df_freq$ClaimNb),
        model_type = "iid", 
        likelihood_type = "zeroinflatedpoisson1",
        intercept = TRUE,
        n_latent_arg = as.integer(N_LEVELS_BRAND),
        x_idx_arg = as.integer(df_freq$VehBrandIdx - 1) # 0-indexed for rust
    )
})

rinla_zip_expr <- substitute({
    suppressMessages(inla(
        ClaimNb ~ 1 + f(VehBrandIdx, model="iid"),
        data = df_freq,
        family = "zeroinflatedpoisson1",
        control.compute = list(config=FALSE),
        num.threads = 1
    ))
})

cat("-- Running Rusty-INLA (ZIP) --\n")
rusty_perf_zip <- track_perf(rusty_zip_expr)

cat("-- Running R-INLA (ZIP) --\n")
rinla_perf_zip <- track_perf(rinla_zip_expr)

# -----------------------------------------------------------------------------
# 2. Tweedie Example & Benchmark
# -----------------------------------------------------------------------------
cat("\n==================================================\n")
cat("2. Tweedie Benchmark (Saddlepoint vs Exact)\n")
cat("==================================================\n")

cat("Preparing combined frequency/severity for pure premium modeling...\n")
agg_sev <- aggregate(ClaimAmount ~ IDpol, data = freMTPL2sev, sum)
df_tw <- merge(df_freq, agg_sev, by = "IDpol", all.x = TRUE)
df_tw$ClaimAmount[is.na(df_tw$ClaimAmount)] <- 0
df_tw$ClaimAmount <- pmin(df_tw$ClaimAmount, 50000)

cat("Response: Aggregated ClaimAmount  | Latent Effect: RW1(AgeGroup)\n")

# Setup Expressions
rusty_tw_expr <- substitute({
    rust_inla_run(
        data = as.numeric(df_tw$ClaimAmount),
        model_type = "rw1",
        likelihood_type = "tweedie",
        intercept = TRUE,
        n_latent_arg = as.integer(N_LEVELS_AGE),
        x_idx_arg = as.integer(df_tw$AgeGroupIdx - 1)
    )
})

rinla_tw_expr <- substitute({
    tryCatch({
        suppressMessages(inla(
            ClaimAmount ~ 1 + f(AgeGroupIdx, model="rw1"),
            data = df_tw,
            family = "tweedie",
            control.compute = list(config=FALSE),
            control.inla = list(strategy = "gaussian")
        ))
    }, error = function(e) {
        warning("INLA failed: ", conditionMessage(e))
        NULL
    })
})

cat("-- Running Rusty-INLA (Tweedie) --\n")
rusty_perf_tw <- track_perf(rusty_tw_expr)

cat("-- Running R-INLA (Tweedie) --\n")
rinla_perf_tw <- track_perf(rinla_tw_expr)

# -----------------------------------------------------------------------------
# Print Final Results Report
# -----------------------------------------------------------------------------
cat("\n==================================================\n")
cat("FINAL BENCHMARK RESULTS\n")
cat("==================================================\n")

cat("\n[1] Zero-Inflated Poisson (ZIP Type-1)\n")
cat(sprintf("Rusty-INLA : %.3f sec execution | %.2f MB Peak Mem\n", rusty_perf_zip$time, rusty_perf_zip$mem))
cat(sprintf("R-INLA     : %.3f sec execution | %.2f MB Peak Mem\n", rinla_perf_zip$time, rinla_perf_zip$mem))

cat("\n[2] Tweedie (Pure Premium)\n")
cat(sprintf("Rusty-INLA : %.3f sec execution | %.2f MB Peak Mem\n", rusty_perf_tw$time, rusty_perf_tw$mem))
if (!is.null(rinla_perf_tw$res)) {
    cat(sprintf("R-INLA     : %.3f sec execution | %.2f MB Peak Mem\n", rinla_perf_tw$time, rinla_perf_tw$mem))
} else {
    cat("R-INLA     : FAILED (Numerical Instability)\n")
}
