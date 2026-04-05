# Benchmark Harness for rustyINLA
# Compare Rusty-INLA with standard R-INLA for Actuarial Likelihoods

library(INLA)
library(CASdatasets)
library(rustyINLA)

# -----------------------------------------------------------------------------
# Data Preparation
# -----------------------------------------------------------------------------
cat("Loading datasets and cutting covariates...\n")
data(freMTPL2freq)
data(freMTPL2sev)

freMTPL2freq$AgeGroup <- cut(
    freMTPL2freq$DrivAge, breaks = c(17, 25, 40, 60, 80, 150), labels = c("18-25", "26-40", "41-60", "61-80", "81+")
)
df_freq <- freMTPL2freq
df_freq$VehBrand <- as.factor(df_freq$VehBrand)

agg_sev <- aggregate(ClaimAmount ~ IDpol, data = freMTPL2sev, sum)
df_tw <- merge(df_freq, agg_sev, by = "IDpol", all.x = TRUE)
df_tw$ClaimAmount[is.na(df_tw$ClaimAmount)] <- 0
# Limit extreme severities for naive INLA fitting
df_tw$ClaimAmount <- pmin(df_tw$ClaimAmount, 50000)

df_sev <- df_tw[df_tw$ClaimNb > 0 & df_tw$ClaimAmount > 0, ]

# -----------------------------------------------------------------------------
# Evaluation Harness
# -----------------------------------------------------------------------------
track_perf <- function(expr_sub) {
    if (exists("gc")) gc(reset=TRUE)
    t0 <- proc.time()
    res <- tryCatch({ eval(expr_sub) }, error = function(e) list(error=e))
    t1 <- proc.time()
    g <- gc()
    max_mem_mb <- sum(g[, 6]) # column 6 is (max used in Mb)
    return(list(res=res, time=(t1-t0)["elapsed"], mem=max_mem_mb))
}

compare_models <- function(name, rusty_perf, rinla_perf) {
   cat(sprintf("\n==================================================\n"))
   cat(sprintf(" %s COMPARISON \n", toupper(name)))
   cat(sprintf("==================================================\n"))
   cat("\n[ PERFORMANCE ]\n")
   cat(sprintf("  Rusty-INLA : %6.2f sec | %6.2f MB Peak Memory\n", rusty_perf$time, rusty_perf$mem))
   
   has_rinla <- !is.null(rinla_perf$res) && inherits(rinla_perf$res, "inla")
   if (has_rinla) {
      cat(sprintf("  R-INLA     : %6.2f sec | %6.2f MB Peak Memory\n", rinla_perf$time, rinla_perf$mem))
      
      cat("\n[ FIXED EFFECTS (Intercept Mean) ]\n")
      r_int <- rusty_perf$res$summary.fixed["(Intercept)", "mean"]
      i_int <- rinla_perf$res$summary.fixed["(Intercept)", "mean"]
      cat(sprintf("  Rusty-INLA : %8.4f\n", r_int))
      cat(sprintf("  R-INLA     : %8.4f\n", i_int))
      cat(sprintf("  Difference : %8.4f\n", abs(r_int - i_int)))
      
      cat("\n[ RANDOM EFFECTS (First 5 Levels - Mean Estimates) ]\n")
      rnd_name <- names(rusty_perf$res$summary.random)[1]
      r_rnd <- rusty_perf$res$summary.random[[rnd_name]]$mean
      r_rnd_sd <- rusty_perf$res$summary.random[[rnd_name]]$sd
      
      inla_rnd_name <- names(rinla_perf$res$summary.random)[1]
      i_rnd <- rinla_perf$res$summary.random[[inla_rnd_name]]$mean
      i_rnd_sd <- rinla_perf$res$summary.random[[inla_rnd_name]]$sd
      
      max_len <- min(5, length(r_rnd), length(i_rnd))
      
      df_comp <- data.frame(
         Level = 1:max_len,
         Rusty_Mean = round(r_rnd[1:max_len], 5),
         INLA_Mean = round(i_rnd[1:max_len], 5),
         Diff_Mean = abs(round(r_rnd[1:max_len] - i_rnd[1:max_len], 5)),
         Rusty_SD = round(r_rnd_sd[1:max_len], 5),
         INLA_SD = round(i_rnd_sd[1:max_len], 5)
      )
      print(df_comp, row.names=FALSE)
   } else {
      cat("  R-INLA     : FAILED or Skipped\n")
      if(!is.null(rinla_perf$res$error)) {
          cat("  Error details: ", conditionMessage(rinla_perf$res$error), "\n")
      }
   }
}

cat("\nExecuting models (This may take a few minutes)...\n")

# 1. Poisson
cat("\nRunning Model 1/4: Poisson (ClaimNb ~ 1 + f(VehBrand))...\n")
rusty_pois <- track_perf(substitute(rusty_inla(ClaimNb ~ 1 + f(VehBrand, model="iid"), data = df_freq, family = "poisson")))
rinla_pois <- track_perf(substitute(suppressWarnings(suppressMessages(inla(ClaimNb ~ 1 + f(VehBrand, model="iid"), data = df_freq, family = "poisson", control.compute = list(config=FALSE), num.threads=1)))))
compare_models("Poisson (Frequency)", rusty_pois, rinla_pois)

# 2. Gamma
cat("\nRunning Model 2/4: Gamma (ClaimAmount ~ 1 + f(AgeGroup))...\n")
rusty_gam <- track_perf(substitute(rusty_inla(ClaimAmount ~ 1 + f(AgeGroup, model="rw1"), data = df_sev, family = "gamma")))
rinla_gam <- track_perf(substitute(suppressWarnings(suppressMessages(inla(ClaimAmount ~ 1 + f(AgeGroup, model="rw1"), data = df_sev, family = "gamma", control.compute = list(config=FALSE), num.threads=1)))))
compare_models("Gamma (Severity)", rusty_gam, rinla_gam)

# 3. ZIP
cat("\nRunning Model 3/4: ZIP (ClaimNb ~ 1 + f(VehBrand))...\n")
rusty_zip <- track_perf(substitute(rusty_inla(ClaimNb ~ 1 + f(VehBrand, model="iid"), data = df_freq, family = "zeroinflatedpoisson1")))
rinla_zip <- track_perf(substitute(suppressWarnings(suppressMessages(inla(ClaimNb ~ 1 + f(VehBrand, model="iid"), data = df_freq, family = "zeroinflatedpoisson1", control.compute = list(config=FALSE), num.threads=1)))))
compare_models("Zero-Inflated Poisson", rusty_zip, rinla_zip)

# 4. Tweedie
cat("\nRunning Model 4/4: Tweedie (ClaimAmount ~ 1 + f(AgeGroup))...\n")
rusty_tw <- track_perf(substitute(rusty_inla(ClaimAmount ~ 1 + f(AgeGroup, model="rw1"), data = df_tw, family = "tweedie")))
# Note: Tweaking R-INLA Tweedie strategy to 'gaussian' to give it a better chance of not failing numerically
rinla_tw <- track_perf(substitute(suppressWarnings(suppressMessages(inla(ClaimAmount ~ 1 + f(AgeGroup, model="rw1"), data = df_tw, family = "tweedie", control.compute = list(config=FALSE), control.inla = list(strategy = "gaussian"), num.threads=1)))))
compare_models("Tweedie (Pure Premium)", rusty_tw, rinla_tw)
