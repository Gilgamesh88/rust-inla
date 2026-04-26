local_rustyinla_lib <- Sys.getenv("RUSTYINLA_LIB", "")
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
df$AgeGroup <- cut(
    df$DrivAge,
    breaks = c(17, 25, 40, 60, 80, 150),
    labels = c("18-25", "26-40", "41-60", "61-80", "81+"),
    ordered_result = TRUE
)
df$AgeIndex <- as.integer(df$AgeGroup)

run_case <- function(formula, family) {
    rusty <- rusty_inla(formula, data = df, family = family)
    rinla <- suppressWarnings(suppressMessages(
        inla(
            formula,
            data = df,
            family = family,
            control.compute = list(config = FALSE),
            control.predictor = list(compute = TRUE),
            num.threads = 1
        )
    ))

    y <- df$ClaimNb
    mu_rusty <- pmax(as.numeric(rusty$summary.fitted.values$mean), 1e-12)
    mu_inla <- pmax(as.numeric(rinla$summary.fitted.values$mean), 1e-12)

    data.frame(
        intercept_rusty = rusty$summary.fixed["(Intercept)", "mean"],
        intercept_inla = rinla$summary.fixed["(Intercept)", "mean"],
        intercept_diff = rusty$summary.fixed["(Intercept)", "mean"] -
            rinla$summary.fixed["(Intercept)", "mean"],
        theta1_rusty = if (nrow(rusty$summary.hyperpar) >= 1) {
            rusty$summary.hyperpar[1, "mean"]
        } else {
            NA_real_
        },
        theta1_rusty_prec = if (nrow(rusty$summary.hyperpar) >= 1) {
            exp(rusty$summary.hyperpar[1, "mean"])
        } else {
            NA_real_
        },
        hyper1_inla = if (nrow(rinla$summary.hyperpar) >= 1) {
            rinla$summary.hyperpar[1, "mean"]
        } else {
            NA_real_
        },
        random_mean_rusty = if (length(rusty$summary.random) > 0) {
            mean(rusty$summary.random[[1]]$mean)
        } else {
            NA_real_
        },
        random_mean_inla = if (length(rinla$summary.random) > 0) {
            mean(rinla$summary.random[[1]]$mean)
        } else {
            NA_real_
        },
        plugin_loglik_rusty = sum(dpois(y, lambda = mu_rusty, log = TRUE)),
        plugin_loglik_inla = sum(dpois(y, lambda = mu_inla, log = TRUE)),
        plugin_loglik_diff = sum(dpois(y, lambda = mu_rusty, log = TRUE)) -
            sum(dpois(y, lambda = mu_inla, log = TRUE)),
        mlik_rusty = as.numeric(rusty$mlik[1]),
        mlik_inla = as.numeric(rinla$mlik[1]),
        mlik_diff = as.numeric(rusty$mlik[1]) - as.numeric(rinla$mlik[1])
    )
}

iid_stats <- run_case(
    ClaimNb ~ 1 + offset(log(Exposure)) + f(VehBrand, model = "iid"),
    "poisson"
)
ar1_stats <- run_case(
    ClaimNb ~ 1 + offset(log(Exposure)) + f(AgeIndex, model = "ar1"),
    "poisson"
)

cat("IID case\n")
print(iid_stats)
cat("\nAR1 case\n")
print(ar1_stats)
