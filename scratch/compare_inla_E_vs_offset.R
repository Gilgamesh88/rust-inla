suppressPackageStartupMessages({
    library(INLA)
    library(CASdatasets)
})

options(digits = 15)

data(freMTPL2freq)

df <- freMTPL2freq
df$VehBrand <- as.factor(df$VehBrand)

ctrl_compute <- list(config = FALSE, dic = FALSE, waic = FALSE, cpo = FALSE)
ctrl_inla <- list(strategy = "auto", int.strategy = "auto")

fit_offset_time <- system.time({
    fit_offset <- inla(
        ClaimNb ~ 1 + offset(log(Exposure)) + f(VehBrand, model = "iid"),
        family = "poisson",
        data = df,
        control.compute = ctrl_compute,
        control.inla = ctrl_inla,
        verbose = FALSE
    )
})

fit_E_time <- system.time({
    fit_E <- inla(
        ClaimNb ~ 1 + f(VehBrand, model = "iid"),
        family = "poisson",
        data = df,
        E = Exposure,
        control.compute = ctrl_compute,
        control.inla = ctrl_inla,
        verbose = FALSE
    )
})

summary_table <- data.frame(
    path = c("offset(log(E))", "E="),
    elapsed_sec = c(unname(fit_offset_time["elapsed"]), unname(fit_E_time["elapsed"])),
    intercept = c(
        fit_offset$summary.fixed["(Intercept)", "mean"],
        fit_E$summary.fixed["(Intercept)", "mean"]
    ),
    precision_mean = c(
        fit_offset$summary.hyperpar[1, "mean"],
        fit_E$summary.hyperpar[1, "mean"]
    ),
    mlik = c(fit_offset$mlik[1, 1], fit_E$mlik[1, 1]),
    stringsAsFactors = FALSE
)

cat("INLA Poisson iid: offset(log(E)) versus E=\n")
print(summary_table, row.names = FALSE)

cat("\nDifferences (offset - E)\n")
cat("intercept =", summary_table$intercept[1] - summary_table$intercept[2], "\n")
cat(
    "precision_mean =",
    summary_table$precision_mean[1] - summary_table$precision_mean[2],
    "\n"
)
cat("mlik =", summary_table$mlik[1] - summary_table$mlik[2], "\n")

cat("\ncontrol.inla defaults seen by the fitted objects\n")
cat("strategy_offset =", fit_offset$.args$control.inla$strategy, "\n")
cat("strategy_E =", fit_E$.args$control.inla$strategy, "\n")
cat("int_strategy_offset =", fit_offset$.args$control.inla$int.strategy, "\n")
cat("int_strategy_E =", fit_E$.args$control.inla$int.strategy, "\n")

models <- inla.models()
cat("\nRelevant default hyper initials\n")
cat("iid_initial =", models$latent$iid$hyper$theta$initial, "\n")
cat("ar1_precision_initial =", models$latent$ar1$hyper$theta1$initial, "\n")
cat("ar1_rho_initial =", models$latent$ar1$hyper$theta2$initial, "\n")
cat("zip1_initial =", models$likelihood$zeroinflatedpoisson1$hyper$theta$initial, "\n")
cat("tweedie_p_initial =", models$likelihood$tweedie$hyper$theta1$initial, "\n")
cat("tweedie_phi_initial =", models$likelihood$tweedie$hyper$theta2$initial, "\n")
cat("poisson_hyper_count =", length(models$likelihood$poisson$hyper), "\n")
