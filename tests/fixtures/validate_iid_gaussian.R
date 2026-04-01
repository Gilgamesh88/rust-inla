# ============================================================
# Validación R-INLA: Modelo IID Gaussiano
# Equivale a: y[i] ~ N(x[i], 1/tau_lik),  x[i] ~ N(0, 1/tau_x)
# ============================================================

library(INLA)

set.seed(42)
n     <- 100
tau_x <- 4.0   # precision del efecto latente
tau_e <- 2.0   # precision del ruido de observacion

x_true <- rnorm(n, 0, 1/sqrt(tau_x))
y      <- x_true + rnorm(n, 0, 1/sqrt(tau_e))

# ── Ajuste con R-INLA ────────────────────────────────────────
idx    <- 1:n
result <- inla(
  y ~ -1 + f(idx, model = "iid"),
  data             = data.frame(y = y, idx = idx),
  family           = "gaussian",
  control.compute  = list(dic = TRUE, waic = TRUE, config = TRUE),
  control.predictor = list(compute = TRUE)
)

cat("=== HIPERPARAMETROS ===\n")
cat("theta_opt (log-escala interna):\n")
print(result$mode$theta)
cat("\nResumen de hiperparametros:\n")
print(result$summary.hyperpar)

cat("\n=== LOG VEROSIMILITUD MARGINAL ===\n")
cat("log_mlik:", result$mlik[1], "\n")

cat("\n=== EFECTOS LATENTES (primeros 10) ===\n")
sx <- result$summary.random$idx
cat("mean:\n");  print(round(sx$mean[1:10], 4))
cat("sd:\n");    print(round(sx$sd[1:10],   4))
cat("0.025q:\n");print(round(sx$`0.025quant`[1:10], 4))
cat("0.975q:\n");print(round(sx$`0.975quant`[1:10], 4))

cat("\n=== METRICAS GLOBALES ===\n")
cat("max|mean_x|:       ", max(abs(sx$mean)), "\n")
cat("mean(sd_x):        ", mean(sx$sd), "\n")
cat("sd(mean_x):        ", sd(sx$mean), "\n")
cat("DIC:               ", result$dic$dic, "\n")
cat("WAIC:              ", result$waic$waic, "\n")

cat("\n=== MARGINAL DE HIPERPARAMETROS ===\n")
cat("Precision x  — mean:", result$summary.hyperpar$mean[1],
    " sd:", result$summary.hyperpar$sd[1], "\n")
cat("Precision obs— mean:", result$summary.hyperpar$mean[2],
    " sd:", result$summary.hyperpar$sd[2], "\n")

cat("\n=== COMPARACION CON VERDAD ===\n")
cat("Correlacion mean_x vs x_true:", cor(sx$mean, x_true), "\n")
cat("RMSE mean_x vs x_true:       ", sqrt(mean((sx$mean - x_true)^2)), "\n")
