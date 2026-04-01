# ============================================================
# Validación R-INLA: Modelo AR1 + Gamma
# y[t] ~ Gamma(shape=phi, rate=phi/exp(eta[t]))
# x[t] ~ AR1(tau, rho)
# ============================================================

library(INLA)

set.seed(42)
n   <- 50
tau <- 2.0
rho <- 0.7
phi <- 3.0   # shape de la Gamma

# Simular AR1
x <- numeric(n)
x[1] <- rnorm(1, 0, 1/sqrt(tau))
for (t in 2:n) {
  x[t] <- rho * x[t-1] + rnorm(1, 0, sqrt((1 - rho^2) / tau))
}
mu <- exp(x)
y  <- rgamma(n, shape = phi, rate = phi / mu)

# ── Ajuste con R-INLA ────────────────────────────────────────
idx    <- 1:n
result <- inla(
  y ~ -1 + f(idx, model = "ar1"),
  data             = data.frame(y = y, idx = idx),
  family           = "gamma",
  control.compute  = list(dic = TRUE, waic = TRUE),
  control.predictor = list(compute = TRUE)
)

cat("=== HIPERPARAMETROS ===\n")
cat("theta_opt (log-escala interna):\n")
print(result$mode$theta)
# theta[1] = log(tau_ar1), theta[2] = logit-like(rho), theta[3] = log(phi_gamma)

cat("\nResumen de hiperparametros:\n")
print(result$summary.hyperpar)

cat("\n=== LOG VEROSIMILITUD MARGINAL ===\n")
cat("log_mlik:", result$mlik[1], "\n")

cat("\n=== PARAMETROS VERDADEROS vs ESTIMADOS ===\n")
hp <- result$summary.hyperpar
cat("tau — verdad:", tau, "| INLA mean:", hp$mean[1], "sd:", hp$sd[1], "\n")
cat("rho — verdad:", rho, "| INLA mean:", hp$mean[2], "sd:", hp$sd[2], "\n")
cat("phi — verdad:", phi, "| INLA mean:", hp$mean[3], "sd:", hp$sd[3], "\n")

cat("\n=== EFECTOS LATENTES (primeros 10) ===\n")
sx <- result$summary.random$idx
cat("mean:\n");  print(round(sx$mean[1:10], 4))
cat("sd:\n");    print(round(sx$sd[1:10],   4))

cat("\n=== METRICAS GLOBALES ===\n")
cat("max|mean_x - x_true|:", max(abs(sx$mean - x)), "\n")
cat("RMSE mean_x vs x_true:", sqrt(mean((sx$mean - x)^2)), "\n")
cat("Correlacion:          ", cor(sx$mean, x), "\n")
cat("max(sd_x):            ", max(sx$sd), "\n")
cat("mean(sd_x):           ", mean(sx$sd), "\n")
cat("DIC:                  ", result$dic$dic, "\n")

cat("\n=== THETA* EN ESCALA ORIGINAL ===\n")
theta_mode <- result$mode$theta
cat("tau_ar1 = exp(theta[1]) =", exp(theta_mode[1]), "(verdad:", tau, ")\n")
# rho via logit-inversa interna de INLA
cat("phi_gamma = exp(theta[3]) =", exp(theta_mode[3]), "(verdad:", phi, ")\n")
