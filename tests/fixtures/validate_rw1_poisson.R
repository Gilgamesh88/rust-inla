# ============================================================
# Validación R-INLA: Modelo RW1 + Poisson
# y[t] ~ Poisson(exp(mu + x[t]))
# x[t] ~ RW1(tau)
# ============================================================

library(INLA)

set.seed(42)
n   <- 50
tau <- 10.0   # precision de los incrementos
mu  <- -1.0   # intercepto

# Simular RW1
x <- numeric(n)
x[1] <- 0
for (t in 2:n) {
  x[t] <- x[t-1] + rnorm(1, 0, 1/sqrt(tau))
}
lambda <- exp(mu + x)
y      <- rpois(n, lambda)

cat("Datos simulados:\n")
cat("y:", y[1:10], "...\n")
cat("lambda verdadera:", round(lambda[1:10], 2), "...\n")

# ── Ajuste con R-INLA ────────────────────────────────────────
idx    <- 1:n
result <- inla(
  y ~ 1 + f(idx, model = "rw1"),
  data             = data.frame(y = y, idx = idx),
  family           = "poisson",
  control.compute  = list(dic = TRUE, waic = TRUE),
  control.predictor = list(compute = TRUE)
)

cat("\n=== INTERCEPTO ===\n")
cat("Intercepto — verdad:", mu, "\n")
print(result$summary.fixed)

cat("\n=== HIPERPARAMETROS ===\n")
cat("theta_opt (log-escala interna):\n")
print(result$mode$theta)  # theta[1] = log(tau_rw1)

cat("\nResumen de hiperparametros:\n")
print(result$summary.hyperpar)
cat("tau — verdad:", tau,
    "| INLA mean:", result$summary.hyperpar$mean[1],
    "sd:", result$summary.hyperpar$sd[1], "\n")

cat("\n=== LOG VEROSIMILITUD MARGINAL ===\n")
cat("log_mlik:", result$mlik[1], "\n")

cat("\n=== EFECTOS LATENTES RW1 (primeros 10) ===\n")
sx <- result$summary.random$idx
cat("  NOTA: mean_x en R-INLA es el efecto RW1 sin intercepto\n")
cat("  El predictor lineal = intercepto + x\n")
cat("mean_x:\n");  print(round(sx$mean[1:10], 4))
cat("sd_x:\n");    print(round(sx$sd[1:10],   4))

cat("\n=== PREDICTOR LINEAL (intercepto + x) ===\n")
cat("Esto es lo comparable con nuestro engine (sin intercepto separado)\n")
lp <- result$summary.linear.predictor
cat("mean_lp:\n"); print(round(lp$mean[1:10], 4))
cat("sd_lp:\n");   print(round(lp$sd[1:10],   4))

cat("\n=== METRICAS GLOBALES ===\n")
cat("max|mean_x - x_true|:", max(abs(sx$mean - x)), "\n")
cat("RMSE (predictor lineal vs verdad):",
    sqrt(mean((lp$mean - (mu + x))^2)), "\n")
cat("max(sd_x):  ", max(sx$sd), "\n")
cat("mean(sd_x): ", mean(sx$sd), "\n")

cat("\n=== VALORES AJUSTADOS vs OBSERVADOS ===\n")
fitted <- exp(lp$mean)
cat("Correlacion lambda_fitted vs lambda_true:", cor(fitted, lambda), "\n")
cat("RMSE lambda:", sqrt(mean((fitted - lambda)^2)), "\n")
