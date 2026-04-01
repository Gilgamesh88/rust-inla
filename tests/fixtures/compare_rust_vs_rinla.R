# ============================================================
# Comparación directa: rust-inla vs R-INLA
# Usa el mismo fixture JSON que usa cargo test -- --ignored
# ============================================================

library(INLA)
library(jsonlite)

# ── Leer fixture ──────────────────────────────────────────────
fixture_path <- "tests/fixtures/iid_gaussian.json"
if (!file.exists(fixture_path)) {
  stop("Corre este script desde la raiz del repo rust-inla")
}
fx <- fromJSON(fixture_path)

cat("=== FIXTURE iid_gaussian ===\n")
cat("n =", length(fx$y), "\n")
cat("mean(y) =", mean(fx$y), "\n")

# ── R-INLA sobre los mismos datos ─────────────────────────────
y   <- fx$y
n   <- length(y)
idx <- 1:n

result <- inla(
  y ~ -1 + f(idx, model = "iid"),
  data            = data.frame(y = y, idx = idx),
  family          = "gaussian",
  control.compute = list(config = TRUE)
)

# ── Comparación ───────────────────────────────────────────────
sx <- result$summary.random$idx

cat("\n=== R-INLA: hiperparametros en el modo ===\n")
cat("theta_mode:", result$mode$theta, "\n")
cat("  => tau_x   = exp(theta[1]) =", exp(result$mode$theta[1]), "\n")
cat("  => tau_obs = exp(theta[2]) =", exp(result$mode$theta[2]), "\n")

cat("\n=== R-INLA: log_mlik ===\n")
cat("log_mlik:", result$mlik[1], "\n")

cat("\n=== Diferencia R-INLA vs fixture (mean_x del fixture = nuestro rust-inla) ===\n")
our_means <- fx$mean_x
rinla_means <- sx$mean
diff <- our_means - rinla_means
cat("max|rust - rinla|:", max(abs(diff)), "\n")
cat("rmse(rust vs rinla):", sqrt(mean(diff^2)), "\n")
cat("Primeros 5 — rust:", round(our_means[1:5], 4),
    "\n              INLA:", round(rinla_means[1:5], 4), "\n")

cat("\n=== R-INLA SD de efectos latentes (primeros 10) ===\n")
cat("sd_x:", round(sx$sd[1:10], 4), "\n")
cat("mean(sd_x):", mean(sx$sd), "\n")

# ── Exportar para comparación en Rust ─────────────────────────
cat("\n=== VALORES A PEGAR EN CONTEXT_SESION.md ===\n")
cat("R-INLA theta_opt (iid_gaussian):",
    round(result$mode$theta, 6), "\n")
cat("R-INLA log_mlik  (iid_gaussian):",
    round(result$mlik[1], 4), "\n")
cat("R-INLA mean(sd_x):", round(mean(sx$sd), 4), "\n")
cat("R-INLA max(sd_x): ", round(max(sx$sd), 4), "\n")
