# tests/fixtures/

Fixtures de referencia generados por R-INLA.

## Archivos

| Archivo | Descripción | Tolerancias |
|---|---|---|
| `iid_gaussian.json` | Modelo iid Gaussiano | log_mlik: 1e-4, mean_x: 1e-3 |
| `rw1_poisson.json` | RW1 Poisson (frecuencia) | log_mlik: 1e-4, mean_x: 1e-3 |
| `ar1_gamma.json` | AR1 Gamma (severidad) | log_mlik: 1e-4, mean_x: 1e-3 |
| `cholesky_rw1.json` | Cholesky directo n=10,000 | log_det: 1e-8, sol: 1e-8 |

## Cómo regenerar

```bash
Rscript tests/fixtures/generate_fixtures.R
```

Requiere R con INLA instalado:
```r
install.packages("INLA", repos = c(getOption("repos"),
  INLA = "https://inla.r-inla-download.org/R/stable"))
```

## Notas sobre los fixtures actuales

- `cholesky_rw1.json`: generado con n=10,000 (no n=10 del script original).
  Es un stress test de escala real. El log_det=2.399 es correcto para
  Q = DᵀD + 1e-6·I con D la matriz de diferencias de orden 10,000.

- `iid_gaussian.json`: generado con n≈500 (no n=50 del script original).
