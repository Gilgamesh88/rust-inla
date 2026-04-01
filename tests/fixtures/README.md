# Scripts de validación R-INLA

Corre cada script desde la raíz del repo con:

```r
source("tests/fixtures/r_validation/validate_iid_gaussian.R")
source("tests/fixtures/r_validation/validate_ar1_gamma.R")
source("tests/fixtures/r_validation/validate_rw1_poisson.R")
source("tests/fixtures/r_validation/compare_rust_vs_rinla.R")  # requiere fixture JSON
```

## Qué mide cada script

| Script | Qué valida |
|---|---|
| `validate_iid_gaussian.R` | Caso base: Q diagonal, likelihood Gaussian exacta |
| `validate_ar1_gamma.R` | Prior no-diagonal (fill-in en L), likelihood no-conjugada |
| `validate_rw1_poisson.R` | Prior impropio, likelihood no-conjugada, intercepto |
| `compare_rust_vs_rinla.R` | Diferencia directa rust-inla vs R-INLA en fixture JSON |

## Métricas clave a comparar

Para cada modelo, R-INLA reporta:

- `result$mode$theta` — θ* en escala interna (log, arctanh, etc.)
- `result$mlik[1]` — log p(y|M) verosimilitud marginal
- `result$summary.random$idx$mean` — medias posteriores E[xᵢ|y]
- `result$summary.random$idx$sd` — desviaciones posteriores SD[xᵢ|y]
- `result$summary.hyperpar` — posterior de hiperparámetros (media, sd, cuantiles)
- `result$summary.linear.predictor` — predictor lineal η = x + efectos fijos

## Bugs identificados en rust-inla (diagnóstico 2025-04-01)

| Fixture | Bug | Síntoma |
|---|---|---|
| iid_gaussian (n=5000) | Optimizer no converge | theta_opt=[0,0] sin moverse |
| rw1_poisson | SD varianzas explotan | sd_err=23M (fallback O(n²) en Q singular) |
| ar1_gamma | Optimizer se aleja del óptimo | log_mlik=-364 vs ref -453 |

## Orden de corrección sugerido

1. **iid_gaussian**: diagnosticar por qué el gradiente es 0 en theta=[0,0]
2. **rw1_poisson SD**: la varianza de Rw1 debe usar (Q+W)⁻¹ no Q⁻¹
3. **ar1_gamma optimizer**: revisar línea de búsqueda con warm start
