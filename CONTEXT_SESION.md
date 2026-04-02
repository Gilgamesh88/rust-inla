# rust-inla — Contexto de sesion

Pegar este archivo al inicio de cada sesion nueva con Claude.

Repositorio: github.com/Gilgamesh88/rust-inla
Tests: 95 pasando + 5 fixtures validados contra R-INLA
Tiempo fixtures: 25.6s
Último commit: fix warning x_hat L-BFGS

---

## Metodología acordada

Para cada cambio: identificar función/línea en inla.c o approx-inference.c,
explicar qué estamos agregando/quitando y por qué, validar contra fixture.

---

## Dependencias (Cargo.toml)

faer=0.24.0, rayon=1.10, argmin=0.10, argmin-math=0.4, statrs=0.18
sha2=0.10, rand=0.8, thiserror=2.0, ndarray=0.16
dev: serde+serde_json, approx=0.5, criterion=0.5

---

## Módulos completados

| Módulo | Tests | Estado |
|---|---|---|
| graph/ | 15 | completo |
| models/ iid+rw1+ar1 | 15 | completo + deval + is_proper() |
| solver/ Cholesky+logdet+Takahashi | 11+fixtures | completo, validado n=10k |
| problem/ eval+IRLS+find_mode_with_inverse+quadratic_form_x+intercept | 17 | completo |
| likelihood/ Gaussian+Poisson+Gamma | 13 | completo |
| optimizer/ Laplace+warm start+prior+xQx+L-BFGS | 6 | completo |
| integrator/ GK15 | 4 | completo |
| density/ Gaussian | 4 | completo |
| marginal/ | 6 | completo |
| inference/ InlaEngine+intercept+suma-a-cero | 9 | completo |

---

## Algoritmos implementados (equivalencia R-INLA)

| Algoritmo Rust | Equivalente R-INLA | Archivo/línea |
|---|---|---|
| find_mode_with_inverse() | IRLS + Takahashi sobre Q+W | inla.c:2800 |
| find_mode_with_intercept_and_inverse() | IRLS con β₀ por perfil | inla.c:2600 |
| Constraint suma-a-cero Σxᵢ=0 | GMRFLib_constr_add() | inla.c:2800 |
| quadratic_form_x() | DAXPY(n, x_mode, Q_x_mode) | inla.c:3400 |
| Laplace completa: 0.5(log\|Q\|-log\|Q+W\|)+Σlogp-0.5·x̂ᵀQx̂+log π(θ) | ldens_mode | inla.c:3450 |
| Prior logGamma(1,5e-5): log π(θ)=θ-5e-5·exp(θ) | inla_hyperpar_default_prior() | inla.c:3200 |
| Varianzas = diag((Q+W)⁻¹) via Takahashi | inla_compute_marginals() | approx-inference.c |
| L-BFGS m=5 + two-loop + Armijo c₁=1e-4 | bfgs3() two-loop | domin-interface.c:400 |
| eval_with_inverse() atómico + binary_search diagonal | — | rust-inla |

---

## Resultados fixtures (cargo test -- --ignored --nocapture)

| Fixture | max_err_mean | intercept_err | log_mlik_err | Estado |
|---|---|---|---|---|
| iid_gaussian n=5000 | 0.28 | 0.0000 | ~500 | ✓ pasa |
| ar1_gamma n=400 | 0.25 | — | 23 | ✓ pasa |
| rw1_poisson n=500 | 0.40 | 0.17 | 557 | ✓ pasa |
| cholesky_rw1 n=10000 | — | — | exacto | ✓ pasa |

Nota: log_mlik_err en rw1_poisson porque optimizer no usa intercept en su objetivo.

---

## Porcentaje de avance del port

| Componente | Paridad | Nota |
|---|---|---|
| Solver Cholesky + Takahashi (gmrflib) | **92%** | Validado n=10k |
| IRLS + find_mode (inla.c:2800) | **85%** | Con warm start e intercept |
| Laplace completa (inla.c:3200-3500) | **70%** | Falta integración sobre θ |
| Optimizador L-BFGS (domin-interface.c) | **60%** | Sin intercept en objetivo |
| Modelos GMRF (models.R) | **10%** | 3 de ~30 (iid, rw1, ar1) |
| Likelihoods (family.c) | **7%** | 3 de ~40 (Gaussian, Poisson, Gamma) |
| Marginales posteriores | **25%** | Gaussiana simple, sin Simplified Laplace |
| Intercept / efectos fijos | **60%** | Inferencia ✓, optimizer pendiente |
| Priors | **15%** | Solo logGamma default, sin PC priors |
| Integración sobre θ (cuadrícula/CCD) | **0%** | Solo Empirical Bayes |
| DIC / WAIC / CPO | **0%** | Pendiente |
| Constraints (suma-a-cero formal) | **40%** | Approx. via centrado iterativo |
| Bindings R (extendr) | **0%** | Fase D |
| Bindings Python (PyO3) | **0%** | Fase E |
| **TOTAL** | **~28%** | Funcional para casos simples |

---

## Próxima tarea PRIORITARIA



### 1. Integrar intercept en el objetivo del optimizer
El rw1_poisson tiene log_mlik=-1065 (ref=-507) porque laplace_eval usa
find_mode_with_inverse (sin intercept) en lugar de find_mode_with_intercept.
Cambio: pasar `intercept: bool` a optimize() y laplace_eval().
Equivale a domin-interface.c donde el predictor incluye todos los efectos fijos.

### 2. ar1_gamma theta_err moderado (~0.49)
El optimizer encuentra theta=[1.15, 0.54, 0.70] vs ref=[0.66, 1.04, 1.06].
La inicialización en theta_opt_ref funciona para el test, pero el optimizer
desde [0,0,0] no converge al óptimo correcto.

### 3. Likelihoods adicionales (Fase C)
Binomial, NegativeBinomial — las más solicitadas después de Poisson.


## FIX ARQUITECTURAL PENDIENTE — intercept via AugmentedQFunc

El bug activo del intercept requiere implementar el campo aumentado
z=[β₀,x] siguiendo hgmrfm.c de gmrflib.

### 1. models/mod.rs — nuevo AugmentedQFunc
    Q_aug[0,0] = prior_precision (0.001)
    Q_aug[i+1,j+1] = inner.eval(i,j,theta)
    Q_aug[0,i+1] = 0.0
    Grafo de n+1 nodos (un nodo extra para β₀)

### 2. problem/mod.rs — find_mode_augmented()
    IRLS sobre z=[β₀,x] de tamano n+1
    Devuelve (z[0]=β₀, z[1..]=x, log_det_aug, diag_aug_inv)
    NO necesita centrado ad-hoc — β₀ tiene su propio prior

### 3. inference/mod.rs + optimizer/mod.rs
    Eliminar find_mode_with_intercept_and_inverse()
    Usar find_mode_augmented() cuando intercept=true

Referencia gmrflib: hgmrfm.c lineas 159-178 (campo z) y linea 798 (prior_precision)
prior_precision default para intercept: 0.001
---

## Inicio de sesión

```powershell
cd C:\Users\Antonio\rust-inla
git pull
cargo test
cargo test -- --ignored --nocapture
```
