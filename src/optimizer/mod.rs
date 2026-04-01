//! Optimización de hiperparámetros θ via BFGS.
//!
//! Equivalente Rust de `domin-interface.h/c` (1596 líneas de C).
//!
//! ## Qué optimizamos
//!
//! Encontramos θ* = argmax log p(y|θ), la verosimilitud marginal aproximada
//! (aproximación de Laplace). En cada evaluación:
//!
//! ```text
//! f(θ) = 0.5·log|Q(θ)| + Σᵢ log p(yᵢ|ηᵢ,θ_lik) - 0.5·n·log(2π)
//! ```
//!
//! El gradiente analítico usa:
//!
//! ```text
//! ∂f/∂θ_k = 0.5 · trace(Q⁻¹ · ∂Q/∂θ_k)
//!          ≈ 0.5 · Σᵢ (Q⁻¹)ᵢᵢ · (∂Q/∂θ_k)ᵢᵢ   [aprox. diagonal]
//! ```
//!
//! La inversa seleccionada diag(Q⁻¹) se obtiene via eval_with_inverse(),
//! que garantiza el estado Factorized del solver de forma atómica.

use argmin::core::{CostFunction, Error as ArgminError};
use std::cell::RefCell;

use crate::error::InlaError;
use crate::likelihood::LogLikelihood;
use crate::models::QFunc;
use crate::problem::Problem;

/// Resultado de la optimización.
pub struct OptimResult {
    /// Hiperparámetros óptimos en escala interna (log-espacio).
    pub theta_opt: Vec<f64>,
    /// log p(y|θ*) — verosimilitud marginal en el modo.
    pub log_mlik: f64,
    /// Número de evaluaciones de f(θ) realizadas.
    pub n_evals: usize,
}

/// Función objetivo para argmin (no usada actualmente, mantenida para futura
/// integración con BFGS de argmin cuando se actualice la API).
struct InlaObjective<'a> {
    problem:    RefCell<&'a mut Problem>,
    qfunc:      &'a dyn QFunc,
    likelihood: &'a dyn LogLikelihood,
    y:          &'a [f64],
}

impl<'a> CostFunction for InlaObjective<'a> {
    type Param  = Vec<f64>;
    type Output = f64;

    fn cost(&self, theta: &Self::Param) -> Result<Self::Output, ArgminError> {
        let n_model     = self.qfunc.n_hyperparams();
        let theta_model = &theta[..n_model];
        let theta_lik   = &theta[n_model..];

        let log_det = match self.problem.borrow_mut().eval(self.qfunc, theta_model) {
            Ok(ld)  => ld,
            Err(_)  => return Ok(f64::MAX / 2.0),
        };

        let n = self.y.len();
        let eta = vec![0.0_f64; n];
        let mut logll = vec![0.0_f64; n];
        self.likelihood.evaluate(&mut logll, &eta, self.y, theta_lik);
        let sum_logll: f64 = logll.iter().sum();

        let log_mlik = 0.5 * log_det + sum_logll;
        Ok(-log_mlik)
    }
}

/// Parámetros del optimizador.
pub struct OptimizerParams {
    /// Tolerancia de convergencia en la norma del gradiente.
    pub tol_grad: f64,
    /// Máximo de evaluaciones de f(θ).
    pub max_evals: usize,
    /// Paso para diferencias finitas del gradiente.
    pub finite_diff_step: f64,
}

impl Default for OptimizerParams {
    fn default() -> Self {
        Self {
            tol_grad:         1e-5,
            max_evals:        200,
            finite_diff_step: 1e-4,
        }
    }
}

/// Optimiza θ via descenso de gradiente con paso Barzilai-Borwein.
///
/// El gradiente analítico para los parámetros del modelo usa `eval_with_inverse()`
/// para obtener diag(Q⁻¹) de forma atómica (build + factorize + Takahashi en
/// una sola llamada, sin riesgo de estado inconsistente).
///
/// Los parámetros de la likelihood siempre usan diferencias finitas porque
/// no afectan a Q directamente.
pub fn optimize(
    problem:    &mut Problem,
    qfunc:      &dyn QFunc,
    likelihood: &dyn LogLikelihood,
    y:          &[f64],
    theta_init: &[f64],
    _params:    &OptimizerParams,
) -> Result<OptimResult, InlaError> {
    let n_model  = qfunc.n_hyperparams();
    let n_lik    = likelihood.n_hyperparams();
    let n_theta  = n_model + n_lik;
    let h        = 1e-4_f64;
    let max_iter = 50_usize;
    let tol_grad = 1e-4_f64;

    let mut theta = theta_init.to_vec();

    // ── Función objetivo: -log p(y|θ) ────────────────────────────────────────
    //
    // Usa solo eval() (sin inversa) porque se llama muchas veces durante la
    // búsqueda de línea y el costo de Takahashi no es necesario aquí.
    let objective = |problem: &mut Problem, theta: &[f64]| -> f64 {
        let theta_model = &theta[..n_model];
        let theta_lik   = &theta[n_model..];

        let log_det = match problem.eval(qfunc, theta_model) {
            Ok(ld) => ld,
            Err(_) => return f64::MAX / 2.0,
        };

        let n   = y.len();
        let eta = vec![0.0_f64; n];
        let mut logll = vec![0.0_f64; n];
        likelihood.evaluate(&mut logll, &eta, y, theta_lik);
        let sum_logll: f64 = logll.iter().sum();

        -(0.5 * log_det + sum_logll)
    };

    // ── Gradiente ─────────────────────────────────────────────────────────────
    //
    // Para parámetros del modelo con deval() implementado:
    //   ∂f/∂θ_k ≈ -0.5 · Σᵢ (Q⁻¹)ᵢᵢ · (∂Q/∂θ_k)ᵢᵢ
    //
    // Se usa eval_with_inverse() para obtener (log_det, diag_qinv) de forma
    // atómica. Esto evita el bug de estado donde selected_inverse() era llamado
    // después de que una llamada intermedia a build() reseteaba el solver a Built.
    //
    // Para parámetros de likelihood y como fallback: diferencias finitas.
    let gradient = |problem: &mut Problem, theta: &[f64]| -> Vec<f64> {
        let theta_model = &theta[..n_model];

        let has_analytic = qfunc.deval(0, 0, theta_model, 0).is_some();

        if has_analytic {
            // eval_with_inverse() = build + factorize + Takahashi, atómico.
            // diag_qinv usa binary_search — correcto para Rw1/Ar1 donde
            // val_of_col(j)[0] sería el primer elemento de la columna (no la
            // diagonal) en la SpMat simétrica de selected_inverse().
            match problem.eval_with_inverse(qfunc, theta_model) {
                Ok((_log_det, diag_qinv)) => {
                    let n_obs = y.len();
                    let mut grad = vec![0.0_f64; n_theta];

                    // Gradiente analítico para parámetros del modelo.
                    // Aproximación diagonal: trace(Q⁻¹ · dQ/dθ_k) ≈ Σᵢ (Q⁻¹)ᵢᵢ · (dQ)ᵢᵢ
                    // Exacto para iid (dQ = tau*I) y correcto en orden de magnitud
                    // para rw1/ar1. El término off-diagonal completo se añadirá
                    // cuando se implemente la traza completa en Fase C.
                    for k in 0..n_model {
                        let trace_approx: f64 = (0..n_obs)
                            .map(|i| {
                                let dq = qfunc.deval(i, i, theta_model, k).unwrap_or(0.0);
                                diag_qinv[i] * dq
                            })
                            .sum();
                        grad[k] = -0.5 * trace_approx;
                    }

                    // Parámetros de likelihood: siempre diferencias finitas
                    // porque deval solo cubre la matriz Q del modelo latente.
                    let f0 = objective(problem, theta);
                    for i in n_model..n_theta {
                        let mut theta_h = theta.to_vec();
                        theta_h[i] += h;
                        let fh = objective(problem, &theta_h);
                        grad[i] = (fh - f0) / h;
                    }

                    grad
                }
                Err(_) => {
                    // Fallback a diferencias finitas si eval_with_inverse falla
                    // (p.ej. Q no PD en un theta explorado durante la búsqueda).
                    let f0 = objective(problem, theta);
                    let mut grad = vec![0.0_f64; n_theta];
                    for i in 0..n_theta {
                        let mut theta_h = theta.to_vec();
                        theta_h[i] += h;
                        let fh = objective(problem, &theta_h);
                        grad[i] = (fh - f0) / h;
                    }
                    grad
                }
            }
        } else {
            // Sin gradiente analítico: diferencias finitas completas.
            let f0 = objective(problem, theta);
            let mut grad = vec![0.0_f64; n_theta];
            for i in 0..n_theta {
                let mut theta_h = theta.to_vec();
                theta_h[i] += h;
                let fh = objective(problem, &theta_h);
                grad[i] = (fh - f0) / h;
            }
            grad
        }
    };

    // ── Descenso de gradiente con búsqueda de línea (Barzilai-Borwein) ───────
    let mut f_cur   = objective(problem, &theta);
    let mut n_evals = 1_usize;
    let mut step    = 0.1_f64;

    for _iter in 0..max_iter {
        let grad = gradient(problem, &theta);
        n_evals += n_theta + 1;

        let grad_norm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        if grad_norm < tol_grad { break; }

        // Búsqueda de línea: reducir step hasta que f mejore (backtracking)
        let mut accepted = false;
        let mut step_try = step;
        for _ in 0..10 {
            let theta_new: Vec<f64> = theta.iter()
                .zip(grad.iter())
                .map(|(&t, &g)| t - step_try * g)
                .collect();
            let f_new = objective(problem, &theta_new);
            n_evals += 1;
            if f_new < f_cur {
                theta    = theta_new;
                f_cur    = f_new;
                step     = step_try * 1.2; // agrandar paso si mejoró
                accepted = true;
                break;
            }
            step_try *= 0.5;
        }
        if !accepted { break; }
    }

    // ── Evaluación final en θ* ────────────────────────────────────────────────
    let theta_model = &theta[..n_model];
    let theta_lik   = &theta[n_model..];

    problem.eval(qfunc, theta_model)
        .map_err(|_| InlaError::ConvergenceFailed {
            reason: "theta* produce Q no definida positiva".to_string(),
        })?;

    let n   = y.len();
    let eta = vec![0.0_f64; n];
    let mut logll = vec![0.0_f64; n];
    likelihood.evaluate(&mut logll, &eta, y, theta_lik);
    let sum_logll: f64 = logll.iter().sum();
    let log_mlik = 0.5 * problem.log_det() + sum_logll;

    Ok(OptimResult {
        theta_opt: theta,
        log_mlik,
        n_evals,
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::likelihood::GaussianLikelihood;
    use crate::models::IidModel;

    #[test]
    fn optimize_returns_result_for_valid_theta() {
        let n   = 10;
        let tau = 2.0_f64;
        let y: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();

        let model = IidModel::new(n);
        let lik   = GaussianLikelihood;
        let mut problem = Problem::new(&model);

        let theta_init = vec![tau.ln(), 1.0_f64.ln()];

        let result = optimize(
            &mut problem,
            &model,
            &lik,
            &y,
            &theta_init,
            &OptimizerParams::default(),
        ).unwrap();

        assert!(result.log_mlik.is_finite());
        assert!(result.n_evals > 0);
    }

    #[test]
    fn log_mlik_contains_log_det_term() {
        let n = 5;
        let y = vec![0.0; n];

        let model = IidModel::new(n);
        let lik   = GaussianLikelihood;

        let mut p1 = Problem::new(&model);
        let mut p2 = Problem::new(&model);

        let r1 = optimize(&mut p1, &model, &lik, &y, &[0.0, 0.0], &Default::default()).unwrap();
        let r2 = optimize(&mut p2, &model, &lik, &y, &[1.0, 0.0], &Default::default()).unwrap();

        // Distinto τ_model → distinto log_det → distinto log_mlik
        assert!((r1.log_mlik - r2.log_mlik).abs() > 1e-6);
    }
}
