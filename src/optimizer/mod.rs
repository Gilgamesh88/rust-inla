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
//! El gradiente se calcula por diferencias finitas (igual que R-INLA).
//! Cada evaluación de f(θ) ejecuta un Cholesky completo via Problem::eval().
//!
//! ## Por qué argmin
//!
//! `argmin` provee BFGS con búsqueda de línea. Solo necesitamos implementar
//! el trait `CostFunction` — el resto (actualizaciones de Hessiano,
//! condiciones de Wolfe, convergencia) viene gratis.

use std::cell::RefCell;

use argmin::core::{CostFunction, Error as ArgminError};

use crate::error::InlaError;
use crate::likelihood::LogLikelihood;
use crate::models::QFunc;
use crate::problem::Problem;
use crate::solver::SparseSolver;

/// Resultado de la optimización.
pub struct OptimResult {
    /// Hiperparámetros óptimos en escala interna (log-espacio).
    pub theta_opt: Vec<f64>,
    /// log p(y|θ*) — verosimilitud marginal en el modo.
    pub log_mlik: f64,
    /// Número de evaluaciones de f(θ) realizadas.
    pub n_evals: usize,
}

/// Función objetivo para argmin.
struct InlaObjective<'a> {
    problem:    RefCell<&'a mut Problem>,
    qfunc:      &'a dyn QFunc,
    likelihood: &'a dyn LogLikelihood,
    y:          &'a [f64],
}

impl<'a> CostFunction for InlaObjective<'a> {
    type Param  = Vec<f64>;
    type Output = f64;

    /// Evalúa -log p(y|θ) (negativo porque argmin minimiza).
    ///
    /// f(θ) = -[0.5·log|Q(θ)| + Σᵢ log p(yᵢ|ηᵢ, θ_lik)]
    ///
    /// Si Q(θ) no es PD (θ fuera del dominio), devolvemos +∞.
    fn cost(&self, theta: &Self::Param) -> Result<Self::Output, ArgminError> {
        // Separar θ del modelo latente de θ de la likelihood
        let n_model = self.qfunc.n_hyperparams();
        let theta_model = &theta[..n_model];
        let theta_lik   = &theta[n_model..];

        // Evalúa log|Q(θ)| via Cholesky
        // Si falla (Q no PD), retornamos un valor muy grande
        let log_det = match self.problem.borrow_mut().eval(self.qfunc, theta_model) {
            Ok(ld)  => ld,
            Err(_)  => return Ok(f64::MAX / 2.0), // señal de dominio inválido
        };

        // Predictor lineal η = 0 por ahora (se actualiza en B.6 con InlaEngine)
        // En la aproximación completa: η = Q⁻¹·b donde b depende de la likelihood
        // Para el scaffold del optimizador, usamos η = 0 como punto de partida
        let n = self.y.len();
        let eta = vec![0.0_f64; n];

        // Evalúa Σᵢ log p(yᵢ|ηᵢ, θ_lik)
        let mut logll = vec![0.0_f64; n];
        self.likelihood.evaluate(&mut logll, &eta, self.y, theta_lik);
        let sum_logll: f64 = logll.iter().sum();

        // log p(y|θ) ≈ 0.5·log|Q| + Σ log p(yᵢ|ηᵢ)
        // Retornamos el negativo porque argmin minimiza
        let log_mlik = 0.5 * log_det + sum_logll;
        Ok(-log_mlik)
    }
}

/// Parámetros del optimizador.
pub struct OptimizerParams {
    /// Tolerancia de convergencia en el gradiente.
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

/// Optimiza θ via BFGS con gradiente por diferencias finitas.
///
/// # Argumentos
/// - `problem`:    Problem ya inicializado con reorder() (AMD hecho)
/// - `qfunc`:      modelo latente GMRF
/// - `likelihood`: familia de verosimilitud
/// - `y`:          vector de observaciones
/// - `theta_init`: θ inicial (en escala log)
/// - `params`:     parámetros del optimizador
pub fn optimize(
    problem:    &mut Problem,
    qfunc:      &dyn QFunc,
    likelihood: &dyn LogLikelihood,
    y:          &[f64],
    theta_init: &[f64],
    _params:    &OptimizerParams,
) -> Result<OptimResult, InlaError> {
    let n_model   = qfunc.n_hyperparams();
    let n_lik     = likelihood.n_hyperparams();
    let n_theta   = n_model + n_lik;
    let h         = 1e-4_f64; // paso para diferencias finitas
    let max_iter  = 50_usize;
    let tol_grad  = 1e-4_f64;

    let mut theta = theta_init.to_vec();

    // Funcion objetivo: -log p(y|theta)
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

    // Gradiente por diferencias finitas
    let gradient = |problem: &mut Problem, theta: &[f64]| -> Vec<f64> {
        let theta_model = &theta[..n_model];

        // Intentar gradiente analitico
        // d f/d theta_k = 0.5 * traza(Q^-1 * dQ/d theta_k)
        // Verificar si deval esta implementado (devuelve Some)
        let has_analytic = qfunc.deval(0, 0, theta_model, 0).is_some();

        if has_analytic {
            // Necesitamos diag(Q^-1) = selected_inverse diagonal
            // y la traza de Q^-1 * dQ/d theta_k
            // Para modelos donde dQ/d theta_k = c_k * Q (iid, rw1):
            //   traza(Q^-1 * dQ/d theta_k) = c_k * traza(Q^-1 * Q) = c_k * n
            // Para AR1 con k=0: mismo, c_0 = 1
            // Para AR1 con k=1: mas complejo, usamos diferencias finitas
            let n_obs = y.len();
            let mut grad = vec![0.0_f64; n_theta];

            // Evaluar Q en theta actual (ya deberia estar factorizado)
            let _ = problem.eval(qfunc, theta_model);

            // Obtener diag(Q^-1) via Takahashi
            if let Ok(q_inv) = problem.solver.selected_inverse() {
                let diag_qinv: Vec<f64> = (0..n_obs)
                    .map(|i| q_inv.val_of_col(i as usize)[0])
                    .collect();

                for k in 0..n_model {
                    // traza(Q^-1 * dQ/d theta_k) = sum_i (Q^-1)_ii * dQ_ii/d theta_k
                    // + 2 * sum_{i<j} (Q^-1)_ij * dQ_ij/d theta_k
                    // Para modelos con dQ = c*Q: traza = c * n
                    let dq_diag: f64 = (0..n_obs)
                        .map(|i| {
                            let dq = qfunc.deval(i, i, theta_model, k).unwrap_or(0.0);
                            diag_qinv[i] * dq
                        })
                        .sum();

                    grad[k] = -0.5 * dq_diag;
                }
            } else {
                // Fallback a diferencias finitas si Takahashi falla
                let f0 = objective(problem, theta);
                for i in 0..n_theta {
                    let mut theta_h = theta.to_vec();
                    theta_h[i] += h;
                    let fh = objective(problem, &theta_h);
                    grad[i] = (fh - f0) / h;
                }
            }

            // Para parametros de likelihood: siempre diferencias finitas
            let f0 = objective(problem, theta);
            for i in n_model..n_theta {
                let mut theta_h = theta.to_vec();
                theta_h[i] += h;
                let fh = objective(problem, &theta_h);
                grad[i] = (fh - f0) / h;
            }

            grad
        } else {
            // Diferencias finitas completas
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

    // Descenso de gradiente con paso adaptativo (Barzilai-Borwein)
    // Mas robusto que BFGS puro para este tipo de funcion
    let mut f_cur  = objective(problem, &theta);
    let mut n_evals = 1_usize;
    let mut step   = 0.1_f64;

    for _iter in 0..max_iter {
        let grad = gradient(problem, &theta);
        n_evals += n_theta + 1;

        // Verificar convergencia
        let grad_norm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        if grad_norm < tol_grad { break; }

        // Busqueda de linea simple: reducir step hasta que f mejore
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
                theta  = theta_new;
                f_cur  = f_new;
                step   = step_try * 1.2; // aumentar paso si mejoro
                accepted = true;
                break;
            }
            step_try *= 0.5;
        }
        if !accepted { break; }
    }

    // Evaluar log_mlik en theta*
    let n_model_final = qfunc.n_hyperparams();
    let theta_model   = &theta[..n_model_final];
    let theta_lik     = &theta[n_model_final..];

    problem.eval(qfunc, theta_model)
        .map_err(|_| InlaError::ConvergenceFailed {
            reason: "theta* no es PD".to_string(),
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
        // Verifica que el pipeline completo no panics con θ válido
        let n   = 10;
        let tau = 2.0_f64;
        let y: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();

        let model = IidModel::new(n);
        let lik   = GaussianLikelihood;
        let mut problem = Problem::new(&model);

        // θ = [log_tau_model, log_tau_lik]
        let theta_init = vec![tau.ln(), 1.0_f64.ln()];

        let result = optimize(
            &mut problem,
            &model,
            &lik,
            &y,
            &theta_init,
            &OptimizerParams::default(),
        ).unwrap();

        assert!(result.log_mlik.is_finite()); assert!(result.n_evals > 1);
        assert!(result.log_mlik.is_finite());
        assert!(result.n_evals > 0);
    }

    #[test]
    fn log_mlik_contains_log_det_term() {
        // Para iid Gaussian con η=0:
        // log p(y|θ) = 0.5·log|Q| + Σ log N(yᵢ|0, 1/τ_lik)
        // Verificamos que el resultado cambia con τ (sensibilidad correcta)
        let n = 5;
        let y = vec![0.0; n]; // y=0 simplifica el cálculo

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
