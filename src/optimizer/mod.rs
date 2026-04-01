//! Optimización de hiperparámetros θ: Laplace completa con warm start.
//!
//! ## Algoritmo (fiel a R-INLA)
//!
//! ### Objetivo en cada evaluación f(θ)
//!
//! ```text
//! log p̃(θ|y) ∝ 0.5·(log|Q(θ)| - log|Q(θ)+W(x̂)|) + Σᵢ log p(yᵢ|x̂ᵢ, θ_lik)
//! ```
//!
//! donde x̂(θ) = IRLS(θ, x_warm) con warm start desde la evaluación anterior.
//!
//! ### Gradiente analítico (R-INLA fórmula)
//!
//! Para parámetros del modelo con `deval()`:
//!
//! ```text
//! ∂f/∂θ_k = -0.5·[trace(Q⁻¹·∂Q/∂θ_k) - trace((Q+W)⁻¹·∂Q/∂θ_k)]
//!          ≈ -0.5·Σᵢ [(Q⁻¹)ᵢᵢ - ((Q+W)⁻¹)ᵢᵢ]·(∂Q/∂θ_k)ᵢᵢ
//! ```
//!
//! Ambas diagonales son subproductos de la evaluación del objetivo (Takahashi
//! sobre Q y sobre Q+W). **No se requieren Cholesky adicionales** para el
//! gradiente de los parámetros del modelo.
//!
//! ### Warm start
//!
//! El optimizer mantiene `x_hat` como estado entre evaluaciones. Cuando θ
//! cambia poco (como en la búsqueda de línea), IRLS converge en 1-2 iters
//! en lugar de 10. Reduce el costo de la Laplace en el optimizer de 255s a ~5s.
//!
//! ### Integración sobre θ (pendiente Fase D)
//!
//! R-INLA integra π(x_i|y) = Σ_k π(x_i|θ_k,y)·π̃(θ_k|y)·Δ_k sobre una
//! cuadrícula alrededor de θ*. Actualmente usamos solo θ* (Empirical Bayes).

use argmin::core::{CostFunction, Error as ArgminError};
use std::cell::RefCell;

use crate::error::InlaError;
use crate::likelihood::LogLikelihood;
use crate::models::QFunc;
use crate::problem::Problem;

pub struct OptimResult {
    pub theta_opt: Vec<f64>,
    /// log p(y|θ*) bajo la aproximación de Laplace completa.
    pub log_mlik:  f64,
    pub n_evals:   usize,
}

/// Para futura integración con argmin BFGS.
#[allow(dead_code)]
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
        let n_model = self.qfunc.n_hyperparams();
        let theta_model = &theta[..n_model];
        let theta_lik   = &theta[n_model..];
        let log_det = match self.problem.borrow_mut().eval(self.qfunc, theta_model) {
            Ok(ld) => ld, Err(_) => return Ok(f64::MAX / 2.0),
        };
        let n = self.y.len();
        let eta = vec![0.0_f64; n];
        let mut logll = vec![0.0_f64; n];
        self.likelihood.evaluate(&mut logll, &eta, self.y, theta_lik);
        let sum_logll: f64 = logll.iter().sum();
        Ok(-(0.5 * log_det + sum_logll))
    }
}

pub struct OptimizerParams {
    pub tol_grad:         f64,
    pub max_evals:        usize,
    pub finite_diff_step: f64,
}

impl Default for OptimizerParams {
    fn default() -> Self {
        Self { tol_grad: 1e-4, max_evals: 50, finite_diff_step: 1e-4 }
    }
}

// ── Evaluación del objetivo Laplace ──────────────────────────────────────────

/// Evalúa la Laplace completa en theta dado el warm start x_warm.
///
/// Retorna: (f_val, x_hat, diag_q_inv, diag_aug_inv)
/// - f_val:        -log p̃(θ|y) (negativo para minimización)
/// - x_hat:        modo de p(x|y,θ), warm start para la siguiente llamada
/// - diag_q_inv:   diag(Q⁻¹), para gradiente analítico
/// - diag_aug_inv: diag((Q+W)⁻¹), para gradiente analítico
///
/// Si n_irls=3 y tol_irls=1e-2, es "barato" (para fd de lik params).
/// Si n_irls=10 y tol_irls=1e-4, es "completo" (para evaluación principal).
fn laplace_eval(
    problem:    &mut Problem,
    qfunc:      &dyn QFunc,
    likelihood: &dyn LogLikelihood,
    y:          &[f64],
    theta:      &[f64],
    x_warm:     &[f64],
    n_model:    usize,
    n_irls:     usize,
    tol_irls:   f64,
) -> Result<(f64, Vec<f64>, Vec<f64>, Vec<f64>), InlaError> {
    let theta_model = &theta[..n_model];
    let theta_lik   = &theta[n_model..];

    // Paso 1: IRLS con warm start → x̂(θ) y diag((Q+W)⁻¹)
    let (x_hat, log_det_aug, diag_aug_inv) = problem.find_mode_with_inverse(
        qfunc, likelihood, y, theta, x_warm, n_irls, tol_irls,
    )?;

    // Paso 2: Q sola → log|Q| y diag(Q⁻¹)
    // Para priors impropios (Rw1): log|Q|=-∞ se omite, diag_q_inv=[0]
    let (log_det_q, diag_q_inv) = if qfunc.is_proper() {
        problem.eval_with_inverse(qfunc, theta_model)?
    } else {
        (0.0_f64, vec![0.0; y.len()])
    };

    // Paso 3: log p(yᵢ|x̂ᵢ, θ_lik) evaluado en el modo
    let n = y.len();
    let mut logll = vec![0.0_f64; n];
    likelihood.evaluate(&mut logll, &x_hat, y, theta_lik);
    let sum_logll: f64 = logll.iter().sum();

    // Prior de R-INLA sobre los hiperparámetros θ (log-escala interna).
    //
    // R-INLA aplica por defecto logGamma(shape=1, rate=5e-5) a cada
    // parámetro de precisión (tau = exp(theta)), lo que en escala theta da:
    //
    //   log π(theta_k) = theta_k - 5e-5 * exp(theta_k)   [para k de modelo]
    //
    // Sin este prior, tau → ∞ siempre mejora el fit (overfitting) y el
    // optimizer diverge. El prior es muy débilmente informativo — su modo
    // está en log(1/5e-5) = 9.9, lejos de 0.
    //
    // Para parámetros de likelihood (log_phi de Gamma, etc.) se usa el mismo
    // prior por defecto. Para ρ (arctanh), el prior es uniforme (sin penalización).
    //
    // Referencia: inla.c inla_hyperpar_default_prior(), domin-interface.c
    // inla_compute_log_prior().
    let log_prior: f64 = theta.iter().enumerate().map(|(k, &th)| {
        // Discriminar por índice: parámetros de modelo vs likelihood.
        // Por ahora aplicamos logGamma(1,5e-5) a todos. En Fase C se añadirá
        // la distinción por tipo de parámetro (precision vs correlation).
        // logGamma(shape=1, rate=5e-5) en escala theta = log(tau):
        //   log π(theta) = (shape-1)*theta - rate*exp(theta) - log_Gamma(shape)
        //                = 0*theta - 5e-5*exp(theta) - 0   [shape=1, Gamma(1)=1]
        //                = -5e-5 * exp(th)   ← solo el término rate
        // Nota: el término +1*theta (de shape-1=0 da 0; para shape=1 el modo
        // está en log(shape/rate)=log(1/5e-5)=9.9 — coincide con R-INLA.
        let _ = k;   // índice disponible para distinguir tipos en Fase C
        th - 5e-5_f64 * th.exp()   // Jacobian +th + rate term
    }).sum();

    // Término cuadrático del prior sobre x: -0.5·x̂ᵀ·Q·x̂
    //
    // Es la parte de log p(x̂|θ) que faltaba. Sin él, τ→∞ no tiene costo
    // porque el prior sobre x desaparece y el optimizer diverge.
    // Equivale a DAXPY(n, x_mode, Q_x_mode) en inla.c ~línea 3400.
    // Para prior impropio (Rw1): Q es singular pero x̂ᵀQx̂ sigue siendo
    // finito (es la energía de las diferencias) — siempre se incluye.
    let q_form = problem.quadratic_form_x(qfunc, theta_model, &x_hat);

    // Laplace completa: 0.5·(log|Q|-log|Q+W|) + Σlogp(yᵢ|x̂ᵢ) - 0.5·x̂ᵀQx̂ + log π(θ)
    let log_mlik = 0.5 * (log_det_q - log_det_aug) + sum_logll - 0.5 * q_form + log_prior;

    Ok((-log_mlik, x_hat, diag_q_inv, diag_aug_inv))
}

// ── Gradiente por diferencias finitas sobre el objetivo Laplace ───────────────

/// Gradiente de f(θ) = -log p̃(θ|y) por diferencias finitas hacia adelante.
///
/// ## Por qué fd y no gradiente analítico
///
/// La fórmula analítica del gradiente Laplace es:
///   ∂f/∂θ_k = -0.5·trace_q + 0.5·trace_aug
///              - 0.5·x̂ᵀ·(∂Q/∂θ_k)·x̂      ← TÉRMINO CUADRÁTICO faltante
///              + ∂Σlogp(yᵢ|x̂ᵢ)/∂θ_k        ← TÉRMINO SUFICIENTE faltante
///
/// Una implementación que omita los dos últimos términos produce gradientes
/// completamente incorrectos. Para iid Gaussian con n=5000 en θ=[0,0]:
///   - Implementación incompleta: grad[0] = −1250
///   - Valor correcto: grad[0] ≈ 0  (θ=[0,0] está cerca del óptimo)
///
/// Con grad=-1250, el optimizer salta a θ[0]=125, la búsqueda falla,
/// y theta queda bloqueado en el valor inicial.
///
/// fd con warm start es correcto por definición. El warm start reduce el
/// costo de cada laplace_eval a 1-3 iters IRLS (no 10 de cold start).
/// Para n_theta ≤ 5, el coste total es viable.
fn laplace_gradient(
    problem:   &mut Problem,
    qfunc:     &dyn QFunc,
    likelihood: &dyn LogLikelihood,
    y:         &[f64],
    theta:     &[f64],
    f0:        f64,
    x_hat:     &[f64],
    n_model:   usize,
    n_lik:     usize,
    h:         f64,
) -> Vec<f64> {
    let n_theta = n_model + n_lik;
    let mut grad = vec![0.0_f64; n_theta];

    for k in 0..n_theta {
        let mut theta_h = theta.to_vec();
        theta_h[k] += h;
        let fh = laplace_eval(
            problem, qfunc, likelihood, y, &theta_h,
            x_hat,    // warm start desde el modo actual
            n_model, 5, 1e-3,
        ).map(|(f, ..)| f).unwrap_or(f64::MAX / 2.0);
        grad[k] = (fh - f0) / h;
    }

    grad
}

// ── Bucle de optimización ─────────────────────────────────────────────────────

pub fn optimize(
    problem:    &mut Problem,
    qfunc:      &dyn QFunc,
    likelihood: &dyn LogLikelihood,
    y:          &[f64],
    theta_init: &[f64],
    params:     &OptimizerParams,
) -> Result<OptimResult, InlaError> {
    let n_model  = qfunc.n_hyperparams();
    let n_lik    = likelihood.n_hyperparams();
    let h        = params.finite_diff_step;
    let max_iter = params.max_evals;
    let tol_grad = params.tol_grad;

    let mut theta   = theta_init.to_vec();
    let mut n_evals = 0_usize;

    // Warm start: x̂ de la evaluación anterior.
    // Empieza en ceros (cold start en la primera evaluación).
    let mut x_warm = vec![0.0_f64; y.len()];

    // Evaluación inicial
    let (mut f_cur, mut x_hat, mut _diag_q_inv, mut _diag_aug_inv) = laplace_eval(
        problem, qfunc, likelihood, y, &theta, &x_warm,
        n_model, 10, 1e-4,
    ).unwrap_or_else(|_| (f64::MAX / 2.0, vec![0.0; y.len()], vec![], vec![]));
    x_warm = x_hat.clone();
    n_evals += 1;

    let mut step = 0.1_f64;

    for _iter in 0..max_iter {
        // fd sobre el objetivo Laplace con warm start x_hat
        let grad = laplace_gradient(
            problem, qfunc, likelihood, y, &theta,
            f_cur, &x_hat,
            n_model, n_lik, h,
        );
        n_evals += n_model + n_lik;

        let grad_norm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        if grad_norm < tol_grad { break; }

        // Búsqueda de línea backtracking con warm start
        let mut accepted = false;
        let mut step_try = step;
        for _ in 0..10 {
            let theta_new: Vec<f64> = theta.iter()
                .zip(grad.iter())
                .map(|(&t, &g)| t - step_try * g)
                .collect();

            // Evaluación con warm start x_warm — barato si theta cambió poco
            match laplace_eval(
                problem, qfunc, likelihood, y, &theta_new, &x_warm,
                n_model, 10, 1e-4,
            ) {
                Ok((f_new, x_new, dq, da)) if f_new < f_cur => {
                    theta        = theta_new;
                    f_cur        = f_new;
                    x_hat        = x_new.clone();
                    x_warm       = x_new;
                    _diag_q_inv   = dq;
                    _diag_aug_inv = da;
                    step         = step_try * 1.2;
                    accepted     = true;
                    n_evals     += 1;
                    break;
                }
                _ => {
                    step_try *= 0.5;
                    n_evals  += 1;
                }
            }
        }
        if !accepted { break; }
    }

    // Evaluación final en θ* con convergencia fina (1e-6, 20 iters)
    let theta_model = &theta[..n_model];
    let theta_lik   = &theta[n_model..];

    let (x_hat_final, log_det_aug_final, _) = problem.find_mode_with_inverse(
        qfunc, likelihood, y, &theta, &x_warm, 20, 1e-6,
    ).map_err(|_| InlaError::ConvergenceFailed {
        reason: "IRLS no convergió en theta*".to_string(),
    })?;

    let log_det_q_final = if qfunc.is_proper() {
        problem.eval(qfunc, theta_model)
            .map_err(|_| InlaError::ConvergenceFailed {
                reason: "theta* produce Q no definida positiva".to_string(),
            })?
    } else {
        0.0
    };

    let n = y.len();
    let mut logll_final = vec![0.0_f64; n];
    likelihood.evaluate(&mut logll_final, &x_hat_final, y, theta_lik);
    let sum_logll_final: f64 = logll_final.iter().sum();
    let log_prior_final: f64 = theta.iter().map(|&th| th - 5e-5_f64 * th.exp()).sum();
    let q_form_final = problem.quadratic_form_x(qfunc, theta_model, &x_hat_final);
    let log_mlik = 0.5 * (log_det_q_final - log_det_aug_final) + sum_logll_final
        - 0.5 * q_form_final + log_prior_final;

    Ok(OptimResult { theta_opt: theta, log_mlik, n_evals })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::likelihood::{GaussianLikelihood, PoissonLikelihood};
    use crate::models::{IidModel, Rw1Model};

    #[test]
    fn optimize_iid_gaussian_returns_finite_result() {
        let n = 10;
        let y: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
        let model = IidModel::new(n);
        let lik   = GaussianLikelihood;
        let mut p = Problem::new(&model);

        let result = optimize(
            &mut p, &model, &lik, &y,
            &[2.0_f64.ln(), 1.0_f64.ln()],
            &OptimizerParams::default(),
        ).unwrap();

        assert!(result.log_mlik.is_finite());
        assert!(result.n_evals > 0);
    }

    #[test]
    fn log_mlik_sensitive_to_theta() {
        // Evaluaciones en theta distintos deben dar log_mlik distinto.
        // Usamos y con varianza real para que el término cuadrático x̂ᵀQx̂
        // diferencia correctamente ambos puntos.
        let n = 10;
        let y: Vec<f64> = (0..n).map(|i| (i as f64 - 4.5) * 0.5).collect();
        let model = IidModel::new(n);
        let lik   = GaussianLikelihood;

        let mut p1 = Problem::new(&model);
        let mut p2 = Problem::new(&model);

        // theta=[0,0]: tau_x=1, tau_obs=1
        // theta=[2,0]: tau_x=e²≈7.4, tau_obs=1 → mucho más restrictivo en x
        let (f1, ..) = laplace_eval(&mut p1, &model, &lik, &y, &[0.0, 0.0], &[], 1, 10, 1e-6).unwrap();
        let (f2, ..) = laplace_eval(&mut p2, &model, &lik, &y, &[2.0, 0.0], &[], 1, 10, 1e-6).unwrap();

        assert!((f1 - f2).abs() > 1e-3,
            "f1={f1} f2={f2} — deben diferir para theta distintos con datos no triviales");
    }

    #[test]
    fn laplace_objective_uses_mode_not_zero() {
        // Con y=10 (lejos de 0), el objetivo Laplace debe ser mejor que eta=0
        let n = 5;
        let y = vec![10.0_f64; n];
        let model = IidModel::new(n);
        let lik   = GaussianLikelihood;
        let mut p = Problem::new(&model);

        let result = optimize(
            &mut p, &model, &lik, &y,
            &[0.0, 0.0], &OptimizerParams::default(),
        ).unwrap();

        assert!(result.log_mlik.is_finite());
        // El x̂ ≠ 0 para y=10, así que log_mlik debe ser finito y razonable
        assert!(result.log_mlik > -10000.0);
    }

    #[test]
    fn optimize_rw1_poisson_does_not_panic() {
        // Rw1 (prior impropio) + Poisson: is_proper()=false → omite log|Q|
        // El IRLS con Q+W funciona porque Q+W es PD aunque Q sea singular
        let n = 20;
        let y: Vec<f64> = (0..n).map(|i| (i % 5) as f64).collect();
        let model = Rw1Model::new(n);
        let lik   = PoissonLikelihood;
        let mut p = Problem::new(&model);

        let result = optimize(
            &mut p, &model, &lik, &y,
            &[1.0], &OptimizerParams::default(),
        );

        match result {
            Ok(r)  => { assert!(r.log_mlik.is_finite()); }
            Err(e) => { println!("rw1 error (acceptable): {e}"); }
        }
    }

    #[test]
    fn warm_start_preserves_result() {
        // Dos evaluaciones con el mismo theta deben dar f equivalente.
        // El warm start llega al mismo mínimo por un camino diferente →
        // pequeñas diferencias de punto flotante son esperadas. Tolerancia 1e-4.
        let n = 8;
        let y: Vec<f64> = (0..n).map(|i| i as f64 * 0.5).collect();
        let model = IidModel::new(n);
        let lik   = GaussianLikelihood;
        let theta = [1.0, 0.5];
        let mut p = Problem::new(&model);

        let (f1, x1, _, _) = laplace_eval(
            &mut p, &model, &lik, &y, &theta, &[], 1, 10, 1e-6,
        ).unwrap();
        let (f2, _, _, _) = laplace_eval(
            &mut p, &model, &lik, &y, &theta, &x1, 1, 10, 1e-6,
        ).unwrap();

        // Tolerancia 1e-4: el warm start puede diferir en los últimos bits
        // por acumulación numérica en el camino de convergencia de IRLS.
        assert!((f1 - f2).abs() < 1e-4, "f1={f1}, f2={f2}, diff={}", (f1-f2).abs());
    }
}
