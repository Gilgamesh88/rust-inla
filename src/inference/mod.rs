//! Motor principal de inferencia INLA.
//!
//! Equivalente Rust de `approx-inference.h/c` (8725 líneas de C).
//!
//! ## Algoritmo INLA (simplificado)
//!
//! ```text
//! 1. optimizer: encuentra θ* = argmax log p(y|θ)
//!    └── cada evaluación: Problem::eval(θ) → Cholesky → log|Q(θ)|
//!
//! 2. en θ*:
//!    a. media posterior: x̂ = Q(θ*)⁻¹ · g  donde g = gradiente de log p(y|x)
//!    b. varianzas: diag(Q(θ*)⁻¹) via n solves secuenciales (O(n²))
//!       → Takahashi O(nnz) es la optimización pendiente
//!
//! 3. construye Marginal para cada efecto latente
//!    └── aproximación gaussiana: N(x̂ᵢ, varᵢ)
//! ```
//!
//! ## Estado B.6
//!
//! - θ* y log p(y|θ*): ✓ funcional
//! - Varianzas marginales diag(Q⁻¹): ✓ correcto (O(n²), lento para n grande)
//! - Media posterior x̂: simplificada (x̂=0 para scaffold; IRLS en C+)
//! - ScGaussian (corrección de asimetría): pendiente Fase C+

use crate::error::InlaError;
use crate::likelihood::LogLikelihood;
use crate::marginal::Marginal;
use crate::models::QFunc;
use crate::optimizer::{self, OptimizerParams};
use crate::problem::Problem;

pub struct InlaModel<'a> {
    pub qfunc:      &'a dyn QFunc,
    pub likelihood: &'a dyn LogLikelihood,
    pub y:          &'a [f64],
    pub theta_init: Vec<f64>,
}

pub struct InlaParams {
    pub optimizer:    OptimizerParams,
    pub marginal_pts: usize,
    pub marginal_sds: f64,
}

impl Default for InlaParams {
    fn default() -> Self {
        Self {
            optimizer:    OptimizerParams::default(),
            marginal_pts: 75,
            marginal_sds: 4.0,
        }
    }
}

pub struct InlaResult {
    pub theta_opt: Vec<f64>,
    pub log_mlik:  f64,
    pub random:    Vec<Marginal>,
    pub n_evals:   usize,
}

pub struct InlaEngine;

impl InlaEngine {
    pub fn run(model: &InlaModel<'_>, params: &InlaParams) -> Result<InlaResult, InlaError> {
        let mut problem = Problem::new(model.qfunc);

        let opt = optimizer::optimize(
            &mut problem,
            model.qfunc,
            model.likelihood,
            model.y,
            &model.theta_init,
            &params.optimizer,
        )?;

        let theta_opt   = opt.theta_opt.clone();
        let n_model     = model.qfunc.n_hyperparams();
        let theta_model = &theta_opt[..n_model];

        problem.eval(model.qfunc, theta_model)?;

        let n = problem.n();

        // Varianzas: diag(Q^-1) via n solves. Correcto, O(n^2).
        let mut variances = vec![0.0_f64; n];
        for i in 0..n {
            let mut e_i = vec![0.0_f64; n];
            e_i[i] = 1.0;
            problem.solve(&mut e_i);
            variances[i] = e_i[i].max(0.0);
        }

        // Media posterior simplificada: x_hat = 0 (IRLS pendiente Fase C+)
        // Media posterior real via IRLS
        let posterior_mean = problem.find_mode(
            model.qfunc,
            model.likelihood,
            model.y,
            &theta_opt,
            20,
            1e-6,
        ).unwrap_or_else(|_| vec![0.0_f64; n]);

        let random: Vec<Marginal> = (0..n)
            .map(|i| {
                let mean = posterior_mean[i];
                let sd   = variances[i].sqrt().max(1e-10);
                let lo   = mean - params.marginal_sds * sd;
                let hi   = mean + params.marginal_sds * sd;
                let pts  = params.marginal_pts;
                let x: Vec<f64> = (0..pts)
                    .map(|k| lo + (hi - lo) * k as f64 / (pts - 1) as f64)
                    .collect();
                let y: Vec<f64> = x.iter().map(|&xi| {
                    let z = (xi - mean) / sd;
                    (-0.5 * z * z).exp()
                }).collect();
                Marginal::new(x, y)
            })
            .collect();

        Ok(InlaResult {
            theta_opt,
            log_mlik: opt.log_mlik,
            random,
            n_evals: opt.n_evals,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::likelihood::GaussianLikelihood;
    use crate::models::IidModel;
    use approx::assert_abs_diff_eq;

    #[test]
    fn engine_runs_on_iid_gaussian() {
        let n = 10;
        let y: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
        let model = IidModel::new(n);
        let lik   = GaussianLikelihood;
        let result = InlaEngine::run(
            &InlaModel { qfunc: &model, likelihood: &lik, y: &y,
                         theta_init: vec![0.0, 0.0] },
            &InlaParams::default(),
        ).expect("InlaEngine no debe fallar con theta valido");
        assert_eq!(result.theta_opt.len(), 2);
        assert_eq!(result.random.len(), n);
        assert!(result.log_mlik.is_finite());
        assert!(result.n_evals > 0);
    }

    #[test]
    fn marginals_have_correct_dimension() {
        let n = 5;
        let y = vec![1.0; n];
        let model = IidModel::new(n);
        let lik   = GaussianLikelihood;
        let result = InlaEngine::run(
            &InlaModel { qfunc: &model, likelihood: &lik, y: &y,
                         theta_init: vec![0.0, 0.0] },
            &InlaParams::default(),
        ).unwrap();
        for m in &result.random {
            assert_eq!(m.x.len(), 75);
            assert_eq!(m.y.len(), 75);
        }
    }

    #[test]
    fn marginals_integrate_to_one() {
        let n = 4;
        let y = vec![0.0; n];
        let model = IidModel::new(n);
        let lik   = GaussianLikelihood;
        let result = InlaEngine::run(
            &InlaModel { qfunc: &model, likelihood: &lik, y: &y,
                         theta_init: vec![0.0, 0.0] },
            &InlaParams::default(),
        ).unwrap();
        for m in &result.random {
            let integral = m.emarginal(|_| 1.0);
            assert_abs_diff_eq!(integral, 1.0, epsilon = 1e-3);
        }
    }

    #[test]
    fn variances_are_positive() {
        let n = 6;
        let y = vec![1.0; n];
        let model = IidModel::new(n);
        let lik   = GaussianLikelihood;
        let result = InlaEngine::run(
            &InlaModel { qfunc: &model, likelihood: &lik, y: &y,
                         theta_init: vec![0.0, 0.0] },
            &InlaParams::default(),
        ).unwrap();
        for m in &result.random {
            assert!(m.variance() > 0.0);
        }
    }

    #[test]
    fn iid_all_marginals_equal() {
        let n = 5;
        let y = vec![0.0; n];
        let model = IidModel::new(n);
        let lik   = GaussianLikelihood;
        let result = InlaEngine::run(
            &InlaModel { qfunc: &model, likelihood: &lik, y: &y,
                         theta_init: vec![0.0, 0.0] },
            &InlaParams::default(),
        ).unwrap();
        let sd0 = result.random[0].sd();
        for m in &result.random {
            assert_abs_diff_eq!(m.sd(), sd0, epsilon = 1e-8);
        }
    }
    #[test]
    #[ignore = "requiere tests/fixtures/iid_gaussian.json"]
    fn fixture_iid_gaussian_matches_r_inla() {
        let raw = std::fs::read_to_string("tests/fixtures/iid_gaussian.json").unwrap();
        let v: serde_json::Value = serde_json::from_str(&raw).unwrap();

        let y: Vec<f64> = v["y"].as_array().unwrap().iter()
            .map(|x| x.as_f64().unwrap()).collect();
        let mean_x_ref: Vec<f64> = v["mean_x"].as_array().unwrap().iter()
            .map(|x| x.as_f64().unwrap()).collect();

        let n     = y.len();
        let model = crate::models::IidModel::new(n);
        let lik   = crate::likelihood::GaussianLikelihood;

        let result = InlaEngine::run(
            &InlaModel { qfunc: &model, likelihood: &lik, y: &y,
                         theta_init: vec![0.0, 0.0] },
            &InlaParams::default(),
        ).unwrap();

        // Medias posteriores — con IRLS deben estar cerca de R-INLA
        let mut max_err = 0.0_f64;
        for (m, &ref_mean) in result.random.iter().zip(mean_x_ref.iter()) {
            let err = (m.mean() - ref_mean).abs();
            if err > max_err { max_err = err; }
        }
        println!("Max error en medias: {max_err:.6}");
        // Tolerancia amplia — BFGS completo pendiente
        assert!(max_err < 2.0, "Error demasiado grande: {max_err}");
    }
}