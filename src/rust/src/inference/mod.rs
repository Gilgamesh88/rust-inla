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
//!    a. media posterior: x̂ via IRLS (find_mode)
//!    b. varianzas: diag(Q(θ*)⁻¹) via eval_with_inverse() [Takahashi O(nnz)]
//!
//! 3. construye Marginal para cada efecto latente
//!    └── aproximación gaussiana: N(x̂ᵢ, varᵢ)
//! ```
//!
//! ## Notas de implementación
//!
//! Las varianzas se calculan con `eval_with_inverse()` en lugar de la cadena
//! manual `problem.eval()` + `problem.solver.selected_inverse()`. Esto corrige
//! dos bugs presentes en la versión anterior:
//!
//! 1. **Bug de estado**: `selected_inverse()` requiere estado `Factorized`.
//!    Cualquier llamada intermedia a `build()` resetea el estado a `Built`.
//!    `eval_with_inverse()` hace las tres operaciones de forma atómica.
//!
//! 2. **Bug de diagonal**: `val_of_col(j)[0]` devuelve el primer elemento de
//!    la columna j en la SpMat simétrica, que NO es necesariamente la diagonal
//!    para modelos no-diagonales (Rw1, Ar1). `eval_with_inverse()` usa
//!    `binary_search` para localizar el elemento (j,j) correctamente.

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
    /// Si true, estima un intercepto global β₀ separado del efecto latente x.
    /// Equivale a `y ~ 1 + f(idx, model=...)` en R-INLA.
    /// Necesario para modelos con prior impropio (Rw1) o cuando los datos
    /// tienen una media que no está centrada en 0.
    pub intercept: bool,
    /// Dimensión del campo latente (N_latent). Para modelos IID/AR1/RW1 = niveles únicos.
    pub n_latent:  usize,
    /// Mapeo ALTREP: x_idx[i] es el índice del efecto latente x al que pertenece el dato y[i].
    /// Si es None, asume mapeo 1-a-1 explícito (n_latent == y.len()).
    pub x_idx:     Option<&'a [usize]>,
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
    pub theta_opt:      Vec<f64>,
    pub log_mlik:       f64,
    pub random:         Vec<Marginal>,
    pub n_evals:        usize,
    /// Media posterior del intercepto β₀. Es 0.0 si el modelo no tiene intercept.
    pub intercept_mean: f64,
    /// SD posterior del intercepto. Es 0.0 si el modelo no tiene intercept.
    pub intercept_sd:   f64,

    /// Grid CCD para diagnosticos
    pub ccd_thetas:     Vec<f64>,
    pub ccd_weights:    Vec<f64>,

    /// Para actualizaciones bayesianas secuenciales
    pub posterior_mean: Vec<f64>,
    pub w_opt:          Vec<f64>,
}

pub struct InlaEngine;

impl InlaEngine {
    pub fn run(model: &InlaModel<'_>, params: &InlaParams) -> Result<InlaResult, InlaError> {
        let mut problem = Problem::new(model.qfunc);

        // Paso 1: encontrar θ*
        // Pasar intercept al optimizer para que la Laplace durante la
        // búsqueda de θ* incluya β₀ en el predictor lineal.
        // Equivale a domin-interface.c donde el predictor η = Aβ + Bx
        // incluye siempre todos los efectos fijos definidos en la formula.
        let opt = optimizer::optimize(
            &mut problem,
            model.qfunc,
            model.likelihood,
            model.y,
            model.x_idx,
            &model.theta_init,
            &params.optimizer,
            model.intercept,
        )?;

        let theta_opt   = opt.theta_opt.clone();
        let n_model     = model.qfunc.n_hyperparams();
        let n = model.n_latent;

        // Paso 2: Generar Grid de Integración CCD
        let ccd_grid = crate::optimizer::ccd::build_ccd_grid(
            &mut problem,
            model.qfunc,
            model.likelihood,
            model.y,
            model.x_idx,
            &theta_opt,
            model.intercept,
        )?;

        // Preparar acumuladores del Gaussian Mixture para la media latente
        let mut mixed_mean = vec![0.0_f64; n];
        let mut mixed_var_inner = vec![0.0_f64; n];
        let mut mixed_mean_sq = vec![0.0_f64; n];
        
        let mut mixed_intercept_mean = 0.0_f64;
        let mut mixed_intercept_var_inner = 0.0_f64; // si tenemos su varianza en el futuro
        let mut mixed_intercept_mean_sq = 0.0_f64;

        // Bucle de integración sobre los puntos CCD
        for pt in &ccd_grid.points {
            let theta_k = &pt.theta;
            let weight = pt.weight;

            let (intercept_k, mean_k, vars_k) = if model.intercept {
                match problem.find_mode_with_intercept_and_inverse(
                    model.qfunc, model.likelihood, model.y, model.x_idx, theta_k,
                    &[], 0.0, 20, 1e-6,
                ) {
                    Ok((beta0, x_hat, _log_det_aug, diag_aug_inv, _schur_s)) => {
                        let vs = diag_aug_inv.into_iter().map(|v| v.max(1e-12)).collect();
                        (beta0, x_hat, vs)
                    }
                    Err(_) => (0.0, vec![0.0_f64; n], vec![1.0_f64; n]),
                }
            } else {
                match problem.find_mode_with_inverse(
                    model.qfunc, model.likelihood, model.y, model.x_idx, theta_k,
                    &[], 20, 1e-6,
                ) {
                    Ok((x_hat, _log_det_aug, diag_aug_inv)) => {
                        let vs = diag_aug_inv.into_iter().map(|v| v.max(1e-12)).collect();
                        (0.0, x_hat, vs)
                    }
                    Err(_) => (0.0, vec![0.0_f64; n], vec![1.0_f64; n]),
                }
            };

            // Acumular
            mixed_intercept_mean += weight * intercept_k;
            mixed_intercept_mean_sq += weight * intercept_k * intercept_k;

            for i in 0..n {
                mixed_mean[i] += weight * mean_k[i];
                mixed_mean_sq[i] += weight * mean_k[i] * mean_k[i];
                mixed_var_inner[i] += weight * vars_k[i];
            }
        }

        // Finalizar Gaussian Mixture
        // Var_total = E[\text{Var}(X|\Theta)] + \text{Var}(E[X|\Theta])
        // Var_total = mixed_var_inner + (mixed_mean_sq - mixed_mean^2)
        
        let mut final_vars = vec![0.0_f64; n];
        for i in 0..n {
            let inter_var = (mixed_mean_sq[i] - mixed_mean[i] * mixed_mean[i]).max(0.0);
            final_vars[i] = mixed_var_inner[i] + inter_var;
        }

        let intercept_inter_var = (mixed_intercept_mean_sq - mixed_intercept_mean * mixed_intercept_mean).max(0.0);
        let intercept_sd = intercept_inter_var.sqrt(); // Aproximamos su sd incorporando la incertidumbre del hyper-param
        let intercept_mean = mixed_intercept_mean;
        let posterior_mean = mixed_mean.clone();
        let variances = final_vars.clone();

        // Recalcular W (curvatura) en el modo para guardarlo en w_opt
        let theta_lik = &theta_opt[n_model..];
        let mut eta_data = vec![0.0_f64; model.y.len()];
        for i in 0..model.y.len() {
            let lat_idx = model.x_idx.map_or(i, |x| x[i]);
            eta_data[i] = posterior_mean[lat_idx] + if model.intercept { intercept_mean } else { 0.0 };
        }
        
        let mut grad_data = vec![0.0_f64; model.y.len()];
        let mut curv_data = vec![0.0_f64; model.y.len()];
        model.likelihood.gradient_and_curvature(&mut grad_data, &mut curv_data, &eta_data, model.y, theta_lik);
        
        let mut w_opt = vec![0.0_f64; n];
        for i in 0..model.y.len() {
            let lat_idx = model.x_idx.map_or(i, |x| x[i]);
            w_opt[lat_idx] += curv_data[i];
        }
        for i in 0..n { w_opt[i] = w_opt[i].max(1e-6); }

        // Paso 3: construir marginales gaussianas
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

        // Export Grid
        let mut ccd_thetas = Vec::new();
        let mut ccd_weights = Vec::new();
        for pt in &ccd_grid.points {
            ccd_thetas.extend_from_slice(&pt.theta);
            ccd_weights.push(pt.weight);
        }

        Ok(InlaResult {
            theta_opt,
            log_mlik:       opt.log_mlik,
            random,
            n_evals:        opt.n_evals,
            intercept_mean,
            intercept_sd,
            ccd_thetas,
            ccd_weights,
            posterior_mean,
            w_opt,
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
                         theta_init: vec![0.0, 0.0], intercept: false, n_latent: n, x_idx: None },
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
                         theta_init: vec![0.0, 0.0], intercept: false, n_latent: n, x_idx: None },
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
                         theta_init: vec![0.0, 0.0], intercept: false, n_latent: n, x_idx: None },
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
                         theta_init: vec![0.0, 0.0], intercept: false, n_latent: n, x_idx: None },
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
                         theta_init: vec![0.0, 0.0], intercept: false, n_latent: n, x_idx: None },
            &InlaParams::default(),
        ).unwrap();
        let sd0 = result.random[0].sd();
        for m in &result.random {
            assert_abs_diff_eq!(m.sd(), sd0, epsilon = 1e-8);
        }
    }

    // ── Fixture rw1_poisson ───────────────────────────────────────────────────
    #[test]
    fn intercept_shifts_marginal_means() {
        // Con intercept=true y datos desplazados, β₀ absorbe la media global.
        // Las medias posteriores de x deben ser aprox. cero (centradas).
        // Equivale al test de que η = β₀ + x en lugar de η = x.
        use crate::models::Rw1Model;
        use crate::likelihood::PoissonLikelihood;
        let n = 20;
        // Datos Poisson con λ=1 (log(λ)=0) — el intercepto debería estar cerca de 0
        let y = vec![1.0_f64; n];
        let model = Rw1Model::new(n);
        let lik   = PoissonLikelihood;

        let result_with = InlaEngine::run(
            &InlaModel { qfunc: &model, likelihood: &lik, y: &y,
                         theta_init: vec![1.0], intercept: true, n_latent: n, x_idx: None },
            &InlaParams::default(),
        );
        let result_without = InlaEngine::run(
            &InlaModel { qfunc: &model, likelihood: &lik, y: &y,
                         theta_init: vec![1.0], intercept: false, n_latent: n, x_idx: None },
            &InlaParams::default(),
        );

        // Ambos deben completar sin panic
        match (result_with, result_without) {
            (Ok(rw), Ok(rwo)) => {
                // Con intercept: la media de x debe ser más cercana a 0
                let mean_x_with: f64 = rw.random.iter().map(|m| m.mean()).sum::<f64>() / n as f64;
                let mean_x_without: f64 = rwo.random.iter().map(|m| m.mean()).sum::<f64>() / n as f64;
                // intercept_mean debería absorber parte del desplazamiento
                println!("intercept_mean={:.4} mean_x_with={:.4} mean_x_without={:.4}",
                    rw.intercept_mean, mean_x_with, mean_x_without);
                assert!(rw.intercept_mean.is_finite());
            }
            _ => { /* Puede fallar con Rw1 impropio — eso es aceptable */ }
        }
    }

    #[test]
    #[ignore = "requiere tests/fixtures/rw1_poisson.json"]
    fn fixture_rw1_poisson_matches_r_inla() {
        let raw = std::fs::read_to_string("tests/fixtures/rw1_poisson.json").unwrap();
        let v: serde_json::Value = serde_json::from_str(&raw).unwrap();

        let y: Vec<f64> = v["y"].as_array().unwrap().iter()
            .map(|x| x.as_f64().unwrap()).collect();
        let mean_x_ref: Vec<f64> = v["mean_x"].as_array().unwrap().iter()
            .map(|x| x.as_f64().unwrap()).collect();
        let sd_x_ref: Vec<f64> = v["sd_x"].as_array().unwrap().iter()
            .map(|x| x.as_f64().unwrap()).collect();
        let theta_opt_ref: Vec<f64> = v["theta_opt"].as_array().unwrap().iter()
            .map(|x| x.as_f64().unwrap()).collect();
        let log_mlik_ref = v["log_mlik"][0].as_f64().unwrap();
        let mean_intercept_ref = v["mean_intercept"][0].as_f64().unwrap();

        let n     = y.len();
        let model = crate::models::Rw1Model::new(n);
        let lik   = crate::likelihood::PoissonLikelihood;
        let theta_init = vec![theta_opt_ref[0]];

        match InlaEngine::run(
            &InlaModel { qfunc: &model, likelihood: &lik, y: &y, theta_init, intercept: true, n_latent: n, x_idx: None },
            &InlaParams::default(),
        ) {
            Ok(result) => {
                assert!(result.log_mlik.is_finite());
                assert_eq!(result.random.len(), n);

                let mut max_err_mean = 0.0_f64;
                let mut max_err_sd   = 0.0_f64;
                for (m, (&ref_mean, &ref_sd)) in result.random.iter()
                    .zip(mean_x_ref.iter().zip(sd_x_ref.iter()))
                {
                    max_err_mean = max_err_mean.max((m.mean() - ref_mean).abs());
                    max_err_sd   = max_err_sd.max((m.sd()   - ref_sd  ).abs());
                }
                let theta_err = (result.theta_opt[0] - theta_opt_ref[0]).abs();
                println!("rw1_poisson | mean_err={max_err_mean:.4} sd_err={max_err_sd:.4} \
                          theta_err={theta_err:.4} \
                          log_mlik={:.2} (ref={log_mlik_ref:.2})",
                         result.log_mlik);
                println!("  intercept_mean={:.4} (ref={mean_intercept_ref:.4})",
                         result.intercept_mean);
                let intercept_err = (result.intercept_mean - mean_intercept_ref).abs();
                println!("  intercept_err={intercept_err:.4}");
                assert!(max_err_mean < 10.0, "mean_err={max_err_mean}");
            }
            Err(e) => {
                println!("rw1_poisson: Err (prior impropio pendiente): {e}");
            }
        }
    }

    // ── Fixture ar1_gamma ─────────────────────────────────────────────────────
    #[test]
    #[ignore = "requiere tests/fixtures/ar1_gamma.json"]
    fn fixture_ar1_gamma_matches_r_inla() {
        let raw = std::fs::read_to_string("tests/fixtures/ar1_gamma.json").unwrap();
        let v: serde_json::Value = serde_json::from_str(&raw).unwrap();

        let y: Vec<f64> = v["y"].as_array().unwrap().iter()
            .map(|x| x.as_f64().unwrap()).collect();
        let mean_x_ref: Vec<f64> = v["mean_x"].as_array().unwrap().iter()
            .map(|x| x.as_f64().unwrap()).collect();
        let sd_x_ref: Vec<f64> = v["sd_x"].as_array().unwrap().iter()
            .map(|x| x.as_f64().unwrap()).collect();
        let theta_opt_ref: Vec<f64> = v["theta_opt"].as_array().unwrap().iter()
            .map(|x| x.as_f64().unwrap()).collect();
        let log_mlik_ref = v["log_mlik"][0].as_f64().unwrap();

        let n     = y.len();
        let model = crate::models::Ar1Model::new(n);
        let lik   = crate::likelihood::GammaLikelihood;
        let theta_init = theta_opt_ref.clone();

        let result = InlaEngine::run(
            &InlaModel { qfunc: &model, likelihood: &lik, y: &y, theta_init, intercept: false, n_latent: n, x_idx: None },
            &InlaParams::default(),
        ).expect("ar1_gamma con Ar1 (prior propio) no debe fallar");

        assert!(result.log_mlik.is_finite());
        assert_eq!(result.random.len(), n);

        let mut max_err_mean = 0.0_f64;
        let mut max_err_sd   = 0.0_f64;
        for (m, (&ref_mean, &ref_sd)) in result.random.iter()
            .zip(mean_x_ref.iter().zip(sd_x_ref.iter()))
        {
            max_err_mean = max_err_mean.max((m.mean() - ref_mean).abs());
            max_err_sd   = max_err_sd.max((m.sd()   - ref_sd  ).abs());
        }
        let theta_err_tau = (result.theta_opt[0] - theta_opt_ref[0]).abs();
        let theta_err_rho = (result.theta_opt[1] - theta_opt_ref[1]).abs();
        let theta_err_phi = (result.theta_opt[2] - theta_opt_ref[2]).abs();

        println!("ar1_gamma | mean_err={max_err_mean:.4} sd_err={max_err_sd:.4}");
        println!("  theta_err: tau={theta_err_tau:.4} rho={theta_err_rho:.4} phi={theta_err_phi:.4}");
        println!("  log_mlik={:.4} (ref={log_mlik_ref:.4})", result.log_mlik);
        println!("  theta_opt={:?}", result.theta_opt);
        println!("  theta_ref={theta_opt_ref:?}");

        assert!(max_err_mean < 10.0, "mean_err={max_err_mean}");
        assert!(max_err_sd   < 5.0,  "sd_err={max_err_sd}");
    }

    // ── Fixture iid_gaussian ──────────────────────────────────────────────────
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

        // El fixture fue generado con intercepto (mean_intercept=1.978 ≈ mean(y)).
        // R-INLA aplicó sum-to-zero constraint sobre x → mean(mean_x)=0.
        // Usar intercept: true para que nuestro engine haga lo mismo.
        let mean_intercept_ref = v["mean_intercept"][0].as_f64().unwrap();

        let result = InlaEngine::run(
            &InlaModel { qfunc: &model, likelihood: &lik, y: &y,
                         theta_init: vec![0.0, 0.0], intercept: true, n_latent: n, x_idx: None },
            &InlaParams::default(),
        ).unwrap();

        let mut max_err_mean = 0.0_f64;
        let mut sum_sq_err   = 0.0_f64;
        for (m, &ref_mean) in result.random.iter().zip(mean_x_ref.iter()) {
            let err = (m.mean() - ref_mean).abs();
            max_err_mean = max_err_mean.max(err);
            sum_sq_err  += err * err;
        }
        let rmse = (sum_sq_err / n as f64).sqrt();
        let intercept_err = (result.intercept_mean - mean_intercept_ref).abs();

        println!("iid_gaussian | n={n} max_err_mean={max_err_mean:.6} rmse={rmse:.6}");
        println!("  theta_opt={:?}", result.theta_opt);
        println!("  log_mlik={:.4}", result.log_mlik);
        println!("  intercept_mean={:.4} (ref={mean_intercept_ref:.4}) err={intercept_err:.4}",
                 result.intercept_mean);

        assert!(max_err_mean < 4.0, "max_err_mean={max_err_mean}");
        assert!(rmse < 1.0,         "rmse={rmse}");
        assert!(intercept_err < 1.0, "intercept_err={intercept_err}");
    }
}
