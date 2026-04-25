use std::time::Instant;

use crate::diagnostics::RunDiagnosticsSummary;
use crate::error::InlaError;
use crate::likelihood::LogLikelihood;
use crate::marginal::Marginal;
use crate::models::QFunc;
use crate::optimizer::{self, LaplaceDecomposition, OptimizerParams};
use crate::problem::Problem;

pub struct InlaModel<'a> {
    pub qfunc: &'a dyn QFunc,
    pub likelihood: &'a dyn LogLikelihood,
    pub y: &'a [f64],
    pub theta_init: Vec<f64>,
    pub latent_init: Vec<f64>,
    pub fixed_init: Vec<f64>,
    pub fixed_matrix: Option<&'a [f64]>,
    pub n_fixed: usize,
    pub n_latent: usize,
    pub a_i: Option<&'a [usize]>,
    pub a_j: Option<&'a [usize]>,
    pub a_x: Option<&'a [f64]>,
    pub offset: Option<&'a [f64]>,
    pub extr_constr: Option<&'a [f64]>,
    pub n_constr: usize,
}

pub struct InlaParams {
    pub optimizer: OptimizerParams,
    pub marginal_pts: usize,
    pub marginal_sds: f64,
    pub skip_ccd: bool,
}

impl Default for InlaParams {
    fn default() -> Self {
        Self {
            optimizer: OptimizerParams::default(),
            marginal_pts: 75,
            marginal_sds: 4.0,
            skip_ccd: false,
        }
    }
}

pub struct InlaResult {
    pub theta_opt: Vec<f64>,
    pub log_mlik: f64,
    pub log_mlik_theta_opt: f64,
    pub log_mlik_theta_laplace: f64,
    pub theta_laplace_correction: f64,
    pub random: Vec<Marginal>,
    pub fitted: Vec<Marginal>,
    pub n_evals: usize,
    pub fixed_means: Vec<f64>,
    pub fixed_sds: Vec<f64>,
    pub fixed_var_theta_opt: Vec<f64>,
    pub ccd_thetas: Vec<f64>,
    pub ccd_base_weights: Vec<f64>,
    pub ccd_weights: Vec<f64>,
    pub ccd_log_mlik: Vec<f64>,
    pub ccd_log_weight: Vec<f64>,
    pub ccd_hessian_eigenvalues: Vec<f64>,
    pub posterior_mean: Vec<f64>,
    pub latent_var_theta_opt: Vec<f64>,
    pub latent_var_within_theta: Vec<f64>,
    pub latent_var_between_theta: Vec<f64>,
    pub w_opt: Vec<f64>,
    pub laplace_terms: LaplaceDecomposition,
    pub mode_x: Vec<f64>,
    pub mode_beta: Vec<f64>,
    pub mode_eta: Vec<f64>,
    pub mode_grad: Vec<f64>,
    pub mode_curvature_raw: Vec<f64>,
    pub mode_curvature: Vec<f64>,
    pub diagnostics: RunDiagnosticsSummary,
}

pub struct InlaEngine;

impl InlaEngine {
    pub fn run(model: &InlaModel<'_>, params: &InlaParams) -> Result<InlaResult, InlaError> {
        let mut problem = Problem::new(model);

        let opt = optimizer::optimize(&mut problem, model, &model.theta_init, &params.optimizer)?;

        let theta_opt = opt.theta_opt.clone();
        let n_model = model.qfunc.n_hyperparams();
        let n = model.n_latent;
        let k = model.n_fixed;

        let mut fixed_var_theta_opt = vec![0.0_f64; k];
        let latent_var_theta_opt = if k > 0 {
            let (_, _, _, diag_aug_inv, fixed_cov, _, _) = problem
                .find_mode_with_fixed_effects_with_cov(
                    model,
                    &theta_opt,
                    &opt.mode_x,
                    &opt.mode_beta,
                    20,
                    1e-8,
                )
                .map_err(|err| InlaError::ConvergenceFailed {
                    reason: format!(
                        "Theta-opt conditional covariance solve failed while solving fixed effects: {err}"
                    ),
                })?;
            for j in 0..k {
                fixed_var_theta_opt[j] = fixed_cov[j * k + j];
            }
            diag_aug_inv.into_iter().map(|v| v.max(1e-12)).collect()
        } else {
            let (_, _, diag_aug_inv) = problem
                .find_mode_with_inverse(model, &theta_opt, &opt.mode_x, 20, 1e-8)
                .map_err(|err| InlaError::ConvergenceFailed {
                    reason: format!(
                        "Theta-opt conditional covariance solve failed while solving latent mode: {err}"
                    ),
                })?;
            diag_aug_inv.into_iter().map(|v| v.max(1e-12)).collect()
        };

        let ccd_grid = if params.skip_ccd {
            crate::optimizer::ccd::CcdIntegration {
                points: vec![crate::optimizer::ccd::CcdPoint {
                    theta: theta_opt.clone(),
                    base_weight: 1.0,
                    weight: 1.0,
                    log_mlik: opt.log_mlik,
                    log_weight: opt.log_mlik,
                }],
                hessian_eigenvalues: vec![],
            }
        } else {
            crate::optimizer::ccd::build_ccd_grid(&mut problem, model, &theta_opt)?
        };
        let log_mlik_theta_opt = opt.log_mlik;
        let theta_laplace_correction = ccd_grid.theta_laplace_correction();
        let log_mlik_theta_laplace = log_mlik_theta_opt + theta_laplace_correction;

        let mut mixed_mean = vec![0.0_f64; n];
        let mut mixed_var_inner = vec![0.0_f64; n];
        let mut mixed_mean_sq = vec![0.0_f64; n];

        let mut mixed_fixed_mean = vec![0.0_f64; k];
        let mut mixed_fixed_second_moment = vec![0.0_f64; k * k];
        let mut mixed_fixed_cov_inner = vec![0.0_f64; k * k];
        let mut mixed_latent_fixed_second_moment = vec![0.0_f64; n * k];
        let mut mixed_latent_fixed_cov_inner = vec![0.0_f64; n * k];
        let mut next_x_warm = opt.mode_x.clone();
        let mut next_beta_warm = opt.mode_beta.clone();

        for (pt_idx, pt) in ccd_grid.points.iter().enumerate() {
            let theta_k = &pt.theta;
            let weight = pt.weight;

            let x_warm = if next_x_warm.len() == n {
                next_x_warm.clone()
            } else {
                vec![0.0_f64; n]
            };
            let mut beta_warm = if next_beta_warm.len() == k {
                next_beta_warm.clone()
            } else {
                vec![0.0_f64; k]
            };

            // Frequency Regime Log-Link Warm Start
            // Bypasses the Newton-Raphson hurdle when the target frequency is extremely low (e.g. freMTPL2freq ~ 0.05).
            if k > 0
                && matches!(
                    model.likelihood.link(),
                    crate::likelihood::LinkFunction::Log
                )
            {
                let valid_y: Vec<f64> = model.y.iter().copied().filter(|y| !y.is_nan()).collect();
                if !valid_y.is_empty() {
                    let avg_y = valid_y.iter().sum::<f64>() / valid_y.len() as f64;
                    if avg_y > 0.0 && avg_y < 0.2 {
                        beta_warm[0] = avg_y.ln();
                    }
                }
            }

            let (fixed_k, mean_k, vars_k, fixed_cov_k, latent_fixed_cov_k) = if k > 0 {
                let (beta, x_hat, _, diag_aug_inv, fixed_cov, latent_fixed_cov, _) = problem
                    .find_mode_with_fixed_effects_with_cov(
                        model, theta_k, &x_warm, &beta_warm, 20, 1e-8,
                    )
                    .map_err(|err| InlaError::ConvergenceFailed {
                        reason: format!(
                            "CCD point {pt_idx} failed while solving fixed effects: {err}"
                        ),
                    })?;
                let vs: Vec<f64> = diag_aug_inv.into_iter().map(|v| v.max(1e-12)).collect();
                (beta, x_hat, vs, fixed_cov, latent_fixed_cov)
            } else {
                let (x_hat, _, diag_aug_inv) = problem
                    .find_mode_with_inverse(model, theta_k, &x_warm, 20, 1e-8)
                    .map_err(|err| InlaError::ConvergenceFailed {
                        reason: format!(
                            "CCD point {pt_idx} failed while solving latent mode: {err}"
                        ),
                    })?;
                let vs: Vec<f64> = diag_aug_inv.into_iter().map(|v| v.max(1e-12)).collect();
                (vec![], x_hat, vs, vec![], vec![])
            };

            next_x_warm = mean_k.clone();
            next_beta_warm = fixed_k.clone();

            for (j1, fixed_value) in fixed_k.iter().enumerate().take(k) {
                mixed_fixed_mean[j1] += weight * *fixed_value;
                for j2 in 0..k {
                    mixed_fixed_second_moment[j1 * k + j2] += weight * *fixed_value * fixed_k[j2];
                    mixed_fixed_cov_inner[j1 * k + j2] += weight * fixed_cov_k[j1 * k + j2];
                }
            }

            for i in 0..n {
                mixed_mean[i] += weight * mean_k[i];
                mixed_mean_sq[i] += weight * mean_k[i] * mean_k[i];
                mixed_var_inner[i] += weight * vars_k[i];
                for j in 0..k {
                    mixed_latent_fixed_second_moment[i + j * n] += weight * mean_k[i] * fixed_k[j];
                    mixed_latent_fixed_cov_inner[i + j * n] +=
                        weight * latent_fixed_cov_k[i + j * n];
                }
            }
        }

        let mut inter_var = vec![0.0_f64; n];
        let mut final_vars = vec![0.0_f64; n];
        for i in 0..n {
            inter_var[i] = (mixed_mean_sq[i] - mixed_mean[i] * mixed_mean[i]).max(0.0);
            final_vars[i] = mixed_var_inner[i] + inter_var[i];
        }

        let mut mixed_fixed_cov = mixed_fixed_cov_inner;
        for j1 in 0..k {
            for j2 in 0..k {
                mixed_fixed_cov[j1 * k + j2] += mixed_fixed_second_moment[j1 * k + j2]
                    - mixed_fixed_mean[j1] * mixed_fixed_mean[j2];
            }
        }

        let mut mixed_latent_fixed_cov = mixed_latent_fixed_cov_inner;
        for j in 0..k {
            for i in 0..n {
                mixed_latent_fixed_cov[i + j * n] += mixed_latent_fixed_second_moment[i + j * n]
                    - mixed_mean[i] * mixed_fixed_mean[j];
            }
        }

        let mut fixed_sds = vec![0.0_f64; k];
        for j in 0..k {
            fixed_sds[j] = mixed_fixed_cov[j * k + j].max(0.0).sqrt();
        }

        let mut posterior_mean = mixed_mean.clone();
        let variances = final_vars.clone();

        // --------------------------------------------------------------------
        // SUM-TO-ZERO identifiability adjustment for intrinsic latent fields.
        //
        // INLA devel defaults differ by latent model: intrinsic models like rw1
        // and rw2 use constraints by default, while proper models like iid and
        // ar1 do not. Applying this projection to proper models shifts the
        // latent mean into the intercept and creates an artificial intercept gap.
        //
        // This is still a coarse whole-field switch for compound models. When we
        // add mixed proper/improper blocks, this needs to become block-specific.
        // --------------------------------------------------------------------
        if k > 0 && n > 0 && !model.qfunc.is_proper() {
            let mean_x = posterior_mean.iter().sum::<f64>() / n as f64;
            for posterior_mean_i in posterior_mean.iter_mut().take(n) {
                *posterior_mean_i -= mean_x;
            }
            // Assume the first fixed effect is the global intercept.
            mixed_fixed_mean[0] += mean_x;
        }

        let mut a_rows = vec![vec![]; model.y.len()];
        if let (Some(a_i), Some(a_j), Some(a_x)) = (model.a_i, model.a_j, model.a_x) {
            for k in 0..a_i.len() {
                a_rows[a_i[k]].push((a_j[k], a_x[k]));
            }
        } else {
            for (i, row) in a_rows.iter_mut().enumerate().take(model.y.len().min(n)) {
                row.push((i, 1.0));
            }
        }

        let theta_lik = &theta_opt[n_model..];
        let mut eta_data = vec![0.0_f64; model.y.len()];
        for i in 0..model.y.len() {
            let mut ax_sum = 0.0;
            for &(j, ax) in &a_rows[i] {
                ax_sum += ax * posterior_mean[j];
            }
            let mut xb = 0.0;
            if let Some(fixed_matrix) = model.fixed_matrix {
                for (j, fixed_mean) in mixed_fixed_mean.iter().enumerate().take(k) {
                    xb += fixed_matrix[i + j * model.y.len()] * *fixed_mean;
                }
            }
            eta_data[i] = ax_sum + xb;
        }
        if let Some(offset) = model.offset {
            for (eta_i, offset_i) in eta_data.iter_mut().zip(offset.iter()) {
                *eta_i += *offset_i;
            }
        }

        let likelihood_started = Instant::now();
        let mut grad_data = vec![0.0_f64; model.y.len()];
        let mut curv_data = vec![0.0_f64; model.y.len()];
        model.likelihood.gradient_and_curvature(
            &mut grad_data,
            &mut curv_data,
            &eta_data,
            model.y,
            theta_lik,
        );
        problem.diagnostics_mut().likelihood_assembly_time += likelihood_started.elapsed();

        let mut w_opt = vec![0.0_f64; n];
        for i in 0..model.y.len() {
            for &(j, _) in &a_rows[i] {
                // Just assign curvature to the involved latent nodes
                w_opt[j] += curv_data[i];
            }
        }
        for w_opt_i in w_opt.iter_mut().take(n) {
            *w_opt_i = (*w_opt_i).max(1e-6);
        }

        let random: Vec<Marginal> = (0..n)
            .map(|i| {
                let mean = posterior_mean[i];
                let sd = variances[i].sqrt().max(1e-10);
                let lo = mean - params.marginal_sds * sd;
                let hi = mean + params.marginal_sds * sd;
                let pts = params.marginal_pts;
                let x: Vec<f64> = (0..pts)
                    .map(|_k| lo + (hi - lo) * _k as f64 / (pts - 1) as f64)
                    .collect();
                let y: Vec<f64> = x
                    .iter()
                    .map(|&xi| {
                        let z = (xi - mean) / sd;
                        (-0.5 * z * z).exp()
                    })
                    .collect();
                Marginal::new(x, y)
            })
            .collect();

        let fitted: Vec<Marginal> = (0..model.y.len())
            .map(|i| {
                let mean = eta_data[i];
                let mut var = 0.0;
                for &(j, ax) in &a_rows[i] {
                    var += ax * ax * variances[j];
                }
                if let Some(fixed_matrix) = model.fixed_matrix {
                    for j1 in 0..k {
                        let x_i_j1 = fixed_matrix[i + j1 * model.y.len()];
                        for j2 in 0..k {
                            var += x_i_j1
                                * mixed_fixed_cov[j1 * k + j2]
                                * fixed_matrix[i + j2 * model.y.len()];
                        }
                    }
                    for &(latent_idx, ax) in &a_rows[i] {
                        for j in 0..k {
                            var += 2.0
                                * ax
                                * mixed_latent_fixed_cov[latent_idx + j * n]
                                * fixed_matrix[i + j * model.y.len()];
                        }
                    }
                }
                let sd = var.sqrt().max(1e-10);
                let lo = mean - params.marginal_sds * sd;
                let hi = mean + params.marginal_sds * sd;
                let pts = params.marginal_pts;
                let x: Vec<f64> = (0..pts)
                    .map(|_k| lo + (hi - lo) * _k as f64 / (pts - 1) as f64)
                    .collect();
                let y: Vec<f64> = x
                    .iter()
                    .map(|&xi| {
                        let z = (xi - mean) / sd;
                        (-0.5 * z * z).exp()
                    })
                    .collect();
                Marginal::new(x, y)
            })
            .collect();

        let mut ccd_thetas = Vec::new();
        let mut ccd_base_weights = Vec::new();
        let mut ccd_weights = Vec::new();
        let mut ccd_log_mlik = Vec::new();
        let mut ccd_log_weight = Vec::new();
        for pt in &ccd_grid.points {
            ccd_thetas.extend_from_slice(&pt.theta);
            ccd_base_weights.push(pt.base_weight);
            ccd_weights.push(pt.weight);
            ccd_log_mlik.push(if pt.theta.is_empty() {
                log_mlik_theta_opt
            } else {
                pt.log_mlik
            });
            ccd_log_weight.push(if pt.theta.is_empty() {
                log_mlik_theta_opt
            } else {
                pt.log_weight
            });
        }
        let diagnostics = problem.diagnostics_summary();

        Ok(InlaResult {
            theta_opt,
            log_mlik: log_mlik_theta_opt,
            log_mlik_theta_opt,
            log_mlik_theta_laplace,
            theta_laplace_correction,
            random,
            fitted,
            n_evals: opt.n_evals,
            fixed_means: mixed_fixed_mean,
            fixed_sds,
            fixed_var_theta_opt,
            ccd_thetas,
            ccd_base_weights,
            ccd_weights,
            ccd_log_mlik,
            ccd_log_weight,
            ccd_hessian_eigenvalues: ccd_grid.hessian_eigenvalues,
            posterior_mean,
            latent_var_theta_opt,
            latent_var_within_theta: mixed_var_inner,
            latent_var_between_theta: inter_var,
            w_opt,
            laplace_terms: opt.laplace_terms,
            mode_x: opt.mode_x,
            mode_beta: opt.mode_beta,
            mode_eta: opt.mode_eta,
            mode_grad: opt.mode_grad,
            mode_curvature_raw: opt.mode_curvature_raw,
            mode_curvature: opt.mode_curvature,
            diagnostics,
        })
    }
}
