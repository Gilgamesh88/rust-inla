use crate::error::InlaError;
use crate::likelihood::LogLikelihood;
use crate::marginal::Marginal;
use crate::models::QFunc;
use crate::optimizer::{self, OptimizerParams};
use crate::problem::Problem;

/// Full model description passed to InlaEngine::run().
///
/// Centralises all model components so internal functions receive a single
/// struct instead of 10+ loose parameters.
pub struct InlaModel<'a> {
    pub qfunc:        &'a dyn QFunc,
    pub likelihood:   &'a dyn LogLikelihood,
    pub y:            &'a [f64],
    pub theta_init:   Vec<f64>,
    pub fixed_matrix: Option<&'a [f64]>,
    pub n_fixed:      usize,
    pub n_latent:     usize,
    /// A-matrix row indices (observation → latent mapping). None = identity.
    pub a_i: Option<&'a [usize]>,
    /// A-matrix column indices (latent node index per observation).
    pub a_j: Option<&'a [usize]>,
    /// A-matrix values.
    pub a_x: Option<&'a [f64]>,
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
    pub fixed_means:    Vec<f64>,
    pub fixed_sds:      Vec<f64>,
    pub ccd_thetas:     Vec<f64>,
    pub ccd_weights:    Vec<f64>,
    pub posterior_mean: Vec<f64>,
    pub w_opt:          Vec<f64>,
}

pub struct InlaEngine;

impl InlaEngine {
    pub fn run(model: &InlaModel<'_>, params: &InlaParams) -> Result<InlaResult, InlaError> {
        let mut problem = Problem::new(model);

        // ── Optimise θ* via L-BFGS ──────────────────────────────────────────
        let opt = optimizer::optimize(
            &mut problem,
            model.qfunc,
            model.likelihood,
            model.y,
            model.a_i, model.a_j, model.a_x,
            model.fixed_matrix,
            model.n_fixed,
            &model.theta_init,
            &params.optimizer,
        )?;

        let theta_opt = opt.theta_opt.clone();
        let n_model   = model.qfunc.n_hyperparams();
        let n         = model.n_latent;
        let k         = model.n_fixed;

        // ── CCD grid over θ ──────────────────────────────────────────────────
        let ccd_grid = crate::optimizer::ccd::build_ccd_grid(
            &mut problem,
            model.qfunc,
            model.likelihood,
            model.y,
            model.a_i, model.a_j, model.a_x,
            model.fixed_matrix,
            k,
            &theta_opt,
        )?;

        // ── Gaussian mixture synthesis (law of total variance) ───────────────
        let mut mixed_mean      = vec![0.0_f64; n];
        let mut mixed_var_inner = vec![0.0_f64; n];
        let mut mixed_mean_sq   = vec![0.0_f64; n];
        let mut mixed_fixed_mean    = vec![0.0_f64; k];
        let mut mixed_fixed_mean_sq = vec![0.0_f64; k];

        for pt in &ccd_grid.points {
            let theta_k   = &pt.theta;
            let weight    = pt.weight;
            let x_warm    = vec![0.0_f64; n];
            let beta_warm = vec![0.0_f64; k];

            let (fixed_k, mean_k, vars_k) = if k > 0 {
                match problem.find_mode_with_fixed_effects(
                    model.qfunc, model.likelihood, model.y,
                    model.a_i, model.a_j, model.a_x,
                    model.fixed_matrix, k, theta_k,
                    &x_warm, &beta_warm, 20, 1e-6,
                ) {
                    Ok((beta, x_hat, _, diag_aug_inv, _)) => {
                        let vs = diag_aug_inv
                            .into_iter()
                            .map(|v| v.max(1e-12))
                            .collect();
                        (beta, x_hat, vs)
                    }
                    Err(_) => (
                        vec![0.0_f64; k],
                        vec![0.0_f64; n],
                        vec![1.0_f64; n],
                    ),
                }
            } else {
                match problem.find_mode_with_inverse(
                    model.qfunc, model.likelihood, model.y,
                    model.a_i, model.a_j, model.a_x,
                    theta_k, &x_warm, 20, 1e-6,
                ) {
                    Ok((x_hat, _, diag_aug_inv)) => {
                        let vs = diag_aug_inv
                            .into_iter()
                            .map(|v| v.max(1e-12))
                            .collect();
                        (vec![], x_hat, vs)
                    }
                    Err(_) => (vec![], vec![0.0_f64; n], vec![1.0_f64; n]),
                }
            };

            for j in 0..k {
                mixed_fixed_mean[j]    += weight * fixed_k[j];
                mixed_fixed_mean_sq[j] += weight * fixed_k[j] * fixed_k[j];
            }
            for i in 0..n {
                mixed_mean[i]      += weight * mean_k[i];
                mixed_mean_sq[i]   += weight * mean_k[i] * mean_k[i];
                mixed_var_inner[i] += weight * vars_k[i];
            }
        }

        // Total variance = intra-θ + inter-θ.
        let mut final_vars = vec![0.0_f64; n];
        for i in 0..n {
            let inter_var = (mixed_mean_sq[i] - mixed_mean[i] * mixed_mean[i]).max(0.0);
            final_vars[i] = mixed_var_inner[i] + inter_var;
        }

        let mut fixed_sds = vec![0.0_f64; k];
        for j in 0..k {
            let iv = (mixed_fixed_mean_sq[j]
                - mixed_fixed_mean[j] * mixed_fixed_mean[j])
                .max(0.0);
            fixed_sds[j] = iv.sqrt();
        }

        let mut posterior_mean = mixed_mean.clone();
        let variances          = final_vars.clone();

        // ── Sum-to-zero identifiability projection ───────────────────────────
        // Project the global mean out of the latent field and absorb it into
        // the intercept (fixed_means[0]), matching R-INLA's constr=TRUE default.
        if k > 0 && n > 0 {
            let mean_x = posterior_mean.iter().sum::<f64>() / n as f64;
            for i in 0..n {
                posterior_mean[i] -= mean_x;
            }
            mixed_fixed_mean[0] += mean_x;
        }

        // ── Build η at posterior mean for IRLS weight diagnostics ────────────
        let theta_lik   = &theta_opt[n_model..];
        let mut eta_data = vec![0.0_f64; model.y.len()];
        for i in 0..model.y.len() {
            // FIX: was `let lat_idx = model.a_i, model.a_j, model.a_x.map_or(...)` — invalid syntax.
            let lat_idx = model.a_j.map_or(i, |aj| aj[i]);
            let mut xb = 0.0;
            if k > 0 {
                if let Some(xm) = model.fixed_matrix {
                    for j in 0..k {
                        xb += xm[i + j * model.y.len()] * mixed_fixed_mean[j];
                    }
                }
            }
            eta_data[i] = posterior_mean[lat_idx] + xb;
        }

        let mut grad_data = vec![0.0_f64; model.y.len()];
        let mut curv_data = vec![0.0_f64; model.y.len()];
        model.likelihood.gradient_and_curvature(
            &mut grad_data, &mut curv_data, &eta_data, model.y, theta_lik,
        );

        let mut w_opt = vec![0.0_f64; n];
        for i in 0..model.y.len() {
            // FIX: same invalid syntax as above.
            let lat_idx = model.a_j.map_or(i, |aj| aj[i]);
            w_opt[lat_idx] += curv_data[i];
        }
        for i in 0..n {
            w_opt[i] = w_opt[i].max(1e-6);
        }

        // ── Gaussian marginals for each latent node ──────────────────────────
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

        // ── Pack CCD diagnostics ─────────────────────────────────────────────
        let mut ccd_thetas  = Vec::new();
        let mut ccd_weights = Vec::new();
        for pt in &ccd_grid.points {
            ccd_thetas.extend_from_slice(&pt.theta);
            ccd_weights.push(pt.weight);
        }

        Ok(InlaResult {
            theta_opt,
            log_mlik: opt.log_mlik,
            random,
            n_evals: opt.n_evals,
            fixed_means: mixed_fixed_mean,
            fixed_sds,
            ccd_thetas,
            ccd_weights,
            posterior_mean,
            w_opt,
        })
    }
}
