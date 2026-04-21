use std::time::Instant;

use crate::diagnostics::LaplacePhase;
use crate::error::InlaError;
use crate::inference::InlaModel;
use crate::problem::Problem;

pub mod ccd;

#[derive(Clone, Debug, Default)]
pub struct LaplaceDecomposition {
    pub sum_loglik: f64,
    pub log_prior_model: f64,
    pub log_prior_likelihood: f64,
    pub log_prior: f64,
    pub latent_log_det_q: f64,
    pub latent_log_det_aug: f64,
    pub fixed_log_det_penalty: f64,
    pub schur_complement_adjustment: f64,
    pub final_log_det_q: f64,
    pub final_log_det_aug: f64,
    pub latent_q_form: f64,
    pub fixed_q_form: f64,
    pub final_q_form: f64,
    pub log_mlik: f64,
    pub neg_log_mlik: f64,
}

#[derive(Clone, Debug)]
pub(crate) struct LaplaceEvalResult {
    pub neg_log_mlik: f64,
    pub x_hat: Vec<f64>,
    pub beta_hat: Vec<f64>,
    pub eta_hat: Vec<f64>,
    pub decomposition: LaplaceDecomposition,
}

pub struct OptimResult {
    pub theta_opt: Vec<f64>,
    pub log_mlik: f64,
    pub n_evals: usize,
    pub laplace_terms: LaplaceDecomposition,
    pub mode_x: Vec<f64>,
    pub mode_beta: Vec<f64>,
    pub mode_eta: Vec<f64>,
    pub mode_grad: Vec<f64>,
    pub mode_curvature_raw: Vec<f64>,
    pub mode_curvature: Vec<f64>,
}

#[derive(Clone, Copy)]
pub(crate) struct LaplaceEvalConfig {
    pub phase: LaplacePhase,
    pub n_irls: usize,
    pub tol_irls: f64,
}

const OPTIMIZE_IRLS_MAX_ITER: usize = 20;
const OPTIMIZE_IRLS_TOL: f64 = 1e-6;
const FINAL_IRLS_MAX_ITER: usize = 60;
const FINAL_IRLS_TOL: f64 = 1e-8;

pub struct OptimizerParams {
    pub tol_grad: f64,
    pub max_evals: usize,
    pub finite_diff_step: f64,
    pub lbfgs_memory: usize,
}

impl Default for OptimizerParams {
    fn default() -> Self {
        Self {
            tol_grad: 1e-4,
            max_evals: 50,
            // Hyperparameter profiles are fairly flat on the INLA scale.
            // A larger finite-difference step is needed so the outer gradient
            // is not drowned out by inner IRLS approximation error.
            finite_diff_step: 1e-2,
            lbfgs_memory: 5,
        }
    }
}

fn build_linear_predictor(
    model: &InlaModel<'_>,
    x: &[f64],
    beta: &[f64],
) -> Result<Vec<f64>, InlaError> {
    let n_data = model.y.len();
    let mut eta = vec![0.0_f64; n_data];

    if model.n_fixed > 0 {
        let fixed_matrix = model.fixed_matrix.ok_or(InlaError::DimensionMismatch {
            expected: n_data * model.n_fixed,
            got: 0,
        })?;
        for (j, beta_j) in beta.iter().enumerate().take(model.n_fixed) {
            for (i, eta_i) in eta.iter_mut().enumerate().take(n_data) {
                *eta_i += fixed_matrix[i + j * n_data] * *beta_j;
            }
        }
    }

    if let (Some(a_i), Some(a_j), Some(a_x)) = (model.a_i, model.a_j, model.a_x) {
        for ((&row, &col), &weight) in a_i.iter().zip(a_j.iter()).zip(a_x.iter()) {
            eta[row] += weight * x[col];
        }
    } else {
        for (i, eta_i) in eta.iter_mut().enumerate().take(n_data.min(x.len())) {
            *eta_i += x[i];
        }
    }

    if let Some(offset) = model.offset {
        for (eta_i, offset_i) in eta.iter_mut().zip(offset.iter()) {
            *eta_i += *offset_i;
        }
    }

    Ok(eta)
}

fn compensated_sum<I>(iter: I) -> f64
where
    I: IntoIterator<Item = f64>,
{
    let mut sum = 0.0_f64;
    let mut c = 0.0_f64;
    for x in iter {
        let y = x - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    sum
}

fn gaussian_integrated_data_term(
    model: &InlaModel<'_>,
    theta_lik: &[f64],
    eta_with_offset: &[f64],
) -> Option<f64> {
    let tau_obs = model.likelihood.gaussian_observation_precision(theta_lik)?;
    let log_norm = 0.5 * ((tau_obs).ln() - std::f64::consts::TAU.ln());
    let n_obs = model.y.iter().filter(|y_i| !y_i.is_nan()).count();
    let centered_response_ss = compensated_sum((0..model.y.len()).filter_map(|i| {
        let y_i = model.y[i];
        if y_i.is_nan() {
            return None;
        }
        let offset_i = model.offset.map_or(0.0_f64, |offset| offset[i]);
        let centered_response = y_i - offset_i;
        Some(centered_response * centered_response)
    }));
    let rhs_mode_quad = compensated_sum((0..model.y.len()).filter_map(|i| {
        let y_i = model.y[i];
        if y_i.is_nan() {
            return None;
        }
        let offset_i = model.offset.map_or(0.0_f64, |offset| offset[i]);
        let centered_response = y_i - offset_i;
        let centered_mode = eta_with_offset[i] - offset_i;
        Some(centered_response * centered_mode)
    }));

    Some(
        (n_obs as f64) * log_norm - 0.5 * tau_obs * centered_response_ss
            + 0.5 * tau_obs * rhs_mode_quad,
    )
}

pub(crate) fn laplace_eval(
    problem: &mut Problem,
    model: &InlaModel<'_>,
    theta: &[f64],
    x_warm: &[f64],
    beta_warm: &[f64],
    config: LaplaceEvalConfig,
) -> Result<LaplaceEvalResult, InlaError> {
    problem.record_laplace_eval(config.phase);
    let n_model = model.qfunc.n_hyperparams();
    let theta_model = &theta[..n_model];
    let theta_lik = &theta[n_model..];

    let (x_hat, log_det_aug, eta_for_lik, beta_out, schur_s) = if model.n_fixed > 0 {
        let (beta, x, ld, s) = problem.find_mode_with_fixed_effects_logdet(
            model,
            theta,
            x_warm,
            beta_warm,
            config.n_irls,
            config.tol_irls,
        )?;
        let eta = build_linear_predictor(model, &x, &beta)?;
        (x, ld, eta, beta, s)
    } else {
        let (x, ld) = problem.find_mode_with_logdet_and_warm(
            model,
            theta,
            x_warm,
            config.n_irls,
            config.tol_irls,
        )?;
        let eta = build_linear_predictor(model, &x, &[])?;
        (x, ld, eta, vec![], 0.0_f64)
    };

    let log_det_q = if model.qfunc.is_proper() {
        problem.eval(model.qfunc, theta_model)?
    } else {
        model.qfunc.log_det_term(theta_model)
    };

    let n = model.y.len();
    let mut logll = vec![0.0_f64; n];
    model
        .likelihood
        .evaluate(&mut logll, &eta_for_lik, model.y, theta_lik);
    let sum_logll: f64 = logll.iter().sum();

    let log_prior_model = model.qfunc.log_prior(theta_model);
    let log_prior_likelihood = model.likelihood.log_prior(theta_lik);
    let log_prior = log_prior_model + log_prior_likelihood;
    let latent_q_form = problem.quadratic_form_x(model.qfunc, theta_model, &x_hat);

    let (fixed_log_det_penalty, schur_complement_adjustment, fixed_q_form) = if model.n_fixed > 0 {
        let penalty = crate::problem::PRIOR_PREC_BETA.ln() * (model.n_fixed as f64);
        let beta_qf: f64 = beta_out.iter().map(|b| b * b).sum();
        (penalty, schur_s, crate::problem::PRIOR_PREC_BETA * beta_qf)
    } else {
        (0.0_f64, 0.0_f64, 0.0_f64)
    };

    let final_log_det_q = log_det_q + fixed_log_det_penalty;
    let final_log_det_aug = log_det_aug + schur_complement_adjustment;
    let final_q_form = latent_q_form + fixed_q_form;

    let log_mlik =
        if let Some(data_term) = gaussian_integrated_data_term(model, theta_lik, &eta_for_lik) {
            0.5 * (final_log_det_q - final_log_det_aug) + data_term + log_prior
        } else {
            0.5 * (final_log_det_q - final_log_det_aug) + sum_logll - 0.5 * final_q_form + log_prior
        };

    let decomposition = LaplaceDecomposition {
        sum_loglik: sum_logll,
        log_prior_model,
        log_prior_likelihood,
        log_prior,
        latent_log_det_q: log_det_q,
        latent_log_det_aug: log_det_aug,
        fixed_log_det_penalty,
        schur_complement_adjustment,
        final_log_det_q,
        final_log_det_aug,
        latent_q_form,
        fixed_q_form,
        final_q_form,
        log_mlik,
        neg_log_mlik: -log_mlik,
    };

    Ok(LaplaceEvalResult {
        neg_log_mlik: -log_mlik,
        x_hat,
        beta_hat: beta_out,
        eta_hat: eta_for_lik,
        decomposition,
    })
}

fn laplace_gradient(
    problem: &mut Problem,
    model: &InlaModel<'_>,
    theta: &[f64],
    f0: f64,
    x_hat: &[f64],
    beta_warm: &[f64],
    h: f64,
) -> Result<Vec<f64>, InlaError> {
    let n_theta = model.qfunc.n_hyperparams() + model.likelihood.n_hyperparams();
    let mut grad = vec![0.0_f64; n_theta];
    for k in 0..n_theta {
        let mut theta_h = theta.to_vec();
        theta_h[k] += h;
        let fh = laplace_eval(
            problem,
            model,
            &theta_h,
            x_hat,
            beta_warm,
            LaplaceEvalConfig {
                phase: LaplacePhase::Optimize,
                n_irls: OPTIMIZE_IRLS_MAX_ITER,
                tol_irls: OPTIMIZE_IRLS_TOL,
            },
        )?
        .neg_log_mlik;
        grad[k] = (fh - f0) / h;
    }
    Ok(grad)
}

fn laplace_eval_trial(
    problem: &mut Problem,
    model: &InlaModel<'_>,
    theta: &[f64],
    x_warm: &[f64],
    beta_warm: &[f64],
    cold_fallback_threshold: Option<f64>,
) -> Result<ProbeAccept, InlaError> {
    let warm_eval = laplace_eval(
        problem,
        model,
        theta,
        x_warm,
        beta_warm,
        LaplaceEvalConfig {
            phase: LaplacePhase::Optimize,
            n_irls: OPTIMIZE_IRLS_MAX_ITER,
            tol_irls: OPTIMIZE_IRLS_TOL,
        },
    );

    if let Ok(warm_result) = &warm_eval {
        let needs_cold_fallback = cold_fallback_threshold
            .map(|threshold| warm_result.neg_log_mlik > threshold)
            .unwrap_or(false);
        if !needs_cold_fallback {
            return Ok((
                theta.to_vec(),
                warm_result.neg_log_mlik,
                warm_result.x_hat.clone(),
                warm_result.beta_hat.clone(),
            ));
        }
    }

    let cold_x = vec![0.0_f64; model.n_latent];
    let cold_beta = vec![0.0_f64; model.n_fixed];
    let cold_eval = laplace_eval(
        problem,
        model,
        theta,
        &cold_x,
        &cold_beta,
        LaplaceEvalConfig {
            phase: LaplacePhase::Optimize,
            n_irls: OPTIMIZE_IRLS_MAX_ITER,
            tol_irls: OPTIMIZE_IRLS_TOL,
        },
    );

    match (warm_eval, cold_eval) {
        (Ok(warm_result), Ok(cold_result)) => {
            if warm_result.neg_log_mlik <= cold_result.neg_log_mlik {
                Ok((
                    theta.to_vec(),
                    warm_result.neg_log_mlik,
                    warm_result.x_hat,
                    warm_result.beta_hat,
                ))
            } else {
                Ok((
                    theta.to_vec(),
                    cold_result.neg_log_mlik,
                    cold_result.x_hat,
                    cold_result.beta_hat,
                ))
            }
        }
        (Ok(warm_result), Err(_)) => Ok((
            theta.to_vec(),
            warm_result.neg_log_mlik,
            warm_result.x_hat,
            warm_result.beta_hat,
        )),
        (Err(_), Ok(cold_result)) => Ok((
            theta.to_vec(),
            cold_result.neg_log_mlik,
            cold_result.x_hat,
            cold_result.beta_hat,
        )),
        (Err(err), Err(_)) => Err(err),
    }
}

fn lbfgs_direction(grad: &[f64], s_list: &[Vec<f64>], y_list: &[Vec<f64>]) -> Vec<f64> {
    let m = s_list.len();
    let n = grad.len();
    let mut q = grad.to_vec();
    let mut alpha = vec![0.0_f64; m];

    for i in (0..m).rev() {
        let sy: f64 = s_list[i]
            .iter()
            .zip(y_list[i].iter())
            .map(|(s, y)| s * y)
            .sum();
        if sy.abs() < 1e-14 {
            continue;
        }
        let rho = 1.0 / sy;
        let sq: f64 = s_list[i].iter().zip(q.iter()).map(|(s, qi)| s * qi).sum();
        alpha[i] = rho * sq;
        for j in 0..n {
            q[j] -= alpha[i] * y_list[i][j];
        }
    }

    let gamma = if m > 0 {
        let sy: f64 = s_list[m - 1]
            .iter()
            .zip(y_list[m - 1].iter())
            .map(|(s, y)| s * y)
            .sum();
        let yy: f64 = y_list[m - 1].iter().map(|y| y * y).sum();
        if yy > 1e-14 {
            sy / yy
        } else {
            1.0
        }
    } else {
        1.0
    };

    let mut r: Vec<f64> = q.iter().map(|qi| gamma * qi).collect();

    for i in 0..m {
        let sy: f64 = s_list[i]
            .iter()
            .zip(y_list[i].iter())
            .map(|(s, y)| s * y)
            .sum();
        if sy.abs() < 1e-14 {
            continue;
        }
        let rho = 1.0 / sy;
        let yr: f64 = y_list[i].iter().zip(r.iter()).map(|(y, ri)| y * ri).sum();
        let beta = rho * yr;
        for j in 0..n {
            r[j] += s_list[i][j] * (alpha[i] - beta);
        }
    }

    r.iter().map(|ri| -ri).collect()
}

type ProbeAccept = (Vec<f64>, f64, Vec<f64>, Vec<f64>);

fn coordinate_probe(
    problem: &mut Problem,
    model: &InlaModel<'_>,
    theta: &[f64],
    f_cur: f64,
    x_warm: &[f64],
    beta_warm: &[f64],
    h: f64,
) -> Result<(usize, Option<ProbeAccept>), InlaError> {
    let probe_scales = [100.0_f64, 50.0_f64, 25.0_f64];
    let mut n_evals = 0usize;
    let mut best_f = f_cur;
    let mut best: Option<ProbeAccept> = None;
    problem.diagnostics_mut().coordinate_probe_calls += 1;

    for k in 0..theta.len() {
        for scale in probe_scales {
            let delta = h * scale;
            for sign in [-1.0_f64, 1.0_f64] {
                let mut theta_try = theta.to_vec();
                theta_try[k] += sign * delta;
                n_evals += 1;
                problem.diagnostics_mut().coordinate_probe_evals += 1;
                let improve_threshold = best_f - 1e-8;
                if let Ok((_, f_try, x_try, beta_try)) = laplace_eval_trial(
                    problem,
                    model,
                    &theta_try,
                    x_warm,
                    beta_warm,
                    Some(improve_threshold),
                ) {
                    if f_try + 1e-8 < best_f {
                        best_f = f_try;
                        best = Some((theta_try, f_try, x_try, beta_try));
                    }
                }
            }
        }
    }

    Ok((n_evals, best))
}

pub fn optimize(
    problem: &mut Problem,
    model: &InlaModel<'_>,
    theta_init: &[f64],
    params: &OptimizerParams,
) -> Result<OptimResult, InlaError> {
    let optimize_started = Instant::now();
    let n_model = model.qfunc.n_hyperparams();
    let n_lik = model.likelihood.n_hyperparams();
    let h = params.finite_diff_step;
    let max_iter = params.max_evals;
    let tol_grad = params.tol_grad;
    let m = params.lbfgs_memory;

    let mut theta = theta_init.to_vec();
    let mut n_evals = 0_usize;
    let mut x_warm = if model.latent_init.len() == model.n_latent {
        model.latent_init.clone()
    } else {
        vec![0.0_f64; model.n_latent]
    };
    let mut beta_warm = if model.fixed_init.len() == model.n_fixed {
        model.fixed_init.clone()
    } else {
        vec![0.0_f64; model.n_fixed]
    };

    let mut s_list: Vec<Vec<f64>> = Vec::with_capacity(m);
    let mut y_list: Vec<Vec<f64>> = Vec::with_capacity(m);

    let initial_eval = laplace_eval(
        problem,
        model,
        &theta,
        &x_warm,
        &beta_warm,
        LaplaceEvalConfig {
            phase: LaplacePhase::Optimize,
            n_irls: OPTIMIZE_IRLS_MAX_ITER,
            tol_irls: OPTIMIZE_IRLS_TOL,
        },
    )?;
    let mut f_cur = initial_eval.neg_log_mlik;
    x_warm = initial_eval.x_hat;
    beta_warm = initial_eval.beta_hat;
    n_evals += 1;

    let mut grad = laplace_gradient(problem, model, &theta, f_cur, &x_warm, &beta_warm, h)?;
    n_evals += n_model + n_lik;

    for _iter in 0..max_iter {
        problem.diagnostics_mut().optimizer_outer_iterations += 1;
        let grad_norm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        if grad_norm < tol_grad {
            let (probe_evals, probe_best) =
                coordinate_probe(problem, model, &theta, f_cur, &x_warm, &beta_warm, h)?;
            n_evals += probe_evals;
            if let Some((theta_new, f_new, x_new, beta_new)) = probe_best {
                problem.diagnostics_mut().coordinate_probe_accepts += 1;
                theta = theta_new;
                f_cur = f_new;
                x_warm = x_new;
                beta_warm = beta_new;
                s_list.clear();
                y_list.clear();
                grad = laplace_gradient(problem, model, &theta, f_cur, &x_warm, &beta_warm, h)?;
                n_evals += n_model + n_lik;
                continue;
            }
            break;
        }

        let direction = lbfgs_direction(&grad, &s_list, &y_list);

        let c1 = 1e-4_f64;
        let grad_d: f64 = grad.iter().zip(direction.iter()).map(|(g, d)| g * d).sum();
        let mut alpha = 1.0_f64;
        let mut accepted = false;

        for _ in 0..20 {
            problem.diagnostics_mut().line_search_trial_evals += 1;
            let theta_new: Vec<f64> = theta
                .iter()
                .zip(direction.iter())
                .map(|(&t, &d)| t + alpha * d)
                .collect();
            let accept_threshold = f_cur + c1 * alpha * grad_d;

            match laplace_eval_trial(
                problem,
                model,
                &theta_new,
                &x_warm,
                &beta_warm,
                Some(accept_threshold),
            ) {
                Ok((_, f_new, x_new, beta_new)) if f_new <= accept_threshold => {
                    let s_k: Vec<f64> = theta_new
                        .iter()
                        .zip(theta.iter())
                        .map(|(tn, t)| tn - t)
                        .collect();

                    let grad_new =
                        laplace_gradient(problem, model, &theta_new, f_new, &x_new, &beta_new, h)?;
                    n_evals += n_model + n_lik;

                    let y_k: Vec<f64> = grad_new
                        .iter()
                        .zip(grad.iter())
                        .map(|(gn, g)| gn - g)
                        .collect();

                    let sy: f64 = s_k.iter().zip(y_k.iter()).map(|(s, y)| s * y).sum();
                    if sy > 1e-10 {
                        if s_list.len() == m {
                            s_list.remove(0);
                            y_list.remove(0);
                        }
                        s_list.push(s_k);
                        y_list.push(y_k);
                    }

                    theta = theta_new;
                    f_cur = f_new;
                    x_warm = x_new;
                    beta_warm = beta_new;
                    grad = grad_new;
                    n_evals += 1;
                    problem.diagnostics_mut().line_search_trial_accepts += 1;
                    accepted = true;
                    break;
                }
                _ => {
                    alpha *= 0.5;
                    n_evals += 1;
                }
            }
        }

        if !accepted {
            let (probe_evals, probe_best) =
                coordinate_probe(problem, model, &theta, f_cur, &x_warm, &beta_warm, h)?;
            n_evals += probe_evals;
            if let Some((theta_new, f_new, x_new, beta_new)) = probe_best {
                problem.diagnostics_mut().coordinate_probe_accepts += 1;
                theta = theta_new;
                f_cur = f_new;
                x_warm = x_new;
                beta_warm = beta_new;
                s_list.clear();
                y_list.clear();
                grad = laplace_gradient(problem, model, &theta, f_cur, &x_warm, &beta_warm, h)?;
                n_evals += n_model + n_lik;
                continue;
            }
            break;
        }
    }

    let final_eval = laplace_eval(
        problem,
        model,
        &theta,
        &x_warm,
        &beta_warm,
        LaplaceEvalConfig {
            phase: LaplacePhase::Optimize,
            n_irls: FINAL_IRLS_MAX_ITER,
            tol_irls: FINAL_IRLS_TOL,
        },
    )
    .map_err(|e| InlaError::ConvergenceFailed {
        reason: format!("IRLS no convergiÃ³ en theta*. Cause: {:?}", e),
    })?;

    let log_mlik = -final_eval.neg_log_mlik;
    let theta_lik = &theta[n_model..];
    let mut mode_grad = vec![0.0_f64; model.y.len()];
    let mut mode_curvature_raw = vec![0.0_f64; model.y.len()];
    model.likelihood.gradient_and_curvature(
        &mut mode_grad,
        &mut mode_curvature_raw,
        &final_eval.eta_hat,
        model.y,
        theta_lik,
    );
    let mut mode_curvature = mode_curvature_raw.clone();
    for curv_i in &mut mode_curvature {
        *curv_i = (*curv_i).max(1e-6);
    }
    problem.diagnostics_mut().optimizer_time += optimize_started.elapsed();

    Ok(OptimResult {
        theta_opt: theta,
        log_mlik,
        n_evals,
        laplace_terms: final_eval.decomposition,
        mode_x: final_eval.x_hat,
        mode_beta: final_eval.beta_hat,
        mode_eta: final_eval.eta_hat,
        mode_grad,
        mode_curvature_raw,
        mode_curvature,
    })
}

#[cfg(test)]
mod tests {
    use super::{build_linear_predictor, laplace_eval, LaplaceEvalConfig};
    use crate::inference::InlaModel;
    use crate::likelihood::{GaussianLikelihood, LogLikelihood, PoissonLikelihood};
    use crate::models::{IidModel, QFunc};
    use crate::problem::{Problem, PRIOR_PREC_BETA};
    use approx::assert_abs_diff_eq;

    fn dense_cholesky_solve(mut a: Vec<f64>, mut b: Vec<f64>, n: usize) -> (Vec<f64>, f64) {
        let mut log_det = 0.0_f64;
        for i in 0..n {
            for j in 0..=i {
                let mut sum = a[i * n + j];
                for k in 0..j {
                    sum -= a[i * n + k] * a[j * n + k];
                }
                if i == j {
                    assert!(sum > 1e-12, "dense test matrix must remain SPD");
                    let l_ii = sum.sqrt();
                    a[i * n + j] = l_ii;
                    log_det += 2.0 * l_ii.ln();
                } else {
                    a[i * n + j] = sum / a[j * n + j];
                }
            }
        }

        for i in 0..n {
            let mut sum = b[i];
            for k in 0..i {
                sum -= a[i * n + k] * b[k];
            }
            b[i] = sum / a[i * n + i];
        }
        for i in (0..n).rev() {
            let mut sum = b[i];
            for k in (i + 1)..n {
                sum -= a[k * n + i] * b[k];
            }
            b[i] = sum / a[i * n + i];
        }

        (b, log_det)
    }

    fn dense_q_matrix(qfunc: &dyn QFunc, theta_model: &[f64], n: usize) -> Vec<f64> {
        let mut q = vec![0.0_f64; n * n];
        for i in 0..n {
            q[i * n + i] = qfunc.eval(i, i, theta_model);
            for j in (i + 1)..n {
                if qfunc.graph().are_neighbors(i, j) {
                    let val = qfunc.eval(i, j, theta_model);
                    q[i * n + j] = val;
                    q[j * n + i] = val;
                }
            }
        }
        q
    }

    #[test]
    fn linear_predictor_includes_fixed_random_and_offset_terms() {
        let y = vec![0.0, 1.0, 2.0];
        let fixed_matrix = vec![1.0, 1.0, 1.0];
        let x = vec![0.5, -0.25];
        let beta = vec![2.0];
        let offset = vec![0.1, 0.2, 0.3];
        let a_i = vec![0, 1, 2];
        let a_j = vec![0, 1, 0];
        let a_x = vec![1.0, 1.0, 1.0];

        let qfunc = IidModel::new(2);
        let likelihood = PoissonLikelihood;
        let model = InlaModel {
            qfunc: &qfunc,
            likelihood: &likelihood,
            y: &y,
            theta_init: vec![0.0],
            latent_init: vec![],
            fixed_init: vec![],
            fixed_matrix: Some(&fixed_matrix),
            n_fixed: 1,
            n_latent: 2,
            a_i: Some(&a_i),
            a_j: Some(&a_j),
            a_x: Some(&a_x),
            offset: Some(&offset),
            extr_constr: None,
            n_constr: 0,
        };

        let eta = build_linear_predictor(&model, &x, &beta).unwrap();

        assert_eq!(eta.len(), 3);
        assert_abs_diff_eq!(eta[0], 2.6, epsilon = 1e-12);
        assert_abs_diff_eq!(eta[1], 1.95, epsilon = 1e-12);
        assert_abs_diff_eq!(eta[2], 2.8, epsilon = 1e-12);
    }

    #[test]
    fn gaussian_laplace_eval_matches_dense_integrated_marginal_with_fixed_effects() {
        let y = vec![1.2, f64::NAN, -0.4, 0.9];
        let offset = vec![0.1, -0.2, 0.05, -0.1];
        let fixed_matrix = vec![1.0, 1.0, 1.0, 1.0];
        let a_i = vec![0, 1, 2, 3];
        let a_j = vec![0, 1, 0, 1];
        let a_x = vec![1.0, 1.0, 1.0, 1.0];
        let theta = vec![0.4, 1.1];

        let qfunc = IidModel::new(2);
        let likelihood = GaussianLikelihood;
        let model = InlaModel {
            qfunc: &qfunc,
            likelihood: &likelihood,
            y: &y,
            theta_init: theta.clone(),
            latent_init: vec![],
            fixed_init: vec![],
            fixed_matrix: Some(&fixed_matrix),
            n_fixed: 1,
            n_latent: 2,
            a_i: Some(&a_i),
            a_j: Some(&a_j),
            a_x: Some(&a_x),
            offset: Some(&offset),
            extr_constr: None,
            n_constr: 0,
        };

        let mut problem = Problem::new(&model);
        let eval = laplace_eval(
            &mut problem,
            &model,
            &theta,
            &[0.0; 2],
            &[0.0; 1],
            LaplaceEvalConfig {
                phase: crate::diagnostics::LaplacePhase::Optimize,
                n_irls: 10,
                tol_irls: 1e-10,
            },
        )
        .unwrap();

        let tau_obs = theta[1].exp();
        let centered_y = [y[0] - offset[0], y[2] - offset[2], y[3] - offset[3]];

        let q_prior = dense_q_matrix(&qfunc, &theta[..1], 2);
        let log_det_prior = 2.0 * theta[0] + PRIOR_PREC_BETA.ln();

        let mut q_post = vec![
            q_prior[0],
            q_prior[1],
            0.0,
            q_prior[2],
            q_prior[3],
            0.0,
            0.0,
            0.0,
            PRIOR_PREC_BETA,
        ];
        let design_rows = [[1.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]];
        let mut rhs = vec![0.0_f64; 3];
        for (row, &centered) in design_rows.iter().zip(centered_y.iter()) {
            for i in 0..3 {
                rhs[i] += tau_obs * row[i] * centered;
                for j in 0..3 {
                    q_post[i * 3 + j] += tau_obs * row[i] * row[j];
                }
            }
        }

        let (mode_joint, log_det_post) = dense_cholesky_solve(q_post, rhs.clone(), 3);
        let b_mode_quad: f64 = rhs
            .iter()
            .zip(mode_joint.iter())
            .map(|(b_i, z_i)| b_i * z_i)
            .sum();
        let n_obs = centered_y.len() as f64;
        let response_ss: f64 = centered_y.iter().map(|value| value * value).sum();
        let expected_log_mlik = 0.5 * (log_det_prior - log_det_post)
            + 0.5 * n_obs * (theta[1] - std::f64::consts::TAU.ln())
            - 0.5 * tau_obs * response_ss
            + 0.5 * b_mode_quad
            + qfunc.log_prior(&theta[..1])
            + likelihood.log_prior(&theta[1..]);

        assert_abs_diff_eq!(-eval.neg_log_mlik, expected_log_mlik, epsilon = 1e-6);
    }
}
