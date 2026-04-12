use argmin::core::{CostFunction, Error as ArgminError};
use std::cell::RefCell;

use crate::error::InlaError;
pub mod ccd;
use crate::likelihood::LogLikelihood;
use crate::models::QFunc;
use crate::problem::Problem;

pub struct OptimResult {
    pub theta_opt: Vec<f64>,
    pub log_mlik:  f64,
    pub n_evals:   usize,
}

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
        let n_model     = self.qfunc.n_hyperparams();
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
    pub lbfgs_memory:     usize,
}

impl Default for OptimizerParams {
    fn default() -> Self {
        Self { tol_grad: 1e-4, max_evals: 50, finite_diff_step: 1e-4, lbfgs_memory: 5 }
    }
}

pub(crate) fn laplace_eval(
    problem:    &mut Problem,
    model:      &InlaModel<'_>,
    theta:      &[f64],
    x_warm:     &[f64],
    beta_warm:  &[f64],
    n_model:    usize,
    n_irls:     usize,
    tol_irls:   f64,
) -> Result<(f64, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>), InlaError> {
    let theta_model = &theta[..n_model];
    let theta_lik   = &theta[n_model..];

    let (x_hat, log_det_aug, diag_aug_inv, eta_for_lik, beta_out, schur_s) = if model.n_fixed > 0 {
        let (beta, x, ld, d, s) = problem.find_mode_with_fixed_effects(
            model, theta, x_warm, beta_warm, n_irls, tol_irls,
        )?;
        let mut eta = vec![0.0_f64; model.y.len()];
        for i in 0..model.y.len() {
            let mut xb = 0.0;
            for j in 0..model.n_fixed { xb += model.fixed_matrix.unwrap()[i + j * model.y.len()] * beta[j]; }
            eta[i] = xb;
        }
        if let (Some(a_i), Some(a_j), Some(a_x)) = (model.a_i, model.a_j, model.a_x) {
            for k in 0..a_i.len() {
                eta[a_i[k]] += a_x[k] * x[a_j[k]];
            }
        }
        (x, ld, d, eta, beta, s)
    } else {
        let (x, ld, d) = problem.find_mode_with_inverse(
            model, theta, x_warm, n_irls, tol_irls,
        )?;
        let mut eta = vec![0.0_f64; model.y.len()];
        if let (Some(a_i), Some(a_j), Some(a_x)) = (model.a_i, model.a_j, model.a_x) {
            for k in 0..a_i.len() {
                eta[a_i[k]] += a_x[k] * x[a_j[k]];
            }
        }
        (x, ld, d, eta, vec![], 0.0_f64)
    };

    let (log_det_q, diag_q_inv) = if model.qfunc.is_proper() {
        problem.eval_with_inverse(model.qfunc, theta_model)?
    } else {
        (0.0_f64, vec![0.0; model.y.len()])
    };

    let n = model.y.len();
    let mut logll = vec![0.0_f64; n];
    model.likelihood.evaluate(&mut logll, &eta_for_lik, model.y, theta_lik);
    let sum_logll: f64 = logll.iter().sum();

    let log_prior: f64 = theta.iter().map(|&th| th - 5e-5_f64 * th.exp()).sum();
    let q_form = problem.quadratic_form_x(model.qfunc, theta_model, &x_hat);

    let (final_log_det_q, final_log_det_aug, final_q_form) = if model.n_fixed > 0 {
        let penalty = crate::problem::PRIOR_PREC_BETA.ln() * (model.n_fixed as f64);
        let beta_qf: f64 = beta_out.iter().map(|b| b * b).sum();
        (
            log_det_q + penalty,
            log_det_aug + schur_s,
            q_form + crate::problem::PRIOR_PREC_BETA * beta_qf
        )
    } else {
        (log_det_q, log_det_aug, q_form)
    };

    let log_mlik = 0.5 * (final_log_det_q - final_log_det_aug) + sum_logll - 0.5 * final_q_form + log_prior;
    
    Ok((-log_mlik, x_hat, beta_out, diag_q_inv, diag_aug_inv))
}

fn laplace_gradient(
    problem:    &mut Problem,
    model:      &InlaModel<'_>,
    theta:      &[f64],
    f0:         f64,
    x_hat:      &[f64],
    beta_warm:  &[f64],
    n_model:    usize,
    n_lik:      usize,
    h:          f64,
) -> Vec<f64> {
    let n_theta = n_model + n_lik;
    let mut grad = vec![0.0_f64; n_theta];
    for k in 0..n_theta {
        let mut theta_h = theta.to_vec();
        theta_h[k] += h;
        let fh = laplace_eval(
            problem, qfunc, likelihood, y, a_i, a_j, a_x, x_mat, n_fixed, &theta_h,
            x_hat, beta_warm, n_model, 5, 1e-3,
        ).map(|(f, ..)| f).unwrap_or(f64::MAX / 2.0);
        grad[k] = (fh - f0) / h;
    }
    grad
}

fn lbfgs_direction(
    grad: &[f64],
    s_list: &[Vec<f64>],
    y_list: &[Vec<f64>],
) -> Vec<f64> {
    let m = s_list.len();
    let n = grad.len();
    let mut q = grad.to_vec();
    let mut alpha = vec![0.0_f64; m];

    for i in (0..m).rev() {
        let sy: f64 = s_list[i].iter().zip(y_list[i].iter()).map(|(s,y)| s*y).sum();
        if sy.abs() < 1e-14 { continue; }
        let rho = 1.0 / sy;
        let sq: f64 = s_list[i].iter().zip(q.iter()).map(|(s,qi)| s*qi).sum();
        alpha[i] = rho * sq;
        for j in 0..n { q[j] -= alpha[i] * y_list[i][j]; }
    }

    let gamma = if m > 0 {
        let sy: f64 = s_list[m-1].iter().zip(y_list[m-1].iter()).map(|(s,y)| s*y).sum();
        let yy: f64 = y_list[m-1].iter().map(|y| y*y).sum();
        if yy > 1e-14 { sy / yy } else { 1.0 }
    } else { 1.0 };
    
    let mut r: Vec<f64> = q.iter().map(|qi| gamma * qi).collect();

    for i in 0..m {
        let sy: f64 = s_list[i].iter().zip(y_list[i].iter()).map(|(s,y)| s*y).sum();
        if sy.abs() < 1e-14 { continue; }
        let rho = 1.0 / sy;
        let yr: f64 = y_list[i].iter().zip(r.iter()).map(|(y,ri)| y*ri).sum();
        let beta = rho * yr;
        for j in 0..n { r[j] += s_list[i][j] * (alpha[i] - beta); }
    }

    r.iter().map(|ri| -ri).collect()
}

pub fn optimize(
    problem:    &mut Problem,
    qfunc:      &dyn QFunc,
    likelihood: &dyn LogLikelihood,
    y:          &[f64],
    a_i: Option<&[usize]>, a_j: Option<&[usize]>, a_x: Option<&[f64]>,
    x_mat:      Option<&[f64]>,
    n_fixed:    usize,
    theta_init: &[f64],
    params:     &OptimizerParams,
) -> Result<OptimResult, InlaError> {
    let n_model  = qfunc.n_hyperparams();
    let n_lik    = likelihood.n_hyperparams();
    let h        = params.finite_diff_step;
    let max_iter = params.max_evals;
    let tol_grad = params.tol_grad;
    let m        = params.lbfgs_memory;

    let mut theta   = theta_init.to_vec();
    let mut n_evals = 0_usize;
    let mut x_warm  = vec![0.0_f64; y.len()];
    let mut beta_warm = vec![0.0_f64; n_fixed];

    let mut s_list: Vec<Vec<f64>> = Vec::with_capacity(m);
    let mut y_list: Vec<Vec<f64>> = Vec::with_capacity(m);

    let (mut f_cur, _, beta_init_warm, _, _) = laplace_eval(
        problem, qfunc, likelihood, y, a_i, a_j, a_x, x_mat, n_fixed, &theta, &x_warm, &beta_warm,
        n_model, 10, 1e-4,
    ).unwrap_or_else(|_| (f64::MAX / 2.0, vec![0.0; problem.n()], vec![0.0; n_fixed], vec![], vec![]));
    beta_warm = beta_init_warm;
    n_evals += 1;

    let mut grad = laplace_gradient(
        problem, qfunc, likelihood, y, a_i, a_j, a_x, x_mat, n_fixed, &theta, f_cur, &x_warm, &beta_warm,
        n_model, n_lik, h,
    );
    n_evals += n_model + n_lik;

    for _iter in 0..max_iter {
        let grad_norm: f64 = grad.iter().map(|g| g*g).sum::<f64>().sqrt();
        if grad_norm < tol_grad { break; }

        let direction = lbfgs_direction(&grad, &s_list, &y_list);

        let c1      = 1e-4_f64;
        let grad_d: f64 = grad.iter().zip(direction.iter()).map(|(g,d)| g*d).sum();
        let mut alpha   = 1.0_f64;
        let mut accepted = false;

        for _ in 0..20 {
            let theta_new: Vec<f64> = theta.iter().zip(direction.iter()).map(|(&t, &d)| t + alpha * d).collect();

            match laplace_eval(
                problem, qfunc, likelihood, y, a_i, a_j, a_x, x_mat, n_fixed, &theta_new, &x_warm, &beta_warm,
                n_model, 10, 1e-4,
            ) {
                Ok((f_new, x_new, beta_new, ..)) if f_new <= f_cur + c1 * alpha * grad_d => {
                    let s_k: Vec<f64> = theta_new.iter().zip(theta.iter()).map(|(tn, t)| tn - t).collect();

                    let grad_new = laplace_gradient(
                        problem, qfunc, likelihood, y, a_i, a_j, a_x, x_mat, n_fixed, &theta_new,
                        f_new, &x_new, &beta_new, n_model, n_lik, h,
                    );
                    n_evals += n_model + n_lik;

                    let y_k: Vec<f64> = grad_new.iter().zip(grad.iter()).map(|(gn, g)| gn - g).collect();

                    let sy: f64 = s_k.iter().zip(y_k.iter()).map(|(s,y)| s*y).sum();
                    if sy > 1e-10 {
                        if s_list.len() == m {
                            s_list.remove(0);
                            y_list.remove(0);
                        }
                        s_list.push(s_k);
                        y_list.push(y_k);
                    }

                    theta      = theta_new;
                    f_cur      = f_new;
                    x_warm     = x_new;
                    beta_warm  = beta_new;
                    grad       = grad_new;
                    n_evals += 1;
                    accepted = true;
                    break;
                }
                _ => {
                    alpha  *= 0.5;
                    n_evals += 1;
                }
            }
        }
        if !accepted { break; }
    }

    let (neg_log_mlik, _, _, _, _) = laplace_eval(
        problem, qfunc, likelihood, y, a_i, a_j, a_x, x_mat, n_fixed, &theta, &x_warm, &beta_warm, n_model, 20, 1e-6,
    ).map_err(|_| InlaError::ConvergenceFailed { reason: "IRLS no convergió en theta*".to_string() })?;
    
    let log_mlik = -neg_log_mlik;

    Ok(OptimResult { theta_opt: theta, log_mlik, n_evals })
}

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
        let result = optimize(&mut p, &model, &lik, &y, None, None, None, None, 0, &[2.0_f64.ln(), 1.0_f64.ln()], &OptimizerParams::default()).unwrap();
        assert!(result.log_mlik.is_finite());
    }

    #[test]
    fn warm_start_preserves_result() {
        let n = 8;
        let y: Vec<f64> = (0..n).map(|i| i as f64 * 0.5).collect();
        let model = IidModel::new(n);
        let lik   = GaussianLikelihood;
        let theta = [1.0, 0.5];
        let mut p = Problem::new(&model);

        let (f1, x1, _, _, _) = laplace_eval(&mut p, &model, &lik, &y, None, None, None, None, 0, &theta, &[], &[], 1, 10, 1e-6).unwrap();
        let (f2, _, _, _, _) = laplace_eval(&mut p, &model, &lik, &y, None, None, None, None, 0, &theta, &x1, &[], 1, 10, 1e-6).unwrap();

        assert!((f1 - f2).abs() < 1e-4);
    }
}
