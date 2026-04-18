use std::f64::consts::PI;
use std::time::Instant;

use crate::diagnostics::LaplacePhase;
use crate::error::InlaError;
use crate::inference::InlaModel;
use crate::optimizer::laplace_eval;
use crate::problem::Problem;

pub struct CcdPoint {
    pub theta: Vec<f64>,
    pub base_weight: f64,
    pub weight: f64,
    pub log_mlik: f64,
    pub log_weight: f64,
}

pub struct CcdIntegration {
    pub points: Vec<CcdPoint>,
    pub hessian_eigenvalues: Vec<f64>,
}

impl CcdIntegration {
    pub fn theta_laplace_correction(&self) -> f64 {
        theta_laplace_correction(&self.hessian_eigenvalues)
    }
}

fn theta_laplace_correction(hessian_eigenvalues: &[f64]) -> f64 {
    if hessian_eigenvalues.is_empty() {
        return 0.0;
    }

    let d = hessian_eigenvalues.len() as f64;
    0.5 * d * (2.0 * PI).ln()
        - 0.5
            * hessian_eigenvalues
                .iter()
                .map(|x| x.max(1e-12).ln())
                .sum::<f64>()
}

const CCD_F0: f64 = 1.1;
const CCD_EVAL_IRLS_MAX_ITER: usize = 60;
const CCD_EVAL_IRLS_TOL: f64 = 1e-8;

fn ccd_rule_weights(d: usize, include_factorial: bool) -> (f64, f64) {
    if d == 0 {
        return (1.0, 0.0);
    }

    let n_noncentral = 2 * d + if include_factorial { 1usize << d } else { 0 };
    let f0_sq = CCD_F0 * CCD_F0;
    let delta = 1.0 / (1.0 + (-0.5 * d as f64 * f0_sq).exp() * (f0_sq - 1.0));
    let center_weight = 1.0 - delta;
    let noncentral_weight = delta / n_noncentral as f64;

    (center_weight.max(0.0), noncentral_weight.max(0.0))
}

fn approx_hessian(
    problem: &mut Problem,
    model: &InlaModel<'_>,
    theta_opt: &[f64],
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), InlaError> {
    let d = theta_opt.len();
    let h = 0.1;
    let mut hessian = vec![0.0_f64; d * d];

    let x_warm = vec![0.0_f64; model.n_latent];
    let beta_warm = vec![0.0_f64; model.n_fixed];

    let opt_eval = laplace_eval(
        problem,
        model,
        LaplacePhase::Ccd,
        theta_opt,
        &x_warm,
        &beta_warm,
        20,
        1e-6,
    )?;
    let neg_log_mlik_opt = opt_eval.neg_log_mlik;
    let x_mode_opt = opt_eval.x_hat;
    let beta_mode_opt = opt_eval.beta_hat;

    for i in 0..d {
        let mut t_plus = theta_opt.to_vec();
        t_plus[i] += h;
        let mut t_minus = theta_opt.to_vec();
        t_minus[i] -= h;

        let f_plus = laplace_eval(
            problem,
            model,
            LaplacePhase::Ccd,
            &t_plus,
            &x_mode_opt,
            &beta_mode_opt,
            20,
            1e-6,
        )?
        .neg_log_mlik;
        let f_minus = laplace_eval(
            problem,
            model,
            LaplacePhase::Ccd,
            &t_minus,
            &x_mode_opt,
            &beta_mode_opt,
            20,
            1e-6,
        )?
        .neg_log_mlik;

        let d2 = (f_plus - 2.0 * neg_log_mlik_opt + f_minus) / (h * h);
        hessian[i * d + i] = d2;
    }

    for i in 0..d {
        for j in (i + 1)..d {
            let mut t_pp = theta_opt.to_vec();
            t_pp[i] += h;
            t_pp[j] += h;
            let mut t_mm = theta_opt.to_vec();
            t_mm[i] -= h;
            t_mm[j] -= h;
            let mut t_pm = theta_opt.to_vec();
            t_pm[i] += h;
            t_pm[j] -= h;
            let mut t_mp = theta_opt.to_vec();
            t_mp[i] -= h;
            t_mp[j] += h;

            let f_pp = laplace_eval(
                problem,
                model,
                LaplacePhase::Ccd,
                &t_pp,
                &x_mode_opt,
                &beta_mode_opt,
                20,
                1e-6,
            )?
            .neg_log_mlik;
            let f_mm = laplace_eval(
                problem,
                model,
                LaplacePhase::Ccd,
                &t_mm,
                &x_mode_opt,
                &beta_mode_opt,
                20,
                1e-6,
            )?
            .neg_log_mlik;
            let f_pm = laplace_eval(
                problem,
                model,
                LaplacePhase::Ccd,
                &t_pm,
                &x_mode_opt,
                &beta_mode_opt,
                20,
                1e-6,
            )?
            .neg_log_mlik;
            let f_mp = laplace_eval(
                problem,
                model,
                LaplacePhase::Ccd,
                &t_mp,
                &x_mode_opt,
                &beta_mode_opt,
                20,
                1e-6,
            )?
            .neg_log_mlik;

            let d2 = (f_pp - f_pm - f_mp + f_mm) / (4.0 * h * h);
            hessian[i * d + j] = d2;
            hessian[j * d + i] = d2;
        }
    }

    Ok((hessian, x_mode_opt, beta_mode_opt))
}

pub fn build_ccd_grid(
    problem: &mut Problem,
    model: &InlaModel<'_>,
    theta_opt: &[f64],
) -> Result<CcdIntegration, InlaError> {
    let ccd_started = Instant::now();
    let d = theta_opt.len();

    if d == 0 {
        problem.diagnostics_mut().ccd_time += ccd_started.elapsed();
        return Ok(CcdIntegration {
            points: vec![CcdPoint {
                theta: vec![],
                base_weight: 1.0,
                weight: 1.0,
                log_mlik: 0.0,
                log_weight: 0.0,
            }],
            hessian_eigenvalues: vec![],
        });
    }

    let (hessian, mut x_warm, mut beta_warm) = approx_hessian(problem, model, theta_opt)?;
    let (v, s) = simple_jacobi_eigenvalue(&hessian, d);

    let mut points = Vec::new();
    let include_factorial = d > 1 && d <= 5;
    let (center_weight, noncentral_weight) = ccd_rule_weights(d, include_factorial);
    let factorial_z = CCD_F0;
    let axial_z = CCD_F0 * (d as f64).sqrt();

    let mut t_matrix = vec![0.0_f64; d * d];
    for i in 0..d {
        let std_dev = if s[i] > 1e-12 { 1.0 / s[i].sqrt() } else { 0.0 };
        for j in 0..d {
            t_matrix[j * d + i] = v[j * d + i] * std_dev;
        }
    }

    points.push(CcdPoint {
        theta: theta_opt.to_vec(),
        base_weight: center_weight,
        weight: 1.0,
        log_mlik: 0.0,
        log_weight: 0.0,
    });

    for i in 0..d {
        for sign in [-1.0_f64, 1.0_f64] {
            let mut th = theta_opt.to_vec();
            for j in 0..d {
                th[j] += t_matrix[j * d + i] * (sign * axial_z);
            }
            points.push(CcdPoint {
                theta: th,
                base_weight: noncentral_weight,
                weight: 1.0,
                log_mlik: 0.0,
                log_weight: 0.0,
            });
        }
    }

    if include_factorial {
        let n_fac = 1 << d;
        for mask in 0..n_fac {
            let mut th = theta_opt.to_vec();
            for idx in 0..d {
                let sign = if (mask & (1 << idx)) == 0 { -1.0 } else { 1.0 };
                for j in 0..d {
                    th[j] += t_matrix[j * d + idx] * (sign * factorial_z);
                }
            }
            points.push(CcdPoint {
                theta: th,
                base_weight: noncentral_weight,
                weight: 1.0,
                log_mlik: 0.0,
                log_weight: 0.0,
            });
        }
    }

    let mut log_weights = Vec::with_capacity(points.len());
    let mut max_log_w = -f64::MAX;

    for pt in &points {
        let eval = laplace_eval(
            problem,
            model,
            LaplacePhase::Ccd,
            &pt.theta,
            &x_warm,
            &beta_warm,
            CCD_EVAL_IRLS_MAX_ITER,
            CCD_EVAL_IRLS_TOL,
        )?;
        let log_mlik = -eval.neg_log_mlik;
        let log_w = pt.base_weight.max(1e-300).ln() + log_mlik;
        log_weights.push(log_w);
        if log_w > max_log_w {
            max_log_w = log_w;
        }
        x_warm = eval.x_hat;
        beta_warm = eval.beta_hat;
    }

    let mut sum_w = 0.0;
    for (pt, log_w) in points.iter_mut().zip(log_weights.iter()) {
        let weight = (*log_w - max_log_w).exp();
        pt.weight = weight;
        pt.log_mlik = *log_w - pt.base_weight.max(1e-300).ln();
        pt.log_weight = *log_w;
        sum_w += weight;
    }
    for pt in &mut points {
        pt.weight /= sum_w;
    }

    problem.diagnostics_mut().ccd_time += ccd_started.elapsed();
    Ok(CcdIntegration {
        points,
        hessian_eigenvalues: s,
    })
}

fn simple_jacobi_eigenvalue(mat: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut v = vec![0.0_f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }

    let mut a = mat.to_vec();

    for _ in 0..50 {
        let mut max_off = 0.0;
        let mut p = 0;
        let mut q = 0;
        for i in 0..n {
            for j in (i + 1)..n {
                let off = a[i * n + j].abs();
                if off > max_off {
                    max_off = off;
                    p = i;
                    q = j;
                }
            }
        }
        if max_off < 1e-12 {
            break;
        }

        let app = a[p * n + p];
        let aqq = a[q * n + q];
        let apq = a[p * n + q];

        let theta_ang = (aqq - app) / (2.0 * apq);
        let t = if theta_ang >= 0.0 {
            1.0 / (theta_ang + (theta_ang * theta_ang + 1.0_f64).sqrt())
        } else {
            1.0 / (theta_ang - (theta_ang * theta_ang + 1.0_f64).sqrt())
        };
        let c = 1.0 / (t * t + 1.0_f64).sqrt();
        let s = t * c;

        a[p * n + p] = app - t * apq;
        a[q * n + q] = aqq + t * apq;
        a[p * n + q] = 0.0;
        a[q * n + p] = 0.0;

        for r in 0..n {
            if r != p && r != q {
                let arp = a[r * n + p];
                let arq = a[r * n + q];
                a[r * n + p] = c * arp - s * arq;
                a[p * n + r] = c * arp - s * arq;
                a[r * n + q] = s * arp + c * arq;
                a[q * n + r] = s * arp + c * arq;
            }
            let vrp = v[r * n + p];
            let vrq = v[r * n + q];
            v[r * n + p] = c * vrp - s * vrq;
            v[r * n + q] = s * vrp + c * vrq;
        }
    }

    let mut eigenvals = vec![0.0; n];
    for (i, eigenval) in eigenvals.iter_mut().enumerate().take(n) {
        *eigenval = a[i * n + i].max(1e-12);
    }

    (v, eigenvals)
}

#[cfg(test)]
mod tests {
    use super::{ccd_rule_weights, theta_laplace_correction, CCD_F0};
    use approx::assert_abs_diff_eq;

    #[test]
    fn theta_laplace_correction_matches_closed_form() {
        let eigs = [4.0_f64, 9.0_f64];
        let expected = (2.0 * std::f64::consts::PI).ln() - 0.5 * (36.0_f64).ln();
        assert_abs_diff_eq!(theta_laplace_correction(&eigs), expected, epsilon = 1e-12);
    }

    #[test]
    fn theta_laplace_correction_is_zero_without_hyperparameters() {
        assert_abs_diff_eq!(theta_laplace_correction(&[]), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn ccd_rule_weights_sum_to_one_with_factorial_points() {
        let d = 2usize;
        let (k0, k1) = ccd_rule_weights(d, true);
        let n_noncentral = 2 * d + (1usize << d);

        assert!(k0 > 0.0);
        assert!(k1 > 0.0);
        assert_abs_diff_eq!(k0 + n_noncentral as f64 * k1, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn ccd_rule_weights_reduce_to_center_and_axial_only() {
        let d = 1usize;
        let (k0, k1) = ccd_rule_weights(d, false);
        let expected_delta =
            1.0 / (1.0 + (-0.5 * d as f64 * CCD_F0 * CCD_F0).exp() * (CCD_F0 * CCD_F0 - 1.0));

        assert_abs_diff_eq!(k0, 1.0 - expected_delta, epsilon = 1e-12);
        assert_abs_diff_eq!(2.0 * k1, expected_delta, epsilon = 1e-12);
    }
}
