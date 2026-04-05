use crate::error::InlaError;
use crate::likelihood::LogLikelihood;
use crate::models::QFunc;
use crate::problem::Problem;
use crate::optimizer::laplace_eval;

pub struct CcdPoint {
    pub theta: Vec<f64>,
    pub weight: f64,
}

pub struct CcdIntegration {
    pub points: Vec<CcdPoint>,
}

fn compute_hessian(
    problem: &mut Problem,
    qfunc: &dyn QFunc,
    likelihood: &dyn LogLikelihood,
    y: &[f64],
    x_idx: Option<&[usize]>,
    x_mat: Option<&[f64]>,
    n_fixed: usize,
    theta_opt: &[f64],
    h: f64,
) -> Result<Vec<f64>, InlaError> {
    let n_model = qfunc.n_hyperparams();
    let n_lik = likelihood.n_hyperparams();
    let d = n_model + n_lik;
    let mut hessian = vec![0.0_f64; d * d];

    let x_warm = vec![0.0_f64; y.len()];
    let beta_warm = vec![0.0_f64; n_fixed];

    let (neg_log_mlik_opt, _, _, _, _) = laplace_eval(
        problem, qfunc, likelihood, y, x_idx, x_mat, n_fixed, theta_opt, &x_warm, &beta_warm, n_model, 20, 1e-6,
    )?;

    for i in 0..d {
        let mut t_plus = theta_opt.to_vec();
        t_plus[i] += h;
        let mut t_minus = theta_opt.to_vec();
        t_minus[i] -= h;

        let (f_plus, ..) = laplace_eval(
            problem, qfunc, likelihood, y, x_idx, x_mat, n_fixed, &t_plus, &x_warm, &beta_warm, n_model, 20, 1e-6,
        ).unwrap_or((neg_log_mlik_opt + 1e3, vec![], vec![], vec![], vec![]));

        let (f_minus, ..) = laplace_eval(
            problem, qfunc, likelihood, y, x_idx, x_mat, n_fixed, &t_minus, &x_warm, &beta_warm, n_model, 20, 1e-6,
        ).unwrap_or((neg_log_mlik_opt + 1e3, vec![], vec![], vec![], vec![]));

        let d2 = (f_plus - 2.0 * neg_log_mlik_opt + f_minus) / (h * h);
        hessian[i * d + i] = d2;
    }

    for i in 0..d {
        for j in (i + 1)..d {
            let mut t_pp = theta_opt.to_vec(); t_pp[i] += h; t_pp[j] += h;
            let mut t_mm = theta_opt.to_vec(); t_mm[i] -= h; t_mm[j] -= h;
            let mut t_pm = theta_opt.to_vec(); t_pm[i] += h; t_pm[j] -= h;
            let mut t_mp = theta_opt.to_vec(); t_mp[i] -= h; t_mp[j] += h;

            let (f_pp, ..) = laplace_eval(problem, qfunc, likelihood, y, x_idx, x_mat, n_fixed, &t_pp, &x_warm, &beta_warm, n_model, 20, 1e-6)
                .unwrap_or((neg_log_mlik_opt + 1e3, vec![], vec![], vec![], vec![]));
            let (f_mm, ..) = laplace_eval(problem, qfunc, likelihood, y, x_idx, x_mat, n_fixed, &t_mm, &x_warm, &beta_warm, n_model, 20, 1e-6)
                .unwrap_or((neg_log_mlik_opt + 1e3, vec![], vec![], vec![], vec![]));
            let (f_pm, ..) = laplace_eval(problem, qfunc, likelihood, y, x_idx, x_mat, n_fixed, &t_pm, &x_warm, &beta_warm, n_model, 20, 1e-6)
                .unwrap_or((neg_log_mlik_opt + 1e3, vec![], vec![], vec![], vec![]));
            let (f_mp, ..) = laplace_eval(problem, qfunc, likelihood, y, x_idx, x_mat, n_fixed, &t_mp, &x_warm, &beta_warm, n_model, 20, 1e-6)
                .unwrap_or((neg_log_mlik_opt + 1e3, vec![], vec![], vec![], vec![]));

            let d2 = (f_pp - f_pm - f_mp + f_mm) / (4.0 * h * h);
            hessian[i * d + j] = d2;
            hessian[j * d + i] = d2;
        }
    }

    Ok(hessian)
}

pub fn build_ccd_grid(
    problem: &mut Problem,
    qfunc: &dyn QFunc,
    likelihood: &dyn LogLikelihood,
    y: &[f64],
    x_idx: Option<&[usize]>,
    x_mat: Option<&[f64]>,
    n_fixed: usize,
    theta_opt: &[f64],
) -> Result<CcdIntegration, InlaError> {
    let d = theta_opt.len();

    if d == 0 {
        return Ok(CcdIntegration {
            points: vec![CcdPoint { theta: vec![], weight: 1.0 }]
        });
    }

    let hessian = compute_hessian(problem, qfunc, likelihood, y, x_idx, x_mat, n_fixed, theta_opt, 1e-4)?;
    let (v, s) = simple_jacobi_eigenvalue(&hessian, d);

    let mut points = Vec::new();
    let fz = 1.0; 

    let mut t_matrix = vec![0.0_f64; d * d];
    for i in 0..d {
        let std_dev = if s[i] > 1e-12 { 1.0 / s[i].sqrt() } else { 0.0 };
        for j in 0..d {
            t_matrix[j * d + i] = v[j * d + i] * std_dev;
        }
    }

    points.push(CcdPoint { theta: theta_opt.to_vec(), weight: 1.0 });

    for i in 0..d {
        for sign in [-1.0_f64, 1.0_f64] {
            let mut th = theta_opt.to_vec();
            for j in 0..d {
                th[j] += t_matrix[j * d + i] * (sign * fz);
            }
            points.push(CcdPoint { theta: th, weight: 1.0 });
        }
    }

    if d > 1 && d <= 5 {
        let n_fac = 1 << d;
        let fac_z = if d <= 2 { 1.0 } else { 1.0 }; 
        for mut i in 0..n_fac {
            let mut th = theta_opt.to_vec();
            for idx in 0..d {
                let sign = if (i & 1) == 0 { -1.0 } else { 1.0 };
                i >>= 1;
                for j in 0..d {
                    th[j] += t_matrix[j * d + idx] * (sign * fac_z);
                }
            }
            points.push(CcdPoint { theta: th, weight: 1.0 });
        }
    }

    let x_warm = vec![0.0_f64; y.len()];
    let beta_warm = vec![0.0_f64; n_fixed];
    
    let mut log_weights = Vec::with_capacity(points.len());
    let mut max_log_w = -f64::MAX;

    for pt in &points {
        let (neg_log_mlik, _, _, _, _) = laplace_eval(
            problem, qfunc, likelihood, y, x_idx, x_mat, n_fixed, &pt.theta, &x_warm, &beta_warm, qfunc.n_hyperparams(), 10, 1e-4,
        ).unwrap_or((f64::MAX / 2.0, vec![], vec![], vec![], vec![]));
        
        let log_w = -neg_log_mlik; 
        log_weights.push(log_w);
        if log_w > max_log_w { max_log_w = log_w; }
    }

    let mut sum_w = 0.0;
    for i in 0..points.len() {
        let w = (log_weights[i] - max_log_w).exp();
        points[i].weight = w;
        sum_w += w;
    }
    for i in 0..points.len() {
        points[i].weight /= sum_w;
    }

    Ok(CcdIntegration { points })
}

fn simple_jacobi_eigenvalue(mat: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut v = vec![0.0_f64; n * n];
    for i in 0..n { v[i * n + i] = 1.0; }
    
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
        if max_off < 1e-12 { break; }

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
    for i in 0..n { eigenvals[i] = a[i * n + i].max(1e-12); }

    (v, eigenvals)
}
