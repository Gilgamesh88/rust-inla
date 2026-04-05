use crate::error::InlaError;
use crate::likelihood::LogLikelihood;
use crate::models::QFunc;
use crate::problem::Problem;
use crate::optimizer::{laplace_eval, OptimResult};

use faer::Mat;

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
    theta_opt: &[f64],
    intercept: bool,
    h: f64,
) -> Result<Mat<f64>, InlaError> {
    let n_model = qfunc.n_hyperparams();
    let n_lik = likelihood.n_hyperparams();
    let d = n_model + n_lik;
    let mut hessian = Mat::<f64>::zeros(d, d);

    let x_warm = vec![0.0_f64; y.len()];
    let beta0_warm = 0.0_f64;

    // Evaluación de f_opt (neg log mlik en el modo)
    let (neg_log_mlik_opt, _, _, _, _) = laplace_eval(
        problem, qfunc, likelihood, y, x_idx, theta_opt, &x_warm, beta0_warm, intercept, n_model, 20, 1e-6,
    )?;

    // Compute diagonal H_{ii}
    for i in 0..d {
        let mut t_plus = theta_opt.to_vec();
        t_plus[i] += h;
        let mut t_minus = theta_opt.to_vec();
        t_minus[i] -= h;

        let (f_plus, ..) = laplace_eval(
            problem, qfunc, likelihood, y, x_idx, &t_plus, &x_warm, beta0_warm, intercept, n_model, 20, 1e-6,
        ).unwrap_or((neg_log_mlik_opt + 1e3, vec![], 0.0, vec![], vec![]));

        let (f_minus, ..) = laplace_eval(
            problem, qfunc, likelihood, y, x_idx, &t_minus, &x_warm, beta0_warm, intercept, n_model, 20, 1e-6,
        ).unwrap_or((neg_log_mlik_opt + 1e3, vec![], 0.0, vec![], vec![]));

        let d2 = (f_plus - 2.0 * neg_log_mlik_opt + f_minus) / (h * h);
        hessian[(i, i)] = d2;
    }

    // Compute off-diagonal H_{ij}
    for i in 0..d {
        for j in (i + 1)..d {
            let mut t_pp = theta_opt.to_vec(); t_pp[i] += h; t_pp[j] += h;
            let mut t_mm = theta_opt.to_vec(); t_mm[i] -= h; t_mm[j] -= h;
            let mut t_pm = theta_opt.to_vec(); t_pm[i] += h; t_pm[j] -= h;
            let mut t_mp = theta_opt.to_vec(); t_mp[i] -= h; t_mp[j] += h;

            let (f_pp, ..) = laplace_eval(problem, qfunc, likelihood, y, x_idx, &t_pp, &x_warm, beta0_warm, intercept, n_model, 20, 1e-6)
                .unwrap_or((neg_log_mlik_opt + 1e3, vec![], 0.0, vec![], vec![]));
            let (f_mm, ..) = laplace_eval(problem, qfunc, likelihood, y, x_idx, &t_mm, &x_warm, beta0_warm, intercept, n_model, 20, 1e-6)
                .unwrap_or((neg_log_mlik_opt + 1e3, vec![], 0.0, vec![], vec![]));
            let (f_pm, ..) = laplace_eval(problem, qfunc, likelihood, y, x_idx, &t_pm, &x_warm, beta0_warm, intercept, n_model, 20, 1e-6)
                .unwrap_or((neg_log_mlik_opt + 1e3, vec![], 0.0, vec![], vec![]));
            let (f_mp, ..) = laplace_eval(problem, qfunc, likelihood, y, x_idx, &t_mp, &x_warm, beta0_warm, intercept, n_model, 20, 1e-6)
                .unwrap_or((neg_log_mlik_opt + 1e3, vec![], 0.0, vec![], vec![]));

            let d2 = (f_pp - f_pm - f_mp + f_mm) / (4.0 * h * h);
            hessian[(i, j)] = d2;
            hessian[(j, i)] = d2;
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
    theta_opt: &[f64],
    intercept: bool,
) -> Result<CcdIntegration, InlaError> {
    let d = theta_opt.len();

    // Si D=0, el CCD es solo un punto (el modo)
    if d == 0 {
        return Ok(CcdIntegration {
            points: vec![CcdPoint { theta: vec![], weight: 1.0 }]
        });
    }

    // 1. Matriz Jacobiana/Hessiana numérica en el modo
    // Extraemos la curvatura local
    let hessian = compute_hessian(problem, qfunc, likelihood, y, x_idx, theta_opt, intercept, 1e-4)?;

    // 2. Eigendecomposition of Hessian H
    // Debido a que faer.svd da un error similar a "no method svd", vamos a usar descomposición simple
    // o SVD manual si se expone diferentemente en 0.24. Para este puerto aseguraremos que usamos algo genérico
    // o el solver Cholesky si es Pos Def.
    // Asumimos H es semi-definida positiva. En lugar de eigendecomposition, en 0.24 iteraremos:
    // SVD se accedería en faer 0.24 normalmente vía svd. 
    // Para simplificar, implementaré un Jacobi eigenvalue algorithm simple para D < 10 dado que faer-0.24 API cambió dramáticamente.
    
    let (v, s) = simple_jacobi_eigenvalue(&hessian);

    // 3. Crear CCD points de acuerdo a z = V * Sigma^(-1/2) * z_std
    let mut points = Vec::new();

    // delta shift = 1.0 standard dev factor para star points (usualmente \sqrt{2} o f(D))
    // Usaremos un shift escalado simple: INLA por defecto z = +- 1 * delta para las estrellas.
    // Usaremos z-step = 1.0 o lo que dicte un fz de CCD estandar.
    let fz = 1.0; 

    // Compute transformations: z -> delta_theta
    // delta_theta = V * diag(1/sqrt(s)) * z
    let mut t_matrix = Mat::<f64>::zeros(d, d);
    for i in 0..d {
        let std_dev = if s[i] > 1e-12 { 1.0 / s[i].sqrt() } else { 0.0 };
        for j in 0..d {
            t_matrix[(j, i)] = v[(j, i)] * std_dev;
        }
    }

    // Punto central
    points.push(CcdPoint { theta: theta_opt.to_vec(), weight: 1.0 });

    // Star points: +- fz a lo largo de cada eje de componente principal
    for i in 0..d {
        for sign in [-1.0_f64, 1.0_f64] {
            let mut th = theta_opt.to_vec();
            for j in 0..d {
                th[j] += t_matrix[(j, i)] * (sign * fz);
            }
            points.push(CcdPoint { theta: th, weight: 1.0 });
        }
    }

    // Fractional / Full Factorial (2^D) se añade si D > 1 
    if d > 1 && d <= 5 {
        let n_fac = 1 << d;
        let fac_z = if d <= 2 { 1.0 } else { 1.0 }; // En diseos de CCD reales esto depende, para small D = 1.0
        for mut i in 0..n_fac {
            let mut th = theta_opt.to_vec();
            for idx in 0..d {
                let sign = if (i & 1) == 0 { -1.0 } else { 1.0 };
                i >>= 1;
                for j in 0..d {
                    th[j] += t_matrix[(j, idx)] * (sign * fac_z);
                }
            }
            points.push(CcdPoint { theta: th, weight: 1.0 });
        }
    }

    // 4. Evaluar log_pi a lo largo de los puntos para calcular integration weights
    // log pi_tilde(theta | y) ∝ log marginal likelihood en ese point (porque prior esta integrado)
    // Ya que optimizer minimizaba `-log_mlik`, usamos `laplace_eval` y leemos el 1er val.
    
    let x_warm = vec![0.0_f64; y.len()];
    
    let mut log_weights = Vec::with_capacity(points.len());
    let mut max_log_w = -f64::MAX;

    for pt in &points {
        let (neg_log_mlik, _, _, _, _) = laplace_eval(
            problem, qfunc, likelihood, y, x_idx, &pt.theta, &x_warm, 0.0, intercept, qfunc.n_hyperparams(), 10, 1e-4,
        ).unwrap_or((f64::MAX / 2.0, vec![], 0.0, vec![], vec![]));
        
        let log_w = -neg_log_mlik; 
        log_weights.push(log_w);
        if log_w > max_log_w { max_log_w = log_w; }
    }

    // Normalize weights
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

/// A simple Jacobi eigenvalue method for small symmetric matrices. 
/// Extremely robust, 100% fine for D < 10.
fn simple_jacobi_eigenvalue(mat: &Mat<f64>) -> (Mat<f64>, Vec<f64>) {
    let n = mat.nrows();
    let mut v = Mat::<f64>::zeros(n, n);
    for i in 0..n {
        v[(i, i)] = 1.0;
    }
    
    let mut a = Mat::<f64>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            a[(i, j)] = mat[(i, j)];
        }
    }

    for _ in 0..50 {
        let mut max_off = 0.0;
        let mut p = 0;
        let mut q = 0;
        for i in 0..n {
            for j in (i + 1)..n {
                let off = a[(i, j)].abs();
                if off > max_off {
                    max_off = off;
                    p = i;
                    q = j;
                }
            }
        }
        if max_off < 1e-12 { break; }

        let app = a[(p, p)];
        let aqq = a[(q, q)];
        let apq = a[(p, q)];

        let theta_ang = (aqq - app) / (2.0 * apq);
        let t = if theta_ang >= 0.0 {
            1.0 / (theta_ang + (theta_ang * theta_ang + 1.0_f64).sqrt())
        } else {
            1.0 / (theta_ang - (theta_ang * theta_ang + 1.0_f64).sqrt())
        };
        let c = 1.0 / (t * t + 1.0_f64).sqrt();
        let s = t * c;

        a[(p, p)] = app - t * apq;
        a[(q, q)] = aqq + t * apq;
        a[(p, q)] = 0.0;
        a[(q, p)] = 0.0;

        for r in 0..n {
            if r != p && r != q {
                let arp = a[(r, p)];
                let arq = a[(r, q)];
                a[(r, p)] = c * arp - s * arq;
                a[(p, r)] = c * arp - s * arq;
                a[(r, q)] = s * arp + c * arq;
                a[(q, r)] = s * arp + c * arq;
            }
            let vrp = v[(r, p)];
            let vrq = v[(r, q)];
            v[(r, p)] = c * vrp - s * vrq;
            v[(r, q)] = s * vrp + c * vrq;
        }
    }

    let mut eigenvals = vec![0.0; n];
    for i in 0..n {
        eigenvals[i] = a[(i, i)].max(1e-12); // clamp to avoid negative values from small numerical errors
    }

    (v, eigenvals)
}
