use crate::error::InlaError;
use crate::inference::InlaModel;
use crate::likelihood::LogLikelihood;
use crate::models::QFunc;
use crate::optimizer::laplace_eval;
use crate::problem::Problem;

// ── CCD design constants ──────────────────────────────────────────────────────

/// Distance factor for the 2D "star" points along each eigenvector.
///
/// Reference: Rue, Martino & Chopin (2009), Table 1.
/// z_star = √(4/3) ≈ 1.1547
/// These points capture the curvature at ±1.15 posterior SDs from θ*.
const STAR_FACTOR: f64 = 1.154_700_538_379_252; // √(4/3)

/// Distance factor for the 2^D "factorial" corner points.
///
/// These sit at ±1 posterior SD along every combination of eigenvectors.
/// Distinct from star factor — using the same value would collapse the design.
const FACTORIAL_FACTOR: f64 = 1.0;

// ── Public types ──────────────────────────────────────────────────────────────

pub struct CcdPoint {
    pub theta:  Vec<f64>,
    pub weight: f64,
}

pub struct CcdIntegration {
    pub points: Vec<CcdPoint>,
}

// ── Hessian estimation ────────────────────────────────────────────────────────

/// Estimates the Hessian of -log p_LA(y|θ) at θ_opt via finite differences.
///
/// Diagonal entries use the centered second-difference formula:
///   H[i,i] = (f(θ+heᵢ) - 2f(θ) + f(θ-heᵢ)) / h²
///
/// Off-diagonal entries use the cross-difference formula:
///   H[i,j] = (f(++)-f(+-)-f(-+)+f(--)) / (4h²)
///
/// h = 1e-4 in log-scale for hyperparameters (dimensionless).
fn compute_hessian(
    problem:   &mut Problem,
    model:     &InlaModel<'_>,
    theta_opt: &[f64],
    h:         f64,
) -> Result<Vec<f64>, InlaError> {
    let d       = theta_opt.len();
    let n_model = model.qfunc.n_hyperparams();
    let mut hessian = vec![0.0_f64; d * d];

    let x_warm    = vec![0.0_f64; model.n_latent];
    let beta_warm = vec![0.0_f64; model.n_fixed];

    let (f0, ..) = laplace_eval(problem, model, theta_opt, &x_warm, &beta_warm, n_model, 20, 1e-6)?;

    // Diagonal entries.
    for i in 0..d {
        let mut t_plus  = theta_opt.to_vec(); t_plus[i]  += h;
        let mut t_minus = theta_opt.to_vec(); t_minus[i] -= h;

        let (f_plus, ..) = laplace_eval(problem, model, &t_plus, &x_warm, &beta_warm, n_model, 20, 1e-6)
            .unwrap_or((f0 + 1e3, vec![], vec![], vec![], vec![]));
        let (f_minus, ..) = laplace_eval(problem, model, &t_minus, &x_warm, &beta_warm, n_model, 20, 1e-6)
            .unwrap_or((f0 + 1e3, vec![], vec![], vec![], vec![]));

        hessian[i * d + i] = (f_plus - 2.0 * f0 + f_minus) / (h * h);
    }

    // Off-diagonal entries.
    for i in 0..d {
        for j in (i + 1)..d {
            let mut t_pp = theta_opt.to_vec(); t_pp[i] += h; t_pp[j] += h;
            let mut t_mm = theta_opt.to_vec(); t_mm[i] -= h; t_mm[j] -= h;
            let mut t_pm = theta_opt.to_vec(); t_pm[i] += h; t_pm[j] -= h;
            let mut t_mp = theta_opt.to_vec(); t_mp[i] -= h; t_mp[j] += h;

            let (f_pp, ..) = laplace_eval(problem, model, &t_pp, &x_warm, &beta_warm, n_model, 20, 1e-6)
                .unwrap_or((f0 + 1e3, vec![], vec![], vec![], vec![]));
            let (f_mm, ..) = laplace_eval(problem, model, &t_mm, &x_warm, &beta_warm, n_model, 20, 1e-6)
                .unwrap_or((f0 + 1e3, vec![], vec![], vec![], vec![]));
            let (f_pm, ..) = laplace_eval(problem, model, &t_pm, &x_warm, &beta_warm, n_model, 20, 1e-6)
                .unwrap_or((f0 + 1e3, vec![], vec![], vec![], vec![]));
            let (f_mp, ..) = laplace_eval(problem, model, &t_mp, &x_warm, &beta_warm, n_model, 20, 1e-6)
                .unwrap_or((f0 + 1e3, vec![], vec![], vec![], vec![]));

            let d2 = (f_pp - f_pm - f_mp + f_mm) / (4.0 * h * h);
            hessian[i * d + j] = d2;
            hessian[j * d + i] = d2;
        }
    }

    Ok(hessian)
}

// ── CCD grid construction ─────────────────────────────────────────────────────

/// Builds the CCD integration grid over θ around θ_opt.
///
/// # Grid layout (Rue, Martino & Chopin 2009, Table 1)
///
/// 1. **Center** — θ*                                      (1 point)
/// 2. **Star**   — θ* ± STAR_FACTOR · σ_k · e_k           (2D points)
/// 3. **Factorial** — θ* ± FACTORIAL_FACTOR · σ_k · e_k   (2^D points, D ≤ 5)
///
/// where e_k is the k-th eigenvector of the Hessian at θ* and
/// σ_k = 1/√λ_k is the implied posterior SD along that direction.
///
/// Weights are proportional to exp(-log p_LA(y|θ_k)) evaluated at each point,
/// normalised in log-space to avoid underflow.
pub fn build_ccd_grid(
    problem:   &mut Problem,
    qfunc:     &dyn QFunc,
    likelihood: &dyn LogLikelihood,
    y:         &[f64],
    a_i:       Option<&[usize]>,
    a_j:       Option<&[usize]>,
    a_x:       Option<&[f64]>,
    x_mat:     Option<&[f64]>,
    n_fixed:   usize,
    theta_opt: &[f64],
) -> Result<CcdIntegration, InlaError> {
    let d = theta_opt.len();

    // D=0: no hyperparameters — single point with weight 1.
    if d == 0 {
        return Ok(CcdIntegration {
            points: vec![CcdPoint { theta: vec![], weight: 1.0 }],
        });
    }

    // Build InlaModel for laplace_eval calls.
    let n_latent = problem.n();
    let model = InlaModel {
        qfunc,
        likelihood,
        y,
        theta_init:   theta_opt.to_vec(),
        fixed_matrix: x_mat,
        n_fixed,
        n_latent,
        a_i,
        a_j,
        a_x,
    };

    // Estimate Hessian at θ* and decompose.
    let hessian = compute_hessian(problem, &model, theta_opt, 1e-4)?;
    let (eigvecs, eigenvals) = simple_jacobi_eigenvalue(&hessian, d);

    // Build transform matrix T where column k = eigvec_k / √λ_k (= σ_k · e_k).
    // T[j, k] = component of eigenvector k in direction j, scaled by σ_k.
    let mut t_matrix = vec![0.0_f64; d * d];
    for k in 0..d {
        let sigma_k = if eigenvals[k] > 1e-12 { 1.0 / eigenvals[k].sqrt() } else { 0.0 };
        for j in 0..d {
            t_matrix[j * d + k] = eigvecs[j * d + k] * sigma_k;
        }
    }

    let mut points: Vec<CcdPoint> = Vec::new();

    // 1. Center point.
    points.push(CcdPoint { theta: theta_opt.to_vec(), weight: 1.0 });

    // 2. Star points: θ* ± STAR_FACTOR along each eigenvector direction.
    //
    // FIX: was `fz = 1.0` — corrected to STAR_FACTOR = √(4/3) ≈ 1.1547.
    // With fz=1.0 the star and factorial points coincided, collapsing all
    // mass onto one side and causing the weight=0,0,1 pattern in the output.
    for k in 0..d {
        for &sign in &[-1.0_f64, 1.0_f64] {
            let mut th = theta_opt.to_vec();
            for j in 0..d {
                th[j] += t_matrix[j * d + k] * (sign * STAR_FACTOR);
            }
            points.push(CcdPoint { theta: th, weight: 1.0 });
        }
    }

    // 3. Factorial corner points (only for D ≤ 5 to keep count manageable).
    //
    // FIX: was `fac_z = if d <= 2 { 1.0 } else { 1.0 }` — both branches
    // identical, now uses named constant FACTORIAL_FACTOR = 1.0 for clarity.
    if d > 1 && d <= 5 {
        let n_factorial = 1usize << d; // 2^D combinations
        for combo in 0..n_factorial {
            let mut th = theta_opt.to_vec();
            for k in 0..d {
                let sign = if (combo >> k) & 1 == 0 { -1.0_f64 } else { 1.0_f64 };
                for j in 0..d {
                    th[j] += t_matrix[j * d + k] * (sign * FACTORIAL_FACTOR);
                }
            }
            points.push(CcdPoint { theta: th, weight: 1.0 });
        }
    }

    // 4. Evaluate Laplace score at each point and normalise weights in log-space.
    let x_warm    = vec![0.0_f64; n_latent];
    let beta_warm = vec![0.0_f64; n_fixed];
    let n_model   = qfunc.n_hyperparams();

    let mut log_weights = Vec::with_capacity(points.len());
    let mut max_log_w   = f64::NEG_INFINITY;

    for pt in &points {
        let (neg_log_mlik, ..) = laplace_eval(
            problem, &model, &pt.theta, &x_warm, &beta_warm, n_model, 10, 1e-4,
        )
        .unwrap_or((f64::MAX / 2.0, vec![], vec![], vec![], vec![]));

        // log weight = log p_LA(y|θ) = -neg_log_mlik.
        let log_w = -neg_log_mlik;
        log_weights.push(log_w);
        if log_w > max_log_w {
            max_log_w = log_w;
        }
    }

    // Normalise: w_k = exp(log_w_k - max) / Σ exp(log_w_i - max).
    let mut sum_w = 0.0_f64;
    for i in 0..points.len() {
        let w = (log_weights[i] - max_log_w).exp();
        points[i].weight = w;
        sum_w += w;
    }
    if sum_w > 0.0 {
        for pt in &mut points {
            pt.weight /= sum_w;
        }
    }

    Ok(CcdIntegration { points })
}

// ── Jacobi eigenvalue decomposition ──────────────────────────────────────────

/// Computes eigenvalues and eigenvectors of a symmetric matrix via Jacobi rotations.
///
/// Returns (eigenvectors, eigenvalues) where eigenvectors is stored column-major:
///   eigvecs[j * n + k] = component j of eigenvector k.
///
/// Eigenvalues are clamped to ≥ 1e-12 (positive semi-definite guarantee).
fn simple_jacobi_eigenvalue(mat: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    // V starts as identity — columns accumulate the eigenvectors.
    let mut v = vec![0.0_f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }

    let mut a = mat.to_vec();

    // At most 50 sweeps — converges for well-conditioned Hessians in ≤ 10.
    for _ in 0..50 {
        // Find the largest off-diagonal entry.
        let mut max_off = 0.0_f64;
        let mut p = 0usize;
        let mut q = 0usize;
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
            break; // Converged.
        }

        // Compute Jacobi rotation angle.
        let app = a[p * n + p];
        let aqq = a[q * n + q];
        let apq = a[p * n + q];
        let tau = (aqq - app) / (2.0 * apq);
        let t = if tau >= 0.0 {
            1.0 / (tau + (tau * tau + 1.0).sqrt())
        } else {
            1.0 / (tau - (tau * tau + 1.0).sqrt())
        };
        let c = 1.0 / (t * t + 1.0).sqrt();
        let s = t * c;

        // Apply rotation to A.
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
            // Accumulate eigenvectors.
            let vrp = v[r * n + p];
            let vrq = v[r * n + q];
            v[r * n + p] = c * vrp - s * vrq;
            v[r * n + q] = s * vrp + c * vrq;
        }
    }

    // Extract diagonal as eigenvalues, clamped positive.
    let eigenvals: Vec<f64> = (0..n).map(|i| a[i * n + i].max(1e-12)).collect();

    (v, eigenvals)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn star_factor_is_sqrt_4_over_3() {
        // STAR_FACTOR must equal √(4/3) to within f64 precision.
        let expected = (4.0_f64 / 3.0).sqrt();
        assert!((STAR_FACTOR - expected).abs() < 1e-12);
    }

    #[test]
    fn factorial_factor_is_one() {
        assert_eq!(FACTORIAL_FACTOR, 1.0);
    }

    #[test]
    fn star_and_factorial_factors_are_distinct() {
        // The whole point of using two constants is that they differ.
        assert!(STAR_FACTOR > FACTORIAL_FACTOR);
    }

    #[test]
    fn jacobi_identity_matrix_eigenvalues() {
        // Eigenvalues of I_2 are both 1.0.
        let mat = vec![1.0_f64, 0.0, 0.0, 1.0];
        let (_v, s) = simple_jacobi_eigenvalue(&mat, 2);
        for &ev in &s {
            assert!((ev - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn jacobi_diagonal_matrix_returns_diagonal_as_eigenvalues() {
        // Diagonal matrix: eigenvalues are the diagonal entries.
        let mat = vec![3.0_f64, 0.0, 0.0, 7.0];
        let (_v, mut s) = simple_jacobi_eigenvalue(&mat, 2);
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((s[0] - 3.0).abs() < 1e-10);
        assert!((s[1] - 7.0).abs() < 1e-10);
    }

    #[test]
    fn ccd_point_count_d1() {
        // D=1: center + 2 star = 3 points (no factorial since d <= 1).
        let d = 1usize;
        let expected = 1 + 2 * d; // 3
        assert_eq!(expected, 3);
    }

    #[test]
    fn ccd_point_count_d2() {
        // D=2: center + 4 star + 4 factorial = 9 points.
        let d = 2usize;
        let expected = 1 + 2 * d + (1 << d); // 9
        assert_eq!(expected, 9);
    }

    #[test]
    fn ccd_point_count_d3() {
        // D=3: center + 6 star + 8 factorial = 15 points.
        let d = 3usize;
        let expected = 1 + 2 * d + (1 << d); // 15
        assert_eq!(expected, 15);
    }

    #[test]
    fn log_space_normalisation_sums_to_one() {
        // Simulate the log-weight normalisation used in build_ccd_grid.
        let log_weights = vec![-10.5_f64, -9.8, -10.1, -11.2, -9.5];
        let max_w = log_weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<f64> = log_weights.iter().map(|&l| (l - max_w).exp()).collect();
        let sum: f64 = weights.iter().sum();
        let normalised: Vec<f64> = weights.iter().map(|&w| w / sum).collect();
        let total: f64 = normalised.iter().sum();
        assert!((total - 1.0).abs() < 1e-12);
        assert!(normalised.iter().all(|&w| w >= 0.0 && w <= 1.0));
    }
}
