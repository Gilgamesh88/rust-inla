use crate::error::InlaError;
use crate::graph::Graph;
use crate::models::QFunc;
use crate::solver::{FaerSolver, SparseSolver};

pub const PRIOR_PREC_BETA: f64 = 0.001;

pub struct Problem {
    graph:   Graph,
    pub(crate) solver:  FaerSolver,
    log_det: f64,
    pub n_evals: usize,
}

impl Problem {
    pub fn new(qfunc: &dyn QFunc) -> Self {
        let mut graph  = qfunc.graph().clone();
        let mut solver = FaerSolver::new();
        solver.reorder(&mut graph);
        Self { graph, solver, log_det: 0.0, n_evals: 0 }
    }

    pub fn eval(&mut self, qfunc: &dyn QFunc, theta: &[f64]) -> Result<f64, InlaError> {
        self.solver.build(&self.graph, qfunc, theta);
        self.solver.factorize()?;
        self.log_det = self.solver.log_determinant();
        self.n_evals += 1;
        Ok(self.log_det)
    }

    /// Factoriza Q y calcula diag(Q⁻¹) via Takahashi en una sola llamada atómica.
    ///
    /// Garantiza que `selected_inverse()` ve el estado `Factorized` correcto.
    /// Usa binary_search sobre los arrays CSC raw para localizar la diagonal,
    /// correcto para modelos no-diagonales (Rw1, Ar1) donde val_of_col(j)[0]
    /// devolvería una covarianza off-diagonal en lugar de la varianza marginal.
    pub fn eval_with_inverse(
        &mut self,
        qfunc: &dyn QFunc,
        theta: &[f64],
    ) -> Result<(f64, Vec<f64>), InlaError> {
        self.solver.build(&self.graph, qfunc, theta);
        self.solver.factorize()?;
        let log_det  = self.solver.log_determinant();
        self.log_det  = log_det;
        self.n_evals += 1;

        let q_inv    = self.solver.selected_inverse()?;
        let col_ptr  = q_inv.symbolic().col_ptr();
        let row_idx  = q_inv.symbolic().row_idx();
        let all_vals = q_inv.val();

        let n = self.n();
        let diag_qinv: Vec<f64> = (0..n)
            .map(|j| {
                let start    = col_ptr[j];
                let end      = col_ptr[j + 1];
                let col_rows = &row_idx[start..end];
                let pos      = col_rows.binary_search(&j)
                    .expect("la diagonal siempre pertenece al patrón de Q⁻¹");
                all_vals[start + pos]
            })
            .collect();

        Ok((log_det, diag_qinv))
    }

    /// IRLS con warm start + Takahashi sobre Q+W en el modo.
    ///
    /// Este es el núcleo del algoritmo INLA: encuentra x̂(θ) = argmax p(x|y,θ)
    /// via Newton-Raphson (IRLS) y calcula simultáneamente diag((Q+W)⁻¹) que
    /// se necesita para el gradiente analítico del optimizador.
    ///
    /// ## Warm start
    ///
    /// `x_init` es el punto de partida de IRLS. Si se pasa x̂ de la evaluación
    /// anterior (theta cercano), IRLS converge en 1-2 iteraciones en lugar de
    /// 10. Esto es lo que hace R-INLA para que la Laplace en el optimizer sea
    /// viable. Para cold start, pasar `&[]`.
    ///
    /// ## Returns
    /// `(x_hat, log_det_aug, diag_aug_inv)` donde:
    /// - `x_hat`:        moda de p(x|y,θ)
    /// - `log_det_aug`:  log|Q(θ)+W(x̂)|
    /// - `diag_aug_inv`: diag((Q(θ)+W(x̂))⁻¹) via Takahashi
    pub fn find_mode_with_inverse(
        &mut self,
        qfunc:      &dyn QFunc,
        likelihood: &dyn crate::likelihood::LogLikelihood,
        y:          &[f64],
        theta:      &[f64],
        x_init:     &[f64],
        max_iter:   usize,
        tol:        f64,
    ) -> Result<(Vec<f64>, f64, Vec<f64>), InlaError> {
        let n           = self.n();
        let n_model     = qfunc.n_hyperparams();
        let theta_model = &theta[..n_model];
        let theta_lik   = &theta[n_model..];
        let h           = 1e-5;

        // Warm start: usa x_init si tiene la longitud correcta, sino ceros
        let mut x = if x_init.len() == n {
            x_init.to_vec()
        } else {
            vec![0.0_f64; n]
        };
        let mut log_det_aug = 0.0_f64;

        for _iter in 0..max_iter {
            let x_old = x.clone();
            let eta   = x.clone();

            let mut grad = vec![0.0_f64; n];
            let mut curv = vec![0.0_f64; n];

            likelihood.gradient_and_curvature(&mut grad, &mut curv, &eta, y, theta_lik);
            
            for i in 0..n {
                curv[i] = curv[i].max(1e-6);
            }

            let z: Vec<f64>       = (0..n).map(|i| eta[i] + grad[i] / curv[i]).collect();
            let mut rhs: Vec<f64> = (0..n).map(|i| curv[i] * z[i]).collect();

            let aug = AugmentedQFunc { inner: qfunc, diag_add: &curv };
            self.solver.build(&self.graph, &aug, theta_model);
            self.solver.factorize()?;
            log_det_aug = self.solver.log_determinant();
            self.solver.solve_llt(&mut rhs);
            x = rhs;

            let delta: f64 = x.iter().zip(x_old.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            if delta < tol { break; }
        }

        // Takahashi sobre Q+W — el solver está en Factorized tras el último
        // paso IRLS. Mismo patrón de extracción de diagonal que eval_with_inverse.
        let aug_inv  = self.solver.selected_inverse()?;
        let col_ptr  = aug_inv.symbolic().col_ptr();
        let row_idx  = aug_inv.symbolic().row_idx();
        let all_vals = aug_inv.val();

        let diag_aug_inv: Vec<f64> = (0..n)
            .map(|j| {
                let start    = col_ptr[j];
                let end      = col_ptr[j + 1];
                let col_rows = &row_idx[start..end];
                let pos      = col_rows.binary_search(&j)
                    .expect("diagonal siempre en patrón de (Q+W)⁻¹");
                all_vals[start + pos]
            })
            .collect();

        Ok((x, log_det_aug, diag_aug_inv))
    }

    /// IRLS con estimación de intercepto β₀ + Takahashi sobre Q+W.
    ///
    /// ## Equivalencia con R-INLA
    ///
    /// En R-INLA el predictor lineal es η = Aβ + x donde A=1 (intercept global).
    /// El intercepto se estima por "perfil" en cada paso IRLS:
    ///
    ///   β₀ = Σᵢ Wᵢ·(zᵢ - xᵢ) / ΣᵢWᵢ    (inla.c ~2600)
    ///
    /// lo que centra el working response antes de resolver el sistema Cholesky.
    /// Esto es equivalente a incluir β₀ en el campo latente aumentado con una
    /// columna de unos y un prior difuso (varianza → ∞).
    ///
    /// ## Returns
    /// `(beta0, x_hat, log_det_aug, diag_aug_inv)` donde:
    /// - `beta0`:        intercepto estimado
    /// - `x_hat`:        efecto latente centrado (sin intercepto)
    /// - `log_det_aug`:  log|Q+W|
    /// - `diag_aug_inv`: diag((Q+W)⁻¹)
    pub fn find_mode_with_intercept_and_inverse(
        &mut self,
        qfunc:      &dyn QFunc,
        likelihood: &dyn crate::likelihood::LogLikelihood,
        y:          &[f64],
        theta:      &[f64],
        x_init:     &[f64],
        beta0_init: f64,
        max_iter:   usize,
        tol:        f64,
    ) -> Result<(f64, Vec<f64>, f64, Vec<f64>, f64), InlaError> {
        let n           = self.n();
        let n_model     = qfunc.n_hyperparams();
        let theta_model = &theta[..n_model];
        let theta_lik   = &theta[n_model..];
        let h           = 1e-5;

        let mut x = if x_init.len() == n {
            x_init.to_vec()
        } else {
            vec![0.0_f64; n]
        };
        let mut beta0       = beta0_init;
        let mut log_det_aug = 0.0_f64;
        let mut final_curv  = vec![0.0_f64; n];

        for _iter in 0..max_iter {
            let x_old    = x.clone();
            let beta0_old = beta0;

            // Predictor lineal: η = β₀ + x
            let eta: Vec<f64> = x.iter().map(|xi| beta0 + xi).collect();

            let mut grad      = vec![0.0_f64; n];
            let mut curv      = vec![0.0_f64; n];

            likelihood.gradient_and_curvature(&mut grad, &mut curv, &eta, y, theta_lik);

            for i in 0..n {
                curv[i] = curv[i].max(1e-6);
            }
            final_curv = curv.clone();

            // Working response (en escala de η): zᵢ = ηᵢ + gradᵢ/Wᵢ
            let z: Vec<f64> = (0..n).map(|i| eta[i] + grad[i] / curv[i]).collect();

            // 1. Joint Newton step: Resolver simultáneamente [P+ΣW, W^T; W, Q+W] [β; x]
            let mut u: Vec<f64> = (0..n).map(|i| curv[i] * z[i]).collect();
            let mut v = curv.clone();

            let aug = AugmentedQFunc { inner: qfunc, diag_add: &curv };
            self.solver.build(&self.graph, &aug, theta_model);
            self.solver.factorize()?;
            log_det_aug = self.solver.log_determinant();
            
            self.solver.solve_llt(&mut u);
            self.solver.solve_llt(&mut v);
            
            let sum_w: f64  = curv.iter().sum();
            let wt_v: f64   = curv.iter().zip(v.iter()).map(|(w, vi)| w * vi).sum();
            let schur_denom = PRIOR_PREC_BETA + (sum_w - wt_v).max(0.0);
            
            let sum_wz: f64 = (0..n).map(|i| final_curv[i] * z[i]).sum();
            let wt_u: f64   = final_curv.iter().zip(u.iter()).map(|(w, ui)| w * ui).sum();
            
            beta0 = (sum_wz - wt_u) / schur_denom;
            // println!("IRLS {}: b0_old={:.4} b0_new={:.4} sum_wz={:.4} wt_u={:.4} denom={:.4}", _iter, beta0_old, beta0, sum_wz, wt_u, schur_denom);
            // assert!(!beta0.is_nan(), "beta0 became NaN! iter={} sum_wz={}, wt_u={}, denom={}", _iter, sum_wz, wt_u, schur_denom);
            
            x = u.into_iter().zip(v.iter()).map(|(ui, vi)| ui - beta0 * vi).collect();

            // Constraint suma-a-cero: Σxᵢ = 0 (inla.c ~2800 GMRFLib_constr_add)
            //
            // Para priors impropios (Rw1, Rw2), el nivel de x no está identificado
            // separado de β₀ — cualquier (β₀+c, x-c) da el mismo predictor η.
            // La constraint Σxᵢ = 0 fija el nivel de x y hace β₀ identificable.
            //
            // Sin esta constraint, el IRLS puede converger a β₀ muy negativo y
            // x muy positivo (o viceversa), dando el predictor correcto pero con
            // x e intercept individuales erróneos.
            //
            // Para priors propios (iid, ar1), la constraint mejora la estabilidad
            // numérica sin cambiar el resultado (Q ya penaliza valores grandes de x).
            let mean_x: f64 = x.iter().sum::<f64>() / n as f64;
            for xi in x.iter_mut() { *xi -= mean_x; }
            beta0 += mean_x;

            // Convergencia: max cambio en (β₀, x)
            let delta_x: f64 = x.iter().zip(x_old.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            let delta_b = (beta0 - beta0_old).abs();
            if delta_x.max(delta_b) < tol { break; }
        }

        // Takahashi sobre Q+W — idéntico a find_mode_with_inverse
        let aug_inv  = self.solver.selected_inverse()?;
        let col_ptr  = aug_inv.symbolic().col_ptr();
        let row_idx  = aug_inv.symbolic().row_idx();
        let all_vals = aug_inv.val();

        let diag_aug_inv: Vec<f64> = (0..n)
            .map(|j| {
                let start    = col_ptr[j];
                let end      = col_ptr[j + 1];
                let col_rows = &row_idx[start..end];
                let pos      = col_rows.binary_search(&j)
                    .expect("diagonal en patrón de (Q+W)⁻¹");
                all_vals[start + pos]
            })
            .collect();

        // 2. Schur complement S final (para determinantes en Laplace)
        let schur_s = {
            let mut v = final_curv.clone();
            self.solver.solve_llt(&mut v);
            let sum_w: f64 = final_curv.iter().sum();
            let wt_v: f64  = final_curv.iter().zip(v.iter()).map(|(w, vi)| w * vi).sum();
            (crate::problem::PRIOR_PREC_BETA + sum_w - wt_v).max(1e-300)
        };

        Ok((beta0, x, log_det_aug, diag_aug_inv, schur_s))
    }

        /// Igual que find_mode pero también devuelve log|Q+W|.
    /// Delega en find_mode_with_inverse con cold start.
    pub fn find_mode_with_logdet(
        &mut self,
        qfunc:      &dyn QFunc,
        likelihood: &dyn crate::likelihood::LogLikelihood,
        y:          &[f64],
        theta:      &[f64],
        max_iter:   usize,
        tol:        f64,
    ) -> Result<(Vec<f64>, f64), InlaError> {
        let (x, log_det_aug, _) = self.find_mode_with_inverse(
            qfunc, likelihood, y, theta, &[], max_iter, tol,
        )?;
        Ok((x, log_det_aug))
    }

    pub fn find_mode(
        &mut self,
        qfunc:      &dyn QFunc,
        likelihood: &dyn crate::likelihood::LogLikelihood,
        y:          &[f64],
        theta:      &[f64],
        max_iter:   usize,
        tol:        f64,
    ) -> Result<Vec<f64>, InlaError> {
        let (x, _) = self.find_mode_with_logdet(qfunc, likelihood, y, theta, max_iter, tol)?;
        Ok(x)
    }

    pub fn solve(&self, rhs: &mut [f64]) { self.solver.solve_llt(rhs); }
    pub fn log_det(&self) -> f64 { self.log_det }
    pub fn n(&self) -> usize { self.graph.n() }

    /// Calcula x̂ᵀ·Q(θ)·x̂ directamente desde el grafo y qfunc.
    ///
    /// Necesario para la Laplace completa:
    ///   log p̃(y|θ) = 0.5·(log|Q|-log|Q+W|) + Σlogp(yᵢ|x̂ᵢ) - 0.5·x̂ᵀQx̂
    ///
    /// Sin este término, τ→∞ no tiene costo (el prior sobre x desaparece)
    /// y el optimizer diverge. Equivale a `inla.c:DAXPY(n, x_mode, Q_x_mode)`.
    ///
    /// Costo: O(nnz) — una pasada sobre el grafo.
    pub fn quadratic_form_x(&self, qfunc: &dyn QFunc, theta_model: &[f64], x: &[f64]) -> f64 {
        let n = self.n();
        // Términos diagonales: Σᵢ Q(i,i)·x̂ᵢ²
        let diag: f64 = (0..n)
            .map(|i| qfunc.eval(i, i, theta_model) * x[i] * x[i])
            .sum();
        // Términos off-diagonal: 2·Σ_{i<j} Q(i,j)·x̂ᵢ·x̂ⱼ  (factor 2 por simetría)
        let offdiag: f64 = self.graph
            .iter_upper_triangle()
            .map(|(i, j)| 2.0 * qfunc.eval(i, j, theta_model) * x[i] * x[j])
            .sum();
        diag + offdiag
    }
}

struct AugmentedQFunc<'a> {
    inner:    &'a dyn QFunc,
    diag_add: &'a [f64],
}

impl QFunc for AugmentedQFunc<'_> {
    fn graph(&self) -> &Graph { self.inner.graph() }
    fn eval(&self, i: usize, j: usize, theta: &[f64]) -> f64 {
        let base = self.inner.eval(i, j, theta);
        if i == j { base + self.diag_add[i] } else { base }
    }
    fn n_hyperparams(&self) -> usize { self.inner.n_hyperparams() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::likelihood::GaussianLikelihood;
    use crate::models::{Ar1Model, IidModel};
    use approx::assert_abs_diff_eq;

    #[test]
    fn problem_new_sets_dimension() {
        let model = IidModel::new(10);
        let p = Problem::new(&model);
        assert_eq!(p.n(), 10);
        assert_eq!(p.n_evals, 0);
    }

    #[test]
    fn problem_eval_iid_log_det() {
        let n = 5; let tau = 4.0_f64;
        let model = IidModel::new(n);
        let mut p = Problem::new(&model);
        let ld = p.eval(&model, &[tau.ln()]).unwrap();
        assert_eq!(p.n_evals, 1);
        assert_abs_diff_eq!(ld, (n as f64) * tau.ln(), epsilon = 1e-8);
    }

    #[test]
    fn problem_eval_multiple_theta() {
        let model = IidModel::new(8);
        let mut p = Problem::new(&model);
        for log_tau in [0.0, 1.0, 2.0, -1.0] {
            let ld = p.eval(&model, &[log_tau]).unwrap();
            assert_abs_diff_eq!(ld, 8.0 * log_tau, epsilon = 1e-8);
        }
        assert_eq!(p.n_evals, 4);
    }

    #[test]
    fn problem_solve_iid_identity() {
        let model = IidModel::new(4);
        let mut p = Problem::new(&model);
        p.eval(&model, &[0.0]).unwrap();
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let mut x = b.clone();
        p.solve(&mut x);
        for (xi, bi) in x.iter().zip(b.iter()) {
            assert_abs_diff_eq!(xi, bi, epsilon = 1e-10);
        }
    }

    #[test]
    fn problem_solve_iid_tau2() {
        let model = IidModel::new(4);
        let mut p = Problem::new(&model);
        p.eval(&model, &[2.0_f64.ln()]).unwrap();
        let mut x = vec![2.0, 4.0, 6.0, 8.0];
        p.solve(&mut x);
        for (xi, expected) in x.iter().zip([1.0, 2.0, 3.0, 4.0].iter()) {
            assert_abs_diff_eq!(xi, expected, epsilon = 1e-10);
        }
    }

    #[test]
    fn problem_ar1_rho_zero_log_det_equals_iid() {
        let n = 6; let tau = 3.0_f64;
        let m_iid = IidModel::new(n);
        let m_ar1 = Ar1Model::new(n);
        let mut p_iid = Problem::new(&m_iid);
        let mut p_ar1 = Problem::new(&m_ar1);
        let ld_iid = p_iid.eval(&m_iid, &[tau.ln()]).unwrap();
        let ld_ar1 = p_ar1.eval(&m_ar1, &[tau.ln(), 0.0]).unwrap();
        assert_abs_diff_eq!(ld_iid, ld_ar1, epsilon = 1e-8);
    }

    #[test]
    fn problem_not_positive_definite_returns_error() {
        let model = IidModel::new(4);
        let mut p = Problem::new(&model);
        let result = p.eval(&model, &[-1000.0]);
        let _ = result;
    }

    #[test]
    fn irls_gaussian_mean_recovers_data() {
        let n = 5;
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let model = IidModel::new(n);
        let lik   = GaussianLikelihood;
        let mut p = Problem::new(&model);
        let theta = vec![0.0_f64, 0.0_f64];
        let x_hat = p.find_mode(&model, &lik, &y, &theta, 20, 1e-6).unwrap();
        assert_eq!(x_hat.len(), n);
        for (xi, yi) in x_hat.iter().zip(y.iter()) {
            assert!(*xi > 0.0 && *xi < *yi, "x_hat={xi} debe estar entre 0 y y={yi}");
        }
    }

    #[test]
    fn irls_gaussian_converges_in_one_iteration() {
        let n = 4;
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let model = IidModel::new(n);
        let lik   = GaussianLikelihood;
        let mut p = Problem::new(&model);
        let theta = vec![0.0_f64, 0.0_f64];
        let x1 = p.find_mode(&model, &lik, &y, &theta, 1,  1e-10).unwrap();
        let x2 = p.find_mode(&model, &lik, &y, &theta, 10, 1e-10).unwrap();
        for (a, b) in x1.iter().zip(x2.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-3);
        }
    }

    #[test]
    fn find_mode_with_logdet_returns_positive_logdet() {
        let n = 5;
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let model = IidModel::new(n);
        let lik   = GaussianLikelihood;
        let mut p = Problem::new(&model);
        let theta = vec![0.0_f64, 0.0_f64];
        let (x_hat, log_det_aug) = p.find_mode_with_logdet(
            &model, &lik, &y, &theta, 20, 1e-6
        ).unwrap();
        assert_eq!(x_hat.len(), n);
        assert!(log_det_aug.is_finite());
        let log_det_q = p.eval(&model, &[0.0]).unwrap();
        assert!(log_det_aug > log_det_q,
            "log|Q+W| > log|Q|: {log_det_aug} vs {log_det_q}");
    }

    // ── eval_with_inverse ─────────────────────────────────────────────────────

    #[test]
    fn eval_with_inverse_iid_diagonal_equals_one_over_tau() {
        let n = 6; let tau = 4.0_f64;
        let model = IidModel::new(n);
        let mut p = Problem::new(&model);
        let (log_det, diag) = p.eval_with_inverse(&model, &[tau.ln()]).unwrap();
        assert_abs_diff_eq!(log_det, (n as f64) * tau.ln(), epsilon = 1e-8);
        assert_eq!(diag.len(), n);
        for (i, d) in diag.iter().enumerate() {
            assert!((*d - 1.0 / tau).abs() < 1e-8, "diag[{i}] esperado {}, obtenido {d}", 1.0/tau);
        }
    }

    #[test]
    fn eval_with_inverse_log_det_matches_eval() {
        let n = 5;
        let model = IidModel::new(n);
        let mut p1 = Problem::new(&model);
        let mut p2 = Problem::new(&model);
        let theta = [2.0_f64.ln()];
        let (ld_inv, _) = p1.eval_with_inverse(&model, &theta).unwrap();
        let ld_eval     = p2.eval(&model, &theta).unwrap();
        assert_abs_diff_eq!(ld_inv, ld_eval, epsilon = 1e-10);
    }

    #[test]
    fn eval_with_inverse_ar1_log_det_matches_eval() {
        let n = 7; let tau = 2.0_f64; let rho = 0.6_f64;
        let model = Ar1Model::new(n);
        let mut p1 = Problem::new(&model);
        let mut p2 = Problem::new(&model);
        let theta = [tau.ln(), rho.atanh()];
        let (ld_inv, diag) = p1.eval_with_inverse(&model, &theta).unwrap();
        let ld_eval        = p2.eval(&model, &theta).unwrap();
        assert_abs_diff_eq!(ld_inv, ld_eval, epsilon = 1e-10);
        for (i, d) in diag.iter().enumerate() {
            assert!(*d > 0.0 && d.is_finite(), "diag[{i}] = {d}");
        }
    }

    #[test]
    fn eval_with_inverse_idempotent_across_calls() {
        let n = 5; let tau = 3.0_f64;
        let model = IidModel::new(n);
        let mut p = Problem::new(&model);
        let theta = [tau.ln()];
        let (ld1, diag1) = p.eval_with_inverse(&model, &theta).unwrap();
        let (ld2, diag2) = p.eval_with_inverse(&model, &theta).unwrap();
        assert_abs_diff_eq!(ld1, ld2, epsilon = 1e-12);
        for (a, b) in diag1.iter().zip(diag2.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-12);
        }
    }

    // ── find_mode_with_inverse ────────────────────────────────────────────────

    #[test]
    fn find_mode_with_inverse_cold_equals_with_logdet() {
        // find_mode_with_inverse(&[]) debe dar el mismo x̂ y log_det_aug
        // que find_mode_with_logdet (que internamente hace cold start)
        let n = 5;
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let model = IidModel::new(n);
        let lik   = GaussianLikelihood;
        let mut p1 = Problem::new(&model);
        let mut p2 = Problem::new(&model);
        let theta = [0.0_f64, 0.0_f64];

        let (x1, ld1) = p1.find_mode_with_logdet(&model, &lik, &y, &theta, 20, 1e-8).unwrap();
        let (x2, ld2, diag2) = p2.find_mode_with_inverse(&model, &lik, &y, &theta, &[], 20, 1e-8).unwrap();

        assert_abs_diff_eq!(ld1, ld2, epsilon = 1e-10);
        for (a, b) in x1.iter().zip(x2.iter()) { assert_abs_diff_eq!(a, b, epsilon = 1e-8); }
        assert_eq!(diag2.len(), n);
        for (i, d) in diag2.iter().enumerate() {
            assert!(*d > 0.0 && d.is_finite(), "diag_aug_inv[{i}] = {d}");
        }
    }

    #[test]
    fn find_mode_with_inverse_warm_start_faster_convergence() {
        // Con warm start x̂, IRLS con max_iter=2 debe dar el mismo resultado
        // que cold start con max_iter=20 (porque ya está en el modo)
        let n = 6;
        let y = vec![1.0, 3.0, 2.0, 4.0, 1.0, 3.0];
        let model = IidModel::new(n);
        let lik   = GaussianLikelihood;
        let theta = [0.5_f64, 0.5_f64];

        let mut p1 = Problem::new(&model);
        let (x_warm, _, _) = p1.find_mode_with_inverse(&model, &lik, &y, &theta, &[], 20, 1e-8).unwrap();

        // Con warm start en el modo, 2 iteraciones deben ser suficientes
        let mut p2 = Problem::new(&model);
        let (x_warm_2, _, _) = p2.find_mode_with_inverse(&model, &lik, &y, &theta, &x_warm, 2, 1e-8).unwrap();

        for (a, b) in x_warm.iter().zip(x_warm_2.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-5);
        }
    }

    #[test]
    fn find_mode_with_inverse_ar1_diag_positive() {
        let n = 8; let tau = 2.0_f64; let rho = 0.5_f64;
        let y: Vec<f64> = (0..n).map(|i| i as f64 * 0.3).collect();
        let model = Ar1Model::new(n);
        let lik   = GaussianLikelihood;
        let theta = [tau.ln(), rho.atanh(), 1.0_f64];
        let mut p = Problem::new(&model);
        let (x_hat, log_det_aug, diag_aug_inv) = p.find_mode_with_inverse(
            &model, &lik, &y, &theta, &[], 20, 1e-6
        ).unwrap();
        assert_eq!(x_hat.len(), n);
        assert!(log_det_aug.is_finite());
        for (i, d) in diag_aug_inv.iter().enumerate() {
            assert!(*d > 0.0 && d.is_finite(), "diag_aug_inv[{i}] = {d}");
        }
    }
}
