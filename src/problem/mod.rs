use crate::error::InlaError;
use crate::graph::Graph;
use crate::models::QFunc;
use crate::solver::{FaerSolver, SparseSolver};

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

    /// Evaluación atómica: factoriza Q y calcula la inversa seleccionada
    /// en una sola llamada, garantizando que `selected_inverse()` ve siempre
    /// el estado `Factorized` correcto.
    ///
    /// # Returns
    /// `(log_det, diag_qinv)` donde:
    /// - `log_det`   = log|Q(θ)|
    /// - `diag_qinv` = diag(Q⁻¹), vector de longitud n
    ///
    /// # Por qué existe este método
    /// `selected_inverse()` requiere estado `Factorized`. Cuando `build()` y
    /// `factorize()` se invocan desde rutas distintas (p.ej. el bucle del
    /// optimizador), cualquier llamada intermedia a `build()` resetea el
    /// estado a `Built` y `selected_inverse()` falla. Al fusionar las tres
    /// operaciones aquí, el estado nunca puede quedar inconsistente.
    ///
    /// # Nota sobre la diagonal en SpMat simétrica
    /// `selected_inverse()` emite triplets lower+upper → la SpMat resultante
    /// es simétrica. En la columna j, el elemento diagonal (fila == j) NO es
    /// necesariamente el primero: puede estar precedido por filas i < j.
    /// Se usa `binary_search` sobre `row_idx_of_col(j)`, que siempre está
    /// ordenado ascendentemente por invariante CSC.
    pub fn eval_with_inverse(
        &mut self,
        qfunc: &dyn QFunc,
        theta: &[f64],
    ) -> Result<(f64, Vec<f64>), InlaError> {
        // Paso 1: build + factorize
        self.solver.build(&self.graph, qfunc, theta);
        self.solver.factorize()?;
        let log_det  = self.solver.log_determinant();
        self.log_det  = log_det;
        self.n_evals += 1;

        // Paso 2: Takahashi — el solver está en Factorized, garantizado
        // porque build+factorize y selected_inverse viven en la misma llamada.
        let q_inv = self.solver.selected_inverse()?;

        // Paso 3: extraer diagonal usando los arrays CSC raw.
        //
        // row_idx_of_col() en faer 0.24 devuelve un iterador, no un slice,
        // por lo que binary_search no está disponible directamente en él.
        // Accedemos a col_ptr y row_idx del simbólico (igual que en
        // faer_solver.rs) para obtener slices sobre los que sí funciona
        // binary_search. Esto es correcto para cualquier modelo: en la SpMat
        // simétrica que devuelve selected_inverse(), el elemento diagonal (j,j)
        // puede NO ser el primero de la columna (hay entradas i < j antes),
        // por lo que val_of_col(j)[0] sería incorrecto para Rw1/Ar1.
        let n        = self.n();
        let col_ptr  = q_inv.symbolic().col_ptr();
        let row_idx  = q_inv.symbolic().row_idx();
        let all_vals = q_inv.val();

        let diag_qinv: Vec<f64> = (0..n)
            .map(|j| {
                let start    = col_ptr[j];
                let end      = col_ptr[j + 1];
                let col_rows = &row_idx[start..end];
                // col_rows está ordenado ascendentemente (invariante CSC).
                let pos = col_rows
                    .binary_search(&j)
                    .expect("la diagonal siempre pertenece al patrón de Q⁻¹");
                all_vals[start + pos]
            })
            .collect();

        Ok((log_det, diag_qinv))
    }

    pub fn solve(&self, rhs: &mut [f64]) {
        self.solver.solve_llt(rhs);
    }

    pub fn log_det(&self) -> f64 { self.log_det }
    pub fn n(&self) -> usize { self.graph.n() }

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

    /// Igual que find_mode pero tambien devuelve log|Q+W| de la ultima iteracion.
    /// Necesario para la aproximacion de Laplace completa en el optimizador.
    pub fn find_mode_with_logdet(
        &mut self,
        qfunc:      &dyn QFunc,
        likelihood: &dyn crate::likelihood::LogLikelihood,
        y:          &[f64],
        theta:      &[f64],
        max_iter:   usize,
        tol:        f64,
    ) -> Result<(Vec<f64>, f64), InlaError> {
        let n           = self.n();
        let n_model     = qfunc.n_hyperparams();
        let theta_model = &theta[..n_model];
        let theta_lik   = &theta[n_model..];
        let h           = 1e-5;
        let mut x       = vec![0.0_f64; n];
        let mut log_det_aug = 0.0_f64;

        for _iter in 0..max_iter {
            let x_old = x.clone();
            let eta   = x.clone();

            let mut ll_center = vec![0.0_f64; n];
            let mut ll_plus   = vec![0.0_f64; n];
            let mut ll_minus  = vec![0.0_f64; n];
            let mut grad      = vec![0.0_f64; n];
            let mut curv      = vec![0.0_f64; n];

            likelihood.evaluate(&mut ll_center, &eta, y, theta_lik);

            for i in 0..n {
                let mut eta_plus  = eta.clone();
                let mut eta_minus = eta.clone();
                eta_plus[i]  += h;
                eta_minus[i] -= h;
                likelihood.evaluate(&mut ll_plus,  &eta_plus,  y, theta_lik);
                likelihood.evaluate(&mut ll_minus, &eta_minus, y, theta_lik);
                grad[i] = (ll_plus[i] - ll_minus[i]) / (2.0 * h);
                curv[i] = (-(ll_plus[i] - 2.0 * ll_center[i] + ll_minus[i])
                           / (h * h)).max(1e-6);
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

        Ok((x, log_det_aug))
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
        let n = 5;
        let tau = 4.0_f64;
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
        let n = 6;
        let tau = 3.0_f64;
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
            assert!(*xi > 0.0 && *xi < *yi,
                "x_hat={xi} debe estar entre 0 y y={yi}");
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
        assert!(log_det_aug.is_finite(), "log_det_aug debe ser finito");
        let log_det_q = p.eval(&model, &[0.0]).unwrap();
        assert!(log_det_aug > log_det_q,
            "log|Q+W| debe ser mayor que log|Q|: {log_det_aug} vs {log_det_q}");
    }

    // ── Tests de eval_with_inverse ────────────────────────────────────────────

    #[test]
    fn eval_with_inverse_iid_diagonal_equals_one_over_tau() {
        // Q = tau*I  →  Q⁻¹ = (1/tau)*I
        let n   = 6;
        let tau = 4.0_f64;
        let model = IidModel::new(n);
        let mut p = Problem::new(&model);
        let (log_det, diag) = p.eval_with_inverse(&model, &[tau.ln()]).unwrap();

        assert_abs_diff_eq!(log_det, (n as f64) * tau.ln(), epsilon = 1e-8);
        assert_eq!(diag.len(), n);
        for (i, d) in diag.iter().enumerate() {
            assert!(
                (*d - 1.0 / tau).abs() < 1e-8,
                "diag[{i}] esperado {}, obtenido {d}", 1.0 / tau
            );
        }
    }

    #[test]
    fn eval_with_inverse_log_det_matches_eval() {
        // log_det debe ser idéntico al de eval() con los mismos theta
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
    fn eval_with_inverse_diag_all_positive() {
        let n = 8;
        let model = IidModel::new(n);
        let mut p = Problem::new(&model);
        let (_, diag) = p.eval_with_inverse(&model, &[1.5_f64.ln()]).unwrap();
        for (i, d) in diag.iter().enumerate() {
            assert!(*d > 0.0, "diag[{i}] = {d} debe ser positivo");
        }
    }

    #[test]
    fn eval_with_inverse_ar1_log_det_matches_eval() {
        // Ar1 tiene fill-in no-trivial en L — verifica binary_search y log_det
        let n   = 7;
        let tau = 2.0_f64;
        let rho = 0.6_f64;
        let model = Ar1Model::new(n);
        let mut p1 = Problem::new(&model);
        let mut p2 = Problem::new(&model);
        let theta = [tau.ln(), rho.atanh()];

        let (ld_inv, diag) = p1.eval_with_inverse(&model, &theta).unwrap();
        let ld_eval        = p2.eval(&model, &theta).unwrap();

        assert_abs_diff_eq!(ld_inv, ld_eval, epsilon = 1e-10);
        assert_eq!(diag.len(), n);
        for (i, d) in diag.iter().enumerate() {
            assert!(*d > 0.0 && d.is_finite(),
                "diag[{i}] = {d} debe ser positivo y finito");
        }
    }

    #[test]
    fn eval_with_inverse_n_evals_increments() {
        let model = IidModel::new(4);
        let mut p = Problem::new(&model);
        assert_eq!(p.n_evals, 0);
        p.eval_with_inverse(&model, &[0.0]).unwrap();
        assert_eq!(p.n_evals, 1);
        p.eval_with_inverse(&model, &[1.0]).unwrap();
        assert_eq!(p.n_evals, 2);
    }

    #[test]
    fn eval_with_inverse_idempotent_across_calls() {
        // Dos llamadas seguidas con los mismos theta deben dar resultados idénticos.
        // Verifica que el estado interno se resetea correctamente en cada llamada.
        let n   = 5;
        let tau = 3.0_f64;
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
}
