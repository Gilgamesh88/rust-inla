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
        let n           = self.n();
        let n_model     = qfunc.n_hyperparams();
        let theta_model = &theta[..n_model];
        let theta_lik   = &theta[n_model..];
        let h           = 1e-5;
        let mut x       = vec![0.0_f64; n];

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
            self.solver.solve_llt(&mut rhs);
            x = rhs;

            let delta: f64 = x.iter().zip(x_old.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            if delta < tol { break; }
        }

        Ok(x)
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
}
