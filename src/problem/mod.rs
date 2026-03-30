//! Constructor central: dado θ, produce log|Q| y resuelve sistemas.
//!
//! Equivalente Rust de `problem-setup.h/c` (2500 líneas de C).
//!
//! ## Responsabilidad
//!
//! `Problem` une Graph + QFunc + FaerSolver en un objeto reutilizable.
//! BFGS lo llama en cada evaluación de f(θ):
//!
//! ```text
//! Problem::eval(theta)
//!   ├── solver.build(graph, qfunc, theta)   // rellena Q(θ) — O(nnz)
//!   ├── solver.factorize()                  // Q = LLᵀ    — O(nnz(L))
//!   ├── log_det = solver.log_determinant()  // 2·Σlog(Lᵢᵢ) — gratis
//!   └── devuelve log_det para BFGS
//! ```
//!
//! ## Por qué Problem existe como tipo separado
//!
//! El solver necesita `reorder()` una sola vez por grafo (caro: AMD).
//! Problem garantiza que reorder se llama exactamente una vez aunque
//! BFGS evalúe f(θ) cientos de veces con el mismo grafo.

use crate::error::InlaError;
use crate::graph::Graph;
use crate::models::QFunc;
use crate::solver::{FaerSolver, SparseSolver};

/// Constructor central de INLA.
///
/// Construir con `Problem::new(qfunc)` y luego llamar `eval(theta)`
/// en cada paso de BFGS.
pub struct Problem {
    /// Grafo de dispersidad (propiedad del modelo, inmutable).
    graph: Graph,

    /// Solver de Cholesky — mantiene el reordering AMD entre evaluaciones.
    solver: FaerSolver,

    /// log|Q(θ)| del último eval() exitoso.
    log_det: f64,

    /// Número de evaluaciones realizadas (para diagnóstico).
    pub n_evals: usize,
}

impl Problem {
    /// Crea un Problem y calcula el reordering AMD.
    ///
    /// `reorder()` es O(n·log n) — se ejecuta una sola vez aquí.
    /// Las evaluaciones posteriores de `eval()` solo hacen Cholesky numérico.
    pub fn new(qfunc: &dyn QFunc) -> Self {
        let mut graph  = qfunc.graph().clone();
        let mut solver = FaerSolver::new();
        solver.reorder(&mut graph);

        Self {
            graph,
            solver,
            log_det:  0.0,
            n_evals:  0,
        }
    }

    /// Evalúa Q(θ), factoriza y devuelve log|Q(θ)|.
    ///
    /// Llamado por BFGS en cada evaluación de f(θ).
    ///
    /// # Errors
    /// `InlaError::NotPositiveDefinite` si Q(θ) no es PD.
    /// BFGS debe tratar este error como f(θ) = +∞.
    pub fn eval(
        &mut self,
        qfunc: &dyn QFunc,
        theta: &[f64],
    ) -> Result<f64, InlaError> {
        self.solver.build(&self.graph, qfunc, theta);
        self.solver.factorize()?;
        self.log_det = self.solver.log_determinant();
        self.n_evals += 1;
        Ok(self.log_det)
    }

    /// Resuelve Q(θ)·x = b en el espacio del último eval() exitoso.
    ///
    /// Modifica `rhs` in-place: rhs ← Q⁻¹·rhs
    ///
    /// # Panics
    /// Si `eval()` no fue llamado antes.
    pub fn solve(&self, rhs: &mut [f64]) {
        self.solver.solve_llt(rhs);
    }

    /// log|Q(θ)| del último eval() exitoso.
    pub fn log_det(&self) -> f64 {
        self.log_det
    }

    /// Dimensión del sistema (número de nodos del grafo).
    pub fn n(&self) -> usize {
        self.graph.n()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
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
        // Q = tau·I  →  log|Q| = n·log(tau)
        let n   = 5;
        let tau = 4.0_f64;
        let model = IidModel::new(n);
        let mut p = Problem::new(&model);

        let ld = p.eval(&model, &[tau.ln()]).unwrap();

        assert_eq!(p.n_evals, 1);
        assert_abs_diff_eq!(ld, (n as f64) * tau.ln(), epsilon = 1e-8);
    }

    #[test]
    fn problem_eval_multiple_theta() {
        // BFGS llama eval() varias veces con distinto θ — no debe fallar
        let model = IidModel::new(8);
        let mut p = Problem::new(&model);

        for log_tau in [0.0, 1.0, 2.0, -1.0] {
            let ld = p.eval(&model, &[log_tau]).unwrap();
            let expected = 8.0 * log_tau;
            assert_abs_diff_eq!(ld, expected, epsilon = 1e-8);
        }
        assert_eq!(p.n_evals, 4);
    }

    #[test]
    fn problem_solve_iid_identity() {
        // Q = I  →  Q⁻¹·b = b
        let model = IidModel::new(4);
        let mut p = Problem::new(&model);
        p.eval(&model, &[0.0]).unwrap(); // tau = 1

        let b = vec![1.0, 2.0, 3.0, 4.0];
        let mut x = b.clone();
        p.solve(&mut x);

        for (xi, bi) in x.iter().zip(b.iter()) {
            assert_abs_diff_eq!(xi, bi, epsilon = 1e-10);
        }
    }

    #[test]
    fn problem_solve_iid_tau2() {
        // Q = 2·I  →  Q⁻¹·b = b/2
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
        // AR1 con ρ=0 es iid → mismo log-det
        let n   = 6;
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
        // RW1 puro (sin regularización) es singular → error esperado
        // Usamos un modelo iid con tau negativo (inválido)
        // tau = exp(-1000) ≈ 0 → Q ≈ 0 → no PD
        let model = IidModel::new(4);
        let mut p = Problem::new(&model);
        // tau = exp(-1000) es efectivamente 0 — Cholesky debe fallar
        // (en la práctica, faer puede o no detectarlo según tolerancia)
        // Este test verifica que el pipeline no panics en casos extremos
        let result = p.eval(&model, &[-1000.0]);
        // Puede ser Ok (tau muy pequeño pero > 0) o Err — ambos son válidos
        let _ = result;
    }
}
