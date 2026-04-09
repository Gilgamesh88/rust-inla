//! Modelos GMRF. Cada modelo implementa `QFunc`.
//!
//! ## Modelos implementados
//! | Modelo   | θ                    | Propio |
//! |----------|----------------------|--------|
//! | IidModel | [log τ]              | sí     |
//! | Rw1Model | [log τ]              | no     |
//! | Ar1Model | [log τ, arctanh(ρ)] | sí     |
//!
//! ## Pseudodeterminante para priors impropios
//!
//! Para Rw1 (kernel = [1,...,1]), los eigenvalores no nulos de DᵀD son:
//!   λ_k = 2(1-cos(kπ/n)),  k=1..n-1
//!
//! log|Q_rw1|* = (n-1)·log(τ) + Σ_{k=1}^{n-1} log(2(1-cos(kπ/n)))
//!
//! Equivalente a `rw.c` en gmrflib.

use crate::graph::Graph;

pub trait QFunc: Send + Sync {
    fn graph(&self) -> &Graph;
    fn eval(&self, i: usize, j: usize, theta: &[f64]) -> f64;
    fn n_hyperparams(&self) -> usize;

    fn deval(&self, _i: usize, _j: usize, _theta: &[f64], _k: usize) -> Option<f64> {
        None
    }

    /// true = Q es PD (prior propio). false = Q es PSD (prior impropio, e.g. Rw1).
    fn is_proper(&self) -> bool { true }

    /// Log-pseudodeterminante de Q(θ) para priors impropios.
    ///
    /// Para un prior de rango r < n, el pseudodeterminante es el producto
    /// de los r eigenvalores no nulos.
    ///
    /// - `Some(v)`: usar v como log|Q_x|* en la Laplace.
    /// - `None`:    usar Cholesky (si is_proper) o 0.0 (fallback).
    ///
    /// Solo implementar si `is_proper() == false`.
    fn log_pseudo_det(&self, _theta: &[f64]) -> Option<f64> { None }
}

// ── IidModel ──────────────────────────────────────────────────────────────────

pub struct IidModel { graph: Graph }

impl IidModel {
    pub fn new(n: usize) -> Self { Self { graph: Graph::iid(n) } }
}

impl QFunc for IidModel {
    fn graph(&self) -> &Graph { &self.graph }
    fn eval(&self, i: usize, j: usize, theta: &[f64]) -> f64 {
        debug_assert_eq!(i, j);
        theta[0].exp()
    }
    fn n_hyperparams(&self) -> usize { 1 }
    fn deval(&self, i: usize, j: usize, theta: &[f64], k: usize) -> Option<f64> {
        if k != 0 { return Some(0.0); }
        Some(self.eval(i, j, theta))
    }
}

// ── Rw1Model ──────────────────────────────────────────────────────────────────

pub struct Rw1Model { graph: Graph }

impl Rw1Model {
    pub fn new(n: usize) -> Self { Self { graph: Graph::linear(n) } }
}

impl QFunc for Rw1Model {
    fn graph(&self) -> &Graph { &self.graph }

    fn eval(&self, i: usize, j: usize, theta: &[f64]) -> f64 {
        let tau = theta[0].exp();
        let n   = self.graph.n();
        if i == j {
            let degree = if i == 0 || i == n - 1 { 1.0 } else { 2.0 };
            tau * degree
        } else { -tau }
    }

    fn n_hyperparams(&self) -> usize { 1 }

    fn deval(&self, i: usize, j: usize, theta: &[f64], k: usize) -> Option<f64> {
        if k != 0 { return Some(0.0); }
        Some(self.eval(i, j, theta))
    }

    fn is_proper(&self) -> bool { false }

    /// log|Q_rw1|* = (n-1)·log(τ) + Σ_{k=1}^{n-1} log(2(1-cos(kπ/n)))
    ///
    /// Equivalente a rw.c en gmrflib.
    fn log_pseudo_det(&self, theta: &[f64]) -> Option<f64> {
        let n       = self.graph.n();
        let log_tau = theta[0];
        let sum_log_eig: f64 = (1..n)
            .map(|k| {
                let angle = std::f64::consts::PI * k as f64 / n as f64;
                (2.0 * (1.0 - angle.cos())).ln()
            })
            .sum();
        Some((n as f64 - 1.0) * log_tau + sum_log_eig)
    }
}

// ── Ar1Model ──────────────────────────────────────────────────────────────────

pub struct Ar1Model { graph: Graph }

impl Ar1Model {
    pub fn new(n: usize) -> Self { Self { graph: Graph::ar1(n) } }
}

impl QFunc for Ar1Model {
    fn graph(&self) -> &Graph { &self.graph }

    fn eval(&self, i: usize, j: usize, theta: &[f64]) -> f64 {
        let tau = theta[0].exp();
        let rho = theta[1].tanh();
        let n   = self.graph.n();
        if i == j {
            if i == 0 || i == n - 1 { tau }
            else { tau * (1.0 + rho * rho) }
        } else { -tau * rho }
    }

    fn n_hyperparams(&self) -> usize { 2 }

    fn deval(&self, i: usize, j: usize, theta: &[f64], k: usize) -> Option<f64> {
        let tau = theta[0].exp();
        let rho = theta[1].tanh();
        let n   = self.graph.n();
        match k {
            0 => Some(self.eval(i, j, theta)),
            1 => {
                let drho = 1.0 - rho * rho;
                if i == j {
                    if i == 0 || i == n - 1 { Some(0.0) }
                    else { Some(tau * 2.0 * rho * drho) }
                } else { Some(-tau * drho) }
            }
            _ => Some(0.0),
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn iid_graph_is_diagonal() {
        let m = IidModel::new(10);
        assert_eq!(m.graph().n(), 10);
        assert_eq!(m.graph().nnz(), 10);
        assert_eq!(m.n_hyperparams(), 1);
    }

    #[test]
    fn iid_diagonal_equals_tau() {
        let m = IidModel::new(5);
        let theta = [2.0_f64.ln()];
        for i in 0..5 { assert_abs_diff_eq!(m.eval(i, i, &theta), 2.0, epsilon = 1e-12); }
    }

    #[test]
    fn iid_tau_increases_with_theta() {
        let m = IidModel::new(3);
        assert!(m.eval(0, 0, &[2.0]) > m.eval(0, 0, &[0.0]));
    }

    #[test]
    fn iid_log_pseudo_det_is_none() {
        assert!(IidModel::new(5).log_pseudo_det(&[0.0]).is_none());
    }

    #[test]
    fn rw1_graph_is_tridiagonal() {
        let m = Rw1Model::new(5);
        assert_eq!(m.graph().n(), 5);
        assert_eq!(m.graph().nnz(), 13);
        assert_eq!(m.n_hyperparams(), 1);
    }

    #[test]
    fn rw1_boundary_diagonal_is_tau() {
        let m = Rw1Model::new(5);
        assert_abs_diff_eq!(m.eval(0, 0, &[0.0]), 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(m.eval(4, 4, &[0.0]), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn rw1_interior_diagonal_is_two_tau() {
        let m = Rw1Model::new(5);
        for i in 1..4 { assert_abs_diff_eq!(m.eval(i, i, &[0.0]), 2.0, epsilon = 1e-12); }
    }

    #[test]
    fn rw1_offdiagonal_is_minus_tau() {
        let m = Rw1Model::new(5);
        let tau = 1.0_f64.exp();
        for i in 0..4 { assert_abs_diff_eq!(m.eval(i, i+1, &[1.0]), -tau, epsilon = 1e-12); }
    }

    #[test]
    fn rw1_is_symmetric() {
        let m = Rw1Model::new(5);
        for i in 0..4 {
            assert_abs_diff_eq!(m.eval(i, i+1, &[0.5]), m.eval(i+1, i, &[0.5]), epsilon = 1e-12);
        }
    }

    #[test]
    fn rw1_log_pseudo_det_n3_tau1() {
        // n=3, τ=1: eigenvalores de DᵀD = {1, 3} → log(1·3) = log(3)
        let m = Rw1Model::new(3);
        assert_abs_diff_eq!(m.log_pseudo_det(&[0.0]).unwrap(), 3.0_f64.ln(), epsilon = 1e-10);
    }

    #[test]
    fn rw1_log_pseudo_det_n3_tau2() {
        // n=3, τ=2: eigenvalores de Q = {2,6} → log(12)
        let m = Rw1Model::new(3);
        assert_abs_diff_eq!(m.log_pseudo_det(&[2.0_f64.ln()]).unwrap(), 12.0_f64.ln(), epsilon = 1e-10);
    }

    #[test]
    fn rw1_log_pseudo_det_n2() {
        let m = Rw1Model::new(2);
        assert_abs_diff_eq!(m.log_pseudo_det(&[0.0]).unwrap(), 2.0_f64.ln(), epsilon = 1e-10);
    }

    #[test]
    fn rw1_log_pseudo_det_scales_linearly_with_log_tau() {
        let n   = 10;
        let m   = Rw1Model::new(n);
        let lpd1 = m.log_pseudo_det(&[0.0]).unwrap();
        let lpd2 = m.log_pseudo_det(&[3.0_f64.ln()]).unwrap();
        assert_abs_diff_eq!(lpd2 - lpd1, (n as f64 - 1.0) * 3.0_f64.ln(), epsilon = 1e-10);
    }

    #[test]
    fn rw1_is_improper_has_pseudo_det() {
        let m = Rw1Model::new(10);
        assert!(!m.is_proper());
        assert!(m.log_pseudo_det(&[0.0]).is_some());
    }

    #[test]
    fn ar1_graph_is_tridiagonal() {
        let m = Ar1Model::new(10);
        assert_eq!(m.graph().n(), 10);
        assert_eq!(m.graph().nnz(), 10 + 2 * 9);
        assert_eq!(m.n_hyperparams(), 2);
    }

    #[test]
    fn ar1_with_rho_zero_is_iid() {
        let m_ar1 = Ar1Model::new(5);
        let m_iid = IidModel::new(5);
        for i in 0..5 {
            assert_abs_diff_eq!(m_ar1.eval(i, i, &[0.0, 0.0]), m_iid.eval(i, i, &[0.0]), epsilon = 1e-12);
        }
        for i in 0..4 {
            assert_abs_diff_eq!(m_ar1.eval(i, i+1, &[0.0, 0.0]), 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn ar1_boundary_diagonal_is_tau() {
        let m = Ar1Model::new(5);
        let theta = [3.0_f64.ln(), 0.7_f64.atanh()];
        assert_abs_diff_eq!(m.eval(0, 0, &theta), 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(m.eval(4, 4, &theta), 3.0, epsilon = 1e-10);
    }

    #[test]
    fn ar1_interior_diagonal_has_rho_correction() {
        let m = Ar1Model::new(5);
        let (tau, rho) = (3.0_f64, 0.7_f64);
        let theta = [tau.ln(), rho.atanh()];
        let expected = tau * (1.0 + rho * rho);
        for i in 1..4 { assert_abs_diff_eq!(m.eval(i, i, &theta), expected, epsilon = 1e-10); }
    }

    #[test]
    fn ar1_offdiagonal_is_minus_tau_rho() {
        let m = Ar1Model::new(5);
        let (tau, rho) = (3.0_f64, 0.7_f64);
        let theta = [tau.ln(), rho.atanh()];
        for i in 0..4 {
            assert_abs_diff_eq!(m.eval(i, i+1, &theta), -tau*rho, epsilon = 1e-10);
        }
    }

    #[test]
    fn ar1_is_symmetric() {
        let m = Ar1Model::new(5);
        let theta = [1.0, 0.5_f64.atanh()];
        for i in 0..4 {
            assert_abs_diff_eq!(m.eval(i, i+1, &theta), m.eval(i+1, i, &theta), epsilon = 1e-12);
        }
    }

    #[test]
    fn ar1_is_diagonally_dominant_for_valid_rho() {
        let m = Ar1Model::new(10);
        let theta = [2.0_f64.ln(), 0.8_f64.atanh()];
        for i in 0..10 {
            let diag    = m.eval(i, i, &theta);
            let off_sum: f64 = m.graph().neighbors_of(i).iter()
                .map(|&j| m.eval(i, j, &theta).abs()).sum();
            let off_total = if i == 0 || i == 9 { off_sum } else { off_sum * 2.0 };
            assert!(diag > off_total - 1e-10, "nodo {i}: diag={diag:.4} off={off_total:.4}");
        }
    }

    #[test]
    fn ar1_log_pseudo_det_is_none() {
        assert!(Ar1Model::new(5).log_pseudo_det(&[0.0, 0.0]).is_none());
    }
}
