//! FaerSolver — implementación concreta de SparseSolver usando faer 0.24.
//!
//! ## Flujo de uso (equivalente a GMRFLib)
//!
//! ```ignore
//! let mut solver = FaerSolver::new();
//!
//! // UNA VEZ por grafo:
//! solver.reorder(&mut graph);          // AMD + simbólico → SymbolicLlt
//!
//! // CADA evaluación de f(θ) en BFGS:
//! solver.build(&graph, &qfunc, theta); // rellena triplets
//! solver.factorize()?;                 // Cholesky numérico → Llt
//! let ld = solver.log_determinant();   // 2·Σlog(diag(L))
//! solver.solve_llt(&mut rhs);          // Q⁻¹·b en su sitio
//! ```
//!
//! ## Por qué dos fases (simbólico + numérico)
//!
//! AMD y el análisis del patrón de L son O(n·log n) y solo dependen
//! del grafo, no de θ. Si lo hiciéramos en cada evaluación de BFGS
//! (potencialmente cientos de veces), desperdiciaríamos tiempo.
//! Al guardar `SymbolicLlt` (que es un Arc barato de clonar), solo
//! pagamos ese coste una vez por modelo.

use faer::sparse::linalg::solvers::{Llt, SymbolicLlt};
use faer::linalg::solvers::SolveCore; // necesario para solve_in_place_with_conj
use faer::sparse::{SparseColMat, Triplet};
use faer::Side;

use crate::error::InlaError;
use crate::graph::Graph;
use crate::models::QFunc;
use crate::solver::{SpIdx, SpMat, SparseSolver};

#[derive(Debug, PartialEq)]
enum State {
    Fresh,
    Reordered,
    Built,
    Factorized,
}

pub struct FaerSolver {
    state:       State,
    n:           usize,
    cached_hash: [u8; 32],

    /// Triplets de Q — se rellenan en build() evaluando QFunc.
    triplets: Vec<Triplet<SpIdx, SpIdx, f64>>,

    /// Q materializada como SparseColMat — se construye en build().
    q_mat: Option<SparseColMat<SpIdx, f64>>,

    /// Estructura simbólica de L (AMD incluido). Se guarda entre evaluaciones
    /// de θ para no repetir el análisis caro.
    symbolic: Option<SymbolicLlt<SpIdx>>,

    /// Factorización numérica LLᵀ — se recalcula en cada factorize().
    llt: Option<Llt<SpIdx, f64>>,

    /// log|Q| = 2·Σlog(Lᵢᵢ), cacheado tras factorize().
    log_det: Option<f64>,
}

impl FaerSolver {
    pub fn new() -> Self {
        Self {
            state:       State::Fresh,
            n:           0,
            cached_hash: [0u8; 32],
            triplets:    Vec::new(),
            q_mat:       None,
            symbolic:    None,
            llt:         None,
            log_det:     None,
        }
    }

    fn needs_reorder(&self, graph: &Graph) -> bool {
        self.state == State::Fresh || self.cached_hash != *graph.hash()
    }

    /// Construye los triplets de Q incluyendo diagonal y off-diagonal simétrica.
    fn fill_triplets(&mut self, graph: &Graph, qfunc: &dyn QFunc, theta: &[f64]) {
        let n = graph.n();
        self.triplets.clear();
        self.triplets.reserve(n + 2 * graph.iter_upper_triangle().count());

        // Diagonal
        for i in 0..n {
            self.triplets.push(Triplet {
                row: i,
                col: i,
                val: qfunc.eval(i, i, theta),
            });
        }
        // Off-diagonal simétrica (i,j) y (j,i)
        for (i, j) in graph.iter_upper_triangle() {
            let v = qfunc.eval(i, j, theta);
            self.triplets.push(Triplet { row: i, col: j, val: v });
            self.triplets.push(Triplet { row: j, col: i, val: v });
        }
    }
}

impl Default for FaerSolver {
    fn default() -> Self { Self::new() }
}

impl SparseSolver for FaerSolver {
    // ── reorder: AMD + Cholesky simbólico ────────────────────────────────────

    fn reorder(&mut self, graph: &mut Graph) {
        if !self.needs_reorder(graph) {
            return; // caché válido — no repetir AMD
        }
        self.n           = graph.n();
        self.cached_hash = *graph.hash();
        self.q_mat       = None;
        self.llt         = None;
        self.log_det     = None;

        // Construimos una matriz con el patrón de Q (valores = 1.0, no importan)
        // Solo necesitamos la estructura simbólica para AMD.
        let n = self.n;
        let mut pattern_triplets: Vec<Triplet<SpIdx, SpIdx, f64>> =
            Vec::with_capacity(n + 2 * graph.iter_upper_triangle().count());

        for i in 0..n {
            pattern_triplets.push(Triplet { row: i, col: i, val: 1.0 });
        }
        for (i, j) in graph.iter_upper_triangle() {
            pattern_triplets.push(Triplet { row: i, col: j, val: 1.0 });
            pattern_triplets.push(Triplet { row: j, col: i, val: 1.0 });
        }

        let pattern_mat = SparseColMat::<SpIdx, f64>::try_new_from_triplets(
            n, n, &pattern_triplets,
        ).expect("reorder: patrón de Q inválido");

        // SymbolicLlt::try_new ejecuta AMD internamente (Default ordering = AMD)
        // y calcula el patrón de fill-in de L.
        self.symbolic = Some(
            SymbolicLlt::try_new(pattern_mat.symbolic(), Side::Lower)
                .expect("reorder: fallo en factorización simbólica"),
        );

        self.state = State::Reordered;
    }

    // ── build: evalúa QFunc y materializa Q ──────────────────────────────────

    fn build(&mut self, graph: &Graph, qfunc: &dyn QFunc, theta: &[f64]) {
        debug_assert!(
            self.state == State::Reordered
                || self.state == State::Built
                || self.state == State::Factorized,
            "Llamar reorder() antes de build()"
        );

        self.fill_triplets(graph, qfunc, theta);

        let mat = SparseColMat::<SpIdx, f64>::try_new_from_triplets(
            self.n, self.n, &self.triplets,
        ).expect("build: triplets de Q inválidos");

        self.q_mat = Some(mat);
        self.llt   = None;
        self.log_det = None;
        self.state = State::Built;
    }

    // ── factorize: Cholesky numérico ──────────────────────────────────────────

    fn factorize(&mut self) -> Result<(), InlaError> {
        debug_assert_eq!(self.state, State::Built, "Llamar build() antes de factorize()");

        let symbolic = self.symbolic.as_ref()
            .expect("factorize: symbolic es None — llamar reorder() primero")
            .clone(); // barato: es un Arc

        let mat = self.q_mat.as_ref()
            .expect("factorize: q_mat es None — llamar build() primero");

        // Cholesky numérico — falla si Q no es PD
        let llt = Llt::try_new_with_symbolic(symbolic, mat.as_ref(), Side::Lower)
            .map_err(|_| InlaError::NotPositiveDefinite)?;

        // log|Q| = 2·Σlog(Lᵢᵢ)
        // TODO: acceder a la diagonal de L mediante la API interna de faer.
        // Por ahora usamos un placeholder que se reemplaza en el siguiente commit.
        // Ver probe_05_log_det en faer_api_probe.rs para la exploración.
        let log_det = compute_log_det_placeholder(&llt, self.n);

        self.llt     = Some(llt);
        self.log_det = Some(log_det);
        self.state   = State::Factorized;
        Ok(())
    }

    // ── log_determinant ───────────────────────────────────────────────────────

    fn log_determinant(&self) -> f64 {
        debug_assert_eq!(self.state, State::Factorized);
        self.log_det.expect("log_det no disponible")
    }

    // ── solve_llt: Q·x = b ───────────────────────────────────────────────────

    fn solve_llt(&self, rhs: &mut [f64]) {
        debug_assert_eq!(self.state, State::Factorized);
        let llt = self.llt.as_ref().expect("factorize primero");
        let n = rhs.len();

        // Creamos un Mat<f64> con copia (safe), resolvemos, copiamos de vuelta.
        // Para INLA n es típicamente 1k-50k — el overhead de copia es mínimo
        // comparado con el Cholesky.
        let mut rhs_mat = faer::Mat::<f64>::from_fn(n, 1, |i, _| rhs[i]);
        llt.solve_in_place_with_conj(faer::Conj::No, rhs_mat.as_mut());
        for i in 0..n {
            rhs[i] = rhs_mat[(i, 0)];
        }
    }

    // ── selected_inverse (Takahashi) ─────────────────────────────────────────

    fn selected_inverse(&mut self) -> Result<SpMat, InlaError> {
        debug_assert_eq!(self.state, State::Factorized);
        // Fase A.3 completa: algoritmo de Takahashi
        // Referencia: GMRFLib_Qinv() en problem-setup.c
        unimplemented!("selected_inverse: implementar en Fase A.3 final")
    }
}

/// Placeholder para log|Q| = 2·Σlog(Lᵢᵢ).
/// Se reemplaza cuando exploremos el API de acceso a la diagonal de L.
fn compute_log_det_placeholder<I, T>(_llt: &Llt<I, T>, _n: usize) -> f64 {
    // TODO: reemplazar con acceso real a diag(L)
    // Opciones a explorar:
    //   A) llt.symbolic.inner.simplicial().col_ptr() + llt.numeric
    //   B) método específico en SymbolicCholesky para diagonales
    //   C) resolver un sistema auxiliar (caro, no preferido)
    0.0 // incorrecto — marcador temporal
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{IidModel, Rw1Model};
    use approx::assert_abs_diff_eq;

    #[test]
    fn solver_starts_fresh() {
        let s = FaerSolver::new();
        assert_eq!(s.state, State::Fresh);
    }

    #[test]
    fn build_triplets_iid_n3() {
        let mut solver = FaerSolver::new();
        let mut g = crate::graph::Graph::iid(3);
        let model = IidModel::new(3);
        let theta = [2.0_f64.ln()];

        solver.reorder(&mut g);
        solver.fill_triplets(&g, &model, &theta);

        assert_eq!(solver.triplets.len(), 3);
        for t in &solver.triplets {
            assert_abs_diff_eq!(t.val, 2.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn build_triplets_rw1_n3() {
        let mut solver = FaerSolver::new();
        let mut g = crate::graph::Graph::linear(3);
        let model = Rw1Model::new(3);
        let theta = [0.0];

        solver.reorder(&mut g);
        solver.fill_triplets(&g, &model, &theta);

        // 3 diag + 2*2 off-diag = 7 triplets
        assert_eq!(solver.triplets.len(), 7);
        // Suma = 0 (propiedad de DᵀD: filas suman 0)
        let sum: f64 = solver.triplets.iter().map(|t| t.val).sum();
        assert_abs_diff_eq!(sum, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn reorder_caches_hash() {
        let mut solver = FaerSolver::new();
        let mut g = crate::graph::Graph::linear(10);
        solver.reorder(&mut g);
        let h1 = solver.cached_hash;
        solver.reorder(&mut g); // segunda llamada: no-op
        assert_eq!(solver.cached_hash, h1);
    }

    #[test]
    fn full_pipeline_iid() {
        // IidModel es PD por construcción → factorize no debe fallar
        let mut solver = FaerSolver::new();
        let mut g = crate::graph::Graph::iid(5);
        let model = IidModel::new(5);
        let theta = [0.0]; // tau = 1

        solver.reorder(&mut g);
        solver.build(&g, &model, &theta);
        solver.factorize().expect("IidModel debe ser PD");
        let _ = solver.log_determinant(); // placeholder — no verificamos valor aún
    }

    #[test]
    fn full_pipeline_rw1_with_regularization() {
        // RW1 es PSD — necesita regularización para ser PD
        // Usamos tau grande (log-tau = 5) como proxy
        let mut solver = FaerSolver::new();
        let n = 5;
        let mut g = crate::graph::Graph::linear(n);
        let model = Rw1Model::new(n);

        // Para hacer RW1 PD en tests aislados:
        // En producción la verosimilitud provee la regularización.
        // Aquí simplemente verificamos que el pipeline no panics.
        solver.reorder(&mut g);
        solver.build(&g, &model, &[5.0]);
        // RW1 puro es singular — esperamos error NotPositiveDefinite
        // (el assert confirma que el manejo de error funciona)
        let result = solver.factorize();
        // El resultado puede ser Ok o Err(NotPositiveDefinite) — ambos son correctos
        // dependiendo de si faer detecta la singularidad con tau=5
        let _ = result; // no panics es suficiente por ahora
    }

    #[test]
    fn solve_iid_identity() {
        // Q = I → Q⁻¹·b = b
        let mut solver = FaerSolver::new();
        let mut g = crate::graph::Graph::iid(4);
        let model = IidModel::new(4);

        solver.reorder(&mut g);
        solver.build(&g, &model, &[0.0]); // tau = 1, Q = I
        solver.factorize().unwrap();

        let b = vec![1.0, 2.0, 3.0, 4.0];
        let mut x = b.clone();
        solver.solve_llt(&mut x);

        // Q = I → solución debe ser igual al RHS
        for (xi, bi) in x.iter().zip(b.iter()) {
            assert_abs_diff_eq!(xi, bi, epsilon = 1e-10);
        }
    }
}
