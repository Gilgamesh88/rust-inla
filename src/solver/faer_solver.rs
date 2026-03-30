//! FaerSolver — Cholesky simplicial de bajo nivel (faer 0.24).
//!
//! Usamos la API simplicial (no supernodal) porque nos da acceso directo
//! a `L_values: Vec<f64>`, necesario para:
//! - `log|Q| = 2·Σlog(L_ii)`  donde  `L[j,j] = L_values[col_ptr[j]]`
//! - Inversa seleccionada (Takahashi) — Fase A.3 final
//!
//! Trade-off: más fill-in que supernodal para matrices grandes.
//! AMD se añade en Fase A.3 final para mejorar rendimiento.

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::llt::factor::LltRegularization;
use faer::sparse::linalg::cholesky::simplicial::{
    self, EliminationTreeRef, SimplicialLltRef,
    SymbolicSimplicialCholesky,
};
use faer::sparse::{SparseColMat, Triplet};
use faer::{Conj, Par};

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

    /// Triplets de Q (triangular superior) — rellenos en build().
    triplets: Vec<Triplet<SpIdx, SpIdx, f64>>,

    /// Q materializada como CSC upper triangular.
    q_upper: Option<SparseColMat<SpIdx, f64>>,

    /// Estructura simbólica de L (patrón de fill-in).
    /// Calculada una vez por grafo en reorder().
    symbolic: Option<SymbolicSimplicialCholesky<SpIdx>>,

    /// Valores numéricos de L (triangular inferior).
    /// L[j,j] = l_values[symbolic.col_ptr()[j]]  ← clave para log-det.
    l_values: Vec<f64>,

    /// log|Q| = 2·Σlog(L_ii), cacheado tras factorize().
    log_det: Option<f64>,
}

impl FaerSolver {
    pub fn new() -> Self {
        Self {
            state:       State::Fresh,
            n:           0,
            cached_hash: [0u8; 32],
            triplets:    Vec::new(),
            q_upper:     None,
            symbolic:    None,
            l_values:    Vec::new(),
            log_det:     None,
        }
    }

    fn needs_reorder(&self, graph: &Graph) -> bool {
        self.state == State::Fresh || self.cached_hash != *graph.hash()
    }

    /// Construye los triplets del triángulo superior de Q.
    fn fill_triplets(&mut self, graph: &Graph, qfunc: &dyn QFunc, theta: &[f64]) {
        let n = graph.n();
        self.triplets.clear();
        self.triplets.reserve(n + graph.iter_upper_triangle().count());

        for i in 0..n {
            self.triplets.push(Triplet::new(i, i, qfunc.eval(i, i, theta)));
        }
        for (i, j) in graph.iter_upper_triangle() {
            // Solo upper triangle — la API simplicial accede solo a esa parte
            self.triplets.push(Triplet::new(i, j, qfunc.eval(i, j, theta)));
        }
    }
}

impl Default for FaerSolver {
    fn default() -> Self { Self::new() }
}

impl SparseSolver for FaerSolver {

    // ── reorder: estructura simbólica de L ───────────────────────────────────

    fn reorder(&mut self, graph: &mut Graph) {
        if !self.needs_reorder(graph) { return; }

        self.n           = graph.n();
        self.cached_hash = *graph.hash();
        self.q_upper     = None;
        self.l_values    = Vec::new();
        self.log_det     = None;

        let n = self.n;

        // Patrón de Q (triangular superior, valores = 1.0, no importan)
        let pattern: Vec<Triplet<SpIdx, SpIdx, f64>> = (0..n)
            .map(|i| Triplet::new(i, i, 1.0))
            .chain(graph.iter_upper_triangle().map(|(i,j)| Triplet::new(i, j, 1.0)))
            .collect();

        let q_pat = SparseColMat::<SpIdx, f64>::try_new_from_triplets(n, n, &pattern)
            .expect("reorder: patrón inválido");

        // ── Paso 1: árbol de eliminación y conteo de columnas ────────────────
        let scratch1 = simplicial::prefactorize_symbolic_cholesky_scratch::<SpIdx>(
            n, q_pat.compute_nnz(),
        );
        let mut mem1 = MemBuffer::try_new(scratch1)
            .expect("reorder: no hay memoria para scratch1");
        let stack1 = MemStack::new(&mut mem1);

        let mut etree:      Vec<isize> = vec![0; n];
        let mut col_counts: Vec<SpIdx> = vec![0; n];

        simplicial::prefactorize_symbolic_cholesky(
            &mut etree,
            &mut col_counts,
            q_pat.symbolic(),
            stack1,
        );

        // ── Paso 2: Cholesky simbólico ────────────────────────────────────────
        let scratch2 = simplicial::factorize_simplicial_symbolic_cholesky_scratch::<SpIdx>(n);
        let mut mem2 = MemBuffer::try_new(scratch2)
            .expect("reorder: no hay memoria para scratch2");
        let stack2 = MemStack::new(&mut mem2);

        let symbolic = simplicial::factorize_simplicial_symbolic_cholesky(
            q_pat.symbolic(),
            // SAFETY: etree fue rellenado por prefactorize_symbolic_cholesky
            unsafe { EliminationTreeRef::from_inner(&etree) },
            &col_counts,
            stack2,
        ).expect("reorder: fallo en Cholesky simbólico");

        self.symbolic = Some(symbolic);
        self.state    = State::Reordered;
    }

    // ── build: evalúa QFunc y construye Q upper ───────────────────────────────

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
        ).expect("build: triplets inválidos");

        self.q_upper = Some(mat);
        self.l_values.clear();
        self.log_det = None;
        self.state   = State::Built;
    }

    // ── factorize: Cholesky numérico + log-det ────────────────────────────────

    fn factorize(&mut self) -> Result<(), InlaError> {
        debug_assert_eq!(self.state, State::Built);

        let symbolic = self.symbolic.as_ref()
            .expect("llamar reorder() primero");
        let q_upper = self.q_upper.as_ref()
            .expect("llamar build() primero");

        let n = self.n;

        // Alojar L_values
        self.l_values.clear();
        self.l_values.resize(symbolic.len_val(), 0.0);

        // Scratch para la factorización numérica
        let scratch = simplicial::factorize_simplicial_numeric_llt_scratch::<SpIdx, f64>(n);
        let mut mem = MemBuffer::try_new(scratch)
            .map_err(|_| InlaError::NotPositiveDefinite)?;
        let stack = MemStack::new(&mut mem);

        match simplicial::factorize_simplicial_numeric_llt::<SpIdx, f64>(
            &mut self.l_values,
            q_upper.as_ref(),
            LltRegularization::default(),
            symbolic,
            stack,
        ) {
            Ok(_)                  => {}
            Err(_) => {
                return Err(InlaError::NotPositiveDefinite);
            }
        }

        // ── log|Q| = 2·Σlog(L[j,j]) ──────────────────────────────────────────
        // En CSC lower triangular: el primer elemento de la columna j
        // es la diagonal L[j,j], que está en l_values[col_ptr[j]].
        let col_ptr = symbolic.col_ptr(); // &[usize]
        let log_det: f64 = (0..n)
            .map(|j| {
                let diag = self.l_values[col_ptr[j]];
                diag.ln()
            })
            .sum::<f64>()
            * 2.0;

        self.log_det = Some(log_det);
        self.state   = State::Factorized;
        Ok(())
    }

    // ── log_determinant ───────────────────────────────────────────────────────

    fn log_determinant(&self) -> f64 {
        debug_assert_eq!(self.state, State::Factorized);
        self.log_det.expect("log_det no disponible")
    }

    // ── solve_llt ─────────────────────────────────────────────────────────────

    fn solve_llt(&self, rhs: &mut [f64]) {
        debug_assert_eq!(self.state, State::Factorized);

        let symbolic = self.symbolic.as_ref().expect("reorder primero");
        let n = rhs.len();

        // Scratch para el solve
        let scratch = symbolic.solve_in_place_scratch::<f64>(1);
        let mut mem = MemBuffer::try_new(scratch)
            .expect("solve_llt: no hay memoria");
        let stack = MemStack::new(&mut mem);

        // Convertir rhs a Mat<f64> (copia in/out — correcto para n típico de INLA)
        let mut rhs_mat = faer::Mat::<f64>::from_fn(n, 1, |i, _| rhs[i]);

        let llt_ref = SimplicialLltRef::<SpIdx, f64>::new(symbolic, &self.l_values);
        llt_ref.solve_in_place_with_conj(Conj::No, rhs_mat.as_mut(), Par::Seq, stack);

        for i in 0..n {
            rhs[i] = rhs_mat[(i, 0)];
        }
    }

    // ── selected_inverse ──────────────────────────────────────────────────────

    fn selected_inverse(&mut self) -> Result<SpMat, InlaError> {
        debug_assert_eq!(self.state, State::Factorized);
        // Fase A.3 final: algoritmo de Takahashi
        // Con l_values y symbolic.col_ptr() ya tenemos todo lo necesario.
        unimplemented!("selected_inverse: pendiente Fase A.3 final")
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{Ar1Model, IidModel, Rw1Model};
    use approx::assert_abs_diff_eq;

    #[test]
    fn solver_starts_fresh() {
        assert_eq!(FaerSolver::new().state, State::Fresh);
    }

    #[test]
    fn build_triplets_iid_n3() {
        let mut solver = FaerSolver::new();
        let mut g = Graph::iid(3);
        let model = IidModel::new(3);
        solver.reorder(&mut g);
        solver.fill_triplets(&g, &model, &[2.0_f64.ln()]);
        assert_eq!(solver.triplets.len(), 3); // solo diagonal
        for t in &solver.triplets {
            assert_abs_diff_eq!(t.val, 2.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn build_triplets_rw1_n3() {
        let mut solver = FaerSolver::new();
        let mut g = Graph::linear(3);
        let model = Rw1Model::new(3);
        solver.reorder(&mut g);
        solver.fill_triplets(&g, &model, &[0.0]);
        // upper triangle: 3 diag + 2 off-diag = 5
        assert_eq!(solver.triplets.len(), 5);
        let sum: f64 = solver.triplets.iter().map(|t| t.val).sum();
        // suma = (1+2+1) + (-1-1) = 2  (triángulo superior de DᵀD)
        assert_abs_diff_eq!(sum, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn reorder_caches_hash() {
        let mut solver = FaerSolver::new();
        let mut g = Graph::linear(10);
        solver.reorder(&mut g);
        let h1 = solver.cached_hash;
        solver.reorder(&mut g);
        assert_eq!(solver.cached_hash, h1);
    }

    #[test]
    fn full_pipeline_iid_log_det() {
        // Q = tau * I  →  log|Q| = n * log(tau)
        let n   = 4;
        let tau = 3.0_f64;
        let mut solver = FaerSolver::new();
        let mut g = Graph::iid(n);
        let model = IidModel::new(n);

        solver.reorder(&mut g);
        solver.build(&g, &model, &[tau.ln()]);
        solver.factorize().unwrap();

        let expected = (n as f64) * tau.ln();
        assert_abs_diff_eq!(solver.log_determinant(), expected, epsilon = 1e-8);
    }

    #[test]
    fn solve_iid_identity() {
        // Q = I  →  Q⁻¹·b = b
        let mut solver = FaerSolver::new();
        let mut g = Graph::iid(4);
        let model = IidModel::new(4);

        solver.reorder(&mut g);
        solver.build(&g, &model, &[0.0]);
        solver.factorize().unwrap();

        let b = vec![1.0, 2.0, 3.0, 4.0];
        let mut x = b.clone();
        solver.solve_llt(&mut x);

        for (xi, bi) in x.iter().zip(b.iter()) {
            assert_abs_diff_eq!(xi, bi, epsilon = 1e-10);
        }
    }

    #[test]
    fn solve_iid_tau2() {
        // Q = 2·I  →  Q⁻¹·b = b/2
        let mut solver = FaerSolver::new();
        let mut g = Graph::iid(4);
        let model = IidModel::new(4);

        solver.reorder(&mut g);
        solver.build(&g, &model, &[2.0_f64.ln()]);
        solver.factorize().unwrap();

        let b = vec![2.0, 4.0, 6.0, 8.0];
        let mut x = b.clone();
        solver.solve_llt(&mut x);

        for (xi, bi) in x.iter().zip(b.iter()) {
            assert_abs_diff_eq!(*xi, bi / 2.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn log_det_ar1_rho_zero_equals_iid() {
        // AR1 con ρ=0 es iid → mismo log-det que IidModel
        let n = 5;
        let tau = 2.0_f64;

        let mut s_iid = FaerSolver::new();
        let mut g_iid = Graph::iid(n);
        s_iid.reorder(&mut g_iid);
        s_iid.build(&g_iid, &IidModel::new(n), &[tau.ln()]);
        s_iid.factorize().unwrap();

        let mut s_ar1 = FaerSolver::new();
        let mut g_ar1 = Graph::ar1(n);
        s_ar1.reorder(&mut g_ar1);
        s_ar1.build(&g_ar1, &Ar1Model::new(n), &[tau.ln(), 0.0]);
        s_ar1.factorize().unwrap();

        assert_abs_diff_eq!(
            s_iid.log_determinant(),
            s_ar1.log_determinant(),
            epsilon = 1e-8
        );
    }

    // Fixture: valida log_det y solve contra R-INLA (activar con fixtures en repo)
    #[test]
    #[ignore = "requiere tests/fixtures/cholesky_rw1.json"]
    fn fixture_cholesky_log_det_and_solve() {
        let raw = std::fs::read_to_string("tests/fixtures/cholesky_rw1.json").unwrap();
        let v: serde_json::Value = serde_json::from_str(&raw).unwrap();

        let n       = v["n"][0].as_u64().unwrap() as usize;
        let rows: Vec<usize> = v["rows"].as_array().unwrap().iter()
            .map(|x| x.as_u64().unwrap() as usize).collect();
        let cols: Vec<usize> = v["cols"].as_array().unwrap().iter()
            .map(|x| x.as_u64().unwrap() as usize).collect();
        let vals: Vec<f64>   = v["vals"].as_array().unwrap().iter()
            .map(|x| x.as_f64().unwrap()).collect();
        let rhs_ref: Vec<f64> = v["rhs"].as_array().unwrap().iter()
            .map(|x| x.as_f64().unwrap()).collect();
        let sol_ref: Vec<f64> = v["sol"].as_array().unwrap().iter()
            .map(|x| x.as_f64().unwrap()).collect();
        let log_det_ref = v["log_det"][0].as_f64().unwrap();

        // Construir Q desde triplets del fixture (simetría: solo upper tri)
        let triplets: Vec<Triplet<usize, usize, f64>> = rows.iter()
            .zip(cols.iter())
            .zip(vals.iter())
            .filter(|((&r, &c), _)| r <= c)
            .map(|((&r, &c), &v)| Triplet::new(r, c, v))
            .collect();

        let q = SparseColMat::<usize, f64>::try_new_from_triplets(n, n, &triplets).unwrap();

        // Factorizar directamente con el solver low-level
        let mut etree      = vec![0isize; n];
        let mut col_counts = vec![0usize; n];
        {
            let sc = simplicial::prefactorize_symbolic_cholesky_scratch::<usize>(n, q.compute_nnz());
            let mut mem = MemBuffer::try_new(sc).unwrap();
            simplicial::prefactorize_symbolic_cholesky(&mut etree, &mut col_counts, q.symbolic(), MemStack::new(&mut mem));
        }
        let symbolic = {
            let sc = simplicial::factorize_simplicial_symbolic_cholesky_scratch::<usize>(n);
            let mut mem = MemBuffer::try_new(sc).unwrap();
            simplicial::factorize_simplicial_symbolic_cholesky(
                q.symbolic(),
                unsafe { EliminationTreeRef::from_inner(&etree) },
                &col_counts,
                MemStack::new(&mut mem),
            ).unwrap()
        };
        let mut l_values = vec![0.0f64; symbolic.len_val()];
        {
            let sc = simplicial::factorize_simplicial_numeric_llt_scratch::<usize, f64>(n);
            let mut mem = MemBuffer::try_new(sc).unwrap();
            simplicial::factorize_simplicial_numeric_llt(&mut l_values, q.as_ref(), LltRegularization::default(), &symbolic, MemStack::new(&mut mem)).unwrap();
        }

        // Verificar log-det
        let col_ptr = symbolic.col_ptr();
        let log_det: f64 = 2.0 * (0..n).map(|j| l_values[col_ptr[j]].ln()).sum::<f64>();
        assert_abs_diff_eq!(log_det, log_det_ref, epsilon = 1e-4);

        // Verificar solve
        let mut sol = rhs_ref.clone();
        let sc = symbolic.solve_in_place_scratch::<f64>(1);
        let mut mem = MemBuffer::try_new(sc).unwrap();
        let llt_ref_obj = SimplicialLltRef::<usize, f64>::new(&symbolic, &l_values);
        let mut rhs_mat = faer::Mat::<f64>::from_fn(n, 1, |i, _| sol[i]);
        llt_ref_obj.solve_in_place_with_conj(Conj::No, rhs_mat.as_mut(), Par::Seq, MemStack::new(&mut mem));
        for i in 0..n { sol[i] = rhs_mat[(i, 0)]; }

        for (s, r) in sol.iter().zip(sol_ref.iter()) {
            assert_abs_diff_eq!(s, r, epsilon = 1e-8);
        }
    }
}
