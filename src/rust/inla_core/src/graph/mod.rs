//! Estructura de dispersidad del campo aleatorio Q.
//!
//! `Graph` codifica qué pares (i,j) tienen entradas no-nulas en la matriz
//! de precisión Q. Es el equivalente Rust de `graph.c` (2127 líneas).
//!
//! ## Convención de almacenamiento
//!
//! Solo guardamos la triangular superior (j > i). La diagonal es implícita.
//!
//! ```text
//! Para Q tridiagonal n=4:
//!   neighbors[0] = [1]   ← Q(0,1)
//!   neighbors[1] = [2]   ← Q(1,2)
//!   neighbors[2] = [3]   ← Q(2,3)
//!   neighbors[3] = []
//! ```

use sha2::{Digest, Sha256};

/// Grafo de dispersidad de Q.
#[derive(Debug, Clone)]
pub struct Graph {
    pub n: usize,
    pub(crate) neighbors: Vec<Vec<usize>>,
    #[allow(dead_code)] // poblado por FaerSolver::reorder() en Fase A.3
    pub(crate) fill_pattern: Vec<Vec<usize>>,
    pub(crate) graph_hash: [u8; 32],
}

impl Graph {
    // ── Accesores ─────────────────────────────────────────────────────────────

    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }

    /// ¿Son i y j vecinos en Q? Funciona en ambas direcciones.
    pub fn are_neighbors(&self, i: usize, j: usize) -> bool {
        let (lo, hi) = if i < j { (i, j) } else { (j, i) };
        self.neighbors[lo].binary_search(&hi).is_ok()
    }

    /// Vecinos del nodo i en la triangular superior (j > i).
    /// El solver itera esto para evaluar QFunc::eval(i, j, theta).
    #[inline]
    pub fn neighbors_of(&self, i: usize) -> &[usize] {
        &self.neighbors[i]
    }

    /// nnz(Q) = n (diagonal) + 2 * aristas (simetría)
    pub fn nnz(&self) -> usize {
        let off_diag: usize = self.neighbors.iter().map(|v| v.len()).sum();
        self.n + 2 * off_diag
    }

    pub fn hash(&self) -> &[u8; 32] {
        &self.graph_hash
    }

    /// Itera todos los pares (i, j) con j > i.
    /// El solver llama a QFunc::eval(i, j, theta) para cada par
    /// y rellena Q en formato CSC.
    pub fn iter_upper_triangle(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.neighbors
            .iter()
            .enumerate()
            .flat_map(|(i, nbrs)| nbrs.iter().map(move |&j| (i, j)))
    }

    // ── Constructores ─────────────────────────────────────────────────────────

    /// Q diagonal. Uso: modelo iid.
    pub fn empty(n: usize) -> Self {
        let neighbors = vec![vec![]; n];
        let hash = Self::compute_hash(&neighbors, n);
        Self {
            n,
            neighbors,
            fill_pattern: vec![],
            graph_hash: hash,
        }
    }

    /// Alias semántico de empty para contextos iid.
    pub fn iid(n: usize) -> Self {
        Self::empty(n)
    }

    /// Q tridiagonal (cadena). Uso: modelos RW1.
    pub fn linear(n: usize) -> Self {
        let mut neighbors = vec![vec![]; n];
        for (i, row) in neighbors.iter_mut().enumerate().take(n.saturating_sub(1)) {
            row.push(i + 1);
        }
        let hash = Self::compute_hash(&neighbors, n);
        Self {
            n,
            neighbors,
            fill_pattern: vec![],
            graph_hash: hash,
        }
    }

    /// Alias semántico de linear para contextos AR1.
    /// AR1 y RW1 tienen la misma estructura de grafo (cadena).
    /// La diferencia está en los valores de Q(i,j,theta), no en la estructura.
    pub fn ar1(n: usize) -> Self {
        Self::linear(n)
    }

    /// Q pentadiagonal de cadena de segundo orden. Uso: modelo RW2.
    pub fn rw2(n: usize) -> Self {
        let mut edges = Vec::with_capacity(n.saturating_sub(1) + n.saturating_sub(2));
        for i in 0..n.saturating_sub(1) {
            edges.push((i, i + 1));
        }
        for i in 0..n.saturating_sub(2) {
            edges.push((i, i + 2));
        }
        Self::from_neighbors(n, &edges)
    }

    /// Alias semántico de rw2 para contextos AR2.
    /// AR2 y RW2 comparten el mismo patrón de banda de segundo orden.
    pub fn ar2(n: usize) -> Self {
        Self::rw2(n)
    }

    /// Constructor genérico desde lista de aristas.
    ///
    /// Los pares pueden venir en cualquier orden y con duplicados:
    /// el constructor normaliza a (lo, hi), deduplica y ordena.
    ///
    /// # Panics (debug)
    /// Si algún índice >= n o si hay self-loop (i == j).
    pub fn from_neighbors(n: usize, edges: &[(usize, usize)]) -> Self {
        let mut neighbors: Vec<Vec<usize>> = vec![vec![]; n];

        for &(a, b) in edges {
            debug_assert!(a != b, "self-loops no permitidos");
            debug_assert!(a < n && b < n, "índice fuera de rango");

            let (lo, hi) = if a < b { (a, b) } else { (b, a) };

            // Inserción ordenada sin duplicados.
            // Para grafos dispersos (grado típico 1-6 en INLA) es O(k) por arista.
            match neighbors[lo].binary_search(&hi) {
                Ok(_) => {} // duplicado, ignorar
                Err(pos) => neighbors[lo].insert(pos, hi),
            }
        }

        let hash = Self::compute_hash(&neighbors, n);
        Self {
            n,
            neighbors,
            fill_pattern: vec![],
            graph_hash: hash,
        }
    }

    /// Une varios grafos disjuntos desplazando sus índices por `start`.
    pub fn disjoint_union(parts: &[(&Graph, usize)]) -> Self {
        let n = parts
            .iter()
            .map(|(graph, start)| start + graph.n())
            .max()
            .unwrap_or(0);
        let mut edges = Vec::new();
        for (graph, start) in parts {
            edges.extend(
                graph
                    .iter_upper_triangle()
                    .map(|(i, j)| (i + *start, j + *start)),
            );
        }
        Self::from_neighbors(n, &edges)
    }

    /// Descubre las conexiones inducidas por la matriz A de observaciones cruzadas.
    pub fn build_a_t_a_edges(a_i: &[usize], a_j: &[usize], n_data: usize) -> Vec<(usize, usize)> {
        let mut row_to_cols: Vec<Vec<usize>> = vec![vec![]; n_data];
        for k in 0..a_i.len() {
            row_to_cols[a_i[k]].push(a_j[k]);
        }

        let mut edges = Vec::new();
        for cols in row_to_cols {
            for m in 0..cols.len() {
                for n in (m + 1)..cols.len() {
                    edges.push((cols[m], cols[n]));
                }
            }
        }
        edges
    }

    /// Combina aristas en el grafo actual de manera determinista (sin duplicados, rehashea).
    pub fn merge_edges(&mut self, edges: &[(usize, usize)]) {
        for &(a, b) in edges {
            if a == b {
                continue;
            }
            let (lo, hi) = if a < b { (a, b) } else { (b, a) };
            if hi >= self.n {
                continue;
            }
            match self.neighbors[lo].binary_search(&hi) {
                Ok(_) => {}
                Err(pos) => self.neighbors[lo].insert(pos, hi),
            }
        }
        self.graph_hash = Self::compute_hash(&self.neighbors, self.n);
    }

    // ── Internals ─────────────────────────────────────────────────────────────

    fn compute_hash(neighbors: &[Vec<usize>], n: usize) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(n.to_le_bytes());
        for (i, nbrs) in neighbors.iter().enumerate() {
            hasher.update(i.to_le_bytes());
            for &j in nbrs {
                hasher.update(j.to_le_bytes());
            }
            hasher.update([0xFF]); // separador para evitar colisiones
        }
        hasher.finalize().into()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Constructores ─────────────────────────────────────────────────────────

    #[test]
    fn empty_graph_has_no_edges() {
        let g = Graph::empty(5);
        assert_eq!(g.n(), 5);
        assert_eq!(g.nnz(), 5);
        for i in 0..5 {
            for j in 0..5 {
                if i != j {
                    assert!(!g.are_neighbors(i, j));
                }
            }
        }
    }

    #[test]
    fn iid_equals_empty() {
        let g1 = Graph::iid(10);
        let g2 = Graph::empty(10);
        assert_eq!(g1.hash(), g2.hash());
    }

    #[test]
    fn linear_graph_has_chain_edges() {
        let g = Graph::linear(5);
        assert!(g.are_neighbors(0, 1));
        assert!(g.are_neighbors(3, 4));
        assert!(!g.are_neighbors(0, 2));
        assert_eq!(g.nnz(), 13); // 5 + 2*4
    }

    #[test]
    fn ar1_equals_linear() {
        let g1 = Graph::ar1(20);
        let g2 = Graph::linear(20);
        assert_eq!(g1.hash(), g2.hash());
        assert_eq!(g1.nnz(), g2.nnz());
    }

    #[test]
    fn rw2_graph_has_second_order_chain_edges() {
        let g = Graph::rw2(5);
        assert!(g.are_neighbors(0, 1));
        assert!(g.are_neighbors(0, 2));
        assert!(g.are_neighbors(2, 4));
        assert!(!g.are_neighbors(0, 3));
        assert_eq!(g.nnz(), 5 + 2 * (4 + 3));
    }

    #[test]
    fn rw2_matches_explicit_neighbor_list() {
        let n = 7;
        let edges: Vec<(usize, usize)> = (0..n - 1)
            .map(|i| (i, i + 1))
            .chain((0..n - 2).map(|i| (i, i + 2)))
            .collect();
        let g_from = Graph::from_neighbors(n, &edges);
        let g_rw2 = Graph::rw2(n);
        assert_eq!(g_from.hash(), g_rw2.hash());
        assert_eq!(g_from.nnz(), g_rw2.nnz());
    }

    #[test]
    fn ar2_equals_rw2_graph_pattern() {
        let g_ar2 = Graph::ar2(9);
        let g_rw2 = Graph::rw2(9);
        assert_eq!(g_ar2.hash(), g_rw2.hash());
        assert_eq!(g_ar2.nnz(), g_rw2.nnz());
    }

    // ── from_neighbors ────────────────────────────────────────────────────────

    #[test]
    fn from_neighbors_builds_chain() {
        let edges = vec![(0, 1), (1, 2), (2, 3), (3, 4)];
        let g = Graph::from_neighbors(5, &edges);
        assert_eq!(g.n(), 5);
        assert_eq!(g.nnz(), 13);
        assert!(g.are_neighbors(0, 1));
        assert!(g.are_neighbors(3, 4));
        assert!(!g.are_neighbors(0, 4));
    }

    #[test]
    fn from_neighbors_deduplicates() {
        // (0,1) aparece tres veces — debe contar solo una
        let edges = vec![(0, 1), (1, 0), (0, 1), (1, 2)];
        let g = Graph::from_neighbors(3, &edges);
        assert_eq!(g.nnz(), 3 + 2 * 2); // 3 diag + 2 aristas únicas × 2
    }

    #[test]
    fn from_neighbors_handles_reversed_pairs() {
        let edges = vec![(3, 1), (2, 0)];
        let g = Graph::from_neighbors(4, &edges);
        assert!(g.are_neighbors(1, 3));
        assert!(g.are_neighbors(0, 2));
    }

    #[test]
    fn from_neighbors_matches_linear() {
        let n = 10;
        let edges: Vec<(usize, usize)> = (0..n - 1).map(|i| (i, i + 1)).collect();
        let g_from = Graph::from_neighbors(n, &edges);
        let g_linear = Graph::linear(n);
        assert_eq!(g_from.hash(), g_linear.hash());
    }

    // ── Iteradores ────────────────────────────────────────────────────────────

    #[test]
    fn disjoint_union_keeps_blocks_separate() {
        let g1 = Graph::iid(2);
        let g2 = Graph::linear(3);
        let g = Graph::disjoint_union(&[(&g1, 0), (&g2, 2)]);

        assert_eq!(g.n(), 5);
        assert!(g.are_neighbors(2, 3));
        assert!(g.are_neighbors(3, 4));
        assert!(!g.are_neighbors(1, 2));
    }

    #[test]
    fn iter_upper_triangle_visits_all_edges() {
        let g = Graph::linear(5);
        let pairs: Vec<(usize, usize)> = g.iter_upper_triangle().collect();
        assert_eq!(pairs.len(), 4);
        assert!(pairs.contains(&(0, 1)));
        assert!(pairs.contains(&(3, 4)));
    }

    #[test]
    fn iter_upper_triangle_empty_graph_yields_nothing() {
        let g = Graph::empty(10);
        assert_eq!(g.iter_upper_triangle().count(), 0);
    }

    #[test]
    fn neighbors_of_returns_correct_slice() {
        let g = Graph::linear(5);
        assert_eq!(g.neighbors_of(0), &[1]);
        assert_eq!(g.neighbors_of(1), &[2]);
        assert!(g.neighbors_of(4).is_empty()); // último nodo, sin vecinos hacia adelante
    }

    // ── Hash ──────────────────────────────────────────────────────────────────

    #[test]
    fn hash_changes_with_structure() {
        assert_ne!(Graph::empty(5).hash(), Graph::linear(5).hash());
    }

    #[test]
    fn hash_is_stable() {
        assert_eq!(Graph::linear(10).hash(), Graph::linear(10).hash());
    }

    #[test]
    fn hash_differs_by_size() {
        assert_ne!(Graph::linear(10).hash(), Graph::linear(11).hash());
    }

    // ── Fixture: cholesky_rw1.json ────────────────────────────────────────────
    //
    // Valida que Graph::linear(10_000) coincide con el fixture de R-INLA.
    //
    // Para activar:
    //   1. git add tests/fixtures/*.json && git commit
    //   2. cargo test -- --ignored
    #[test]
    #[ignore = "requiere tests/fixtures/cholesky_rw1.json en el repositorio"]
    fn fixture_cholesky_rw1_graph_structure() {
        use std::collections::HashSet;

        let raw = std::fs::read_to_string("tests/fixtures/cholesky_rw1.json")
            .expect("fixture no encontrado");

        let v: serde_json::Value = serde_json::from_str(&raw).expect("JSON inválido");

        let n = v["n"][0].as_u64().unwrap() as usize;
        let rows = v["rows"].as_array().unwrap();
        let cols = v["cols"].as_array().unwrap();

        // Aristas únicas del triángulo superior (r < c)
        let edges: Vec<(usize, usize)> = rows
            .iter()
            .zip(cols.iter())
            .filter_map(|(r, c)| {
                let ri = r.as_u64()? as usize;
                let ci = c.as_u64()? as usize;
                if ri < ci {
                    Some((ri, ci))
                } else {
                    None
                }
            })
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();

        let g = Graph::from_neighbors(n, &edges);

        assert_eq!(g.n(), n);
        assert_eq!(g.nnz(), n + 2 * (n - 1), "RW1 tridiagonal n={n}");
        assert!(g.are_neighbors(0, 1));
        assert!(g.are_neighbors(n - 2, n - 1));
        assert!(!g.are_neighbors(0, 2));

        // Debe ser idéntico a linear(n)
        assert_eq!(g.hash(), Graph::linear(n).hash());
    }
}
