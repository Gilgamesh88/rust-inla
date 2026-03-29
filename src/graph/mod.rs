//! Estructura de dispersidad del campo aleatorio Q.
//!
//! `Graph` codifica qué pares (i,j) tienen entradas no-nulas en la matriz
//! de precisión Q. Es el equivalente Rust de `graph.c` (2127 líneas).
//!
//! ## Responsabilidades de Graph
//!
//! 1. **Patrón de Q** — vecinos de cada nodo (CSC o lista de adyacencia).
//!    Usado por `QFunc::eval` para saber qué entradas evaluar.
//!
//! 2. **Patrón de llenado de L** (post-reordering AMD).
//!    Calculado por Cholesky simbólico. Usado por `selected_inverse`.
//!    IMPORTANTE: este patrón puede ser más denso que el de Q.
//!
//! 3. **Hash SHA-2** del grafo de Q, para invalidar/reutilizar el caché
//!    de reorderings entre evaluaciones de BFGS.
//!
//! ## Por qué guardar el patrón de L aquí
//!
//! En INLA, la inversa seleccionada Q⁻¹ se evalúa SOLO en las entradas
//! (i,j) del patrón de L (no de Q). Si Graph no expone ese patrón,
//! el solver no sabe sobre qué entradas iterar. Diseñar esto bien en A.1
//! evita reescribir Graph en A.3 (cuando implementemos selected_inverse).

use sha2::{Digest, Sha256};

/// Grafo de dispersidad de Q.
///
/// Fase A.1 implementa los constructores y el hash.
/// Fase A.3 añade `fill_pattern` (patrón de L post-AMD).
#[derive(Debug, Clone)]
pub struct Graph {
    /// Número de nodos (= dimensión de Q).
    pub n: usize,

    /// Lista de adyacencia: `neighbors[i]` contiene los índices j > i
    /// con Q(i,j) ≠ 0 (solo triangular superior, la diagonal se omite).
    ///
    /// Por convención, los vecinos están ordenados ascendentemente.
    /// Esto es equivalente al `adjacency` de GMRFLib.
    pub(crate) neighbors: Vec<Vec<usize>>,

    /// Patrón de llenado de L después del reordering AMD.
    /// `fill_pattern[i]` = índices j en la columna i de L (j ≥ i).
    ///
    /// Poblado por `FaerSolver::reorder()`. Vacío hasta entonces.
    /// Fase A.3 lo completa.
    pub(crate) fill_pattern: Vec<Vec<usize>>,

    /// SHA-256 del patrón de Q. Se calcula una vez en el constructor.
    /// `FaerSolver` lo compara con el hash cacheado para reutilizar
    /// el reordering AMD sin recomputarlo.
    pub(crate) graph_hash: [u8; 32],
}

impl Graph {
    /// Número de nodos del grafo.
    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }

    /// ¿Son i y j vecinos en Q? (i ≠ j, sin importar el orden)
    pub fn are_neighbors(&self, i: usize, j: usize) -> bool {
        let (lo, hi) = if i < j { (i, j) } else { (j, i) };
        self.neighbors[lo].binary_search(&hi).is_ok()
    }

    /// Número total de entradas no-nulas en Q (incluyendo diagonal).
    /// nnz(Q) = n + 2 * Σ|neighbors[i]|
    pub fn nnz(&self) -> usize {
        let off_diag: usize = self.neighbors.iter().map(|v| v.len()).sum();
        self.n + 2 * off_diag
    }

    /// Hash del patrón de Q para caché de reorderings.
    pub fn hash(&self) -> &[u8; 32] {
        &self.graph_hash
    }

    // ── Constructores (Fase A.1) ──────────────────────────────────────────────

    /// Grafo vacío (sin aristas). Equivale a modelo iid: Q diagonal.
    /// Cada nodo es su propio vecino solo en la diagonal.
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

    /// Grafo lineal (cadena): nodo i conectado a i+1.
    /// Patrón de Q para RW1, RW2 (tridiagonal).
    pub fn linear(n: usize) -> Self {
        let mut neighbors = vec![vec![]; n];
        for i in 0..n.saturating_sub(1) {
            neighbors[i].push(i + 1);
        }
        let hash = Self::compute_hash(&neighbors, n);
        Self {
            n,
            neighbors,
            fill_pattern: vec![],
            graph_hash: hash,
        }
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
            // Separador entre nodos para evitar colisiones de concatenación.
            hasher.update([0xFF]);
        }
        hasher.finalize().into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_graph_has_no_edges() {
        let g = Graph::empty(5);
        assert_eq!(g.n(), 5);
        assert_eq!(g.nnz(), 5); // solo diagonal
        for i in 0..5 {
            for j in 0..5 {
                if i != j {
                    assert!(!g.are_neighbors(i, j));
                }
            }
        }
    }

    #[test]
    fn linear_graph_has_chain_edges() {
        let g = Graph::linear(5);
        assert!(g.are_neighbors(0, 1));
        assert!(g.are_neighbors(1, 2));
        assert!(g.are_neighbors(2, 3));
        assert!(g.are_neighbors(3, 4));
        assert!(!g.are_neighbors(0, 2)); // no hay aristas largas
        // nnz = 5 (diag) + 2*4 (off-diag simétrico) = 13
        assert_eq!(g.nnz(), 13);
    }

    #[test]
    fn hash_changes_with_structure() {
        let g1 = Graph::empty(5);
        let g2 = Graph::linear(5);
        assert_ne!(g1.hash(), g2.hash());
    }

    #[test]
    fn hash_is_stable() {
        let g1 = Graph::linear(10);
        let g2 = Graph::linear(10);
        assert_eq!(g1.hash(), g2.hash());
    }
}
