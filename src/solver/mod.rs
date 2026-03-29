//! Solver disperso de Cholesky para la matriz de precisión Q.
//!
//! Equivalente Rust de `smtp-taucs.c` + `sparse-interface.h`.
//!
//! ## El algoritmo en tres fases
//!
//! ```text
//! 1. reorder(graph)
//!    └── AMD sobre el patrón de Q → permutación P
//!    └── Cholesky simbólico → patrón de llenado de L
//!    └── Almacena P y el patrón (caché por hash del grafo)
//!
//! 2. build(graph, qfunc, theta)
//!    └── Evalúa QFunc::eval(i,j,theta) para cada (i,j) en el patrón
//!    └── Rellena la matriz numérica Q en orden CSC reordenado
//!
//! 3. factorize()
//!    └── Cholesky numérico: Q = L Lᵀ
//!    └── log|Q| = 2 · Σ log(diag(L))    ← gratis de la factorización
//!
//! 4. [opcional] selected_inverse()
//!    └── Algoritmo de inversa seleccionada (Takahashi et al.)
//!    └── Calcula Q⁻¹ SOLO en el patrón de L  ← el corazón de INLA
//! ```
//!
//! ## Por qué `selected_inverse` devuelve `SparseColMat` (no `&mut Problem`)
//!
//! El diseño original del CONTEXT.md tenía `compute_qinv(&mut Problem)`,
//! lo que crea un ciclo de dependencias: solver/ → problem/ y problem/ → solver/.
//!
//! La solución: el solver devuelve la matriz Q⁻¹ (solo el patrón de L),
//! y Problem la almacena. Dependencia unidireccional:
//!
//! ```text
//! problem/ → solver/   ✓  (Problem llama al solver)
//! solver/              ✓  (solver no conoce Problem)
//! ```
//!
//! ## Alias de tipos para faer 0.24
//!
//! Centralizamos aquí para cambiar el índice (usize → u32 para matrices
//! grandes) en un solo lugar cuando hagamos profiling.

use faer::sparse::SparseColMat;

use crate::error::InlaError;
use crate::graph::Graph;
use crate::models::QFunc;

/// Tipo de índice para matrices sparse. `usize` es conservador;
/// cambiar a `u32` reduce memoria a la mitad para n < 4B.
pub type SpIdx = usize;

/// Matriz sparse CSC de f64 con índice `SpIdx`.
/// Tipo de retorno de `selected_inverse`.
pub type SpMat = SparseColMat<SpIdx, f64>;

/// Contrato del solver de Cholesky disperso.
///
/// La implementación concreta `FaerSolver` (Fase A.3) usa faer 0.24.
/// El trait existe para desacoplar `Problem` de la implementación específica.
pub trait SparseSolver {
    // ── Fase de setup (una sola vez por grafo) ────────────────────────────────

    /// Calcula el reordering AMD y el patrón de llenado de L (Cholesky simbólico).
    ///
    /// Debe llamarse **una sola vez** por grafo distinto.
    /// Si el hash del grafo ya está en caché, reutiliza el reordering.
    ///
    /// Efecto secundario: puebla `graph.fill_pattern` con el patrón de L.
    fn reorder(&mut self, graph: &mut Graph);

    // ── Fase numérica (una vez por evaluación de f(θ) en BFGS) ───────────────

    /// Evalúa Q(θ) y la almacena en formato CSC reordenado.
    ///
    /// Precondición: `reorder` ya fue llamado con el mismo grafo.
    /// Si no, el comportamiento es indefinido (posiblemente panic en debug).
    fn build(&mut self, graph: &Graph, qfunc: &dyn QFunc, theta: &[f64]);

    /// Factorización numérica de Cholesky: Q = L Lᵀ.
    ///
    /// # Errors
    /// Devuelve `InlaError::NotPositiveDefinite` si Q no es PD.
    /// Esto ocurre normalmente durante BFGS cuando θ está fuera del dominio;
    /// el optimizador debe tratar este error como "f(θ) = +∞".
    fn factorize(&mut self) -> Result<(), InlaError>;

    // ── Resultados (tras factorize exitoso) ───────────────────────────────────

    /// log|Q| = 2 · Σᵢ log(Lᵢᵢ).
    ///
    /// Gratis de la factorización — no hay costo adicional.
    ///
    /// # Panics (debug)
    /// Si `factorize` no fue llamado antes.
    fn log_determinant(&self) -> f64;

    /// Resuelve Q·x = b en el espacio reordenado.
    ///
    /// Operación: b ← L⁻ᵀ L⁻¹ b (forward + backward substitution).
    /// El resultado se escribe in-place en `rhs`.
    ///
    /// # Panics (debug)
    /// Si `factorize` no fue llamado antes.
    fn solve_llt(&self, rhs: &mut [f64]);

    /// Inversa seleccionada: Q⁻¹ evaluada en el patrón de L.
    ///
    /// Implementa el algoritmo de Takahashi-Fagan-Chin (1973), también
    /// conocido como "supernodal selected inversion" en el contexto de INLA.
    ///
    /// **Solo calcula las entradas (i,j) con L(i,j) ≠ 0.**
    /// Las entradas fuera del patrón son cero en Q pero no en Q⁻¹;
    /// INLA las ignora porque solo necesita las varianzas marginales
    /// (diagonal) y covarianzas locales.
    ///
    /// # Por qué `&mut self`
    /// El algoritmo modifica L in-place durante el cálculo.
    /// La factorización Cholesky no es reutilizable después de esta llamada.
    ///
    /// # Errors
    /// `InlaError::SolverNotInitialized` si no se llamó `factorize` antes.
    fn selected_inverse(&mut self) -> Result<SpMat, InlaError>;
}

// ── Implementación concreta (Fase A.3) ────────────────────────────────────────
// `FaerSolver` se añade en src/solver/faer_solver.rs en Fase A.3.
