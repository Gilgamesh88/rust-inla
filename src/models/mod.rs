//! Modelos de campo aleatorio gaussiano de Markov (GMRF).
//!
//! Equivalente Rust de `hgmrfm.h/c` y la sección de modelos latentes
//! de `models.R`.
//!
//! ## Contrato: trait `QFunc`
//!
//! Cada modelo latente implementa `QFunc`. El trait define dos cosas:
//! - La **estructura** del grafo de dispersidad (fija, independiente de θ)
//! - La **evaluación** de la entrada (i,j) de Q en función de θ
//!
//! ## Por qué `graph()` no recibe `theta`
//!
//! En todos los modelos estándar (iid, RW1, RW2, AR1), el patrón de
//! dispersidad de Q no cambia con los hiperparámetros θ. Solo cambian
//! los *valores* de las entradas. Si `graph` dependiera de θ:
//!
//! - El hash del grafo cambiaría en cada evaluación de BFGS
//! - El caché de reorderings AMD se invalidaría constantemente
//! - Performance: O(n·log n) de AMD en cada llamada a f(θ)
//!
//! Si en el futuro existe un modelo con grafo dinámico, necesita un
//! trait diferente — no una extensión de este.
//!
//! ## Orden de evaluación esperado por Problem
//!
//! ```text
//! // Una sola vez por modelo:
//! let g = qfunc.graph();
//! solver.reorder(g);          // AMD — O(n·log n)
//!
//! // En cada evaluación de f(θ) dentro de BFGS:
//! solver.build(g, &qfunc, theta);   // rellena valores de Q — O(nnz)
//! solver.factorize()?;              // Cholesky — O(nnz(L))
//! let log_det = solver.log_determinant();
//! ```

use crate::graph::Graph;

/// Contrato de cada modelo GMRF latente.
///
/// Los implementadores definen la estructura dispersa de Q y cómo
/// evaluar cada entrada en función de los hiperparámetros θ.
pub trait QFunc: Send + Sync {
    /// Grafo de dispersidad de Q.
    ///
    /// La referencia es `'static` en la práctica (el modelo posee el grafo),
    /// pero el lifetime está ligado a `&self` para evitar `unsafe`.
    ///
    /// **Garantía**: el grafo no cambia durante la vida del modelo.
    /// Si necesitas un grafo que cambia con θ, implementa un trait distinto.
    fn graph(&self) -> &Graph;

    /// Evalúa la entrada Q(i, j) para los hiperparámetros `theta`.
    ///
    /// - Solo se llama para pares (i,j) que son vecinos en `graph()`
    ///   (o la diagonal, i == j).
    /// - Debe ser **simétrica**: eval(i,j,θ) == eval(j,i,θ).
    /// - `theta.len()` == `n_hyperparams()`.
    ///
    /// # Panics
    /// No debe hacer panic. Si θ está fuera de rango, devuelve `f64::INFINITY`
    /// para señalar al optimizador que ese punto no es válido.
    fn eval(&self, i: usize, j: usize, theta: &[f64]) -> f64;

    /// Número de hiperparámetros θ de este modelo.
    ///
    /// - iid:  1 (log-precisión τ)
    /// - RW1:  1 (log-precisión de las diferencias)
    /// - RW2:  1 (idem, orden 2)
    /// - AR1:  2 (log-precisión marginal, arctanh(ρ))
    fn n_hyperparams(&self) -> usize;
}

// ── Implementaciones (Fase A.2) ───────────────────────────────────────────────
// Las structs concretas se añaden en Fase A.2:
//   - IidModel
//   - Rw1Model
//   - Rw2Model
//   - Ar1Model
