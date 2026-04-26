//! Tipo de error central de rust-inla.
//!
//! Todos los módulos devuelven `Result<T, InlaError>`. Usar un único tipo
//! de error evita el anti-patrón de `Box<dyn Error>` en boundaries de módulo
//! y hace que el manejo de errores sea exhaustivo.

/// Errores que puede producir cualquier operación de rust-inla.
#[derive(Debug, thiserror::Error)]
pub enum InlaError {
    // ── Solver ────────────────────────────────────────────────────────────────
    /// Q no es definida positiva. Ocurre si θ está fuera del dominio válido
    /// o si hay una inconsistencia numérica en QFunc::eval.
    #[error("Cholesky failed: Q is not positive definite (theta may be out of range)")]
    NotPositiveDefinite,
    #[error("Cholesky failed: {0}")]
    NotPositiveDefiniteContext(String),

    /// Se llamó log_determinant / solve_llt / selected_inverse antes de
    /// factorize(). El solver requiere el orden: reorder → build → factorize.
    #[error("Solver not initialized — call reorder(), build(), factorize() in order")]
    SolverNotInitialized,

    // ── Optimizador ──────────────────────────────────────────────────────────
    /// BFGS no convergió dentro del límite de evaluaciones.
    #[error("BFGS did not converge: {reason}")]
    ConvergenceFailed { reason: String },

    // ── Fixtures / IO (solo tests) ────────────────────────────────────────────
    /// Error al leer o parsear un fixture JSON de referencia.
    #[error("Fixture error: {0}")]
    Fixture(String),

    // ── Dimensiones ──────────────────────────────────────────────────────────
    /// Incompatibilidad de tamaños entre estructuras.
    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
}
