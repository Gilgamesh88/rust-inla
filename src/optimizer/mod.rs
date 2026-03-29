//! Optimización de hiperparámetros θ vía BFGS.
//!
//! Equivalente Rust de `domin-interface.h/c` (1596 líneas).
//!
//! Usa el crate `argmin` 0.10. La función objetivo es:
//!   f(θ) = -log p(y|θ) = -log p(y|x,θ) - log p(x|θ) + cte
//!
//! El gradiente se calcula por diferencias finitas (igual que R-INLA).
//! Cada evaluación de f(θ) ejecuta una factorización de Cholesky completa.

// Fase B.2: implementar InlaOptimizer
