//! Densidades marginales aproximadas.
//!
//! Equivalente Rust de `density.h/c` (2413 líneas).
//!
//! ## Tipos de densidad
//!
//! - `Density::Gaussian` — aproximación gaussiana simple
//! - `Density::ScGaussian` — Gaussiana + corrección log-spline (default INLA)
//!
//! La densidad default de INLA es SCGAUSSIAN (Skew-Corrected Gaussian),
//! que añade una corrección de asimetría a la aproximación Laplace.

// Fase B.4: implementar enum Density { Gaussian, ScGaussian }
