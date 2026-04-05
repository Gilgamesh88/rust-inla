//! # rust-inla
//!
//! Port de R-INLA a Rust puro. Inferencia Bayesiana Aproximada para modelos
//! de campo aleatorio gaussiano de Markov (GMRF) vía el algoritmo INLA.
//!
//! ## Arquitectura de módulos
//!
//! ```text
//! inference/   ← InlaEngine::run() — integra todo
//!   ├── optimizer/   ← BFGS sobre θ (argmin)
//!   ├── problem/     ← constructor central: μ, log|Q|, Q⁻¹
//!   │    ├── graph/      ← dispersidad de Q + patrón de L
//!   │    ├── models/     ← trait QFunc: iid, rw1, rw2, ar1
//!   │    └── solver/     ← Cholesky faer + selected inverse
//!   ├── likelihood/  ← trait LogLikelihood: Gaussian, Poisson, Gamma...
//!   ├── integrator/  ← Gauss-Kronrod 15pts
//!   ├── density/     ← Gaussian / SCGAUSSIAN
//!   └── marginal/    ← zmarginal, emarginal, tmarginal
//! ```
//!
//! ## Estado de implementación
//!
//! Fase 0 (fixtures R-INLA): ✓ completa
//! Fase A (core): en progreso
//! Fase B (inference): pendiente

// ── Módulos públicos ──────────────────────────────────────────────────────────

pub mod error;
pub mod graph;
pub mod likelihood;
pub mod bindings;
pub mod models;
pub mod solver;

// Módulos con implementación pendiente (stubs vacíos por ahora)
pub mod density;
pub mod inference;
pub mod integrator;
pub mod marginal;
pub mod optimizer;
pub mod problem;

// ── Re-exportaciones de conveniencia ─────────────────────────────────────────

pub use error::InlaError;
pub use graph::Graph;
pub use likelihood::{LinkFunction, LogLikelihood};
pub use models::QFunc;
pub use solver::{SpIdx, SpMat, SparseSolver, FaerSolver};
