//! Constructor central: dado θ, produce μ y log|Q|.
//!
//! Equivalente Rust de `problem-setup.h/c` (2500 líneas).
//!
//! ## Responsabilidad de Problem
//!
//! Por cada evaluación de f(θ) dentro de BFGS:
//!
//! ```text
//! Problem::eval(theta)
//!   ├── solver.build(graph, qfunc, theta)   // rellena Q(θ)
//!   ├── solver.factorize()                  // Q = LLᵀ
//!   ├── log_det = solver.log_determinant()  // 2·Σlog(diag(L))
//!   ├── mu = solver.solve_llt(b)            // media posterior
//!   └── [si se necesita Qinv]
//!       qinv = solver.selected_inverse()    // almacenado aquí
//! ```
//!
//! Problem posee el `SpMat` de Q⁻¹ (devuelto por el solver),
//! rompiendo la dependencia circular del diseño original.

// Fase A.4: implementar Problem struct
