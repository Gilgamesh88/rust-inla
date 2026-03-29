//! Familias de verosimilitud p(y | η, θ).
//!
//! Equivalente Rust de la sección de likelihood de `models.R` e `inla.c`.
//!
//! ## Familias Fase B core
//!
//! | Familia    | Link    | Parámetros θ extra |
//! |------------|---------|-------------------|
//! | Gaussian   | identity| 1 (log-precisión) |
//! | Poisson    | log     | 0                 |
//! | Gamma      | log     | 1 (log-shape)     |
//! | Binomial   | logit   | 0                 |
//! | NegBinomial| log     | 1 (log-overdispersión) |
//! | Beta       | logit   | 1 (log-precisión) |

/// Función de enlace η → μ (o μ → η en la dirección inversa).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkFunction {
    /// μ = η (Gaussian)
    Identity,
    /// μ = exp(η) (Poisson, Gamma, NegBinomial)
    Log,
    /// μ = 1/(1+exp(-η)) (Binomial, Beta)
    Logit,
}

impl LinkFunction {
    /// Aplica la función de enlace inversa: η → μ.
    #[inline]
    pub fn inverse(&self, eta: f64) -> f64 {
        match self {
            LinkFunction::Identity => eta,
            LinkFunction::Log     => eta.exp(),
            LinkFunction::Logit   => 1.0 / (1.0 + (-eta).exp()),
        }
    }

    /// Aplica la función de enlace directa: μ → η.
    #[inline]
    pub fn forward(&self, mu: f64) -> f64 {
        match self {
            LinkFunction::Identity => mu,
            LinkFunction::Log     => mu.ln(),
            LinkFunction::Logit   => (mu / (1.0 - mu)).ln(),
        }
    }
}

/// Contrato de cada familia de verosimilitud.
///
/// Equivalente al callback de likelihood en `approx-inference.h`.
pub trait LogLikelihood: Send + Sync {
    /// Evalúa log p(yᵢ | ηᵢ, θ) para las observaciones en `y_slice`.
    ///
    /// # Argumentos
    /// - `logll`  — slice de salida donde escribir log p(yᵢ | ηᵢ, θ)
    /// - `eta`    — predictores lineales ηᵢ = xᵢ + βᵀzᵢ
    /// - `y`      — observaciones correspondientes
    /// - `theta`  — hiperparámetros de la likelihood (puede ser vacío)
    ///
    /// # Contrato
    /// `logll.len() == eta.len() == y.len()`
    fn evaluate(
        &self,
        logll: &mut [f64],
        eta:   &[f64],
        y:     &[f64],
        theta: &[f64],
    );

    /// ¿Tiene derivadas analíticas? Si no, Problem usa diferencias finitas.
    /// Por defecto: false (diferencias finitas, como R-INLA).
    fn has_exact_derivatives(&self) -> bool {
        false
    }

    /// Función de enlace de esta familia.
    fn link(&self) -> LinkFunction;
}

// ── Implementaciones (Fase B.1) ───────────────────────────────────────────────
// GaussianLikelihood, PoissonLikelihood, GammaLikelihood, etc.
