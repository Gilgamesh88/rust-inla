//! Familias de verosimilitud p(y | η, θ).
//!
//! Equivalente Rust de la sección likelihood de `models.R` e `inla.c`.
//!
//! ## Contrato matemático
//!
//! Cada familia implementa `LogLikelihood::evaluate()` que calcula:
//!
//!   logll[i] = log p(yᵢ | ηᵢ, θ)
//!
//! donde ηᵢ es el predictor lineal (efecto latente + efectos fijos).
//!
//! ## Familias implementadas
//!
//! | Familia   | Link     | θ extra        | Uso típico            |
//! |-----------|----------|----------------|-----------------------|
//! | Gaussian  | identity | [log τ]        | respuesta continua    |
//! | Poisson   | log      | ninguno        | frecuencia siniestros |
//! | Gamma     | log      | [log shape]    | severidad siniestros  |


/// Función de enlace η → μ.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkFunction {
    Identity,
    Log,
    Logit,
}

impl LinkFunction {
    /// Aplica la función inversa: η → μ
    #[inline]
    pub fn inverse(&self, eta: f64) -> f64 {
        match self {
            LinkFunction::Identity => eta,
            LinkFunction::Log     => eta.exp(),
            LinkFunction::Logit   => 1.0 / (1.0 + (-eta).exp()),
        }
    }
}

/// Contrato de cada familia de verosimilitud.
pub trait LogLikelihood: Send + Sync {
    /// Evalúa log p(yᵢ | ηᵢ, θ) para cada observación.
    ///
    /// - `logll`: slice de salida, mismo largo que `eta` e `y`
    /// - `eta`:   predictores lineales ηᵢ
    /// - `y`:     observaciones
    /// - `theta`: hiperparámetros de la likelihood (puede ser vacío)
    fn evaluate(&self, logll: &mut [f64], eta: &[f64], y: &[f64], theta: &[f64]);

    /// Función de enlace de esta familia.
    fn link(&self) -> LinkFunction;

    /// Número de hiperparámetros propios de la likelihood.
    fn n_hyperparams(&self) -> usize;

    /// Evalúa analíticamente la primera derivada (gradiente) y 
    /// el negativo de la segunda derivada (curvatura observada) respecto a ηᵢ.
    fn gradient_and_curvature(&self, grad: &mut [f64], curv: &mut [f64], eta: &[f64], y: &[f64], theta: &[f64]);
}

// ── Gaussian ──────────────────────────────────────────────────────────────────

/// y ~ N(η, 1/τ)
///
/// log p(y|η,τ) = 0.5·log(τ) - 0.5·τ·(y-η)² - 0.5·log(2π)
///
/// θ[0] = log τ  (log-precisión del ruido)
pub struct GaussianLikelihood;

impl LogLikelihood for GaussianLikelihood {
    fn evaluate(&self, logll: &mut [f64], eta: &[f64], y: &[f64], theta: &[f64]) {
        let tau     = theta[0].exp();
        let log_tau = theta[0];
        let log2pi  = std::f64::consts::TAU.ln(); // ln(2π)

        for ((ll, &ei), &yi) in logll.iter_mut().zip(eta).zip(y) {
            let resid = yi - ei;
            *ll = 0.5 * log_tau - 0.5 * tau * resid * resid - 0.5 * log2pi;
        }
    }

    fn link(&self) -> LinkFunction { LinkFunction::Identity }
    fn n_hyperparams(&self) -> usize { 1 }

    fn gradient_and_curvature(&self, grad: &mut [f64], curv: &mut [f64], eta: &[f64], y: &[f64], theta: &[f64]) {
        let tau = theta[0].exp();
        for i in 0..eta.len() {
            grad[i] = tau * (y[i] - eta[i]);
            curv[i] = tau;
        }
    }
}

// ── Poisson ───────────────────────────────────────────────────────────────────

/// y ~ Poisson(exp(η))
///
/// log p(y|η) = y·η - exp(η) - log(y!)
///
/// Sin hiperparámetros propios (θ vacío).
pub struct PoissonLikelihood;

impl LogLikelihood for PoissonLikelihood {
    fn evaluate(&self, logll: &mut [f64], eta: &[f64], y: &[f64], _theta: &[f64]) {
        for ((ll, &ei), &yi) in logll.iter_mut().zip(eta).zip(y) {
            // log(y!) usando la aproximación exacta por lgamma
            // lgamma(y+1) = log(y!) para y entero
            let safe_eta = ei.clamp(-50.0, 50.0);
            let log_y_factorial = statrs::function::gamma::ln_gamma(yi + 1.0);
            *ll = yi * safe_eta - safe_eta.exp() - log_y_factorial;
        }
    }

    fn link(&self) -> LinkFunction { LinkFunction::Log }
    fn n_hyperparams(&self) -> usize { 0 }

    fn gradient_and_curvature(&self, grad: &mut [f64], curv: &mut [f64], eta: &[f64], y: &[f64], _theta: &[f64]) {
        for i in 0..eta.len() {
            // Clamp eta to prevent Inf/NaN in exp() during wild Newton steps
            let safe_eta = eta[i].clamp(-50.0, 50.0);
            let lambda = safe_eta.exp();
            grad[i] = y[i] - lambda;
            curv[i] = lambda;
        }
    }
}

// ── Gamma ─────────────────────────────────────────────────────────────────────

/// y ~ Gamma(shape=φ, rate=φ/exp(η))
///
/// E[y] = exp(η),  Var[y] = exp(η)²/φ
///
/// log p(y|η,φ) = φ·log(φ) - φ·log(μ) + (φ-1)·log(y) - φ·y/μ - log Γ(φ)
///
/// donde μ = exp(η).
///
/// θ[0] = log φ  (log-shape)
pub struct GammaLikelihood;

impl LogLikelihood for GammaLikelihood {
    fn evaluate(&self, logll: &mut [f64], eta: &[f64], y: &[f64], theta: &[f64]) {
        let phi     = theta[0].exp();
        let log_phi = theta[0];
        let log_gamma_phi = statrs::function::gamma::ln_gamma(phi);

        for ((ll, &ei), &yi) in logll.iter_mut().zip(eta).zip(y) {
            let safe_eta = ei.clamp(-50.0, 50.0);
            let mu = safe_eta.exp(); // link log: μ = exp(η)
            *ll = phi * log_phi
                - phi * mu.ln()
                + (phi - 1.0) * yi.ln()
                - phi * yi / mu
                - log_gamma_phi;
        }
    }

    fn link(&self) -> LinkFunction { LinkFunction::Log }
    fn n_hyperparams(&self) -> usize { 1 }

    fn gradient_and_curvature(&self, grad: &mut [f64], curv: &mut [f64], eta: &[f64], y: &[f64], theta: &[f64]) {
        let phi = theta[0].clamp(-50.0, 50.0).exp();
        for i in 0..eta.len() {
            let safe_eta = eta[i].clamp(-50.0, 50.0);
            let mu = safe_eta.exp();
            grad[i] = phi * (y[i] / mu - 1.0);
            curv[i] = phi * y[i] / mu;
        }
    }
}

// ── Zero-Inflated Poisson (ZIP) Type-1 ─────────────────────────────────────────

/// y ~ ZIP(p, μ)
/// 
/// p = logit^{-1}(θ_0) (probabilidad de exceso de ceros)
/// μ = exp(η) (media de Poisson)
///
/// log p(y|η,p) = 
///   y=0: log(p + (1-p)e^{-μ})
///   y>0: log(1-p) + y·log(μ) - μ - log(y!)
pub struct ZipLikelihood;

impl LogLikelihood for ZipLikelihood {
    fn evaluate(&self, logll: &mut [f64], eta: &[f64], y: &[f64], theta: &[f64]) {
        let p = 1.0 / (1.0 + (-theta[0]).exp());
        
        for ((ll, &ei), &yi) in logll.iter_mut().zip(eta).zip(y) {
            let safe_eta = ei.clamp(-50.0, 50.0);
            let mu = safe_eta.exp();
            
            if yi == 0.0 {
                let p0_pois = (-mu).exp();
                *ll = (p + (1.0 - p) * p0_pois).ln();
            } else {
                let log_y_factorial = statrs::function::gamma::ln_gamma(yi + 1.0);
                *ll = (1.0 - p).ln() + yi * safe_eta - mu - log_y_factorial;
            }
        }
    }

    fn link(&self) -> LinkFunction { LinkFunction::Log }
    fn n_hyperparams(&self) -> usize { 1 }

    fn gradient_and_curvature(&self, grad: &mut [f64], curv: &mut [f64], eta: &[f64], y: &[f64], theta: &[f64]) {
        let p = 1.0 / (1.0 + (-theta[0]).exp());
        
        for i in 0..eta.len() {
            let safe_eta = eta[i].clamp(-50.0, 50.0);
            let mu = safe_eta.exp();
            
            if y[i] == 0.0 {
                let p0_pois = (-mu).exp();
                let l0 = p + (1.0 - p) * p0_pois;
                // w is the posterior probability that a 0 came from Poisson
                let w = (1.0 - p) * p0_pois / l0;
                
                grad[i] = -mu * w;
                // Observed Fisher info for ZIP at y=0: w·μ·(1 - μ·(1-w)).
                // Can be negative for large μ and small p — fall back to
                // expected Fisher info (w·μ) to keep IRLS positive-definite.
                curv[i] = w * mu * (1.0 - mu * (1.0 - w));
                if curv[i] <= 0.0 {
                    curv[i] = w * mu; // expected Fisher info fallback
                }
            } else {
                grad[i] = y[i] - mu;
                curv[i] = mu;
            }
        }
    }
}

// ── Tweedie (Saddlepoint Approximation) ───────────────────────────────────────

/// y ~ Tweedie(μ, φ, p)
/// 
/// φ = exp(θ_0) (dispersion)
/// p = 1.0 + logit^{-1}(θ_1) (power, bounded between 1 and 2)
/// μ = exp(η)
pub struct TweedieLikelihood;

impl LogLikelihood for TweedieLikelihood {
    fn evaluate(&self, logll: &mut [f64], eta: &[f64], y: &[f64], theta: &[f64]) {
        let phi = theta[0].exp();
        // Clamp p_power away from singularities: (1-p) and (2-p) are denominators.
        let p_power = (1.0 + 1.0 / (1.0 + (-theta[1]).exp()))
            .clamp(1.0 + 1e-6, 2.0 - 1e-6);

        for ((ll, &ei), &yi) in logll.iter_mut().zip(eta).zip(y) {
            let safe_eta = ei.clamp(-50.0, 50.0);
            let mu = safe_eta.exp();
            
            if yi == 0.0 {
                *ll = - (mu.powf(2.0 - p_power)) / (phi * (2.0 - p_power));
            } else {
                let d = 2.0 * (
                    (yi.powf(2.0 - p_power)) / ((1.0 - p_power) * (2.0 - p_power))
                    - (yi * mu.powf(1.0 - p_power)) / (1.0 - p_power)
                    + (mu.powf(2.0 - p_power)) / (2.0 - p_power)
                );
                
                let log_term = -0.5 * (2.0 * std::f64::consts::PI * phi * yi.powf(p_power)).ln();
                *ll = log_term - d / (2.0 * phi);
            }
        }
    }

    fn link(&self) -> LinkFunction { LinkFunction::Log }
    fn n_hyperparams(&self) -> usize { 2 }

    fn gradient_and_curvature(&self, grad: &mut [f64], curv: &mut [f64], eta: &[f64], y: &[f64], theta: &[f64]) {
        let phi = theta[0].exp();
        let p_power = (1.0 + 1.0 / (1.0 + (-theta[1]).exp()))
            .clamp(1.0 + 1e-6, 2.0 - 1e-6);

        for i in 0..eta.len() {
            let safe_eta = eta[i].clamp(-50.0, 50.0);
            let mu = safe_eta.exp();
            
            if y[i] == 0.0 {
                grad[i] = - (mu.powf(2.0 - p_power)) / phi;
                curv[i] = (2.0 - p_power) * mu.powf(2.0 - p_power) / phi;
            } else {
                grad[i] = (y[i] * mu.powf(1.0 - p_power) - mu.powf(2.0 - p_power)) / phi;
                // Observed Fisher information for Tweedie y>0.
                curv[i] = ((2.0 - p_power) * mu.powf(2.0 - p_power)
                    - y[i] * (1.0 - p_power) * mu.powf(1.0 - p_power))
                    / phi;
            }
            
            // To ensure positive definiteness in non-convex regions, fallback to expected info if curvature is negative
            if curv[i] <= 0.0 {
                curv[i] = mu.powf(2.0 - p_power) / phi;
            }
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use std::f64::consts::PI;

    // ── Gaussian ──────────────────────────────────────────────────────────────

    #[test]
    fn gaussian_at_mean_is_maximum() {
        // Cuando y = η, el residuo es 0 → log-lik máximo para ese τ
        let lik = GaussianLikelihood;
        let tau = 2.0_f64;
        let eta = vec![3.0];
        let y   = vec![3.0]; // y = η → residuo cero
        let mut ll1 = vec![0.0];
        lik.evaluate(&mut ll1, &eta, &y, &[tau.ln()]);

        let y2   = vec![3.5]; // y ≠ η → menor log-lik
        let mut ll2 = vec![0.0];
        lik.evaluate(&mut ll2, &eta, &y2, &[tau.ln()]);

        assert!(ll1[0] > ll2[0], "log-lik en la media debe ser mayor");
    }

    #[test]
    fn gaussian_formula_manual() {
        // log p(y|η,τ) = 0.5·log(τ) - 0.5·τ·(y-η)² - 0.5·log(2π)
        let lik = GaussianLikelihood;
        let tau = 1.0_f64;
        let eta = vec![0.0];
        let y   = vec![1.0];
        let mut ll = vec![0.0];
        lik.evaluate(&mut ll, &eta, &y, &[tau.ln()]);

        // Manual: 0.5·log(1) - 0.5·1·1² - 0.5·log(2π) = -0.5·log(2π) - 0.5
        let expected = -0.5 * (2.0 * PI).ln() - 0.5;
        assert_abs_diff_eq!(ll[0], expected, epsilon = 1e-12);
    }

    #[test]
    fn gaussian_higher_tau_penalizes_residual_more() {
        let lik = GaussianLikelihood;
        let eta = vec![0.0];
        let y   = vec![1.0]; // residuo = 1

        let mut ll_low  = vec![0.0];
        let mut ll_high = vec![0.0];
        lik.evaluate(&mut ll_low,  &eta, &y, &[0.0]); // tau = 1
        lik.evaluate(&mut ll_high, &eta, &y, &[2.0]); // tau = e² ≈ 7.4

        // Mayor tau → mayor penalización por residuo
        assert!(ll_low[0] > ll_high[0]);
    }

    // ── Poisson ───────────────────────────────────────────────────────────────

    #[test]
    fn poisson_y0_formula() {
        // y=0: log p(0|η) = -exp(η)  (log(0!) = 0, 0·η = 0)
        let lik = PoissonLikelihood;
        let eta = vec![1.0];
        let y   = vec![0.0];
        let mut ll = vec![0.0];
        lik.evaluate(&mut ll, &eta, &y, &[]);

        assert_abs_diff_eq!(ll[0], -1.0_f64.exp(), epsilon = 1e-12);
    }

    #[test]
    fn poisson_y1_formula() {
        // y=1, η=0 (μ=1): log p(1|η=0) = 1·0 - 1 - log(1!) = -1
        let lik = PoissonLikelihood;
        let eta = vec![0.0];
        let y   = vec![1.0];
        let mut ll = vec![0.0];
        lik.evaluate(&mut ll, &eta, &y, &[]);

        assert_abs_diff_eq!(ll[0], -1.0, epsilon = 1e-10);
    }

    #[test]
    fn poisson_mode_at_y_equals_mu() {
        // Para μ = exp(η) = y, la log-lik es máxima
        let lik = PoissonLikelihood;
        let mu  = 5.0_f64;
        let eta = vec![mu.ln()]; // η = log(μ)

        let y_mode  = vec![5.0]; // y = μ (modo)
        let y_other = vec![3.0];

        let mut ll_mode  = vec![0.0];
        let mut ll_other = vec![0.0];
        lik.evaluate(&mut ll_mode,  &eta, &y_mode,  &[]);
        lik.evaluate(&mut ll_other, &eta, &y_other, &[]);

        assert!(ll_mode[0] > ll_other[0]);
    }

    // ── Gamma ─────────────────────────────────────────────────────────────────

    #[test]
    fn gamma_link_is_log() {
        assert_eq!(GammaLikelihood.link(), LinkFunction::Log);
    }

    #[test]
    fn gamma_n_hyperparams() {
        assert_eq!(GammaLikelihood.n_hyperparams(), 1);
    }

    #[test]
    fn gamma_higher_shape_less_variance() {
        // Mayor φ → distribución más concentrada → mayor log-lik en la media
        let lik = GammaLikelihood;
        let mu  = 2.0_f64;
        let eta = vec![mu.ln()];
        let y   = vec![mu]; // y = media

        let mut ll_low  = vec![0.0];
        let mut ll_high = vec![0.0];
        lik.evaluate(&mut ll_low,  &eta, &y, &[1.0_f64.ln()]); // φ=1 (exp)
        lik.evaluate(&mut ll_high, &eta, &y, &[5.0_f64.ln()]); // φ=5

        // φ=5 → más concentrada → mayor log-lik en la media
        assert!(ll_high[0] > ll_low[0]);
    }

    #[test]
    fn gamma_phi1_is_exponential() {
        // Con φ=1, Gamma = Exponencial con media μ
        // log p(y|η,φ=1) = -log(μ) - y/μ = -η - y·exp(-η)
        let lik = GammaLikelihood;
        let mu  = 3.0_f64;
        let eta = vec![mu.ln()];
        let y   = vec![2.0_f64];
        let mut ll = vec![0.0];
        lik.evaluate(&mut ll, &eta, &y, &[0.0]); // log(φ=1) = 0

        let expected = -mu.ln() - y[0] / mu;
        assert_abs_diff_eq!(ll[0], expected, epsilon = 1e-10);
    }

    // ── Link functions ────────────────────────────────────────────────────────

    #[test]
    fn link_identity_roundtrip() {
        let l = LinkFunction::Identity;
        assert_abs_diff_eq!(l.inverse(3.0), 3.0, epsilon = 1e-12);
    }

    #[test]
    fn link_log_inverse_is_exp() {
        let l = LinkFunction::Log;
        assert_abs_diff_eq!(l.inverse(2.0), 2.0_f64.exp(), epsilon = 1e-12);
    }

    #[test]
    fn link_logit_at_zero_is_half() {
        let l = LinkFunction::Logit;
        assert_abs_diff_eq!(l.inverse(0.0), 0.5, epsilon = 1e-12);
    }
}
