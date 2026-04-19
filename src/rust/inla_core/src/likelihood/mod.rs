//! Familias de verosimilitud p(y | Î·, Î¸).
//!
//! Equivalente Rust de la secciÃ³n likelihood de `models.R` e `inla.c`.
//!
//! ## Contrato matemÃ¡tico
//!
//! Cada familia implementa `LogLikelihood::evaluate()` que calcula:
//!
//!   logll[i] = log p(yáµ¢ | Î·áµ¢, Î¸)
//!
//! donde Î·áµ¢ es el predictor lineal (efecto latente + efectos fijos).
//!
//! ## Familias implementadas
//!
//! | Familia   | Link     | Î¸ extra        | Uso tÃ­pico            |
//! |-----------|----------|----------------|-----------------------|
//! | Gaussian  | identity | [log Ï„]        | respuesta continua    |
//! | Poisson   | log      | ninguno        | frecuencia siniestros |
//! | Gamma     | log      | [log shape]    | severidad siniestros  |

/// FunciÃ³n de enlace Î· â†’ Î¼.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkFunction {
    Identity,
    Log,
    Logit,
}

impl LinkFunction {
    /// Aplica la funciÃ³n inversa: Î· â†’ Î¼
    #[inline]
    pub fn inverse(&self, eta: f64) -> f64 {
        match self {
            LinkFunction::Identity => eta,
            LinkFunction::Log => eta.exp(),
            LinkFunction::Logit => 1.0 / (1.0 + (-eta).exp()),
        }
    }
}

#[inline]
fn loggamma_on_log_scale(theta: f64, shape: f64, rate: f64) -> f64 {
    shape * rate.ln() - statrs::function::gamma::ln_gamma(shape) + shape * theta
        - rate * theta.exp()
}

#[inline]
fn gaussian_prior_kernel(theta: f64, mean: f64, precision: f64) -> f64 {
    0.5 * (precision.ln() - std::f64::consts::TAU.ln())
        - 0.5 * precision * (theta - mean) * (theta - mean)
}

/// Contrato de cada familia de verosimilitud.
pub trait LogLikelihood: Send + Sync {
    /// EvalÃºa log p(yáµ¢ | Î·áµ¢, Î¸) para cada observaciÃ³n.
    ///
    /// - `logll`: slice de salida, mismo largo que `eta` e `y`
    /// - `eta`:   predictores lineales Î·áµ¢
    /// - `y`:     observaciones
    /// - `theta`: hiperparÃ¡metros de la likelihood (puede ser vacÃ­o)
    fn evaluate(&self, logll: &mut [f64], eta: &[f64], y: &[f64], theta: &[f64]);

    /// FunciÃ³n de enlace de esta familia.
    fn link(&self) -> LinkFunction;

    /// NÃºmero de hiperparÃ¡metros propios de la likelihood.
    fn n_hyperparams(&self) -> usize;

    /// EvalÃºa analÃ­ticamente la primera derivada (gradiente) y
    /// el negativo de la segunda derivada (curvatura observada) respecto a Î·áµ¢.
    fn gradient_and_curvature(
        &self,
        grad: &mut [f64],
        curv: &mut [f64],
        eta: &[f64],
        y: &[f64],
        theta: &[f64],
    );

    /// Log-prior density on the likelihood's internal theta scale.
    fn log_prior(&self, theta: &[f64]) -> f64 {
        theta
            .iter()
            .map(|&th| loggamma_on_log_scale(th, 1.0, 5e-5))
            .sum()
    }
}

//  Gaussiana

/// y ~ N(Î·, 1/Ï„)
///
/// log p(y|Î·,Ï„) = 0.5Â·log(Ï„) - 0.5Â·Ï„Â·(y-Î·)Â² - 0.5Â·log(2Ï€)
///
/// Î¸[0] = log Ï„  (log-precisiÃ³n del ruido)
pub struct GaussianLikelihood;

impl LogLikelihood for GaussianLikelihood {
    fn evaluate(&self, logll: &mut [f64], eta: &[f64], y: &[f64], theta: &[f64]) {
        let tau = theta[0].exp();
        let log_tau = theta[0];
        let log2pi = std::f64::consts::TAU.ln(); // ln(2Ï€)

        for ((ll, &ei), &yi) in logll.iter_mut().zip(eta).zip(y) {
            if yi.is_nan() {
                *ll = 0.0;
                continue;
            }
            let resid = yi - ei;
            *ll = 0.5 * log_tau - 0.5 * tau * resid * resid - 0.5 * log2pi;
        }
    }

    fn link(&self) -> LinkFunction {
        LinkFunction::Identity
    }
    fn n_hyperparams(&self) -> usize {
        1
    }

    fn gradient_and_curvature(
        &self,
        grad: &mut [f64],
        curv: &mut [f64],
        eta: &[f64],
        y: &[f64],
        theta: &[f64],
    ) {
        let tau = theta[0].exp();
        for i in 0..eta.len() {
            if y[i].is_nan() {
                grad[i] = 0.0;
                curv[i] = 0.0;
                continue;
            }
            grad[i] = tau * (y[i] - eta[i]);
            curv[i] = tau;
        }
    }

    fn log_prior(&self, theta: &[f64]) -> f64 {
        loggamma_on_log_scale(theta[0], 1.0, 5e-5)
    }
}

// â”€â”€ Poisson â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// y ~ Poisson(exp(Î·))
///
/// log p(y|Î·) = yÂ·Î· - exp(Î·) - log(y!)
///
/// Sin hiperparÃ¡metros propios (Î¸ vacÃ­o).
pub struct PoissonLikelihood;

impl LogLikelihood for PoissonLikelihood {
    fn evaluate(&self, logll: &mut [f64], eta: &[f64], y: &[f64], _theta: &[f64]) {
        for ((ll, &ei), &yi) in logll.iter_mut().zip(eta).zip(y) {
            if yi.is_nan() {
                *ll = 0.0;
                continue;
            }
            // log(y!) usando la aproximaciÃ³n exacta por lgamma
            // lgamma(y+1) = log(y!) para y entero
            let safe_eta = ei.clamp(-50.0, 50.0);
            let log_y_factorial = statrs::function::gamma::ln_gamma(yi + 1.0);
            *ll = yi * safe_eta - safe_eta.exp() - log_y_factorial;
        }
    }

    fn link(&self) -> LinkFunction {
        LinkFunction::Log
    }
    fn n_hyperparams(&self) -> usize {
        0
    }

    fn gradient_and_curvature(
        &self,
        grad: &mut [f64],
        curv: &mut [f64],
        eta: &[f64],
        y: &[f64],
        _theta: &[f64],
    ) {
        for i in 0..eta.len() {
            if y[i].is_nan() {
                grad[i] = 0.0;
                curv[i] = 0.0;
                continue;
            }
            // Clamp eta to prevent Inf/NaN in exp() during wild Newton steps
            let safe_eta = eta[i].clamp(-50.0, 50.0);
            let lambda = safe_eta.exp();
            grad[i] = y[i] - lambda;
            curv[i] = lambda;
        }
    }

    fn log_prior(&self, _theta: &[f64]) -> f64 {
        0.0
    }
}

// â”€â”€ Gamma â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// y ~ Gamma(shape=Ï†, rate=Ï†/exp(Î·))
///
/// E[y] = exp(Î·),  Var[y] = exp(Î·)Â²/Ï†
///
/// log p(y|Î·,Ï†) = Ï†Â·log(Ï†) - Ï†Â·log(Î¼) + (Ï†-1)Â·log(y) - Ï†Â·y/Î¼ - log Î“(Ï†)
///
/// donde Î¼ = exp(Î·).
///
/// Î¸[0] = log Ï†  (log-shape)
pub struct GammaLikelihood;

impl LogLikelihood for GammaLikelihood {
    fn evaluate(&self, logll: &mut [f64], eta: &[f64], y: &[f64], theta: &[f64]) {
        let phi = theta[0].exp();
        let log_phi = theta[0];
        let log_gamma_phi = statrs::function::gamma::ln_gamma(phi);

        for ((ll, &ei), &yi) in logll.iter_mut().zip(eta).zip(y) {
            if yi.is_nan() {
                *ll = 0.0;
                continue;
            }
            let safe_eta = ei.clamp(-50.0, 50.0);
            let mu = safe_eta.exp(); // link log: Î¼ = exp(Î·)
            *ll = phi * log_phi - phi * mu.ln() + (phi - 1.0) * yi.ln()
                - phi * yi / mu
                - log_gamma_phi;
        }
    }

    fn link(&self) -> LinkFunction {
        LinkFunction::Log
    }
    fn n_hyperparams(&self) -> usize {
        1
    }

    fn gradient_and_curvature(
        &self,
        grad: &mut [f64],
        curv: &mut [f64],
        eta: &[f64],
        y: &[f64],
        theta: &[f64],
    ) {
        let phi = theta[0].clamp(-50.0, 50.0).exp();
        for i in 0..eta.len() {
            if y[i].is_nan() {
                grad[i] = 0.0;
                curv[i] = 0.0;
                continue;
            }
            let safe_eta = eta[i].clamp(-50.0, 50.0);
            let mu = safe_eta.exp();
            grad[i] = phi * (y[i] / mu - 1.0);
            curv[i] = phi * y[i] / mu;
        }
    }

    fn log_prior(&self, theta: &[f64]) -> f64 {
        loggamma_on_log_scale(theta[0], 1.0, 0.01)
    }
}

//  Zero-Inflated Poisson (ZIP) Type-1

/// y ~ ZIP(p, Î¼)
///
/// p = logit^{-1}(Î¸_0) (probabilidad de exceso de ceros)
/// Î¼ = exp(Î·) (media de Poisson)
///
/// log p(y|Î·,p) =
///   y=0: log(p + (1-p)e^{-Î¼})
///   y>0: log(1-p) + yÂ·log(Î¼) - Î¼ - log(y!)
pub struct ZipLikelihood;

impl LogLikelihood for ZipLikelihood {
    fn evaluate(&self, logll: &mut [f64], eta: &[f64], y: &[f64], theta: &[f64]) {
        let p = 1.0 / (1.0 + (-theta[0]).exp());

        for ((ll, &ei), &yi) in logll.iter_mut().zip(eta).zip(y) {
            if yi.is_nan() {
                *ll = 0.0;
                continue;
            }
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

    fn link(&self) -> LinkFunction {
        LinkFunction::Log
    }
    fn n_hyperparams(&self) -> usize {
        1
    }

    fn gradient_and_curvature(
        &self,
        grad: &mut [f64],
        curv: &mut [f64],
        eta: &[f64],
        y: &[f64],
        theta: &[f64],
    ) {
        let p = 1.0 / (1.0 + (-theta[0]).exp());

        for i in 0..eta.len() {
            if y[i].is_nan() {
                grad[i] = 0.0;
                curv[i] = 0.0;
                continue;
            }
            let safe_eta = eta[i].clamp(-50.0, 50.0);
            let mu = safe_eta.exp();

            if y[i] == 0.0 {
                let p0_pois = (-mu).exp();
                let l0 = p + (1.0 - p) * p0_pois;
                // w is the posterior probability that a 0 came from Poisson
                let w = (1.0 - p) * p0_pois / l0;

                grad[i] = -mu * w;
                curv[i] = w * mu * (1.0 - mu * (1.0 - w));
                if curv[i] <= 0.0 {
                    // In non-concave ZIP regions, reuse expected information so
                    // the IRLS weight matrix stays positive definite.
                    let expected_info = (1.0 - p) * mu - mu * mu * p * w;
                    curv[i] = expected_info.max(1e-12);
                }
            } else {
                grad[i] = y[i] - mu;
                curv[i] = mu;
            }
        }
    }

    fn log_prior(&self, theta: &[f64]) -> f64 {
        gaussian_prior_kernel(theta[0], -1.0, 0.2)
    }
}

//  Tweedie (Saddlepoint Approximation)

/// y ~ Tweedie(Î¼, Ï†, p)
///
/// Ï† = exp(Î¸_0) (dispersion)
/// p = 1.0 + logit^{-1}(Î¸_1) (power, bounded between 1 and 2)
/// Î¼ = exp(Î·)
pub struct TweedieLikelihood;

impl LogLikelihood for TweedieLikelihood {
    fn evaluate(&self, logll: &mut [f64], eta: &[f64], y: &[f64], theta: &[f64]) {
        let phi = theta[0].exp();
        let p_power = 1.0 + 1.0 / (1.0 + (-theta[1]).exp()); // maps to (1,2)

        for ((ll, &ei), &yi) in logll.iter_mut().zip(eta).zip(y) {
            if yi.is_nan() {
                *ll = 0.0;
                continue;
            }
            let safe_eta = ei.clamp(-50.0, 50.0);
            let mu = safe_eta.exp();

            if yi == 0.0 {
                *ll = -(mu.powf(2.0 - p_power)) / (phi * (2.0 - p_power));
            } else {
                let d = 2.0
                    * ((yi.powf(2.0 - p_power)) / ((1.0 - p_power) * (2.0 - p_power))
                        - (yi * mu.powf(1.0 - p_power)) / (1.0 - p_power)
                        + (mu.powf(2.0 - p_power)) / (2.0 - p_power));

                let log_term = -0.5 * (2.0 * std::f64::consts::PI * phi * yi.powf(p_power)).ln();
                *ll = log_term - d / (2.0 * phi);
            }
        }
    }

    fn link(&self) -> LinkFunction {
        LinkFunction::Log
    }
    fn n_hyperparams(&self) -> usize {
        2
    }

    fn gradient_and_curvature(
        &self,
        grad: &mut [f64],
        curv: &mut [f64],
        eta: &[f64],
        y: &[f64],
        theta: &[f64],
    ) {
        let phi = theta[0].exp();
        let p_power = 1.0 + 1.0 / (1.0 + (-theta[1]).exp());

        for i in 0..eta.len() {
            if y[i].is_nan() {
                grad[i] = 0.0;
                curv[i] = 0.0;
                continue;
            }
            let safe_eta = eta[i].clamp(-50.0, 50.0);
            let mu = safe_eta.exp();

            if y[i] == 0.0 {
                grad[i] = -(mu.powf(2.0 - p_power)) / phi;
                curv[i] = (2.0 - p_power) * mu.powf(2.0 - p_power) / phi;
            } else {
                grad[i] = (y[i] * mu.powf(1.0 - p_power) - mu.powf(2.0 - p_power)) / phi;
                let expected_info = false; // Choose observed info
                if expected_info {
                    curv[i] = mu.powf(2.0 - p_power) / phi;
                } else {
                    curv[i] = ((2.0 - p_power) * mu.powf(2.0 - p_power)
                        - y[i] * (1.0 - p_power) * mu.powf(1.0 - p_power))
                        / phi;
                }
            }

            // To ensure positive definiteness in non-convex regions, fallback to expected info if curvature is negative
            if curv[i] <= 0.0 {
                curv[i] = mu.powf(2.0 - p_power) / phi;
            }
        }
    }
}

//

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use std::f64::consts::PI;

    //  Gaussian
    #[test]
    fn gaussian_at_mean_is_maximum() {
        // Cuando y = Î·, el residuo es 0 â†’ log-lik mÃ¡ximo para ese Ï„
        let lik = GaussianLikelihood;
        let tau = 2.0_f64;
        let eta = vec![3.0];
        let y = vec![3.0]; // y = Î· â†’ residuo cero
        let mut ll1 = vec![0.0];
        lik.evaluate(&mut ll1, &eta, &y, &[tau.ln()]);

        let y2 = vec![3.5]; // y â‰  Î· â†’ menor log-lik
        let mut ll2 = vec![0.0];
        lik.evaluate(&mut ll2, &eta, &y2, &[tau.ln()]);

        assert!(ll1[0] > ll2[0], "log-lik en la media debe ser mayor");
    }

    #[test]
    fn gaussian_formula_manual() {
        // log p(y|Î·,Ï„) = 0.5Â·log(Ï„) - 0.5Â·Ï„Â·(y-Î·)Â² - 0.5Â·log(2Ï€)
        let lik = GaussianLikelihood;
        let tau = 1.0_f64;
        let eta = vec![0.0];
        let y = vec![1.0];
        let mut ll = vec![0.0];
        lik.evaluate(&mut ll, &eta, &y, &[tau.ln()]);

        // Manual: 0.5Â·log(1) - 0.5Â·1Â·1Â² - 0.5Â·log(2Ï€) = -0.5Â·log(2Ï€) - 0.5
        let expected = -0.5 * (2.0 * PI).ln() - 0.5;
        assert_abs_diff_eq!(ll[0], expected, epsilon = 1e-12);
    }

    #[test]
    fn gaussian_higher_tau_penalizes_residual_more() {
        let lik = GaussianLikelihood;
        let eta = vec![0.0];
        let y = vec![1.0]; // residuo = 1

        let mut ll_low = vec![0.0];
        let mut ll_high = vec![0.0];
        lik.evaluate(&mut ll_low, &eta, &y, &[0.0]); // tau = 1
        lik.evaluate(&mut ll_high, &eta, &y, &[2.0]); // tau = eÂ² â‰ˆ 7.4

        // Mayor tau â†’ mayor penalizaciÃ³n por residuo
        assert!(ll_low[0] > ll_high[0]);
    }

    //  Poisson
    #[test]
    fn poisson_y0_formula() {
        // y=0: log p(0|Î·) = -exp(Î·)  (log(0!) = 0, 0Â·Î· = 0)
        let lik = PoissonLikelihood;
        let eta = vec![1.0];
        let y = vec![0.0];
        let mut ll = vec![0.0];
        lik.evaluate(&mut ll, &eta, &y, &[]);

        assert_abs_diff_eq!(ll[0], -1.0_f64.exp(), epsilon = 1e-12);
    }

    #[test]
    fn poisson_y1_formula() {
        // y=1, Î·=0 (Î¼=1): log p(1|Î·=0) = 1Â·0 - 1 - log(1!) = -1
        let lik = PoissonLikelihood;
        let eta = vec![0.0];
        let y = vec![1.0];
        let mut ll = vec![0.0];
        lik.evaluate(&mut ll, &eta, &y, &[]);

        assert_abs_diff_eq!(ll[0], -1.0, epsilon = 1e-10);
    }

    #[test]
    fn poisson_mode_at_y_equals_mu() {
        // Para Î¼ = exp(Î·) = y, la log-lik es mÃ¡xima
        let lik = PoissonLikelihood;
        let mu = 5.0_f64;
        let eta = vec![mu.ln()]; // Î· = log(Î¼)

        let y_mode = vec![5.0]; // y = Î¼ (modo)
        let y_other = vec![3.0];

        let mut ll_mode = vec![0.0];
        let mut ll_other = vec![0.0];
        lik.evaluate(&mut ll_mode, &eta, &y_mode, &[]);
        lik.evaluate(&mut ll_other, &eta, &y_other, &[]);

        assert!(ll_mode[0] > ll_other[0]);
    }

    //  Gamma

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
        // Mayor Ï† â†’ distribuciÃ³n mÃ¡s concentrada â†’ mayor log-lik en la media
        let lik = GammaLikelihood;
        let mu = 2.0_f64;
        let eta = vec![mu.ln()];
        let y = vec![mu]; // y = media

        let mut ll_low = vec![0.0];
        let mut ll_high = vec![0.0];
        lik.evaluate(&mut ll_low, &eta, &y, &[1.0_f64.ln()]); // Ï†=1 (exp)
        lik.evaluate(&mut ll_high, &eta, &y, &[5.0_f64.ln()]); // Ï†=5

        // Ï†=5 â†’ mÃ¡s concentrada â†’ mayor log-lik en la media
        assert!(ll_high[0] > ll_low[0]);
    }

    #[test]
    fn gamma_phi1_is_exponential() {
        // Con Ï†=1, Gamma = Exponencial con media Î¼
        // log p(y|Î·,Ï†=1) = -log(Î¼) - y/Î¼ = -Î· - yÂ·exp(-Î·)
        let lik = GammaLikelihood;
        let mu = 3.0_f64;
        let eta = vec![mu.ln()];
        let y = vec![2.0_f64];
        let mut ll = vec![0.0];
        lik.evaluate(&mut ll, &eta, &y, &[0.0]); // log(Ï†=1) = 0

        let expected = -mu.ln() - y[0] / mu;
        assert_abs_diff_eq!(ll[0], expected, epsilon = 1e-10);
    }

    // â”€â”€ Link functions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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

    #[test]
    fn zip_log_prior_matches_inla_gaussian_density() {
        let theta = [0.430_880_877_129_124_f64];
        let precision = 0.2_f64;
        let expected = 0.5 * (precision.ln() - std::f64::consts::TAU.ln())
            - 0.5 * precision * (theta[0] + 1.0).powi(2);
        assert_abs_diff_eq!(ZipLikelihood.log_prior(&theta), expected, epsilon = 1e-12);
    }

    #[test]
    fn zip_curvature_uses_expected_info_when_observed_curvature_turns_negative() {
        let lik = ZipLikelihood;
        let eta = [1.0_f64];
        let y = [0.0_f64];
        let theta = [4.466_099_f64];
        let mut grad = [0.0_f64];
        let mut curv = [0.0_f64];

        lik.gradient_and_curvature(&mut grad, &mut curv, &eta, &y, &theta);

        let p = 1.0 / (1.0 + (-theta[0]).exp());
        let mu = eta[0].exp();
        let p0_pois = (-mu).exp();
        let l0 = p + (1.0 - p) * p0_pois;
        let w = (1.0 - p) * p0_pois / l0;
        let observed_curv = w * mu * (1.0 - mu * (1.0 - w));
        let expected_info = (1.0 - p) * mu - mu * mu * p * w;

        assert!(observed_curv < 0.0);
        assert_abs_diff_eq!(grad[0], -mu * w, epsilon = 1e-12);
        assert_abs_diff_eq!(curv[0], expected_info.max(1e-12), epsilon = 1e-12);
        assert!(curv[0] > 0.0);
    }

    #[test]
    fn gamma_log_prior_matches_inla_loggamma_density() {
        let theta = [4.605_170_185_988_09_f64];
        let rate = 0.01_f64;
        let expected = rate.ln() + theta[0] - rate * theta[0].exp();
        assert_abs_diff_eq!(GammaLikelihood.log_prior(&theta), expected, epsilon = 1e-12);
    }
}
