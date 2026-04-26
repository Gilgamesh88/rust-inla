//! Representación de densidades marginales aproximadas.
//!
//! Equivalente Rust de `density.h/c` (2413 líneas de C).
//!
//! ## Tipos de densidad
//!
//! - `Gaussian`: aproximación gaussiana simple (media + sd)
//! - `ScGaussian`: Gaussiana + corrección log-spline (default de INLA)
//!   Captura asimetría que la Gaussiana pura pierde.
//!
//! En Fase B.6 (InlaEngine) se construyen densidades desde los valores
//! de log p(x|y,θ) evaluados en una cuadrícula de puntos.

use crate::integrator::gauss_kronrod_15;

/// Densidad marginal aproximada.
#[derive(Debug, Clone)]
pub enum Density {
    /// Aproximación gaussiana: media y desviación estándar.
    Gaussian { mean: f64, sd: f64 },

    /// Gaussiana + corrección log-spline (Skew-Corrected Gaussian).
    /// Default de INLA para marginales de efectos latentes.
    ///
    /// La corrección log-spline captura asimetría y curtosis
    /// que la aproximación Laplace pura no representa.
    ScGaussian {
        mean: f64,
        sd: f64,
        /// Nodos de evaluación (escala estandarizada)
        nodes: Vec<f64>,
        /// log p(x|y,θ) normalizado en los nodos
        log_density: Vec<f64>,
    },
}

impl Density {
    /// Crea una densidad gaussiana.
    pub fn gaussian(mean: f64, sd: f64) -> Self {
        Self::Gaussian { mean, sd }
    }

    /// Evalúa la densidad en el punto x.
    pub fn evaluate(&self, x: f64) -> f64 {
        match self {
            Self::Gaussian { mean, sd } => {
                let z = (x - mean) / sd;
                (-0.5 * z * z).exp() / (sd * (2.0 * std::f64::consts::PI).sqrt())
            }
            Self::ScGaussian {
                mean,
                sd,
                nodes,
                log_density,
            } => {
                // Interpola en la cuadrícula y multiplica por la Gaussiana base
                let z = (x - mean) / sd;
                let correction = interpolate_log_density(z, nodes, log_density);
                (-0.5 * z * z + correction).exp() / (sd * (2.0 * std::f64::consts::PI).sqrt())
            }
        }
    }

    /// Media de la densidad.
    pub fn mean(&self) -> f64 {
        match self {
            Self::Gaussian { mean, .. } => *mean,
            Self::ScGaussian { mean, .. } => *mean,
        }
    }

    /// Desviación estándar.
    pub fn sd(&self) -> f64 {
        match self {
            Self::Gaussian { sd, .. } => *sd,
            Self::ScGaussian { sd, .. } => *sd,
        }
    }

    /// Varianza.
    pub fn variance(&self) -> f64 {
        let s = self.sd();
        s * s
    }

    /// Cuantil q (0 < q < 1) por bisección numérica.
    pub fn quantile(&self, q: f64) -> f64 {
        debug_assert!(q > 0.0 && q < 1.0, "q debe estar en (0, 1)");

        let m = self.mean();
        let s = self.sd();
        let mut lo = m - 10.0 * s;
        let mut hi = m + 10.0 * s;

        // Bisección: 50 iteraciones → precisión ~1e-15
        for _ in 0..50 {
            let mid = (lo + hi) / 2.0;
            let cdf_mid = gauss_kronrod_15(|x| self.evaluate(x), lo - 5.0 * s, mid);
            if cdf_mid < q {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        (lo + hi) / 2.0
    }
}

/// Interpolación lineal del log de la corrección de densidad.
fn interpolate_log_density(z: f64, nodes: &[f64], log_density: &[f64]) -> f64 {
    if nodes.is_empty() {
        return 0.0;
    }

    // Extrapolación constante fuera del rango
    if z <= nodes[0] {
        return log_density[0];
    }
    if z >= *nodes.last().unwrap() {
        return *log_density.last().unwrap();
    }

    // Búsqueda binaria del intervalo
    let idx = nodes.partition_point(|&n| n < z).saturating_sub(1);
    let t = (z - nodes[idx]) / (nodes[idx + 1] - nodes[idx]);
    log_density[idx] * (1.0 - t) + log_density[idx + 1] * t
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn gaussian_integrates_to_one() {
        let d = Density::gaussian(2.0, 1.5);
        let m = d.mean();
        let s = d.sd();
        let total = gauss_kronrod_15(|x| d.evaluate(x), m - 6.0 * s, m + 6.0 * s);
        assert_abs_diff_eq!(total, 1.0, epsilon = 1e-4);
    }

    #[test]
    fn gaussian_mean_and_sd() {
        let d = Density::gaussian(3.0, 0.5);
        assert_abs_diff_eq!(d.mean(), 3.0, epsilon = 1e-12);
        assert_abs_diff_eq!(d.sd(), 0.5, epsilon = 1e-12);
        assert_abs_diff_eq!(d.variance(), 0.25, epsilon = 1e-12);
    }

    #[test]
    fn gaussian_mode_at_mean() {
        let d = Density::gaussian(1.0, 1.0);
        // f(media) > f(media + 1sd)
        assert!(d.evaluate(1.0) > d.evaluate(2.0));
        assert!(d.evaluate(1.0) > d.evaluate(0.0));
    }

    #[test]
    fn gaussian_quantile_median_is_mean() {
        let d = Density::gaussian(2.0, 1.0);
        // Para gaussiana, cuantil 0.5 = media
        let q50 = d.quantile(0.5);
        assert_abs_diff_eq!(q50, 2.0, epsilon = 1e-4);
    }
}
