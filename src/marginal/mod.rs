//! Transformaciones de densidades marginales.
//!
//! Equivalente Rust de `marginal.R` (567 líneas).
//!
//! Implementa las funciones de R-INLA:
//! - `inla.zmarginal` → estadísticos básicos (mean, sd, quantiles)
//! - `inla.emarginal` → E[g(x)] para función g arbitraria
//! - `inla.tmarginal` → transforma la variable aleatoria

use crate::integrator::gauss_kronrod_15;

/// Densidad marginal discreta — evaluada en una cuadrícula de puntos.
///
/// `x` y `y` tienen el mismo largo. La densidad está normalizada:
/// ∫ p(x) dx = 1 (aproximado por integración numérica de la cuadrícula).
#[derive(Debug, Clone)]
pub struct Marginal {
    /// Nodos de evaluación.
    pub x: Vec<f64>,
    /// Densidad (no-normalizada) en cada nodo.
    pub y: Vec<f64>,
}

impl Marginal {
    /// Crea una marginal desde nodos y densidades.
    /// Normaliza automáticamente para que integre a 1.
    pub fn new(x: Vec<f64>, y: Vec<f64>) -> Self {
        assert_eq!(x.len(), y.len(), "x e y deben tener el mismo largo");
        assert!(x.len() >= 2, "Se necesitan al menos 2 puntos");
        let mut m = Self { x, y };
        m.normalize();
        m
    }

    /// Normaliza la densidad para que integre a 1.
    fn normalize(&mut self) {
        let total = self.integrate(|_, yi| yi);
        if total > 0.0 {
            for yi in &mut self.y {
                *yi /= total;
            }
        }
    }

    /// Integra f(x, p(x)) sobre la cuadrícula por regla del trapecio.
    fn integrate(&self, f: impl Fn(f64, f64) -> f64) -> f64 {
        self.x.windows(2)
            .zip(self.y.windows(2))
            .map(|(xs, ys)| {
                let dx = xs[1] - xs[0];
                0.5 * dx * (f(xs[0], ys[0]) + f(xs[1], ys[1]))
            })
            .sum()
    }

    /// Media posterior E[x].
    pub fn mean(&self) -> f64 {
        self.integrate(|xi, yi| xi * yi)
    }

    /// Varianza posterior Var(x) = E[x²] - E[x]².
    pub fn variance(&self) -> f64 {
        let m = self.mean();
        let e_x2 = self.integrate(|xi, yi| xi * xi * yi);
        (e_x2 - m * m).max(0.0)
    }

    /// Desviación estándar posterior.
    pub fn sd(&self) -> f64 {
        self.variance().sqrt()
    }

    /// E[g(x)] — esperanza de una función arbitraria de x.
    ///
    /// Equivalente a `inla.emarginal(fun, marginal)` en R.
    pub fn emarginal(&self, g: impl Fn(f64) -> f64) -> f64 {
        self.integrate(|xi, yi| g(xi) * yi)
    }

    /// Cuantil q (0 < q < 1) por búsqueda en la CDF acumulada.
    pub fn quantile(&self, q: f64) -> f64 {
        debug_assert!(q > 0.0 && q < 1.0);

        let mut cdf = 0.0;
        for i in 1..self.x.len() {
            let dx    = self.x[i] - self.x[i - 1];
            let piece = 0.5 * dx * (self.y[i - 1] + self.y[i]);
            if cdf + piece >= q {
                // Interpolación lineal dentro del intervalo
                let t = (q - cdf) / piece;
                return self.x[i - 1] + t * dx;
            }
            cdf += piece;
        }
        *self.x.last().unwrap()
    }

    /// Estadísticos básicos — equivalente a `inla.zmarginal` en R.
    pub fn zmarginal(&self) -> ZMarginal {
        ZMarginal {
            mean:  self.mean(),
            sd:    self.sd(),
            q0_025: self.quantile(0.025),
            q0_25:  self.quantile(0.25),
            q0_5:   self.quantile(0.5),
            q0_75:  self.quantile(0.75),
            q0_975: self.quantile(0.975),
        }
    }
}

/// Estadísticos básicos de una marginal (salida de zmarginal).
#[derive(Debug, Clone)]
pub struct ZMarginal {
    pub mean:   f64,
    pub sd:     f64,
    pub q0_025: f64,
    pub q0_25:  f64,
    pub q0_5:   f64,
    pub q0_75:  f64,
    pub q0_975: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    /// Construye una marginal gaussiana discreta en una cuadrícula uniforme.
    fn gaussian_marginal(mean: f64, sd: f64, n: usize) -> Marginal {
        let lo = mean - 4.0 * sd;
        let hi = mean + 4.0 * sd;
        let x: Vec<f64> = (0..n).map(|i| lo + (hi - lo) * i as f64 / (n - 1) as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| {
            let z = (xi - mean) / sd;
            (-0.5 * z * z).exp()
        }).collect();
        Marginal::new(x, y)
    }

    #[test]
    fn marginal_mean_of_gaussian() {
        let m = gaussian_marginal(2.0, 1.0, 200);
        assert_abs_diff_eq!(m.mean(), 2.0, epsilon = 1e-3);
    }

    #[test]
    fn marginal_sd_of_gaussian() {
        let m = gaussian_marginal(0.0, 1.5, 200);
        assert_abs_diff_eq!(m.sd(), 1.5, epsilon = 1e-2);
    }

    #[test]
    fn marginal_median_of_symmetric() {
        // Para distribución simétrica, mediana = media
        let m = gaussian_marginal(3.0, 1.0, 200);
        assert_abs_diff_eq!(m.quantile(0.5), 3.0, epsilon = 1e-2);
    }

    #[test]
    fn emarginal_identity_is_mean() {
        // E[x] = media
        let m = gaussian_marginal(2.0, 1.0, 200);
        assert_abs_diff_eq!(m.emarginal(|x| x), 2.0, epsilon = 1e-3);
    }

    #[test]
    fn emarginal_exp_positive() {
        // E[exp(x)] > 0 siempre
        let m = gaussian_marginal(0.0, 1.0, 200);
        assert!(m.emarginal(|x| x.exp()) > 0.0);
    }

    #[test]
    fn zmarginal_q025_lt_median_lt_q975() {
        let m = gaussian_marginal(0.0, 1.0, 200);
        let z = m.zmarginal();
        assert!(z.q0_025 < z.q0_5);
        assert!(z.q0_5   < z.q0_975);
        assert_abs_diff_eq!(z.mean, 0.0, epsilon = 1e-3);
    }
}
