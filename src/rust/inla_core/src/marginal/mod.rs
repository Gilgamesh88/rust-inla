//! Transformaciones de densidades marginales.
//!
//! Equivalente Rust de `marginal.R` (567 lÃ­neas).
//!
//! Implementa las funciones de R-INLA:
//! - `inla.zmarginal` â†’ estadÃ­sticos bÃ¡sicos (mean, sd, quantiles)
//! - `inla.emarginal` â†’ E[g(x)] para funciÃ³n g arbitraria
//! - `inla.tmarginal` â†’ transforma la variable aleatoria

/// Densidad marginal discreta â€” evaluada en una cuadrÃ­cula de puntos.
///
/// `x` y `y` tienen el mismo largo. La densidad estÃ¡ normalizada:
/// âˆ« p(x) dx = 1 (aproximado por integraciÃ³n numÃ©rica de la cuadrÃ­cula).
#[derive(Debug, Clone)]
pub struct Marginal {
    /// Nodos de evaluaciÃ³n.
    pub x: Vec<f64>,
    /// Densidad (no-normalizada) en cada nodo.
    pub y: Vec<f64>,
}

impl Marginal {
    /// Crea una marginal desde nodos y densidades.
    /// Normaliza automÃ¡ticamente para que integre a 1.
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

    /// Integra f(x, p(x)) sobre la cuadrÃ­cula por regla del trapecio.
    fn integrate(&self, f: impl Fn(f64, f64) -> f64) -> f64 {
        self.x
            .windows(2)
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

    /// Varianza posterior Var(x) = E[xÂ²] - E[x]Â².
    pub fn variance(&self) -> f64 {
        let m = self.mean();
        let e_x2 = self.integrate(|xi, yi| xi * xi * yi);
        (e_x2 - m * m).max(0.0)
    }

    /// DesviaciÃ³n estÃ¡ndar posterior.
    pub fn sd(&self) -> f64 {
        self.variance().sqrt()
    }

    /// E[g(x)] â€” esperanza de una funciÃ³n arbitraria de x.
    ///
    /// Equivalente a `inla.emarginal(fun, marginal)` en R.
    pub fn emarginal(&self, g: impl Fn(f64) -> f64) -> f64 {
        self.integrate(|xi, yi| g(xi) * yi)
    }

    /// Cuantil q (0 < q < 1) por bÃºsqueda en la CDF acumulada.
    pub fn quantile(&self, q: f64) -> f64 {
        debug_assert!(q > 0.0 && q < 1.0);

        let mut cdf = 0.0;
        for i in 1..self.x.len() {
            let dx = self.x[i] - self.x[i - 1];
            let piece = 0.5 * dx * (self.y[i - 1] + self.y[i]);
            if cdf + piece >= q {
                // InterpolaciÃ³n lineal dentro del intervalo
                let t = (q - cdf) / piece;
                return self.x[i - 1] + t * dx;
            }
            cdf += piece;
        }
        *self.x.last().unwrap()
    }

    /// EstadÃ­sticos bÃ¡sicos â€” equivalente a `inla.zmarginal` en R.
    pub fn zmarginal(&self) -> ZMarginal {
        ZMarginal {
            mean: self.mean(),
            sd: self.sd(),
            q0_025: self.quantile(0.025),
            q0_25: self.quantile(0.25),
            q0_5: self.quantile(0.5),
            q0_75: self.quantile(0.75),
            q0_975: self.quantile(0.975),
        }
    }
}

/// EstadÃ­sticos bÃ¡sicos de una marginal (salida de zmarginal).
#[derive(Debug, Clone)]
pub struct ZMarginal {
    pub mean: f64,
    pub sd: f64,
    pub q0_025: f64,
    pub q0_25: f64,
    pub q0_5: f64,
    pub q0_75: f64,
    pub q0_975: f64,
}

/// Construye una marginal gaussiana discreta en una cuadrÃ­cula uniforme.
pub fn build_build_gaussian_marginal(mean: f64, sd: f64, n: usize) -> Marginal {
    let lo = mean - 4.0 * sd;
    let hi = mean + 4.0 * sd;
    let x: Vec<f64> = (0..n)
        .map(|i| lo + (hi - lo) * i as f64 / (n - 1) as f64)
        .collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| {
            let z = (xi - mean) / sd;
            (-0.5 * z * z).exp()
        })
        .collect();
    Marginal::new(x, y)
}
