//! Integración numérica Gauss-Kronrod de 15 puntos.
//!
//! Equivalente Rust de `integrator.c` (1054 líneas de C).
//!
//! Las constantes están hardcodeadas igual que en el C original —
//! son los nodos y pesos de la cuadratura GK15, valores fijos
//! calculados una vez y válidos para cualquier intervalo [a,b].
//!
//! Se usa para integrar densidades marginales sobre θ en InlaEngine.

/// Nodos de Gauss-Kronrod 15 puntos en [-1, 1].
/// Los nodos de Gauss de 7 puntos son un subconjunto (índices pares).
const GK15_NODES: [f64; 15] = [
    -0.991_455_371_120_813,
    -0.949_107_912_342_759,
    -0.864_864_423_359_769,
    -0.741_531_185_599_394,
    -0.586_087_235_467_691,
    -0.405_845_151_377_397,
    -0.207_784_955_007_898,
     0.000_000_000_000_000,
     0.207_784_955_007_898,
     0.405_845_151_377_397,
     0.586_087_235_467_691,
     0.741_531_185_599_394,
     0.864_864_423_359_769,
     0.949_107_912_342_759,
     0.991_455_371_120_813,
];

/// Pesos de Kronrod (15 puntos).
const GK15_WEIGHTS: [f64; 15] = [
    0.022_935_322_010_529,
    0.063_092_092_629_979,
    0.104_790_010_322_250,
    0.140_653_259_715_525,
    0.169_004_726_639_267,
    0.190_350_578_064_785,
    0.204_432_940_075_298,
    0.209_482_141_084_728,
    0.204_432_940_075_298,
    0.190_350_578_064_785,
    0.169_004_726_639_267,
    0.140_653_259_715_525,
    0.104_790_010_322_250,
    0.063_092_092_629_979,
    0.022_935_322_010_529,
];

/// Integra f sobre [a, b] usando Gauss-Kronrod de 15 puntos.
///
/// Error O(h^30) para funciones suaves — suficiente para densidades
/// posteriores de INLA que son aproximadamente gaussianas.
///
/// # Ejemplo
/// ```
/// use rust_inla::integrator::gauss_kronrod_15;
/// // ∫₀¹ x² dx = 1/3
/// let result = gauss_kronrod_15(|x| x * x, 0.0, 1.0);
/// assert!((result - 1.0/3.0).abs() < 1e-10);
/// ```
pub fn gauss_kronrod_15(f: impl Fn(f64) -> f64, a: f64, b: f64) -> f64 {
    let mid      = (a + b) / 2.0;
    let half_len = (b - a) / 2.0;

    GK15_NODES
        .iter()
        .zip(GK15_WEIGHTS.iter())
        .map(|(&t, &w)| w * f(mid + half_len * t))
        .sum::<f64>()
        * half_len
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn integrates_constant() {
        // ∫₀¹ 3 dx = 3
        let r = gauss_kronrod_15(|_| 3.0, 0.0, 1.0);
        assert_abs_diff_eq!(r, 3.0, epsilon = 1e-12);
    }

    #[test]
    fn integrates_polynomial() {
        // ∫₀¹ x² dx = 1/3
        let r = gauss_kronrod_15(|x| x * x, 0.0, 1.0);
        assert_abs_diff_eq!(r, 1.0 / 3.0, epsilon = 1e-12);
    }

    #[test]
    fn integrates_gaussian_density() {
        // ∫₋₅⁵ N(x|0,1) dx ≈ 1.0 (casi toda la masa)
        let r = gauss_kronrod_15(
            |x| (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt(),
            -5.0, 5.0,
        );
        assert_abs_diff_eq!(r, 1.0, epsilon = 1e-5);
    }

    #[test]
    fn integrates_on_negative_interval() {
        // ∫₋₁⁰ x dx = -0.5
        let r = gauss_kronrod_15(|x| x, -1.0, 0.0);
        assert_abs_diff_eq!(r, -0.5, epsilon = 1e-12);
    }
}
