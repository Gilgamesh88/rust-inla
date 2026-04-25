//! Modelos de campo aleatorio gaussiano de Markov (GMRF).
//!
//! Cada modelo implementa el trait `QFunc`, que define:
//! - La estructura dispersa de Q (a través de un `Graph`)
//! - El valor de cada entrada Q(i,j) en función de los hiperparámetros θ
//!
//! ## Modelos implementados
//!
//! | Modelo     | Graph       | θ                          | Notas                    |
//! |------------|-------------|----------------------------|--------------------------|
//! | IidModel   | empty (diag)| [log τ]                    | Q = τ·I                  |
//! | Rw1Model   | linear      | [log τ]                    | Q = τ·DᵀD (impropio)     |
//! | Rw2Model   | rw2         | [log τ]                    | Q = τ·D₂ᵀD₂ (impropio)   |
//! | Ar1Model   | linear      | [log τ, arctanh(ρ)]        | Q propio si |ρ| < 1       |
//! | Ar2Model   | ar2         | [log τ, pacf1, pacf2]      | Q propio, banda ±2       |
//!
//! ## RW1 es un prior impropio
//!
//! Q_rw1 = τ·DᵀD donde D es la matriz de diferencias (n-1)×n.
//! El vector [1,1,...,1] está en el kernel de DᵀD → eigenvalor cero → Q singular.
//! Esto es correcto matemáticamente: un paseo aleatorio no tiene distribución
//! estacionaria. En INLA la verosimilitud p(y|x) hace que el posterior sea propio.
//! En tests aislados del solver se añade ε·I para obtener Q definida positiva.

use crate::graph::Graph;

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

#[inline]
fn inla_correlation_from_theta(theta: f64) -> f64 {
    (theta * 0.5).tanh()
}

#[inline]
fn pc_cor0_log_prior(theta: f64, u: f64, alpha: f64) -> f64 {
    let lambda = -alpha.ln() / (-(1.0 - u * u).ln()).sqrt();
    let rho = inla_correlation_from_theta(theta);
    let mu = (-(1.0 - rho * rho).ln()).sqrt();

    if mu < 1e-12 {
        return lambda.ln() - 4.0_f64.ln();
    }

    lambda.ln() - lambda * mu + rho.abs().ln() - mu.ln() - 4.0_f64.ln()
}

/// Contrato de cada modelo GMRF latente.
pub trait QFunc: Send + Sync {
    /// Grafo de dispersidad de Q (fijo, independiente de θ).
    fn graph(&self) -> &Graph;

    /// Valor de la entrada Q(i, j) para los hiperparámetros `theta`.
    ///
    /// Solo se llama para pares (i,j) que son vecinos en `graph()`,
    /// o para la diagonal (i == j). Debe ser simétrica: eval(i,j) == eval(j,i).
    fn eval(&self, i: usize, j: usize, theta: &[f64]) -> f64;

    /// Número de hiperparámetros θ de este modelo.
    fn n_hyperparams(&self) -> usize;
    /// Derivada de Q(i,j,theta) respecto a theta[k].
    /// Por defecto None — usa diferencias finitas en el optimizador.
    /// Implementar para gradientes exactos y BFGS eficiente.
    fn deval(&self, _i: usize, _j: usize, _theta: &[f64], _k: usize) -> Option<f64> {
        None
    }

    /// Si true, Q es definida positiva (prior propio).
    /// Si false, Q es solo semidefinida positiva (prior impropio, e.g. Rw1, Rw2).
    ///
    /// El optimizador usa este flag para omitir el término 0.5·log|Q| cuando
    /// Q es singular: para un prior impropio log|Q| = -∞ y no contribuye al
    /// modo de p(y|θ). Solo la verosimilitud y log|Q+W| determinan θ*.
    fn is_proper(&self) -> bool {
        true
    }

    /// Theta-dependent contribution of log|Q| to the Laplace objective.
    ///
    /// Proper models use an exact sparse log-determinant elsewhere. Intrinsic
    /// models still need their generalized-determinant scaling term on theta
    /// so the outer optimizer sees the correct precision penalty.
    fn log_det_term(&self, _theta: &[f64]) -> f64 {
        0.0
    }

    /// Log-prior density on the model's internal theta scale.
    fn log_prior(&self, theta: &[f64]) -> f64 {
        theta
            .iter()
            .map(|&th| loggamma_on_log_scale(th, 1.0, 5e-5))
            .sum()
    }
}

// ── IidModel ──────────────────────────────────────────────────────────────────

/// Modelo de efectos independientes idénticamente distribuidos.
///
/// Q = τ·I  (diagonal, sin correlación espacial/temporal)
///
/// Hiperparámetros:
/// - θ[0] = log τ   (log-precisión marginal)
pub struct IidModel {
    graph: Graph,
}

impl IidModel {
    pub fn new(n: usize) -> Self {
        Self {
            graph: Graph::iid(n),
        }
    }
}

impl QFunc for IidModel {
    fn graph(&self) -> &Graph {
        &self.graph
    }

    fn eval(&self, i: usize, j: usize, theta: &[f64]) -> f64 {
        if i != j {
            return 0.0;
        }
        theta[0].exp() // τ = exp(log τ)
    }

    fn n_hyperparams(&self) -> usize {
        1
    }

    fn deval(&self, i: usize, j: usize, theta: &[f64], k: usize) -> Option<f64> {
        if k != 0 {
            return Some(0.0);
        }
        // d/d(log_tau) [tau * structure] = tau * structure = Q(i,j)
        Some(self.eval(i, j, theta))
    }

    fn log_prior(&self, theta: &[f64]) -> f64 {
        loggamma_on_log_scale(theta[0], 1.0, 5e-5)
    }
}

// ── Rw1Model ──────────────────────────────────────────────────────────────────

/// Paseo aleatorio de primer orden (Random Walk 1).
///
/// Q = τ · DᵀD  donde D es la matriz de diferencias (n-1)×n:
///
/// ```text
///   DᵀD =  [  1  -1   0   0  ...]
///           [ -1   2  -1   0  ...]
///           [  0  -1   2  -1  ...]
///           [         ...       ]
///           [  0   0  -1   1  ]
/// ```
///
/// Equivale a: Q(i,i) = τ × grado(i), Q(i,i±1) = -τ
///
/// Hiperparámetros:
/// - θ[0] = log τ   (log-precisión de los incrementos)
///
/// **Nota:** Q es solo semidefinida positiva (PSD), no PD.
/// El vector [1,1,...,1] está en el kernel. Ver docstring del módulo.
pub struct Rw1Model {
    graph: Graph,
}

impl Rw1Model {
    pub fn new(n: usize) -> Self {
        Self {
            graph: Graph::linear(n),
        }
    }
}

impl QFunc for Rw1Model {
    fn graph(&self) -> &Graph {
        &self.graph
    }

    fn eval(&self, i: usize, j: usize, theta: &[f64]) -> f64 {
        let tau = theta[0].exp();
        let n = self.graph.n();

        if i == j {
            // Grado del nodo: nodos interiores tienen 2 vecinos, extremos tienen 1
            let degree = if i == 0 || i == n - 1 { 1.0 } else { 2.0 };
            tau * degree
        } else {
            // Off-diagonal: siempre -τ para nodos adyacentes en la cadena
            -tau
        }
    }

    fn n_hyperparams(&self) -> usize {
        1
    }
    fn deval(&self, i: usize, j: usize, theta: &[f64], k: usize) -> Option<f64> {
        if k != 0 {
            return Some(0.0);
        }
        // d/d(log_tau) [tau * structure] = tau * structure = Q(i,j)
        Some(self.eval(i, j, theta))
    }
    /// Rw1 es un prior impropio: Q = τ·DᵀD es solo semidefinida positiva.
    /// El vector constante [1,1,...,1] está en el kernel → log|Q| = -∞.
    fn is_proper(&self) -> bool {
        false
    }

    fn log_det_term(&self, theta: &[f64]) -> f64 {
        (self.graph.n().saturating_sub(1) as f64) * theta[0]
    }

    fn log_prior(&self, theta: &[f64]) -> f64 {
        loggamma_on_log_scale(theta[0], 1.0, 5e-5)
    }
}

// ── Rw2Model ──────────────────────────────────────────────────────────────────

/// Paseo aleatorio de segundo orden (Random Walk 2).
///
/// Q = τ · D₂ᵀD₂ donde D₂ es la matriz de segundas diferencias (n-2)×n:
///
/// ```text
///   (D₂x)_r = x_r - 2 x_{r+1} + x_{r+2}
/// ```
///
/// La estructura resultante es pentadiagonal:
///
/// ```text
///   diag    = [1, 5, 6, ..., 6, 5, 1]
///   off ±1  = [-2, -4, ..., -4, -2]
///   off ±2  = [1, 1, ..., 1]
/// ```
///
/// Hiperparámetros:
/// - θ[0] = log τ   (log-precisión de las segundas diferencias)
///
/// **Nota:** Q es impropia. El espacio nulo está generado por funciones
/// constantes y lineales sobre el índice.
pub struct Rw2Model {
    graph: Graph,
    diag: Vec<f64>,
    off1: Vec<f64>,
    off2: Vec<f64>,
}

impl Rw2Model {
    pub fn new(n: usize) -> Self {
        let values: Vec<f64> = (0..n).map(|idx| idx as f64).collect();
        Self::new_with_values(&values)
            .expect("equally spaced rw2 values should always produce a valid structure")
    }

    pub fn new_with_values(values: &[f64]) -> Result<Self, String> {
        if values.len() < 3 {
            return Err("rw2 requires at least 3 levels".to_string());
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err("rw2 structure values must be finite".to_string());
        }
        for idx in 0..values.len() - 1 {
            if values[idx + 1] <= values[idx] {
                return Err("rw2 structure values must be strictly increasing".to_string());
            }
        }

        let n = values.len();
        let mut diag = vec![0.0; n];
        let mut off1 = vec![0.0; n - 1];
        let mut off2 = vec![0.0; n - 2];

        for row in 0..(n - 2) {
            let h_left = values[row + 1] - values[row];
            let h_right = values[row + 2] - values[row + 1];
            let weight = 2.0 / (h_left + h_right);
            let coeffs = [1.0 / h_left, -(1.0 / h_left + 1.0 / h_right), 1.0 / h_right];

            diag[row] += weight * coeffs[0] * coeffs[0];
            diag[row + 1] += weight * coeffs[1] * coeffs[1];
            diag[row + 2] += weight * coeffs[2] * coeffs[2];
            off1[row] += weight * coeffs[0] * coeffs[1];
            off1[row + 1] += weight * coeffs[1] * coeffs[2];
            off2[row] += weight * coeffs[0] * coeffs[2];
        }

        Ok(Self {
            graph: Graph::rw2(n),
            diag,
            off1,
            off2,
        })
    }

    #[inline]
    fn structure_value(&self, i: usize, j: usize) -> f64 {
        let (lo, hi) = if i <= j { (i, j) } else { (j, i) };
        match hi - lo {
            0 => self.diag[lo],
            1 => self.off1[lo],
            2 => self.off2[lo],
            _ => 0.0,
        }
    }
}

impl QFunc for Rw2Model {
    fn graph(&self) -> &Graph {
        &self.graph
    }

    fn eval(&self, i: usize, j: usize, theta: &[f64]) -> f64 {
        theta[0].exp() * self.structure_value(i, j)
    }

    fn n_hyperparams(&self) -> usize {
        1
    }

    fn deval(&self, i: usize, j: usize, theta: &[f64], k: usize) -> Option<f64> {
        if k != 0 {
            return Some(0.0);
        }
        Some(self.eval(i, j, theta))
    }

    fn is_proper(&self) -> bool {
        false
    }

    fn log_det_term(&self, theta: &[f64]) -> f64 {
        (self.graph.n().saturating_sub(2) as f64) * theta[0]
    }

    fn log_prior(&self, theta: &[f64]) -> f64 {
        loggamma_on_log_scale(theta[0], 1.0, 5e-5)
    }
}

// ── Ar1Model ──────────────────────────────────────────────────────────────────

/// Proceso autoregresivo de primer orden (AR1).
///
/// x_t = ρ · x_{t-1} + ε_t,  ε_t ~ N(0, 1/τ_ε)
///
/// La matriz de precisión del proceso estacionario es:
///
/// ```text
///   Q(0,0)      = κ / (1 - ρ²)                  (nodo inicial)
///   Q(i,i)      = κ·(1 + ρ²) / (1 - ρ²)         (nodos interiores)
///   Q(n-1,n-1)  = κ / (1 - ρ²)                  (nodo final)
///   Q(i,i+1)    = Q(i+1,i) = -κ·ρ / (1 - ρ²)    (off-diagonal)
/// ```
///
/// donde κ es la precisión *marginal* estacionaria (κ = 1 / Var(x_t)).
/// Esto coincide con la parametrización interna de INLA para `model = "ar1"`.
///
/// Hiperparámetros:
/// - θ[0] = log τ        (log-precisión marginal)
/// - θ[1] = log((1+ρ)/(1-ρ)) = 2 * atanh(ρ)
///   (ρ = 2*exp(θ[1])/(1+exp(θ[1])) - 1 = tanh(θ[1] / 2), same internal scale as INLA)
///
/// **Nota:** Q es PD si y solo si |ρ| < 1, que se garantiza con la
/// parametrización logit-like above.
pub struct Ar1Model {
    graph: Graph,
}

impl Ar1Model {
    pub fn new(n: usize) -> Self {
        Self {
            graph: Graph::ar1(n),
        }
    }
}

impl QFunc for Ar1Model {
    fn graph(&self) -> &Graph {
        &self.graph
    }

    fn eval(&self, i: usize, j: usize, theta: &[f64]) -> f64 {
        let kappa = theta[0].exp(); // precisión marginal estacionaria
        let rho = inla_correlation_from_theta(theta[1]); // autocorrelación ∈ (-1, 1), same as INLA
        let n = self.graph.n();
        let scale = kappa / (1.0 - rho * rho);

        if i == j {
            if i == 0 || i == n - 1 {
                scale
            } else {
                scale * (1.0 + rho * rho)
            }
        } else {
            -scale * rho
        }
    }

    fn n_hyperparams(&self) -> usize {
        2
    }

    fn deval(&self, i: usize, j: usize, theta: &[f64], k: usize) -> Option<f64> {
        let kappa = theta[0].exp();
        let rho = inla_correlation_from_theta(theta[1]);
        let n = self.graph.n();
        let scale = kappa / (1.0 - rho * rho);
        match k {
            0 => Some(self.eval(i, j, theta)), // d/d(log_kappa) = Q(i,j)
            1 => {
                // d/d(theta_inla) where rho = tanh(theta_inla / 2)
                let drho = 0.5 * (1.0 - rho * rho);
                let dscale = scale * rho;
                if i == j {
                    if i == 0 || i == n - 1 {
                        Some(dscale)
                    } else {
                        Some(dscale * (1.0 + rho * rho) + scale * 2.0 * rho * drho)
                    }
                } else {
                    Some(-(dscale * rho + scale * drho))
                }
            }
            _ => Some(0.0),
        }
    }

    fn log_prior(&self, theta: &[f64]) -> f64 {
        loggamma_on_log_scale(theta[0], 1.0, 5e-5) + gaussian_prior_kernel(theta[1], 0.0, 0.15)
    }
}

// ── Ar2Model ─────────────────────────────────────────────────────────────────────────────

/// Proceso autoregresivo de segundo orden (AR2) parametrizado por PACF.
///
/// INLA expone el modelo AR(p) en términos de:
///
/// - θ[0] = log κ, donde κ es la precisión marginal estacionaria
/// - θ[1] = logit-like(pacf1)
/// - θ[2] = logit-like(pacf2)
///
/// Con `pacf_j = tanh(theta[j] / 2)`, las PACF quedan en (-1, 1) y la
/// recursión de Durbin-Levinson produce coeficientes AR2 estacionarios.
///
/// Para p = 2:
///
/// ```text
/// phi2 = pacf2
/// phi1 = pacf1 * (1 - pacf2)
/// ```
///
/// La matriz de precisión exacta para una realización finita estacionaria se
/// compone de:
///
/// - el bloque de precisión de los dos primeros estados estacionarios
/// - más las innovaciones AR2 para t = 3, ..., n
///
/// El resultado es una matriz pentadiagonal propia con el mismo patrón de
/// banda que `rw2`.
pub struct Ar2Model {
    graph: Graph,
}

#[derive(Clone, Copy)]
struct Ar2State {
    kappa: f64,
    phi1: f64,
    phi2: f64,
    innovation_scale: f64,
    initial_diag: f64,
    initial_offdiag: f64,
}

impl Ar2Model {
    pub fn new(n: usize) -> Self {
        Self {
            graph: Graph::ar2(n),
        }
    }

    #[inline]
    fn state(theta: &[f64]) -> Ar2State {
        let kappa = theta[0].exp();
        let pacf1 = inla_correlation_from_theta(theta[1]);
        let pacf2 = inla_correlation_from_theta(theta[2]);

        // Same p=2 Durbin-Levinson recursion used by INLA's ar.R helpers.
        let phi2 = pacf2;
        let phi1 = pacf1 * (1.0 - pacf2);

        let rho1 = phi1 / (1.0 - phi2);
        let rho2 = phi1 * rho1 + phi2;
        let innovation_factor = (1.0 - phi1 * rho1 - phi2 * rho2).max(1e-12);
        let innovation_scale = 1.0 / innovation_factor;
        let initial_scale = 1.0 / (1.0 - rho1 * rho1).max(1e-12);

        Ar2State {
            kappa,
            phi1,
            phi2,
            innovation_scale,
            initial_diag: initial_scale,
            initial_offdiag: -rho1 * initial_scale,
        }
    }

    #[inline]
    fn structure_value(&self, i: usize, j: usize, theta: &[f64]) -> f64 {
        let n = self.graph.n();
        if n == 1 {
            return if i == 0 && j == 0 { 1.0 } else { 0.0 };
        }

        let state = Self::state(theta);
        let (lo, hi) = if i <= j { (i, j) } else { (j, i) };
        match hi - lo {
            0 => {
                let innovation = (if lo >= 2 { 1.0 } else { 0.0 })
                    + (if lo >= 1 && lo + 1 < n {
                        state.phi1 * state.phi1
                    } else {
                        0.0
                    })
                    + (if lo + 2 < n {
                        state.phi2 * state.phi2
                    } else {
                        0.0
                    });
                let initial = if lo < 2 { state.initial_diag } else { 0.0 };
                initial + state.innovation_scale * innovation
            }
            1 => {
                let innovation = (if lo >= 1 { -state.phi1 } else { 0.0 })
                    + (if lo + 2 < n {
                        state.phi1 * state.phi2
                    } else {
                        0.0
                    });
                let initial = if lo == 0 { state.initial_offdiag } else { 0.0 };
                initial + state.innovation_scale * innovation
            }
            2 => {
                if lo + 2 < n {
                    -state.innovation_scale * state.phi2
                } else {
                    0.0
                }
            }
            _ => 0.0,
        }
    }
}

impl QFunc for Ar2Model {
    fn graph(&self) -> &Graph {
        &self.graph
    }

    fn eval(&self, i: usize, j: usize, theta: &[f64]) -> f64 {
        let state = Self::state(theta);
        state.kappa * self.structure_value(i, j, theta)
    }

    fn n_hyperparams(&self) -> usize {
        3
    }

    fn deval(&self, i: usize, j: usize, theta: &[f64], k: usize) -> Option<f64> {
        match k {
            0 => Some(self.eval(i, j, theta)),
            _ => None,
        }
    }

    fn log_prior(&self, theta: &[f64]) -> f64 {
        loggamma_on_log_scale(theta[0], 1.0, 5e-5)
            + pc_cor0_log_prior(theta[1], 0.5, 0.5)
            + pc_cor0_log_prior(theta[2], 0.5, 0.4)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

struct CompoundBlock {
    start: usize,
    len: usize,
    theta_start: usize,
    qfunc: Box<dyn QFunc>,
}

/// Modelo compuesto block-diagonal para múltiples términos `f(...)`.
pub struct CompoundQFunc {
    graph: Graph,
    blocks: Vec<CompoundBlock>,
}

impl CompoundQFunc {
    pub fn new(blocks: Vec<(usize, Box<dyn QFunc>)>) -> Self {
        let graph_parts: Vec<(&Graph, usize)> = blocks
            .iter()
            .map(|(start, qfunc)| (qfunc.graph(), *start))
            .collect();
        let graph = Graph::disjoint_union(&graph_parts);

        let mut theta_start = 0usize;
        let blocks = blocks
            .into_iter()
            .map(|(start, qfunc)| {
                let len = qfunc.graph().n();
                let n_theta = qfunc.n_hyperparams();
                let block = CompoundBlock {
                    start,
                    len,
                    theta_start,
                    qfunc,
                };
                theta_start += n_theta;
                block
            })
            .collect();

        Self { graph, blocks }
    }

    fn block_for_index(&self, idx: usize) -> Option<&CompoundBlock> {
        self.blocks
            .iter()
            .find(|block| idx >= block.start && idx < block.start + block.len)
    }
}

impl QFunc for CompoundQFunc {
    fn graph(&self) -> &Graph {
        &self.graph
    }

    fn eval(&self, i: usize, j: usize, theta: &[f64]) -> f64 {
        let Some(block) = self.block_for_index(i) else {
            return 0.0;
        };
        if j < block.start || j >= block.start + block.len {
            return 0.0;
        }
        let local_i = i - block.start;
        let local_j = j - block.start;
        let theta_end = block.theta_start + block.qfunc.n_hyperparams();
        block
            .qfunc
            .eval(local_i, local_j, &theta[block.theta_start..theta_end])
    }

    fn n_hyperparams(&self) -> usize {
        self.blocks
            .iter()
            .map(|block| block.qfunc.n_hyperparams())
            .sum()
    }

    fn deval(&self, i: usize, j: usize, theta: &[f64], k: usize) -> Option<f64> {
        let block = self.block_for_index(i)?;
        if j < block.start || j >= block.start + block.len {
            return Some(0.0);
        }
        let theta_end = block.theta_start + block.qfunc.n_hyperparams();
        if k < block.theta_start || k >= theta_end {
            return Some(0.0);
        }
        block.qfunc.deval(
            i - block.start,
            j - block.start,
            &theta[block.theta_start..theta_end],
            k - block.theta_start,
        )
    }

    fn is_proper(&self) -> bool {
        self.blocks.iter().all(|block| block.qfunc.is_proper())
    }

    fn log_det_term(&self, theta: &[f64]) -> f64 {
        self.blocks
            .iter()
            .map(|block| {
                let theta_end = block.theta_start + block.qfunc.n_hyperparams();
                block
                    .qfunc
                    .log_det_term(&theta[block.theta_start..theta_end])
            })
            .sum()
    }

    fn log_prior(&self, theta: &[f64]) -> f64 {
        self.blocks
            .iter()
            .map(|block| {
                let theta_end = block.theta_start + block.qfunc.n_hyperparams();
                block.qfunc.log_prior(&theta[block.theta_start..theta_end])
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn inla_ar1_theta_from_rho(rho: f64) -> f64 {
        ((1.0 + rho) / (1.0 - rho)).ln()
    }

    fn dense_cholesky_solve(mut a: Vec<f64>, mut b: Vec<f64>, n: usize) -> Vec<f64> {
        for i in 0..n {
            for j in 0..=i {
                let mut sum = a[i * n + j];
                for k in 0..j {
                    sum -= a[i * n + k] * a[j * n + k];
                }
                if i == j {
                    assert!(sum > 1e-12, "dense reference matrix must remain SPD");
                    a[i * n + j] = sum.sqrt();
                } else {
                    a[i * n + j] = sum / a[j * n + j];
                }
            }
        }

        for i in 0..n {
            let mut sum = b[i];
            for k in 0..i {
                sum -= a[i * n + k] * b[k];
            }
            b[i] = sum / a[i * n + i];
        }
        for i in (0..n).rev() {
            let mut sum = b[i];
            for k in (i + 1)..n {
                sum -= a[k * n + i] * b[k];
            }
            b[i] = sum / a[i * n + i];
        }

        b
    }

    fn dense_spd_inverse(a: &[f64], n: usize) -> Vec<f64> {
        let mut inv = vec![0.0_f64; n * n];
        for col in 0..n {
            let mut rhs = vec![0.0_f64; n];
            rhs[col] = 1.0;
            let sol = dense_cholesky_solve(a.to_vec(), rhs, n);
            for row in 0..n {
                inv[row * n + col] = sol[row];
            }
        }
        inv
    }

    fn ar2_phi_from_pacf(pacf1: f64, pacf2: f64) -> (f64, f64) {
        (pacf1 * (1.0 - pacf2), pacf2)
    }

    fn ar2_acf_from_phi(phi1: f64, phi2: f64, lag_max: usize) -> Vec<f64> {
        let rho1 = phi1 / (1.0 - phi2);
        let rho2 = phi1 * rho1 + phi2;
        let mut acf = vec![0.0_f64; lag_max + 1];
        acf[0] = 1.0;
        if lag_max >= 1 {
            acf[1] = rho1;
        }
        if lag_max >= 2 {
            acf[2] = rho2;
        }
        for lag in 3..=lag_max {
            acf[lag] = phi1 * acf[lag - 1] + phi2 * acf[lag - 2];
        }
        acf
    }

    // ── IidModel ──────────────────────────────────────────────────────────────

    #[test]
    fn iid_graph_is_diagonal() {
        let m = IidModel::new(10);
        assert_eq!(m.graph().n(), 10);
        assert_eq!(m.graph().nnz(), 10); // solo diagonal
        assert_eq!(m.n_hyperparams(), 1);
    }

    #[test]
    fn iid_diagonal_equals_tau() {
        let m = IidModel::new(5);
        // θ[0] = log(2.0)  →  τ = 2.0
        let theta = [2.0_f64.ln()];
        for i in 0..5 {
            assert_abs_diff_eq!(m.eval(i, i, &theta), 2.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn iid_tau_increases_with_theta() {
        let m = IidModel::new(3);
        let q_low = m.eval(0, 0, &[0.0]); // τ = exp(0) = 1
        let q_high = m.eval(0, 0, &[2.0]); // τ = exp(2) ≈ 7.39
        assert!(q_high > q_low);
    }

    // ── Rw1Model ──────────────────────────────────────────────────────────────

    #[test]
    fn rw1_graph_is_tridiagonal() {
        let m = Rw1Model::new(5);
        assert_eq!(m.graph().n(), 5);
        // nnz = 5 (diag) + 2*4 (off-diag) = 13
        assert_eq!(m.graph().nnz(), 13);
        assert_eq!(m.n_hyperparams(), 1);
    }

    #[test]
    fn rw1_boundary_diagonal_is_tau() {
        let m = Rw1Model::new(5);
        let theta = [0.0]; // τ = 1.0
                           // Nodos extremos tienen grado 1 → Q(0,0) = Q(4,4) = τ
        assert_abs_diff_eq!(m.eval(0, 0, &theta), 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(m.eval(4, 4, &theta), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn rw1_interior_diagonal_is_two_tau() {
        let m = Rw1Model::new(5);
        let theta = [0.0]; // τ = 1.0
                           // Nodos interiores tienen grado 2 → Q(i,i) = 2τ
        for i in 1..4 {
            assert_abs_diff_eq!(m.eval(i, i, &theta), 2.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn rw1_offdiagonal_is_minus_tau() {
        let m = Rw1Model::new(5);
        let theta = [1.0]; // τ = e ≈ 2.718
        let tau = 1.0_f64.exp();
        // Q(i, i+1) = -τ para todos los pares adyacentes
        for i in 0..4 {
            assert_abs_diff_eq!(m.eval(i, i + 1, &theta), -tau, epsilon = 1e-12);
        }
    }

    #[test]
    fn rw1_is_symmetric() {
        let m = Rw1Model::new(5);
        let theta = [0.5];
        for i in 0..4 {
            assert_abs_diff_eq!(
                m.eval(i, i + 1, &theta),
                m.eval(i + 1, i, &theta),
                epsilon = 1e-12
            );
        }
    }

    #[test]
    fn rw1_log_det_term_matches_intrinsic_rank_scaling() {
        let m = Rw1Model::new(6);
        let theta = [1.7_f64];
        assert_abs_diff_eq!(m.log_det_term(&theta), 5.0 * theta[0], epsilon = 1e-12);
    }

    // ── Rw2Model ──────────────────────────────────────────────────────────────

    #[test]
    fn rw2_graph_is_pentadiagonal() {
        let m = Rw2Model::new(5);
        assert_eq!(m.graph().n(), 5);
        assert_eq!(m.graph().nnz(), 5 + 2 * (4 + 3));
        assert_eq!(m.n_hyperparams(), 1);
    }

    #[test]
    fn rw2_diagonal_matches_second_difference_structure() {
        let m = Rw2Model::new(6);
        let theta = [0.0];
        let expected = [1.0, 5.0, 6.0, 6.0, 5.0, 1.0];
        for (idx, value) in expected.iter().enumerate() {
            assert_abs_diff_eq!(m.eval(idx, idx, &theta), *value, epsilon = 1e-12);
        }
    }

    #[test]
    fn rw2_first_offdiagonal_matches_second_difference_structure() {
        let m = Rw2Model::new(6);
        let theta = [0.0];
        let expected = [-2.0, -4.0, -4.0, -4.0, -2.0];
        for (idx, value) in expected.iter().enumerate() {
            assert_abs_diff_eq!(m.eval(idx, idx + 1, &theta), *value, epsilon = 1e-12);
        }
    }

    #[test]
    fn rw2_second_offdiagonal_is_tau() {
        let m = Rw2Model::new(6);
        let theta = [1.0];
        let tau = 1.0_f64.exp();
        for i in 0..4 {
            assert_abs_diff_eq!(m.eval(i, i + 2, &theta), tau, epsilon = 1e-12);
        }
    }

    #[test]
    fn rw2_irregular_spacing_matches_inla_reference_precision() {
        let values = [1.0, 2.0, 4.0, 7.0, 11.0];
        let m = Rw2Model::new_with_values(&values).unwrap();
        let theta = [0.0];
        let expected = [
            [0.6666666666666666, -1.0, 0.3333333333333333, 0.0, 0.0],
            [-1.0, 1.6, -0.6666666666666666, 0.06666666666666667, 0.0],
            [
                0.3333333333333333,
                -0.6666666666666666,
                0.47619047619047616,
                -0.16666666666666666,
                0.023809523809523808,
            ],
            [
                0.0,
                0.06666666666666667,
                -0.16666666666666666,
                0.14166666666666666,
                -0.041666666666666664,
            ],
            [
                0.0,
                0.0,
                0.023809523809523808,
                -0.041666666666666664,
                0.017857142857142856,
            ],
        ];

        for (i, row) in expected.iter().enumerate() {
            for (j, value) in row.iter().enumerate() {
                assert_abs_diff_eq!(m.eval(i, j, &theta), *value, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn rw2_is_symmetric() {
        let m = Rw2Model::new(6);
        let theta = [0.5];
        for i in 0..5 {
            for j in 0..6 {
                assert_abs_diff_eq!(m.eval(i, j, &theta), m.eval(j, i, &theta), epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn rw2_derivative_matches_finite_difference() {
        let m = Rw2Model::new(6);
        let theta = [0.7_f64];
        let eps = 1e-6_f64;
        let theta_plus = [theta[0] + eps];
        let theta_minus = [theta[0] - eps];

        for &(i, j) in &[(0_usize, 0_usize), (1, 2), (0, 2), (2, 4)] {
            let analytic = m.deval(i, j, &theta, 0).unwrap();
            let numeric = (m.eval(i, j, &theta_plus) - m.eval(i, j, &theta_minus)) / (2.0 * eps);
            assert_abs_diff_eq!(analytic, numeric, epsilon = 1e-6);
        }
    }

    #[test]
    fn rw2_log_det_term_matches_intrinsic_rank_scaling() {
        let m = Rw2Model::new(6);
        let theta = [1.7_f64];
        assert_abs_diff_eq!(m.log_det_term(&theta), 4.0 * theta[0], epsilon = 1e-12);
    }

    #[test]
    fn rw2_requires_strictly_increasing_values() {
        assert!(Rw2Model::new_with_values(&[1.0, 1.0, 2.0]).is_err());
        assert!(Rw2Model::new_with_values(&[1.0, f64::NAN, 2.0]).is_err());
    }

    // ── Ar1Model ──────────────────────────────────────────────────────────────

    #[test]
    fn ar1_graph_is_tridiagonal() {
        let m = Ar1Model::new(10);
        assert_eq!(m.graph().n(), 10);
        assert_eq!(m.graph().nnz(), 10 + 2 * 9); // 10 + 18 = 28
        assert_eq!(m.n_hyperparams(), 2);
    }

    #[test]
    fn ar1_with_rho_zero_is_iid() {
        // ρ = tanh(0) = 0  →  AR1 degenera en iid
        // Q(i,i) = τ·(1+0²) = τ para todos, Q(i,j) = 0 para i≠j
        let m_ar1 = Ar1Model::new(5);
        let m_iid = IidModel::new(5);
        let theta_ar1 = [0.0, 0.0]; // τ=1, ρ=0
        let theta_iid = [0.0]; // τ=1

        for i in 0..5 {
            assert_abs_diff_eq!(
                m_ar1.eval(i, i, &theta_ar1),
                m_iid.eval(i, i, &theta_iid),
                epsilon = 1e-12
            );
        }
        for i in 0..4 {
            // off-diagonal debe ser 0 cuando ρ=0
            assert_abs_diff_eq!(m_ar1.eval(i, i + 1, &theta_ar1), 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn ar1_boundary_diagonal_uses_stationary_scaling() {
        let m = Ar1Model::new(5);
        // θ = [log(3), arctanh(0.7)]  →  κ=3, ρ=0.7
        let kappa = 3.0_f64;
        let rho = 0.7_f64;
        let theta = [kappa.ln(), inla_ar1_theta_from_rho(rho)];
        let expected = kappa / (1.0 - rho * rho);

        assert_abs_diff_eq!(m.eval(0, 0, &theta), expected, epsilon = 1e-10);
        assert_abs_diff_eq!(m.eval(4, 4, &theta), expected, epsilon = 1e-10);
    }

    #[test]
    fn ar1_interior_diagonal_has_stationary_scaling() {
        let m = Ar1Model::new(5);
        let kappa = 3.0_f64;
        let rho = 0.7_f64;
        let theta = [kappa.ln(), inla_ar1_theta_from_rho(rho)];

        let expected = kappa * (1.0 + rho * rho) / (1.0 - rho * rho);
        for i in 1..4 {
            assert_abs_diff_eq!(m.eval(i, i, &theta), expected, epsilon = 1e-10);
        }
    }

    #[test]
    fn ar1_offdiagonal_has_stationary_scaling() {
        let m = Ar1Model::new(5);
        let kappa = 3.0_f64;
        let rho = 0.7_f64;
        let theta = [kappa.ln(), inla_ar1_theta_from_rho(rho)];

        let expected = -kappa * rho / (1.0 - rho * rho);
        for i in 0..4 {
            assert_abs_diff_eq!(m.eval(i, i + 1, &theta), expected, epsilon = 1e-10);
        }
    }

    #[test]
    fn ar1_correlation_derivative_matches_finite_difference() {
        let m = Ar1Model::new(5);
        let theta = [1.2_f64, inla_ar1_theta_from_rho(0.7)];
        let eps = 1e-6_f64;
        let theta_plus = [theta[0], theta[1] + eps];
        let theta_minus = [theta[0], theta[1] - eps];

        for &(i, j) in &[(0_usize, 0_usize), (2, 2), (2, 3)] {
            let analytic = m.deval(i, j, &theta, 1).unwrap();
            let numeric = (m.eval(i, j, &theta_plus) - m.eval(i, j, &theta_minus)) / (2.0 * eps);
            assert_abs_diff_eq!(analytic, numeric, epsilon = 1e-6);
        }
    }

    #[test]
    fn ar1_is_symmetric() {
        let m = Ar1Model::new(5);
        let theta = [1.0, inla_ar1_theta_from_rho(0.5)];
        for i in 0..4 {
            assert_abs_diff_eq!(
                m.eval(i, i + 1, &theta),
                m.eval(i + 1, i, &theta),
                epsilon = 1e-12
            );
        }
    }

    #[test]
    fn ar1_is_diagonally_dominant_for_valid_rho() {
        // Para |ρ| < 1, Q debe ser diagonalmente dominante → PD.
        // Interior: Q(i,i) = κ(1+ρ²)/(1-ρ²) > 2κρ/(1-ρ²) iff (1-ρ)² > 0 ✓
        let m = Ar1Model::new(10);
        let kappa = 2.0_f64;
        let rho = 0.8_f64;
        let theta = [kappa.ln(), inla_ar1_theta_from_rho(rho)];

        for i in 0..10 {
            let diag = m.eval(i, i, &theta);
            let off_sum: f64 = m
                .graph()
                .neighbors_of(i)
                .iter()
                .map(|&j| m.eval(i, j, &theta).abs())
                .sum();
            // Para nodos extremos también sumamos el vecino hacia "atrás"
            let off_total = if i == 0 || i == 9 {
                off_sum
            } else {
                off_sum * 2.0 // dos vecinos, eval solo sobre triangular superior
            };
            assert!(
                diag > off_total - 1e-10,
                "nodo {i}: diag={diag:.4} off={off_total:.4} — no diagonalmente dominante"
            );
        }
    }

    #[test]
    fn compound_qfunc_is_block_diagonal() {
        let model = CompoundQFunc::new(vec![
            (0, Box::new(IidModel::new(2))),
            (2, Box::new(Ar1Model::new(3))),
        ]);
        let theta = [2.0_f64.ln(), 3.0_f64.ln(), inla_ar1_theta_from_rho(0.5)];
        let ar1_scale = 3.0 / (1.0 - 0.5 * 0.5);

        assert_eq!(model.graph().n(), 5);
        assert_eq!(model.n_hyperparams(), 3);
        assert!(!model.graph().are_neighbors(1, 2));
        assert!(model.graph().are_neighbors(2, 3));
        assert!(model.graph().are_neighbors(3, 4));

        assert_abs_diff_eq!(model.eval(0, 0, &theta), 2.0, epsilon = 1e-12);
        assert_abs_diff_eq!(model.eval(2, 2, &theta), ar1_scale, epsilon = 1e-12);
        assert_abs_diff_eq!(
            model.eval(3, 3, &theta),
            ar1_scale * (1.0 + 0.5 * 0.5),
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(model.eval(2, 3, &theta), -ar1_scale * 0.5, epsilon = 1e-12);
        assert_abs_diff_eq!(model.eval(0, 2, &theta), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn compound_qfunc_is_improper_if_any_block_is_improper() {
        let model = CompoundQFunc::new(vec![
            (0, Box::new(IidModel::new(2))),
            (2, Box::new(Rw1Model::new(3))),
        ]);
        assert!(!model.is_proper());
    }

    #[test]
    fn compound_qfunc_with_rw2_uses_second_order_block_values() {
        let model = CompoundQFunc::new(vec![
            (0, Box::new(IidModel::new(2))),
            (2, Box::new(Rw2Model::new(4))),
        ]);
        let theta = [2.0_f64.ln(), 3.0_f64.ln()];
        let tau_rw2 = 3.0_f64;

        assert_eq!(model.graph().n(), 6);
        assert_eq!(model.n_hyperparams(), 2);
        assert!(!model.graph().are_neighbors(1, 2));
        assert!(model.graph().are_neighbors(2, 3));
        assert!(model.graph().are_neighbors(2, 4));
        assert_abs_diff_eq!(model.eval(0, 0, &theta), 2.0, epsilon = 1e-12);
        assert_abs_diff_eq!(model.eval(2, 2, &theta), tau_rw2, epsilon = 1e-12);
        assert_abs_diff_eq!(model.eval(3, 3, &theta), 5.0 * tau_rw2, epsilon = 1e-12);
        assert_abs_diff_eq!(model.eval(2, 3, &theta), -2.0 * tau_rw2, epsilon = 1e-12);
        assert_abs_diff_eq!(model.eval(2, 4, &theta), tau_rw2, epsilon = 1e-12);
    }

    #[test]
    fn compound_qfunc_log_det_term_sums_intrinsic_blocks() {
        let model = CompoundQFunc::new(vec![
            (0, Box::new(IidModel::new(2))),
            (2, Box::new(Rw1Model::new(3))),
            (5, Box::new(Rw2Model::new(4))),
        ]);
        let theta = [0.1_f64, 0.3_f64, 0.7_f64];
        let expected = 2.0 * theta[1] + 2.0 * theta[2];
        assert_abs_diff_eq!(model.log_det_term(&theta), expected, epsilon = 1e-12);
    }

    #[test]
    fn ar1_internal_theta_matches_inla_transform() {
        let theta = 2.0_f64;
        let rho = (theta * 0.5).tanh();
        assert_abs_diff_eq!(rho, 0.761_594_155_955_764_9, epsilon = 1e-12);
    }

    #[test]
    fn ar1_log_prior_uses_normal_prior_density_for_correlation_theta() {
        let model = Ar1Model::new(5);
        let theta = [2.0_f64.ln(), 2.0_f64];
        let rate = 5e-5_f64;
        let precision = 0.15_f64;
        let expected = rate.ln() + theta[0] - rate * theta[0].exp()
            + 0.5 * (precision.ln() - std::f64::consts::TAU.ln())
            - 0.5 * precision * theta[1] * theta[1];
        assert_abs_diff_eq!(model.log_prior(&theta), expected, epsilon = 1e-12);
    }

    // ── Ar2Model ───────────────────────────────────────────────────────────────────────

    #[test]
    fn ar2_graph_is_pentadiagonal() {
        let m = Ar2Model::new(8);
        assert_eq!(m.graph().n(), 8);
        assert_eq!(m.graph().nnz(), 8 + 2 * (7 + 6));
        assert_eq!(m.n_hyperparams(), 3);
    }

    #[test]
    fn ar2_with_zero_pacf_reduces_to_iid() {
        let m_ar2 = Ar2Model::new(6);
        let m_iid = IidModel::new(6);
        let theta_ar2 = [0.0, 0.0, 0.0];
        let theta_iid = [0.0];

        for i in 0..6 {
            assert_abs_diff_eq!(
                m_ar2.eval(i, i, &theta_ar2),
                m_iid.eval(i, i, &theta_iid),
                epsilon = 1e-12
            );
        }
        for i in 0..5 {
            assert_abs_diff_eq!(m_ar2.eval(i, i + 1, &theta_ar2), 0.0, epsilon = 1e-12);
        }
        for i in 0..4 {
            assert_abs_diff_eq!(m_ar2.eval(i, i + 2, &theta_ar2), 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn ar2_with_zero_second_pacf_matches_ar1() {
        let m_ar2 = Ar2Model::new(7);
        let m_ar1 = Ar1Model::new(7);
        let rho = 0.6_f64;
        let theta_corr = inla_ar1_theta_from_rho(rho);
        let theta_ar2 = [1.3_f64, theta_corr, 0.0];
        let theta_ar1 = [1.3_f64, theta_corr];

        for i in 0..7 {
            assert_abs_diff_eq!(
                m_ar2.eval(i, i, &theta_ar2),
                m_ar1.eval(i, i, &theta_ar1),
                epsilon = 1e-10
            );
        }
        for i in 0..6 {
            assert_abs_diff_eq!(
                m_ar2.eval(i, i + 1, &theta_ar2),
                m_ar1.eval(i, i + 1, &theta_ar1),
                epsilon = 1e-10
            );
        }
        for i in 0..5 {
            assert_abs_diff_eq!(m_ar2.eval(i, i + 2, &theta_ar2), 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn ar2_matches_dense_stationary_precision() {
        let n = 6usize;
        let pacf1 = 0.6_f64;
        let pacf2 = -0.25_f64;
        let theta = [
            1.8_f64,
            inla_ar1_theta_from_rho(pacf1),
            inla_ar1_theta_from_rho(pacf2),
        ];
        let model = Ar2Model::new(n);
        let (phi1, phi2) = ar2_phi_from_pacf(pacf1, pacf2);
        let acf = ar2_acf_from_phi(phi1, phi2, n - 1);

        let mut sigma = vec![0.0_f64; n * n];
        let marginal_var = (-theta[0]).exp();
        for i in 0..n {
            for j in 0..n {
                sigma[i * n + j] = marginal_var * acf[i.abs_diff(j)];
            }
        }
        let expected_q = dense_spd_inverse(&sigma, n);

        for i in 0..n {
            for j in 0..n {
                let got = if i == j || model.graph().are_neighbors(i, j) {
                    model.eval(i, j, &theta)
                } else {
                    0.0
                };
                assert_abs_diff_eq!(got, expected_q[i * n + j], epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn ar2_is_symmetric() {
        let m = Ar2Model::new(6);
        let theta = [
            1.1_f64,
            inla_ar1_theta_from_rho(0.4),
            inla_ar1_theta_from_rho(-0.2),
        ];
        for i in 0..6 {
            for j in 0..6 {
                assert_abs_diff_eq!(m.eval(i, j, &theta), m.eval(j, i, &theta), epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn ar2_log_prior_is_finite_on_inla_internal_scale() {
        let m = Ar2Model::new(6);
        let theta = [2.0_f64.ln(), 1.0_f64, 0.0_f64];
        assert!(m.log_prior(&theta).is_finite());
    }
}
