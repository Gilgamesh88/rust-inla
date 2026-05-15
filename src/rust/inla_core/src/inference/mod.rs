use std::time::Instant;

use crate::diagnostics::RunDiagnosticsSummary;
use crate::error::InlaError;
use crate::likelihood::LogLikelihood;
use crate::marginal::Marginal;
use crate::models::QFunc;
use crate::optimizer::{self, LaplaceDecomposition, OptimizerParams};
use crate::problem::Problem;

pub struct ThetaStateEvidence<'a> {
    pub n_support: usize,
    pub support: &'a [f64],
    pub fixed_precision: &'a [f64],
    pub fixed_linear: &'a [f64],
    pub latent_precision_diag: &'a [f64],
    pub latent_linear: &'a [f64],
    pub latent_precision_i: Option<&'a [usize]>,
    pub latent_precision_j: Option<&'a [usize]>,
    pub latent_precision_x: Option<&'a [f64]>,
    pub latent_fixed_precision: &'a [f64],
    pub log_constant: &'a [f64],
}

pub struct SelectedStateEvidence {
    pub fixed_precision: Option<Vec<f64>>,
    pub fixed_linear: Option<Vec<f64>>,
    pub latent_precision_diag: Option<Vec<f64>>,
    pub latent_linear: Option<Vec<f64>>,
    pub latent_precision_i: Option<Vec<usize>>,
    pub latent_precision_j: Option<Vec<usize>>,
    pub latent_precision_x: Option<Vec<f64>>,
    pub latent_fixed_precision: Option<Vec<f64>>,
    pub log_constant: f64,
}

impl SelectedStateEvidence {
    pub fn has_any(&self) -> bool {
        self.fixed_precision.is_some()
            || self.fixed_linear.is_some()
            || self.latent_precision_diag.is_some()
            || self.latent_linear.is_some()
            || self.latent_precision_x.is_some()
            || self.latent_fixed_precision.is_some()
            || self.log_constant != 0.0
    }
}

pub struct InlaModel<'a> {
    pub qfunc: &'a dyn QFunc,
    pub likelihood: &'a dyn LogLikelihood,
    pub y: &'a [f64],
    pub theta_init: Vec<f64>,
    pub latent_init: Vec<f64>,
    pub fixed_init: Vec<f64>,
    pub fixed_matrix: Option<&'a [f64]>,
    pub n_fixed: usize,
    pub n_latent: usize,
    pub a_i: Option<&'a [usize]>,
    pub a_j: Option<&'a [usize]>,
    pub a_x: Option<&'a [f64]>,
    pub offset: Option<&'a [f64]>,
    pub extr_constr: Option<&'a [f64]>,
    pub n_constr: usize,
    pub fixed_state_precision: Option<&'a [f64]>,
    pub fixed_state_linear: Option<&'a [f64]>,
    pub latent_state_precision_diag: Option<&'a [f64]>,
    pub latent_state_linear: Option<&'a [f64]>,
    pub latent_state_precision_i: Option<&'a [usize]>,
    pub latent_state_precision_j: Option<&'a [usize]>,
    pub latent_state_precision_x: Option<&'a [f64]>,
    pub latent_fixed_state_precision: Option<&'a [f64]>,
    pub theta_state_evidence: Option<ThetaStateEvidence<'a>>,
}

impl<'a> InlaModel<'a> {
    fn theta_state_blend(
        &self,
        theta: &[f64],
        state: &ThetaStateEvidence<'_>,
    ) -> (usize, usize, f64) {
        if state.n_support == 0 || theta.is_empty() {
            return (0, 0, 0.0);
        }

        let n_theta = theta.len();
        if n_theta == 1 {
            let theta_value = theta[0];
            let mut order: Vec<usize> = (0..state.n_support).collect();
            order.sort_by(|&lhs, &rhs| {
                state.support[lhs]
                    .partial_cmp(&state.support[rhs])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let first = order[0];
            let last = order[state.n_support - 1];
            if theta_value <= state.support[first] {
                return (first, first, 0.0);
            }
            if theta_value >= state.support[last] {
                return (last, last, 0.0);
            }

            for pair in order.windows(2) {
                let left = pair[0];
                let right = pair[1];
                let theta_left = state.support[left];
                let theta_right = state.support[right];
                if theta_value >= theta_left && theta_value <= theta_right {
                    let denom = theta_right - theta_left;
                    let right_weight = if denom.abs() <= f64::EPSILON {
                        0.0
                    } else {
                        ((theta_value - theta_left) / denom).clamp(0.0, 1.0)
                    };
                    return (left, right, right_weight);
                }
            }
        }

        let mut best = 0usize;
        let mut best_dist = f64::INFINITY;
        for support_idx in 0..state.n_support {
            let mut dist = 0.0_f64;
            for (theta_idx, theta_value) in theta.iter().enumerate().take(n_theta) {
                let diff = *theta_value - state.support[support_idx * n_theta + theta_idx];
                dist += diff * diff;
            }
            if dist < best_dist {
                best_dist = dist;
                best = support_idx;
            }
        }
        (best, best, 0.0)
    }

    fn theta_state_blend_weights(
        &self,
        theta: &[f64],
        state: &ThetaStateEvidence<'_>,
    ) -> Vec<(usize, f64)> {
        if state.n_support == 0 || theta.is_empty() {
            return vec![(0, 1.0)];
        }
        if theta.len() == 1 {
            let (left, right, right_weight) = self.theta_state_blend(theta, state);
            if left == right || right_weight <= 0.0 {
                return vec![(left, 1.0)];
            }
            return vec![(left, 1.0 - right_weight), (right, right_weight)];
        }

        let n_theta = theta.len();
        let mut scale = vec![1.0_f64; n_theta];
        let mut min_support = vec![f64::INFINITY; n_theta];
        let mut max_support = vec![-f64::INFINITY; n_theta];
        for (theta_idx, scale_i) in scale.iter_mut().enumerate().take(n_theta) {
            let mut min_value = f64::INFINITY;
            let mut max_value = -f64::INFINITY;
            for support_idx in 0..state.n_support {
                let value = state.support[support_idx * n_theta + theta_idx];
                min_value = min_value.min(value);
                max_value = max_value.max(value);
            }
            let range = max_value - min_value;
            if range.is_finite() && range > 1e-12 {
                *scale_i = range;
            }
            min_support[theta_idx] = min_value;
            max_support[theta_idx] = max_value;
        }

        let mut distances: Vec<(usize, f64)> = (0..state.n_support)
            .map(|support_idx| {
                let mut dist = 0.0_f64;
                for (theta_idx, theta_value) in theta.iter().enumerate().take(n_theta) {
                    let diff = (*theta_value - state.support[support_idx * n_theta + theta_idx])
                        / scale[theta_idx];
                    dist += diff * diff;
                }
                (support_idx, dist)
            })
            .collect();
        distances.sort_by(|lhs, rhs| {
            lhs.1
                .partial_cmp(&rhs.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if distances[0].1 <= 1e-12 {
            return vec![(distances[0].0, 1.0)];
        }

        let outside_support = theta.iter().enumerate().any(|(theta_idx, theta_value)| {
            *theta_value < min_support[theta_idx] - 1e-10
                || *theta_value > max_support[theta_idx] + 1e-10
        });
        if outside_support {
            return vec![(distances[0].0, 1.0)];
        }

        let target_count = (2 * n_theta + 1).min(state.n_support).max(1);
        let radius_limit = (distances[0].1 * 4.0).max(1e-10);
        let mut selected = Vec::new();
        for &(support_idx, dist) in &distances {
            if selected.len() < target_count || dist <= radius_limit {
                selected.push((support_idx, dist));
            }
            if selected.len() >= target_count && dist > radius_limit {
                break;
            }
        }

        let mut raw_weights = Vec::with_capacity(selected.len());
        let mut weight_sum = 0.0_f64;
        for (support_idx, dist) in selected {
            let weight = 1.0 / dist.max(1e-12);
            raw_weights.push((support_idx, weight));
            weight_sum += weight;
        }
        if weight_sum <= 0.0 || !weight_sum.is_finite() {
            return vec![(distances[0].0, 1.0)];
        }
        raw_weights
            .into_iter()
            .map(|(support_idx, weight)| (support_idx, weight / weight_sum))
            .collect()
    }

    fn blend_theta_state_slice_weighted(
        source: &[f64],
        per_support: usize,
        weights: &[(usize, f64)],
    ) -> Vec<f64> {
        let mut out = vec![0.0_f64; per_support];
        for &(support_idx, weight) in weights {
            let start = support_idx * per_support;
            let slice = &source[start..start + per_support];
            for (target, value) in out.iter_mut().zip(slice.iter()) {
                *target += weight * *value;
            }
        }
        out
    }

    pub fn selected_state_evidence(&self, theta: &[f64]) -> SelectedStateEvidence {
        if let Some(theta_state) = &self.theta_state_evidence {
            let weights = self.theta_state_blend_weights(theta, theta_state);
            let log_constant = weights
                .iter()
                .map(|(support_idx, weight)| weight * theta_state.log_constant[*support_idx])
                .sum();
            return SelectedStateEvidence {
                fixed_precision: Some(Self::blend_theta_state_slice_weighted(
                    theta_state.fixed_precision,
                    self.n_fixed * self.n_fixed,
                    &weights,
                )),
                fixed_linear: Some(Self::blend_theta_state_slice_weighted(
                    theta_state.fixed_linear,
                    self.n_fixed,
                    &weights,
                )),
                latent_precision_diag: Some(Self::blend_theta_state_slice_weighted(
                    theta_state.latent_precision_diag,
                    self.n_latent,
                    &weights,
                )),
                latent_linear: Some(Self::blend_theta_state_slice_weighted(
                    theta_state.latent_linear,
                    self.n_latent,
                    &weights,
                )),
                latent_precision_i: theta_state.latent_precision_i.map(|values| values.to_vec()),
                latent_precision_j: theta_state.latent_precision_j.map(|values| values.to_vec()),
                latent_precision_x: theta_state.latent_precision_x.map(|values| {
                    let n_edges = theta_state.latent_precision_i.map_or(0, |idx| idx.len());
                    Self::blend_theta_state_slice_weighted(values, n_edges, &weights)
                }),
                latent_fixed_precision: Some(Self::blend_theta_state_slice_weighted(
                    theta_state.latent_fixed_precision,
                    self.n_latent * self.n_fixed,
                    &weights,
                )),
                log_constant,
            };
        }

        SelectedStateEvidence {
            fixed_precision: self.fixed_state_precision.map(|values| values.to_vec()),
            fixed_linear: self.fixed_state_linear.map(|values| values.to_vec()),
            latent_precision_diag: self
                .latent_state_precision_diag
                .map(|values| values.to_vec()),
            latent_linear: self.latent_state_linear.map(|values| values.to_vec()),
            latent_precision_i: self.latent_state_precision_i.map(|values| values.to_vec()),
            latent_precision_j: self.latent_state_precision_j.map(|values| values.to_vec()),
            latent_precision_x: self.latent_state_precision_x.map(|values| values.to_vec()),
            latent_fixed_precision: self
                .latent_fixed_state_precision
                .map(|values| values.to_vec()),
            log_constant: 0.0,
        }
    }
}

pub struct InlaParams {
    pub optimizer: OptimizerParams,
    pub marginal_pts: usize,
    pub marginal_sds: f64,
    pub skip_ccd: bool,
}

impl Default for InlaParams {
    fn default() -> Self {
        Self {
            optimizer: OptimizerParams::default(),
            marginal_pts: 75,
            marginal_sds: 4.0,
            skip_ccd: false,
        }
    }
}

pub struct InlaResult {
    pub theta_opt: Vec<f64>,
    pub log_mlik: f64,
    pub log_mlik_theta_opt: f64,
    pub log_mlik_theta_laplace: f64,
    pub theta_laplace_correction: f64,
    pub random: Vec<Marginal>,
    pub fitted: Vec<Marginal>,
    pub n_evals: usize,
    pub fixed_means: Vec<f64>,
    pub fixed_sds: Vec<f64>,
    pub fixed_var_theta_opt: Vec<f64>,
    pub fixed_cov_theta_opt: Vec<f64>,
    pub ccd_thetas: Vec<f64>,
    pub ccd_base_weights: Vec<f64>,
    pub ccd_weights: Vec<f64>,
    pub ccd_log_mlik: Vec<f64>,
    pub ccd_log_weight: Vec<f64>,
    pub ccd_hessian_eigenvalues: Vec<f64>,
    pub theta_evidence_fixed_precision: Vec<f64>,
    pub theta_evidence_fixed_linear: Vec<f64>,
    pub theta_evidence_latent_precision_diag: Vec<f64>,
    pub theta_evidence_latent_linear: Vec<f64>,
    pub theta_evidence_latent_precision_i: Vec<usize>,
    pub theta_evidence_latent_precision_j: Vec<usize>,
    pub theta_evidence_latent_precision_x: Vec<f64>,
    pub theta_evidence_latent_fixed_precision: Vec<f64>,
    pub theta_evidence_log_constant: Vec<f64>,
    pub posterior_mean: Vec<f64>,
    pub latent_var_theta_opt: Vec<f64>,
    pub latent_var_within_theta: Vec<f64>,
    pub latent_var_between_theta: Vec<f64>,
    pub w_opt: Vec<f64>,
    pub laplace_terms: LaplaceDecomposition,
    pub mode_x: Vec<f64>,
    pub mode_beta: Vec<f64>,
    pub mode_eta: Vec<f64>,
    pub mode_grad: Vec<f64>,
    pub mode_curvature_raw: Vec<f64>,
    pub mode_curvature: Vec<f64>,
    pub diagnostics: RunDiagnosticsSummary,
}

pub struct InlaEngine;

fn build_a_rows(model: &InlaModel<'_>) -> Vec<Vec<(usize, f64)>> {
    let mut a_rows = vec![vec![]; model.y.len()];
    if let (Some(a_i), Some(a_j), Some(a_x)) = (model.a_i, model.a_j, model.a_x) {
        for idx in 0..a_i.len() {
            a_rows[a_i[idx]].push((a_j[idx], a_x[idx]));
        }
    } else {
        for (i, row) in a_rows
            .iter_mut()
            .enumerate()
            .take(model.y.len().min(model.n_latent))
        {
            row.push((i, 1.0));
        }
    }
    a_rows
}

fn linear_predictor_for_state(
    model: &InlaModel<'_>,
    a_rows: &[Vec<(usize, f64)>],
    x: &[f64],
    beta: &[f64],
) -> Vec<f64> {
    let n_data = model.y.len();
    let mut eta = vec![0.0_f64; n_data];

    if let Some(fixed_matrix) = model.fixed_matrix {
        for (j, beta_j) in beta.iter().enumerate().take(model.n_fixed) {
            for i in 0..n_data {
                eta[i] += fixed_matrix[i + j * n_data] * *beta_j;
            }
        }
    }

    for i in 0..n_data {
        for &(latent_idx, ax) in &a_rows[i] {
            eta[i] += ax * x[latent_idx];
        }
    }

    if let Some(offset) = model.offset {
        for (eta_i, offset_i) in eta.iter_mut().zip(offset.iter()) {
            *eta_i += *offset_i;
        }
    }

    eta
}

struct ThetaEvidenceBlocks {
    fixed_precision: Vec<f64>,
    fixed_linear: Vec<f64>,
    latent_precision_diag: Vec<f64>,
    latent_linear: Vec<f64>,
    latent_precision_x: Vec<f64>,
    latent_fixed_precision: Vec<f64>,
    log_constant: f64,
}

fn theta_evidence_pair_keys(a_rows: &[Vec<(usize, f64)>]) -> Vec<(usize, usize)> {
    let mut pair_keys = Vec::new();
    for row in a_rows {
        for (idx, &(lhs, _)) in row.iter().enumerate() {
            for &(rhs, _) in &row[(idx + 1)..] {
                if lhs == rhs {
                    continue;
                }
                let key = if lhs < rhs { (lhs, rhs) } else { (rhs, lhs) };
                if !pair_keys.contains(&key) {
                    pair_keys.push(key);
                }
            }
        }
    }
    pair_keys.sort_unstable();
    pair_keys
}

fn theta_evidence_blocks(
    model: &InlaModel<'_>,
    a_rows: &[Vec<(usize, f64)>],
    pair_keys: &[(usize, usize)],
    theta: &[f64],
    x: &[f64],
    beta: &[f64],
) -> ThetaEvidenceBlocks {
    let n_model = model.qfunc.n_hyperparams();
    let theta_lik = &theta[n_model..];
    let n_data = model.y.len();
    let n_latent = model.n_latent;
    let n_fixed = model.n_fixed;
    let eta = linear_predictor_for_state(model, a_rows, x, beta);

    let mut logll = vec![0.0_f64; n_data];
    model
        .likelihood
        .evaluate(&mut logll, &eta, model.y, theta_lik);

    let mut grad = vec![0.0_f64; n_data];
    let mut curvature = vec![0.0_f64; n_data];
    model
        .likelihood
        .gradient_and_curvature(&mut grad, &mut curvature, &eta, model.y, theta_lik);
    for curv_i in &mut curvature {
        *curv_i = (*curv_i).max(1e-6);
    }

    let mut fixed_precision = vec![0.0_f64; n_fixed * n_fixed];
    let mut fixed_linear = vec![0.0_f64; n_fixed];
    let mut latent_precision_diag = vec![0.0_f64; n_latent];
    let mut latent_linear = vec![0.0_f64; n_latent];
    let mut latent_precision_x = vec![0.0_f64; pair_keys.len()];
    let mut latent_fixed_precision = vec![0.0_f64; n_latent * n_fixed];
    let mut weighted_pseudo = vec![0.0_f64; n_data];
    let mut log_constant = 0.0_f64;

    for i in 0..n_data {
        let offset_i = model.offset.map_or(0.0_f64, |offset| offset[i]);
        let centered_mode = eta[i] - offset_i;
        weighted_pseudo[i] = curvature[i] * centered_mode + grad[i];
        log_constant +=
            logll[i] - grad[i] * centered_mode - 0.5 * curvature[i] * centered_mode * centered_mode;
    }

    if let Some(fixed_matrix) = model.fixed_matrix {
        for j1 in 0..n_fixed {
            for i in 0..n_data {
                fixed_linear[j1] += fixed_matrix[i + j1 * n_data] * weighted_pseudo[i];
            }
            for j2 in 0..n_fixed {
                let mut value = 0.0_f64;
                for i in 0..n_data {
                    value += fixed_matrix[i + j1 * n_data]
                        * curvature[i]
                        * fixed_matrix[i + j2 * n_data];
                }
                fixed_precision[j1 * n_fixed + j2] = value;
            }
        }
    }

    for i in 0..n_data {
        for &(latent_idx, ax) in &a_rows[i] {
            latent_precision_diag[latent_idx] += ax * curvature[i] * ax;
            latent_linear[latent_idx] += ax * weighted_pseudo[i];
            if let Some(fixed_matrix) = model.fixed_matrix {
                for j in 0..n_fixed {
                    latent_fixed_precision[latent_idx + j * n_latent] +=
                        ax * curvature[i] * fixed_matrix[i + j * n_data];
                }
            }
        }
        for (idx, &(lhs, ax_lhs)) in a_rows[i].iter().enumerate() {
            for &(rhs, ax_rhs) in &a_rows[i][(idx + 1)..] {
                let value = ax_lhs * curvature[i] * ax_rhs;
                if lhs == rhs {
                    latent_precision_diag[lhs] += 2.0 * value;
                    continue;
                }
                let key = if lhs < rhs { (lhs, rhs) } else { (rhs, lhs) };
                if let Ok(pair_idx) = pair_keys.binary_search(&key) {
                    latent_precision_x[pair_idx] += value;
                }
            }
        }
    }

    ThetaEvidenceBlocks {
        fixed_precision,
        fixed_linear,
        latent_precision_diag,
        latent_linear,
        latent_precision_x,
        latent_fixed_precision,
        log_constant,
    }
}

impl InlaEngine {
    pub fn run(model: &InlaModel<'_>, params: &InlaParams) -> Result<InlaResult, InlaError> {
        let mut problem = Problem::new(model);

        let opt = optimizer::optimize(&mut problem, model, &model.theta_init, &params.optimizer)?;

        let theta_opt = opt.theta_opt.clone();
        let n_model = model.qfunc.n_hyperparams();
        let n = model.n_latent;
        let k = model.n_fixed;

        let mut fixed_var_theta_opt = vec![0.0_f64; k];
        let mut fixed_cov_theta_opt = vec![0.0_f64; k * k];
        let latent_var_theta_opt = if k > 0 {
            let (_, _, _, diag_aug_inv, fixed_cov, _, _) = problem
                .find_mode_with_fixed_effects_with_cov(
                    model,
                    &theta_opt,
                    &opt.mode_x,
                    &opt.mode_beta,
                    20,
                    1e-8,
                )
                .map_err(|err| InlaError::ConvergenceFailed {
                    reason: format!(
                        "Theta-opt conditional covariance solve failed while solving fixed effects: {err}"
                    ),
                })?;
            for j in 0..k {
                fixed_var_theta_opt[j] = fixed_cov[j * k + j];
            }
            fixed_cov_theta_opt = fixed_cov;
            diag_aug_inv.into_iter().map(|v| v.max(1e-12)).collect()
        } else {
            let (_, _, diag_aug_inv) = problem
                .find_mode_with_inverse(model, &theta_opt, &opt.mode_x, 20, 1e-8)
                .map_err(|err| InlaError::ConvergenceFailed {
                    reason: format!(
                        "Theta-opt conditional covariance solve failed while solving latent mode: {err}"
                    ),
                })?;
            diag_aug_inv.into_iter().map(|v| v.max(1e-12)).collect()
        };

        let ccd_grid = if params.skip_ccd {
            crate::optimizer::ccd::CcdIntegration {
                points: vec![crate::optimizer::ccd::CcdPoint {
                    theta: theta_opt.clone(),
                    base_weight: 1.0,
                    weight: 1.0,
                    log_mlik: opt.log_mlik,
                    log_weight: opt.log_mlik,
                }],
                hessian_eigenvalues: vec![],
            }
        } else {
            crate::optimizer::ccd::build_ccd_grid(&mut problem, model, &theta_opt)?
        };
        let log_mlik_theta_opt = opt.log_mlik;
        let theta_laplace_correction = ccd_grid.theta_laplace_correction();
        let log_mlik_theta_laplace = log_mlik_theta_opt + theta_laplace_correction;

        let mut mixed_mean = vec![0.0_f64; n];
        let mut mixed_var_inner = vec![0.0_f64; n];
        let mut mixed_mean_sq = vec![0.0_f64; n];

        let mut mixed_fixed_mean = vec![0.0_f64; k];
        let mut mixed_fixed_second_moment = vec![0.0_f64; k * k];
        let mut mixed_fixed_cov_inner = vec![0.0_f64; k * k];
        let mut mixed_latent_fixed_second_moment = vec![0.0_f64; n * k];
        let mut mixed_latent_fixed_cov_inner = vec![0.0_f64; n * k];
        let mut next_x_warm = opt.mode_x.clone();
        let mut next_beta_warm = opt.mode_beta.clone();
        let a_rows = build_a_rows(model);
        let theta_evidence_pair_keys = theta_evidence_pair_keys(&a_rows);
        let mut theta_evidence_fixed_precision = Vec::new();
        let mut theta_evidence_fixed_linear = Vec::new();
        let mut theta_evidence_latent_precision_diag = Vec::new();
        let mut theta_evidence_latent_linear = Vec::new();
        let theta_evidence_latent_precision_i: Vec<usize> =
            theta_evidence_pair_keys.iter().map(|(i, _)| *i).collect();
        let theta_evidence_latent_precision_j: Vec<usize> =
            theta_evidence_pair_keys.iter().map(|(_, j)| *j).collect();
        let mut theta_evidence_latent_precision_x = Vec::new();
        let mut theta_evidence_latent_fixed_precision = Vec::new();
        let mut theta_evidence_log_constant = Vec::new();

        for (pt_idx, pt) in ccd_grid.points.iter().enumerate() {
            let theta_k = &pt.theta;
            let weight = pt.weight;

            let x_warm = if next_x_warm.len() == n {
                next_x_warm.clone()
            } else {
                vec![0.0_f64; n]
            };
            let mut beta_warm = if next_beta_warm.len() == k {
                next_beta_warm.clone()
            } else {
                vec![0.0_f64; k]
            };

            // Frequency Regime Log-Link Warm Start
            // Bypasses the Newton-Raphson hurdle when the target frequency is extremely low (e.g. freMTPL2freq ~ 0.05).
            if k > 0
                && matches!(
                    model.likelihood.link(),
                    crate::likelihood::LinkFunction::Log
                )
            {
                let valid_y: Vec<f64> = model.y.iter().copied().filter(|y| !y.is_nan()).collect();
                if !valid_y.is_empty() {
                    let avg_y = valid_y.iter().sum::<f64>() / valid_y.len() as f64;
                    if avg_y > 0.0 && avg_y < 0.2 {
                        beta_warm[0] = avg_y.ln();
                    }
                }
            }

            let (fixed_k, mean_k, vars_k, fixed_cov_k, latent_fixed_cov_k) = if k > 0 {
                let (beta, x_hat, _, diag_aug_inv, fixed_cov, latent_fixed_cov, _) = problem
                    .find_mode_with_fixed_effects_with_cov(
                        model, theta_k, &x_warm, &beta_warm, 20, 1e-8,
                    )
                    .map_err(|err| InlaError::ConvergenceFailed {
                        reason: format!(
                            "CCD point {pt_idx} failed while solving fixed effects: {err}"
                        ),
                    })?;
                let vs: Vec<f64> = diag_aug_inv.into_iter().map(|v| v.max(1e-12)).collect();
                (beta, x_hat, vs, fixed_cov, latent_fixed_cov)
            } else {
                let (x_hat, _, diag_aug_inv) = problem
                    .find_mode_with_inverse(model, theta_k, &x_warm, 20, 1e-8)
                    .map_err(|err| InlaError::ConvergenceFailed {
                        reason: format!(
                            "CCD point {pt_idx} failed while solving latent mode: {err}"
                        ),
                    })?;
                let vs: Vec<f64> = diag_aug_inv.into_iter().map(|v| v.max(1e-12)).collect();
                (vec![], x_hat, vs, vec![], vec![])
            };

            next_x_warm = mean_k.clone();
            next_beta_warm = fixed_k.clone();

            if n > 0 && k > 0 {
                let evidence = theta_evidence_blocks(
                    model,
                    &a_rows,
                    &theta_evidence_pair_keys,
                    theta_k,
                    &mean_k,
                    &fixed_k,
                );
                theta_evidence_fixed_precision.extend(evidence.fixed_precision);
                theta_evidence_fixed_linear.extend(evidence.fixed_linear);
                theta_evidence_latent_precision_diag.extend(evidence.latent_precision_diag);
                theta_evidence_latent_linear.extend(evidence.latent_linear);
                theta_evidence_latent_precision_x.extend(evidence.latent_precision_x);
                theta_evidence_latent_fixed_precision.extend(evidence.latent_fixed_precision);
                theta_evidence_log_constant.push(evidence.log_constant);
            }

            for (j1, fixed_value) in fixed_k.iter().enumerate().take(k) {
                mixed_fixed_mean[j1] += weight * *fixed_value;
                for j2 in 0..k {
                    mixed_fixed_second_moment[j1 * k + j2] += weight * *fixed_value * fixed_k[j2];
                    mixed_fixed_cov_inner[j1 * k + j2] += weight * fixed_cov_k[j1 * k + j2];
                }
            }

            for i in 0..n {
                mixed_mean[i] += weight * mean_k[i];
                mixed_mean_sq[i] += weight * mean_k[i] * mean_k[i];
                mixed_var_inner[i] += weight * vars_k[i];
                for j in 0..k {
                    mixed_latent_fixed_second_moment[i + j * n] += weight * mean_k[i] * fixed_k[j];
                    mixed_latent_fixed_cov_inner[i + j * n] +=
                        weight * latent_fixed_cov_k[i + j * n];
                }
            }
        }

        let mut inter_var = vec![0.0_f64; n];
        let mut final_vars = vec![0.0_f64; n];
        for i in 0..n {
            inter_var[i] = (mixed_mean_sq[i] - mixed_mean[i] * mixed_mean[i]).max(0.0);
            final_vars[i] = mixed_var_inner[i] + inter_var[i];
        }

        let mut mixed_fixed_cov = mixed_fixed_cov_inner;
        for j1 in 0..k {
            for j2 in 0..k {
                mixed_fixed_cov[j1 * k + j2] += mixed_fixed_second_moment[j1 * k + j2]
                    - mixed_fixed_mean[j1] * mixed_fixed_mean[j2];
            }
        }

        let mut mixed_latent_fixed_cov = mixed_latent_fixed_cov_inner;
        for j in 0..k {
            for i in 0..n {
                mixed_latent_fixed_cov[i + j * n] += mixed_latent_fixed_second_moment[i + j * n]
                    - mixed_mean[i] * mixed_fixed_mean[j];
            }
        }

        let mut fixed_sds = vec![0.0_f64; k];
        for j in 0..k {
            fixed_sds[j] = mixed_fixed_cov[j * k + j].max(0.0).sqrt();
        }

        let mut posterior_mean = mixed_mean.clone();
        let variances = final_vars.clone();

        // --------------------------------------------------------------------
        // SUM-TO-ZERO identifiability adjustment for intrinsic latent fields.
        //
        // INLA devel defaults differ by latent model: intrinsic models like rw1
        // and rw2 use constraints by default, while proper models like iid and
        // ar1 do not. Applying this projection to proper models shifts the
        // latent mean into the intercept and creates an artificial intercept gap.
        //
        // This is still a coarse whole-field switch for compound models. When we
        // add mixed proper/improper blocks, this needs to become block-specific.
        // --------------------------------------------------------------------
        if k > 0 && n > 0 && !model.qfunc.is_proper() {
            let mean_x = posterior_mean.iter().sum::<f64>() / n as f64;
            for posterior_mean_i in posterior_mean.iter_mut().take(n) {
                *posterior_mean_i -= mean_x;
            }
            // Assume the first fixed effect is the global intercept.
            mixed_fixed_mean[0] += mean_x;
        }

        let theta_lik = &theta_opt[n_model..];
        let mut eta_data = vec![0.0_f64; model.y.len()];
        for i in 0..model.y.len() {
            let mut ax_sum = 0.0;
            for &(j, ax) in &a_rows[i] {
                ax_sum += ax * posterior_mean[j];
            }
            let mut xb = 0.0;
            if let Some(fixed_matrix) = model.fixed_matrix {
                for (j, fixed_mean) in mixed_fixed_mean.iter().enumerate().take(k) {
                    xb += fixed_matrix[i + j * model.y.len()] * *fixed_mean;
                }
            }
            eta_data[i] = ax_sum + xb;
        }
        if let Some(offset) = model.offset {
            for (eta_i, offset_i) in eta_data.iter_mut().zip(offset.iter()) {
                *eta_i += *offset_i;
            }
        }

        let likelihood_started = Instant::now();
        let mut grad_data = vec![0.0_f64; model.y.len()];
        let mut curv_data = vec![0.0_f64; model.y.len()];
        model.likelihood.gradient_and_curvature(
            &mut grad_data,
            &mut curv_data,
            &eta_data,
            model.y,
            theta_lik,
        );
        problem.diagnostics_mut().likelihood_assembly_time += likelihood_started.elapsed();

        let mut w_opt = vec![0.0_f64; n];
        for i in 0..model.y.len() {
            for &(j, _) in &a_rows[i] {
                // Just assign curvature to the involved latent nodes
                w_opt[j] += curv_data[i];
            }
        }
        for w_opt_i in w_opt.iter_mut().take(n) {
            *w_opt_i = (*w_opt_i).max(1e-6);
        }

        let random: Vec<Marginal> = (0..n)
            .map(|i| {
                let mean = posterior_mean[i];
                let sd = variances[i].sqrt().max(1e-10);
                let lo = mean - params.marginal_sds * sd;
                let hi = mean + params.marginal_sds * sd;
                let pts = params.marginal_pts;
                let x: Vec<f64> = (0..pts)
                    .map(|_k| lo + (hi - lo) * _k as f64 / (pts - 1) as f64)
                    .collect();
                let y: Vec<f64> = x
                    .iter()
                    .map(|&xi| {
                        let z = (xi - mean) / sd;
                        (-0.5 * z * z).exp()
                    })
                    .collect();
                Marginal::new(x, y)
            })
            .collect();

        let fitted: Vec<Marginal> = (0..model.y.len())
            .map(|i| {
                let mean = eta_data[i];
                let mut var = 0.0;
                for &(j, ax) in &a_rows[i] {
                    var += ax * ax * variances[j];
                }
                if let Some(fixed_matrix) = model.fixed_matrix {
                    for j1 in 0..k {
                        let x_i_j1 = fixed_matrix[i + j1 * model.y.len()];
                        for j2 in 0..k {
                            var += x_i_j1
                                * mixed_fixed_cov[j1 * k + j2]
                                * fixed_matrix[i + j2 * model.y.len()];
                        }
                    }
                    for &(latent_idx, ax) in &a_rows[i] {
                        for j in 0..k {
                            var += 2.0
                                * ax
                                * mixed_latent_fixed_cov[latent_idx + j * n]
                                * fixed_matrix[i + j * model.y.len()];
                        }
                    }
                }
                let sd = var.sqrt().max(1e-10);
                let lo = mean - params.marginal_sds * sd;
                let hi = mean + params.marginal_sds * sd;
                let pts = params.marginal_pts;
                let x: Vec<f64> = (0..pts)
                    .map(|_k| lo + (hi - lo) * _k as f64 / (pts - 1) as f64)
                    .collect();
                let y: Vec<f64> = x
                    .iter()
                    .map(|&xi| {
                        let z = (xi - mean) / sd;
                        (-0.5 * z * z).exp()
                    })
                    .collect();
                Marginal::new(x, y)
            })
            .collect();

        let mut ccd_thetas = Vec::new();
        let mut ccd_base_weights = Vec::new();
        let mut ccd_weights = Vec::new();
        let mut ccd_log_mlik = Vec::new();
        let mut ccd_log_weight = Vec::new();
        for pt in &ccd_grid.points {
            ccd_thetas.extend_from_slice(&pt.theta);
            ccd_base_weights.push(pt.base_weight);
            ccd_weights.push(pt.weight);
            ccd_log_mlik.push(if pt.theta.is_empty() {
                log_mlik_theta_opt
            } else {
                pt.log_mlik
            });
            ccd_log_weight.push(if pt.theta.is_empty() {
                log_mlik_theta_opt
            } else {
                pt.log_weight
            });
        }
        let diagnostics = problem.diagnostics_summary();

        Ok(InlaResult {
            theta_opt,
            log_mlik: log_mlik_theta_opt,
            log_mlik_theta_opt,
            log_mlik_theta_laplace,
            theta_laplace_correction,
            random,
            fitted,
            n_evals: opt.n_evals,
            fixed_means: mixed_fixed_mean,
            fixed_sds,
            fixed_var_theta_opt,
            fixed_cov_theta_opt,
            ccd_thetas,
            ccd_base_weights,
            ccd_weights,
            ccd_log_mlik,
            ccd_log_weight,
            ccd_hessian_eigenvalues: ccd_grid.hessian_eigenvalues,
            theta_evidence_fixed_precision,
            theta_evidence_fixed_linear,
            theta_evidence_latent_precision_diag,
            theta_evidence_latent_linear,
            theta_evidence_latent_precision_i,
            theta_evidence_latent_precision_j,
            theta_evidence_latent_precision_x,
            theta_evidence_latent_fixed_precision,
            theta_evidence_log_constant,
            posterior_mean,
            latent_var_theta_opt,
            latent_var_within_theta: mixed_var_inner,
            latent_var_between_theta: inter_var,
            w_opt,
            laplace_terms: opt.laplace_terms,
            mode_x: opt.mode_x,
            mode_beta: opt.mode_beta,
            mode_eta: opt.mode_eta,
            mode_grad: opt.mode_grad,
            mode_curvature_raw: opt.mode_curvature_raw,
            mode_curvature: opt.mode_curvature,
            diagnostics,
        })
    }
}
