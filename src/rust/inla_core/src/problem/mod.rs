use std::{collections::HashMap, time::Instant};

use crate::diagnostics::{LaplacePhase, RunDiagnostics, RunDiagnosticsSummary};
use crate::error::InlaError;
use crate::graph::Graph;
use crate::models::QFunc;
use crate::solver::{FaerSolver, SpMat, SparseSolver};

pub const PRIOR_PREC_BETA: f64 = 0.001;
type ModeCoreResult = (Vec<f64>, f64, Option<Vec<f64>>);
type FixedEffectsModeResult = (Vec<f64>, Vec<f64>, f64, Vec<f64>, f64);
type FixedEffectsModeCoreResult = (Vec<f64>, Vec<f64>, f64, Option<Vec<f64>>, f64);

#[derive(Clone, Copy)]
struct ModeControl {
    max_iter: usize,
    tol: f64,
    need_diag_aug_inv: bool,
}

pub struct Problem {
    graph: Graph,
    a_rows: Vec<Vec<(usize, f64)>>,
    a_single_j: Option<Vec<usize>>,
    a_single_x: Option<Vec<f64>>,
    pub(crate) solver: FaerSolver,
    log_det: f64,
    pub n_evals: usize,
    diagnostics: RunDiagnostics,
}

fn build_a_rows(
    model: &crate::inference::InlaModel<'_>,
    n_data: usize,
    n_latent: usize,
) -> Vec<Vec<(usize, f64)>> {
    let mut a_rows = vec![vec![]; n_data];
    if let (Some(a_i), Some(a_j), Some(a_x)) = (model.a_i, model.a_j, model.a_x) {
        for k in 0..a_i.len() {
            a_rows[a_i[k]].push((a_j[k], a_x[k]));
        }
    } else {
        for (i, row) in a_rows.iter_mut().enumerate().take(n_data.min(n_latent)) {
            row.push((i, 1.0));
        }
    }
    a_rows
}

fn add_offset(eta_data: &mut [f64], offset: Option<&[f64]>) {
    if let Some(offset) = offset {
        for (eta_i, offset_i) in eta_data.iter_mut().zip(offset.iter()) {
            *eta_i += *offset_i;
        }
    }
}

fn extract_singleton_a_rows(a_rows: &[Vec<(usize, f64)>]) -> Option<(Vec<usize>, Vec<f64>)> {
    let mut a_j = Vec::with_capacity(a_rows.len());
    let mut a_x = Vec::with_capacity(a_rows.len());
    for row in a_rows {
        if row.len() != 1 {
            return None;
        }
        a_j.push(row[0].0);
        a_x.push(row[0].1);
    }
    Some((a_j, a_x))
}

fn clamp_curvature(curv_data: &mut [f64]) {
    for curv_i in curv_data.iter_mut() {
        *curv_i = (*curv_i).max(1e-6);
    }
}

fn compensated_sum<I>(iter: I) -> f64
where
    I: IntoIterator<Item = f64>,
{
    let mut sum = 0.0_f64;
    let mut c = 0.0_f64;
    for x in iter {
        let y = x - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    sum
}

enum AtwaStorage {
    Diagonal(Vec<f64>),
    Split {
        diag: Vec<f64>,
        offdiag: HashMap<(usize, usize), f64>,
    },
}

impl AtwaStorage {
    fn diag(&self, i: usize) -> f64 {
        match self {
            Self::Diagonal(diag) => diag[i],
            Self::Split { diag, .. } => diag[i],
        }
    }

    fn offdiag(&self, i: usize, j: usize) -> f64 {
        match self {
            Self::Diagonal(_) => 0.0,
            Self::Split { offdiag, .. } => offdiag.get(&(i, j)).copied().unwrap_or(0.0),
        }
    }

    fn value(&self, i: usize, j: usize) -> f64 {
        if i == j {
            self.diag(i)
        } else {
            let (min_j, max_j) = if i < j { (i, j) } else { (j, i) };
            self.offdiag(min_j, max_j)
        }
    }
}

fn schur_stabilization_jitter(
    s_mat: &[f64],
    xtwx_diag: &[f64],
    wcv_diag: &[f64],
    n: usize,
) -> Option<f64> {
    let mut max_scale = 0.0_f64;
    let mut min_diag = f64::INFINITY;
    for j in 0..n {
        max_scale = max_scale.max(xtwx_diag[j].abs()).max(wcv_diag[j].abs());
        min_diag = min_diag.min(s_mat[j * n + j]);
    }

    if !(max_scale.is_finite()) || max_scale <= 0.0 {
        return None;
    }

    // When X'WX and C'Q^{-1}C are both enormous and nearly cancel, the Schur
    // complement can become slightly negative from floating-point loss even if
    // the exact dense block is SPD. Add only a machine-precision-scale ridge.
    let ridge_floor = 1e-12 * max_scale;
    if min_diag <= ridge_floor && min_diag.abs() <= 1e-8 * max_scale {
        Some((ridge_floor - min_diag).max(0.0))
    } else {
        None
    }
}

fn try_dense_cholesky_with_schur_stabilization(
    a: &mut [f64],
    b: &mut [f64],
    n: usize,
    xtwx_diag: &[f64],
    wcv_diag: &[f64],
) -> Result<(f64, Option<f64>), InlaError> {
    if let Ok(log_det) = dense_cholesky_solve(a, b, n) {
        return Ok((log_det, None));
    }

    let jitter = schur_stabilization_jitter(a, xtwx_diag, wcv_diag, n).ok_or_else(|| {
        InlaError::NotPositiveDefiniteContext(
            "dense_cholesky_solve encountered a non-positive pivot".to_string(),
        )
    })?;

    for j in 0..n {
        a[j * n + j] += jitter;
    }

    let log_det = dense_cholesky_solve(a, b, n)?;
    Ok((log_det, Some(jitter)))
}

fn dense_spd_inverse_with_schur_stabilization(
    a: &[f64],
    n: usize,
    xtwx_diag: &[f64],
    wcv_diag: &[f64],
) -> Result<(Vec<f64>, Option<f64>), InlaError> {
    let jitter = schur_stabilization_jitter(a, xtwx_diag, wcv_diag, n);
    let mut base = a.to_vec();
    if let Some(j) = jitter {
        for idx in 0..n {
            base[idx * n + idx] += j;
        }
    }

    let mut inv = vec![0.0_f64; n * n];
    for col in 0..n {
        let mut chol = base.clone();
        let mut rhs = vec![0.0_f64; n];
        rhs[col] = 1.0;
        dense_cholesky_solve(&mut chol, &mut rhs, n)?;
        for row in 0..n {
            inv[row * n + col] = rhs[row];
        }
    }
    Ok((inv, jitter))
}

fn extract_sparse_diagonal(matrix: &SpMat, n: usize) -> Vec<f64> {
    let col_ptr = matrix.symbolic().col_ptr();
    let row_idx = matrix.symbolic().row_idx();
    let all_vals = matrix.val();

    (0..n)
        .map(|j| {
            let pos = row_idx[col_ptr[j]..col_ptr[j + 1]]
                .binary_search(&j)
                .unwrap();
            all_vals[col_ptr[j] + pos]
        })
        .collect()
}

fn build_kriging_system<S: SparseSolver>(
    solver: &mut S,
    constr: &[f64],
    n_latent: usize,
    n_constr: usize,
) -> Result<(Vec<f64>, Vec<f64>), InlaError> {
    let mut v_c = vec![0.0_f64; n_latent * n_constr];
    for c in 0..n_constr {
        let mut vc_col = constr[c * n_latent..(c + 1) * n_latent].to_vec();
        solver.solve_llt(&mut vc_col);
        for k in 0..n_latent {
            v_c[c * n_latent + k] = vc_col[k];
        }
    }

    let mut c_vc = vec![0.0_f64; n_constr * n_constr];
    for c1 in 0..n_constr {
        for c2 in 0..n_constr {
            for k in 0..n_latent {
                c_vc[c1 * n_constr + c2] += constr[c1 * n_latent + k] * v_c[c2 * n_latent + k];
            }
        }
    }

    let mut c_vc_inv = vec![0.0_f64; n_constr * n_constr];
    for c in 0..n_constr {
        let mut rhs = vec![0.0_f64; n_constr];
        rhs[c] = 1.0;
        let mut cvc_tmp = c_vc.clone();
        dense_cholesky_solve(&mut cvc_tmp, &mut rhs, n_constr).map_err(|_| {
            InlaError::NotPositiveDefiniteContext("Kriging C V_c block singular".to_string())
        })?;
        for c2 in 0..n_constr {
            c_vc_inv[c * n_constr + c2] = rhs[c2];
        }
    }

    Ok((v_c, c_vc_inv))
}

fn apply_kriging_correction_to_diagonal(
    diag: &mut [f64],
    v_c: &[f64],      // n_constr x n_latent, row-major
    c_vc_inv: &[f64], // n_constr x n_constr
    n_latent: usize,
    n_constr: usize,
) {
    for i in 0..n_latent {
        let mut correction = 0.0_f64;
        for c1 in 0..n_constr {
            let vc1_i = v_c[c1 * n_latent + i];
            for c2 in 0..n_constr {
                correction += vc1_i * c_vc_inv[c1 * n_constr + c2] * v_c[c2 * n_latent + i];
            }
        }
        diag[i] = (diag[i] - correction).max(1e-12);
    }
}

impl Problem {
    pub fn new(model: &crate::inference::InlaModel<'_>) -> Self {
        let mut graph = model.qfunc.graph().clone();

        // Merge the connections generated by A^T A (Random effects interacting via the same Observation rows)
        if let (Some(a_i), Some(a_j)) = (model.a_i, model.a_j) {
            let a_edges = Graph::build_a_t_a_edges(a_i, a_j, model.y.len());
            graph.merge_edges(&a_edges);
        }

        let mut solver = FaerSolver::new();
        solver.reorder(&mut graph);
        let a_rows = build_a_rows(model, model.y.len(), model.n_latent);
        let (a_single_j, a_single_x) = extract_singleton_a_rows(&a_rows)
            .map(|(j, x)| (Some(j), Some(x)))
            .unwrap_or((None, None));
        Self {
            graph,
            a_rows,
            a_single_j,
            a_single_x,
            solver,
            log_det: 0.0,
            n_evals: 0,
            diagnostics: RunDiagnostics::default(),
        }
    }

    pub fn record_laplace_eval(&mut self, phase: LaplacePhase) {
        self.diagnostics.record_laplace_eval(phase);
    }

    pub fn diagnostics_mut(&mut self) -> &mut RunDiagnostics {
        &mut self.diagnostics
    }

    pub fn diagnostics_summary(&self) -> RunDiagnosticsSummary {
        RunDiagnosticsSummary::from_parts(&self.diagnostics, self.solver.diagnostics())
    }

    pub fn eval(&mut self, qfunc: &dyn QFunc, theta: &[f64]) -> Result<f64, InlaError> {
        self.solver.build(&self.graph, qfunc, theta);
        self.solver.factorize().map_err(|_| {
            crate::error::InlaError::NotPositiveDefiniteContext(
                "Problem::eval Q factorize failed".to_string(),
            )
        })?;
        self.log_det = self.solver.log_determinant();
        self.n_evals += 1;
        Ok(self.log_det)
    }

    pub fn eval_with_inverse(
        &mut self,
        qfunc: &dyn QFunc,
        theta: &[f64],
    ) -> Result<(f64, Vec<f64>), InlaError> {
        self.solver.build(&self.graph, qfunc, theta);
        self.solver.factorize().map_err(|_| {
            crate::error::InlaError::NotPositiveDefiniteContext(
                "Q prior factorize failed".to_string(),
            )
        })?;
        let log_det = self.solver.log_determinant();
        self.log_det = log_det;
        self.n_evals += 1;

        let q_inv = self.solver.selected_inverse()?;
        let col_ptr = q_inv.symbolic().col_ptr();
        let row_idx = q_inv.symbolic().row_idx();
        let all_vals = q_inv.val();

        let n = self.n();
        let diag_qinv: Vec<f64> = (0..n)
            .map(|j| {
                let start = col_ptr[j];
                let end = col_ptr[j + 1];
                let col_rows = &row_idx[start..end];
                let pos = col_rows.binary_search(&j).unwrap();
                all_vals[start + pos]
            })
            .collect();

        Ok((log_det, diag_qinv))
    }

    fn find_mode_core(
        &mut self,
        model: &crate::inference::InlaModel<'_>,
        theta: &[f64],
        x_init: &[f64],
        control: ModeControl,
    ) -> Result<ModeCoreResult, InlaError> {
        self.diagnostics.latent_mode_solve_calls += 1;
        let solve_started = Instant::now();
        let result = (|| {
            let n_latent = self.n();
            let n_data = model.y.len();
            let n_model = model.qfunc.n_hyperparams();
            let theta_model = &theta[..n_model];
            let theta_lik = &theta[n_model..];

            let mut x = if x_init.len() == n_latent {
                x_init.to_vec()
            } else {
                vec![0.0_f64; n_latent]
            };
            let mut log_det_aug = 0.0_f64;
            let mut iterations = 0_usize;
            let mut converged = false;

            for _iter in 0..control.max_iter {
                iterations += 1;
                let x_old = x.clone();
                let assembly_started = Instant::now();

                let mut eta_data = vec![0.0_f64; n_data];
                if let (Some(a_single_j), Some(a_single_x)) = (&self.a_single_j, &self.a_single_x) {
                    for (i, eta_i) in eta_data.iter_mut().enumerate().take(n_data) {
                        *eta_i = a_single_x[i] * x[a_single_j[i]];
                    }
                } else {
                    for (i, eta_i) in eta_data.iter_mut().enumerate().take(n_data) {
                        for &(j, ax) in &self.a_rows[i] {
                            *eta_i += ax * x[j];
                        }
                    }
                }
                add_offset(&mut eta_data, model.offset);

                let mut grad_data = vec![0.0_f64; n_data];
                let mut curv_data = vec![0.0_f64; n_data];
                model.likelihood.gradient_and_curvature(
                    &mut grad_data,
                    &mut curv_data,
                    &eta_data,
                    model.y,
                    theta_lik,
                );

                clamp_curvature(&mut curv_data);

                let (b_x, atwa) = if let (Some(a_single_j), Some(a_single_x)) =
                    (&self.a_single_j, &self.a_single_x)
                {
                    let mut b_x = vec![0.0_f64; n_latent];
                    let mut atwa_diag = vec![0.0_f64; n_latent];
                    for i in 0..n_data {
                        let k = a_single_j[i];
                        let ax = a_single_x[i];
                        let grad = grad_data[i];
                        let curv = curv_data[i];
                        atwa_diag[k] += ax * curv * ax;

                        let z_i = eta_data[i] + grad / curv;
                        let offset_i = model.offset.map_or(0.0_f64, |offset| offset[i]);
                        let wz = curv * (z_i - offset_i);
                        b_x[k] += ax * wz;
                    }
                    (b_x, AtwaStorage::Diagonal(atwa_diag))
                } else {
                    let mut b_x = vec![0.0_f64; n_latent];
                    let mut atwa_diag = vec![0.0_f64; n_latent];
                    let mut atwa_offdiag = HashMap::new();
                    for i in 0..n_data {
                        let grad = grad_data[i];
                        let w = curv_data[i];
                        let row = &self.a_rows[i];
                        let z_i = eta_data[i] + grad / w;
                        let offset_i = model.offset.map_or(0.0_f64, |offset| offset[i]);
                        let wz = w * (z_i - offset_i);
                        for (idx, &(j1, ax1)) in row.iter().enumerate() {
                            b_x[j1] += ax1 * wz;
                            atwa_diag[j1] += ax1 * w * ax1;
                            for &(j2, ax2) in &row[(idx + 1)..] {
                                let (min_j, max_j) = if j1 < j2 { (j1, j2) } else { (j2, j1) };
                                *atwa_offdiag.entry((min_j, max_j)).or_insert(0.0) += ax1 * w * ax2;
                            }
                        }
                    }
                    (
                        b_x,
                        AtwaStorage::Split {
                            diag: atwa_diag,
                            offdiag: atwa_offdiag,
                        },
                    )
                };

                // Solve the same IRLS normal equations used by the fixed-effects
                // path, but without the Schur complement:
                //   (Q + A'WA) x = A'W(z - offset)
                // This keeps the latent-only solver consistent with the
                // fixed-effects solver and avoids depending on the graph's
                // upper-triangular neighbor storage when assembling the RHS.
                let mut rhs = b_x;
                self.diagnostics.likelihood_assembly_time += assembly_started.elapsed();

                let aug = AugmentedQFunc {
                    inner: model.qfunc,
                    atwa: &atwa,
                };
                self.solver.build(&self.graph, &aug, theta_model);
                self.solver.factorize().map_err(|_| {
                    crate::error::InlaError::NotPositiveDefiniteContext(
                        "Sparse Q_aug block 1 singular".to_string(),
                    )
                })?;
                log_det_aug = self.solver.log_determinant();
                self.solver.solve_llt(&mut rhs);
                if let Some(constr) = model.extr_constr {
                    let n_c = model.n_constr;
                    if n_c > 0 {
                        let (v_c, c_vc_inv) =
                            build_kriging_system(&mut self.solver, constr, n_latent, n_c)?;
                        apply_kriging_correction(&mut rhs, &v_c, &c_vc_inv, constr, n_latent, n_c);
                    }
                }
                x = rhs;

                let delta: f64 = x
                    .iter()
                    .zip(x_old.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f64, f64::max);
                if delta < control.tol {
                    converged = true;
                    break;
                }
            }
            self.diagnostics.latent_mode_iterations_total += iterations;
            if !converged {
                self.diagnostics.latent_mode_max_iter_hits += 1;
            }

            let diag_aug_inv = if control.need_diag_aug_inv {
                let mut diag = extract_sparse_diagonal(&self.solver.selected_inverse()?, n_latent);
                if let Some(constr) = model.extr_constr {
                    let n_c = model.n_constr;
                    if n_c > 0 {
                        let (v_c, c_vc_inv) =
                            build_kriging_system(&mut self.solver, constr, n_latent, n_c)?;
                        apply_kriging_correction_to_diagonal(
                            &mut diag, &v_c, &c_vc_inv, n_latent, n_c,
                        );
                    }
                }
                Some(diag)
            } else {
                None
            };

            Ok((x, log_det_aug, diag_aug_inv))
        })();
        self.diagnostics.latent_mode_solve_time += solve_started.elapsed();
        result
    }

    pub fn find_mode_with_inverse(
        &mut self,
        model: &crate::inference::InlaModel<'_>,
        theta: &[f64],
        x_init: &[f64],
        max_iter: usize,
        tol: f64,
    ) -> Result<(Vec<f64>, f64, Vec<f64>), InlaError> {
        let (x, log_det_aug, diag_aug_inv) = self.find_mode_core(
            model,
            theta,
            x_init,
            ModeControl {
                max_iter,
                tol,
                need_diag_aug_inv: true,
            },
        )?;
        Ok((x, log_det_aug, diag_aug_inv.unwrap_or_default()))
    }

    pub fn find_mode_with_logdet_and_warm(
        &mut self,
        model: &crate::inference::InlaModel<'_>,
        theta: &[f64],
        x_init: &[f64],
        max_iter: usize,
        tol: f64,
    ) -> Result<(Vec<f64>, f64), InlaError> {
        let (x, log_det_aug, _) = self.find_mode_core(
            model,
            theta,
            x_init,
            ModeControl {
                max_iter,
                tol,
                need_diag_aug_inv: false,
            },
        )?;
        Ok((x, log_det_aug))
    }

    /// Implements the General Fixed Effects Solver via Matrix Schur Complement
    /// Replaces beta0 with a K-dimensional vector beta.
    fn find_mode_with_fixed_effects_core(
        &mut self,
        model: &crate::inference::InlaModel<'_>,
        theta: &[f64],
        x_init: &[f64],
        beta_init: &[f64],
        control: ModeControl,
    ) -> Result<FixedEffectsModeCoreResult, InlaError> {
        if model.n_fixed == 0 || model.fixed_matrix.is_none() {
            let (x, ld, di) = self.find_mode_core(model, theta, x_init, control)?;
            return Ok((vec![], x, ld, di, 0.0));
        }
        self.diagnostics.latent_mode_solve_calls += 1;
        let solve_started = Instant::now();
        let result = (|| {
            let x_m = model.fixed_matrix.unwrap();
            let n_fixed = model.n_fixed;
            let n_latent = self.n();
            let n_data = model.y.len();
            let n_model = model.qfunc.n_hyperparams();
            let theta_model = &theta[..n_model];
            let theta_lik = &theta[n_model..];

            let mut x = if x_init.len() == n_latent {
                x_init.to_vec()
            } else {
                vec![0.0_f64; n_latent]
            };
            let mut beta = if beta_init.len() == n_fixed {
                beta_init.to_vec()
            } else {
                vec![0.0_f64; n_fixed]
            };

            let mut log_det_aug = 0.0_f64;
            let mut final_w_cross = vec![0.0_f64; n_latent * n_fixed];
            let mut final_w_data = vec![0.0_f64; n_data];
            let mut iterations = 0_usize;
            let mut converged = false;

            for _iter in 0..control.max_iter {
                iterations += 1;
                let x_old = x.clone();
                let beta_old = beta.clone();
                let assembly_started = Instant::now();

                let mut eta_data = vec![0.0_f64; n_data];
                if let (Some(a_single_j), Some(a_single_x)) = (&self.a_single_j, &self.a_single_x) {
                    for (i, eta_i) in eta_data.iter_mut().enumerate().take(n_data) {
                        *eta_i = a_single_x[i] * x[a_single_j[i]];
                        let mut xb = 0.0;
                        for j in 0..n_fixed {
                            xb += x_m[i + j * n_data] * beta[j];
                        }
                        *eta_i += xb;
                    }
                } else {
                    for (i, eta_i) in eta_data.iter_mut().enumerate().take(n_data) {
                        for &(j, ax) in &self.a_rows[i] {
                            *eta_i += ax * x[j];
                        }
                        let mut xb = 0.0;
                        for j in 0..n_fixed {
                            xb += x_m[i + j * n_data] * beta[j];
                        }
                        *eta_i += xb;
                    }
                }
                add_offset(&mut eta_data, model.offset);

                let mut grad_data = vec![0.0_f64; n_data];
                let mut curv_data = vec![0.0_f64; n_data];
                model.likelihood.gradient_and_curvature(
                    &mut grad_data,
                    &mut curv_data,
                    &eta_data,
                    model.y,
                    theta_lik,
                );

                clamp_curvature(&mut curv_data);
                final_w_data = curv_data.clone();

                let (atwa, w_cross, b_beta, b_x) = if let (Some(a_single_j), Some(a_single_x)) =
                    (&self.a_single_j, &self.a_single_x)
                {
                    let mut atwa_diag = vec![0.0_f64; n_latent];
                    let mut w_cross = vec![0.0_f64; n_latent * n_fixed];
                    let mut b_beta = vec![0.0_f64; n_fixed];
                    let mut b_x = vec![0.0_f64; n_latent];
                    for i in 0..n_data {
                        let k = a_single_j[i];
                        let ax = a_single_x[i];
                        let grad = grad_data[i];
                        let curv = curv_data[i];
                        atwa_diag[k] += ax * curv * ax;

                        let z_i = eta_data[i] + grad / curv;
                        let offset_i = model.offset.map_or(0.0_f64, |offset| offset[i]);
                        let wz = curv * (z_i - offset_i);
                        b_x[k] += ax * wz;
                        for j in 0..n_fixed {
                            let x_ij = x_m[i + j * n_data];
                            w_cross[k + j * n_latent] += ax * curv * x_ij;
                            b_beta[j] += x_ij * wz;
                        }
                    }
                    (AtwaStorage::Diagonal(atwa_diag), w_cross, b_beta, b_x)
                } else {
                    let mut atwa_diag = vec![0.0_f64; n_latent];
                    let mut atwa_offdiag = HashMap::new();
                    let mut w_cross = vec![0.0_f64; n_latent * n_fixed];
                    let mut b_beta = vec![0.0_f64; n_fixed];
                    let mut b_x = vec![0.0_f64; n_latent];
                    for i in 0..n_data {
                        let grad = grad_data[i];
                        let curv = curv_data[i];
                        let row = &self.a_rows[i];
                        let z_i = eta_data[i] + grad / curv;
                        let offset_i = model.offset.map_or(0.0_f64, |offset| offset[i]);
                        let wz = curv * (z_i - offset_i);

                        for j in 0..n_fixed {
                            let x_ij = x_m[i + j * n_data];
                            b_beta[j] += x_ij * wz;
                            let w_x_ij = curv * x_ij;
                            for &(k, ax) in row {
                                w_cross[k + j * n_latent] += ax * w_x_ij;
                            }
                        }

                        for (idx, &(j1, ax1)) in row.iter().enumerate() {
                            atwa_diag[j1] += ax1 * curv * ax1;
                            b_x[j1] += ax1 * wz;
                            for &(j2, ax2) in &row[(idx + 1)..] {
                                let (min_j, max_j) = if j1 < j2 { (j1, j2) } else { (j2, j1) };
                                *atwa_offdiag.entry((min_j, max_j)).or_insert(0.0) +=
                                    ax1 * curv * ax2;
                            }
                        }
                    }
                    (
                        AtwaStorage::Split {
                            diag: atwa_diag,
                            offdiag: atwa_offdiag,
                        },
                        w_cross,
                        b_beta,
                        b_x,
                    )
                };
                final_w_cross = w_cross.clone();
                self.diagnostics.likelihood_assembly_time += assembly_started.elapsed();

                let aug = AugmentedQFunc {
                    inner: model.qfunc,
                    atwa: &atwa,
                };
                self.solver.build(&self.graph, &aug, theta_model);

                self.solver.factorize().map_err(|_| {
                    crate::error::InlaError::NotPositiveDefiniteContext(format!(
                        "Sparse Q_aug block 2 singular at iter {}",
                        _iter
                    ))
                })?;

                log_det_aug = self.solver.log_determinant();

                let mut u = b_x.clone();
                self.solver.solve_llt(&mut u);

                let mut v = vec![0.0_f64; n_latent * n_fixed];
                for j in 0..n_fixed {
                    let mut v_col = w_cross[j * n_latent..(j + 1) * n_latent].to_vec();
                    self.solver.solve_llt(&mut v_col);
                    for k in 0..n_latent {
                        v[k + j * n_latent] = v_col[k];
                    }
                }

                if let Some(constr) = model.extr_constr {
                    let n_c = model.n_constr;
                    if n_c > 0 {
                        let (v_c, c_vc_inv) =
                            build_kriging_system(&mut self.solver, constr, n_latent, n_c)?;

                        apply_kriging_correction(&mut u, &v_c, &c_vc_inv, constr, n_latent, n_c);

                        for j in 0..n_fixed {
                            let mut vj = v[j * n_latent..(j + 1) * n_latent].to_vec();
                            apply_kriging_correction(
                                &mut vj, &v_c, &c_vc_inv, constr, n_latent, n_c,
                            );
                            for k in 0..n_latent {
                                v[k + j * n_latent] = vj[k];
                            }
                        }
                    }
                }

                let mut s_mat = vec![0.0_f64; n_fixed * n_fixed];
                let mut s_xtwx_diag = vec![0.0_f64; n_fixed];
                let mut s_wcv_diag = vec![0.0_f64; n_fixed];
                for j1 in 0..n_fixed {
                    for j2 in 0..n_fixed {
                        let mut xtwx =
                            compensated_sum((0..n_data).map(|i| {
                                x_m[i + j1 * n_data] * curv_data[i] * x_m[i + j2 * n_data]
                            }));
                        if j1 == j2 {
                            xtwx += PRIOR_PREC_BETA;
                        }

                        let wcv = compensated_sum(
                            (0..n_latent)
                                .map(|k| w_cross[k + j1 * n_latent] * v[k + j2 * n_latent]),
                        );
                        s_mat[j1 * n_fixed + j2] = xtwx - wcv;
                        if j1 == j2 {
                            s_xtwx_diag[j1] = xtwx;
                            s_wcv_diag[j1] = wcv;
                        }
                    }
                }

                let mut b_s = vec![0.0_f64; n_fixed];
                for j in 0..n_fixed {
                    let wcu =
                        compensated_sum((0..n_latent).map(|k| w_cross[k + j * n_latent] * u[k]));
                    b_s[j] = b_beta[j] - wcu;
                }

                let s_mat_bkp = s_mat.clone();
                match try_dense_cholesky_with_schur_stabilization(
                    &mut s_mat,
                    &mut b_s,
                    n_fixed,
                    &s_xtwx_diag,
                    &s_wcv_diag,
                ) {
                    Ok((_, retry)) => retry,
                    Err(_) => {
                        let w_min = curv_data.iter().fold(f64::INFINITY, |acc, &x| acc.min(x));
                        let w_max = curv_data
                            .iter()
                            .fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));
                        return Err(crate::error::InlaError::NotPositiveDefiniteContext(
                        format!(
                            "Schur S_mat loop singular, diag0: {:?}, xtwx0: {:?}, wcv0: {:?}, w_min: {:?}, w_max: {:?}, theta: {:?}",
                            s_mat_bkp[0],
                            s_xtwx_diag.first().copied().unwrap_or(0.0),
                            s_wcv_diag.first().copied().unwrap_or(0.0),
                            w_min,
                            w_max,
                            theta,
                        ),
                    ));
                    }
                };
                beta[..n_fixed].copy_from_slice(&b_s[..n_fixed]);

                for k in 0..n_latent {
                    let mut vb = 0.0;
                    for j in 0..n_fixed {
                        vb += v[k + j * n_latent] * beta[j];
                    }
                    x[k] = u[k] - vb;
                }

                let mut delta_max = 0.0_f64;
                for (a, b_val) in x.iter().zip(x_old.iter()) {
                    delta_max = delta_max.max((a - b_val).abs());
                }
                for (a, b_val) in beta.iter().zip(beta_old.iter()) {
                    delta_max = delta_max.max((a - b_val).abs());
                }

                if delta_max < control.tol {
                    converged = true;
                    break;
                }
            }
            self.diagnostics.latent_mode_iterations_total += iterations;
            if !converged {
                self.diagnostics.latent_mode_max_iter_hits += 1;
            }

            let mut diag_aug_inv = if control.need_diag_aug_inv {
                Some(extract_sparse_diagonal(
                    &self.solver.selected_inverse()?,
                    n_latent,
                ))
            } else {
                None
            };

            let mut v = vec![0.0_f64; n_latent * n_fixed];
            for j in 0..n_fixed {
                let mut v_col = final_w_cross[j * n_latent..(j + 1) * n_latent].to_vec();
                self.solver.solve_llt(&mut v_col);
                for k in 0..n_latent {
                    v[k + j * n_latent] = v_col[k];
                }
            }

            if let Some(constr) = model.extr_constr {
                let n_c = model.n_constr;
                if n_c > 0 {
                    let (v_c, c_vc_inv) =
                        build_kriging_system(&mut self.solver, constr, n_latent, n_c)?;
                    for j in 0..n_fixed {
                        let mut vj = v[j * n_latent..(j + 1) * n_latent].to_vec();
                        apply_kriging_correction(&mut vj, &v_c, &c_vc_inv, constr, n_latent, n_c);
                        for k in 0..n_latent {
                            v[k + j * n_latent] = vj[k];
                        }
                    }
                    if let Some(diag_latent_cov) = diag_aug_inv.as_mut() {
                        apply_kriging_correction_to_diagonal(
                            diag_latent_cov,
                            &v_c,
                            &c_vc_inv,
                            n_latent,
                            n_c,
                        );
                    }
                }
            }

            let mut s_mat = vec![0.0_f64; n_fixed * n_fixed];
            let mut s_xtwx_diag = vec![0.0_f64; n_fixed];
            let mut s_wcv_diag = vec![0.0_f64; n_fixed];
            for j1 in 0..n_fixed {
                for j2 in 0..n_fixed {
                    let mut xtwx =
                        compensated_sum((0..n_data).map(|i| {
                            x_m[i + j1 * n_data] * final_w_data[i] * x_m[i + j2 * n_data]
                        }));
                    if j1 == j2 {
                        xtwx += PRIOR_PREC_BETA;
                    }
                    let wcv = compensated_sum(
                        (0..n_latent)
                            .map(|k| final_w_cross[k + j1 * n_latent] * v[k + j2 * n_latent]),
                    );
                    s_mat[j1 * n_fixed + j2] = xtwx - wcv;
                    if j1 == j2 {
                        s_xtwx_diag[j1] = xtwx;
                        s_wcv_diag[j1] = wcv;
                    }
                }
            }

            if let Some(diag_latent_cov) = diag_aug_inv.as_mut() {
                let (schur_inv, schur_inv_jitter) = dense_spd_inverse_with_schur_stabilization(
                    &s_mat,
                    n_fixed,
                    &s_xtwx_diag,
                    &s_wcv_diag,
                )
                .map_err(|_| {
                    crate::error::InlaError::NotPositiveDefiniteContext(
                        "Schur S_mat final inverse failed".to_string(),
                    )
                })?;

                if let Some(jitter) = schur_inv_jitter {
                    for j in 0..n_fixed {
                        s_mat[j * n_fixed + j] += jitter;
                    }
                }

                // Joint fixed/random Gaussian covariance:
                // Var(x | y, theta) = Q_aug^{-1} + Q_aug^{-1} C S^{-1} C^T Q_aug^{-1}
                // with C = A^T W X and S the fixed-effects Schur complement.
                // Keeping only diag(Q_aug^{-1}) makes latent SDs too tight whenever
                // fixed effects are present and coupled to the latent field.
                for k in 0..n_latent {
                    let mut schur_var = 0.0_f64;
                    for j1 in 0..n_fixed {
                        let vk_j1 = v[k + j1 * n_latent];
                        for j2 in 0..n_fixed {
                            schur_var +=
                                vk_j1 * schur_inv[j1 * n_fixed + j2] * v[k + j2 * n_latent];
                        }
                    }
                    diag_latent_cov[k] = (diag_latent_cov[k] + schur_var).max(1e-12);
                }
            }

            let mut dummy_b = vec![0.0_f64; n_fixed];
            let schur_log_det = try_dense_cholesky_with_schur_stabilization(
                &mut s_mat,
                &mut dummy_b,
                n_fixed,
                &s_xtwx_diag,
                &s_wcv_diag,
            )
            .map(|(log_det, _)| log_det)
            .map_err(|_| {
                crate::error::InlaError::NotPositiveDefiniteContext(
                    "Schur S_mat final singular".to_string(),
                )
            })?;

            Ok((beta, x, log_det_aug, diag_aug_inv, schur_log_det))
        })();
        self.diagnostics.latent_mode_solve_time += solve_started.elapsed();
        result
    }

    pub fn find_mode_with_fixed_effects(
        &mut self,
        model: &crate::inference::InlaModel<'_>,
        theta: &[f64],
        x_init: &[f64],
        beta_init: &[f64],
        max_iter: usize,
        tol: f64,
    ) -> Result<FixedEffectsModeResult, InlaError> {
        let (beta, x, log_det_aug, diag_aug_inv, schur_log_det) = self
            .find_mode_with_fixed_effects_core(
                model,
                theta,
                x_init,
                beta_init,
                ModeControl {
                    max_iter,
                    tol,
                    need_diag_aug_inv: true,
                },
            )?;
        Ok((
            beta,
            x,
            log_det_aug,
            diag_aug_inv.unwrap_or_default(),
            schur_log_det,
        ))
    }

    pub fn find_mode_with_fixed_effects_logdet(
        &mut self,
        model: &crate::inference::InlaModel<'_>,
        theta: &[f64],
        x_init: &[f64],
        beta_init: &[f64],
        max_iter: usize,
        tol: f64,
    ) -> Result<(Vec<f64>, Vec<f64>, f64, f64), InlaError> {
        let (beta, x, log_det_aug, _, schur_log_det) = self.find_mode_with_fixed_effects_core(
            model,
            theta,
            x_init,
            beta_init,
            ModeControl {
                max_iter,
                tol,
                need_diag_aug_inv: false,
            },
        )?;
        Ok((beta, x, log_det_aug, schur_log_det))
    }

    pub fn find_mode_with_logdet(
        &mut self,
        model: &crate::inference::InlaModel<'_>,
        theta: &[f64],
        max_iter: usize,
        tol: f64,
    ) -> Result<(Vec<f64>, f64), InlaError> {
        self.find_mode_with_logdet_and_warm(model, theta, &[], max_iter, tol)
    }

    pub fn find_mode(
        &mut self,
        model: &crate::inference::InlaModel<'_>,
        theta: &[f64],
        max_iter: usize,
        tol: f64,
    ) -> Result<Vec<f64>, InlaError> {
        let (x, _) = self.find_mode_with_logdet(model, theta, max_iter, tol)?;
        Ok(x)
    }

    pub fn solve(&self, rhs: &mut [f64]) {
        self.solver.solve_llt(rhs);
    }
    pub fn log_det(&self) -> f64 {
        self.log_det
    }
    pub fn n(&self) -> usize {
        self.graph.n()
    }

    pub fn quadratic_form_x(&self, qfunc: &dyn QFunc, theta_model: &[f64], x: &[f64]) -> f64 {
        let n = self.n();
        let diag: f64 = (0..n)
            .map(|i| qfunc.eval(i, i, theta_model) * x[i] * x[i])
            .sum();
        let offdiag: f64 = qfunc
            .graph()
            .iter_upper_triangle()
            .map(|(i, j)| 2.0 * qfunc.eval(i, j, theta_model) * x[i] * x[j])
            .sum();
        diag + offdiag
    }
}

// Native dense Cholesky implementation for tiny exact K x K matrices.
fn dense_cholesky_solve(a: &mut [f64], b: &mut [f64], n: usize) -> Result<f64, InlaError> {
    let mut log_det = 0.0;

    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i * n + j];
            for k in 0..j {
                sum -= a[i * n + k] * a[j * n + k];
            }
            if i == j {
                if sum <= 1e-12 {
                    return Err(InlaError::NotPositiveDefiniteContext(
                        "dense_cholesky_solve encountered a non-positive pivot".to_string(),
                    ));
                }
                let l_ii = sum.sqrt();
                a[i * n + j] = l_ii;
                log_det += 2.0 * l_ii.ln();
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
    Ok(log_det)
}

struct AugmentedQFunc<'a> {
    inner: &'a dyn QFunc,
    atwa: &'a AtwaStorage,
}

impl QFunc for AugmentedQFunc<'_> {
    fn graph(&self) -> &Graph {
        self.inner.graph()
    }
    fn eval(&self, i: usize, j: usize, theta: &[f64]) -> f64 {
        let mut base = self.inner.eval(i, j, theta);
        if i == j && !self.inner.is_proper() {
            base += 1e-6; // Intrinsic regularization
        }
        base + self.atwa.value(i, j)
    }
    fn n_hyperparams(&self) -> usize {
        self.inner.n_hyperparams()
    }
}

// Kriging solver: u_new = u_old - V_c (C V_c)^-1 C u_old
fn apply_kriging_correction(
    u: &mut [f64],
    v_c: &[f64],      // n_latent x n_constr
    c_vc_inv: &[f64], // n_constr x n_constr
    constr: &[f64],   // n_constr x n_latent
    n_latent: usize,
    n_constr: usize,
) {
    let mut cu = vec![0.0_f64; n_constr];
    for c in 0..n_constr {
        for i in 0..n_latent {
            cu[c] += constr[c * n_latent + i] * u[i];
        }
    }
    let mut lambda = vec![0.0_f64; n_constr];
    for c1 in 0..n_constr {
        for c2 in 0..n_constr {
            lambda[c1] += c_vc_inv[c1 * n_constr + c2] * cu[c2];
        }
    }
    for i in 0..n_latent {
        for c in 0..n_constr {
            u[i] -= v_c[c * n_latent + i] * lambda[c];
        }
    }
}
