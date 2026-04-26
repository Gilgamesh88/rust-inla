use std::time::Duration;

#[derive(Clone, Copy, Debug)]
pub enum LaplacePhase {
    Optimize,
    Ccd,
}

#[derive(Clone, Debug, Default)]
pub struct SolverDiagnostics {
    pub factorization_count: usize,
    pub factorization_time: Duration,
    pub selected_inverse_count: usize,
    pub selected_inverse_time: Duration,
}

#[derive(Clone, Debug, Default)]
pub struct RunDiagnostics {
    pub optimizer_outer_iterations: usize,
    pub line_search_trial_evals: usize,
    pub line_search_trial_accepts: usize,
    pub coordinate_probe_calls: usize,
    pub coordinate_probe_evals: usize,
    pub coordinate_probe_accepts: usize,
    pub laplace_eval_calls_total: usize,
    pub laplace_eval_calls_optimizer: usize,
    pub laplace_eval_calls_ccd: usize,
    pub latent_mode_solve_calls: usize,
    pub latent_mode_iterations_total: usize,
    pub latent_mode_max_iter_hits: usize,
    pub latent_mode_restarts: usize,
    pub latent_mode_step_ramp_solves: usize,
    pub latent_mode_step_factor_min: f64,
    pub latent_mode_solve_time: Duration,
    pub likelihood_assembly_time: Duration,
    pub optimizer_time: Duration,
    pub ccd_time: Duration,
}

impl RunDiagnostics {
    pub fn record_laplace_eval(&mut self, phase: LaplacePhase) {
        self.laplace_eval_calls_total += 1;
        match phase {
            LaplacePhase::Optimize => self.laplace_eval_calls_optimizer += 1,
            LaplacePhase::Ccd => self.laplace_eval_calls_ccd += 1,
        }
    }

    pub fn record_mode_step_factor(&mut self, step_factor: f64) {
        if !step_factor.is_finite() {
            return;
        }
        if self.latent_mode_step_factor_min == 0.0 {
            self.latent_mode_step_factor_min = step_factor;
        } else {
            self.latent_mode_step_factor_min = self.latent_mode_step_factor_min.min(step_factor);
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct RunDiagnosticsSummary {
    pub optimizer_outer_iterations: usize,
    pub line_search_trial_evals: usize,
    pub line_search_trial_accepts: usize,
    pub coordinate_probe_calls: usize,
    pub coordinate_probe_evals: usize,
    pub coordinate_probe_accepts: usize,
    pub laplace_eval_calls_total: usize,
    pub laplace_eval_calls_optimizer: usize,
    pub laplace_eval_calls_ccd: usize,
    pub latent_mode_solve_calls: usize,
    pub latent_mode_iterations_total: usize,
    pub latent_mode_max_iter_hits: usize,
    pub latent_mode_restarts: usize,
    pub latent_mode_step_ramp_solves: usize,
    pub latent_mode_step_factor_min: f64,
    pub factorization_count: usize,
    pub selected_inverse_count: usize,
    pub optimizer_time_sec: f64,
    pub ccd_time_sec: f64,
    pub latent_mode_solve_time_sec: f64,
    pub likelihood_assembly_time_sec: f64,
    pub sparse_factorization_time_sec: f64,
    pub selected_inverse_time_sec: f64,
}

impl RunDiagnosticsSummary {
    pub fn from_parts(run: &RunDiagnostics, solver: &SolverDiagnostics) -> Self {
        Self {
            optimizer_outer_iterations: run.optimizer_outer_iterations,
            line_search_trial_evals: run.line_search_trial_evals,
            line_search_trial_accepts: run.line_search_trial_accepts,
            coordinate_probe_calls: run.coordinate_probe_calls,
            coordinate_probe_evals: run.coordinate_probe_evals,
            coordinate_probe_accepts: run.coordinate_probe_accepts,
            laplace_eval_calls_total: run.laplace_eval_calls_total,
            laplace_eval_calls_optimizer: run.laplace_eval_calls_optimizer,
            laplace_eval_calls_ccd: run.laplace_eval_calls_ccd,
            latent_mode_solve_calls: run.latent_mode_solve_calls,
            latent_mode_iterations_total: run.latent_mode_iterations_total,
            latent_mode_max_iter_hits: run.latent_mode_max_iter_hits,
            latent_mode_restarts: run.latent_mode_restarts,
            latent_mode_step_ramp_solves: run.latent_mode_step_ramp_solves,
            latent_mode_step_factor_min: if run.latent_mode_step_factor_min > 0.0 {
                run.latent_mode_step_factor_min
            } else {
                1.0
            },
            factorization_count: solver.factorization_count,
            selected_inverse_count: solver.selected_inverse_count,
            optimizer_time_sec: run.optimizer_time.as_secs_f64(),
            ccd_time_sec: run.ccd_time.as_secs_f64(),
            latent_mode_solve_time_sec: run.latent_mode_solve_time.as_secs_f64(),
            likelihood_assembly_time_sec: run.likelihood_assembly_time.as_secs_f64(),
            sparse_factorization_time_sec: solver.factorization_time.as_secs_f64(),
            selected_inverse_time_sec: solver.selected_inverse_time.as_secs_f64(),
        }
    }
}
