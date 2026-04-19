#![allow(non_snake_case)]
// The R package and extendr entrypoint intentionally keep the public
// `rustyINLA` name so the compiled module matches the package-facing symbol
// expected by the current R integration.

use extendr_api::prelude::*;
use inla_core::inference::{InlaEngine, InlaModel, InlaParams};
use inla_core::likelihood::{
    GammaLikelihood, GaussianLikelihood, LogLikelihood, PoissonLikelihood, TweedieLikelihood,
    ZipLikelihood,
};
use inla_core::models::{Ar1Model, CompoundQFunc, IidModel, QFunc, Rw1Model};
use std::collections::HashMap;

type BridgeResult<T> = std::result::Result<T, String>;

struct LatentBlockSpec {
    model_type: String,
    n_levels: usize,
    start: usize,
}

struct BackendSpec {
    y: Vec<f64>,
    likelihood_type: String,
    fixed_matrix: Option<Vec<f64>>,
    n_fixed: usize,
    n_latent: usize,
    a_i: Option<Vec<usize>>,
    a_j: Option<Vec<usize>>,
    a_x: Option<Vec<f64>>,
    offset: Option<Vec<f64>>,
    extr_constr: Option<Vec<f64>>,
    n_constr: usize,
    latent_blocks: Vec<LatentBlockSpec>,
    theta_init: Option<Vec<f64>>,
    latent_init: Option<Vec<f64>>,
    fixed_init: Option<Vec<f64>>,
    optimizer_max_evals: Option<usize>,
    skip_ccd: Option<bool>,
}

fn list_to_map(list: &List) -> BridgeResult<HashMap<String, Robj>> {
    HashMap::<String, Robj>::try_from(list)
        .map_err(|err| format!("Invalid named list in backend spec: {err:?}"))
}

fn get_required_field<'a>(spec: &'a HashMap<String, Robj>, field: &str) -> BridgeResult<&'a Robj> {
    spec.get(field)
        .ok_or_else(|| format!("Missing backend spec field '{field}'"))
}

fn parse_required_string(obj: &Robj, field: &str) -> BridgeResult<String> {
    obj.as_str()
        .map(|value| value.to_string())
        .ok_or_else(|| format!("Backend spec field '{field}' must be a string"))
}

fn parse_required_usize(obj: &Robj, field: &str) -> BridgeResult<usize> {
    obj.as_integer()
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| format!("Backend spec field '{field}' must be a non-negative integer"))
}

fn parse_required_real_vec(obj: &Robj, field: &str) -> BridgeResult<Vec<f64>> {
    obj.as_real_slice()
        .map(|slice| slice.to_vec())
        .ok_or_else(|| format!("Backend spec field '{field}' must be a numeric vector"))
}

fn parse_optional_real_vec(obj: &Robj, field: &str) -> BridgeResult<Option<Vec<f64>>> {
    if obj.is_null() {
        Ok(None)
    } else {
        parse_required_real_vec(obj, field).map(Some)
    }
}

fn parse_optional_usize_vec(obj: &Robj, field: &str) -> BridgeResult<Option<Vec<usize>>> {
    if obj.is_null() {
        Ok(None)
    } else {
        obj.as_integer_slice()
            .map(|slice| {
                slice
                    .iter()
                    .map(|&value| usize::try_from(value).unwrap_or(usize::MAX))
                    .collect()
            })
            .ok_or_else(|| format!("Backend spec field '{field}' must be an integer vector"))
            .and_then(|values: Vec<usize>| {
                if values.contains(&usize::MAX) {
                    Err(format!(
                        "Backend spec field '{field}' contains negative indices"
                    ))
                } else {
                    Ok(Some(values))
                }
            })
    }
}

fn parse_optional_usize(obj: &Robj, field: &str) -> BridgeResult<Option<usize>> {
    if obj.is_null() {
        Ok(None)
    } else {
        parse_required_usize(obj, field).map(Some)
    }
}

fn parse_optional_bool(obj: &Robj, field: &str) -> BridgeResult<Option<bool>> {
    if obj.is_null() {
        Ok(None)
    } else {
        obj.as_bool()
            .map(Some)
            .ok_or_else(|| format!("Backend spec field '{field}' must be TRUE/FALSE"))
    }
}

fn parse_latent_blocks(obj: &Robj) -> BridgeResult<Vec<LatentBlockSpec>> {
    if obj.is_null() {
        return Ok(vec![]);
    }

    let blocks = obj
        .as_list()
        .ok_or_else(|| "Backend spec field 'latent_blocks' must be a list".to_string())?;

    blocks
        .values()
        .enumerate()
        .map(|(idx, block_obj)| {
            let block_list = block_obj.as_list().ok_or_else(|| {
                format!(
                    "Backend spec latent_blocks[[{}]] must be a named list",
                    idx + 1
                )
            })?;
            let block_map = list_to_map(&block_list)?;
            Ok(LatentBlockSpec {
                model_type: parse_required_string(
                    get_required_field(&block_map, "model")?,
                    "latent_blocks$model",
                )?,
                n_levels: parse_required_usize(
                    get_required_field(&block_map, "n_levels")?,
                    "latent_blocks$n_levels",
                )?,
                start: parse_required_usize(
                    get_required_field(&block_map, "start")?,
                    "latent_blocks$start",
                )?,
            })
        })
        .collect()
}

fn build_single_qfunc(model_type: &str, n_levels: usize) -> BridgeResult<Box<dyn QFunc>> {
    match model_type {
        "iid" => Ok(Box::new(IidModel::new(n_levels))),
        "rw1" => Ok(Box::new(Rw1Model::new(n_levels))),
        "ar1" => Ok(Box::new(Ar1Model::new(n_levels))),
        _ => Err(format!("Unknown model_type: {model_type}")),
    }
}

fn build_qfunc(latent_blocks: &[LatentBlockSpec]) -> BridgeResult<Box<dyn QFunc>> {
    if latent_blocks.len() == 1 {
        let block = &latent_blocks[0];
        return build_single_qfunc(&block.model_type, block.n_levels);
    }

    let mut blocks = Vec::with_capacity(latent_blocks.len());
    for block in latent_blocks {
        blocks.push((
            block.start,
            build_single_qfunc(&block.model_type, block.n_levels)?,
        ));
    }
    Ok(Box::new(CompoundQFunc::new(blocks)))
}

fn default_model_theta_init(latent_blocks: &[LatentBlockSpec]) -> BridgeResult<Vec<f64>> {
    let mut theta_init = Vec::new();
    for block in latent_blocks {
        match block.model_type.as_str() {
            "iid" | "rw1" => theta_init.push(4.0),
            "ar1" => {
                theta_init.push(4.0);
                theta_init.push(2.0);
            }
            _ => {
                return Err(format!(
                    "No default theta initial values configured for latent model '{}'",
                    block.model_type
                ))
            }
        }
    }
    Ok(theta_init)
}

fn default_likelihood_theta_init(likelihood_type: &str) -> BridgeResult<Vec<f64>> {
    match likelihood_type {
        "gaussian" => Ok(vec![4.0]),
        "poisson" => Ok(vec![]),
        "gamma" => Ok(vec![4.605_170_185_988_09]),
        "zeroinflatedpoisson1" => Ok(vec![-1.0]),
        "tweedie" => Ok(vec![0.0, -4.0]),
        _ => Err(format!(
            "No default theta initial values configured for likelihood '{}'",
            likelihood_type
        )),
    }
}

fn validate_backend_spec(spec: &BackendSpec) -> BridgeResult<()> {
    let n_data = spec.y.len();

    if let Some(fixed_matrix) = &spec.fixed_matrix {
        let expected = n_data * spec.n_fixed;
        if fixed_matrix.len() != expected {
            return Err(format!(
                "fixed_matrix length {} does not match nrow(data) * n_fixed = {}",
                fixed_matrix.len(),
                expected
            ));
        }
    } else if spec.n_fixed > 0 {
        return Err("n_fixed > 0 requires a fixed_matrix".to_string());
    }

    match (&spec.a_i, &spec.a_j, &spec.a_x) {
        (None, None, None) => {}
        (Some(a_i), Some(a_j), Some(a_x)) => {
            if a_i.len() != a_j.len() || a_i.len() != a_x.len() {
                return Err(
                    "A matrix triplets must have matching lengths for a_i, a_j and a_x".to_string(),
                );
            }
            for &row in a_i {
                if row >= n_data {
                    return Err(format!(
                        "A matrix row index {} is out of range for {} observations",
                        row, n_data
                    ));
                }
            }
            for &col in a_j {
                if col >= spec.n_latent {
                    return Err(format!(
                        "A matrix column index {} is out of range for {} latent nodes",
                        col, spec.n_latent
                    ));
                }
            }
        }
        _ => {
            return Err(
                "A matrix triplets must provide a_i, a_j and a_x together or leave all three NULL"
                    .to_string(),
            )
        }
    }

    if let Some(offset) = &spec.offset {
        if offset.len() != n_data {
            return Err(format!(
                "offset length {} does not match data length {}",
                offset.len(),
                n_data
            ));
        }
    }

    if let Some(extr_constr) = &spec.extr_constr {
        let expected = spec.n_constr * spec.n_latent;
        if extr_constr.len() != expected {
            return Err(format!(
                "extr_constr length {} does not match n_constr * n_latent = {}",
                extr_constr.len(),
                expected
            ));
        }
    } else if spec.n_constr > 0 {
        return Err("n_constr > 0 requires extr_constr".to_string());
    }

    if spec.latent_blocks.is_empty() {
        return Err(
            "At least one latent block from f(...) is required by the current Rust bridge"
                .to_string(),
        );
    }

    let mut expected_start = 0usize;
    let mut total_levels = 0usize;
    for block in &spec.latent_blocks {
        if block.start != expected_start {
            return Err(
                "latent_blocks must be contiguous and ordered by their start positions".to_string(),
            );
        }
        expected_start += block.n_levels;
        total_levels += block.n_levels;
    }

    if total_levels != spec.n_latent {
        return Err(format!(
            "Sum of latent block sizes {} does not match n_latent {}",
            total_levels, spec.n_latent
        ));
    }

    if let Some(theta_init) = &spec.theta_init {
        let expected = spec
            .latent_blocks
            .iter()
            .map(|block| match block.model_type.as_str() {
                "iid" | "rw1" => 1usize,
                "ar1" => 2usize,
                _ => 0usize,
            })
            .sum::<usize>()
            + match spec.likelihood_type.as_str() {
                "gaussian" => 1usize,
                "poisson" => 0usize,
                "gamma" => 1usize,
                "zeroinflatedpoisson1" => 1usize,
                "tweedie" => 2usize,
                _ => 0usize,
            };
        if theta_init.len() != expected {
            return Err(format!(
                "theta_init length {} does not match expected hyperparameter length {}",
                theta_init.len(),
                expected
            ));
        }
    }

    if let Some(latent_init) = &spec.latent_init {
        if latent_init.len() != spec.n_latent {
            return Err(format!(
                "latent_init length {} does not match n_latent {}",
                latent_init.len(),
                spec.n_latent
            ));
        }
    }

    if let Some(fixed_init) = &spec.fixed_init {
        if fixed_init.len() != spec.n_fixed {
            return Err(format!(
                "fixed_init length {} does not match n_fixed {}",
                fixed_init.len(),
                spec.n_fixed
            ));
        }
    }

    Ok(())
}

fn parse_backend_spec(spec_arg: Robj) -> BridgeResult<BackendSpec> {
    let spec_list = spec_arg
        .as_list()
        .ok_or_else(|| "rust_inla_run expects a named backend spec list".to_string())?;
    let spec_map = list_to_map(&spec_list)?;

    let spec = BackendSpec {
        y: parse_required_real_vec(get_required_field(&spec_map, "y")?, "y")?,
        likelihood_type: parse_required_string(
            get_required_field(&spec_map, "likelihood")?,
            "likelihood",
        )?,
        fixed_matrix: parse_optional_real_vec(
            get_required_field(&spec_map, "fixed_matrix")?,
            "fixed_matrix",
        )?,
        n_fixed: parse_required_usize(get_required_field(&spec_map, "n_fixed")?, "n_fixed")?,
        n_latent: parse_required_usize(get_required_field(&spec_map, "n_latent")?, "n_latent")?,
        a_i: parse_optional_usize_vec(get_required_field(&spec_map, "a_i")?, "a_i")?,
        a_j: parse_optional_usize_vec(get_required_field(&spec_map, "a_j")?, "a_j")?,
        a_x: parse_optional_real_vec(get_required_field(&spec_map, "a_x")?, "a_x")?,
        offset: parse_optional_real_vec(get_required_field(&spec_map, "offset")?, "offset")?,
        extr_constr: parse_optional_real_vec(
            get_required_field(&spec_map, "extr_constr")?,
            "extr_constr",
        )?,
        n_constr: parse_required_usize(get_required_field(&spec_map, "n_constr")?, "n_constr")?,
        latent_blocks: parse_latent_blocks(get_required_field(&spec_map, "latent_blocks")?)?,
        theta_init: spec_map
            .get("theta_init")
            .map(|obj| parse_optional_real_vec(obj, "theta_init"))
            .transpose()?
            .flatten(),
        latent_init: spec_map
            .get("latent_init")
            .map(|obj| parse_optional_real_vec(obj, "latent_init"))
            .transpose()?
            .flatten(),
        fixed_init: spec_map
            .get("fixed_init")
            .map(|obj| parse_optional_real_vec(obj, "fixed_init"))
            .transpose()?
            .flatten(),
        optimizer_max_evals: spec_map
            .get("optimizer_max_evals")
            .map(|obj| parse_optional_usize(obj, "optimizer_max_evals"))
            .transpose()?
            .flatten(),
        skip_ccd: spec_map
            .get("skip_ccd")
            .map(|obj| parse_optional_bool(obj, "skip_ccd"))
            .transpose()?
            .flatten(),
    };

    validate_backend_spec(&spec)?;
    Ok(spec)
}

/// Execute rust-inla backend from R.
///
/// @param spec A named backend specification list built on the R side.
/// @export
#[extendr]
fn rust_inla_run(spec_arg: Robj) -> Robj {
    let spec = match parse_backend_spec(spec_arg) {
        Ok(spec) => spec,
        Err(err) => return r!(format!("Error: {err}")),
    };

    let qfunc = match build_qfunc(&spec.latent_blocks) {
        Ok(qfunc) => qfunc,
        Err(err) => return r!(format!("Error: {err}")),
    };

    let lik: Box<dyn LogLikelihood> = match spec.likelihood_type.as_str() {
        "gaussian" => Box::new(GaussianLikelihood),
        "poisson" => Box::new(PoissonLikelihood),
        "gamma" => Box::new(GammaLikelihood),
        "zeroinflatedpoisson1" => Box::new(ZipLikelihood),
        "tweedie" => Box::new(TweedieLikelihood),
        _ => return r!(format!("Unknown likelihood_type: {}", spec.likelihood_type)),
    };

    let mut theta_init = match default_model_theta_init(&spec.latent_blocks) {
        Ok(theta_init) => theta_init,
        Err(err) => return r!(format!("Error: {err}")),
    };
    let theta_lik_init = match default_likelihood_theta_init(&spec.likelihood_type) {
        Ok(theta_init) => theta_init,
        Err(err) => return r!(format!("Error: {err}")),
    };
    theta_init.extend(theta_lik_init);

    if theta_init.len() != qfunc.n_hyperparams() + lik.n_hyperparams() {
        return r!(format!(
            "Error: theta_init length {} does not match model+likelihood hyperparameters {}",
            theta_init.len(),
            qfunc.n_hyperparams() + lik.n_hyperparams()
        ));
    }

    if let Some(theta_init_override) = spec.theta_init.clone() {
        theta_init = theta_init_override;
    }

    let model = InlaModel {
        qfunc: qfunc.as_ref(),
        likelihood: lik.as_ref(),
        y: &spec.y,
        theta_init,
        latent_init: spec.latent_init.clone().unwrap_or_default(),
        fixed_init: spec.fixed_init.clone().unwrap_or_default(),
        fixed_matrix: spec.fixed_matrix.as_deref(),
        n_fixed: spec.n_fixed,
        n_latent: spec.n_latent,
        a_i: spec.a_i.as_deref(),
        a_j: spec.a_j.as_deref(),
        a_x: spec.a_x.as_deref(),
        offset: spec.offset.as_deref(),
        extr_constr: spec.extr_constr.as_deref(),
        n_constr: spec.n_constr,
    };

    let mut params = InlaParams::default();
    if let Some(max_evals) = spec.optimizer_max_evals {
        params.optimizer.max_evals = max_evals;
    }
    if let Some(skip_ccd) = spec.skip_ccd {
        params.skip_ccd = skip_ccd;
    }

    match InlaEngine::run(&model, &params) {
        Ok(res) => {
            // Build return list to R
            // Unpack random marginals (just mean and var for now)
            let mut marg_means = Vec::with_capacity(spec.n_latent);
            let mut marg_vars = Vec::with_capacity(spec.n_latent);

            let mut fitted_mean = Vec::with_capacity(spec.y.len());
            let mut fitted_q025 = Vec::with_capacity(spec.y.len());
            let mut fitted_q500 = Vec::with_capacity(spec.y.len());
            let mut fitted_q975 = Vec::with_capacity(spec.y.len());
            let mut fitted_mode = Vec::with_capacity(spec.y.len());
            let mut eta_mean = Vec::with_capacity(spec.y.len());
            let mut eta_var = Vec::with_capacity(spec.y.len());
            let mut eta_q025 = Vec::with_capacity(spec.y.len());
            let mut eta_q500 = Vec::with_capacity(spec.y.len());
            let mut eta_q975 = Vec::with_capacity(spec.y.len());

            // Because the inverse link (exp/logit) is monotonically increasing,
            // quantiles pass through exactly mapping Quantile(eta) -> Quantile(mu)
            let link_inv = |eta: f64| lik.link().inverse(eta);

            for m in &res.random {
                marg_means.push(m.mean());
                marg_vars.push(m.variance());
            }

            for m in &res.fitted {
                eta_mean.push(m.mean());
                eta_var.push(m.variance());
                eta_q025.push(m.quantile(0.025));
                eta_q500.push(m.quantile(0.500));
                eta_q975.push(m.quantile(0.975));
                fitted_mean.push(m.emarginal(link_inv));
                fitted_q025.push(link_inv(m.quantile(0.025)));
                fitted_q500.push(link_inv(m.quantile(0.500)));
                fitted_q975.push(link_inv(m.quantile(0.975)));

                // Provide a safe response-scale peak approximation
                let mode = match lik.link() {
                    inla_core::likelihood::LinkFunction::Log => (m.mean() - m.variance()).exp(),
                    _ => m.quantile(0.50),
                };
                fitted_mode.push(mode);
            }

            list!(
                log_mlik = res.log_mlik,
                log_mlik_theta_opt = res.log_mlik_theta_opt,
                log_mlik_theta_laplace = res.log_mlik_theta_laplace,
                theta_laplace_correction = res.theta_laplace_correction,
                theta_opt = res.theta_opt,
                theta_init_used = model.theta_init.clone(),
                n_evals = res.n_evals,
                fixed_means = res.fixed_means,
                fixed_sds = res.fixed_sds,
                marg_means = marg_means,
                marg_vars = marg_vars,
                // Predictions mapped to the Response (μ) Scale natively!
                fitted_mean = fitted_mean,
                fitted_mode = fitted_mode,
                fitted_q025 = fitted_q025,
                fitted_q500 = fitted_q500,
                fitted_q975 = fitted_q975,
                eta_mean = eta_mean,
                eta_var = eta_var,
                eta_q025 = eta_q025,
                eta_q500 = eta_q500,
                eta_q975 = eta_q975,
                ccd_thetas = res.ccd_thetas,
                ccd_base_weights = res.ccd_base_weights,
                ccd_weights = res.ccd_weights,
                ccd_log_mlik = res.ccd_log_mlik,
                ccd_log_weight = res.ccd_log_weight,
                ccd_hessian_eigenvalues = res.ccd_hessian_eigenvalues,
                prior_W = res.w_opt,
                prior_mean = res.posterior_mean,
                latent_var_theta_opt = res.latent_var_theta_opt,
                latent_var_within_theta = res.latent_var_within_theta,
                latent_var_between_theta = res.latent_var_between_theta,
                mode_x = res.mode_x,
                mode_beta = res.mode_beta,
                mode_eta = res.mode_eta,
                mode_grad = res.mode_grad,
                mode_curvature_raw = res.mode_curvature_raw,
                mode_curvature = res.mode_curvature,
                laplace_terms = list!(
                    sum_loglik = res.laplace_terms.sum_loglik,
                    log_prior_model = res.laplace_terms.log_prior_model,
                    log_prior_likelihood = res.laplace_terms.log_prior_likelihood,
                    log_prior = res.laplace_terms.log_prior,
                    latent_log_det_q = res.laplace_terms.latent_log_det_q,
                    latent_log_det_aug = res.laplace_terms.latent_log_det_aug,
                    fixed_log_det_penalty = res.laplace_terms.fixed_log_det_penalty,
                    schur_complement_adjustment = res.laplace_terms.schur_complement_adjustment,
                    final_log_det_q = res.laplace_terms.final_log_det_q,
                    final_log_det_aug = res.laplace_terms.final_log_det_aug,
                    latent_q_form = res.laplace_terms.latent_q_form,
                    fixed_q_form = res.laplace_terms.fixed_q_form,
                    final_q_form = res.laplace_terms.final_q_form,
                    log_mlik = res.laplace_terms.log_mlik,
                    neg_log_mlik = res.laplace_terms.neg_log_mlik
                ),
                diagnostics = list!(
                    optimizer_outer_iterations = res.diagnostics.optimizer_outer_iterations,
                    line_search_trial_evals = res.diagnostics.line_search_trial_evals,
                    line_search_trial_accepts = res.diagnostics.line_search_trial_accepts,
                    coordinate_probe_calls = res.diagnostics.coordinate_probe_calls,
                    coordinate_probe_evals = res.diagnostics.coordinate_probe_evals,
                    coordinate_probe_accepts = res.diagnostics.coordinate_probe_accepts,
                    laplace_eval_calls_total = res.diagnostics.laplace_eval_calls_total,
                    laplace_eval_calls_optimizer = res.diagnostics.laplace_eval_calls_optimizer,
                    laplace_eval_calls_ccd = res.diagnostics.laplace_eval_calls_ccd,
                    latent_mode_solve_calls = res.diagnostics.latent_mode_solve_calls,
                    latent_mode_iterations_total = res.diagnostics.latent_mode_iterations_total,
                    latent_mode_max_iter_hits = res.diagnostics.latent_mode_max_iter_hits,
                    factorization_count = res.diagnostics.factorization_count,
                    selected_inverse_count = res.diagnostics.selected_inverse_count,
                    optimizer_time_sec = res.diagnostics.optimizer_time_sec,
                    ccd_time_sec = res.diagnostics.ccd_time_sec,
                    latent_mode_solve_time_sec = res.diagnostics.latent_mode_solve_time_sec,
                    likelihood_assembly_time_sec = res.diagnostics.likelihood_assembly_time_sec,
                    sparse_factorization_time_sec = res.diagnostics.sparse_factorization_time_sec,
                    selected_inverse_time_sec = res.diagnostics.selected_inverse_time_sec
                ),
            )
            .into_robj()
        }
        Err(e) => r!(format!("Engine Error: {:?}", e)),
    }
}

// Macro to initialize the extendr module.
// Note: name matches the package/module name expected by R
extendr_module! {
    mod rustyINLA;
    fn rust_inla_run;
}
