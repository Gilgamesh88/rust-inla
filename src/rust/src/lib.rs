#![allow(non_snake_case)]
// The R package and extendr entrypoint intentionally keep the public
// `rustyINLA` name so the compiled module matches the package-facing symbol
// expected by the current R integration.

use extendr_api::prelude::*;
use inla_core::inference::{InlaEngine, InlaModel, InlaParams, ThetaStateEvidence};
use inla_core::likelihood::{
    GammaLikelihood, GaussianLikelihood, LogLikelihood, PoissonLikelihood, TweedieLikelihood,
    ZipLikelihood,
};
use inla_core::models::{
    Ar1Model, Ar2Model, CompoundQFunc, FixedOnlyModel, IidModel, QFunc, Rw1Model, Rw2Model,
};
use std::collections::HashMap;

type BridgeResult<T> = std::result::Result<T, String>;

struct LatentBlockSpec {
    model_type: String,
    n_levels: usize,
    start: usize,
    structure_values: Option<Vec<f64>>,
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
    theta_prior_mean: Option<Vec<f64>>,
    theta_prior_precision: Option<Vec<f64>>,
    theta_prior_mask: Option<Vec<usize>>,
    fixed_state_precision: Option<Vec<f64>>,
    fixed_state_linear: Option<Vec<f64>>,
    latent_state_precision_diag: Option<Vec<f64>>,
    latent_state_linear: Option<Vec<f64>>,
    latent_state_precision_i: Option<Vec<usize>>,
    latent_state_precision_j: Option<Vec<usize>>,
    latent_state_precision_x: Option<Vec<f64>>,
    latent_fixed_state_precision: Option<Vec<f64>>,
    theta_state_n_support: Option<usize>,
    theta_state_support: Option<Vec<f64>>,
    theta_state_fixed_precision: Option<Vec<f64>>,
    theta_state_fixed_linear: Option<Vec<f64>>,
    theta_state_latent_precision_diag: Option<Vec<f64>>,
    theta_state_latent_linear: Option<Vec<f64>>,
    theta_state_latent_precision_i: Option<Vec<usize>>,
    theta_state_latent_precision_j: Option<Vec<usize>>,
    theta_state_latent_precision_x: Option<Vec<f64>>,
    theta_state_latent_fixed_precision: Option<Vec<f64>>,
    theta_state_log_constant: Option<Vec<f64>>,
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
                structure_values: block_map
                    .get("structure_values")
                    .map(|obj| parse_optional_real_vec(obj, "latent_blocks$structure_values"))
                    .transpose()?
                    .flatten(),
            })
        })
        .collect()
}

fn iid_log_precision_prior_override(spec: &BackendSpec) -> Option<(f64, f64)> {
    let mean = spec.theta_prior_mean.as_ref()?;
    let precision = spec.theta_prior_precision.as_ref()?;
    let mask = spec.theta_prior_mask.as_ref()?;
    if mask.first().copied() == Some(1usize) {
        Some((mean[0], precision[0]))
    } else {
        None
    }
}

fn build_single_qfunc(
    block: &LatentBlockSpec,
    iid_prior_override: Option<(f64, f64)>,
) -> BridgeResult<Box<dyn QFunc>> {
    match block.model_type.as_str() {
        "iid" => Ok(Box::new(match iid_prior_override {
            Some((mean, precision)) => {
                IidModel::new_with_log_precision_prior(block.n_levels, mean, precision)
            }
            None => IidModel::new(block.n_levels),
        })),
        "rw1" => Ok(Box::new(Rw1Model::new(block.n_levels))),
        "rw2" => match block.structure_values.as_deref() {
            Some(values) => {
                Rw2Model::new_with_values(values).map(|model| Box::new(model) as Box<dyn QFunc>)
            }
            None => Ok(Box::new(Rw2Model::new(block.n_levels))),
        },
        "ar1" => Ok(Box::new(Ar1Model::new(block.n_levels))),
        "ar2" => Ok(Box::new(Ar2Model::new(block.n_levels))),
        _ => Err(format!("Unknown model_type: {}", block.model_type)),
    }
}

fn build_qfunc(spec: &BackendSpec) -> BridgeResult<Box<dyn QFunc>> {
    let latent_blocks = &spec.latent_blocks;
    if latent_blocks.is_empty() {
        return Ok(Box::new(FixedOnlyModel::new()));
    }

    let iid_prior_override = iid_log_precision_prior_override(spec);
    if latent_blocks.len() == 1 {
        let block = &latent_blocks[0];
        return build_single_qfunc(block, iid_prior_override);
    }

    let mut blocks = Vec::with_capacity(latent_blocks.len());
    for block in latent_blocks {
        blocks.push((block.start, build_single_qfunc(block, None)?));
    }
    Ok(Box::new(CompoundQFunc::new(blocks)))
}

fn default_model_theta_init(latent_blocks: &[LatentBlockSpec]) -> BridgeResult<Vec<f64>> {
    let mut theta_init = Vec::new();
    for block in latent_blocks {
        match block.model_type.as_str() {
            "iid" | "rw1" | "rw2" => theta_init.push(4.0),
            "ar1" => {
                theta_init.push(4.0);
                theta_init.push(2.0);
            }
            "ar2" => {
                theta_init.push(4.0);
                theta_init.push(1.0);
                theta_init.push(0.0);
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

fn expected_theta_len(spec: &BackendSpec) -> usize {
    spec.latent_blocks
        .iter()
        .map(|block| match block.model_type.as_str() {
            "iid" | "rw1" | "rw2" => 1usize,
            "ar1" => 2usize,
            "ar2" => 3usize,
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
        }
}

fn validate_backend_spec(spec: &BackendSpec) -> BridgeResult<()> {
    let n_data = spec.y.len();

    if spec.y.iter().any(|value| value.is_infinite()) {
        return Err("y must not contain infinite values".to_string());
    }

    if let Some(fixed_matrix) = &spec.fixed_matrix {
        let expected = n_data * spec.n_fixed;
        if fixed_matrix.len() != expected {
            return Err(format!(
                "fixed_matrix length {} does not match nrow(data) * n_fixed = {}",
                fixed_matrix.len(),
                expected
            ));
        }
        if fixed_matrix.iter().any(|value| !value.is_finite()) {
            return Err("fixed_matrix must contain only finite values".to_string());
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
            if a_x.iter().any(|value| !value.is_finite()) {
                return Err("A matrix values a_x must contain only finite values".to_string());
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
        if offset.iter().any(|value| !value.is_finite()) {
            return Err("offset must contain only finite values".to_string());
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
        if extr_constr.iter().any(|value| !value.is_finite()) {
            return Err("extr_constr must contain only finite values".to_string());
        }
    } else if spec.n_constr > 0 {
        return Err("n_constr > 0 requires extr_constr".to_string());
    }

    if spec.n_fixed == 0 && spec.n_latent == 0 {
        return Err(
            "At least one fixed-effect column or latent f(...) block is required".to_string(),
        );
    }

    if spec.n_latent == 0 && spec.n_constr > 0 {
        return Err("extr_constr requires at least one latent node".to_string());
    }

    let mut expected_start = 0usize;
    let mut total_levels = 0usize;
    for block in &spec.latent_blocks {
        if block.start != expected_start {
            return Err(
                "latent_blocks must be contiguous and ordered by their start positions".to_string(),
            );
        }
        if block.model_type == "rw2" && block.n_levels < 3 {
            return Err("latent model 'rw2' requires at least 3 levels".to_string());
        }
        if let Some(structure_values) = &block.structure_values {
            if structure_values.len() != block.n_levels {
                return Err(format!(
                    "latent block '{}' structure_values length {} does not match n_levels {}",
                    block.model_type,
                    structure_values.len(),
                    block.n_levels
                ));
            }
            if structure_values.iter().any(|value| !value.is_finite()) {
                return Err(format!(
                    "latent block '{}' structure_values must be finite",
                    block.model_type
                ));
            }
            if structure_values.windows(2).any(|pair| pair[1] <= pair[0]) {
                return Err(format!(
                    "latent block '{}' structure_values must be strictly increasing",
                    block.model_type
                ));
            }
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

    let expected = expected_theta_len(spec);
    if let Some(theta_init) = &spec.theta_init {
        if theta_init.len() != expected {
            return Err(format!(
                "theta_init length {} does not match expected hyperparameter length {}",
                theta_init.len(),
                expected
            ));
        }
        if theta_init.iter().any(|value| !value.is_finite()) {
            return Err("theta_init must contain only finite values".to_string());
        }
    }

    match (
        &spec.theta_prior_mean,
        &spec.theta_prior_precision,
        &spec.theta_prior_mask,
    ) {
        (None, None, None) => {}
        (Some(mean), Some(precision), Some(mask)) => {
            if mean.len() != expected || precision.len() != expected || mask.len() != expected {
                return Err(format!(
                    "theta prior override length must match expected hyperparameter length {expected}"
                ));
            }
            if mean.iter().any(|value| !value.is_finite()) {
                return Err("theta_prior_mean must contain only finite values".to_string());
            }
            if precision.iter().any(|value| !value.is_finite() || *value < 0.0) {
                return Err(
                    "theta_prior_precision must contain finite non-negative values".to_string(),
                );
            }
            if mask.iter().any(|value| *value > 1usize) {
                return Err("theta_prior_mask entries must be 0 or 1".to_string());
            }

            let active: Vec<usize> = mask
                .iter()
                .enumerate()
                .filter_map(|(idx, value)| if *value == 1 { Some(idx) } else { None })
                .collect();
            if active != vec![0usize] {
                return Err(
                    "experimental theta prior override can replace only the first iid hyperparameter"
                        .to_string(),
                );
            }
            if precision[0] <= 0.0 {
                return Err(
                    "active theta_prior_precision entries must be strictly positive".to_string(),
                );
            }
            if spec.latent_blocks.len() != 1 || spec.latent_blocks[0].model_type != "iid" {
                return Err(
                    "experimental theta prior override currently supports exactly one iid latent block"
                        .to_string(),
                );
            }
        }
        _ => {
            return Err(
                "theta prior override requires theta_prior_mean, theta_prior_precision, and theta_prior_mask together"
                    .to_string(),
            )
        }
    }

    match (&spec.fixed_state_precision, &spec.fixed_state_linear) {
        (None, None) => {}
        (Some(precision), Some(linear)) => {
            let expected_precision = spec.n_fixed * spec.n_fixed;
            if precision.len() != expected_precision || linear.len() != spec.n_fixed {
                return Err(format!(
                    "fixed state evidence must have precision length {} and linear length {}",
                    expected_precision, spec.n_fixed
                ));
            }
            if precision.iter().any(|value| !value.is_finite())
                || linear.iter().any(|value| !value.is_finite())
            {
                return Err("fixed state evidence must contain only finite values".to_string());
            }
            for j in 0..spec.n_fixed {
                if precision[j * spec.n_fixed + j] < 0.0 {
                    return Err(
                        "fixed state evidence precision must have non-negative diagonal entries"
                            .to_string(),
                    );
                }
            }
        }
        _ => return Err(
            "fixed state evidence requires fixed_state_precision and fixed_state_linear together"
                .to_string(),
        ),
    }

    match (
        &spec.latent_state_precision_diag,
        &spec.latent_state_linear,
    ) {
        (None, None) => {}
        (Some(precision), Some(linear)) => {
            if precision.len() != spec.n_latent || linear.len() != spec.n_latent {
                return Err(format!(
                    "latent state evidence must have precision and linear lengths matching n_latent = {}",
                    spec.n_latent
                ));
            }
            if precision
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
                || linear.iter().any(|value| !value.is_finite())
            {
                return Err(
                    "latent state evidence must contain finite values and non-negative precision"
                        .to_string(),
                );
            }
            if spec.latent_blocks.is_empty()
                || spec
                    .latent_blocks
                    .iter()
                    .any(|block| block.model_type != "iid")
            {
                return Err(
                    "latent state evidence currently supports only iid latent blocks"
                        .to_string(),
                );
            }
        }
        _ => {
            return Err(
                "latent state evidence requires latent_state_precision_diag and latent_state_linear together"
                    .to_string(),
            )
        }
    }

    let latent_sparse_state_present = [
        spec.latent_state_precision_i.is_some(),
        spec.latent_state_precision_j.is_some(),
        spec.latent_state_precision_x.is_some(),
    ];
    if latent_sparse_state_present.iter().any(|present| *present) {
        if !latent_sparse_state_present.iter().all(|present| *present) {
            return Err(
                "latent sparse state evidence requires latent_state_precision_i, latent_state_precision_j, and latent_state_precision_x together"
                    .to_string(),
            );
        }
        if spec.latent_state_precision_diag.is_none() || spec.latent_state_linear.is_none() {
            return Err(
                "latent sparse state evidence requires latent diagonal state evidence".to_string(),
            );
        }
        if spec.latent_blocks.is_empty()
            || spec
                .latent_blocks
                .iter()
                .any(|block| block.model_type != "iid")
        {
            return Err(
                "latent sparse state evidence currently supports only iid latent blocks"
                    .to_string(),
            );
        }
        let rows = spec.latent_state_precision_i.as_ref().unwrap();
        let cols = spec.latent_state_precision_j.as_ref().unwrap();
        let values = spec.latent_state_precision_x.as_ref().unwrap();
        if rows.len() != cols.len() || rows.len() != values.len() {
            return Err(
                "latent sparse state evidence index and value vectors must have matching lengths"
                    .to_string(),
            );
        }
        for idx in 0..values.len() {
            if rows[idx] >= spec.n_latent || cols[idx] >= spec.n_latent || rows[idx] == cols[idx] {
                return Err(format!(
                    "latent sparse state evidence entry {} has invalid latent indices",
                    idx
                ));
            }
            if !values[idx].is_finite() {
                return Err(
                    "latent sparse state evidence precision must contain only finite values"
                        .to_string(),
                );
            }
        }
    }

    if let Some(precision) = &spec.latent_fixed_state_precision {
        let expected = spec.n_latent * spec.n_fixed;
        if precision.len() != expected {
            return Err(format!(
                "latent-fixed state evidence precision length {} does not match n_latent * n_fixed = {}",
                precision.len(),
                expected
            ));
        }
        if precision.iter().any(|value| !value.is_finite()) {
            return Err(
                "latent-fixed state evidence precision must contain only finite values".to_string(),
            );
        }
        if spec.fixed_state_precision.is_none()
            || spec.fixed_state_linear.is_none()
            || spec.latent_state_precision_diag.is_none()
            || spec.latent_state_linear.is_none()
        {
            return Err(
                "latent-fixed state evidence requires fixed and latent state evidence blocks"
                    .to_string(),
            );
        }
        if spec.n_fixed == 0 || spec.n_latent == 0 {
            return Err(
                "latent-fixed state evidence requires positive fixed and latent dimensions"
                    .to_string(),
            );
        }
    }

    let theta_state_present = [
        spec.theta_state_n_support.is_some(),
        spec.theta_state_support.is_some(),
        spec.theta_state_fixed_precision.is_some(),
        spec.theta_state_fixed_linear.is_some(),
        spec.theta_state_latent_precision_diag.is_some(),
        spec.theta_state_latent_linear.is_some(),
        spec.theta_state_latent_fixed_precision.is_some(),
        spec.theta_state_log_constant.is_some(),
    ];
    let theta_sparse_state_present = [
        spec.theta_state_latent_precision_i.is_some(),
        spec.theta_state_latent_precision_j.is_some(),
        spec.theta_state_latent_precision_x.is_some(),
    ];
    if theta_state_present.iter().any(|present| *present) {
        if !theta_state_present.iter().all(|present| *present) {
            return Err(
                "theta state evidence requires all theta_state_* fields together".to_string(),
            );
        }
        if theta_sparse_state_present.iter().any(|present| *present)
            && !theta_sparse_state_present.iter().all(|present| *present)
        {
            return Err(
                "theta sparse state evidence requires theta_state_latent_precision_i, theta_state_latent_precision_j, and theta_state_latent_precision_x together"
                    .to_string(),
            );
        }
        let n_support = spec.theta_state_n_support.unwrap_or(0);
        if n_support == 0 {
            return Err("theta state evidence requires at least one support point".to_string());
        }
        let n_theta = expected_theta_len(spec);
        if spec.latent_blocks.is_empty()
            || spec
                .latent_blocks
                .iter()
                .any(|block| block.model_type != "iid")
        {
            return Err(
                "theta-dependent state evidence currently supports only iid latent blocks"
                    .to_string(),
            );
        }

        let support = spec.theta_state_support.as_ref().unwrap();
        let fixed_precision = spec.theta_state_fixed_precision.as_ref().unwrap();
        let fixed_linear = spec.theta_state_fixed_linear.as_ref().unwrap();
        let latent_precision = spec.theta_state_latent_precision_diag.as_ref().unwrap();
        let latent_linear = spec.theta_state_latent_linear.as_ref().unwrap();
        let latent_fixed = spec.theta_state_latent_fixed_precision.as_ref().unwrap();
        let log_constant = spec.theta_state_log_constant.as_ref().unwrap();
        let theta_sparse_i = spec.theta_state_latent_precision_i.as_ref();
        let theta_sparse_j = spec.theta_state_latent_precision_j.as_ref();
        let theta_sparse_x = spec.theta_state_latent_precision_x.as_ref();

        let expected_support = n_support * n_theta;
        let expected_fixed_precision = n_support * spec.n_fixed * spec.n_fixed;
        let expected_fixed_linear = n_support * spec.n_fixed;
        let expected_latent = n_support * spec.n_latent;
        let expected_latent_fixed = n_support * spec.n_latent * spec.n_fixed;
        if support.len() != expected_support
            || fixed_precision.len() != expected_fixed_precision
            || fixed_linear.len() != expected_fixed_linear
            || latent_precision.len() != expected_latent
            || latent_linear.len() != expected_latent
            || latent_fixed.len() != expected_latent_fixed
            || log_constant.len() != n_support
        {
            return Err(
                "theta state evidence dimensions do not match support, fixed, and iid sizes"
                    .to_string(),
            );
        }
        if let (Some(rows), Some(cols), Some(values)) =
            (theta_sparse_i, theta_sparse_j, theta_sparse_x)
        {
            if rows.len() != cols.len() || values.len() != n_support * rows.len() {
                return Err(
                    "theta sparse state evidence dimensions do not match support and edge count"
                        .to_string(),
                );
            }
            for edge_idx in 0..rows.len() {
                if rows[edge_idx] >= spec.n_latent
                    || cols[edge_idx] >= spec.n_latent
                    || rows[edge_idx] == cols[edge_idx]
                {
                    return Err(format!(
                        "theta sparse state evidence edge {} has invalid latent indices",
                        edge_idx
                    ));
                }
            }
            if values.iter().any(|value| !value.is_finite()) {
                return Err(
                    "theta sparse state evidence precision must contain only finite values"
                        .to_string(),
                );
            }
        }
        if support.iter().any(|value| !value.is_finite())
            || fixed_precision.iter().any(|value| !value.is_finite())
            || fixed_linear.iter().any(|value| !value.is_finite())
            || latent_precision
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
            || latent_linear.iter().any(|value| !value.is_finite())
            || latent_fixed.iter().any(|value| !value.is_finite())
            || log_constant.iter().any(|value| !value.is_finite())
        {
            return Err(
                "theta state evidence must contain finite values and non-negative iid precision"
                    .to_string(),
            );
        }
        for support_idx in 0..n_support {
            let offset = support_idx * spec.n_fixed * spec.n_fixed;
            for j in 0..spec.n_fixed {
                if fixed_precision[offset + j * spec.n_fixed + j] < 0.0 {
                    return Err(
                        "theta state fixed precision must have non-negative diagonal entries"
                            .to_string(),
                    );
                }
            }
        }
    }
    if !theta_state_present.iter().any(|present| *present)
        && theta_sparse_state_present.iter().any(|present| *present)
    {
        return Err(
            "theta sparse state evidence requires the complete theta state evidence block"
                .to_string(),
        );
    }

    if let Some(latent_init) = &spec.latent_init {
        if latent_init.len() != spec.n_latent {
            return Err(format!(
                "latent_init length {} does not match n_latent {}",
                latent_init.len(),
                spec.n_latent
            ));
        }
        if latent_init.iter().any(|value| !value.is_finite()) {
            return Err("latent_init must contain only finite values".to_string());
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
        if fixed_init.iter().any(|value| !value.is_finite()) {
            return Err("fixed_init must contain only finite values".to_string());
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
        theta_prior_mean: spec_map
            .get("theta_prior_mean")
            .map(|obj| parse_optional_real_vec(obj, "theta_prior_mean"))
            .transpose()?
            .flatten(),
        theta_prior_precision: spec_map
            .get("theta_prior_precision")
            .map(|obj| parse_optional_real_vec(obj, "theta_prior_precision"))
            .transpose()?
            .flatten(),
        theta_prior_mask: spec_map
            .get("theta_prior_mask")
            .map(|obj| parse_optional_usize_vec(obj, "theta_prior_mask"))
            .transpose()?
            .flatten(),
        fixed_state_precision: spec_map
            .get("fixed_state_precision")
            .map(|obj| parse_optional_real_vec(obj, "fixed_state_precision"))
            .transpose()?
            .flatten(),
        fixed_state_linear: spec_map
            .get("fixed_state_linear")
            .map(|obj| parse_optional_real_vec(obj, "fixed_state_linear"))
            .transpose()?
            .flatten(),
        latent_state_precision_diag: spec_map
            .get("latent_state_precision_diag")
            .map(|obj| parse_optional_real_vec(obj, "latent_state_precision_diag"))
            .transpose()?
            .flatten(),
        latent_state_linear: spec_map
            .get("latent_state_linear")
            .map(|obj| parse_optional_real_vec(obj, "latent_state_linear"))
            .transpose()?
            .flatten(),
        latent_state_precision_i: spec_map
            .get("latent_state_precision_i")
            .map(|obj| parse_optional_usize_vec(obj, "latent_state_precision_i"))
            .transpose()?
            .flatten(),
        latent_state_precision_j: spec_map
            .get("latent_state_precision_j")
            .map(|obj| parse_optional_usize_vec(obj, "latent_state_precision_j"))
            .transpose()?
            .flatten(),
        latent_state_precision_x: spec_map
            .get("latent_state_precision_x")
            .map(|obj| parse_optional_real_vec(obj, "latent_state_precision_x"))
            .transpose()?
            .flatten(),
        latent_fixed_state_precision: spec_map
            .get("latent_fixed_state_precision")
            .map(|obj| parse_optional_real_vec(obj, "latent_fixed_state_precision"))
            .transpose()?
            .flatten(),
        theta_state_n_support: spec_map
            .get("theta_state_n_support")
            .map(|obj| parse_optional_usize(obj, "theta_state_n_support"))
            .transpose()?
            .flatten(),
        theta_state_support: spec_map
            .get("theta_state_support")
            .map(|obj| parse_optional_real_vec(obj, "theta_state_support"))
            .transpose()?
            .flatten(),
        theta_state_fixed_precision: spec_map
            .get("theta_state_fixed_precision")
            .map(|obj| parse_optional_real_vec(obj, "theta_state_fixed_precision"))
            .transpose()?
            .flatten(),
        theta_state_fixed_linear: spec_map
            .get("theta_state_fixed_linear")
            .map(|obj| parse_optional_real_vec(obj, "theta_state_fixed_linear"))
            .transpose()?
            .flatten(),
        theta_state_latent_precision_diag: spec_map
            .get("theta_state_latent_precision_diag")
            .map(|obj| parse_optional_real_vec(obj, "theta_state_latent_precision_diag"))
            .transpose()?
            .flatten(),
        theta_state_latent_linear: spec_map
            .get("theta_state_latent_linear")
            .map(|obj| parse_optional_real_vec(obj, "theta_state_latent_linear"))
            .transpose()?
            .flatten(),
        theta_state_latent_precision_i: spec_map
            .get("theta_state_latent_precision_i")
            .map(|obj| parse_optional_usize_vec(obj, "theta_state_latent_precision_i"))
            .transpose()?
            .flatten(),
        theta_state_latent_precision_j: spec_map
            .get("theta_state_latent_precision_j")
            .map(|obj| parse_optional_usize_vec(obj, "theta_state_latent_precision_j"))
            .transpose()?
            .flatten(),
        theta_state_latent_precision_x: spec_map
            .get("theta_state_latent_precision_x")
            .map(|obj| parse_optional_real_vec(obj, "theta_state_latent_precision_x"))
            .transpose()?
            .flatten(),
        theta_state_latent_fixed_precision: spec_map
            .get("theta_state_latent_fixed_precision")
            .map(|obj| parse_optional_real_vec(obj, "theta_state_latent_fixed_precision"))
            .transpose()?
            .flatten(),
        theta_state_log_constant: spec_map
            .get("theta_state_log_constant")
            .map(|obj| parse_optional_real_vec(obj, "theta_state_log_constant"))
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

    let qfunc = match build_qfunc(&spec) {
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

    let theta_state_evidence = spec
        .theta_state_n_support
        .map(|n_support| ThetaStateEvidence {
            n_support,
            support: spec.theta_state_support.as_deref().unwrap_or(&[]),
            fixed_precision: spec.theta_state_fixed_precision.as_deref().unwrap_or(&[]),
            fixed_linear: spec.theta_state_fixed_linear.as_deref().unwrap_or(&[]),
            latent_precision_diag: spec
                .theta_state_latent_precision_diag
                .as_deref()
                .unwrap_or(&[]),
            latent_linear: spec.theta_state_latent_linear.as_deref().unwrap_or(&[]),
            latent_precision_i: spec.theta_state_latent_precision_i.as_deref(),
            latent_precision_j: spec.theta_state_latent_precision_j.as_deref(),
            latent_precision_x: spec.theta_state_latent_precision_x.as_deref(),
            latent_fixed_precision: spec
                .theta_state_latent_fixed_precision
                .as_deref()
                .unwrap_or(&[]),
            log_constant: spec.theta_state_log_constant.as_deref().unwrap_or(&[]),
        });

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
        fixed_state_precision: spec.fixed_state_precision.as_deref(),
        fixed_state_linear: spec.fixed_state_linear.as_deref(),
        latent_state_precision_diag: spec.latent_state_precision_diag.as_deref(),
        latent_state_linear: spec.latent_state_linear.as_deref(),
        latent_state_precision_i: spec.latent_state_precision_i.as_deref(),
        latent_state_precision_j: spec.latent_state_precision_j.as_deref(),
        latent_state_precision_x: spec.latent_state_precision_x.as_deref(),
        latent_fixed_state_precision: spec.latent_fixed_state_precision.as_deref(),
        theta_state_evidence,
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
                fixed_var_theta_opt = res.fixed_var_theta_opt,
                fixed_cov_theta_opt = res.fixed_cov_theta_opt,
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
                theta_evidence_fixed_precision = res.theta_evidence_fixed_precision,
                theta_evidence_fixed_linear = res.theta_evidence_fixed_linear,
                theta_evidence_latent_precision_diag = res.theta_evidence_latent_precision_diag,
                theta_evidence_latent_linear = res.theta_evidence_latent_linear,
                theta_evidence_latent_precision_i = res.theta_evidence_latent_precision_i,
                theta_evidence_latent_precision_j = res.theta_evidence_latent_precision_j,
                theta_evidence_latent_precision_x = res.theta_evidence_latent_precision_x,
                theta_evidence_latent_fixed_precision = res.theta_evidence_latent_fixed_precision,
                theta_evidence_log_constant = res.theta_evidence_log_constant,
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
                    state_log_factor = res.laplace_terms.state_log_factor,
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
                    latent_mode_restarts = res.diagnostics.latent_mode_restarts,
                    latent_mode_step_ramp_solves = res.diagnostics.latent_mode_step_ramp_solves,
                    latent_mode_step_factor_min = res.diagnostics.latent_mode_step_factor_min,
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
