use extendr_api::prelude::*;
use crate::models::{IidModel, Rw1Model, Ar1Model, QFunc};
use crate::likelihood::{GaussianLikelihood, PoissonLikelihood, GammaLikelihood, ZipLikelihood, TweedieLikelihood, LogLikelihood};
use crate::inference::{InlaParams, InlaModel, InlaEngine};

/// Execute rust-inla backend from R.
///
/// @param data A numeric array of length N (the response y).
/// @param model_type String identifier ("iid", "rw1", "ar1").
/// @param likelihood_type String identifier ("gaussian", "poisson", "gamma").
/// @param intercept Boolean to estimate global beta0.
/// @export
#[extendr]
fn rust_inla_run(
    data: Robj, 
    model_type: &str, 
    likelihood_type: &str, 
    fixed_matrix_arg: Robj,
    n_fixed_arg: i32,
    n_latent_arg: Robj,
    x_idx_arg: Robj
) -> Robj {
    // Attempt to slice directly into R's memory without allocating if REAL
    let y_slice = match data.as_real_slice() {
        Some(s) => s,
        None => return r!(format!("Error: data must be a real vector")),
    };
    
    let n_data = y_slice.len();
    
    let x_mat_slice = fixed_matrix_arg.as_real_slice();
    let n_fixed = n_fixed_arg as usize;

    let (n_latent, x_idx_vec) = if x_idx_arg.is_null() {
        (n_data, None)
    } else {
        match x_idx_arg.as_integer_slice() {
            Some(s) => {
                let n_lat: i32 = n_latent_arg.as_integer().unwrap_or(s.len() as i32);
                let vec_u: Vec<usize> = s.iter().map(|&v| v as usize).collect();
                (n_lat as usize, Some(vec_u))
            },
            None => return r!(format!("Error: x_idx must be an integer vector")),
        }
    };

    // Box the dispatch to avoid lifetime complexity in dynamic trait objects
    let qfunc: Box<dyn QFunc> = match model_type {
        "iid" => Box::new(IidModel::new(n_latent)),
        "rw1" => Box::new(Rw1Model::new(n_latent)),
        "ar1" => Box::new(Ar1Model::new(n_latent)),
        _ => return r!(format!("Unknown model_type: {}", model_type)),
    };

    let lik: Box<dyn LogLikelihood> = match likelihood_type {
        "gaussian" => Box::new(GaussianLikelihood),
        "poisson" => Box::new(PoissonLikelihood),
        "gamma" => Box::new(GammaLikelihood),
        "zeroinflatedpoisson1" => Box::new(ZipLikelihood),
        "tweedie" => Box::new(TweedieLikelihood),
        _ => return r!(format!("Unknown likelihood_type: {}", likelihood_type)),
    };

    // Initialize theta based on n_hyperparams
    let theta_init = vec![0.0; qfunc.n_hyperparams() + lik.n_hyperparams()];
    
    let model = InlaModel {
        qfunc: qfunc.as_ref(),
        likelihood: lik.as_ref(),
        y: y_slice,
        theta_init,
        fixed_matrix: x_mat_slice,
        n_fixed,
        n_latent,
        x_idx: x_idx_vec.as_deref(),
    };

    let params = InlaParams::default();

    match InlaEngine::run(&model, &params) {
        Ok(res) => {
            // Build return list to R
            // Unpack random marginals (just mean and var for now)
            let mut marg_means = Vec::with_capacity(n_latent);
            let mut marg_vars = Vec::with_capacity(n_latent);
            for m in &res.random {
                marg_means.push(m.mean());
                marg_vars.push(m.variance());
            }

            list!(
                log_mlik = res.log_mlik,
                theta_opt = res.theta_opt,
                fixed_means = res.fixed_means,
                fixed_sds = res.fixed_sds,
                marg_means = marg_means,
                marg_vars = marg_vars,
                ccd_thetas = res.ccd_thetas,
                ccd_weights = res.ccd_weights,
                prior_W = res.w_opt,
                prior_mean = res.posterior_mean,
            ).into_robj()
        },
        Err(e) => r!(format!("Engine Error: {:?}", e)),
    }
}

// Macro to initialize the extendr module.
// Note: name matches the package/module name expected by R
extendr_module! {
    mod rustyINLA;
    fn rust_inla_run;
}
