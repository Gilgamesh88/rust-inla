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
    model_types_arg: &str, 
    likelihood_type: &str, 
    fixed_matrix_arg: Robj,
    n_fixed_arg: i32,
    n_latent_arg: Robj,
    a_i_arg: Robj,
    a_j_arg: Robj,
    a_x_arg: Robj
) -> Robj {
    // Attempt to slice directly into R's memory without allocating if REAL
    let y_slice = match data.as_real_slice() {
        Some(s) => s,
        None => return r!(format!("Error: data must be a real vector")),
    };
    
    let n_data = y_slice.len();
    
    let x_mat_slice = fixed_matrix_arg.as_real_slice();
    let n_fixed = n_fixed_arg as usize;

    let (n_latent, a_i, a_j, a_x) = if a_i_arg.is_null() {
        (n_data, None, None, None)
    } else {
        let n_lat: i32 = n_latent_arg.as_integer().unwrap_or(a_i_arg.len() as i32);
        
        let a_i_slice = match a_i_arg.as_integer_slice() {
            Some(s) => s,
            None => return r!(format!("Error: a_i must be an integer vector")),
        };
        let a_i_vec: Vec<usize> = a_i_slice.iter().map(|&v| v as usize).collect();
        
        let a_j_slice = match a_j_arg.as_integer_slice() {
            Some(s) => s,
            None => return r!(format!("Error: a_j must be an integer vector")),
        };
        let a_j_vec: Vec<usize> = a_j_slice.iter().map(|&v| v as usize).collect();

        // Avoid cloning heavy numeric arrays, just map them
        let a_x_slice = match a_x_arg.as_real_slice() {
            Some(s) => s,
            None => return r!(format!("Error: a_x must be a real vector")),
        };
        let a_x_vec: Vec<f64> = a_x_slice.to_vec();
        
        (n_lat as usize, Some(a_i_vec), Some(a_j_vec), Some(a_x_vec))
    };

    // Temporarily, we will just use the FIRST model_types_arg separated by comma until CompoundQFunc is built!
    let mut model_type_parts = model_types_arg.split(',');
    let primary_model = model_type_parts.next().unwrap_or("iid");

    // Box the dispatch to avoid lifetime complexity in dynamic trait objects
    let qfunc: Box<dyn QFunc> = match primary_model {
        "iid" => Box::new(IidModel::new(n_latent)),
        "rw1" => Box::new(Rw1Model::new(n_latent)),
        "ar1" => Box::new(Ar1Model::new(n_latent)),
        _ => return r!(format!("Unknown model_type: {}", primary_model)),
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
        a_i: a_i.as_deref(),
        a_j: a_j.as_deref(),
        a_x: a_x.as_deref(),
        offset: None,
    };

    let params = InlaParams::default();

    match InlaEngine::run(&model, &params) {
        Ok(res) => {
            // Build return list to R
            // Unpack random marginals (just mean and var for now)
            let mut marg_means = Vec::with_capacity(n_latent);
            let mut marg_vars = Vec::with_capacity(n_latent);

            let mut fitted_mean = Vec::with_capacity(n_latent);
            let mut fitted_q025 = Vec::with_capacity(n_latent);
            let mut fitted_q500 = Vec::with_capacity(n_latent);
            let mut fitted_q975 = Vec::with_capacity(n_latent);
            let mut fitted_mode = Vec::with_capacity(n_latent);

            // Because the inverse link (exp/logit) is monotonically increasing, 
            // quantiles pass through exactly mapping Quantile(eta) -> Quantile(mu)
            let link_inv = |eta: f64| lik.link().inverse(eta);

            for m in &res.random {
                marg_means.push(m.mean());
                marg_vars.push(m.variance());
                
                fitted_mean.push(m.emarginal(&link_inv));
                fitted_q025.push(link_inv(m.quantile(0.025)));
                fitted_q500.push(link_inv(m.quantile(0.500)));
                fitted_q975.push(link_inv(m.quantile(0.975)));
                
                // Provide a safe response-scale peak approximation
                let mode = match lik.link() {
                    crate::likelihood::LinkFunction::Log => (m.mean() - m.variance()).exp(),
                    _ => m.quantile(0.50),
                };
                fitted_mode.push(mode);
            }

            list!(
                log_mlik = res.log_mlik,
                theta_opt = res.theta_opt,
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
