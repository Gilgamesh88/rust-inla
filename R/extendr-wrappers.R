#' Execute rust-inla backend from R.

# nolint start

#'
#' @param data A numeric array of length N (the response y).
#' @param model_type String identifier ("iid", "rw1", "ar1").
#' @param likelihood_type String identifier ("gaussian", "poisson", "gamma").
#' @param intercept Boolean to estimate global beta0.
#' @export
rust_inla_run <- function(data, model_types_arg, likelihood_type, fixed_matrix_arg, n_fixed_arg, n_latent_arg, a_i_arg, a_j_arg, a_x_arg, extr_constr_arg, n_constr_arg) .Call(wrap__rust_inla_run, data, model_types_arg, likelihood_type, fixed_matrix_arg, n_fixed_arg, n_latent_arg, a_i_arg, a_j_arg, a_x_arg, extr_constr_arg, n_constr_arg)


# nolint end
