#' Define a latent random effect in Rusty-INLA
#'
#' @param covariate The variable to map to the latent effect.
#' @param model A string defining the model ("iid", "rw1", "ar1").
#' @return A list containing the covariate name and model type.
#' @export
f <- function(covariate, model) {
    if (missing(model)) stop("Model type required.")
    cov_expr <- substitute(covariate)
    list(covariate_name = deparse(cov_expr), model = model)
}
