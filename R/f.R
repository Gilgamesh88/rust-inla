#' Define a latent random effect in Rusty-INLA
#'
#' @param covariate The variable to map to the latent effect.
#' @param model A string defining the model ("iid", "rw1", "ar1").
#' @param constr Optional logical flag for a sum-to-zero constraint. When left
#'   as `NULL`, Rusty-INLA follows the INLA devel defaults for the currently
#'   supported latent models: `TRUE` for `"rw1"` and `FALSE` for `"iid"` and
#'   `"ar1"`.
#' @return A list containing the covariate name, model type and constraint flag.
#' @export
f <- function(covariate, model, constr = NULL) {
    if (missing(model)) stop("Model type required.")
    cov_expr <- substitute(covariate)
    default_constr <- switch(
        model,
        rw1 = TRUE,
        iid = FALSE,
        ar1 = FALSE,
        NULL
    )

    if (is.null(default_constr)) {
        stop(sprintf("Unsupported latent model '%s'.", model), call. = FALSE)
    }

    if (is.null(constr)) {
        constr <- default_constr
    } else if (!is.logical(constr) || length(constr) != 1L || is.na(constr)) {
        stop("constr must be a single TRUE/FALSE value.", call. = FALSE)
    }

    list(
        covariate_name = deparse(cov_expr),
        model = model,
        constr = isTRUE(constr)
    )
}
