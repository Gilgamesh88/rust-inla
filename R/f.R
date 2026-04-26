#' Define a latent random effect in Rusty-INLA
#'
#' @param covariate The variable to map to the latent effect.
#' @param model A string defining the model ("iid", "rw1", "rw2", "ar1", "ar2").
#' @param constr Optional logical flag for the default intrinsic-model
#'   identifiability constraints. When left as `NULL`, Rusty-INLA follows the
#'   INLA devel defaults for the currently supported latent models: `TRUE` for
#'   `"rw1"` and `"rw2"`, and `FALSE` for `"iid"`, `"ar1"`, and `"ar2"`.
#' @return A list containing the covariate name, model type and constraint flag.
#' @export
f <- function(covariate, model, constr = NULL) {
    if (missing(model)) stop("Model type required.")
    cov_expr <- substitute(covariate)
    if (!is.name(cov_expr) || identical(as.character(cov_expr), ".")) {
        stop("f() covariate must be a single untransformed data column name.", call. = FALSE)
    }
    if (!is.character(model) || length(model) != 1L || is.na(model)) {
        stop("model must be a single supported latent model string.", call. = FALSE)
    }
    default_constr <- switch(
        model,
        rw1 = TRUE,
        rw2 = TRUE,
        iid = FALSE,
        ar1 = FALSE,
        ar2 = FALSE,
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
