#' Execute rust-inla backend from R.

# nolint start

#'
#' @param spec A named backend specification list built by `build_backend_spec()`.
#' @export
rust_inla_run <- function(spec) .Call(wrap__rust_inla_run, spec)


# nolint end
