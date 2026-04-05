#' Native R Formula Interface for Rusty-INLA
#'
#' @param formula A robust R formula `y ~ 1 + f(cov, model="iid")`.
#' @param data A data.frame containing the variables.
#' @param family The likelihood family.
#' @export
rusty_inla <- function(formula, data, family) {
    # 1. Parse formula to extract response
    tf <- terms(formula, specials = "f")
    resp_idx <- attr(tf, "response")
    if (resp_idx == 0) stop("Formula requires a response variable.")
    
    y_var <- as.character(attr(tf, "variables")[[resp_idx + 1]])
    y <- as.numeric(data[[y_var]])
    
    # Check for intercept
    intercept <- as.logical(attr(tf, "intercept"))

    # 2. Extract f() term
    f_idx <- attr(tf, "specials")$f
    if (is.null(f_idx)) {
        stop("Formula must contain exactly one f(covariate, model=...) term for the current RustyINLA prototype.")
    }
    if (length(f_idx) > 1) {
        stop("Current RustyINLA prototype supports only one f() term.")
    }
    
    # Parse the f() term securely. We use a local environment providing our f() function.
    f_call <- attr(tf, "variables")[[f_idx + 1]]
    eval_env <- new.env(parent = emptyenv())
    eval_env$f <- f
    f_res <- eval(f_call, envir = eval_env)
    cov_name <- f_res$covariate_name
    model_type <- f_res$model
    
    # 3. Handle Covariate Indexing Automatically
    cov_data <- data[[cov_name]]
    if (is.factor(cov_data)) {
        idx <- as.numeric(cov_data)
    } else {
        # Force categorical/idx logic for R-INLA parity
        idx <- as.numeric(as.factor(cov_data))
    }
    
    n_latent <- max(idx, na.rm = TRUE)
    
    # 4. Invoke the Rust Core
    res <- rust_inla_run(
        data = y,
        model_type = model_type,
        likelihood_type = family,
        intercept = intercept,
        n_latent_arg = as.integer(n_latent),
        x_idx_arg = as.integer(idx - 1)
    )
    
    # Error handling from backend
    if (is.character(res)) { stop(res) }
    
    # 5. Build Standard Output Structure matching R-INLA expectations
    fit <- list(
        call = match.call(),
        formula = formula,
        mlik = res$log_mlik,
        summary.fixed = data.frame(
            row.names = c("(Intercept)"),
            mean = res$intercept_mean,
            sd = res$intercept_sd
        ),
        summary.random = list()
    )
    
    # Format Hyperparameters securely if present
    if (length(res$theta_opt) > 0) {
        fit$summary.hyperpar <- data.frame(
            row.names = paste("theta", 1:length(res$theta_opt)),
            mean = res$theta_opt
        )
    } else {
        fit$summary.hyperpar <- data.frame()
    }
    
    # Populate Random Effects (Latent margins)
    rnd_df <- data.frame(
        ID = 1:n_latent,
        mean = res$marg_means,
        sd = sqrt(res$marg_vars)
    )
    fit$summary.random[[cov_name]] <- rnd_df
    
    class(fit) <- "rusty_inla"
    return(fit)
}

#' @export
print.rusty_inla <- function(x, ...) {
    cat("Call:\n")
    print(x$call)
    cat(sprintf("\nLog Marginal-Likelihood: %f\n", x$mlik))
    cat("\nFixed effects:\n")
    print(round(x$summary.fixed, 4))
    invisible(x)
}

#' @export
summary.rusty_inla <- function(object, ...) {
    print(object)
    
    cat("\nRandom effects:\n")
    rnd_name <- names(object$summary.random)[1]
    cat(sprintf("  Name '%s' with %d levels\n", rnd_name, nrow(object$summary.random[[rnd_name]])))
    
    if (nrow(object$summary.hyperpar) > 0) {
        cat("\nModel hyperparameters:\n")
        print(round(object$summary.hyperpar, 4))
    }
    
    invisible(object)
}
