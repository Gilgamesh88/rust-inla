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
    
    # 2. Extract fixed terms design matrix
    t_labels <- attr(tf, "term.labels")
    f_term_idx <- grep("^f\\(", t_labels)
    if (length(f_term_idx) > 0) {
        tf_fixed <- drop.terms(tf, f_term_idx, keep.response = FALSE)
        X_fixed <- model.matrix(tf_fixed, data)
    } else {
        X_fixed <- model.matrix(tf, data)
    }
    
    n_fixed <- ncol(X_fixed)
    x_matrix_flat <- as.numeric(X_fixed)
    
    # 3. Setup A Matrix Triplets for Random Effects
    A_i <- integer()
    A_j <- integer()
    A_x <- numeric()
    
    n_latent_total <- 0
    model_types <- character()
    cov_names <- character()
    n_levels <- integer()
    
    f_idx <- attr(tf, "specials")$f
    
    if (!is.null(f_idx)) {
        eval_env <- new.env(parent = emptyenv())
        eval_env$f <- f
        for (idx_f in f_idx) {
            f_call <- attr(tf, "variables")[[idx_f + 1]]
            f_res <- eval(f_call, envir = eval_env)
            c_name <- f_res$covariate_name
            m_type <- f_res$model
            
            cov_data <- data[[c_name]]
            
            if (is.factor(cov_data)) {
                c_idx <- as.numeric(cov_data)
            } else {
                c_idx <- as.numeric(as.factor(cov_data))
            }
            
            n_latent_cov <- max(c_idx, na.rm=TRUE)
            
            # Map into Trips (N_A rows)
            # R arrays are 1-based. A_i, A_j must be 0-based for Rust!
            valid_rows <- which(!is.na(c_idx))
            A_i <- c(A_i, valid_rows - 1)
            A_j <- c(A_j, (c_idx[valid_rows] - 1) + n_latent_total)
            A_x <- c(A_x, rep(1.0, length(valid_rows)))
            
            n_latent_total <- n_latent_total + n_latent_cov
            model_types <- c(model_types, m_type)
            cov_names <- c(cov_names, c_name)
            n_levels <- c(n_levels, n_latent_cov)
        }
    }
    
    # Join models safely into a CSV string to bypass Extendr vec string overhead for now
    model_types_str <- paste(model_types, collapse = ",")
    
    # 4. Invoke the Rust Core
    res <- rust_inla_run(
        data = y,
        model_types_arg = model_types_str,
        likelihood_type = family,
        fixed_matrix_arg = x_matrix_flat,
        n_fixed_arg = as.integer(n_fixed),
        n_latent_arg = as.integer(n_latent_total),
        a_i_arg = as.integer(A_i),
        a_j_arg = as.integer(A_j),
        a_x_arg = as.numeric(A_x)
    )
    
    # Error handling from backend
    if (is.character(res)) { stop(res) }
    
    # 5. Build Standard Output Structure matching R-INLA expectations
    fit <- list(
        call = match.call(),
        formula = formula,
        mlik = res$log_mlik,
        summary.fixed = data.frame(
            row.names = colnames(X_fixed),
            mean = res$fixed_means,
            sd = res$fixed_sds,
            `0.025quant` = res$fixed_means - 1.96 * res$fixed_sds,
            `0.975quant` = res$fixed_means + 1.96 * res$fixed_sds,
            check.names = FALSE
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
    if (length(cov_names) > 0) {
        start_idx <- 1
        for (k in seq_along(cov_names)) {
            c_name <- cov_names[k]
            nl <- n_levels[k]
            end_idx <- start_idx + nl - 1
            
            rnd_mean <- res$marg_means[start_idx:end_idx]
            rnd_var <- res$marg_vars[start_idx:end_idx]
            rnd_sd <- sqrt(rnd_var)
            
            rnd_df <- data.frame(
                ID = 1:nl,
                mean = rnd_mean,
                sd = rnd_sd,
                `0.025quant` = rnd_mean - 1.96 * rnd_sd,
                `0.975quant` = rnd_mean + 1.96 * rnd_sd,
                check.names = FALSE
            )
            fit$summary.random[[c_name]] <- rnd_df
            start_idx <- end_idx + 1
        }
    }
    
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
    
    if (length(object$summary.random) > 0) {
        cat("\nRandom effects:\n")
        for (rnd_name in names(object$summary.random)) {
            cat(sprintf("  Name '%s' with %d levels\n", rnd_name, nrow(object$summary.random[[rnd_name]])))
        }
    }
    
    if (nrow(object$summary.hyperpar) > 0) {
        cat("\nModel hyperparameters:\n")
        print(round(object$summary.hyperpar, 4))
    }
    
    invisible(object)
}
