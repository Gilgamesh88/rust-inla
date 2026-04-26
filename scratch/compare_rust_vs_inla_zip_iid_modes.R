local_rustyinla_lib <- Sys.getenv(
    "RUSTYINLA_LIB",
    "C:/Users/Antonio/Documents/rustyINLA/rustyINLA/scratch/rlib"
)
if (nzchar(local_rustyinla_lib)) {
    .libPaths(c(
        normalizePath(local_rustyinla_lib, winslash = "/", mustWork = TRUE),
        .libPaths()
    ))
}

suppressPackageStartupMessages({
    library(INLA)
    library(CASdatasets)
    library(rustyINLA)
})

options(digits = 15)

data(freMTPL2freq)

df <- freMTPL2freq
df$VehBrand <- as.factor(df$VehBrand)

models_info <- inla.models()
ctrl_compute <- list(config = TRUE, dic = FALSE, waic = FALSE, cpo = FALSE)
ctrl_predictor <- list(compute = TRUE)
ctrl_inla <- list(strategy = "auto", int.strategy = "auto")

expected_response_mean <- function(family, eta, theta_internal) {
    mu <- exp(eta)

    if (identical(family, "poisson")) {
        return(mu)
    }

    if (identical(family, "zeroinflatedpoisson1")) {
        zero_prob <- plogis(theta_internal[[length(theta_internal)]])
        return((1.0 - zero_prob) * mu)
    }

    stop(sprintf("Unsupported family for expected response mean: %s", family))
}

plugin_loglik <- function(family, y, eta, theta_internal) {
    mu <- exp(eta)

    if (identical(family, "poisson")) {
        return(sum(dpois(y, lambda = mu, log = TRUE)))
    }

    if (identical(family, "zeroinflatedpoisson1")) {
        zero_prob <- plogis(theta_internal[[length(theta_internal)]])
        loglik <- ifelse(
            y == 0,
            log(zero_prob + (1.0 - zero_prob) * exp(-mu)),
            log1p(-zero_prob) + dpois(y, lambda = mu, log = TRUE)
        )
        return(sum(loglik))
    }

    stop(sprintf("Unsupported family for plug-in log-likelihood: %s", family))
}

fit_rust_case <- function(formula, family) {
    spec <- rustyINLA:::build_backend_spec(formula, data = df, family = family)
    elapsed <- system.time(raw <- rust_inla_run(spec))
    if (is.character(raw)) {
        stop(raw)
    }
    list(
        spec = spec,
        raw = raw,
        elapsed_sec = unname(elapsed["elapsed"])
    )
}

fit_inla_case <- function(formula, family) {
    elapsed <- system.time(fit <- inla(
        formula,
        family = family,
        data = df,
        control.compute = ctrl_compute,
        control.predictor = ctrl_predictor,
        control.inla = ctrl_inla,
        num.threads = 1,
        verbose = FALSE
    ))
    list(
        fit = fit,
        elapsed_sec = unname(elapsed["elapsed"])
    )
}

case_hyper_specs <- function(case_name) {
    if (identical(case_name, "zip_iid")) {
        return(list(
            list(
                label = "prec(VehBrand)",
                initial = models_info$latent$iid$hyper$theta$initial,
                from_theta = function(x) exp(x)
            ),
            list(
                label = "zero_prob",
                initial = models_info$likelihood$zeroinflatedpoisson1$hyper$theta$initial,
                from_theta = models_info$likelihood$zeroinflatedpoisson1$hyper$theta$from.theta
            )
        ))
    }

    if (identical(case_name, "poisson_iid")) {
        return(list(
            list(
                label = "prec(VehBrand)",
                initial = models_info$latent$iid$hyper$theta$initial,
                from_theta = function(x) exp(x)
            )
        ))
    }

    stop(sprintf("Unknown case_name: %s", case_name))
}

extract_inla_mode_theta <- function(case_name, inla_fit) {
    theta_values <- as.numeric(inla_fit[["mode"]][["theta"]])
    theta_names <- names(inla_fit[["mode"]][["theta"]])

    pick_one <- function(pattern) {
        hit <- grep(pattern, theta_names, ignore.case = TRUE)
        if (length(hit) != 1L) {
            stop(sprintf(
                "Could not uniquely match INLA mode theta '%s' for case '%s'",
                pattern,
                case_name
            ))
        }
        theta_values[[hit]]
    }

    if (identical(case_name, "zip_iid")) {
        return(c(
            pick_one("VehBrand"),
            pick_one("zero-probability")
        ))
    }

    if (identical(case_name, "poisson_iid")) {
        return(c(
            pick_one("VehBrand")
        ))
    }

    stop(sprintf("Unknown case_name: %s", case_name))
}

extract_inla_mode_components <- function(case_name, inla_fit, latent_tag) {
    mode_x <- as.numeric(inla_fit[["mode"]][["x"]])
    contents <- inla_fit[["misc"]][["configs"]][["contents"]]

    get_block <- function(tag_name) {
        hit <- match(tag_name, contents$tag)
        if (is.na(hit)) {
            stop(sprintf("Could not find mode block '%s' in INLA contents", tag_name))
        }
        start <- contents$start[[hit]]
        len <- contents$length[[hit]]
        mode_x[seq.int(start, length.out = len)]
    }

    list(
        theta = extract_inla_mode_theta(case_name, inla_fit),
        eta = get_block("Predictor"),
        x = get_block(latent_tag),
        beta = get_block("(Intercept)")
    )
}

summarize_starts <- function(case_name, rust_raw) {
    hyper_specs <- case_hyper_specs(case_name)
    inla_initial <- vapply(hyper_specs, `[[`, numeric(1), "initial")

    data.frame(
        case = case_name,
        theta_index = seq_along(hyper_specs),
        label = vapply(hyper_specs, `[[`, character(1), "label"),
        rust_start = as.numeric(rust_raw$theta_init_used),
        inla_initial = inla_initial,
        start_diff = as.numeric(rust_raw$theta_init_used) - inla_initial,
        stringsAsFactors = FALSE
    )
}

summarize_mode_hypers <- function(case_name, rust_theta, inla_theta) {
    hyper_specs <- case_hyper_specs(case_name)
    rows <- vector("list", length(hyper_specs))

    for (idx in seq_along(hyper_specs)) {
        spec <- hyper_specs[[idx]]
        rows[[idx]] <- data.frame(
            case = case_name,
            theta_index = idx,
            label = spec$label,
            rust_internal = rust_theta[[idx]],
            inla_internal = inla_theta[[idx]],
            internal_diff = rust_theta[[idx]] - inla_theta[[idx]],
            rust_external = spec$from_theta(rust_theta[[idx]]),
            inla_external = spec$from_theta(inla_theta[[idx]]),
            stringsAsFactors = FALSE
        )
    }

    do.call(rbind, rows)
}

summarize_mode_fixed <- function(case_name, rust_mode, inla_mode) {
    data.frame(
        case = case_name,
        coefficient = "(Intercept)",
        rust_mode = as.numeric(rust_mode$mode_beta[[1]]),
        inla_mode = as.numeric(inla_mode$beta[[1]]),
        diff = as.numeric(rust_mode$mode_beta[[1]]) - as.numeric(inla_mode$beta[[1]]),
        stringsAsFactors = FALSE
    )
}

summarize_mode_latent <- function(case_name, rust_mode, inla_mode, block_name) {
    rust_x <- as.numeric(rust_mode$mode_x)
    inla_x <- as.numeric(inla_mode$x)

    data.frame(
        case = case_name,
        block = block_name,
        rust_mean_avg = mean(rust_x),
        inla_mean_avg = mean(inla_x),
        mode_rmse = sqrt(mean((rust_x - inla_x)^2)),
        mode_max_abs = max(abs(rust_x - inla_x)),
        stringsAsFactors = FALSE
    )
}

summarize_mode_eta <- function(case_name, family, rust_mode, inla_mode, y) {
    rust_eta <- as.numeric(rust_mode$mode_eta)
    inla_eta <- as.numeric(inla_mode$eta)
    rust_theta <- as.numeric(rust_mode$theta_opt)
    inla_theta <- as.numeric(inla_mode$theta)
    rust_mu <- expected_response_mean(family, rust_eta, rust_theta)
    inla_mu <- expected_response_mean(family, inla_eta, inla_theta)

    data.frame(
        case = case_name,
        rust_eta_avg = mean(rust_eta),
        inla_eta_avg = mean(inla_eta),
        eta_rmse = sqrt(mean((rust_eta - inla_eta)^2)),
        eta_max_abs = max(abs(rust_eta - inla_eta)),
        rust_mu_avg = mean(rust_mu),
        inla_mu_avg = mean(inla_mu),
        mu_rmse = sqrt(mean((rust_mu - inla_mu)^2)),
        mu_max_abs = max(abs(rust_mu - inla_mu)),
        plugin_loglik_rust = plugin_loglik(family, y, rust_eta, rust_theta),
        plugin_loglik_inla = plugin_loglik(family, y, inla_eta, inla_theta),
        stringsAsFactors = FALSE
    )
}

run_case <- function(case_name, formula, family, latent_tag) {
    cat("\n============================================================\n")
    cat("Running case:", case_name, "\n")
    cat("Formula:", deparse(formula, width.cutoff = 500L), "\n")
    flush.console()

    rust_fit <- fit_rust_case(formula, family)
    inla_fit <- fit_inla_case(formula, family)
    inla_mode <- extract_inla_mode_components(case_name, inla_fit$fit, latent_tag)

    list(
        headline = data.frame(
            case = case_name,
            rust_elapsed_sec = rust_fit$elapsed_sec,
            inla_elapsed_sec = inla_fit$elapsed_sec,
            rust_mlik = as.numeric(rust_fit$raw$log_mlik[[1]]),
            inla_mlik = as.numeric(inla_fit$fit$mlik[1, 1]),
            stringsAsFactors = FALSE
        ),
        starts = summarize_starts(case_name, rust_fit$raw),
        theta = summarize_mode_hypers(case_name, rust_fit$raw$theta_opt, inla_mode$theta),
        fixed = summarize_mode_fixed(case_name, rust_fit$raw, inla_mode),
        latent = summarize_mode_latent(case_name, rust_fit$raw, inla_mode, latent_tag),
        eta = summarize_mode_eta(case_name, family, rust_fit$raw, inla_mode, df$ClaimNb)
    )
}

case_results <- list(
    run_case(
        case_name = "zip_iid",
        formula = ClaimNb ~ 1 + offset(log(Exposure)) + f(VehBrand, model = "iid"),
        family = "zeroinflatedpoisson1",
        latent_tag = "VehBrand"
    ),
    run_case(
        case_name = "poisson_iid",
        formula = ClaimNb ~ 1 + offset(log(Exposure)) + f(VehBrand, model = "iid"),
        family = "poisson",
        latent_tag = "VehBrand"
    )
)

headline_df <- do.call(rbind, lapply(case_results, `[[`, "headline"))
starts_df <- do.call(rbind, lapply(case_results, `[[`, "starts"))
theta_df <- do.call(rbind, lapply(case_results, `[[`, "theta"))
fixed_df <- do.call(rbind, lapply(case_results, `[[`, "fixed"))
latent_df <- do.call(rbind, lapply(case_results, `[[`, "latent"))
eta_df <- do.call(rbind, lapply(case_results, `[[`, "eta"))

cat("\nHEADLINE SUMMARY\n")
print(headline_df, row.names = FALSE)

cat("\nSTART VALUE COMPARISON\n")
print(starts_df, row.names = FALSE)

cat("\nMODE THETA COMPARISON\n")
print(theta_df, row.names = FALSE)

cat("\nMODE FIXED EFFECT COMPARISON\n")
print(fixed_df, row.names = FALSE)

cat("\nMODE LATENT BLOCK COMPARISON\n")
print(latent_df, row.names = FALSE)

cat("\nMODE PREDICTOR / PLUG-IN LIKELIHOOD COMPARISON\n")
print(eta_df, row.names = FALSE)
