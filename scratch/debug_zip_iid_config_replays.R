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

ctrl_compute <- list(config = TRUE, dic = FALSE, waic = FALSE, cpo = FALSE)
ctrl_predictor <- list(compute = TRUE)
ctrl_inla <- list(strategy = "auto", int.strategy = "auto")

formula_zip <- ClaimNb ~ 1 + offset(log(Exposure)) + f(VehBrand, model = "iid")

fit_inla <- inla(
    formula_zip,
    family = "zeroinflatedpoisson1",
    data = df,
    control.compute = ctrl_compute,
    control.predictor = ctrl_predictor,
    control.inla = ctrl_inla,
    num.threads = 1,
    verbose = FALSE
)

extract_zip_rust_theta_order <- function(inla_fit) {
    theta_names <- names(inla_fit$mode$theta)

    pick_one <- function(pattern) {
        hit <- grep(pattern, theta_names, ignore.case = TRUE)
        if (length(hit) != 1L) {
            stop(sprintf(
                "Could not uniquely match ZIP theta '%s' in INLA mode names",
                pattern
            ))
        }
        hit[[1]]
    }

    c(
        pick_one("VehBrand"),
        pick_one("zero-probability")
    )
}

prefix_columns <- function(df, prefix) {
    names(df) <- paste0(prefix, names(df))
    df
}

safe_rmse <- function(x) {
    x <- x[is.finite(x)]
    if (length(x) == 0L) {
        return(NA_real_)
    }
    sqrt(mean(x^2))
}

safe_mean_abs <- function(x) {
    x <- abs(x[is.finite(x)])
    if (length(x) == 0L) {
        return(NA_real_)
    }
    mean(x)
}

safe_max_abs <- function(x) {
    x <- abs(x[is.finite(x)])
    if (length(x) == 0L) {
        return(NA_real_)
    }
    max(x)
}

configs <- fit_inla$misc$configs$config
theta_mat <- matrix(
    unlist(lapply(configs, function(cfg) as.numeric(cfg$theta))),
    ncol = length(as.numeric(configs[[1]]$theta)),
    byrow = TRUE
)
log_post <- vapply(configs, function(cfg) as.numeric(cfg$log.posterior), numeric(1))
order_idx <- order(log_post, decreasing = TRUE)
rust_order <- extract_zip_rust_theta_order(fit_inla)

replay_one <- function(theta_target) {
    spec <- rustyINLA:::build_backend_spec(formula_zip, data = df, family = "zeroinflatedpoisson1")
    spec$theta_init <- as.numeric(theta_target)
    spec$optimizer_max_evals <- 0L
    spec$skip_ccd <- TRUE
    spec$latent_init <- as.numeric(base_raw$mode_x)
    spec$fixed_init <- as.numeric(base_raw$mode_beta)

    out <- tryCatch(
        rust_inla_run(spec),
        error = function(err) conditionMessage(err)
    )

    if (is.character(out)) {
        return(data.frame(
            success = FALSE,
            rust_log_mlik = NA_real_,
            rust_sum_loglik = NA_real_,
            rust_log_prior = NA_real_,
            rust_det_adjustment = NA_real_,
            rust_quadratic_penalty = NA_real_,
            rust_theta_1 = NA_real_,
            rust_theta_2 = NA_real_,
            message = out,
            stringsAsFactors = FALSE
        ))
    }

    terms <- out$laplace_terms

    data.frame(
        success = TRUE,
        rust_log_mlik = as.numeric(out$log_mlik[[1]]),
        rust_sum_loglik = as.numeric(terms$sum_loglik),
        rust_log_prior = as.numeric(terms$log_prior),
        rust_det_adjustment = 0.5 * (
            as.numeric(terms$final_log_det_q) -
                as.numeric(terms$final_log_det_aug)
        ),
        rust_quadratic_penalty = -0.5 * as.numeric(terms$final_q_form),
        rust_theta_1 = as.numeric(out$theta_opt[[1]]),
        rust_theta_2 = as.numeric(out$theta_opt[[2]]),
        message = NA_character_,
        stringsAsFactors = FALSE
    )
}

base_spec <- rustyINLA:::build_backend_spec(formula_zip, data = df, family = "zeroinflatedpoisson1")
base_raw <- rust_inla_run(base_spec)
if (is.character(base_raw)) {
    stop(base_raw)
}

n_replay <- min(12L, length(order_idx))
rows <- vector("list", n_replay)
for (i in seq_len(n_replay)) {
    idx <- order_idx[[i]]
    theta_inla_order <- theta_mat[idx, , drop = TRUE]
    theta_rust_order <- theta_inla_order[rust_order]
    replay_inla_order <- replay_one(theta_inla_order)
    replay_rust_order <- replay_one(theta_rust_order)

    rows[[i]] <- cbind(
        data.frame(
            config_rank = i,
            original_config_id = idx,
            inla_log_posterior = log_post[[idx]],
            theta_inla_order_1 = theta_inla_order[[1]],
            theta_inla_order_2 = theta_inla_order[[2]],
            theta_rust_order_1 = theta_rust_order[[1]],
            theta_rust_order_2 = theta_rust_order[[2]],
            stringsAsFactors = FALSE
        ),
        prefix_columns(replay_inla_order, "inla_order_"),
        prefix_columns(replay_rust_order, "rust_order_")
    )
}

point_df <- do.call(rbind, rows)
point_df$inla_centered <- point_df$inla_log_posterior - max(point_df$inla_log_posterior)
point_df$inla_order_centered <- point_df$inla_order_rust_log_mlik -
    max(point_df$inla_order_rust_log_mlik[point_df$inla_order_success], na.rm = TRUE)
point_df$rust_order_centered <- point_df$rust_order_rust_log_mlik -
    max(point_df$rust_order_rust_log_mlik[point_df$rust_order_success], na.rm = TRUE)

point_df$inla_order_centered_diff <- point_df$inla_order_centered - point_df$inla_centered
point_df$rust_order_centered_diff <- point_df$rust_order_centered - point_df$inla_centered

point_df$delta_log_mlik <- point_df$inla_order_rust_log_mlik - point_df$rust_order_rust_log_mlik
point_df$delta_sum_loglik <- point_df$inla_order_rust_sum_loglik - point_df$rust_order_rust_sum_loglik
point_df$delta_det_adjustment <- point_df$inla_order_rust_det_adjustment -
    point_df$rust_order_rust_det_adjustment
point_df$delta_quadratic_penalty <- point_df$inla_order_rust_quadratic_penalty -
    point_df$rust_order_rust_quadratic_penalty
point_df$delta_log_prior <- point_df$inla_order_rust_log_prior - point_df$rust_order_rust_log_prior

term_diff_df <- data.frame(
    term = c(
        "log_mlik",
        "sum_loglik",
        "det_adjustment",
        "quadratic_penalty",
        "log_prior"
    ),
    rmse = c(
        safe_rmse(point_df$delta_log_mlik),
        safe_rmse(point_df$delta_sum_loglik),
        safe_rmse(point_df$delta_det_adjustment),
        safe_rmse(point_df$delta_quadratic_penalty),
        safe_rmse(point_df$delta_log_prior)
    ),
    mean_abs = c(
        safe_mean_abs(point_df$delta_log_mlik),
        safe_mean_abs(point_df$delta_sum_loglik),
        safe_mean_abs(point_df$delta_det_adjustment),
        safe_mean_abs(point_df$delta_quadratic_penalty),
        safe_mean_abs(point_df$delta_log_prior)
    ),
    max_abs = c(
        safe_max_abs(point_df$delta_log_mlik),
        safe_max_abs(point_df$delta_sum_loglik),
        safe_max_abs(point_df$delta_det_adjustment),
        safe_max_abs(point_df$delta_quadratic_penalty),
        safe_max_abs(point_df$delta_log_prior)
    ),
    stringsAsFactors = FALSE
)

summary_df <- data.frame(
    theta_mode_name_1 = names(fit_inla$mode$theta)[[1]],
    theta_mode_name_2 = names(fit_inla$mode$theta)[[2]],
    rust_order_index_1 = rust_order[[1]],
    rust_order_index_2 = rust_order[[2]],
    inla_order_centered_rmse = safe_rmse(point_df$inla_order_centered_diff),
    rust_order_centered_rmse = safe_rmse(point_df$rust_order_centered_diff),
    stringsAsFactors = FALSE
)

cat("ZIP CONFIG REPLAY SUMMARY\n")
print(summary_df, row.names = FALSE)

cat("\nTERMWISE DIFFERENCE BETWEEN RAW INLA ORDER AND RUST ORDER REPLAYS\n")
print(term_diff_df, row.names = FALSE)

cat("\nTOP-12 ZIP CONFIG REPLAYS\n")
print(point_df[, c(
    "config_rank",
    "original_config_id",
    "inla_log_posterior",
    "theta_inla_order_1",
    "theta_inla_order_2",
    "theta_rust_order_1",
    "theta_rust_order_2",
    "inla_order_rust_log_mlik",
    "rust_order_rust_log_mlik",
    "delta_log_mlik",
    "delta_sum_loglik",
    "delta_det_adjustment",
    "delta_quadratic_penalty",
    "delta_log_prior"
)], row.names = FALSE)
