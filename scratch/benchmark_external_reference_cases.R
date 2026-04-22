#!/usr/bin/env Rscript

# External reference benchmarks for rustyINLA vs R-INLA.
#
# This script does two things:
# 1. Retrieves or generates benchmark-ready datasets for the five active model
#    families we care about.
# 2. Fits the matching model with rustyINLA and R-INLA and reports accuracy,
#    time, and memory side by side.
#
# Public-source cases:
# - Poisson + iid + offset: spData::nc.sids
# - Poisson + iid + iid + offset: Gelman/Hill frisk_with_noise.dat
# - Gaussian + rw2: SemiPar::lidar
# - Poisson + ar1: MixtureInf::earthquake
#
# Synthetic fallback cases:
# - Gamma + rw1: no clean public exact gamma+rw1 benchmark dataset was found
#   on the first pass, so this follows the official INLA gamma parameterization
#   with an ordered RW1 latent effect.
# - ZIP + iid + offset: no clean public exact type1 ZIP + iid + offset dataset
#   was found on the first pass, so this uses a reproducible synthetic
#   zeroinflatedpoisson1 reference.

local_rustyinla_lib <- Sys.getenv("RUSTYINLA_LIB", "")
if (nzchar(local_rustyinla_lib)) {
    .libPaths(c(
        normalizePath(local_rustyinla_lib, winslash = "/", mustWork = TRUE),
        .libPaths()
    ))
}

install_missing <- identical(Sys.getenv("RUSTYINLA_INSTALL_MISSING", "0"), "1")
rusty_output_profile <- Sys.getenv("RUSTYINLA_OUTPUT_PROFILE", "thin")
if (!(rusty_output_profile %in% c("thin", "benchmark"))) {
    stop("RUSTYINLA_OUTPUT_PROFILE must be 'thin' or 'benchmark'.")
}
external_output_path <- Sys.getenv("RUSTYINLA_EXTERNAL_BENCHMARK_OUT", "")

cache_dir <- Sys.getenv(
    "RUSTYINLA_EXTERNAL_BENCHMARK_DIR",
    file.path(getwd(), "scratch", "external_reference_data")
)
dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)

`%||%` <- function(x, y) {
    if (is.null(x)) y else x
}

ensure_package <- function(pkg) {
    if (requireNamespace(pkg, quietly = TRUE)) {
        return(TRUE)
    }

    if (!install_missing) {
        return(FALSE)
    }

    archive_urls <- c(
        MixtureInf = "https://cran.r-project.org/src/contrib/Archive/MixtureInf/MixtureInf_1.1.tar.gz"
    )

    if (pkg %in% names(archive_urls)) {
        if (pkg == "MixtureInf" && !requireNamespace("quadprog", quietly = TRUE)) {
            install.packages("quadprog", repos = "https://cloud.r-project.org")
        }
        install.packages(archive_urls[[pkg]], repos = NULL, type = "source")
    } else {
        install.packages(pkg, repos = "https://cloud.r-project.org")
    }

    requireNamespace(pkg, quietly = TRUE)
}

download_if_missing <- function(url, dest) {
    if (file.exists(dest)) {
        return(dest)
    }

    utils::download.file(url, destfile = dest, mode = "wb", quiet = TRUE)
    dest
}

peak_mem_mb <- function(gc_res) {
    col_names <- trimws(colnames(gc_res))
    max_used_col <- which(tolower(col_names) == "max used")
    if (length(max_used_col) > 0 && max_used_col[[1]] < ncol(gc_res)) {
        return(sum(gc_res[, max_used_col[[1]] + 1]))
    }

    mb_cols <- which(col_names == "(Mb)")
    if (length(mb_cols) > 0) {
        return(sum(gc_res[, mb_cols[[length(mb_cols)]]]))
    }

    sum(gc_res[, ncol(gc_res)])
}

benchmark_error <- function(message) {
    structure(list(message = message), class = "benchmark_error")
}

track_perf <- function(expr_sub, envir = parent.frame()) {
    gc(reset = TRUE)
    t0 <- proc.time()[["elapsed"]]
    res <- tryCatch(
        eval(expr_sub, envir = envir),
        error = function(e) benchmark_error(conditionMessage(e))
    )
    elapsed <- proc.time()[["elapsed"]] - t0
    mem <- peak_mem_mb(gc())

    list(
        res = res,
        ok = !inherits(res, "benchmark_error"),
        error = if (inherits(res, "benchmark_error")) res$message else NA_character_,
        time = unname(elapsed),
        mem = unname(mem)
    )
}

named_column <- function(df, column) {
    if (is.null(df) || nrow(df) == 0 || !(column %in% names(df))) {
        return(setNames(numeric(), character()))
    }
    stats::setNames(as.numeric(df[[column]]), rownames(df))
}

indexed_column <- function(df, column) {
    if (is.null(df) || nrow(df) == 0 || !(column %in% names(df))) {
        return(setNames(numeric(), character()))
    }
    stats::setNames(as.numeric(df[[column]]), as.character(seq_len(nrow(df))))
}

collect_random_metric <- function(fit, metric) {
    if (is.null(fit$summary.random) || length(fit$summary.random) == 0) {
        return(list())
    }
    out <- lapply(fit$summary.random, function(df) {
        if (!(metric %in% names(df))) {
            return(setNames(numeric(), character()))
        }
        ids <- if ("ID" %in% names(df)) as.character(df$ID) else rownames(df)
        stats::setNames(as.numeric(df[[metric]]), ids)
    })
    out[order(names(out))]
}

compare_named_numeric <- function(lhs, rhs, abs_tol = NULL) {
    shared <- intersect(names(lhs), names(rhs))
    if (length(shared) == 0) {
        return(list(
            n = 0L,
            max_abs = NA_real_,
            mean_abs = NA_real_,
            worst = NA_character_,
            pass = NA
        ))
    }

    lhs_vals <- as.numeric(lhs[shared])
    rhs_vals <- as.numeric(rhs[shared])
    keep <- is.finite(lhs_vals) & is.finite(rhs_vals)
    if (!any(keep)) {
        return(list(
            n = 0L,
            max_abs = NA_real_,
            mean_abs = NA_real_,
            worst = NA_character_,
            pass = NA
        ))
    }

    shared <- shared[keep]
    diffs <- abs(lhs_vals[keep] - rhs_vals[keep])
    worst_idx <- which.max(diffs)
    list(
        n = length(shared),
        max_abs = max(diffs),
        mean_abs = mean(diffs),
        worst = shared[[worst_idx]],
        pass = if (is.null(abs_tol)) NA else max(diffs) <= abs_tol
    )
}

compare_nested_metrics <- function(lhs_list, rhs_list, abs_tol) {
    shared_terms <- intersect(names(lhs_list), names(rhs_list))
    if (length(shared_terms) == 0) {
        return(list(
            n = 0L,
            max_abs = NA_real_,
            mean_abs = NA_real_,
            worst = NA_character_,
            pass = NA
        ))
    }

    diffs <- numeric()
    labels <- character()
    for (term in shared_terms) {
        lhs <- lhs_list[[term]]
        rhs <- rhs_list[[term]]
        shared_ids <- intersect(names(lhs), names(rhs))
        if (length(shared_ids) > 0) {
            term_diffs <- abs(lhs[shared_ids] - rhs[shared_ids])
            diffs <- c(diffs, term_diffs)
            labels <- c(labels, paste(term, shared_ids, sep = "::"))
            next
        }

        if (length(lhs) == length(rhs) && length(lhs) > 0) {
            term_diffs <- abs(unname(lhs) - unname(rhs))
            diffs <- c(diffs, term_diffs)
            labels <- c(
                labels,
                paste(term, seq_along(term_diffs), sep = "::")
            )
        }
    }

    if (length(diffs) == 0) {
        return(list(
            n = 0L,
            max_abs = NA_real_,
            mean_abs = NA_real_,
            worst = NA_character_,
            pass = NA
        ))
    }

    worst_idx <- which.max(diffs)
    list(
        n = length(diffs),
        max_abs = max(diffs),
        mean_abs = mean(diffs),
        worst = labels[[worst_idx]],
        pass = max(diffs) <= abs_tol
    )
}

compare_fitted_mean <- function(rusty_fit, inla_fit, rel_tol) {
    rusty_fitted <- rusty_fit$summary.fitted.values
    inla_fitted <- inla_fit$summary.fitted.values
    if (is.null(rusty_fitted) || is.null(inla_fitted)) {
        return(list(
            n = 0L,
            max_rel = NA_real_,
            mean_rel = NA_real_,
            worst = NA_integer_,
            pass = NA
        ))
    }

    n_shared <- min(nrow(rusty_fitted), nrow(inla_fitted))
    if (n_shared == 0) {
        return(list(
            n = 0L,
            max_rel = NA_real_,
            mean_rel = NA_real_,
            worst = NA_integer_,
            pass = NA
        ))
    }

    rusty_mean <- as.numeric(rusty_fitted$mean[seq_len(n_shared)])
    inla_mean <- as.numeric(inla_fitted$mean[seq_len(n_shared)])
    rel_diff <- abs(rusty_mean - inla_mean) / pmax(1.0, abs(inla_mean))
    worst_idx <- which.max(rel_diff)

    list(
        n = n_shared,
        max_rel = max(rel_diff),
        mean_rel = mean(rel_diff),
        worst = worst_idx,
        pass = max(rel_diff) <= rel_tol
    )
}

extract_mlik <- function(fit) {
    if (is.null(fit$mlik) || length(fit$mlik) == 0) {
        return(NA_real_)
    }

    mlik <- fit$mlik
    if (is.matrix(mlik) || is.data.frame(mlik)) {
        return(as.numeric(mlik[[1]]))
    }

    as.numeric(mlik[[1]])
}

build_gamma_rw1_reference <- function(seed = 20260418L) {
    set.seed(seed)
    n_groups <- 40L
    reps_per_group <- 15L
    rw_index <- rep(seq_len(n_groups), each = reps_per_group)
    u <- cumsum(c(rnorm(1L, sd = 0.08), rnorm(n_groups - 1L, sd = 0.08)))
    u <- u - mean(u)
    intercept <- 6.2
    mu <- exp(intercept + u[rw_index])
    phi <- 18.0
    y <- rgamma(length(mu), shape = phi, rate = phi / mu)

    data.frame(
        y = y,
        rw_index = rw_index
    )
}

build_zip_iid_offset_reference <- function(seed = 20260419L) {
    set.seed(seed)
    n_groups <- 35L
    reps_per_group <- 18L
    group <- factor(rep(seq_len(n_groups), each = reps_per_group))
    exposure <- runif(n_groups * reps_per_group, min = 0.4, max = 2.2)
    u <- rnorm(n_groups, mean = 0, sd = 0.35)
    eta <- -1.1 + u[as.integer(group)]
    lambda <- exposure * exp(eta)
    p_zero <- plogis(-1.0)
    structural_zero <- rbinom(length(lambda), size = 1L, prob = p_zero)
    y <- ifelse(structural_zero == 1L, 0L, rpois(length(lambda), lambda = lambda))

    data.frame(
        y = y,
        exposure = exposure,
        group = group
    )
}

prepare_case_record <- function(
    id,
    label,
    family,
    formula,
    data_path,
    source_type,
    provenance,
    available = TRUE,
    reason = NA_character_,
    tolerances = NULL
) {
    list(
        id = id,
        label = label,
        family = family,
        formula = formula,
        data_path = data_path,
        source_type = source_type,
        provenance = provenance,
        available = available,
        reason = reason,
        tolerances = tolerances
    )
}

prepare_nc_sids_case <- function(cache_dir) {
    if (!ensure_package("spData")) {
        return(prepare_case_record(
            id = "poisson_iid_offset_nc_sids",
            label = "NC SIDS Poisson + iid + offset",
            family = "poisson",
            formula = SID74 ~ 1 + offset(log(BIR74)) + f(CNTY.ID, model = "iid"),
            data_path = NA_character_,
            source_type = "public_exact",
            provenance = "spData::nc.sids",
            available = FALSE,
            reason = "Package 'spData' is not installed."
        ))
    }

    df <- spData::nc.sids
    df$CNTY.ID <- factor(df$CNTY.ID)
    path <- file.path(cache_dir, "poisson_iid_offset_nc_sids.rds")
    saveRDS(df, path)

    prepare_case_record(
        id = "poisson_iid_offset_nc_sids",
        label = "NC SIDS Poisson + iid + offset",
        family = "poisson",
        formula = SID74 ~ 1 + offset(log(BIR74)) + f(CNTY.ID, model = "iid"),
        data_path = path,
        source_type = "public_exact",
        provenance = paste(
            "spData::nc.sids;",
            "INLA book mixed-effects/count examples;"
        )
    )
}

prepare_nyc_stops_case <- function(cache_dir) {
    dest <- file.path(cache_dir, "frisk_with_noise.dat")
    url <- "https://www2.stat.duke.edu/~pdh10/Teaching/560/Data/frisk_with_noise.dat"

    raw_data <- tryCatch(
        {
            download_if_missing(url, dest)
            first_line <- readLines(dest, n = 1L, warn = FALSE)
            skip_lines <- if (grepl("^stops\\s+pop\\s+past\\.arrests\\s+precinct\\s+eth\\s+crime\\s*$", first_line)) {
                0L
            } else {
                6L
            }
            read.table(dest, skip = skip_lines, header = TRUE)
        },
        error = function(e) e
    )

    if (inherits(raw_data, "error")) {
        return(prepare_case_record(
            id = "poisson_iid_iid_offset_nyc_stops",
            label = "NYC Stops Poisson + iid + iid + offset",
            family = "poisson",
            formula = stops ~ eth +
                f(precinct, model = "iid") +
                f(ID, model = "iid") +
                offset(log((15 / 12) * past.arrests)),
            data_path = NA_character_,
            source_type = "public_exact",
            provenance = paste(
                "Gelman/Hill police stop-and-frisk example;",
                "INLA book Chapter 4"
            ),
            available = FALSE,
            reason = conditionMessage(raw_data)
        ))
    }

    raw_data$eth <- factor(raw_data$eth, levels = c(1, 2, 3))
    levels(raw_data$eth) <- c("black", "hispanic", "white")
    raw_data$eth <- stats::relevel(raw_data$eth, "white")

    agg <- aggregate(
        cbind(stops, past.arrests, pop) ~ precinct + eth,
        data = raw_data,
        FUN = sum
    )
    agg$pop <- agg$pop / 4
    agg$precinct <- factor(agg$precinct)
    agg$ID <- factor(seq_len(nrow(agg)))

    path <- file.path(cache_dir, "poisson_iid_iid_offset_nyc_stops.rds")
    saveRDS(agg, path)

    prepare_case_record(
        id = "poisson_iid_iid_offset_nyc_stops",
        label = "NYC Stops Poisson + iid + iid + offset",
        family = "poisson",
        formula = stops ~ eth +
            f(precinct, model = "iid") +
            f(ID, model = "iid") +
            offset(log((15 / 12) * past.arrests)),
        data_path = path,
        source_type = "public_exact",
        provenance = paste(
            "frisk_with_noise.dat;",
            "Gelman/Hill police-stop example;",
            "INLA book Chapter 4"
        ),
        tolerances = list(
            random_mean_abs = 1.00,
            random_sd_abs = 1.00,
            fitted_mean_rel = 0.50
        )
    )
}

prepare_gamma_rw1_case <- function(cache_dir) {
    df <- build_gamma_rw1_reference()
    path <- file.path(cache_dir, "gamma_rw1_reference_synthetic.rds")
    saveRDS(df, path)

    prepare_case_record(
        id = "gamma_rw1_reference",
        label = "Synthetic Gamma + rw1 reference",
        family = "gamma",
        formula = y ~ 1 + f(rw_index, model = "rw1"),
        data_path = path,
        source_type = "synthetic_exact_family",
        provenance = paste(
            "Synthetic reference using official INLA gamma parameterization",
            "with an ordered RW1 latent effect"
        ),
        tolerances = list(
            fitted_mean_rel = 0.35
        )
    )
}

prepare_rw2_lidar_case <- function(cache_dir) {
    if (!ensure_package("SemiPar")) {
        return(prepare_case_record(
            id = "gaussian_rw2_lidar",
            label = "LIDAR Gaussian + rw2 smoothing",
            family = "gaussian",
            formula = logratio ~ -1 + f(range, model = "rw2", constr = FALSE),
            data_path = NA_character_,
            source_type = "public_exact",
            provenance = "SemiPar::lidar; INLA GitBook Chapter 9 smoothing",
            available = FALSE,
            reason = "Package 'SemiPar' is not installed."
        ))
    }

    data("lidar", package = "SemiPar", envir = environment())
    df <- get("lidar", envir = environment())
    path <- file.path(cache_dir, "gaussian_rw2_lidar.rds")
    saveRDS(df, path)

    prepare_case_record(
        id = "gaussian_rw2_lidar",
        label = "LIDAR Gaussian + rw2 smoothing",
        family = "gaussian",
        formula = logratio ~ -1 + f(range, model = "rw2", constr = FALSE),
        data_path = path,
        source_type = "public_exact",
        provenance = "SemiPar::lidar; INLA GitBook Chapter 9 smoothing",
        tolerances = list(
            random_mean_abs = 0.35,
            random_sd_abs = 0.35,
            fitted_mean_rel = 0.35
        )
    )
}

prepare_earthquake_case <- function(cache_dir) {
    if (!ensure_package("MixtureInf")) {
        return(prepare_case_record(
            id = "poisson_ar1_earthquake",
            label = "Earthquake Poisson + ar1",
            family = "poisson",
            formula = number ~ 1 + f(year, model = "ar1"),
            data_path = NA_character_,
            source_type = "public_exact",
            provenance = "MixtureInf::earthquake; INLA book Chapter 8",
            available = FALSE,
            reason = "Package 'MixtureInf' is not installed."
        ))
    }

    data("earthquake", package = "MixtureInf", envir = environment())
    df <- get("earthquake", envir = environment())
    df$year <- seq(1900L, by = 1L, length.out = nrow(df))
    path <- file.path(cache_dir, "poisson_ar1_earthquake.rds")
    saveRDS(df, path)

    prepare_case_record(
        id = "poisson_ar1_earthquake",
        label = "Earthquake Poisson + ar1",
        family = "poisson",
        formula = number ~ 1 + f(year, model = "ar1"),
        data_path = path,
        source_type = "public_exact",
        provenance = "MixtureInf::earthquake; INLA book Chapter 8",
        tolerances = list(
            random_mean_abs = 0.75,
            random_sd_abs = 0.75,
            fitted_mean_rel = 0.35
        )
    )
}

prepare_zip_iid_case <- function(cache_dir) {
    df <- build_zip_iid_offset_reference()
    path <- file.path(cache_dir, "zip_iid_offset_reference_synthetic.rds")
    saveRDS(df, path)

    prepare_case_record(
        id = "zip_iid_offset_reference",
        label = "Synthetic ZIP type1 + iid + offset reference",
        family = "zeroinflatedpoisson1",
        formula = y ~ 1 + offset(log(exposure)) + f(group, model = "iid"),
        data_path = path,
        source_type = "synthetic_exact_family",
        provenance = paste(
            "Synthetic reference using official zeroinflatedpoisson1 semantics",
            "with exposure and an iid latent effect"
        ),
        tolerances = list(
            random_mean_abs = 1.00,
            random_sd_abs = 1.00,
            fitted_mean_rel = 0.50
        )
    )
}

prepare_cases <- function(cache_dir) {
    list(
        prepare_nc_sids_case(cache_dir),
        prepare_nyc_stops_case(cache_dir),
        prepare_gamma_rw1_case(cache_dir),
        prepare_rw2_lidar_case(cache_dir),
        prepare_earthquake_case(cache_dir),
        prepare_zip_iid_case(cache_dir)
    )
}

evaluate_case <- function(case) {
    tolerances <- modifyList(
        list(
            fixed_mean_abs = 0.25,
            random_mean_abs = 0.50,
            random_sd_abs = 0.50,
            fitted_mean_rel = 0.25
        ),
        case$tolerances %||% list()
    )

    if (!case$available) {
        return(data.frame(
            case_id = case$id,
            label = case$label,
            source_type = case$source_type,
            output_profile = rusty_output_profile,
            available = FALSE,
            passed = FALSE,
            reason = case$reason,
            rusty_time = NA_real_,
            inla_time = NA_real_,
            rusty_mem = NA_real_,
            inla_mem = NA_real_,
            mlik_abs_diff = NA_real_,
            fixed_mean_max_abs = NA_real_,
            fixed_sd_max_abs = NA_real_,
            random_mean_max_abs = NA_real_,
            random_sd_max_abs = NA_real_,
            fitted_mean_max_rel = NA_real_,
            hyper_mean_max_abs = NA_real_,
            hyper_sd_max_abs = NA_real_,
            linear_predictor_mean_max_abs = NA_real_,
            linear_predictor_sd_max_abs = NA_real_,
            stringsAsFactors = FALSE
        ))
    }

    df <- readRDS(case$data_path)
    rusty_expr <- bquote(
        rusty_inla(
            .(case$formula),
            data = df,
            family = .(case$family),
            output_profile = .(rusty_output_profile)
        )
    )
    inla_expr <- bquote(
        suppressWarnings(suppressMessages(
            inla(
                .(case$formula),
                data = df,
                family = .(case$family),
                control.compute = list(config = FALSE),
                control.predictor = list(compute = TRUE),
                num.threads = 1
            )
        ))
    )

    rusty_perf <- track_perf(rusty_expr)
    rinla_perf <- track_perf(inla_expr)

    if (!rusty_perf$ok || !rinla_perf$ok || !inherits(rinla_perf$res, "inla")) {
        return(data.frame(
            case_id = case$id,
            label = case$label,
            source_type = case$source_type,
            output_profile = rusty_output_profile,
            available = TRUE,
            passed = FALSE,
            reason = paste(
                c(
                    if (!rusty_perf$ok) paste("rusty:", rusty_perf$error),
                    if (!rinla_perf$ok) paste("inla:", rinla_perf$error)
                ),
                collapse = " | "
            ),
            rusty_time = rusty_perf$time,
            inla_time = rinla_perf$time,
            rusty_mem = rusty_perf$mem,
            inla_mem = rinla_perf$mem,
            mlik_abs_diff = NA_real_,
            fixed_mean_max_abs = NA_real_,
            fixed_sd_max_abs = NA_real_,
            random_mean_max_abs = NA_real_,
            random_sd_max_abs = NA_real_,
            fitted_mean_max_rel = NA_real_,
            hyper_mean_max_abs = NA_real_,
            hyper_sd_max_abs = NA_real_,
            linear_predictor_mean_max_abs = NA_real_,
            linear_predictor_sd_max_abs = NA_real_,
            stringsAsFactors = FALSE
        ))
    }

    fixed <- compare_named_numeric(
        named_column(rusty_perf$res$summary.fixed, "mean"),
        named_column(rinla_perf$res$summary.fixed, "mean"),
        tolerances$fixed_mean_abs
    )
    fixed_sd <- compare_named_numeric(
        named_column(rusty_perf$res$summary.fixed, "sd"),
        named_column(rinla_perf$res$summary.fixed, "sd")
    )
    random_mean <- compare_nested_metrics(
        collect_random_metric(rusty_perf$res, "mean"),
        collect_random_metric(rinla_perf$res, "mean"),
        tolerances$random_mean_abs
    )
    random_sd <- compare_nested_metrics(
        collect_random_metric(rusty_perf$res, "sd"),
        collect_random_metric(rinla_perf$res, "sd"),
        tolerances$random_sd_abs
    )
    fitted <- compare_fitted_mean(
        rusty_perf$res,
        rinla_perf$res,
        tolerances$fitted_mean_rel
    )
    hyper_mean <- compare_named_numeric(
        indexed_column(rusty_perf$res$summary.hyperpar, "mean"),
        indexed_column(rinla_perf$res$summary.hyperpar, "mean")
    )
    hyper_sd <- compare_named_numeric(
        indexed_column(rusty_perf$res$summary.hyperpar, "sd"),
        indexed_column(rinla_perf$res$summary.hyperpar, "sd")
    )
    linear_predictor_mean <- compare_named_numeric(
        indexed_column(rusty_perf$res$summary.linear.predictor, "mean"),
        indexed_column(rinla_perf$res$summary.linear.predictor, "mean")
    )
    linear_predictor_sd <- compare_named_numeric(
        indexed_column(rusty_perf$res$summary.linear.predictor, "sd"),
        indexed_column(rinla_perf$res$summary.linear.predictor, "sd")
    )

    pass_flags <- c(fixed$pass, random_mean$pass, random_sd$pass, fitted$pass)
    pass_flags <- pass_flags[!is.na(pass_flags)]
    passed <- length(pass_flags) > 0 && all(pass_flags)

    data.frame(
        case_id = case$id,
        label = case$label,
        source_type = case$source_type,
        output_profile = rusty_output_profile,
        available = TRUE,
        passed = passed,
        reason = NA_character_,
        rusty_time = rusty_perf$time,
        inla_time = rinla_perf$time,
        rusty_mem = rusty_perf$mem,
        inla_mem = rinla_perf$mem,
        mlik_abs_diff = abs(extract_mlik(rusty_perf$res) - extract_mlik(rinla_perf$res)),
        fixed_mean_max_abs = fixed$max_abs,
        fixed_sd_max_abs = fixed_sd$max_abs,
        random_mean_max_abs = random_mean$max_abs,
        random_sd_max_abs = random_sd$max_abs,
        fitted_mean_max_rel = fitted$max_rel,
        hyper_mean_max_abs = hyper_mean$max_abs,
        hyper_sd_max_abs = hyper_sd$max_abs,
        linear_predictor_mean_max_abs = linear_predictor_mean$max_abs,
        linear_predictor_sd_max_abs = linear_predictor_sd$max_abs,
        stringsAsFactors = FALSE
    )
}

main <- function() {
    suppressPackageStartupMessages({
        library(INLA)
        library(rustyINLA)
    })

    cases <- prepare_cases(cache_dir)

    cat("Prepared case inventory:\n")
    inventory <- data.frame(
        case_id = vapply(cases, `[[`, character(1), "id"),
        source_type = vapply(cases, `[[`, character(1), "source_type"),
        available = vapply(cases, `[[`, logical(1), "available"),
        provenance = vapply(cases, `[[`, character(1), "provenance"),
        reason = vapply(cases, function(x) x$reason %||% NA_character_, character(1)),
        stringsAsFactors = FALSE
    )
    print(inventory, row.names = FALSE)

    cat(sprintf("\nCache directory: %s\n", normalizePath(cache_dir, winslash = "/", mustWork = FALSE)))
    cat(sprintf("Rusty-INLA output profile: %s\n", rusty_output_profile))
    cat("\nRunning available cases...\n")
    results <- do.call(rbind, lapply(cases, evaluate_case))
    print(results, row.names = FALSE)

    if (nzchar(external_output_path)) {
        utils::write.csv(results, external_output_path, row.names = FALSE)
        cat(sprintf("\nWrote external benchmark results to %s\n", external_output_path))
    }

    runnable <- results[results$available, , drop = FALSE]
    if (nrow(runnable) > 0) {
        cat(sprintf(
            "\nRunnable cases passed: %d/%d\n",
            sum(runnable$passed, na.rm = TRUE),
            nrow(runnable)
        ))
    }

    invisible(results)
}

main()
