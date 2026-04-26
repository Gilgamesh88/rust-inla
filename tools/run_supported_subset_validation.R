#!/usr/bin/env Rscript

# Curated supported-subset validation for Phase 7A.
#
# This script intentionally does not source or run the uploaded INLA suites.
# It maps the supported-subset manifest into small deterministic cases that
# exercise the same formula surfaces against rustyINLA and R-INLA.

repo_root <- normalizePath(
    Sys.getenv("RUSTYINLA_REPO_ROOT", getwd()),
    winslash = "/",
    mustWork = TRUE
)

rusty_output_profile <- Sys.getenv("RUSTYINLA_OUTPUT_PROFILE", "thin")
if (!(rusty_output_profile %in% c("thin", "benchmark"))) {
    stop("RUSTYINLA_OUTPUT_PROFILE must be 'thin' or 'benchmark'.", call. = FALSE)
}

supported_output_path <- Sys.getenv("RUSTYINLA_SUPPORTED_SUBSET_OUT", "")

`%||%` <- function(x, y) {
    if (is.null(x)) y else x
}

ensure_inla <- function() {
    if (!requireNamespace("INLA", quietly = TRUE)) {
        stop("Package 'INLA' is required for supported-subset validation.", call. = FALSE)
    }
    suppressPackageStartupMessages(library(INLA))
}

track_fit <- function(expr_sub, envir = parent.frame()) {
    t0 <- proc.time()[["elapsed"]]
    res <- tryCatch(
        eval(expr_sub, envir = envir),
        error = function(e) structure(
            list(message = conditionMessage(e)),
            class = "supported_subset_error"
        )
    )
    elapsed <- proc.time()[["elapsed"]] - t0

    list(
        res = res,
        ok = !inherits(res, "supported_subset_error"),
        error = if (inherits(res, "supported_subset_error")) res$message else NA_character_,
        time = unname(elapsed)
    )
}

named_column <- function(df, column) {
    if (is.null(df) || nrow(df) == 0L || !(column %in% names(df))) {
        return(stats::setNames(numeric(), character()))
    }
    stats::setNames(as.numeric(df[[column]]), rownames(df))
}

collect_random_metric <- function(fit, metric) {
    if (is.null(fit$summary.random) || length(fit$summary.random) == 0L) {
        return(list())
    }

    out <- lapply(fit$summary.random, function(df) {
        if (is.null(df) || nrow(df) == 0L || !(metric %in% names(df))) {
            return(stats::setNames(numeric(), character()))
        }
        ids <- if ("ID" %in% names(df)) as.character(df$ID) else rownames(df)
        stats::setNames(as.numeric(df[[metric]]), ids)
    })
    out[order(names(out))]
}

compare_named_numeric <- function(lhs, rhs, abs_tol = NULL) {
    shared <- intersect(names(lhs), names(rhs))
    if (length(shared) == 0L) {
        return(list(n = 0L, max_abs = NA_real_, pass = NA))
    }

    lhs_vals <- as.numeric(lhs[shared])
    rhs_vals <- as.numeric(rhs[shared])
    keep <- is.finite(lhs_vals) & is.finite(rhs_vals)
    if (!any(keep)) {
        return(list(n = 0L, max_abs = NA_real_, pass = NA))
    }

    diffs <- abs(lhs_vals[keep] - rhs_vals[keep])
    list(
        n = length(diffs),
        max_abs = max(diffs),
        pass = if (is.null(abs_tol)) NA else max(diffs) <= abs_tol
    )
}

compare_nested_metrics <- function(lhs_list, rhs_list, abs_tol) {
    shared_terms <- intersect(names(lhs_list), names(rhs_list))
    if (length(shared_terms) == 0L) {
        return(list(n = 0L, max_abs = NA_real_, pass = NA))
    }

    diffs <- numeric()
    for (term in shared_terms) {
        lhs <- lhs_list[[term]]
        rhs <- rhs_list[[term]]
        shared_ids <- intersect(names(lhs), names(rhs))
        if (length(shared_ids) > 0L) {
            diffs <- c(diffs, abs(lhs[shared_ids] - rhs[shared_ids]))
        } else if (length(lhs) == length(rhs) && length(lhs) > 0L) {
            diffs <- c(diffs, abs(unname(lhs) - unname(rhs)))
        }
    }

    if (length(diffs) == 0L) {
        return(list(n = 0L, max_abs = NA_real_, pass = NA))
    }
    diffs <- diffs[is.finite(diffs)]
    if (length(diffs) == 0L) {
        return(list(n = 0L, max_abs = NA_real_, pass = NA))
    }

    list(n = length(diffs), max_abs = max(diffs), pass = max(diffs) <= abs_tol)
}

compare_fitted_mean <- function(rusty_fit, inla_fit, rel_tol) {
    rusty_fitted <- rusty_fit$summary.fitted.values
    inla_fitted <- inla_fit$summary.fitted.values
    if (is.null(rusty_fitted) || is.null(inla_fitted)) {
        return(list(n = 0L, max_rel = NA_real_, pass = NA))
    }

    n_shared <- min(nrow(rusty_fitted), nrow(inla_fitted))
    if (n_shared == 0L) {
        return(list(n = 0L, max_rel = NA_real_, pass = NA))
    }

    rusty_mean <- as.numeric(rusty_fitted$mean[seq_len(n_shared)])
    inla_mean <- as.numeric(inla_fitted$mean[seq_len(n_shared)])
    keep <- is.finite(rusty_mean) & is.finite(inla_mean)
    if (!any(keep)) {
        return(list(n = 0L, max_rel = NA_real_, pass = NA))
    }

    rel_diff <- abs(rusty_mean[keep] - inla_mean[keep]) / pmax(1.0, abs(inla_mean[keep]))
    list(n = length(rel_diff), max_rel = max(rel_diff), pass = max(rel_diff) <= rel_tol)
}

case_record <- function(id, manifest_source, family, formula, data, tolerances = list()) {
    list(
        id = id,
        manifest_source = manifest_source,
        family = family,
        formula = formula,
        data = data,
        tolerances = modifyList(
            list(
                fixed_mean_abs = 0.35,
                random_mean_abs = 0.75,
                random_sd_abs = 0.75,
                fitted_mean_rel = 0.35
            ),
            tolerances
        )
    )
}

make_fixed_only_poisson_offset <- function(seed = 1701L) {
    set.seed(seed)
    n <- 120L
    x1 <- stats::rnorm(n)
    x2 <- stats::runif(n, -1.0, 1.0)
    promo <- factor(sample(c("base", "promo"), n, replace = TRUE))
    exposure <- stats::runif(n, 0.5, 2.0)
    eta <- -0.35 + 0.45 * x1 - 0.25 * x2 + 0.35 * (promo == "promo") -
        0.20 * x1 * (promo == "promo") + log(exposure)
    data.frame(
        y = stats::rpois(n, lambda = exp(eta)),
        x1 = x1,
        x2 = x2,
        promo = promo,
        exposure = exposure
    )
}

make_poisson_iid_fixed <- function(seed = 1702L) {
    set.seed(seed)
    n_groups <- 16L
    reps <- 8L
    n <- n_groups * reps
    group <- factor(rep(seq_len(n_groups), each = reps))
    x1 <- stats::rnorm(n)
    promo <- factor(sample(c("base", "promo"), n, replace = TRUE))
    exposure <- stats::runif(n, 0.4, 1.8)
    u <- stats::rnorm(n_groups, 0.0, 0.30)
    eta <- -0.55 + 0.30 * x1 + 0.25 * (promo == "promo") +
        u[as.integer(group)] + log(exposure)
    data.frame(
        y = stats::rpois(n, lambda = exp(eta)),
        x1 = x1,
        promo = promo,
        exposure = exposure,
        group = group
    )
}

make_poisson_two_iid <- function(seed = 1703L) {
    set.seed(seed)
    n_group <- 12L
    n_area <- 5L
    reps <- 10L
    n <- n_group * reps
    group <- factor(rep(seq_len(n_group), each = reps))
    area <- factor(rep(seq_len(n_area), length.out = n))
    promo <- factor(sample(c("base", "promo"), n, replace = TRUE))
    exposure <- stats::runif(n, 0.5, 2.3)
    u_group <- stats::rnorm(n_group, 0.0, 0.25)
    u_area <- stats::rnorm(n_area, 0.0, 0.20)
    eta <- -0.70 + 0.35 * (promo == "promo") +
        u_group[as.integer(group)] + u_area[as.integer(area)] + log(exposure)
    data.frame(
        y = stats::rpois(n, lambda = exp(eta)),
        promo = promo,
        exposure = exposure,
        group = group,
        area = area
    )
}

make_gamma_iid_fixed <- function(seed = 1704L) {
    set.seed(seed)
    n_groups <- 15L
    reps <- 7L
    n <- n_groups * reps
    group <- factor(rep(seq_len(n_groups), each = reps))
    x1 <- stats::rnorm(n)
    promo <- factor(sample(c("base", "promo"), n, replace = TRUE))
    u <- stats::rnorm(n_groups, 0.0, 0.18)
    mu <- exp(1.20 + 0.25 * x1 + 0.20 * (promo == "promo") + u[as.integer(group)])
    shape <- 14.0
    data.frame(
        y = stats::rgamma(n, shape = shape, rate = shape / mu),
        x1 = x1,
        promo = promo,
        group = group
    )
}

make_gaussian_ar1_fixed <- function(seed = 1705L) {
    set.seed(seed)
    n <- 90L
    x1 <- stats::rnorm(n)
    latent <- as.numeric(stats::arima.sim(model = list(ar = 0.65), n = n, sd = 0.25))
    data.frame(
        y = 0.55 - 0.30 * x1 + latent + stats::rnorm(n, sd = 0.15),
        x1 = x1,
        time = seq_len(n)
    )
}

load_inla_dataset <- function(name) {
    env <- new.env(parent = emptyenv())
    utils::data(list = name, package = "INLA", envir = env)
    if (!exists(name, envir = env, inherits = FALSE)) {
        stop(sprintf("INLA dataset '%s' was not found.", name), call. = FALSE)
    }
    get(name, envir = env, inherits = FALSE)
}

make_suite_part1_germany <- function() {
    Germany <- load_inla_dataset("Germany")
    data.frame(
        Y = Germany$Y,
        E = Germany$E,
        x = Germany$x,
        region_iid = factor(Germany$region)
    )
}

make_suite_part1_epil <- function() {
    Epil <- load_inla_dataset("Epil")
    data.frame(
        y = Epil$y,
        Trt = Epil$Trt,
        Base = Epil$Base,
        Age = Epil$Age,
        V4 = Epil$V4,
        Ind = factor(Epil$Ind),
        rand = factor(Epil$rand)
    )
}

make_suite_part2_fremtpl_synthetic <- function(seed = 2701L, n = 3000L) {
    set.seed(seed)
    df <- data.frame(
        IDpol = seq_len(n),
        ClaimNb = stats::rpois(n, 0.05),
        Exposure = stats::runif(n, 0.1, 1.0),
        Area = sample(
            LETTERS[1:6],
            n,
            replace = TRUE,
            prob = c(0.15, 0.2, 0.25, 0.2, 0.1, 0.1)
        ),
        VehPower = sample(4:15, n, replace = TRUE),
        VehAge = sample(0:20, n, replace = TRUE),
        DrivAge = sample(
            18:90,
            n,
            replace = TRUE,
            prob = stats::dnorm(18:90, mean = 45, sd = 15)
        ),
        BonusMalus = pmax(50, pmin(230, stats::rnorm(n, 100, 20))),
        VehBrand = sample(paste0("B", 1:11), n, replace = TRUE),
        VehGas = sample(c("Regular", "Diesel"), n, replace = TRUE, prob = c(0.4, 0.6)),
        Density = round(exp(stats::rnorm(n, 5, 1.5))),
        Region = sample(paste0("R", 1:22), n, replace = TRUE)
    )

    df$ClaimNb <- pmin(df$ClaimNb, 4)
    df$Exposure <- pmin(df$Exposure, 1.0)
    has_claim <- df$ClaimNb > 0
    df$TotalClaim <- 0.0
    df$TotalClaim[has_claim] <- stats::rgamma(
        sum(has_claim),
        shape = 2,
        rate = 2 / 2000
    ) * df$ClaimNb[has_claim]
    df$NumClaims <- df$ClaimNb

    df$Area <- factor(df$Area)
    df$VehPower <- factor(pmin(as.numeric(as.character(df$VehPower)), 12))
    df$VehBrand <- factor(df$VehBrand)
    df$Region <- factor(df$Region)
    df$VehGas <- factor(df$VehGas)
    df$log_exposure <- log(df$Exposure)
    df$log_density <- log(df$Density + 1)
    df$obs_id <- seq_len(nrow(df))
    df$HasClaim <- as.integer(df$ClaimNb > 0)
    df$Area_num <- as.integer(df$Area)
    df$Region_num <- as.integer(df$Region)
    df
}

make_suite_part3_scale <- function(seed = 3701L, n = 500L) {
    set.seed(seed)
    x1 <- stats::rnorm(n)
    x2 <- stats::rnorm(n)
    group <- sample(seq_len(min(50L, n %/% 10L)), n, replace = TRUE)
    time <- rep(seq_len(min(100L, n %/% 10L)), length.out = n)
    lambda <- exp(1 + 0.3 * x1 - 0.2 * x2)
    mu_gamma <- exp(5 + 0.2 * x1)

    data.frame(
        y_pois = stats::rpois(n, lambda),
        y_gamma = stats::rgamma(n, shape = 2, rate = 2 / mu_gamma),
        x1 = x1,
        x2 = x2,
        group = factor(group),
        time = time,
        obs_id = seq_len(n)
    )
}

build_cases <- function() {
    suite_part2 <- make_suite_part2_fremtpl_synthetic()
    suite_part2_sev <- suite_part2[suite_part2$TotalClaim > 0, , drop = FALSE]
    suite_part2_sev$sev_id <- seq_len(nrow(suite_part2_sev))

    list(
        case_record(
            id = "suite_part1_germany_poisson_glm",
            manifest_source = "uploaded part1: Germany_Poisson_GLM adapted E to offset",
            family = "poisson",
            formula = Y ~ 1 + x + offset(log(E)),
            data = make_suite_part1_germany(),
            tolerances = list(fixed_mean_abs = 0.10, fitted_mean_rel = 0.10)
        ),
        case_record(
            id = "suite_part1_germany_poisson_iid",
            manifest_source = "uploaded part1: Germany_Poisson_IID adapted E to offset",
            family = "poisson",
            formula = Y ~ 1 + x + offset(log(E)) + f(region_iid, model = "iid"),
            data = make_suite_part1_germany(),
            tolerances = list(fixed_mean_abs = 0.15, random_mean_abs = 0.80, random_sd_abs = 0.80)
        ),
        case_record(
            id = "suite_part1_epil_poisson_glm",
            manifest_source = "uploaded part1: Epil_Poisson_GLM",
            family = "poisson",
            formula = y ~ 1 + Trt + Base + Age + V4,
            data = make_suite_part1_epil(),
            tolerances = list(fixed_mean_abs = 0.20, fitted_mean_rel = 0.20)
        ),
        case_record(
            id = "suite_part1_epil_poisson_iid",
            manifest_source = "uploaded part1: Epil_Poisson_IID",
            family = "poisson",
            formula = y ~ 1 + Trt + Base + Age + V4 + f(Ind, model = "iid"),
            data = make_suite_part1_epil(),
            tolerances = list(fixed_mean_abs = 0.25, random_mean_abs = 0.75, random_sd_abs = 0.75)
        ),
        case_record(
            id = "suite_part2_f01_poisson_offset",
            manifest_source = "uploaded part2: F01_Poisson_Offset synthetic fallback",
            family = "poisson",
            formula = ClaimNb ~ 1 + DrivAge + VehAge + BonusMalus +
                log_density + offset(log_exposure),
            data = suite_part2,
            tolerances = list(fixed_mean_abs = 0.25, fitted_mean_rel = 0.20)
        ),
        case_record(
            id = "suite_part2_s01_gamma_glm",
            manifest_source = "uploaded part2: S01_Gamma_GLM synthetic fallback",
            family = "gamma",
            formula = TotalClaim ~ 1 + DrivAge + VehAge + BonusMalus + log_density,
            data = suite_part2_sev,
            tolerances = list(fixed_mean_abs = 0.35, fitted_mean_rel = 0.25)
        ),
        case_record(
            id = "suite_part3_scale_poisson_glm",
            manifest_source = "uploaded part3: Scale_*_Poisson_GLM",
            family = "poisson",
            formula = y_pois ~ 1 + x1 + x2,
            data = make_suite_part3_scale(),
            tolerances = list(fixed_mean_abs = 0.12, fitted_mean_rel = 0.10)
        ),
        case_record(
            id = "fixed_only_poisson_offset_interaction",
            manifest_source = "new fixed-only GLM subset decision",
            family = "poisson",
            formula = y ~ 1 + x1 * promo + x2 + offset(log(exposure)),
            data = make_fixed_only_poisson_offset(),
            tolerances = list(fixed_mean_abs = 0.25, fitted_mean_rel = 0.20)
        ),
        case_record(
            id = "part1_epil_style_poisson_iid_fixed",
            manifest_source = "part1: Epil_Poisson_IID",
            family = "poisson",
            formula = y ~ 1 + x1 + promo + offset(log(exposure)) + f(group, model = "iid"),
            data = make_poisson_iid_fixed()
        ),
        case_record(
            id = "part1_epil_style_poisson_two_iid",
            manifest_source = "part1: Epil_Poisson_IID2 / stress: MultiRE_2IID",
            family = "poisson",
            formula = y ~ 1 + promo + offset(log(exposure)) +
                f(group, model = "iid") + f(area, model = "iid"),
            data = make_poisson_two_iid(),
            tolerances = list(random_mean_abs = 1.00, random_sd_abs = 1.00, fitted_mean_rel = 0.45)
        ),
        case_record(
            id = "fremtpl_style_gamma_iid_fixed",
            manifest_source = "freMTPL2: S02_Gamma_IID_Area",
            family = "gamma",
            formula = y ~ 1 + x1 + promo + f(group, model = "iid"),
            data = make_gamma_iid_fixed(),
            tolerances = list(fixed_mean_abs = 0.45, fitted_mean_rel = 0.30)
        ),
        case_record(
            id = "stress_timeseries_gaussian_ar1_fixed",
            manifest_source = "stress: TimeSeries_*_AR1",
            family = "gaussian",
            formula = y ~ 1 + x1 + f(time, model = "ar1", constr = FALSE),
            data = make_gaussian_ar1_fixed(),
            tolerances = list(fixed_mean_abs = 0.25, random_mean_abs = 0.50, random_sd_abs = 0.50)
        )
    )
}

evaluate_case <- function(case) {
    df <- case$data
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
            INLA::inla(
                .(case$formula),
                data = df,
                family = .(case$family),
                control.compute = list(config = FALSE),
                control.predictor = list(compute = TRUE),
                num.threads = 1
            )
        ))
    )

    rusty_perf <- track_fit(rusty_expr)
    inla_perf <- track_fit(inla_expr)

    if (!rusty_perf$ok || !inla_perf$ok || !inherits(inla_perf$res, "inla")) {
        return(data.frame(
            case_id = case$id,
            manifest_source = case$manifest_source,
            family = case$family,
            passed = FALSE,
            reason = paste(
                c(
                    if (!rusty_perf$ok) paste("rusty:", rusty_perf$error),
                    if (!inla_perf$ok) paste("inla:", inla_perf$error)
                ),
                collapse = " | "
            ),
            rusty_time = rusty_perf$time,
            inla_time = inla_perf$time,
            fixed_mean_max_abs = NA_real_,
            random_mean_max_abs = NA_real_,
            random_sd_max_abs = NA_real_,
            fitted_mean_max_rel = NA_real_,
            stringsAsFactors = FALSE
        ))
    }

    fixed <- compare_named_numeric(
        named_column(rusty_perf$res$summary.fixed, "mean"),
        named_column(inla_perf$res$summary.fixed, "mean"),
        case$tolerances$fixed_mean_abs
    )
    random_mean <- compare_nested_metrics(
        collect_random_metric(rusty_perf$res, "mean"),
        collect_random_metric(inla_perf$res, "mean"),
        case$tolerances$random_mean_abs
    )
    random_sd <- compare_nested_metrics(
        collect_random_metric(rusty_perf$res, "sd"),
        collect_random_metric(inla_perf$res, "sd"),
        case$tolerances$random_sd_abs
    )
    fitted <- compare_fitted_mean(
        rusty_perf$res,
        inla_perf$res,
        case$tolerances$fitted_mean_rel
    )

    pass_flags <- c(fixed$pass, random_mean$pass, random_sd$pass, fitted$pass)
    pass_flags <- pass_flags[!is.na(pass_flags)]
    passed <- length(pass_flags) > 0L && all(pass_flags)

    data.frame(
        case_id = case$id,
        manifest_source = case$manifest_source,
        family = case$family,
        passed = passed,
        reason = NA_character_,
        rusty_time = rusty_perf$time,
        inla_time = inla_perf$time,
        fixed_mean_max_abs = fixed$max_abs,
        random_mean_max_abs = random_mean$max_abs,
        random_sd_max_abs = random_sd$max_abs,
        fitted_mean_max_rel = fitted$max_rel,
        stringsAsFactors = FALSE
    )
}

main <- function() {
    ensure_inla()

    source(file.path(repo_root, "tools", "load_worktree_package.R"), local = TRUE)
    load_rustyinla_for_benchmarks(repo_root)

    cases <- build_cases()
    inventory <- data.frame(
        case_id = vapply(cases, `[[`, character(1), "id"),
        manifest_source = vapply(cases, `[[`, character(1), "manifest_source"),
        family = vapply(cases, `[[`, character(1), "family"),
        stringsAsFactors = FALSE
    )

    cat("Supported-subset validation inventory:\n")
    print(inventory, row.names = FALSE)
    cat(sprintf("\nRusty-INLA output profile: %s\n", rusty_output_profile))
    cat("Running curated cases against R-INLA...\n")

    results <- do.call(rbind, lapply(cases, evaluate_case))
    print(results, row.names = FALSE)

    if (nzchar(supported_output_path)) {
        utils::write.csv(results, supported_output_path, row.names = FALSE)
        cat(sprintf("\nWrote supported-subset results to %s\n", supported_output_path))
    }

    cat(sprintf(
        "\nCurated supported-subset cases passed: %d/%d\n",
        sum(results$passed, na.rm = TRUE),
        nrow(results)
    ))

    invisible(results)
}

main()
