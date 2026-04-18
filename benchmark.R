# Benchmark Harness for rustyINLA
# Compare rustyINLA with R-INLA on freMTPL2 examples using explicit tolerances.

local_rustyinla_lib <- Sys.getenv("RUSTYINLA_LIB", "")
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

`%||%` <- function(x, y) {
    if (is.null(x)) y else x
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

    diffs <- abs(lhs[shared] - rhs[shared])
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

    cat(sprintf("\nRunning %s...\n", case$label))
    rusty_perf <- track_perf(case$rusty_expr)
    rinla_perf <- track_perf(case$inla_expr)

    if (!rusty_perf$ok || !rinla_perf$ok || !inherits(rinla_perf$res, "inla")) {
        return(list(
            case_id = case$id,
            label = case$label,
            passed = FALSE,
            rusty_perf = rusty_perf,
            rinla_perf = rinla_perf,
            fixed = NULL,
            random_mean = NULL,
            random_sd = NULL,
            fitted = NULL,
            summary = data.frame(
                case_id = case$id,
                passed = FALSE,
                rusty_ok = rusty_perf$ok,
                inla_ok = rinla_perf$ok && inherits(rinla_perf$res, "inla"),
                rusty_time = rusty_perf$time,
                inla_time = rinla_perf$time,
                rusty_mem = rusty_perf$mem,
                inla_mem = rinla_perf$mem,
                rusty_mlik = NA_real_,
                inla_mlik = NA_real_,
                mlik_abs_diff = NA_real_,
                fixed_mean_max_abs = NA_real_,
                random_mean_max_abs = NA_real_,
                random_sd_max_abs = NA_real_,
                fitted_mean_max_rel = NA_real_,
                stringsAsFactors = FALSE
            )
        ))
    }

    fixed <- compare_named_numeric(
        named_column(rusty_perf$res$summary.fixed, "mean"),
        named_column(rinla_perf$res$summary.fixed, "mean"),
        tolerances$fixed_mean_abs
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

    pass_flags <- c(fixed$pass, random_mean$pass, random_sd$pass, fitted$pass)
    pass_flags <- pass_flags[!is.na(pass_flags)]
    passed <- length(pass_flags) > 0 && all(pass_flags)
    rusty_mlik <- extract_mlik(rusty_perf$res)
    inla_mlik <- extract_mlik(rinla_perf$res)

    list(
        case_id = case$id,
        label = case$label,
        passed = passed,
        rusty_perf = rusty_perf,
        rinla_perf = rinla_perf,
        fixed = fixed,
        random_mean = random_mean,
        random_sd = random_sd,
        fitted = fitted,
        summary = data.frame(
            case_id = case$id,
            passed = passed,
            rusty_ok = TRUE,
            inla_ok = TRUE,
            rusty_time = rusty_perf$time,
            inla_time = rinla_perf$time,
            rusty_mem = rusty_perf$mem,
            inla_mem = rinla_perf$mem,
            rusty_mlik = rusty_mlik,
            inla_mlik = inla_mlik,
            mlik_abs_diff = abs(rusty_mlik - inla_mlik),
            fixed_mean_max_abs = fixed$max_abs,
            random_mean_max_abs = random_mean$max_abs,
            random_sd_max_abs = random_sd$max_abs,
            fitted_mean_max_rel = fitted$max_rel,
            stringsAsFactors = FALSE
        )
    )
}

print_case_report <- function(result) {
    cat(sprintf(
        "\n==================================================\n%s\n==================================================\n",
        toupper(result$label)
    ))
    cat(sprintf(
        "Rusty-INLA: %6.2f sec | %6.2f MB\nR-INLA    : %6.2f sec | %6.2f MB\n",
        result$rusty_perf$time,
        result$rusty_perf$mem,
        result$rinla_perf$time,
        result$rinla_perf$mem
    ))

    if (!result$rusty_perf$ok || !result$rinla_perf$ok) {
        cat("Status     : FAILED TO RUN\n")
        if (!result$rusty_perf$ok) {
            cat(sprintf("Rusty error: %s\n", result$rusty_perf$error))
        }
        if (!result$rinla_perf$ok) {
            cat(sprintf("R-INLA err : %s\n", result$rinla_perf$error))
        }
        return(invisible(result))
    }

    cat(sprintf(
        "Marginal log-lik diff    : %s\n",
        format(
            round(
                abs(extract_mlik(result$rusty_perf$res) - extract_mlik(result$rinla_perf$res)),
                6
            ),
            nsmall = 6
        )
    ))

    cat(sprintf(
        "Fixed mean max abs diff : %s\n",
        format(round(result$fixed$max_abs, 6), nsmall = 6)
    ))
    cat(sprintf(
        "Random mean max abs diff: %s\n",
        format(round(result$random_mean$max_abs, 6), nsmall = 6)
    ))
    cat(sprintf(
        "Random sd max abs diff  : %s\n",
        format(round(result$random_sd$max_abs, 6), nsmall = 6)
    ))
    cat(sprintf(
        "Fitted mean max rel diff: %s\n",
        format(round(result$fitted$max_rel, 6), nsmall = 6)
    ))
    cat(sprintf("Status                  : %s\n", if (result$passed) "PASS" else "FAIL"))
    invisible(result)
}

cat("Loading datasets and constructing benchmark data...\n")
data(freMTPL2freq)
data(freMTPL2sev)

df_freq <- freMTPL2freq
df_freq$AgeGroup <- cut(
    df_freq$DrivAge,
    breaks = c(17, 25, 40, 60, 80, 150),
    labels = c("18-25", "26-40", "41-60", "61-80", "81+"),
    ordered_result = TRUE
)
df_freq$AgeIndex <- as.integer(df_freq$AgeGroup)
df_freq$VehBrand <- as.factor(df_freq$VehBrand)
df_freq$Region <- as.factor(df_freq$Region)

df_sev <- merge(
    freMTPL2sev,
    df_freq[, c("IDpol", "AgeGroup")],
    by = "IDpol",
    all.x = FALSE,
    all.y = FALSE
)
df_sev$AgeGroup <- factor(
    df_sev$AgeGroup,
    levels = levels(df_freq$AgeGroup),
    ordered = is.ordered(df_freq$AgeGroup)
)
sev_cutoff <- as.numeric(stats::quantile(
    df_sev$ClaimAmount[df_sev$ClaimAmount > 0],
    probs = 0.90,
    na.rm = TRUE
))
df_sev <- df_sev[df_sev$ClaimAmount > 0 & df_sev$ClaimAmount <= sev_cutoff, ]
cat(sprintf(
    "Gamma severity trim threshold (90th percentile): %.2f\n",
    sev_cutoff
))
cat("Tweedie benchmark excluded from the active parity sweep due to instability.\n")

cases <- list(
    list(
        id = "poisson_iid_offset",
        label = "Poisson Frequency + offset + iid(VehBrand)",
        rusty_expr = quote(
            rusty_inla(
                ClaimNb ~ 1 + offset(log(Exposure)) + f(VehBrand, model = "iid"),
                data = df_freq,
                family = "poisson"
            )
        ),
        inla_expr = quote(
            suppressWarnings(suppressMessages(
                inla(
                    ClaimNb ~ 1 + offset(log(Exposure)) + f(VehBrand, model = "iid"),
                    data = df_freq,
                    family = "poisson",
                    control.compute = list(config = FALSE),
                    control.predictor = list(compute = TRUE),
                    num.threads = 1
                )
            ))
        )
    ),
    list(
        id = "poisson_multi_iid_offset",
        label = "Poisson Frequency + offset + iid(VehBrand) + iid(Region)",
        rusty_expr = quote(
            rusty_inla(
                ClaimNb ~ 1 + offset(log(Exposure)) +
                    f(VehBrand, model = "iid") +
                    f(Region, model = "iid"),
                data = df_freq,
                family = "poisson"
            )
        ),
        inla_expr = quote(
            suppressWarnings(suppressMessages(
                inla(
                    ClaimNb ~ 1 + offset(log(Exposure)) +
                        f(VehBrand, model = "iid") +
                        f(Region, model = "iid"),
                    data = df_freq,
                    family = "poisson",
                    control.compute = list(config = FALSE),
                    control.predictor = list(compute = TRUE),
                    num.threads = 1
                )
            ))
        ),
        tolerances = list(
            random_mean_abs = 0.75,
            random_sd_abs = 0.75,
            fitted_mean_rel = 0.35
        )
    ),
    list(
        id = "gamma_rw1",
        label = "Gamma Severity (freMTPL2sev, top-10% trimmed) + rw1(AgeGroup)",
        rusty_expr = quote(
            rusty_inla(
                ClaimAmount ~ 1 + f(AgeGroup, model = "rw1"),
                data = df_sev,
                family = "gamma"
            )
        ),
        inla_expr = quote(
            suppressWarnings(suppressMessages(
                inla(
                    ClaimAmount ~ 1 + f(AgeGroup, model = "rw1"),
                    data = df_sev,
                    family = "gamma",
                    control.compute = list(config = FALSE),
                    control.predictor = list(compute = TRUE),
                    num.threads = 1
                )
            ))
        ),
        tolerances = list(
            fitted_mean_rel = 0.35
        )
    ),
    list(
        id = "poisson_ar1_offset",
        label = "Poisson Frequency + offset + ar1(AgeIndex)",
        rusty_expr = quote(
            rusty_inla(
                ClaimNb ~ 1 + offset(log(Exposure)) + f(AgeIndex, model = "ar1"),
                data = df_freq,
                family = "poisson"
            )
        ),
        inla_expr = quote(
            suppressWarnings(suppressMessages(
                inla(
                    ClaimNb ~ 1 + offset(log(Exposure)) + f(AgeIndex, model = "ar1"),
                    data = df_freq,
                    family = "poisson",
                    control.compute = list(config = FALSE),
                    control.predictor = list(compute = TRUE),
                    num.threads = 1
                )
            ))
        ),
        tolerances = list(
            random_mean_abs = 0.75,
            random_sd_abs = 0.75,
            fitted_mean_rel = 0.35
        )
    ),
    list(
        id = "zip_iid_offset",
        label = "Zero-Inflated Poisson + offset + iid(VehBrand)",
        rusty_expr = quote(
            rusty_inla(
                ClaimNb ~ 1 + offset(log(Exposure)) + f(VehBrand, model = "iid"),
                data = df_freq,
                family = "zeroinflatedpoisson1"
            )
        ),
        inla_expr = quote(
            suppressWarnings(suppressMessages(
                inla(
                    ClaimNb ~ 1 + offset(log(Exposure)) + f(VehBrand, model = "iid"),
                    data = df_freq,
                    family = "zeroinflatedpoisson1",
                    control.compute = list(config = FALSE),
                    control.predictor = list(compute = TRUE),
                    num.threads = 1
                )
            ))
        ),
        tolerances = list(
            random_mean_abs = 1.00,
            random_sd_abs = 1.00,
            fitted_mean_rel = 0.50
        )
    )
)

cat("\nExecuting parity benchmark cases...\n")
results <- lapply(cases, evaluate_case)
invisible(lapply(results, print_case_report))

summary_table <- do.call(
    rbind,
    lapply(results, function(result) result$summary)
)

cat("\n====================\nPARITY SUMMARY TABLE\n====================\n")
print(summary_table, row.names = FALSE)

cat(sprintf(
    "\nOverall: %d/%d cases passed.\n",
    sum(summary_table$passed, na.rm = TRUE),
    nrow(summary_table)
))
