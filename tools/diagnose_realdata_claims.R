args <- commandArgs(trailingOnly = TRUE)

has_arg <- function(flag) {
    any(args == flag)
}

arg_value <- function(flag, default = NULL) {
    prefix <- paste0(flag, "=")
    inline <- args[startsWith(args, prefix)]
    if (length(inline) > 0L) {
        return(substring(inline[[length(inline)]], nchar(prefix) + 1L))
    }
    idx <- which(args == flag)
    if (length(idx) > 0L && idx[[length(idx)]] < length(args)) {
        return(args[[idx[[length(idx)]] + 1L]])
    }
    default
}

csv_arg <- function(flag, default) {
    value <- arg_value(flag, default)
    value <- strsplit(value, ",", fixed = TRUE)[[1L]]
    trimws(value[nzchar(trimws(value))])
}

repo_root <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
data_path <- arg_value("--data", file.path(repo_root, "data", "real", "inla_sbi_test.RDS"))
if (!grepl("^[A-Za-z]:[/\\\\]|^/", data_path)) {
    data_path <- file.path(repo_root, data_path)
}
data_path <- normalizePath(data_path, winslash = "/", mustWork = TRUE)

report_dir <- arg_value(
    "--report-dir",
    file.path(repo_root, "scratch", "phase8_reports", "realdata_hull_claim_diagnostics")
)
if (!grepl("^[A-Za-z]:[/\\\\]|^/", report_dir)) {
    report_dir <- file.path(repo_root, report_dir)
}
dir.create(report_dir, recursive = TRUE, showWarnings = FALSE)
report_dir <- normalizePath(report_dir, winslash = "/", mustWork = TRUE)

period_col <- arg_value("--period-col", "periodo")
response_col <- arg_value("--response-col", "pt")
exposure_col <- arg_value("--exposure-col", "expuesto")
iid_col <- arg_value("--iid-col", "desc_armadora")
fixed_effects <- csv_arg("--fixed", "modeloc,desc_edo_circula,medio_emit")
drop_low_exposure_col <- arg_value("--drop-low-exposure-col", "")
drop_low_exposure_n <- as.integer(arg_value("--drop-low-exposure-n", "0"))
explicit_drop_levels <- csv_arg("--drop-levels", "")

write_report <- function(x, name) {
    path <- file.path(report_dir, name)
    utils::write.csv(x, path, row.names = FALSE, na = "")
    path
}

as_number <- function(x) {
    if (is.numeric(x)) {
        return(as.numeric(x))
    }
    suppressWarnings(as.numeric(as.character(x)))
}

period_key <- function(x) {
    x <- as.character(x)
    out <- rep(Inf, length(x))
    year_only <- grepl("^\\d{4}_y$", x)
    if (any(year_only)) {
        year <- as.integer(sub("_y$", "", x[year_only]))
        out[year_only] <- year * 10
    }
    quarter <- grepl("^\\d{4}_[1-4]t$", x)
    if (any(quarter)) {
        year <- as.integer(substr(x[quarter], 1L, 4L))
        q <- as.integer(sub("^\\d{4}_([1-4])t$", "\\1", x[quarter]))
        out[quarter] <- year * 10 + q
    }
    month <- grepl("^\\d{4}_(0?[1-9]|1[0-2])m$", x)
    if (any(month)) {
        year <- as.integer(substr(x[month], 1L, 4L))
        m <- as.integer(sub("^\\d{4}_(0?[1-9]|1[0-2])m$", "\\1", x[month]))
        out[month] <- year * 100 + m
    }
    out
}

ordered_periods <- function(x) {
    x <- unique(as.character(x))
    x[order(period_key(x), x, na.last = TRUE)]
}

summarize_periods <- function(dt, label, period_col, response_col, exposure_col) {
    periods <- ordered_periods(dt[[period_col]])
    out <- vector("list", length(periods))
    for (idx in seq_along(periods)) {
        p <- periods[[idx]]
        rows <- dt[[period_col]] == p
        y <- dt[[response_col]][rows]
        e <- dt[[exposure_col]][rows]
        positive <- is.finite(y) & y > 0
        out[[idx]] <- data.frame(
            slice = label,
            period = p,
            rows = sum(rows),
            exposure = sum(e, na.rm = TRUE),
            claims = sum(y, na.rm = TRUE),
            frequency = sum(y, na.rm = TRUE) / max(sum(e, na.rm = TRUE), 1e-12),
            positive_claim_rows = sum(positive),
            positive_claim_exposure = sum(e[positive], na.rm = TRUE),
            zero_claim_rows = sum(is.finite(y) & y == 0),
            missing_response_rows = sum(!is.finite(y)),
            missing_or_nonpositive_exposure_rows = sum(!is.finite(e) | e <= 0),
            max_claim_count = suppressWarnings(max(y, na.rm = TRUE)),
            mean_exposure = mean(e, na.rm = TRUE),
            median_exposure = stats::median(e, na.rm = TRUE),
            max_exposure = suppressWarnings(max(e, na.rm = TRUE)),
            check.names = FALSE
        )
    }
    do.call(rbind, out)
}

summarize_by_group <- function(dt, group_col, period_col, response_col, exposure_col, label) {
    if (!(group_col %in% names(dt))) {
        return(data.frame())
    }
    groups <- sort(unique(as.character(dt[[group_col]])))
    periods <- ordered_periods(dt[[period_col]])
    rows <- vector("list", length(groups) * length(periods))
    pos <- 0L
    for (period in periods) {
        period_rows <- dt[[period_col]] == period
        for (group in groups) {
            idx <- period_rows & as.character(dt[[group_col]]) == group
            if (!any(idx)) {
                next
            }
            pos <- pos + 1L
            e <- dt[[exposure_col]][idx]
            y <- dt[[response_col]][idx]
            rows[[pos]] <- data.frame(
                slice = label,
                period = period,
                group_col = group_col,
                group = group,
                rows = sum(idx),
                exposure = sum(e, na.rm = TRUE),
                claims = sum(y, na.rm = TRUE),
                frequency = sum(y, na.rm = TRUE) / max(sum(e, na.rm = TRUE), 1e-12),
                positive_claim_rows = sum(is.finite(y) & y > 0),
                check.names = FALSE
            )
        }
    }
    do.call(rbind, rows[seq_len(pos)])
}

raw <- readRDS(data_path)
if (!is.data.frame(raw)) {
    raw <- as.data.frame(raw)
}
dt <- as.data.frame(raw)

required_cols <- unique(c(period_col, response_col, exposure_col, iid_col, fixed_effects))
missing_cols <- setdiff(required_cols, names(dt))
if (length(missing_cols) > 0L) {
    stop(sprintf("Missing required column(s): %s", paste(missing_cols, collapse = ", ")), call. = FALSE)
}

dt[[period_col]] <- as.character(dt[[period_col]])
dt[[response_col]] <- as_number(dt[[response_col]])
dt[[exposure_col]] <- as_number(dt[[exposure_col]])

valid_exposure <- is.finite(dt[[exposure_col]]) & dt[[exposure_col]] > 0
valid_response <- is.finite(dt[[response_col]]) & dt[[response_col]] >= 0
valid_period <- !is.na(dt[[period_col]]) & nzchar(dt[[period_col]])
valid_iid <- !is.na(dt[[iid_col]]) & nzchar(as.character(dt[[iid_col]]))
valid_fixed <- rep(TRUE, nrow(dt))
for (col in fixed_effects) {
    valid_fixed <- valid_fixed & !is.na(dt[[col]]) & nzchar(as.character(dt[[col]]))
}
base_model_rows <- valid_exposure & valid_response & valid_period & valid_iid & valid_fixed

drop_level_table <- data.frame(
    column = character(),
    level = character(),
    exposure = numeric(),
    rows = integer(),
    claims = numeric(),
    check.names = FALSE
)
if (nzchar(drop_low_exposure_col)) {
    level_values <- as.character(dt[[drop_low_exposure_col]])
    levels <- sort(unique(level_values[base_model_rows]))
    level_table <- do.call(rbind, lapply(levels, function(level) {
        idx <- base_model_rows & level_values == level
        data.frame(
            column = drop_low_exposure_col,
            level = level,
            exposure = sum(dt[[exposure_col]][idx], na.rm = TRUE),
            rows = sum(idx),
            claims = sum(dt[[response_col]][idx], na.rm = TRUE),
            check.names = FALSE
        )
    }))
    level_table <- level_table[order(level_table$exposure, level_table$rows, level_table$level), , drop = FALSE]
    selected <- character()
    if (is.finite(drop_low_exposure_n) && drop_low_exposure_n > 0L) {
        selected <- union(selected, utils::head(level_table$level, drop_low_exposure_n))
    }
    selected <- union(selected, explicit_drop_levels)
    selected <- intersect(selected, level_table$level)
    if (length(selected) > 0L) {
        drop_level_table <- level_table[match(selected, level_table$level), , drop = FALSE]
    }
}

drop_rows <- rep(FALSE, nrow(dt))
if (nrow(drop_level_table) > 0L) {
    for (idx in seq_len(nrow(drop_level_table))) {
        drop_rows <- drop_rows |
            as.character(dt[[drop_level_table$column[[idx]]]]) == drop_level_table$level[[idx]]
    }
}
clean_rows <- base_model_rows & !drop_rows

candidate_name_pattern <- paste(
    c("fecha", "date", "period", "periodo", "claim", "sini", "reclam", "pago", "ocurr", "report", "cobertura"),
    collapse = "|"
)
candidate_columns <- names(dt)[grepl(candidate_name_pattern, names(dt), ignore.case = TRUE)]

column_profile <- data.frame(
    column = names(dt),
    class = vapply(dt, function(x) paste(class(x), collapse = "|"), character(1)),
    missing_rows = vapply(dt, function(x) sum(is.na(x)), integer(1)),
    distinct_values = vapply(dt, function(x) length(unique(x)), integer(1)),
    candidate_time_or_claim_column = names(dt) %in% candidate_columns,
    check.names = FALSE
)

response <- dt[[response_col]]
exposure <- dt[[exposure_col]]
response_profile <- data.frame(
    metric = c(
        "rows",
        "response_sum",
        "response_positive_rows",
        "response_zero_rows",
        "response_missing_rows",
        "response_negative_rows",
        "response_non_integer_rows",
        "response_max",
        "exposure_sum",
        "exposure_positive_rows",
        "exposure_missing_or_nonpositive_rows"
    ),
    value = c(
        nrow(dt),
        sum(response, na.rm = TRUE),
        sum(is.finite(response) & response > 0),
        sum(is.finite(response) & response == 0),
        sum(!is.finite(response)),
        sum(is.finite(response) & response < 0),
        sum(is.finite(response) & abs(response - round(response)) > 1e-8),
        suppressWarnings(max(response, na.rm = TRUE)),
        sum(exposure, na.rm = TRUE),
        sum(is.finite(exposure) & exposure > 0),
        sum(!is.finite(exposure) | exposure <= 0)
    ),
    check.names = FALSE
)

raw_period <- summarize_periods(dt[valid_period, , drop = FALSE], "raw_valid_period", period_col, response_col, exposure_col)
base_period <- summarize_periods(dt[base_model_rows, , drop = FALSE], "base_model_rows", period_col, response_col, exposure_col)
clean_period <- summarize_periods(dt[clean_rows, , drop = FALSE], "clean_after_drops", period_col, response_col, exposure_col)
period_profile <- rbind(raw_period, base_period, clean_period)

period_profile$claims_prev_ratio <- NA_real_
period_profile$frequency_prev_ratio <- NA_real_
for (slice in unique(period_profile$slice)) {
    idx <- which(period_profile$slice == slice)
    idx <- idx[order(period_key(period_profile$period[idx]))]
    period_profile$claims_prev_ratio[idx] <- c(NA_real_, period_profile$claims[idx[-1L]] / pmax(period_profile$claims[idx[-length(idx)]], 1e-12))
    period_profile$frequency_prev_ratio[idx] <- c(NA_real_, period_profile$frequency[idx[-1L]] / pmax(period_profile$frequency[idx[-length(idx)]], 1e-12))
}

group_profiles <- do.call(rbind, lapply(c("medio_emit", "modeloc", iid_col, "desc_edo_circula"), function(col) {
    summarize_by_group(dt[clean_rows, , drop = FALSE], col, period_col, response_col, exposure_col, "clean_after_drops")
}))

positive_claims <- dt[clean_rows & dt[[response_col]] > 0, , drop = FALSE]
positive_claim_profile <- data.frame()
if (nrow(positive_claims) > 0L) {
    periods <- ordered_periods(positive_claims[[period_col]])
    positive_claim_profile <- do.call(rbind, lapply(periods, function(p) {
        idx <- positive_claims[[period_col]] == p
        y <- positive_claims[[response_col]][idx]
        e <- positive_claims[[exposure_col]][idx]
        data.frame(
            period = p,
            positive_rows = sum(idx),
            positive_claims_sum = sum(y, na.rm = TRUE),
            exposure_on_positive_rows = sum(e, na.rm = TRUE),
            pt_q50 = unname(stats::quantile(y, 0.50, na.rm = TRUE)),
            pt_q90 = unname(stats::quantile(y, 0.90, na.rm = TRUE)),
            pt_q99 = unname(stats::quantile(y, 0.99, na.rm = TRUE)),
            pt_max = suppressWarnings(max(y, na.rm = TRUE)),
            check.names = FALSE
        )
    }))
}

paths <- c(
    write_report(column_profile, "claim_diag_column_profile.csv"),
    write_report(response_profile, "claim_diag_response_profile.csv"),
    write_report(period_profile, "claim_diag_period_profile.csv"),
    write_report(group_profiles, "claim_diag_group_profiles.csv"),
    write_report(positive_claim_profile, "claim_diag_positive_claims.csv"),
    write_report(drop_level_table, "claim_diag_dropped_levels.csv")
)

cat("Real-data claim diagnostics complete.\n")
cat(sprintf("Data: %s\n", data_path))
cat(sprintf("Reports: %s\n", report_dir))
cat(sprintf("Rows: %d; clean rows after schema/drop filters: %d\n", nrow(dt), sum(clean_rows)))
cat("Candidate time/claim columns:\n")
cat(sprintf("  %s\n", paste(candidate_columns, collapse = ", ")))
cat("Report files:\n")
cat(sprintf("  %s\n", paths))
