default_periods <- c(
    "2021_y",
    as.vector(vapply(2022:2030, function(year) {
        sprintf("%d_%dt", year, 1:4)
    }, character(4)))
)

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
data_path <- arg_value(
    "--data",
    Sys.getenv(
        "RUSTYINLA_REALDATA_PATH",
        file.path(repo_root, "data", "real", "inla_sbi_test.RDS")
    )
)
if (!grepl("^[A-Za-z]:[/\\\\]|^/", data_path)) {
    data_path <- file.path(repo_root, data_path)
}
if (!file.exists(data_path) && file.exists(sub("\\.RDS$", ".rds", data_path, ignore.case = FALSE))) {
    data_path <- sub("\\.RDS$", ".rds", data_path, ignore.case = FALSE)
}
data_path <- normalizePath(data_path, winslash = "/", mustWork = TRUE)

report_dir <- arg_value(
    "--report-dir",
    Sys.getenv(
        "RUSTYINLA_REALDATA_REPORT_DIR",
        file.path(repo_root, "scratch", "phase8_reports", "realdata_hull")
    )
)
if (!grepl("^[A-Za-z]:[/\\\\]|^/", report_dir)) {
    report_dir <- file.path(repo_root, report_dir)
}
dir.create(report_dir, recursive = TRUE, showWarnings = FALSE)
report_dir <- normalizePath(report_dir, winslash = "/", mustWork = TRUE)

period_col <- arg_value("--period-col", Sys.getenv("RUSTYINLA_REALDATA_PERIOD_COL", "periodo"))
response_col <- arg_value("--response-col", Sys.getenv("RUSTYINLA_REALDATA_RESPONSE_COL", "pt"))
exposure_col <- arg_value("--exposure-col", Sys.getenv("RUSTYINLA_REALDATA_EXPOSURE_COL", "expuesto"))
iid_col <- arg_value("--iid-col", Sys.getenv("RUSTYINLA_REALDATA_IID_COL", "desc_armadora"))
iid_cols <- csv_arg("--iid-cols", Sys.getenv("RUSTYINLA_REALDATA_IID_COLS", iid_col))
if (length(iid_cols) == 0L) {
    stop("--iid-cols must name at least one iid column.", call. = FALSE)
}
iid_col <- iid_cols[[1L]]
fixed_effects <- csv_arg(
    "--fixed",
    Sys.getenv("RUSTYINLA_REALDATA_FIXED", "modeloc,desc_edo_circula,medio_emit")
)
family <- arg_value("--family", Sys.getenv("RUSTYINLA_REALDATA_FAMILY", "poisson"))
base_period <- arg_value("--base-period", Sys.getenv("RUSTYINLA_REALDATA_BASE_PERIOD", "2021_y"))
requested_periods <- csv_arg(
    "--periods",
    Sys.getenv("RUSTYINLA_REALDATA_PERIODS", paste(default_periods, collapse = ","))
)
requested_updates <- csv_arg("--updates", Sys.getenv("RUSTYINLA_REALDATA_UPDATES", ""))
drop_low_exposure_col <- arg_value(
    "--drop-low-exposure-col",
    Sys.getenv("RUSTYINLA_REALDATA_DROP_LOW_EXPOSURE_COL", "")
)
drop_low_exposure_n <- as.integer(arg_value(
    "--drop-low-exposure-n",
    Sys.getenv("RUSTYINLA_REALDATA_DROP_LOW_EXPOSURE_N", "0")
))
drop_low_exposure_threshold <- suppressWarnings(as.numeric(arg_value(
    "--drop-low-exposure-threshold",
    Sys.getenv("RUSTYINLA_REALDATA_DROP_LOW_EXPOSURE_THRESHOLD", "NA")
)))
explicit_drop_levels <- csv_arg(
    "--drop-levels",
    Sys.getenv("RUSTYINLA_REALDATA_DROP_LEVELS", "")
)
top_exposure_feature_col <- arg_value(
    "--top-exposure-feature-col",
    Sys.getenv("RUSTYINLA_REALDATA_TOP_EXPOSURE_FEATURE_COL", "")
)
top_exposure_feature_n <- as.integer(arg_value(
    "--top-exposure-feature-n",
    Sys.getenv("RUSTYINLA_REALDATA_TOP_EXPOSURE_FEATURE_N", "0")
))
top_exposure_feature_name <- arg_value(
    "--top-exposure-feature-name",
    Sys.getenv("RUSTYINLA_REALDATA_TOP_EXPOSURE_FEATURE_NAME", "")
)
top_exposure_feature_scope <- arg_value(
    "--top-exposure-feature-scope",
    Sys.getenv("RUSTYINLA_REALDATA_TOP_EXPOSURE_FEATURE_SCOPE", "base")
)
top_exposure_feature_other <- arg_value(
    "--top-exposure-feature-other",
    Sys.getenv("RUSTYINLA_REALDATA_TOP_EXPOSURE_FEATURE_OTHER", "OTHER")
)
top_exposure_feature_enabled <- nzchar(top_exposure_feature_col) && is.finite(top_exposure_feature_n) &&
    top_exposure_feature_n > 0L
if (top_exposure_feature_enabled && !nzchar(top_exposure_feature_name)) {
    top_exposure_feature_name <- sprintf("%s_top%d", top_exposure_feature_col, top_exposure_feature_n)
}
if (top_exposure_feature_enabled &&
    !(top_exposure_feature_scope %in% c("base", "all"))) {
    stop("--top-exposure-feature-scope must be either 'base' or 'all'.", call. = FALSE)
}
derived_feature_cols <- if (top_exposure_feature_enabled) top_exposure_feature_name else character()
aggregate_poisson_cells <- !has_arg("--no-aggregate-poisson-cells") &&
    (
        has_arg("--aggregate-poisson-cells") ||
            identical(Sys.getenv("RUSTYINLA_REALDATA_AGGREGATE_POISSON_CELLS", "1"), "1")
    )
run_fits <- has_arg("--run-fits") || identical(Sys.getenv("RUSTYINLA_REALDATA_RUN_FITS", "0"), "1")
schema_only <- has_arg("--schema-only") || !run_fits
joint_refit_schedule <- arg_value(
    "--joint-refit-schedule",
    Sys.getenv("RUSTYINLA_REALDATA_JOINT_REFIT_SCHEDULE", "all")
)
if (!(joint_refit_schedule %in% c("all", "final", "none"))) {
    stop("--joint-refit-schedule must be one of 'all', 'final', or 'none'.", call. = FALSE)
}
max_fit_minutes <- suppressWarnings(as.numeric(arg_value(
    "--max-fit-minutes",
    Sys.getenv("RUSTYINLA_REALDATA_MAX_FIT_MINUTES", "Inf")
)))
if (is.na(max_fit_minutes) || max_fit_minutes <= 0) {
    max_fit_minutes <- Inf
}
fit_budget_started_at <- Sys.time()

fit_budget_elapsed_minutes <- function() {
    as.numeric(difftime(Sys.time(), fit_budget_started_at, units = "mins"))
}

check_fit_budget <- function(context) {
    if (!is.finite(max_fit_minutes)) {
        return(invisible(TRUE))
    }
    elapsed <- fit_budget_elapsed_minutes()
    if (elapsed >= max_fit_minutes) {
        stop(
            sprintf(
                "Stopping before/after %s: elapsed %.2f minutes exceeds --max-fit-minutes=%.2f.",
                context,
                elapsed,
                max_fit_minutes
            ),
            call. = FALSE
        )
    }
    invisible(TRUE)
}

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

formula_name <- function(x) {
    if (grepl("^[A-Za-z.][A-Za-z0-9._]*$", x)) {
        return(x)
    }
    paste0("`", gsub("`", "\\\\`", x), "`")
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

required_cols <- unique(c(
    period_col,
    response_col,
    exposure_col,
    iid_cols,
    setdiff(fixed_effects, derived_feature_cols),
    if (top_exposure_feature_enabled) top_exposure_feature_col else character()
))

raw <- readRDS(data_path)
if (!is.data.frame(raw)) {
    raw <- as.data.frame(raw)
}
df <- as.data.frame(raw)

missing_cols <- setdiff(required_cols, names(df))
if (length(missing_cols) > 0L) {
    stop(
        sprintf("Input data is missing required column(s): %s", paste(missing_cols, collapse = ", ")),
        call. = FALSE
    )
}

response <- as_number(df[[response_col]])
exposure <- as_number(df[[exposure_col]])
period <- as.character(df[[period_col]])
iid <- as.character(df[[iid_col]])
iid_values <- lapply(iid_cols, function(col) as.character(df[[col]]))
names(iid_values) <- iid_cols

present_periods <- ordered_periods(period[!is.na(period) & nzchar(period)])
requested_periods <- intersect(requested_periods, present_periods)
if (!(base_period %in% present_periods)) {
    stop(sprintf("Base period '%s' is not present in '%s'.", base_period, period_col), call. = FALSE)
}
if (!(base_period %in% requested_periods)) {
    requested_periods <- c(base_period, requested_periods)
}
requested_periods <- ordered_periods(requested_periods)

if (length(requested_updates) == 0L) {
    requested_updates <- requested_periods[period_key(requested_periods) > period_key(base_period)]
}
requested_updates <- intersect(requested_updates, present_periods)
requested_updates <- ordered_periods(requested_updates)

top_exposure_feature_table <- data.frame(
    source_column = character(),
    feature_column = character(),
    scope = character(),
    rank = integer(),
    level = character(),
    exposure = numeric(),
    rows = integer(),
    claims = numeric(),
    selected = logical(),
    check.names = FALSE
)
if (top_exposure_feature_enabled) {
    source_values <- as.character(df[[top_exposure_feature_col]])
    source_values[is.na(source_values) | !nzchar(source_values)] <- top_exposure_feature_other
    feature_candidate_rows <- is.finite(exposure) & exposure >= 0 &
        is.finite(response) & response >= 0 &
        !is.na(period) & nzchar(period)
    if (identical(top_exposure_feature_scope, "base")) {
        feature_candidate_rows <- feature_candidate_rows & period == base_period
    }
    if (!any(feature_candidate_rows)) {
        stop("No rows are available to build the top-exposure feature.", call. = FALSE)
    }
    exposure_by_level <- tapply(exposure[feature_candidate_rows], source_values[feature_candidate_rows], sum)
    rows_by_level <- tapply(rep.int(1L, sum(feature_candidate_rows)), source_values[feature_candidate_rows], sum)
    claims_by_level <- tapply(response[feature_candidate_rows], source_values[feature_candidate_rows], sum)
    feature_levels <- data.frame(
        level = names(exposure_by_level),
        exposure = as.numeric(exposure_by_level),
        rows = as.integer(rows_by_level[names(exposure_by_level)]),
        claims = as.numeric(claims_by_level[names(exposure_by_level)]),
        check.names = FALSE
    )
    feature_levels <- feature_levels[order(-feature_levels$exposure, feature_levels$level), , drop = FALSE]
    top_levels <- utils::head(feature_levels$level, top_exposure_feature_n)
    top_exposure_feature_table <- data.frame(
        source_column = top_exposure_feature_col,
        feature_column = top_exposure_feature_name,
        scope = top_exposure_feature_scope,
        rank = seq_len(nrow(feature_levels)),
        level = feature_levels$level,
        exposure = feature_levels$exposure,
        rows = feature_levels$rows,
        claims = feature_levels$claims,
        selected = feature_levels$level %in% top_levels,
        check.names = FALSE
    )
    df[[top_exposure_feature_name]] <- ifelse(
        source_values %in% top_levels,
        source_values,
        top_exposure_feature_other
    )
}

valid_exposure <- if (aggregate_poisson_cells && family == "poisson") {
    is.finite(exposure) & exposure >= 0
} else {
    is.finite(exposure) & exposure > 0
}
strict_positive_exposure <- is.finite(exposure) & exposure > 0
valid_response <- is.finite(response) & response >= 0
valid_period <- !is.na(period) & nzchar(period)
valid_iid <- rep(TRUE, nrow(df))
for (col in iid_cols) {
    values <- df[[col]]
    valid_iid <- valid_iid & !is.na(values) & nzchar(as.character(values))
}
valid_fixed <- rep(TRUE, nrow(df))
for (col in fixed_effects) {
    values <- df[[col]]
    valid_fixed <- valid_fixed & !is.na(values) & nzchar(as.character(values))
}
base_model_rows <- valid_exposure & valid_response & valid_period & valid_iid & valid_fixed

factor_level_exposure <- do.call(rbind, lapply(unique(c(iid_cols, fixed_effects)), function(col) {
    values <- as.character(df[[col]])
    levels <- sort(unique(values[base_model_rows]))
    if (length(levels) == 0L) {
        return(data.frame(
            column = character(),
            level = character(),
            rows = integer(),
            exposure = numeric(),
            claims = numeric(),
            active_periods = integer(),
            check.names = FALSE
        ))
    }
    do.call(rbind, lapply(levels, function(level) {
        rows <- base_model_rows & values == level
        data.frame(
            column = col,
            level = level,
            rows = sum(rows),
            exposure = sum(exposure[rows], na.rm = TRUE),
            claims = sum(response[rows], na.rm = TRUE),
            active_periods = length(unique(period[rows])),
            check.names = FALSE
        )
    }))
}))

drop_level_table <- data.frame(
    column = character(),
    level = character(),
    reason = character(),
    exposure = numeric(),
    rows = integer(),
    check.names = FALSE
)
if (nzchar(drop_low_exposure_col)) {
    if (!(drop_low_exposure_col %in% names(df))) {
        stop(sprintf("drop-low-exposure-col '%s' is not present in the data.", drop_low_exposure_col), call. = FALSE)
    }
    level_rows <- factor_level_exposure[factor_level_exposure$column == drop_low_exposure_col, , drop = FALSE]
    level_rows <- level_rows[order(level_rows$exposure, level_rows$rows, level_rows$level), , drop = FALSE]
    selected <- character()
    if (is.finite(drop_low_exposure_threshold)) {
        selected <- union(selected, level_rows$level[level_rows$exposure < drop_low_exposure_threshold])
    }
    if (is.finite(drop_low_exposure_n) && drop_low_exposure_n > 0L) {
        selected <- union(selected, utils::head(level_rows$level, drop_low_exposure_n))
    }
    if (length(explicit_drop_levels) > 0L) {
        selected <- union(selected, explicit_drop_levels)
    }
    selected <- intersect(selected, level_rows$level)
    if (length(selected) > 0L) {
        selected_rows <- level_rows[match(selected, level_rows$level), , drop = FALSE]
        drop_level_table <- data.frame(
            column = drop_low_exposure_col,
            level = selected_rows$level,
            reason = "low_exposure_fixed_level",
            exposure = selected_rows$exposure,
            rows = selected_rows$rows,
            check.names = FALSE
        )
    }
}

low_exposure_level_drop <- rep(FALSE, nrow(df))
if (nrow(drop_level_table) > 0L) {
    for (idx in seq_len(nrow(drop_level_table))) {
        col <- drop_level_table$column[[idx]]
        level <- drop_level_table$level[[idx]]
        low_exposure_level_drop <- low_exposure_level_drop | as.character(df[[col]]) == level
    }
}
model_rows <- base_model_rows & !low_exposure_level_drop

schema <- data.frame(
    field = c(
        "data_path",
        "rows",
        "columns",
        "object_size_mb",
        "period_col",
        "response_col",
        "exposure_col",
        "iid_col",
        "iid_cols",
        "fixed_effects",
        "family",
        "base_period",
        "update_periods",
        "model_rows",
        "dropped_rows",
        "missing_or_invalid_exposure",
        "missing_or_invalid_response",
        "missing_period",
        "missing_iid",
        "missing_fixed_effect",
        "missing_or_invalid_exposure_removed",
        "non_positive_exposure_rows",
        "drop_low_exposure_col",
        "drop_low_exposure_n",
        "drop_low_exposure_threshold",
        "explicit_drop_levels",
        "dropped_low_exposure_level_rows",
        "top_exposure_feature_col",
        "top_exposure_feature_n",
        "top_exposure_feature_name",
        "top_exposure_feature_scope",
        "top_exposure_feature_other",
        "joint_refit_schedule",
        "aggregate_poisson_cells",
        "aggregation_group_columns"
    ),
    value = c(
        data_path,
        nrow(df),
        ncol(df),
        round(as.numeric(utils::object.size(df)) / 1024^2, 3),
        period_col,
        response_col,
        exposure_col,
        iid_col,
        paste(iid_cols, collapse = ","),
        paste(fixed_effects, collapse = ","),
        family,
        base_period,
        paste(requested_updates, collapse = ","),
        sum(model_rows),
        sum(!model_rows),
        sum(!valid_exposure),
        sum(!valid_response),
        sum(!valid_period),
        sum(!valid_iid),
        sum(!valid_fixed),
        sum(!valid_exposure),
        sum(!strict_positive_exposure),
        drop_low_exposure_col,
        drop_low_exposure_n,
        if (is.finite(drop_low_exposure_threshold)) drop_low_exposure_threshold else "",
        paste(explicit_drop_levels, collapse = ","),
        sum(base_model_rows & low_exposure_level_drop),
        top_exposure_feature_col,
        if (top_exposure_feature_enabled) top_exposure_feature_n else "",
        top_exposure_feature_name,
        top_exposure_feature_scope,
        top_exposure_feature_other,
        joint_refit_schedule,
        aggregate_poisson_cells,
        paste(unique(c(period_col, fixed_effects, iid_cols)), collapse = ",")
    ),
    check.names = FALSE
)

period_summary <- do.call(rbind, lapply(present_periods, function(p) {
    rows <- period == p & !is.na(period)
    model_period_rows <- rows & model_rows
    data.frame(
        period = p,
        rows = sum(rows),
        model_rows = sum(model_period_rows),
        exposure = sum(exposure[model_period_rows], na.rm = TRUE),
        claims = sum(response[model_period_rows], na.rm = TRUE),
        claim_frequency = sum(response[model_period_rows], na.rm = TRUE) /
            max(sum(exposure[model_period_rows], na.rm = TRUE), 1e-12),
        iid_levels = length(unique(iid[model_period_rows])),
        iid_levels_by_block = paste(vapply(iid_cols, function(col) {
            sprintf("%s=%d", col, length(unique(as.character(df[[col]][model_period_rows]))))
        }, character(1)), collapse = "|"),
        check.names = FALSE
    )
}))

factor_level_summary <- do.call(rbind, lapply(unique(c(iid_cols, fixed_effects)), function(col) {
    values <- as.character(df[[col]])
    model_values <- values[model_rows]
    tab <- sort(table(model_values), decreasing = FALSE)
    data.frame(
        column = col,
        levels = length(tab),
        missing_rows = sum(is.na(values) | !nzchar(values)),
        min_rows_per_level = if (length(tab) == 0L) NA_integer_ else as.integer(min(tab)),
        max_rows_per_level = if (length(tab) == 0L) NA_integer_ else as.integer(max(tab)),
        singleton_levels = sum(tab == 1L),
        check.names = FALSE
    )
}))

iid_transition <- do.call(rbind, lapply(iid_cols, function(col) {
    seen <- character()
    dormant_once <- character()
    rows <- vector("list", length(present_periods))
    values <- as.character(df[[col]])
    for (idx in seq_along(present_periods)) {
        p <- present_periods[[idx]]
        period_rows <- model_rows & period == p
        active <- sort(unique(values[period_rows]))
        born <- setdiff(active, seen)
        dormant <- setdiff(seen, active)
        reentered <- intersect(active, dormant_once)
        rows[[idx]] <- data.frame(
            iid_col = col,
            period = p,
            active_levels = length(active),
            born_levels = length(born),
            dormant_levels = length(dormant),
            reentered_levels = length(reentered),
            born_level_names = paste(utils::head(born, 25L), collapse = "|"),
            reentered_level_names = paste(utils::head(reentered, 25L), collapse = "|"),
            check.names = FALSE
        )
        dormant_once <- union(dormant_once, dormant)
        seen <- union(seen, active)
    }
    do.call(rbind, rows)
}))

prepare_model_data <- function(df) {
    out <- df[model_rows, , drop = FALSE]
    out[[period_col]] <- as.character(out[[period_col]])
    out[[response_col]] <- as_number(out[[response_col]])
    out[[exposure_col]] <- as_number(out[[exposure_col]])
    for (col in iid_cols) {
        out[[col]] <- factor(
            as.character(out[[col]]),
            levels = sort(unique(as.character(df[[col]][model_rows])))
        )
    }
    for (col in fixed_effects) {
        values <- out[[col]]
        if (is.character(values) || is.factor(values) || is.logical(values)) {
            out[[col]] <- factor(as.character(values), levels = sort(unique(as.character(df[[col]][model_rows]))))
        }
    }
    out
}

model_data <- prepare_model_data(df)
raw_model_row_count <- nrow(model_data)
aggregation_dropped_cells <- data.frame(
    period = character(),
    dropped_cells = integer(),
    dropped_original_rows = integer(),
    dropped_claims = numeric(),
    dropped_exposure = numeric(),
    check.names = FALSE
)
aggregation_dropped_cell_detail <- data.frame()

aggregate_model_data <- function(data) {
    if (!aggregate_poisson_cells) {
        data$original_rows <- 1L
        return(data)
    }
    if (family != "poisson") {
        stop("Poisson cell aggregation is currently supported only for family = 'poisson'.", call. = FALSE)
    }
    group_cols <- unique(c(period_col, fixed_effects, iid_cols))
    if (!requireNamespace("data.table", quietly = TRUE)) {
        stop("Package 'data.table' is required for Poisson cell aggregation.", call. = FALSE)
    }
    dt <- data.table::as.data.table(data)
    agg <- dt[, list(
        .response = sum(get(response_col)),
        .exposure = sum(get(exposure_col)),
        original_rows = .N,
        zero_or_nonpositive_exposure_rows = sum(!is.finite(get(exposure_col)) | get(exposure_col) <= 0),
        claims_on_zero_or_nonpositive_exposure_rows = sum(
            get(response_col)[!is.finite(get(exposure_col)) | get(exposure_col) <= 0]
        )
    ), by = group_cols]
    data.table::setnames(agg, c(".response", ".exposure"), c(response_col, exposure_col))
    out <- as.data.frame(agg)
    bad_exposure_cell <- !is.finite(out[[exposure_col]]) | out[[exposure_col]] <= 0
    if (any(bad_exposure_cell)) {
        aggregation_dropped_cell_detail <<- out[bad_exposure_cell, , drop = FALSE]
        aggregation_dropped_cells <<- do.call(rbind, lapply(split(out[bad_exposure_cell, , drop = FALSE], out[[period_col]][bad_exposure_cell]), function(rows) {
            data.frame(
                period = rows[[period_col]][[1L]],
                dropped_cells = nrow(rows),
                dropped_original_rows = sum(rows$original_rows),
                dropped_claims = sum(rows[[response_col]], na.rm = TRUE),
                dropped_exposure = sum(rows[[exposure_col]], na.rm = TRUE),
                check.names = FALSE
            )
        }))
        out <- out[!bad_exposure_cell, , drop = FALSE]
    }
    out[[period_col]] <- as.character(out[[period_col]])
    for (col in iid_cols) {
        out[[col]] <- factor(as.character(out[[col]]), levels = levels(data[[col]]))
    }
    for (col in fixed_effects) {
        if (is.factor(data[[col]])) {
            out[[col]] <- factor(as.character(out[[col]]), levels = levels(data[[col]]))
        }
    }
    out
}

model_data <- aggregate_model_data(model_data)

factor_level_period_summary <- {
    factor_cols <- unique(c(iid_cols, fixed_effects))
    do.call(rbind, lapply(factor_cols, function(col) {
        if (!(col %in% names(model_data))) {
            return(data.frame())
        }
        period_groups <- split(model_data, model_data[[period_col]])
        rows <- do.call(rbind, lapply(period_groups, function(period_data) {
            values <- as.character(period_data[[col]])
            level_names <- sort(unique(values))
            do.call(rbind, lapply(level_names, function(level) {
                level_rows <- values == level
                exposure_sum <- sum(period_data[[exposure_col]][level_rows], na.rm = TRUE)
                claims_sum <- sum(period_data[[response_col]][level_rows], na.rm = TRUE)
                data.frame(
                    period = period_data[[period_col]][[1L]],
                    column = col,
                    level = level,
                    fitting_rows = sum(level_rows),
                    original_rows = if ("original_rows" %in% names(period_data)) {
                        sum(period_data$original_rows[level_rows], na.rm = TRUE)
                    } else {
                        sum(level_rows)
                    },
                    exposure = exposure_sum,
                    claims = claims_sum,
                    claim_frequency = claims_sum / max(exposure_sum, 1e-12),
                    check.names = FALSE
                )
            }))
        }))
        rows[order(period_key(rows$period), rows$column, rows$level), , drop = FALSE]
    }))
}

aggregation_summary <- do.call(rbind, lapply(present_periods, function(p) {
    raw_rows <- sum(model_rows & period == p)
    fit_rows <- sum(model_data[[period_col]] == p)
    dropped_cells <- aggregation_dropped_cells[aggregation_dropped_cells$period == p, , drop = FALSE]
    data.frame(
        period = p,
        raw_model_rows = raw_rows,
        fitting_rows = fit_rows,
        compression_ratio = raw_rows / max(fit_rows, 1L),
        original_rows_from_cells = sum(model_data$original_rows[model_data[[period_col]] == p], na.rm = TRUE),
        dropped_zero_exposure_cells = if (nrow(dropped_cells) == 0L) 0L else sum(dropped_cells$dropped_cells),
        dropped_zero_exposure_cell_claims = if (nrow(dropped_cells) == 0L) 0.0 else sum(dropped_cells$dropped_claims),
        check.names = FALSE
    )
}))
aggregation_total <- data.frame(
    period = "ALL",
    raw_model_rows = raw_model_row_count,
    fitting_rows = nrow(model_data),
    compression_ratio = raw_model_row_count / max(nrow(model_data), 1L),
    original_rows_from_cells = sum(model_data$original_rows, na.rm = TRUE),
    dropped_zero_exposure_cells = sum(aggregation_dropped_cells$dropped_cells),
    dropped_zero_exposure_cell_claims = sum(aggregation_dropped_cells$dropped_claims),
    check.names = FALSE
)
aggregation_summary <- rbind(aggregation_total, aggregation_summary)

rank_row <- function(label, data, fixed_formula) {
    if (nrow(data) == 0L) {
        return(data.frame(
            period = label,
            rows = 0L,
            fixed_columns = NA_integer_,
            rank = NA_integer_,
            zero_columns = NA_integer_,
            zero_column_names = "",
            rank_deficient = NA,
            check.names = FALSE
        ))
    }
    mm <- stats::model.matrix(fixed_formula, data = data)
    zero_cols <- colSums(abs(mm)) == 0
    rank <- qr(mm)$rank
    data.frame(
        period = label,
        rows = nrow(data),
        fixed_columns = ncol(mm),
        rank = rank,
        zero_columns = sum(zero_cols),
        zero_column_names = paste(colnames(mm)[zero_cols], collapse = "|"),
        rank_deficient = rank < ncol(mm),
        check.names = FALSE
    )
}

fixed_rank_table <- {
    fixed_formula <- stats::as.formula(paste("~", paste(vapply(fixed_effects, formula_name, character(1)), collapse = " + ")))
    rows <- vector("list", length(present_periods))
    for (idx in seq_along(present_periods)) {
        p <- present_periods[[idx]]
        period_data <- model_data[model_data[[period_col]] == p, , drop = FALSE]
        rows[[idx]] <- rank_row(p, period_data, fixed_formula)
    }
    do.call(rbind, rows)
}

cumulative_rank_table <- {
    fixed_formula <- stats::as.formula(paste("~", paste(vapply(fixed_effects, formula_name, character(1)), collapse = " + ")))
    rolling_periods <- c(base_period, requested_updates)
    rows <- vector("list", length(rolling_periods))
    for (idx in seq_along(rolling_periods)) {
        p <- rolling_periods[[idx]]
        through_key <- period_key(p)
        period_data <- model_data[period_key(model_data[[period_col]]) <= through_key, , drop = FALSE]
        rows[[idx]] <- rank_row(sprintf("through_%s", p), period_data, fixed_formula)
    }
    do.call(rbind, rows)
}

schema_path <- write_report(schema, "realdata_schema.csv")
period_path <- write_report(period_summary, "realdata_period_summary.csv")
levels_path <- write_report(factor_level_summary, "realdata_factor_level_summary.csv")
transition_path <- write_report(iid_transition, "realdata_iid_transition.csv")
rank_path <- write_report(fixed_rank_table, "realdata_fixed_rank_by_period.csv")
cumulative_rank_path <- write_report(cumulative_rank_table, "realdata_fixed_rank_cumulative.csv")
level_exposure_path <- write_report(factor_level_exposure, "realdata_level_exposure.csv")
level_period_path <- write_report(factor_level_period_summary, "realdata_factor_level_by_period.csv")
drop_levels_path <- write_report(drop_level_table, "realdata_dropped_levels.csv")
top_exposure_feature_path <- write_report(top_exposure_feature_table, "realdata_top_exposure_feature_levels.csv")
aggregation_path <- write_report(aggregation_summary, "realdata_aggregation_summary.csv")
aggregation_dropped_cells_path <- write_report(aggregation_dropped_cells, "realdata_aggregation_dropped_zero_exposure_cells.csv")
aggregation_dropped_cell_detail_path <- write_report(
    aggregation_dropped_cell_detail,
    "realdata_aggregation_dropped_zero_exposure_cell_detail.csv"
)

cat("Phase 8 real-data rolling validation scaffold\n")
cat(sprintf("Data: %s\n", data_path))
cat(sprintf("Reports: %s\n", report_dir))
cat(sprintf("Rows: %d total, %d usable model rows, %d dropped by schema checks\n", nrow(df), sum(model_rows), sum(!model_rows)))
cat(sprintf("Base period: %s; updates requested/present: %s\n", base_period, paste(requested_updates, collapse = ", ")))
cat(sprintf("iid random-effect columns: %s\n", paste(iid_cols, collapse = ", ")))
cat(sprintf("Joint refit schedule: %s\n", joint_refit_schedule))
cat("Schema reports written:\n")
cat(sprintf("  %s\n", schema_path))
cat(sprintf("  %s\n", period_path))
cat(sprintf("  %s\n", levels_path))
cat(sprintf("  %s\n", transition_path))
cat(sprintf("  %s\n", rank_path))
cat(sprintf("  %s\n", cumulative_rank_path))
cat(sprintf("  %s\n", level_exposure_path))
cat(sprintf("  %s\n", level_period_path))
cat(sprintf("  %s\n", drop_levels_path))
cat(sprintf("  %s\n", top_exposure_feature_path))
cat(sprintf("  %s\n", aggregation_path))
cat(sprintf("  %s\n", aggregation_dropped_cells_path))
cat(sprintf("  %s\n", aggregation_dropped_cell_detail_path))
cat(sprintf(
    "Fitting rows: %d after%s Poisson cell aggregation (raw model rows: %d).\n",
    nrow(model_data),
    if (aggregate_poisson_cells) "" else " no",
    raw_model_row_count
))
if (nrow(drop_level_table) > 0L) {
    cat("Dropped low-exposure levels:\n")
    print(drop_level_table)
}
if (top_exposure_feature_enabled) {
    selected_feature_levels <- top_exposure_feature_table[top_exposure_feature_table$selected, , drop = FALSE]
    cat(sprintf(
        "Top-exposure feature %s from %s (%s scope): %d selected level(s), else '%s'.\n",
        top_exposure_feature_name,
        top_exposure_feature_col,
        top_exposure_feature_scope,
        nrow(selected_feature_levels),
        top_exposure_feature_other
    ))
    print(selected_feature_levels[, c("rank", "level", "exposure", "rows", "claims"), drop = FALSE])
}

if (schema_only) {
    cat("Schema-only mode complete. Pass --run-fits to run update-vs-joint fitting.\n")
    quit(status = 0L)
}

source(file.path(repo_root, "tools", "load_worktree_package.R"), local = TRUE)
load_rustyinla_for_benchmarks(repo_root)
if (is.finite(max_fit_minutes)) {
    cat(sprintf(
        "Fit budget: %.2f minutes. The harness checks between fit stages; use an outer process timeout for a hard kill.\n",
        max_fit_minutes
    ))
}

timed_fit <- function(expr, label = "fit") {
    check_fit_budget(label)
    invisible(gc())
    before <- gc()
    before_mb <- if ("(Mb)" %in% colnames(before)) sum(before[, "(Mb)"]) else NA_real_
    elapsed <- system.time(value <- eval.parent(substitute(expr)))
    check_fit_budget(label)
    after <- gc()
    after_mb <- if ("(Mb)" %in% colnames(after)) sum(after[, "(Mb)"]) else NA_real_
    list(
        value = value,
        elapsed = unname(elapsed[["elapsed"]]),
        memory_delta_mb = after_mb - before_mb,
        object_mb = as.numeric(utils::object.size(value)) / 1024^2
    )
}

named_column <- function(df, column) {
    if (is.null(df) || nrow(df) == 0L || !(column %in% names(df))) {
        return(stats::setNames(numeric(), character()))
    }
    stats::setNames(as.numeric(df[[column]]), rownames(df))
}

max_abs_named_diff <- function(lhs, rhs) {
    shared <- intersect(names(lhs), names(rhs))
    if (length(shared) == 0L) {
        return(NA_real_)
    }
    max(abs(lhs[shared] - rhs[shared]))
}

min_named_ratio <- function(lhs, rhs) {
    shared <- intersect(names(lhs), names(rhs))
    if (length(shared) == 0L) {
        return(NA_real_)
    }
    min(lhs[shared] / pmax(rhs[shared], 1e-12))
}

max_fitted_rel_diff <- function(fit, joint_fit, joint_rows) {
    lhs <- as.numeric(fit$summary.fitted.values$mean)
    rhs <- as.numeric(joint_fit$summary.fitted.values$mean[joint_rows])
    n <- min(length(lhs), length(rhs))
    if (n == 0L) {
        return(NA_real_)
    }
    max(abs(lhs[seq_len(n)] - rhs[seq_len(n)]) / pmax(1.0, abs(rhs[seq_len(n)])))
}

aligned_fitted_values <- function(fit, joint_fit, joint_rows, column = "mean") {
    if (is.null(fit$summary.fitted.values) ||
        is.null(joint_fit$summary.fitted.values) ||
        !(column %in% names(fit$summary.fitted.values)) ||
        !(column %in% names(joint_fit$summary.fitted.values))) {
        return(data.frame(proxy = numeric(), joint = numeric(), check.names = FALSE))
    }
    lhs <- as.numeric(fit$summary.fitted.values[[column]])
    rhs <- as.numeric(joint_fit$summary.fitted.values[[column]][joint_rows])
    n <- min(length(lhs), length(rhs))
    if (n == 0L) {
        return(data.frame(proxy = numeric(), joint = numeric(), check.names = FALSE))
    }
    data.frame(
        proxy = lhs[seq_len(n)],
        joint = rhs[seq_len(n)],
        check.names = FALSE
    )
}

quantile_or_na <- function(x, prob) {
    x <- x[is.finite(x)]
    if (length(x) == 0L) {
        return(NA_real_)
    }
    as.numeric(stats::quantile(x, prob, names = FALSE, type = 7))
}

fitted_diff_stats <- function(fit, joint_fit, joint_rows, column = "mean") {
    fitted <- aligned_fitted_values(fit, joint_fit, joint_rows, column = column)
    if (nrow(fitted) == 0L) {
        return(list(
            total_proxy = NA_real_,
            total_joint = NA_real_,
            total_abs_diff = NA_real_,
            mean_abs = NA_real_,
            median_abs = NA_real_,
            q95_abs = NA_real_,
            q99_abs = NA_real_,
            max_abs = NA_real_,
            median_rel = NA_real_,
            q95_rel = NA_real_,
            q99_rel = NA_real_,
            max_rel = NA_real_
        ))
    }
    abs_diff <- abs(fitted$proxy - fitted$joint)
    rel_diff <- abs_diff / pmax(1.0, abs(fitted$joint))
    list(
        total_proxy = sum(fitted$proxy),
        total_joint = sum(fitted$joint),
        total_abs_diff = abs(sum(fitted$proxy) - sum(fitted$joint)),
        mean_abs = mean(abs_diff),
        median_abs = stats::median(abs_diff),
        q95_abs = quantile_or_na(abs_diff, 0.95),
        q99_abs = quantile_or_na(abs_diff, 0.99),
        max_abs = max(abs_diff),
        median_rel = stats::median(rel_diff),
        q95_rel = quantile_or_na(rel_diff, 0.95),
        q99_rel = quantile_or_na(rel_diff, 0.99),
        max_rel = max(rel_diff)
    )
}

marginal_cdf <- function(marginal) {
    if (is.null(marginal) || nrow(marginal) < 2L) {
        return(NULL)
    }
    x <- as.numeric(marginal[, "x"])
    y <- as.numeric(marginal[, "y"])
    area <- c(0.0, cumsum(diff(x) * (head(y, -1L) + tail(y, -1L)) / 2.0))
    if (tail(area, 1L) > 0) {
        area <- area / tail(area, 1L)
    }
    data.frame(x = x, cdf = area)
}

theta_ks <- function(proxy_fit, joint_fit) {
    proxy <- proxy_fit$marginals.hyperpar
    joint <- joint_fit$marginals.hyperpar
    if (is.null(proxy) || is.null(joint)) {
        return(NA_real_)
    }
    theta_names <- intersect(names(proxy), names(joint))
    if (length(theta_names) == 0L) {
        return(NA_real_)
    }
    max(vapply(theta_names, function(name) {
        lhs <- marginal_cdf(proxy[[name]])
        rhs <- marginal_cdf(joint[[name]])
        if (is.null(lhs) || is.null(rhs)) {
            return(NA_real_)
        }
        grid <- sort(unique(c(lhs$x, rhs$x)))
        lhs_cdf <- stats::approx(lhs$x, lhs$cdf, xout = grid, rule = 2)$y
        rhs_cdf <- stats::approx(rhs$x, rhs$cdf, xout = grid, rule = 2)$y
        max(abs(lhs_cdf - rhs_cdf))
    }, numeric(1)), na.rm = TRUE)
}

summary_value <- function(df, row_names, column) {
    if (is.null(df) || nrow(df) == 0L || !(column %in% names(df))) {
        return(rep(NA_real_, length(row_names)))
    }
    as.numeric(df[row_names, column])
}

effect_comparison_table <- function(stage, proxy_fit, joint_fit, current_data) {
    fixed_proxy <- proxy_fit$summary.fixed
    fixed_joint <- joint_fit$summary.fixed
    fixed_names <- union(rownames(fixed_proxy), rownames(fixed_joint))
    fixed_table <- data.frame(
        stage = stage,
        effect_type = "fixed",
        effect_group = "fixed",
        parameter = fixed_names,
        current_exposure = NA_real_,
        current_rows = NA_integer_,
        proxy_mean = as.numeric(fixed_proxy[fixed_names, "mean"]),
        true_refit_mean = as.numeric(fixed_joint[fixed_names, "mean"]),
        proxy_sd = as.numeric(fixed_proxy[fixed_names, "sd"]),
        true_refit_sd = as.numeric(fixed_joint[fixed_names, "sd"]),
        proxy_q025 = summary_value(fixed_proxy, fixed_names, "0.025quant"),
        true_refit_q025 = summary_value(fixed_joint, fixed_names, "0.025quant"),
        proxy_q500 = summary_value(fixed_proxy, fixed_names, "0.5quant"),
        true_refit_q500 = summary_value(fixed_joint, fixed_names, "0.5quant"),
        proxy_q975 = summary_value(fixed_proxy, fixed_names, "0.975quant"),
        true_refit_q975 = summary_value(fixed_joint, fixed_names, "0.975quant"),
        check.names = FALSE
    )

    row_weight <- if ("original_rows" %in% names(current_data)) {
        current_data$original_rows
    } else {
        rep.int(1L, nrow(current_data))
    }
    random_tables <- lapply(iid_cols, function(col) {
        random_proxy <- proxy_fit$summary.random[[col]]
        random_joint <- joint_fit$summary.random[[col]]
        random_names <- union(rownames(random_proxy), rownames(random_joint))
        exposure_by_level <- tapply(current_data[[exposure_col]], as.character(current_data[[col]]), sum)
        rows_by_level <- tapply(row_weight, as.character(current_data[[col]]), sum)
        data.frame(
            stage = stage,
            effect_type = "random",
            effect_group = col,
            parameter = random_names,
            current_exposure = as.numeric(exposure_by_level[random_names]),
            current_rows = as.integer(rows_by_level[random_names]),
            proxy_mean = as.numeric(random_proxy[random_names, "mean"]),
            true_refit_mean = as.numeric(random_joint[random_names, "mean"]),
            proxy_sd = as.numeric(random_proxy[random_names, "sd"]),
            true_refit_sd = as.numeric(random_joint[random_names, "sd"]),
            proxy_q025 = summary_value(random_proxy, random_names, "0.025quant"),
            true_refit_q025 = summary_value(random_joint, random_names, "0.025quant"),
            proxy_q500 = summary_value(random_proxy, random_names, "0.5quant"),
            true_refit_q500 = summary_value(random_joint, random_names, "0.5quant"),
            proxy_q975 = summary_value(random_proxy, random_names, "0.975quant"),
            true_refit_q975 = summary_value(random_joint, random_names, "0.975quant"),
            check.names = FALSE
        )
    })
    out <- rbind(fixed_table, do.call(rbind, random_tables))
    out$current_exposure[is.na(out$current_exposure) & out$effect_type == "random"] <- 0.0
    out$current_rows[is.na(out$current_rows) & out$effect_type == "random"] <- 0L
    out$mean_abs_diff <- abs(out$proxy_mean - out$true_refit_mean)
    out$sd_ratio <- out$proxy_sd / pmax(out$true_refit_sd, 1e-12)
    out$proxy_interval_width <- out$proxy_q975 - out$proxy_q025
    out$true_refit_interval_width <- out$true_refit_q975 - out$true_refit_q025
    out$interval_width_ratio <- out$proxy_interval_width / pmax(out$true_refit_interval_width, 1e-12)
    out$interval_overlap_width <- pmax(
        0.0,
        pmin(out$proxy_q975, out$true_refit_q975) - pmax(out$proxy_q025, out$true_refit_q025)
    )
    out$interval_overlap_true_fraction <- out$interval_overlap_width /
        pmax(out$true_refit_interval_width, 1e-12)
    out
}

fitted_comparison_table <- function(stage, update_fit, joint_fit, current_data, joint_rows) {
    fitted <- aligned_fitted_values(update_fit, joint_fit, joint_rows, column = "mean")
    if (nrow(fitted) == 0L) {
        return(data.frame())
    }
    fitted_median <- aligned_fitted_values(update_fit, joint_fit, joint_rows, column = "0.5quant")
    fitted_mode <- aligned_fitted_values(update_fit, joint_fit, joint_rows, column = "mode")
    rows <- current_data[seq_len(nrow(fitted)), c(
        period_col,
        response_col,
        exposure_col,
        iid_cols,
        fixed_effects,
        intersect("original_rows", names(current_data))
    ), drop = FALSE]
    out <- cbind(
        data.frame(stage = stage, cell_index = seq_len(nrow(fitted)), check.names = FALSE),
        rows,
        data.frame(
            proxy_fitted_mean = fitted$proxy,
            true_refit_fitted_mean = fitted$joint,
            fitted_abs_diff = abs(fitted$proxy - fitted$joint),
            fitted_rel_diff_to_joint = abs(fitted$proxy - fitted$joint) / pmax(1.0, abs(fitted$joint)),
            proxy_fitted_median = fitted_median$proxy,
            true_refit_fitted_median = fitted_median$joint,
            fitted_median_abs_diff = abs(fitted_median$proxy - fitted_median$joint),
            proxy_fitted_mode = fitted_mode$proxy,
            true_refit_fitted_mode = fitted_mode$joint,
            fitted_mode_abs_diff = abs(fitted_mode$proxy - fitted_mode$joint),
            check.names = FALSE
        )
    )
    out[order(-out$fitted_abs_diff), , drop = FALSE]
}

fit_metrics <- function(stage, update_fit, update_time, joint_fit, joint_time, joint_rows, base_time) {
    fixed <- named_column(update_fit$summary.fixed, "mean")
    fixed_joint <- named_column(joint_fit$summary.fixed, "mean")
    fixed_sd <- named_column(update_fit$summary.fixed, "sd")
    fixed_joint_sd <- named_column(joint_fit$summary.fixed, "sd")
    random_mean_by_block <- vapply(iid_cols, function(col) {
        max_abs_named_diff(
            named_column(update_fit$summary.random[[col]], "mean"),
            named_column(joint_fit$summary.random[[col]], "mean")
        )
    }, numeric(1))
    random_sd_ratio_by_block <- vapply(iid_cols, function(col) {
        min_named_ratio(
            named_column(update_fit$summary.random[[col]], "sd"),
            named_column(joint_fit$summary.random[[col]], "sd")
        )
    }, numeric(1))
    names(random_mean_by_block) <- paste0("random_mean_max_abs_to_joint_", iid_cols)
    names(random_sd_ratio_by_block) <- paste0("random_sd_min_ratio_to_joint_", iid_cols)
    fitted_stats <- fitted_diff_stats(update_fit, joint_fit, joint_rows)
    fitted_median_stats <- fitted_diff_stats(update_fit, joint_fit, joint_rows, column = "0.5quant")
    fitted_mode_stats <- fitted_diff_stats(update_fit, joint_fit, joint_rows, column = "mode")
    theta_update <- as.numeric(update_fit$mode$theta)
    theta_joint <- as.numeric(joint_fit$mode$theta)
    out <- data.frame(
        stage = stage,
        base_time_sec = base_time$elapsed,
        update_time_sec = update_time$elapsed,
        joint_time_sec = joint_time$elapsed,
        base_object_mb = base_time$object_mb,
        update_object_mb = update_time$object_mb,
        joint_object_mb = joint_time$object_mb,
        base_memory_delta_mb = base_time$memory_delta_mb,
        update_memory_delta_mb = update_time$memory_delta_mb,
        joint_memory_delta_mb = joint_time$memory_delta_mb,
        theta_log_precision_update = theta_update[[1L]],
        theta_log_precision_joint = theta_joint[[1L]],
        theta_internal_update = paste(round(theta_update, 6), collapse = "|"),
        theta_internal_joint = paste(round(theta_joint, 6), collapse = "|"),
        theta_abs_drift = max(abs(theta_update - theta_joint)),
        theta_cdf_ks = theta_ks(update_fit, joint_fit),
        fixed_mean_max_abs_to_joint = max_abs_named_diff(fixed, fixed_joint),
        random_mean_max_abs_to_joint = max(random_mean_by_block, na.rm = TRUE),
        fitted_new_max_rel_to_joint = max_fitted_rel_diff(update_fit, joint_fit, joint_rows),
        fitted_new_total_update = fitted_stats$total_proxy,
        fitted_new_total_joint = fitted_stats$total_joint,
        fitted_new_total_abs_diff = fitted_stats$total_abs_diff,
        fitted_new_mean_abs_to_joint = fitted_stats$mean_abs,
        fitted_new_median_abs_to_joint = fitted_stats$median_abs,
        fitted_new_q95_abs_to_joint = fitted_stats$q95_abs,
        fitted_new_q99_abs_to_joint = fitted_stats$q99_abs,
        fitted_new_max_abs_to_joint = fitted_stats$max_abs,
        fitted_new_median_rel_to_joint = fitted_stats$median_rel,
        fitted_new_q95_rel_to_joint = fitted_stats$q95_rel,
        fitted_new_q99_rel_to_joint = fitted_stats$q99_rel,
        fitted_new_total_median_update = fitted_median_stats$total_proxy,
        fitted_new_total_median_joint = fitted_median_stats$total_joint,
        fitted_new_total_median_abs_diff = fitted_median_stats$total_abs_diff,
        fitted_new_median_q95_abs_to_joint = fitted_median_stats$q95_abs,
        fitted_new_median_max_abs_to_joint = fitted_median_stats$max_abs,
        fitted_new_total_mode_update = fitted_mode_stats$total_proxy,
        fitted_new_total_mode_joint = fitted_mode_stats$total_joint,
        fitted_new_total_mode_abs_diff = fitted_mode_stats$total_abs_diff,
        fitted_new_mode_q95_abs_to_joint = fitted_mode_stats$q95_abs,
        fitted_new_mode_max_abs_to_joint = fitted_mode_stats$max_abs,
        fixed_sd_min_ratio_to_joint = min_named_ratio(fixed_sd, fixed_joint_sd),
        random_sd_min_ratio_to_joint = min(random_sd_ratio_by_block, na.rm = TRUE),
        check.names = FALSE
    )
    for (name in names(random_mean_by_block)) {
        out[[name]] <- random_mean_by_block[[name]]
    }
    for (name in names(random_sd_ratio_by_block)) {
        out[[name]] <- random_sd_ratio_by_block[[name]]
    }
    out
}

rhs_terms <- c(
    "1",
    vapply(fixed_effects, formula_name, character(1)),
    sprintf("offset(log(%s))", formula_name(exposure_col)),
    sprintf("f(%s, model = \"iid\")", vapply(iid_cols, formula_name, character(1)))
)
model_formula <- stats::as.formula(sprintf("%s ~ %s", formula_name(response_col), paste(rhs_terms, collapse = " + ")))
cat(sprintf("Model formula: %s\n", paste(deparse(model_formula), collapse = " ")))

base_data <- model_data[model_data[[period_col]] == base_period, , drop = FALSE]
if (nrow(base_data) == 0L) {
    stop("No usable rows remain for the base period after schema filtering.", call. = FALSE)
}

cat(sprintf("Fitting base period %s (%d rows)...\n", base_period, nrow(base_data)))
base_fit <- timed_fit(
    rusty_inla(model_formula, data = base_data, family = family, output_profile = "benchmark"),
    label = sprintf("base period %s", base_period)
)
cat(sprintf(
    "Base fit complete: %.2f sec, object %.2f MB, memory delta %.2f MB.\n",
    base_fit$elapsed,
    base_fit$object_mb,
    base_fit$memory_delta_mb
))
state <- rusty_update_state(base_fit$value)
cumulative_data <- base_data

metrics <- list()
effect_tables <- list()
fitted_tables <- list()
update_time_tables <- list()
last_requested_update <- if (length(requested_updates) == 0L) NA_character_ else requested_updates[[length(requested_updates)]]
update_times_path <- file.path(report_dir, "realdata_rolling_update_times.csv")

for (p in requested_updates) {
    batch_data <- model_data[model_data[[period_col]] == p, , drop = FALSE]
    if (nrow(batch_data) == 0L) {
        warning(sprintf("Skipping period %s because it has no usable rows.", p), call. = FALSE)
        next
    }
    cumulative_data <- rbind(cumulative_data, batch_data)
    joint_rows <- which(cumulative_data[[period_col]] == p)

    cat(sprintf("Running rolling update for %s (%d rows)...\n", p, nrow(batch_data)))
    update_fit <- timed_fit(rusty_inla(
        model_formula,
        data = batch_data,
        family = family,
        output_profile = "benchmark",
        control.update = list(
            state = state,
            mode = "fixed_iid_cross_theta_evidence"
        )
    ), label = sprintf("rolling update %s", p))

    update_time_tables[[p]] <- data.frame(
        stage = p,
        update_rows = nrow(batch_data),
        cumulative_rows = nrow(cumulative_data),
        update_time_sec = update_fit$elapsed,
        update_object_mb = update_fit$object_mb,
        update_memory_delta_mb = update_fit$memory_delta_mb,
        theta_log_precision_update = as.numeric(update_fit$value$mode$theta[[1L]]),
        theta_internal_update = paste(round(as.numeric(update_fit$value$mode$theta), 6), collapse = "|"),
        theta_evidence_support_points = if (is.null(update_fit$value$posterior_state_used$theta_evidence_support_points)) {
            NA_integer_
        } else {
            as.integer(update_fit$value$posterior_state_used$theta_evidence_support_points)
        },
        theta_evidence_guard_points_added = if (is.null(update_fit$value$posterior_state_used$theta_evidence_guard_points_added)) {
            NA_integer_
        } else {
            as.integer(update_fit$value$posterior_state_used$theta_evidence_guard_points_added)
        },
        compared_to_joint = identical(joint_refit_schedule, "all") ||
            (identical(joint_refit_schedule, "final") && identical(p, last_requested_update)),
        check.names = FALSE
    )
    write_report(do.call(rbind, update_time_tables), "realdata_rolling_update_times.csv")

    compare_to_joint <- identical(joint_refit_schedule, "all") ||
        (identical(joint_refit_schedule, "final") && identical(p, last_requested_update))
    if (compare_to_joint) {
        cat(sprintf("Running cumulative joint refit through %s (%d rows)...\n", p, nrow(cumulative_data)))
        joint_fit <- timed_fit(rusty_inla(
            model_formula,
            data = cumulative_data,
            family = family,
            output_profile = "benchmark"
        ), label = sprintf("cumulative joint refit through %s", p))

        metrics[[p]] <- fit_metrics(p, update_fit$value, update_fit, joint_fit$value, joint_fit, joint_rows, base_fit)
        effect_tables[[p]] <- effect_comparison_table(p, update_fit$value, joint_fit$value, batch_data)
        fitted_tables[[p]] <- fitted_comparison_table(p, update_fit$value, joint_fit$value, batch_data, joint_rows)
    }
    state <- rusty_compose_update_state(state, update_fit$value)
}

update_time_table <- do.call(rbind, update_time_tables)
write_report(update_time_table, "realdata_rolling_update_times.csv")

if (length(metrics) == 0L) {
    if (identical(joint_refit_schedule, "none")) {
        cat("Rolling updates complete without joint-refit comparison.\n")
        cat(sprintf("  %s\n", update_times_path))
        print(update_time_table)
        quit(status = 0L)
    }
    stop("No update periods were compared to a joint refit.", call. = FALSE)
}

metrics_table <- do.call(rbind, metrics)
effect_table <- do.call(rbind, effect_tables)
fitted_table <- do.call(rbind, fitted_tables)

metrics_path <- write_report(metrics_table, "realdata_rolling_metrics.csv")
effect_path <- write_report(effect_table, "realdata_rolling_effects_proxy_vs_joint.csv")
fitted_path <- write_report(fitted_table, "realdata_rolling_fitted_proxy_vs_joint.csv")
update_times_path <- write_report(update_time_table, "realdata_rolling_update_times.csv")

cat("Rolling validation complete.\n")
cat(sprintf("  %s\n", update_times_path))
cat(sprintf("  %s\n", metrics_path))
cat(sprintf("  %s\n", effect_path))
cat(sprintf("  %s\n", fitted_path))
print(metrics_table)
