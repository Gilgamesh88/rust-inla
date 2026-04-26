source_root <- if (file.exists(file.path("R", "interface.R"))) {
    "."
} else if (file.exists(file.path("..", "R", "interface.R"))) {
    ".."
} else {
    NA_character_
}

if (!is.na(source_root)) {
    source(file.path(source_root, "R", "f.R"), local = FALSE)
    source(file.path(source_root, "R", "interface.R"), local = FALSE)
} else {
    library(rustyINLA)
    build_backend_spec <- getFromNamespace("build_backend_spec", "rustyINLA")
}

expect_error_matching <- function(expr, pattern) {
    err <- tryCatch(
        {
            force(expr)
            NULL
        },
        error = function(e) conditionMessage(e)
    )
    if (is.null(err)) {
        stop(sprintf("Expected an error matching '%s', but no error was thrown.", pattern))
    }
    if (!grepl(pattern, err)) {
        stop(sprintf("Expected error matching '%s', got: %s", pattern, err))
    }
}

df <- data.frame(
    y = c(0.9, 1.4, 1.1, 1.8, 1.2, 2.0, 1.5, 2.2),
    x1 = c(-1.0, -0.4, 0.2, 0.8, -0.7, 0.5, 1.1, -0.2),
    x2 = c(0.3, -0.1, 0.6, -0.5, 0.2, 0.9, -0.4, 0.7),
    promo = factor(
        c("base", "promo", "base", "promo", "base", "promo", "promo", "base"),
        levels = c("base", "promo")
    ),
    group = factor(c(1, 1, 2, 2, 3, 3, 4, 4)),
    exposure = c(0.8, 1.1, 1.0, 1.4, 0.9, 1.6, 1.2, 1.5)
)

spec <- build_backend_spec(
    y ~ 1 + x1 * promo + x2 + offset(log(exposure)) + f(group, model = "iid"),
    data = df,
    family = "gaussian"
)

expected_matrix <- model.matrix(
    ~ 1 + x1 * promo + x2 + offset(log(exposure)),
    data = df
)
got_matrix <- matrix(
    spec$fixed_matrix,
    nrow = nrow(df),
    ncol = spec$n_fixed,
    dimnames = list(NULL, spec$fixed_names)
)

stopifnot(identical(spec$n_fixed, as.integer(ncol(expected_matrix))))
stopifnot(identical(spec$fixed_names, colnames(expected_matrix)))
stopifnot(isTRUE(all.equal(
    unname(got_matrix),
    unname(expected_matrix),
    tolerance = 1e-12,
    check.attributes = FALSE
)))
stopifnot(isTRUE(all.equal(spec$offset, log(df$exposure), tolerance = 1e-12)))
stopifnot(identical(spec$n_latent, as.integer(nlevels(df$group))))

rich_df <- data.frame(
    y = c(1.2, 0.8, 1.6, 1.0, 1.4, 1.9, 0.7, 1.1, 1.8, 1.3, 2.0, 0.9),
    x1 = c(-1.2, -0.8, -0.3, 0.1, 0.5, 1.0, -1.0, -0.4, 0.2, 0.7, 1.2, -0.1),
    tier = factor(
        rep(c("low", "mid", "high"), 4L),
        levels = c("low", "mid", "high")
    ),
    flag = rep(c(TRUE, FALSE), 6L),
    group = factor(rep(seq_len(4L), each = 3L)),
    exposure = c(0.9, 1.2, 0.8, 1.5, 1.1, 1.8, 1.0, 1.4, 0.7, 1.6, 1.3, 1.9)
)

rich_spec <- build_backend_spec(
    y ~ 1 + x1 + tier + flag + x1:tier + offset(log(exposure)) + f(group, model = "iid"),
    data = rich_df,
    family = "gaussian"
)
rich_expected_matrix <- model.matrix(
    ~ 1 + x1 + tier + flag + x1:tier + offset(log(exposure)),
    data = rich_df
)
rich_matrix <- matrix(
    rich_spec$fixed_matrix,
    nrow = nrow(rich_df),
    ncol = rich_spec$n_fixed,
    dimnames = list(NULL, rich_spec$fixed_names)
)
stopifnot(identical(rich_spec$n_fixed, as.integer(ncol(rich_expected_matrix))))
stopifnot(identical(rich_spec$fixed_names, colnames(rich_expected_matrix)))
stopifnot(isTRUE(all.equal(
    unname(rich_matrix),
    unname(rich_expected_matrix),
    tolerance = 1e-12,
    check.attributes = FALSE
)))
stopifnot(isTRUE(all.equal(rich_spec$offset, log(rich_df$exposure), tolerance = 1e-12)))
stopifnot(identical(rich_spec$n_latent, as.integer(nlevels(rich_df$group))))

rank_deficient_df <- transform(df, x_dup = x1)
expect_error_matching(
    build_backend_spec(
        y ~ 1 + x1 + x_dup + f(group, model = "iid"),
        data = rank_deficient_df,
        family = "gaussian"
    ),
    "rank-deficient"
)

unused_factor_df <- transform(
    df,
    unused = factor(rep("observed", nrow(df)), levels = c("observed", "empty"))
)
expect_error_matching(
    build_backend_spec(
        y ~ 1 + unused + f(group, model = "iid"),
        data = unused_factor_df,
        family = "gaussian"
    ),
    "rank-deficient"
)

expect_error_matching(
    build_backend_spec(
        y ~ 1 + log(x1) + f(group, model = "iid"),
        data = df,
        family = "gaussian"
    ),
    "Unsupported fixed-effect term"
)

character_df <- transform(df, char_group = as.character(promo))
expect_error_matching(
    build_backend_spec(
        y ~ 1 + char_group + f(group, model = "iid"),
        data = character_df,
        family = "gaussian"
    ),
    "convert it to a factor"
)

nonfinite_fixed_df <- df
nonfinite_fixed_df$x1[[2L]] <- Inf
expect_error_matching(
    build_backend_spec(
        y ~ 1 + x1 + f(group, model = "iid"),
        data = nonfinite_fixed_df,
        family = "gaussian"
    ),
    "Fixed-effects design matrix contains non-finite"
)

expect_error_matching(
    build_backend_spec(
        y ~ 1 + x1:f(group, model = "iid"),
        data = df,
        family = "gaussian"
    ),
    "Latent f\\(\\) terms must be standalone"
)

expect_error_matching(
    build_backend_spec(
        y ~ 1 + f(x1 + x2, model = "iid"),
        data = df,
        family = "gaussian"
    ),
    "f\\(\\) covariate must be a single"
)

expect_error_matching(
    build_backend_spec(
        y ~ 1 + f(group),
        data = df,
        family = "gaussian"
    ),
    "f\\(\\) model must be a literal"
)

expect_error_matching(
    build_backend_spec(
        y ~ 1 + f(group, "iid", model = "iid"),
        data = df,
        family = "gaussian"
    ),
    "model was supplied both"
)

expect_error_matching(
    build_backend_spec(
        y ~ 1 + f(group, model = "iid", constr = 1),
        data = df,
        family = "gaussian"
    ),
    "constr must be a literal"
)

expect_error_matching(
    build_backend_spec(
        y ~ 1 + f(group, model = "iid", hyper = list(prec = list(prior = "pc.prec"))),
        data = df,
        family = "gaussian"
    ),
    "Unsupported f\\(\\) argument"
)

nonfinite_formula_offset_df <- df
nonfinite_formula_offset_df$exposure[[1L]] <- 0
expect_error_matching(
    build_backend_spec(
        y ~ 1 + x1 + offset(log(exposure)) + f(group, model = "iid"),
        data = nonfinite_formula_offset_df,
        family = "gaussian"
    ),
    "Formula offset contains non-finite"
)

expect_error_matching(
    build_backend_spec(
        y ~ 1 + x1 + f(group, model = "iid"),
        data = df,
        family = "gaussian",
        offset = c(0, Inf, rep(0, nrow(df) - 2L))
    ),
    "offset contains non-finite"
)

expect_error_matching(
    build_backend_spec(
        y ~ 1 + x1 + f(group, model = "iid"),
        data = df,
        family = "gaussian",
        offset = c(0, 0)
    ),
    "offset length does not match"
)

fixed_only_spec <- build_backend_spec(
    y ~ 1 + x1 + promo + offset(log(exposure)),
    data = df,
    family = "gaussian"
)
fixed_only_expected <- model.matrix(
    ~ 1 + x1 + promo + offset(log(exposure)),
    data = df
)
fixed_only_matrix <- matrix(
    fixed_only_spec$fixed_matrix,
    nrow = nrow(df),
    ncol = fixed_only_spec$n_fixed,
    dimnames = list(NULL, fixed_only_spec$fixed_names)
)
stopifnot(identical(fixed_only_spec$n_latent, 0L))
stopifnot(identical(length(fixed_only_spec$latent_blocks), 0L))
stopifnot(identical(fixed_only_spec$n_fixed, as.integer(ncol(fixed_only_expected))))
stopifnot(identical(fixed_only_spec$fixed_names, colnames(fixed_only_expected)))
stopifnot(isTRUE(all.equal(
    unname(fixed_only_matrix),
    unname(fixed_only_expected),
    tolerance = 1e-12,
    check.attributes = FALSE
)))
stopifnot(isTRUE(all.equal(fixed_only_spec$offset, log(df$exposure), tolerance = 1e-12)))

expect_error_matching(
    build_backend_spec(
        y ~ 0,
        data = df,
        family = "gaussian"
    ),
    "at least one fixed-effect column or standalone f"
)

expect_error_matching(
    build_backend_spec(
        y ~ 1 + y + f(group, model = "iid"),
        data = df,
        family = "gaussian"
    ),
    "Response variable cannot also appear"
)
