source_root <- if (file.exists(file.path("R", "interface.R"))) {
    "."
} else if (file.exists(file.path("..", "R", "interface.R"))) {
    ".."
} else {
    NA_character_
}

if (!is.na(source_root)) {
    loaded_worktree_package <- FALSE
    loader <- file.path(source_root, "tools", "load_worktree_package.R")
    if (file.exists(loader)) {
        source(loader, local = TRUE)
        loaded_worktree_package <- tryCatch(
            {
                load_rustyinla_for_benchmarks(normalizePath(source_root, winslash = "/"))
                TRUE
            },
            error = function(e) FALSE
        )
    }
    if (!loaded_worktree_package) {
        source(file.path(source_root, "R", "f.R"), local = FALSE)
        source(file.path(source_root, "R", "interface.R"), local = FALSE)
    }
} else {
    library(rustyINLA)
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
    y = c(1.0, 1.4, 1.1, 1.8, 1.2, 2.0, 1.5, 2.2),
    x1 = c(-1.0, -0.4, 0.2, 0.8, -0.7, 0.5, 1.1, -0.2),
    x2 = c(0.3, -0.1, 0.6, -0.5, 0.2, 0.9, -0.4, 0.7),
    promo = factor(
        c("base", "promo", "base", "promo", "base", "promo", "promo", "base"),
        levels = c("base", "promo")
    ),
    group = factor(c(1, 1, 2, 2, 3, 3, 4, 4)),
    exposure = c(0.8, 1.1, 1.0, 1.4, 0.9, 1.6, 1.2, 1.5)
)

expect_error_matching(
    rusty_inla(
        y ~ 1 + log(x1),
        data = df,
        family = "gaussian"
    ),
    "Unsupported fixed-effect term"
)

rank_deficient_df <- transform(df, x_dup = x1)
expect_error_matching(
    rusty_inla(
        y ~ 1 + x1 + x_dup,
        data = rank_deficient_df,
        family = "gaussian"
    ),
    "rank-deficient"
)

character_df <- transform(df, char_group = as.character(promo))
expect_error_matching(
    rusty_inla(
        y ~ 1 + char_group,
        data = character_df,
        family = "gaussian"
    ),
    "convert it to a factor"
)

nonfinite_fixed_df <- df
nonfinite_fixed_df$x1[[2L]] <- Inf
expect_error_matching(
    rusty_inla(
        y ~ 1 + x1,
        data = nonfinite_fixed_df,
        family = "gaussian"
    ),
    "Fixed-effects design matrix contains non-finite"
)

bad_offset <- c(0, Inf, rep(0, nrow(df) - 2L))
expect_error_matching(
    rusty_inla(
        y ~ 1 + x1,
        data = df,
        family = "poisson",
        offset = bad_offset
    ),
    "offset contains non-finite"
)

expect_error_matching(
    rusty_inla(
        y ~ 0,
        data = df,
        family = "gaussian"
    ),
    "at least one fixed-effect column or standalone f"
)

expect_error_matching(
    rusty_inla(
        y ~ 1 + f(group, model = "iid", hyper = list(prec = list(prior = "pc.prec"))),
        data = df,
        family = "gaussian"
    ),
    "Unsupported f\\(\\) argument"
)

cat("Public rusty_inla() formula error cases passed: 7/7\n")
