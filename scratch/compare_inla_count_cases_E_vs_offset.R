suppressPackageStartupMessages({
    library(INLA)
    library(CASdatasets)
})

options(digits = 15)

data(freMTPL2freq)

df <- freMTPL2freq
df$AgeGroup <- cut(
    df$DrivAge,
    breaks = c(17, 25, 40, 60, 80, 150),
    labels = c("18-25", "26-40", "41-60", "61-80", "81+"),
    ordered_result = TRUE
)
df$AgeIndex <- as.integer(df$AgeGroup)
df$VehBrand <- as.factor(df$VehBrand)
df$Region <- as.factor(df$Region)

ctrl_compute <- list(config = FALSE, dic = FALSE, waic = FALSE, cpo = FALSE)
ctrl_inla <- list(strategy = "auto", int.strategy = "auto")
ctrl_predictor <- list(compute = TRUE)

capture_fit_row <- function(case, path, fit, elapsed_sec) {
    data.frame(
        case = case,
        path = path,
        elapsed_sec = elapsed_sec,
        intercept = fit$summary.fixed["(Intercept)", "mean"],
        hyper_1 = if (nrow(fit$summary.hyperpar) >= 1) fit$summary.hyperpar[1, "mean"] else NA_real_,
        hyper_2 = if (nrow(fit$summary.hyperpar) >= 2) fit$summary.hyperpar[2, "mean"] else NA_real_,
        mlik = fit$mlik[1, 1],
        strategy = fit$.args$control.inla$strategy,
        int_strategy = fit$.args$control.inla$int.strategy,
        stringsAsFactors = FALSE
    )
}

run_case <- function(case, offset_expr, e_expr = NULL) {
    rows <- list()

    offset_time <- system.time(offset_fit <- eval(offset_expr))
    rows[[1]] <- capture_fit_row(
        case = case,
        path = "offset(log(E))",
        fit = offset_fit,
        elapsed_sec = unname(offset_time["elapsed"])
    )

    if (!is.null(e_expr)) {
        rows[[2]] <- tryCatch({
            e_time <- system.time(e_fit <- eval(e_expr))
            capture_fit_row(
                case = case,
                path = "E=",
                fit = e_fit,
                elapsed_sec = unname(e_time["elapsed"])
            )
        }, error = function(err) {
            data.frame(
                case = case,
                path = "E=",
                elapsed_sec = NA_real_,
                intercept = NA_real_,
                hyper_1 = NA_real_,
                hyper_2 = NA_real_,
                mlik = NA_real_,
                strategy = NA_character_,
                int_strategy = paste("ERROR:", conditionMessage(err)),
                stringsAsFactors = FALSE
            )
        })
    }

    do.call(rbind, rows)
}

results <- do.call(rbind, list(
    run_case(
        case = "poisson_multi_iid",
        offset_expr = quote(inla(
            ClaimNb ~ 1 + offset(log(Exposure)) +
                f(VehBrand, model = "iid") +
                f(Region, model = "iid"),
            family = "poisson",
            data = df,
            control.compute = ctrl_compute,
            control.predictor = ctrl_predictor,
            control.inla = ctrl_inla,
            num.threads = 1,
            verbose = FALSE
        )),
        e_expr = quote(inla(
            ClaimNb ~ 1 +
                f(VehBrand, model = "iid") +
                f(Region, model = "iid"),
            family = "poisson",
            data = df,
            E = Exposure,
            control.compute = ctrl_compute,
            control.predictor = ctrl_predictor,
            control.inla = ctrl_inla,
            num.threads = 1,
            verbose = FALSE
        ))
    ),
    run_case(
        case = "poisson_ar1",
        offset_expr = quote(inla(
            ClaimNb ~ 1 + offset(log(Exposure)) + f(AgeIndex, model = "ar1"),
            family = "poisson",
            data = df,
            control.compute = ctrl_compute,
            control.predictor = ctrl_predictor,
            control.inla = ctrl_inla,
            num.threads = 1,
            verbose = FALSE
        )),
        e_expr = quote(inla(
            ClaimNb ~ 1 + f(AgeIndex, model = "ar1"),
            family = "poisson",
            data = df,
            E = Exposure,
            control.compute = ctrl_compute,
            control.predictor = ctrl_predictor,
            control.inla = ctrl_inla,
            num.threads = 1,
            verbose = FALSE
        ))
    ),
    run_case(
        case = "zip_iid",
        offset_expr = quote(inla(
            ClaimNb ~ 1 + offset(log(Exposure)) + f(VehBrand, model = "iid"),
            family = "zeroinflatedpoisson1",
            data = df,
            control.compute = ctrl_compute,
            control.predictor = ctrl_predictor,
            control.inla = ctrl_inla,
            num.threads = 1,
            verbose = FALSE
        )),
        e_expr = quote(inla(
            ClaimNb ~ 1 + f(VehBrand, model = "iid"),
            family = "zeroinflatedpoisson1",
            data = df,
            E = Exposure,
            control.compute = ctrl_compute,
            control.predictor = ctrl_predictor,
            control.inla = ctrl_inla,
            num.threads = 1,
            verbose = FALSE
        ))
    )
))

cat("INLA count benchmark cases: offset(log(E)) versus E=\n")
print(results, row.names = FALSE)

cat("\nDifferences (offset - E) by case\n")
for (case_name in unique(results$case)) {
    sub <- results[results$case == case_name, ]
    if (nrow(sub) == 2 && all(is.finite(sub$elapsed_sec))) {
        cat(case_name, "\n")
        cat(
            "  elapsed_sec =",
            sub$elapsed_sec[sub$path == "offset(log(E))"] - sub$elapsed_sec[sub$path == "E="],
            "\n"
        )
        cat(
            "  intercept   =",
            sub$intercept[sub$path == "offset(log(E))"] - sub$intercept[sub$path == "E="],
            "\n"
        )
        cat(
            "  hyper_1     =",
            sub$hyper_1[sub$path == "offset(log(E))"] - sub$hyper_1[sub$path == "E="],
            "\n"
        )
        cat(
            "  hyper_2     =",
            sub$hyper_2[sub$path == "offset(log(E))"] - sub$hyper_2[sub$path == "E="],
            "\n"
        )
        cat(
            "  mlik        =",
            sub$mlik[sub$path == "offset(log(E))"] - sub$mlik[sub$path == "E="],
            "\n"
        )
    }
}
