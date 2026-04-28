source(file.path(getwd(), "tools", "load_worktree_package.R"), local = TRUE)
load_rustyinla_for_benchmarks(getwd())

named_column <- function(df, column) {
    if (is.null(df) || nrow(df) == 0L || !(column %in% names(df))) {
        return(stats::setNames(numeric(), character()))
    }
    stats::setNames(as.numeric(df[[column]]), rownames(df))
}

make_batch <- function(groups, levels, n_per_group, seed, shift = 0.0) {
    set.seed(seed)
    group <- rep(groups, each = n_per_group)
    x <- stats::rnorm(length(group))
    effects <- c("1" = -0.65, "2" = 0.25, "3" = 0.7, "4" = -0.15, "5" = 1.05)
    eta <- -0.5 + shift + 0.35 * x + unname(effects[as.character(group)])
    data.frame(
        y = stats::rpois(length(group), lambda = exp(eta)),
        x = x,
        group = factor(group, levels = levels)
    )
}

part1 <- make_batch(groups = 1:4, levels = 1:4, n_per_group = 60L, seed = 9401L)
part2 <- make_batch(groups = 1:5, levels = 1:5, n_per_group = 35L, seed = 9402L, shift = 0.03)
part3 <- make_batch(groups = 1:4, levels = 1:4, n_per_group = 45L, seed = 9403L, shift = 0.06)
joint123 <- rbind(
    transform(part1, group = factor(as.character(group), levels = 1:5)),
    part2,
    transform(part3, group = factor(as.character(group), levels = 1:5))
)

formula <- y ~ 1 + x + f(group, model = "iid")

fit1 <- rusty_inla(formula, data = part1, family = "poisson")
state1 <- rusty_update_state(fit1)
fit2 <- rusty_inla(
    formula,
    data = part2,
    family = "poisson",
    control.update = list(
        state = state1,
        mode = "fixed_iid_cross_theta_evidence"
    )
)
state12 <- rusty_compose_update_state(state1, fit2)
fit3 <- rusty_inla(
    formula,
    data = part3,
    family = "poisson",
    control.update = list(
        state = state12,
        mode = "fixed_iid_cross_theta_evidence"
    )
)
joint <- rusty_inla(formula, data = joint123, family = "poisson")

metadata <- fit3$posterior_state_used
random_update <- named_column(fit3$summary.random$group, "mean")
random_joint <- named_column(joint$summary.random$group, "mean")
dormant_diff <- abs(random_update[["5"]] - random_joint[["5"]])

cat(sprintf(
    paste(
        "posterior_state_dormant_iid_levels:",
        "dormant %s; carried %s; group5 update %.6f vs joint %.6f; abs diff %.6f\n"
    ),
    paste(metadata$dormant_iid_levels, collapse = ","),
    paste(metadata$carried_iid_levels, collapse = ","),
    random_update[["5"]],
    random_joint[["5"]],
    dormant_diff
))

if (!("5" %in% rownames(fit3$summary.random$group))) {
    stop("Dormant iid level was not retained in summary.random.", call. = FALSE)
}
if (!("5" %in% metadata$dormant_iid_levels)) {
    stop("Dormant iid level was not recorded in posterior_state_used metadata.", call. = FALSE)
}
if (!("5" %in% metadata$carried_iid_levels)) {
    stop("Dormant iid level was not added back to the update factor levels.", call. = FALSE)
}
if (!is.finite(dormant_diff) || dormant_diff > 0.25) {
    stop("Dormant iid level drift is larger than expected versus joint refit.", call. = FALSE)
}

cat("posterior_state_dormant_iid_levels: PASS\n")
