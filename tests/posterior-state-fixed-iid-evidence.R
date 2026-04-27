source(file.path(getwd(), "tools", "load_worktree_package.R"), local = TRUE)
load_rustyinla_for_benchmarks(getwd())

timed_fit <- function(expr) {
    elapsed <- system.time(value <- eval.parent(substitute(expr)))
    list(value = value, elapsed = unname(elapsed[["elapsed"]]))
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

make_batch <- function(groups, levels, n_per_group, seed) {
    set.seed(seed)
    group <- rep(groups, each = n_per_group)
    x <- stats::rnorm(length(group))
    effects <- c("1" = -0.75, "2" = 0.35, "3" = 0.8, "4" = -0.25, "5" = 1.2)
    eta <- -0.55 + 0.45 * x + unname(effects[as.character(group)])
    data.frame(
        y = stats::rpois(length(group), lambda = exp(eta)),
        x = x,
        group = factor(group, levels = levels)
    )
}

old_data <- make_batch(groups = 1:4, levels = 1:4, n_per_group = 90L, seed = 9101L)
new_data <- make_batch(groups = 1:5, levels = 1:5, n_per_group = 22L, seed = 9102L)
joint_data <- rbind(
    transform(old_data, group = factor(as.character(group), levels = 1:5)),
    new_data
)

formula <- y ~ 1 + x + f(group, model = "iid")

old <- timed_fit(rusty_inla(formula, data = old_data, family = "poisson"))
state <- rusty_update_state(old$value)
new_default <- timed_fit(rusty_inla(formula, data = new_data, family = "poisson"))
new_diag <- timed_fit(rusty_inla(
    formula,
    data = new_data,
    family = "poisson",
    control.update = list(
        state = state,
        mode = "fixed_iid_gaussian_evidence"
    )
))
new_cross <- timed_fit(rusty_inla(
    formula,
    data = new_data,
    family = "poisson",
    control.update = list(
        state = state,
        mode = "fixed_iid_cross_gaussian_evidence"
    )
))
joint <- timed_fit(rusty_inla(formula, data = joint_data, family = "poisson"))

joint_new_rows <- seq.int(nrow(old_data) + 1L, nrow(joint_data))
fixed_default <- named_column(new_default$value$summary.fixed, "mean")
fixed_diag <- named_column(new_diag$value$summary.fixed, "mean")
fixed_cross <- named_column(new_cross$value$summary.fixed, "mean")
fixed_joint <- named_column(joint$value$summary.fixed, "mean")
fixed_sd_diag <- named_column(new_diag$value$summary.fixed, "sd")
fixed_sd_cross <- named_column(new_cross$value$summary.fixed, "sd")
fixed_sd_joint <- named_column(joint$value$summary.fixed, "sd")

random_default <- named_column(new_default$value$summary.random$group, "mean")
random_diag <- named_column(new_diag$value$summary.random$group, "mean")
random_cross <- named_column(new_cross$value$summary.random$group, "mean")
random_joint <- named_column(joint$value$summary.random$group, "mean")
random_sd_diag <- named_column(new_diag$value$summary.random$group, "sd")
random_sd_cross <- named_column(new_cross$value$summary.random$group, "sd")
random_sd_joint <- named_column(joint$value$summary.random$group, "sd")

fixed_default_diff <- max_abs_named_diff(fixed_default, fixed_joint)
fixed_diag_diff <- max_abs_named_diff(fixed_diag, fixed_joint)
fixed_cross_diff <- max_abs_named_diff(fixed_cross, fixed_joint)
random_default_diff <- max_abs_named_diff(random_default, random_joint)
random_diag_diff <- max_abs_named_diff(random_diag, random_joint)
random_cross_diff <- max_abs_named_diff(random_cross, random_joint)
fitted_default_diff <- max_fitted_rel_diff(new_default$value, joint$value, joint_new_rows)
fitted_diag_diff <- max_fitted_rel_diff(new_diag$value, joint$value, joint_new_rows)
fitted_cross_diff <- max_fitted_rel_diff(new_cross$value, joint$value, joint_new_rows)
fixed_sd_diag_min_ratio <- min_named_ratio(fixed_sd_diag, fixed_sd_joint)
fixed_sd_cross_min_ratio <- min_named_ratio(fixed_sd_cross, fixed_sd_joint)
random_sd_diag_min_ratio <- min_named_ratio(random_sd_diag, random_sd_joint)
random_sd_cross_min_ratio <- min_named_ratio(random_sd_cross, random_sd_joint)

cat(sprintf(
    paste(
        "posterior_state_fixed_iid_evidence:",
        "old %.3fs, new_default %.3fs, new_diag %.3fs, new_cross %.3fs, joint %.3fs\n"
    ),
    old$elapsed,
    new_default$elapsed,
    new_diag$elapsed,
    new_cross$elapsed,
    joint$elapsed
))
cat(sprintf(
    paste(
        "accuracy vs joint:",
        "fixed default %.6f -> diag %.6f -> cross %.6f;",
        "random default %.6f -> diag %.6f -> cross %.6f;",
        "fitted default %.6f -> diag %.6f -> cross %.6f\n"
    ),
    fixed_default_diff,
    fixed_diag_diff,
    fixed_cross_diff,
    random_default_diff,
    random_diag_diff,
    random_cross_diff,
    fitted_default_diff,
    fitted_diag_diff,
    fitted_cross_diff
))
cat(sprintf(
    "sd ratios vs joint: fixed diag %.6f -> cross %.6f; random diag %.6f -> cross %.6f\n",
    fixed_sd_diag_min_ratio,
    fixed_sd_cross_min_ratio,
    random_sd_diag_min_ratio,
    random_sd_cross_min_ratio
))
cat(sprintf(
    "born level evidence: count %d, posterior mean %.6f\n",
    length(new_cross$value$posterior_state_used$born_iid_levels),
    random_cross[["5"]]
))

if (!inherits(state, "rusty_update_state")) {
    stop("rusty_update_state() did not return the expected S3 class.", call. = FALSE)
}
if (is.null(new_diag$value$posterior_state_used) || is.null(new_cross$value$posterior_state_used)) {
    stop("Evidence update fits did not record posterior_state_used metadata.", call. = FALSE)
}
if (!("5" %in% new_cross$value$posterior_state_used$born_iid_levels)) {
    stop("Born iid level was not recorded as a zero-evidence new level.", call. = FALSE)
}
if (fitted_cross_diff >= fitted_default_diff) {
    stop("Fixed+iid cross evidence update did not improve fitted values versus the joint refit.", call. = FALSE)
}
if (fixed_sd_cross_min_ratio < 0.25 || random_sd_cross_min_ratio < 0.25) {
    stop("Fixed+iid cross evidence update narrowed posterior SDs too aggressively versus the joint refit.", call. = FALSE)
}

cat("posterior_state_fixed_iid_evidence: PASS\n")
