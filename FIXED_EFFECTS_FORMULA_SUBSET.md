# Fixed-Effects Formula Subset

Status: Phase 7A productization slice, 2026-04-25.

This document states the fixed-effects formula subset that `rustyINLA`
currently supports intentionally. The implementation still uses R's
`model.matrix()` for the actual dense fixed-effect design, but the front end
now validates a narrower public contract before building the backend spec.

## Supported

- A single response column on the left-hand side, such as `y ~ ...`.
- Fixed-effect-only GLMs, such as `y ~ 1 + x`, when the formula produces at
  least one fixed-effect design column.
- Fixed-effect intercept handling through normal R formula syntax:
  `1`, `0`, or `-1`.
- Fixed-effect columns that are already present in `data` and are numeric,
  integer, logical, or factor.
- Logical fixed-effect columns, encoded through the normal `model.matrix()`
  treatment.
- Multi-column factor expansions produced by `model.matrix()`, including
  factors with three or more levels.
- Simple fixed-effect interactions among bare data columns, including `:`,
  `*`, and formula expansions such as `(x1 + x2)^2` when they reduce to bare
  columns and their interactions. Interactions can include factors when the
  resulting design remains full rank.
- Formula offsets through `offset(...)`.
- The explicit `offset = ...` argument to `rusty_inla()`, provided it evaluates
  to a finite numeric vector with one value per observation.
- Standalone latent terms of the form
  `f(covariate, model = "iid" | "rw1" | "rw2" | "ar1" | "ar2", constr = TRUE/FALSE)`.

## Rejected For Now

- Formulas with neither a fixed-effect column nor a standalone latent term,
  such as `y ~ 0`.
- Transformed fixed terms such as `log(x)`, `I(x^2)`, `poly(x, 2)`,
  `factor(x)`, or spline bases. Create these columns in `data` first.
- Character fixed-effect columns. Convert them to factors explicitly.
- Rank-deficient or aliased fixed-effect designs.
- Non-finite fixed-effect design values or offsets.
- Fixed-effect factors with unused levels that create aliased/all-zero design
  columns.
- Latent `f()` terms inside interactions, such as `x:f(group, model = "iid")`.
- Transformed latent covariates, such as `f(log(time), model = "rw1")` or
  `f(x + z, model = "iid")`.
- R-INLA `f()` arguments outside the current subset, including `hyper`,
  `scale.model`, `graph`, `group`, `control.group`, and `order`.
- R-INLA surfaces such as `E`, `Ntrials`, custom links, user priors, graph
  inputs, and grouped effects.

## Validation Behavior

Unsupported formula shapes fail in `build_backend_spec()` before Rust is called.
The error should name the unsupported surface rather than allowing an obscure
`model.frame()`, `model.matrix()`, or solver failure.

The Rust bridge also validates hand-built backend specs for non-finite fixed
matrices, offsets, A-matrix values, constraints, and initial values.
Hand-built specs must include at least one fixed-effect design column or one
latent `f(...)` block.

## Test Coverage

The base-R regression script at `tests/fixed-effects-interface.R` covers:

- multiple fixed columns
- multi-level factor expansion
- logical fixed columns
- a simple fixed interaction
- formula offsets mixed with a latent term
- rank-deficient fixed designs
- rank-deficient factor expansions from unused levels
- character fixed columns
- non-finite fixed design values and offsets
- transformed fixed terms
- latent `f()` interactions
- transformed latent covariates
- missing, duplicated, or invalid literal `f()` arguments
- unsupported `f()` arguments
- fixed-effect-only formulas
- formulas with no fixed or latent terms

The public parity script at `tests/fixed-only-parity.R` compares
fixed-effect-only Gaussian, Poisson, and Gamma formulas against R-INLA through
the package-level `rusty_inla()` interface.
