# rust-inla — Session Context

Paste this at the start of every new session.

Repository: https://github.com/Gilgamesh88/rust-inla
Language policy: **English only** — all code, comments, commits, docs.

---

## Current state (April 2026)

**84 tests passing, 0 failed, 0 warnings. Repo pushed to main.**

Global parity with R-INLA: **~47%**

---

## What was completed this session

### Compilation blockers fixed
- `problem/mod.rs` — replaced undefined `x_idx` with `a_j.map_or(i, |aj| aj[i])`
- `optimizer/mod.rs` — `laplace_gradient` now uses `&InlaModel`, `optimize()` builds `InlaModel` internally
- `inference/mod.rs` — fixed invalid `let lat_idx = model.a_i, model.a_j, ...` comma syntax

### Logic bugs fixed
- `optimizer/ccd.rs` — `STAR_FACTOR = √(4/3) ≈ 1.1547` (was hardcoded `1.0`, caused collapsed weights)
- `likelihood/mod.rs` — ZIP negative curvature guard added (fallback to expected Fisher info)
- `likelihood/mod.rs` — Tweedie dead code `expected_info = false` removed
- `likelihood/mod.rs` — Tweedie `p_power` clamped to `(1+1e-6, 2-1e-6)` to prevent singularity

### New in this session (from Gemini sandbox, reviewed and merged)
- `likelihood/mod.rs` — ZIP (zeroinflatedpoisson1) and Tweedie with analytical derivatives
- `problem/mod.rs` — `find_mode_with_fixed_effects()` via Schur complement (k fixed effects)
- `optimizer/ccd.rs` — CCD grid over θ: 1 + 2D + 2^D points, Jacobi eigendecomposition
- `inference/mod.rs` — `InlaModel` struct, CCD mixture synthesis (law of total variance)
- `bindings.rs` — R wrapper via extendr: `rust_inla_run()` dispatches 5 likelihoods, 3 models

---

## Parity table

| Component | Parity | Notes |
|---|---|---|
| Cholesky + Takahashi | 95% | Validated |
| Graph + AMD | 90% | Complete |
| IRLS + find_mode | 85% | Schur complement for fixed effects |
| Fixed effects / intercept | 75% | Schur path in laplace_eval |
| L-BFGS optimizer | 75% | Analytical gradient |
| Marginals (zmarginal, emarginal) | 70% | Complete |
| Laplace approximation | 75% | Intercept in objective |
| CCD θ integration | 60% | Fixed star factor, needs fixture validation |
| Likelihoods (5 of ~40) | 13% | Gaussian, Poisson, Gamma, ZIP, Tweedie |
| R bindings (extendr) | 40% | rust_inla_run() functional |
| GMRF models (3 of ~30) | 10% | iid, rw1, ar1 |
| DIC / WAIC / CPO | 0% | Not started |
| Python bindings | 0% | Phase E |
| **GLOBAL** | **~47%** | |

---

## Immediate next task — fixture validation

Run this once to generate reference JSON files:

```powershell
Rscript generate_fixtures.R
```

Requires R-INLA:
```r
install.packages("INLA", repos = c(INLA = "https://inla.r-inla-download.org/R/stable"))
```

Then run:
```powershell
cargo test
```

The 2 currently-ignored tests will activate automatically once the fixtures exist:
- `fixture_cholesky_log_det_and_solve` — validates solver
- `fixture_cholesky_rw1_graph_structure` — validates graph

Also check CCD weights are now distributed (not collapsed to 0,0,1):
```powershell
cargo test optimizer -- --nocapture 2>&1 | Select-String "ccd"
```

---

## Architecture: how pieces connect

```
InlaEngine::run(model, params)              inference/mod.rs
  ├─► optimizer::optimize()                optimizer/mod.rs
  │     └─► L-BFGS → laplace_eval(problem, &InlaModel, theta)
  │               ├─► find_mode_with_fixed_effects()  if n_fixed > 0
  │               │     └─► Schur complement IRLS     problem/mod.rs
  │               └─► find_mode_with_inverse()        if n_fixed == 0
  │                     └─► AugmentedQFunc(Q+W) IRLS  problem/mod.rs
  │                           └─► FaerSolver (Cholesky + Takahashi)
  ├─► ccd::build_ccd_grid()               optimizer/ccd.rs
  │     ├─► finite-diff Hessian at θ*
  │     ├─► Jacobi eigendecomposition
  │     └─► 1 + 2D + 2^D points, STAR_FACTOR=√(4/3)
  └─► CCD mixture (law of total variance)  inference/mod.rs
        └─► Marginal structs → R via extendr bindings.rs
```

---

## Next priorities (in order)

1. **Run fixtures** — validate CCD weights and marginal means against R-INLA
2. **NegativeBinomial likelihood** — most requested after Poisson, needed for IBNR
3. **LogNormal likelihood** — trivial (jacobian only), needed for severity models
4. **rw2 model** — 2 constraints, needed for smoothing examples on r-inla.org
5. **BESAG (ICAR) model** — needed for spatial examples (Scotland lip cancer)
6. **DIC / WAIC** — trivial once marginals work

---

## Known remaining bugs (logic, not compilation)

- CCD: factorial point loop has a variable shadowing issue (`mut i` reused in inner loop).
  See `optimizer/ccd.rs` around the `for combo in 0..n_factorial` block.
  The fix in this session used `(combo >> k) & 1` which is correct — verify with fixture.

- `a_i` and `a_x` are accepted but unused in `find_mode_with_inverse` (prefixed `_a_i`, `_a_x`).
  Full A-matrix support (non-identity projection) is not implemented yet.
  Currently only identity mapping (observation i → latent i) is supported.

---

## Tooling notes

- Windows PowerShell: use `Select-String` instead of `grep`
- Line endings: `git config core.autocrlf true` silences LF/CRLF warnings
- Antigravity (Google IDE): use **Planning mode** and provide this file as context
- If switching to Gemini mid-session: provide this file + ROADMAP.md as context

---

## Session startup commands

```powershell
cd C:\Users\Antonio\rust-inla
git pull
cargo check
cargo test
```
