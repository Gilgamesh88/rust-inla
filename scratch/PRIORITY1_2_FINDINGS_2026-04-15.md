# Priority 1 and 2 Findings for April 15, 2026

## Priority 1: How `R-INLA-devel` handles `poisson + iid + offset`

### Source inspection

- `rinla/R/create.data.file.R` sends `E` through the Poisson-family response payload as `cbind(ind, E, y.orig)`.
- `rinla/R/inla.R` writes predictor offsets separately through `file.offset`.
- `rinla/R/sections.R` shows that `logoffset` is a dedicated link model with `variant` handling, not just a generic offset alias.
- `rinla/R/models.R` confirms the default hyper initials that matter for the current benchmark family set:
  - `iid`: precision initial `4`
  - `ar1`: precision initial `4`, correlation initial `2`
  - `zeroinflatedpoisson1`: initial `-1`
  - `tweedie`: initials `(0, -4)`
  - `poisson`: no likelihood hyperparameters
- `rinla/R/set.default.arguments.R` keeps `control.inla(strategy = "auto", int.strategy = "auto")` by default.

### Empirical check on the full `freMTPL2freq` Poisson IID case

Using installed `INLA_25.10.19` on:

- `ClaimNb ~ 1 + offset(log(Exposure)) + f(VehBrand, model = "iid")`
- `ClaimNb ~ 1 + f(VehBrand, model = "iid")` with `E = Exposure`

Results:

| Path | Elapsed | Intercept | Precision mean | mlik |
|---|---:|---:|---:|---:|
| `offset(log(E))` | `10.64s` | `-2.56233` | `91.45727` | `-111209.5` |
| `E = Exposure` | `8.19s` | `-2.56233` | `91.45726` | `-111209.5` |

Observed differences (`offset - E`):

- intercept: `-3.66e-08`
- precision mean: `9.79e-07`
- marginal log-likelihood: `-1.35e-05`

Interpretation:

- For this benchmark, `E=` and `offset(log(E))` are numerically equivalent to practical precision.
- The `E=` path is still faster in `R-INLA-devel`, likely because it is routed through the likelihood response encoding instead of only the predictor-offset path.

## Priority 2: Rust slowdown instrumentation

Added diagnostics now report:

- optimizer outer iterations
- total `laplace_eval` calls
- split of optimizer vs CCD `laplace_eval` calls
- line-search trial evaluations
- coordinate-probe calls and evaluations
- factorization count
- selected-inverse count
- total time in:
  - optimizer
  - CCD
  - latent mode solves
  - likelihood/gradient assembly
  - sparse factorization
  - selected inverse

### Focused Poisson IID run (`scratch/profile_poisson_iid_theta.R`)

#### Fixed-theta profile (`optimizer_max_evals = 0`)

Common pattern across theta starts `2:6`:

- `outer_iters = 0`
- `laplace_total = 9`
- `laplace_opt = 3`
- `laplace_ccd = 6`
- `factorizations ~= 191-199`
- `selected_inverse = 21`
- `optimizer_sec ~= 1.6-2.0`
- `ccd_sec ~= 4.1-4.3`
- `latent_mode_sec ~= 8.1-8.5`
- `assembly_sec ~= 7.5-7.8`
- `factorization_sec ~= 0.00030-0.00034`

Interpretation:

- Even without outer iterations, the run already spends most of its time in repeated latent-mode work.
- Raw numeric factorization time is tiny relative to total runtime.

#### Optimization from multiple starts

Key starts:

| theta start | theta opt | precision opt | n_evals | outer iters | laplace total | line trials | coord evals | factorizations | optimizer sec | assembly sec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `4` | `4.15576` | `63.80` | `70` | `3` | `143` | `54` | `12` | `1665` | `69.10s` | `61.93s` |
| `5` | `4.18277` | `65.55` | `66` | `3` | `135` | `50` | `12` | `1574` | `63.47s` | `57.41s` |
| `2` | `4.15576` | `63.80` | `124` | `5` | `249` | `94` | `24` | `2830` | `117.95s` | `100.81s` |

Interpretation:

- The slowdown is being driven by repeated optimizer trial evaluations, not by the raw Cholesky kernel itself.
- Each optimizer run fans out into many `laplace_eval` calls, which in turn trigger large numbers of latent mode solves and selected inverses.
- The dominant measured bucket is likelihood/gradient assembly inside repeated latent mode solves.
- Sparse factorization time is measurable but not the primary bottleneck in the current implementation.

## Immediate next target

The next improvement pass should reduce repeated inner work per optimizer trial, especially:

- repeated full mode solves during line-search trials
- repeated selected-inverse computation in trial evaluations
- repeated likelihood/assembly work that is nearly identical across warm/cold trial checks
