# R-INLA Output Apples-to-Apples Audit

Date: 2026-04-18

This note answers a narrow question:

If we want memory comparisons between `rustyINLA` and `R-INLA` to be more
"apples to apples", which returned objects matter, which are easy to mirror,
and where do we hit a real wall?

## Benchmark Context Used for the Audit

Public reference case:

- `spData::nc.sids`
- formula:
  `SID74 ~ 1 + offset(log(BIR74)) + f(CNTY.ID, model = "iid")`
- `family = "poisson"`

R-INLA call used for the heavier object audit:

```r
inla(
  SID74 ~ 1 + offset(log(BIR74)) + f(CNTY.ID, model = "iid"),
  data = nc.sids,
  family = "poisson",
  control.compute = list(config = TRUE, dic = TRUE, waic = TRUE, cpo = TRUE),
  control.predictor = list(compute = TRUE),
  num.threads = 1
)
```

## Observed Returned Object Sizes

On this case:

- `object.size(rusty_fit) ~= 0.047 MB`
- `object.size(rinla_fit) ~= 1.305 MB`

So `R-INLA` is about `27.8x` larger on the returned fit object alone.

## Largest Top-Level R-INLA Components

Approximate size contribution:

- `.args`: `0.694 MB`
- `misc`: `0.260 MB`
- `marginals.random`: `0.128 MB`
- `logfile`: `0.110 MB`
- `all.hyper`: `0.028 MB`
- `summary.fitted.values`: `0.014 MB`
- `summary.linear.predictor`: `0.014 MB`
- `model.matrix`: `0.009 MB`
- `summary.random`: `0.008 MB`
- `dic`: `0.005 MB`
- `cpo`: `0.003 MB`
- `waic`: `0.002 MB`

The main result is that the memory gap is **not** explained only by Rust vs R.
It is also explained by the fact that `R-INLA` returns substantially more state.

## What R-INLA Is Storing

### 1. `.args`

This is mostly bookkeeping from the original call and defaults.

Largest subfields:

- `control.lp.scale`: `0.633 MB`
- `data`: `0.020 MB`
- `control.inla`: `0.010 MB`
- `control.predictor`: `0.007 MB`
- `control.compute`: `0.005 MB`

This is not the posterior itself. It is mostly argument and control-state
retention.

### 2. `misc`

Largest subfields:

- `configs`: `0.249 MB`
- `opt.trace`: `0.004 MB`
- everything else is much smaller

Inside `misc$configs`, the largest object is:

- `config`: `0.241 MB`

Inside one stored config, the largest items are:

- `Q`: `0.0041 MB`
- `Qinv`: `0.0041 MB`
- `Qprior`: `0.0030 MB`
- `cpodens.moments`: `0.0029 MB`
- `ll.info`: `0.0029 MB`
- `Predictor`: `0.0021 MB`

So `misc$configs` is an actual posterior/integration-state dump, not just
pretty output.

### 3. Marginals

`R-INLA` stores:

- `marginals.fixed`
- `marginals.random`
- `marginals.hyperpar`

but in this audit call it does **not** store:

- `marginals.linear.predictor`
- `marginals.fitted.values`

So even this "heavier" R-INLA object is still not the absolute maximum object
size possible.

## What rustyINLA Currently Returns

Current `rustyINLA` output construction is in:

- `R/interface.R`
- `src/rust/src/lib.rs`

The current fit object is intentionally thin:

- `call`
- `formula`
- `data`
- `family`
- `offset`
- `offset_arg_provided`
- `mlik`
- `summary.fixed`
- `summary.random`
- `summary.fitted.values`
- `summary.hyperpar`
- `diagnostics`
- `theta_init_used`
- `laplace_terms`
- `mode`

Important implementation detail:

- the Rust core already computes full latent/fitted `Marginal` objects in
  `src/rust/inla_core/src/inference/mod.rs`
- but the bridge in `src/rust/src/lib.rs` only exports:
  - random means and variances
  - fitted summary numbers

That means some missing parity items are a **bridge/output** issue, not a core
inference issue.

## Feasibility Map

### Cheap to Add

These should not be a real memory wall and are mostly output/bridge work:

- `marginals.random`
  - We already build these in Rust.
- `marginals.fixed`
  - Can be exported as Gaussian marginals from current mean/sd.
- `.args` or an `.args-lite`
  - Mostly R-side call bookkeeping.
- `summary.linear.predictor`
  - We already have `mode$eta` and latent variances.
- `model.matrix`
  - R-side reconstruction.
- `version` / light metadata.

These would make comparisons much fairer without changing the core algorithm.

### Medium Cost

These are feasible but add either implementation complexity or noticeable
memory:

- `marginals.fitted.values`
  - We already have fitted marginals in Rust, but exporting all of them can get
    large.
- `marginals.hyperpar`
  - Requires constructing proper weighted CCD-based one-dimensional marginals.
- `dic`, `waic`, `cpo`, `po`, `gcpo`, `residuals`
  - Not a memory wall, but more statistics to implement/validate.
- `logfile`
  - Easy to mimic structurally, though not especially useful scientifically.

### Real Wall / Bad Trade Unless Explicitly Requested

These are the places where "matching R-INLA output" becomes expensive enough
that we should probably keep them optional:

#### 1. `misc$configs`

This is the clearest wall.

It stores:

- configuration-level posterior state
- `Q`, `Qinv`, `Qprior`
- predictor-related internal objects
- CPO density moments and likelihood info

This is tightly coupled to the integration/config machinery and grows with the
size of the model state. Matching it exactly is possible in principle, but it
is not just "formatting output". It is storing a substantial internal object
graph.

#### 2. Full Per-Observation Marginals at Portfolio Scale

Rust currently uses `75` grid points per marginal by default.

One marginal stores:

- `x`: 75 doubles
- `y`: 75 doubles

Raw numeric storage only:

- `75 * 2 * 8 = 1200 bytes` per marginal

For `freMTPL2freq`:

- `n = 677,991`
- fitted marginals alone would be about:
  `677,991 * 1200 bytes ~= 776 MiB` raw numeric storage

That is **before** R list overhead, vector headers, and duplication effects.

So:

- full `marginals.fitted.values` on the frequency benchmarks is an
  approximately-GB-class feature
- full `marginals.linear.predictor` would cost roughly the same again

This is a genuine memory wall if enabled by default.

#### 3. Exact R-INLA-Style Config Replay Objects

Anything that requires storing or replaying:

- configuration-local precision matrices
- selected inverses
- full predictor state per config
- internal optimization traces for every config

will move us from "lightweight API result" toward "store the whole engine
state".

## Practical Recommendation

If we want a fairer memory benchmark, the best next target is not
"match everything R-INLA returns".

It is:

### Profile A: Thin

Current `rustyINLA` object.

### Profile B: Benchmark Apples-to-Apples

Add these:

- `marginals.random`
- `marginals.fixed`
- `summary.linear.predictor`
- `model.matrix`
- `.args-lite`
- optional `dic`/`waic`/`cpo` family once implemented

This would cover the scientifically relevant summaries without forcing us to
store GB-scale per-observation marginals or full config internals.

### Profile C: Full Debug / Research

Optional only:

- `marginals.fitted.values`
- `marginals.linear.predictor`
- `misc$configs`-like integration state
- full trace/log objects

This should be explicitly opt-in because this is where the memory wall lives.

## Bottom Line

The current memory win is real, but it is driven by **both**:

- Rust-side efficiency
- thinner returned objects

We are **not** currently comparing identical output contracts.

The largest missing pieces are not all equally important:

- some are easy and worth adding
- some are medium-cost and should be optional
- some, especially `misc$configs` and full per-observation marginals on large
  datasets, are the real wall

Those should remain opt-in debug/research features, not default output.
