# AR2 Extension Example

This note started as a worked example for how an `ar2` latent model would be
added on top of the current `rustyINLA` architecture. `ar2` is now implemented
in this branch, so this document is best read as a post-hoc walkthrough of the
touch points and design choices we used.

It is still intentionally schematic rather than line-by-line. `ar2` remains a
good extension example because it exercises the same surfaces as `rw2` while
introducing a different precision
parameterization.

## 1. Files to touch

For an `ar2` addition, the expected touch points are:

- `src/rust/inla_core/src/models/mod.rs`
- `src/rust/inla_core/src/graph/mod.rs`
- `src/rust/src/lib.rs`
- `R/f.R`
- `R/interface.R` only if `ar2` needs special front-end metadata or defaults
- `src/rust/inla_core/tests/test_basic.rs`
- one reference or benchmark harness under `scratch/`

That is the same high-level path we used for `rw2`, and it is now also the path
used by the shipped `ar2` implementation.

## 2. Core-model shape

The main implementation would live in
[src/rust/inla_core/src/models/mod.rs](C:/Users/Antonio/Documents/rustyINLA/rustyINLA/src/rust/inla_core/src/models/mod.rs).

Suggested skeleton:

```rust
pub struct Ar2Model {
    graph: Graph,
}

impl Ar2Model {
    pub fn new(n: usize) -> Self {
        Self {
            graph: Graph::rw2(n),
        }
    }
}

impl QFunc for Ar2Model {
    fn graph(&self) -> &Graph {
        &self.graph
    }

    fn eval(&self, i: usize, j: usize, theta: &[f64]) -> f64 {
        // theta[0] = log precision
        // theta[1], theta[2] = transformed AR2 coefficients
        // return the pentadiagonal precision entry Q(i, j)
        todo!()
    }

    fn n_hyperparams(&self) -> usize {
        3
    }
}
```

## 3. Graph choice

`ar2` is not a simple first-order chain like `ar1`. Its precision has
second-order temporal coupling, so the sparse pattern is expected to include
`i +/- 1` and `i +/- 2`.

That means the cleanest graph path is either:

- reuse `Graph::rw2(n)` directly, or
- add a semantic alias `Graph::ar2(n)` that delegates to the same
  second-order-chain neighbor pattern

If we want the public code to read more clearly, the alias is probably worth
it even if the structure is identical.

## 4. Hyperparameter contract

The Rust bridge registration would need to expand in
[src/rust/src/lib.rs](C:/Users/Antonio/Documents/rustyINLA/rustyINLA/src/rust/src/lib.rs):

- add `"ar2"` to `build_single_qfunc()`
- add default `theta_init` with length `3`
- add validation so `theta_init` length accounts for the extra AR2
  hyperparameter

The practical recommendation is:

- `theta[0]`: log marginal precision
- `theta[1]`, `theta[2]`: transformed AR coefficients on an unconstrained scale

The most important design choice is not the exact syntax. It is choosing a
parameterization whose inverse transform guarantees stationarity, so the
precision stays valid over unconstrained optimization.

## 5. R-facing registration

Expose the model name in [R/f.R](C:/Users/Antonio/Documents/rustyINLA/rustyINLA/R/f.R)
alongside the current supported set.

Minimal example:

```r
f(x, model = "ar2")
```

In many cases `R/interface.R` would not need special handling beyond default
support, because `ar2` does not require irregular-grid values the way `rw2`
does. If we later support non-unit temporal spacing for `ar2`, then we would
follow the same `structure_values` pattern introduced for `rw2`.

## 6. Tests to add

At minimum:

- model-level symmetry and derivative tests in
  `src/rust/inla_core/src/models/mod.rs`
- end-to-end Gaussian latent tests in
  `src/rust/inla_core/tests/test_basic.rs`
- one comparison against a dense reference precision or posterior

Good first checks:

- `rho1 = rho2 = 0` should reduce toward an iid-like interior structure
- the precision should be symmetric
- selected inverse diagonals should stay positive
- transformed hyperparameters should always map to a stable AR2 region

## 7. Benchmark template

The benchmark pattern should mirror the `rw2` rollout:

1. add a small core end-to-end test
2. add one reference harness case
3. compare fixed effects, random effects, fitted values, and hyperparameters
4. check that existing benchmark cases remain stable

Likely public reference candidates for a future `ar2` case:

- an INLA time-series smoothing example with `model = "ar2"`
- a synthetic Gaussian reference if a clean public exact dataset is not ready

## 8. Why this is a useful next example

`ar2` is a good extension template because it sits between the easy and hard
classes:

- easier than a graph-input model like `besag`
- harder than a pure chain model like `rw1`
- structurally close enough to `rw2` that the current extension path is already
  a useful blueprint

So the takeaway is:

- `rw2` showed the repo path for a new latent model end-to-end
- `ar2` followed the same path, but with a stationarity-safe proper
  precision instead of an intrinsic second-difference precision
