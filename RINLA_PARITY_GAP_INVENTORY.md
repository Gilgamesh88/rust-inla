# R-INLA Parity Gap Inventory

This document is the best practical answer to "what is still missing relative to R-INLA?"

It combines:

- a live registry snapshot from the local `INLA` installation
- the current implemented `rustyINLA` subset
- the larger interface and workflow gaps that are not fully described by `inla.list.models()`

This is not yet a function-by-function inventory of every helper exported by the R-INLA package. It is a model-registry and subsystem-level parity inventory, which is the most useful level for planning future development.

Snapshot used for this inventory:

- local package: `INLA 25.10.19`
- local query method: `capture.output(INLA::inla.list.models())`
- snapshot date: `2026-04-19`

For the architectural intervention map behind these gaps, see [EXTENSION_INTERVENTION_MAP.md](EXTENSION_INTERVENTION_MAP.md). For the current recommended implementation order, see [EXTENSION_BACKLOG.md](EXTENSION_BACKLOG.md). For the split coverage evaluation and public API-surface inventory, see [COVERAGE_EVALUATION_2026-04-19.md](COVERAGE_EVALUATION_2026-04-19.md) and [RINLA_API_SURFACE_INVENTORY.md](RINLA_API_SURFACE_INVENTORY.md).

## 1. Headline parity counts

| Surface | R-INLA registry count | Current rustyINLA status | Gap summary |
| --- | ---: | --- | --- |
| likelihood families | `105` | `5` implemented: `gaussian`, `poisson`, `gamma`, `zeroinflatedpoisson1`, `tweedie` | `100` missing |
| latent models | `52` | `4` direct registry names implemented: `iid`, `rw1`, `rw2`, `ar1`; plus `ar2` as a dedicated `ar(order = 2)` surface | `48` direct registry names still missing, and `ar` remains only partially covered |
| group models | `8` | no dedicated parity surface | effectively missing as a subsystem |
| hazard models | `3` | no dedicated parity surface | effectively missing as a subsystem |
| link registry | `27` | family-default links exist, but no explicit public link registry parity | mostly missing as a public feature |
| prior registry | `48` | a few hardcoded defaults exist, but no public prior registry parity | mostly missing as a public feature |
| mixture registry | `3` | none | missing |
| `scopy` registry | `2` | none | missing |
| wrapper registry | `1` | none | missing |
| predictor registry | `1` | internal predictor path only | missing as a public feature |

## 2. Current implemented subset

These are the model-family and latent-model surfaces already present in the Rust core:

- likelihoods: `gaussian`, `poisson`, `gamma`, `zeroinflatedpoisson1`, `tweedie`
- latent models: `iid`, `rw1`, `rw2`, `ar1`, `ar2`

Important nuance:

- `tweedie` is implemented in the Rust codebase, but still outside the current stable MVP benchmark suite
- some internal priors and default links exist, but there is no R-INLA-style public registry parity for priors or links yet

## 3. Missing latent models from the R-INLA registry

### Main categories of missing latent-model parity

Lower-friction additions that fit the current architecture relatively well:

- `ar`
- `ou`
- `seasonal`
- `prw2`

Important nuance:

- `rw2` is no longer missing
- `ar2` now exists in `rustyINLA` as a dedicated latent-model surface, but `R-INLA` exposes this through the broader `ar` registry entry with `order = 2`, so `ar` is still only partially covered from a registry-parity perspective

Graph-driven or spatial additions that likely need public API widening:

- `besag`
- `besagproper`
- `besagproper2`
- `besag2`
- `bym`
- `bym2`
- `slm`

Larger architecture or workflow milestones:

- `spde`
- `spde2`
- `spde3`
- `dmatern`
- `matern2d`
- `rw2d`
- `rw2diid`

Generic, copy, multivariate, measurement-error, or nonlinear component surfaces that are also currently missing:

- `cgeneric`, `rgeneric`
- `copy`, `scopy`
- `generic`, `generic0`, `generic1`, `generic2`, `generic3`
- `iid1d`, `iid2d`, `iid3d`, `iid4d`, `iid5d`, `iidkd`
- `intslope`
- `meb`, `mec`
- `log1exp`, `logdist`, `sigm`, `revsigm`
- `linear`, `clinear`, `z`

### Exact missing latent-model names

`2diid`, `ar`, `ar1c`, `besag`, `besag2`, `besagproper`, `besagproper2`, `bym`, `bym2`, `cgeneric`, `clinear`, `copy`, `crw2`, `dmatern`, `fgn`, `fgn2`, `generic`, `generic0`, `generic1`, `generic2`, `generic3`, `iid1d`, `iid2d`, `iid3d`, `iid4d`, `iid5d`, `iidkd`, `intslope`, `linear`, `log1exp`, `logdist`, `matern2d`, `meb`, `mec`, `ou`, `prw2`, `revsigm`, `rgeneric`, `rw2`, `rw2d`, `rw2diid`, `scopy`, `seasonal`, `sigm`, `slm`, `spde`, `spde2`, `spde3`, `z`

## 4. Missing likelihood families from the R-INLA registry

### Main categories of missing likelihood parity

Likely lower-friction next families:

- `binomial`
- `nbinomial`
- `nbinomial2`
- `exponential`
- `weibull`
- `lognormal`

Count-model extensions that are likely medium effort:

- `gpoisson`
- `gammacount`
- `zeroinflatedpoisson0`
- `zeroinflatedpoisson2`
- `zeroinflatednbinomial0`
- `zeroinflatednbinomial1`
- `zeroinflatednbinomial2`

Likelihoods that imply richer response metadata or broader interface work:

- `betabinomial`
- `cbinomial`
- `coxph`
- `occupancy`
- `pom`
- `nmix`
- `nmixnb`

Specialized or architecture-expanding families:

- survival families
- extreme-value families
- stochastic-volatility families
- circular families
- user-defined likelihoods

### Exact missing likelihood names

`0binomial`, `0binomialS`, `0poisson`, `0poissonS`, `agaussian`, `bcgaussian`, `bell`, `beta`, `betabinomial`, `betabinomialna`, `bgev`, `binomial`, `binomialmix`, `cbinomial`, `cennbinomial2`, `cenpoisson`, `cenpoisson2`, `circularnormal`, `cloglike`, `coxph`, `dgompertzsurv`, `dgp`, `egp`, `exponential`, `exponentialsurv`, `exppower`, `fl`, `fmri`, `fmrisurv`, `gammacount`, `gammajw`, `gammajwsurv`, `gammasurv`, `gammasv`, `gaussianjw`, `gev`, `ggaussian`, `ggaussianS`, `gompertz`, `gompertzsurv`, `gp`, `gpoisson`, `iidgamma`, `iidlogitbeta`, `loggammafrailty`, `logistic`, `loglogistic`, `loglogisticsurv`, `lognormal`, `lognormalsurv`, `logperiodogram`, `mgamma`, `mgammasurv`, `nbinomial`, `nbinomial2`, `nmix`, `nmixnb`, `npoisson`, `nzpoisson`, `obeta`, `occupancy`, `poisson.special1`, `pom`, `qkumar`, `qloglogistic`, `qloglogisticsurv`, `rcpoisson`, `sem`, `simplex`, `sn`, `stdgaussian`, `stochvol`, `stochvolln`, `stochvolnig`, `stochvolsn`, `stochvolt`, `t`, `tpoisson`, `tstrata`, `vm`, `weibull`, `weibullsurv`, `wrappedcauchy`, `xbinomial`, `xpoisson`, `zeroinflatedbetabinomial0`, `zeroinflatedbetabinomial1`, `zeroinflatedbetabinomial2`, `zeroinflatedbinomial0`, `zeroinflatedbinomial1`, `zeroinflatedbinomial2`, `zeroinflatedcenpoisson0`, `zeroinflatedcenpoisson1`, `zeroinflatednbinomial0`, `zeroinflatednbinomial1`, `zeroinflatednbinomial2`, `zeroinflatedpoisson0`, `zeroinflatedpoisson2`, `zeroninflatedbinomial2`, `zeroninflatedbinomial3`

## 5. Missing registry surfaces beyond the two headline counts

These are not just "more names". They represent whole feature surfaces that R-INLA exposes and `rustyINLA` does not yet expose as first-class public APIs.

### Group models

Exact names:

`ar`, `ar1`, `besag`, `exchangeable`, `exchangeablepos`, `iid`, `rw1`, `rw2`

Current status:

- no dedicated `group` parity surface
- some overlapping math exists through latent models, but not through the same grouped component API

### Hazard models

Exact names:

`iid`, `rw1`, `rw2`

Current status:

- no dedicated hazard-model layer or survival-hazard interface parity

### Link registry

Exact names:

`cauchit`, `ccloglog`, `cgevit`, `cloglog`, `default`, `gevit`, `identity`, `inverse`, `log`, `loga`, `logit`, `logitoffset`, `loglog`, `logoffset`, `neglog`, `powerlogit`, `pquantile`, `probit`, `quantile`, `robit`, `sn`, `special1`, `special2`, `sslogit`, `tan`, `tanpi`, `test1`

Current status:

- family-default links exist internally for the supported subset
- no explicit public link registry parity
- no broad user-selectable link surface comparable to R-INLA

### Prior registry

Exact names:

`betacorrelation`, `dirichlet`, `expression:`, `flat`, `gamma`, `gaussian`, `invalid`, `jeffreystdf`, `laplace`, `linksnintercept`, `logflat`, `loggamma`, `logiflat`, `logitbeta`, `logtgaussian`, `logtnormal`, `minuslogsqrtruncnormal`, `mvnorm`, `none`, `normal`, `pc`, `pc.alphaw`, `pc.ar`, `pc.cor0`, `pc.cor1`, `pc.dof`, `pc.egptail`, `pc.fgnh`, `pc.gamma`, `pc.gammacount`, `pc.gevtail`, `pc.matern`, `pc.mgamma`, `pc.prec`, `pc.prw2.range`, `pc.range`, `pc.sn`, `pc.spde.GA`, `pom`, `ref.ar`, `rprior:`, `table:`, `wishart1d`, `wishart2d`, `wishart3d`, `wishart4d`, `wishart5d`, `wishartkd`

Current status:

- a few default priors are hardcoded in the current models
- there is no public prior registry or expression/table/rprior parity

### Other registry surfaces

Mixtures:

- `gaussian`, `loggamma`, `mloggamma`

`scopy`:

- `rw1`, `rw2`

Wrapper:

- `joint`

Predictor:

- `predictor`

Current status:

- none of these are currently exposed as dedicated parity features

## 6. Important gaps that `inla.list.models()` does not fully capture

This section matters because model-name parity is only part of R-INLA parity.

### Public interface and data-contract gaps

- no generic adjacency or edge-list API in `f(...)`
- limited support for richer family-specific observation metadata
- no general multi-predictor family surface
- no broad multi-likelihood response interface parity

### Mesh and SPDE workflow gaps

- no mesh-generation workflow parity
- no `fmesher`-style spatial workflow
- no public SPDE stack comparable to the R-INLA ecosystem

### Generic model-definition gaps

- no public `rgeneric` parity
- no public `cgeneric` parity
- no user-defined likelihood parity comparable to `cloglike`

### Copy, joint, and grouped-component gaps

- no `copy` parity
- no `scopy` parity
- no `group` component parity
- no `joint` wrapper parity

### Control-surface and compute-output gaps

- no broad parity for the wider `control.*` API surface
- no parity for many optional compute outputs and diagnostics
- no parity for the full internal config dump and debug surfaces returned by R-INLA

### Output-object parity gaps

- current default `rustyINLA` fit objects are intentionally thinner than R-INLA
- benchmark-mode output narrows that gap, but does not yet match full R-INLA output breadth

## 7. Where the real walls are

The hardest missing areas are not just "more names in a registry".

The real walls are:

- graph-driven spatial inputs
- SPDE and mesh workflows
- generic user-defined model and likelihood hooks
- multi-part or multi-predictor likelihood surfaces
- broad prior and control registry parity
- full output-object and diagnostic parity

That is why the recommended implementation order stays conservative:

1. extend the current local-fit subset first
2. add `rw2`
3. add one more GLM-like likelihood
4. widen the graph/input surface
5. then move into `besag` and other spatial models

## 8. How to keep this inventory current

Refresh this document when one of these changes:

- a new likelihood is implemented
- a new latent model is implemented
- a new registry-like public surface becomes available
- a previously experimental feature joins the stable benchmark suite
- the local R-INLA reference installation changes materially

If we later want a truly exhaustive package-level parity inventory, the next step would be a separate pass over:

- exported R-INLA functions
- `control.*` arguments
- optional output objects
- internal debug/config files

That would be a larger API-surface inventory than this model-registry inventory.
