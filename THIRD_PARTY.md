# Third-Party Inventory and Coverage Notes

This document is an engineering inventory of third-party code, libraries, and ecosystem dependencies visible in the local `r-inla-devel` checkout and the current `rustyINLA` port.

It is not legal advice. Its goal is to help us decide what can be cleanly reimplemented, what should be avoided, and what still needs a precise license review before reuse or redistribution.

## 1. Scope of this inventory

This inventory covers four buckets:

1. Third-party code explicitly mentioned in the local INLA license/copyright material.
2. Third-party code and system libraries referenced directly by the native INLA sources.
3. R-side package/runtime dependencies exposed by the local INLA package metadata.
4. Current direct Rust-side dependencies used by `rustyINLA`.

## 2. What the local INLA materials say

The local checkout says:

- most of `R-INLA`, the `inla-program`, and `GMRFLib` are under MIT;
- some bundled or linked pieces are under other terms;
- some of those pieces are permissive;
- some are GPL-derived or have version-specific licensing ambiguity.

That means a whole-tree source translation into Rust is a worse legal posture than an original reimplementation guided by behavior, papers, and benchmark outputs.

## 3. INLA-side third-party components

### 3.1 Bundled or source-derived components called out in `COPYRIGHTS`

| Component | Where it appears locally | Role in INLA | License note seen locally | Current Rust coverage |
| --- | --- | --- | --- | --- |
| `hash.c/.h` from Mapkit | `rinla/inst/COPYRIGHTS` | Hash maps and sparse utility structures | MIT-like permissive notice | Effectively covered by `rustc-hash` and native Rust containers |
| `bfgs3.c/.h` from GSL `vector_bfgs2.c/.h` | `rinla/inst/COPYRIGHTS` | Optimizer implementation | GPL-derived | Functionally replaced by native Rust optimizer logic; should not be source-ported |
| `Algorithm 582` / CALGO | `rinla/inst/COPYRIGHTS` | Legacy numerical routine | ACM/CALGO terms | No direct port identified for current subset; avoid translating source without specific review |
| `GMRFLib-fortran.F` LAPACK-derived code | `rinla/inst/COPYRIGHTS` | Dense linear algebra helpers | Modified BSD-style LAPACK notice | Partially covered by `faer` plus native dense Cholesky code |
| `integrator.c` with cubature / HIntLib / GSL ancestry | `rinla/inst/COPYRIGHTS` | Adaptive multidimensional integration | GPL v2-or-later ancestry explicitly noted | Only partially covered; current Rust has simpler Gauss-Kronrod + CCD logic, not equivalent general integration |

### 3.2 Libraries listed in the INLA license notes

| Component | Role in INLA | License note seen locally | Current Rust coverage | Porting recommendation |
| --- | --- | --- | --- | --- |
| `GSL` | Special functions, optimization helpers, random/statistical routines | GPL | Partial | Reimplement behavior with original Rust code and permissive crates; do not translate GSL-derived source |
| `TAUCS` | Sparse Cholesky and sparse solver infrastructure | TAUCS-specific license | Partial | Prefer replacement with Rust sparse stack instead of vendoring TAUCS logic |
| `AMD` from SuiteSparse | Sparse matrix ordering before factorization | BSD 3-clause in local notes | Not really covered yet | Safe candidate for future reimplementation or replacement with Rust-native ordering |
| `METIS` | Graph partitioning / ordering, including PARDISO-related glue | Ambiguous locally: older restrictive-looking note in one file, Apache 2.0 for METIS 5.0.3 in another | Not covered | Do not depend on it until exact version and license text are pinned |
| `LAPACK`, `BLAS`, `OPENBLAS` | Dense numerical kernels | BSD-like / permissive | Partial | Replace only where needed; do not pull in native stack unless benchmarks require it |
| `muparser` | Expression parser for `EXPRESSION:` priors and formulas | MIT in local notes | Not covered | Defer unless expression-prior support is a release goal |
| `Intel MKL` / `PARDISO` | Optional sparse solver backend and optimized kernels | Separate Intel terms for some binaries | Not covered | Keep out of the baseline Rust engine |

### 3.3 Additional native dependencies visible in the local sources

These are not all highlighted equally in the license notes, but they matter for engineering scope.

| Component | Evidence in local sources | Role in INLA | Current Rust coverage | Notes |
| --- | --- | --- | --- | --- |
| `zstd` utility code | `gmrflib/cores/cores.c`, `gmrflib/cores/LICENSE` | System/utility code, not the statistical core | Not needed | The local `cores` license is a Zstandard dual-license notice; current Rust subset does not need this functionality |
| `iniparser` | `inlaprog/src/iniparser.c/.h` | INI/config parsing for the native program | Not covered | Rust bridge bypasses this whole config-parser layer today |
| `dictionary.c/.h` | `inlaprog/src/dictionary.c/.h` | Companion dictionary structure used by `iniparser` | Not covered | Treat as separate bundled utility code if ever reused |
| `libltdl` / `ltdl.h` | `inlaprog/src/inla.c` | Dynamic loading / plugin-style loading | Not covered | Probably unnecessary for the first Rust-native engine |
| `OpenMP` | Wide native source usage | Native parallelism | Partially replaced by `rayon` | Conceptually replaced, but not feature-for-feature |
| `R` C interface | `R-interface.c`, R headers | Native bridge back to R | Covered differently | Replaced by `extendr-api` on the Rust side |
| `fmesher` file readers | Native sources call `GMRFLib_read_fmesher_file`, R package imports `fmesher` | Mesh/SPDE file and geometry ecosystem | Not covered | One of the clearest deferred subsystems |

## 4. Components the first inventory could easily miss

The first pass focused on sparse solver dependencies. The broader local sweep suggests we should also track:

- `iniparser` and its `dictionary` companion code;
- `libltdl` dynamic loading;
- `OpenMP` as a runtime/parallel dependency;
- `fmesher` and related mesh/SPDE file handling;
- R package imports such as `Matrix`, `MatrixModels`, `fmesher`, `rlang`, and others on the frontend side;
- `OpenBLAS` and `MKL` as optional native backends;
- version ambiguity around `METIS`;
- GPL ancestry in the native integrator and optimizer paths.

## 5. Details on the components that deserve special care

### `muparser`

The local native code conditionally includes `muParserDLL.h` and uses it for `EXPRESSION:` priors. This is not core Laplace machinery; it is a user-expression parsing layer. The local license note says MIT, which is favorable, but the current Rust port has no expression parser and does not need one for the validated actuarial subset.

Recommendation:

- defer `muparser`-equivalent functionality until there is a concrete user need;
- if later required, prefer a Rust-native parser or evaluator rather than a direct source migration.

### `Algorithm 582` / CALGO

The local `COPYRIGHTS` file explicitly calls out Algorithm 582 from ACM CALGO. CALGO algorithms are not a casual copy-paste zone. Even if the implementation seems old or small, it should be treated as provenance-sensitive source.

Recommendation:

- do not translate CALGO source into Rust;
- if the underlying numerical method matters, re-derive it from papers/specifications and validate behavior with golden outputs.

### `zstd`

The `gmrflib/cores` license file is a Zstandard license notice, and the local `cores.c` is utility/system-oriented code rather than an INLA modeling primitive. This is a good example of something that can be safely left out of a first probabilistic-engine port.

Recommendation:

- omit from `rustyINLA` unless a real packaging/runtime need appears;
- if ever needed, use the Rust ecosystem's maintained `zstd` crates instead of porting utility snippets.

### GSL-derived pieces

There are two different GSL issues in the local tree:

1. direct or broad GSL linkage in the native program;
2. explicit GPL-derived source in `bfgs3.c/.h`, plus GPL ancestry in the native `integrator.c`.

This is exactly why clean-room reimplementation is safer than source migration.

Recommendation:

- treat GSL-linked or GSL-derived source as a behavior reference, not an implementation source;
- keep replacing needed math with `statrs`, `faer`, original Rust code, or other permissive Rust crates.

### `hash.c` / Mapkit

This one is relatively benign. The local notes say the code came from Mapkit under an MIT-like permissive notice. Conceptually it is just support infrastructure: hash maps and sparse utility structures.

Recommendation:

- there is no reason to port this code;
- standard Rust collections and `rustc-hash` already cover the role cleanly.

### `METIS`

`METIS` is the trickiest licensing item in the local materials. One local note contains an older-looking distribution policy with an email-permission requirement, while another local note says METIS 5.0.3 is distributed under Apache 2.0.

That means the correct answer depends on the exact METIS version and how INLA actually bundles or links it.

Recommendation:

- do not vendor or rely on METIS until the exact upstream version is pinned;
- for `v0.1`, avoid a hard dependency on METIS;
- if ordering improvements are needed, prefer AMD-first work or a clearly licensed Rust-native alternative.

## 6. R-side package and ecosystem dependencies

The local `INLA` package metadata shows the frontend is not just a thin native wrapper. It also depends on R-side ecosystem pieces.

Important examples:

- `R`
- `Matrix`
- `MatrixModels`
- `fmesher`
- `rlang`
- `parallel`
- `splines`
- `stats`
- `withr`

Engineering interpretation:

- the Rust core is already independent from most of that;
- `SPDE` and mesh-related workflows are the biggest obvious ecosystem gap;
- formula parsing and user-facing ergonomics are also broader than the current Rust-native core.

## 7. Current Rust-side direct dependencies

These are the direct dependencies visible in the current Rust manifests.

### `inla_core`

- `faer`
- `rayon`
- `argmin`
- `argmin-math`
- `statrs`
- `sha2`
- `rand`
- `thiserror`
- `ndarray`
- `rustc-hash`

### R bridge crate

- `extendr-api`
- local path dependency on `inla_core`
- `rayon`
- `rustc-hash`

### Dev/test dependencies currently visible

- `serde`
- `serde_json`
- `approx`
- `criterion`

Note:

- this document is a direct-dependency inventory, not yet a full transitive SPDX audit of every Cargo package in `Cargo.lock`;
- a full Rust crate license audit should still be added to CI later.

## 8. Functional coverage matrix for the current port

| Functional area | INLA-side dependency pressure | Current Rust status | Coverage assessment |
| --- | --- | --- | --- |
| Sparse graph representation | GMRFLib / utility code | Native Rust graph module | Covered |
| Sparse Cholesky, log-det, selected inverse | TAUCS / sparse backend / PARDISO path | `faer` plus native solver code | Covered for current subset, but performance tuning remains |
| Hyperparameter optimization | GSL-derived optimizer ancestry | Native Rust optimizer plus finite differences / CCD | Covered functionally, parity still in progress |
| Dense tiny linear algebra | LAPACK/BLAS-style helpers | Native dense Cholesky helper and `faer` | Partially covered |
| Count and severity likelihoods | Native INLA likelihood code | Gaussian, Poisson, Gamma, ZIP1, Tweedie in Rust | Covered for current subset |
| Latent models | Native INLA/GMRFLib model layer | `iid`, `rw1`, `ar1`, compound blocks | Covered for current subset |
| Ordering heuristics | AMD / possibly METIS-sensitive native paths | No dedicated AMD/METIS-equivalent layer yet | Partial gap |
| Expression priors | `muparser` | Not implemented | Not covered |
| Config/INI parsing | `iniparser`, `dictionary` | Not needed in current bridge | Intentionally not covered |
| Dynamic loading hooks | `libltdl` | Not implemented | Intentionally not covered |
| SPDE / mesh ecosystem | `fmesher`, SPDE native sources | Not implemented | Not covered |
| MKL / PARDISO optimized backend | Intel terms, optional native path | Not implemented | Intentionally not covered |

## 9. Practical conclusions

1. The current Rust port already replaces a meaningful part of the old native stack without linking to it directly.
2. The most important uncovered native dependencies are not utility code like `hash.c`; they are the mesh/SPDE ecosystem, ordering/performance layers, and optional solver backends.
3. The main legal hazards are source-derived GPL/CALGO paths and any attempt to vendor version-ambiguous external code like `METIS` without pinning provenance.
4. The cleanest `v0.1` strategy is still:

- original Rust implementation;
- validated subset;
- no `SPDE`;
- no `PARDISO`/`MKL`;
- no expression-prior parser yet;
- explicit provenance rules for contributors.

## 10. Follow-up work

Recommended next compliance tasks:

1. Choose the outbound project license for `rustyINLA`.
2. Add `PROVENANCE.md`.
3. Add a `THIRD_PARTY_RUST.md` or CI license audit for Cargo dependencies.
4. Pin the exact policy for any future use of AMD or METIS.
5. Keep `SPDE`, `fmesher`, `muparser`, `PARDISO`, and native-config parsing out of `v0.1`.
