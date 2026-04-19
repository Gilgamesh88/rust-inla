# Provenance and Clean-Room Policy

This project aims to be an original Rust-native probabilistic engine inspired by INLA, not a line-by-line translation of the INLA native codebase.

This document defines what sources are safe to use, what sources are restricted, and how contributors should implement new functionality.

It is a project policy document, not legal advice.

## 1. Project posture

`rustyINLA` should be developed as:

- an original implementation;
- behavior-compatible where validated;
- benchmarked against trusted INLA outputs;
- documented so contributors can extend it without copying mixed-license native code.

## 2. Allowed sources for implementation work

These sources are acceptable inputs for design and implementation:

- published papers and academic descriptions of the algorithms;
- official package documentation and user-facing docs;
- public mathematical formulas and derivations;
- black-box behavioral observations from running INLA;
- benchmark outputs and golden reference results;
- independently written design notes and issue discussions in this project;
- upstream license texts used only to understand reuse constraints;
- standard numerical methods re-derived from first principles.

## 3. Restricted sources

These sources must not be translated mechanically into Rust unless a specific legal review says otherwise:

- GPL-derived source files from the INLA tree;
- CALGO or other provenance-sensitive algorithm source code;
- copied chunks of mixed-license native code;
- vendored third-party code whose version or license is unclear;
- direct line-by-line translation of source from `inlaprog`, `gmrflib`, or bundled third-party libraries.

Examples of high-risk areas in the local checkout include:

- GSL-derived optimizer code;
- the native `integrator.c` with GPL ancestry;
- CALGO `Algorithm 582`;
- any source path that depends on version-sensitive `METIS` reuse;
- optional `PARDISO` / `MKL` glue code.

## 4. Safe implementation workflow

For new functionality, contributors should follow this order:

1. Write down the behavior we want.
2. Create or collect a golden reference case.
3. Describe the mathematical method in our own words.
4. Implement original Rust code from that specification.
5. Validate against outputs, not against copied source structure.
6. Record assumptions, approximations, and known deviations.

## 5. Feature categories

### Low-risk features

Usually safe to implement directly from math/specification:

- basic latent models such as `iid`, `rw1`, `ar1`;
- standard likelihoods such as Gaussian, Poisson, Gamma;
- generic sparse graph manipulation;
- generic quadrature and numerical integration methods;
- diagnostics and profiling infrastructure;
- interface layers written specifically for this project.

### Medium-risk features

Require explicit provenance notes:

- ordering heuristics inspired by SuiteSparse/AMD behavior;
- selected inverse strategies tuned to match INLA behavior;
- Tweedie approximations and special-function-heavy likelihood code;
- sparse solver optimizations guided by profiling against INLA.

### High-risk features

Need extra care and probably a dedicated provenance note before implementation:

- `SPDE` and mesh-related ecosystems;
- expression-prior parsing and evaluation;
- METIS- or PARDISO-specific functionality;
- direct rewrites of legacy native optimizer or integrator files;
- anything relying on CALGO/GSL-derived source structure.

## 6. Contributor rules

Contributors should:

- cite the behavior source they used;
- prefer papers, docs, and golden outputs over source translation;
- avoid copying comments, identifier structure, or control flow from restricted source;
- document any nontrivial algorithmic inspiration in the PR description or commit notes;
- add validation cases for every new model, likelihood, or optimizer change.

Contributors should not:

- paste native INLA code into Rust files;
- rephrase restricted code line-by-line;
- vendor third-party source without adding it to the third-party inventory;
- add new dependencies with unclear licenses.

## 7. Required artifacts for each major feature

Every major feature should eventually have:

- a short design note;
- one or more golden-output reference cases;
- tests for numerical behavior;
- explicit statement of supported scope;
- note of any divergence from INLA;
- provenance note if the area is medium- or high-risk.

## 8. How to use INLA for validation

Using INLA as a behavioral oracle is encouraged.

Safe uses include:

- comparing parameter summaries;
- comparing fitted values and log marginal likelihoods;
- profiling runtime and memory;
- observing defaults and internal scales;
- running multiple benchmark variants to infer semantics.

Unsafe uses include:

- copying native routines because the benchmark looked correct;
- treating observed output parity as permission to translate source;
- assuming all code under the local tree has the same license.

## 9. Dependency policy

For Rust dependencies:

- prefer crates with permissive licenses and healthy maintenance;
- keep direct dependencies small and intentional;
- add dependency-license checks to CI;
- document new runtime-critical dependencies in `THIRD_PARTY.md`.

For native dependencies:

- avoid adding new native dependencies to the `v0.1` engine unless clearly necessary;
- avoid version-ambiguous dependencies such as `METIS` until exact provenance is pinned;
- keep `PARDISO` / `MKL` out of the baseline open engine.

## 10. Deferred subsystems for `v0.1`

These areas are explicitly out of scope for the first validated public subset unless a later decision changes it:

- `SPDE`;
- `fmesher`-dependent workflows;
- `muparser`-style expression priors;
- `PARDISO` / `MKL` backends;
- native INI/config parser parity;
- direct plugin/dynamic-loader parity.

## 11. Release philosophy

The project should prefer:

- small validated subset over broad unvalidated scope;
- provenance clarity over feature count;
- original Rust implementation over native-source reuse;
- contributor guidance over undocumented heroics.

That is the path most likely to produce a usable community foundation.
