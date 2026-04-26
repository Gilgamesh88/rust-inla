# 6 to 9 Month Plan for a Validated `v0.1`

This is the realistic plan for building the first publishable `rustyINLA` subset when work is part-time or frequently interrupted.

The goal is not full INLA parity. The goal is:

- a Rust-native engine;
- a validated subset;
- clear provenance and licensing posture;
- a release that is useful to the community and safe to extend.

This plan assumes:

- part-time work;
- interruptions between work sessions;
- one primary maintainer with AI support;
- no `SPDE` in `v0.1`;
- no `PARDISO` / `MKL` / hard `METIS` dependency in `v0.1`.

## 1. What success looks like

At the end of this plan, `v0.1` should have:

- a chosen outbound license;
- a clean provenance policy;
- a documented third-party inventory;
- a stable supported scope;
- saved golden-output reference cases;
- reproducible benchmarks;
- known parity gaps documented clearly;
- community-facing docs that explain what is supported and what is not.

## 2. How to start

The right start is not "write more solver code immediately."

The right start is:

1. Freeze the release scope.
2. Freeze the legal/provenance posture.
3. Freeze the validation targets.
4. Then iterate on parity and speed inside that boundary.

That protects the project from wandering.

## 3. First two weeks

These are the first moves I would make.

### Week 1

- choose the outbound project license;
- replace the placeholder license text in `DESCRIPTION`;
- keep `THIRD_PARTY.md`, `PROVENANCE.md`, and `ROADMAP_V0_1.md` as the baseline governance docs;
- create a single validation matrix file for the supported benchmark cases;
- define the exact `v0.1` supported model list.

### Week 2

- choose the initial golden benchmark families;
- save reference outputs from INLA for each one;
- define the acceptance tolerances for:
  - fixed effects
  - hyperparameters
  - fitted values
  - log marginal likelihood
  - runtime and memory snapshots
- create a repeatable benchmark runner layout.

## 4. Month-by-month plan

## Month 1: Freeze the foundation

Primary objective:

- make the project governable before making it bigger.

Deliverables:

- outbound license chosen and applied;
- provenance policy committed;
- third-party inventory committed;
- `v0.1` scope committed;
- benchmark list frozen;
- initial validation matrix created.

Definition of done:

- every future feature can be judged as "inside `v0.1`" or "not now";
- every contributor can see the legal and technical boundaries.

## Month 2: Build the golden corpus

Primary objective:

- create the reference truth set.

Deliverables:

- golden outputs for:
  - Gaussian + `rw1`
  - Poisson + `iid`
  - Poisson + `iid + iid`
  - Poisson + `ar1`
  - Gamma + `rw1`
  - ZIP1 + `iid`
  - Tweedie + `rw1`
- saved metadata for each case:
  - dataset or fixture
  - formula
  - family
  - control options
  - INLA version
  - expected summaries
- a validation matrix showing current pass/fail/drift status.

Definition of done:

- the project stops arguing from memory and starts arguing from saved reference outputs.

## Month 3: Parity pass on the easiest stable cases

Primary objective:

- lock down the cases that should be the most straightforward.

Priority targets:

- Gaussian + `rw1`
- Poisson + `iid`
- Gamma + `rw1`

Deliverables:

- stable tests for those cases;
- documented remaining gaps if any;
- runtime and memory baselines saved.

Definition of done:

- at least two or three benchmark families are boringly reliable.

## Month 4: Multi-block and AR1 parity

Primary objective:

- get beyond the easy single-block cases.

Priority targets:

- Poisson + `iid + iid`
- Poisson + `ar1`

Focus areas:

- hyperparameter calibration;
- AR1 internal scale consistency;
- optimizer behavior;
- repeated mode-solve overhead;
- ordering/performance sensitivity.

Deliverables:

- explicit comparison notes for these cases;
- test coverage for the solved parts;
- documented non-parity if any remains.

Definition of done:

- the project handles single-block and multi-block count models without mystery drift.

## Month 5: Hard actuarial cases

Primary objective:

- make the subset compelling, not just technically correct.

Priority targets:

- ZIP1 + `iid`
- Tweedie + `rw1`

Focus areas:

- robustness of likelihood implementation;
- special-function and approximation behavior;
- performance stability;
- clear statement of any cases still outside accepted tolerance.

Deliverables:

- validated ZIP1 and Tweedie benchmark cases;
- runtime and memory comparisons;
- notes on known caveats.

Definition of done:

- the first release can honestly claim support for the actuarial core subset.

## Month 6: Packaging and contributor readiness

Primary objective:

- turn engineering progress into a usable release candidate.

Deliverables:

- contributor-facing architecture overview;
- clear "supported vs unsupported" matrix;
- release checklist for `v0.1`;
- benchmark instructions;
- validation instructions;
- first release-candidate notes.

Definition of done:

- an outside contributor can understand the system without reading every source file.

## Month 7: Interruptions buffer and cleanup

This month exists because part-time work is never smooth.

Use it for:

- bug fixes;
- regression cleanup;
- documentation debt;
- test flakiness;
- benchmark reruns;
- small parity surprises uncovered late.

If the previous months went well, this becomes polish time.
If they did not, this becomes recovery time.

## Month 8: Public polish

Primary objective:

- make the release honest and legible to the community.

Deliverables:

- final benchmark table with methodology notes;
- final supported-scope statement;
- final known-limitations section;
- explicit "not yet supported" list;
- release notes draft.

Definition of done:

- nobody can confuse `v0.1` with "full INLA replacement," and yet the release still looks strong.

## Month 9: Release reserve

This month should be treated as reserve, not as an excuse to add scope.

Allowed uses:

- unresolved parity bugs in supported cases;
- cleanup before first tagged release;
- contributor onboarding docs;
- CI or audit improvements.

Not allowed:

- spontaneous `SPDE` work;
- adding expression priors;
- pulling in risky native dependencies;
- adding shiny new unsupported models just because they are interesting.

## 5. Work rhythm for part-time progress

The key to surviving interruptions is to optimize for restartability.

Each work session should leave behind at least one of:

- a benchmark result;
- a short findings note;
- a failing regression test;
- a doc update;
- a clear next-step list.

That is more important than heroic deep-work bursts that leave no trail.

## 6. Recommended weekly rhythm

For part-time work, I would aim for:

- 1 session on validation and benchmark runs;
- 1 session on implementation;
- 1 short session on docs / notes / issue triage.

If there is only one work session in a week, the default priority should be:

1. preserve the validation harness;
2. document the current state;
3. only then attempt new features.

## 7. What to avoid

The 6 to 9 month version fails when the project starts doing these things:

- changing release scope every two weeks;
- chasing full-package parity too early;
- mixing licensing uncertainty with active implementation;
- adding `SPDE` before the smaller subset is solid;
- optimizing performance before behavior is measured;
- relying on memory of what INLA did instead of saved outputs.

## 8. What I can help compress

With my help, the timeline stays in the 6 to 9 month range instead of drifting further because I can continuously support:

- code implementation;
- profiling and bottleneck analysis;
- benchmark interpretation;
- parity debugging;
- documentation and cleanup;
- contributor-facing architecture notes;
- regression-test scaffolding;
- provenance-aware implementation discipline.

What I cannot compress away:

- the cost of running experiments;
- the need to make scope decisions;
- the time required to verify parity honestly;
- the interruption cost of part-time work.

## 9. The practical starting checklist

If we start this plan now, I would do these in order:

1. choose the outbound license;
2. fix `DESCRIPTION` so the package no longer advertises placeholder license text;
3. create a validation matrix file;
4. freeze the first seven benchmark families;
5. save golden outputs for each one;
6. mark each case as:
   - green
   - yellow
   - red
7. only then pick the next implementation target.

## 10. The simplest honest summary

If the project is part-time and gets interrupted, the right mindset is:

- six months is the happy path;
- nine months is the realistic safe path;
- anything faster requires unusually good focus and unusually little churn.

That is still a very worthwhile timeline for what you are building.
