# Release Scope

This repository begins with one clean source-release commit. It contains no
parent repository history.

## Retained Implementation

- Shared probability/action contracts and patient-role checks.
- Full finite-horizon FQE, calibrated propensities, raw/capped SDR and support diagnostics.
- Full common four-policy evaluation, reward/learner sensitivity, paired bootstrap,
  simultaneous contrasts, eligibility gates and post-hoc adversarial controls.
- Logged-only environment and bounded factual reward.
- Behavior clone, cross-fitted teacher construction, observation-only distiller,
  support adapter and probability transforms.
- Prompt trust boundaries, deterministic caches, fingerprints, local Transformers
  and native llama.cpp scoring implementations.
- Synthetic unit cases and new end-to-end public fixture/replay tests.

## Release Changes

Private evidence-bundle entry points are replaced with an independent runner.
Schemas/defaults and prompt absence text are public-release versions; they are
not compatible with private cache hashes. Data-derived fallback defaults are
replaced by uniform probabilities. Default seeds are independent of personal IDs.
The local Transformers adapter defaults to local files only.

Prior-results-specific temperature/blend diagnostic labels become generic
component labels. The greedy always-maintain identity is measured, not asserted
as an imported fact. Evidence metadata distinguishes synthetic characterization
from caller-supplied policies. CSV hash normalization uses LF for all public CSVs.

Release boundary hardening adds a shared policy-column guard including mortality
fields and finite-input checks in the low-level SDR function. Those changes have
their own counterexample tests.

## Excluded

All supplied archives and their contents, original trajectory tables and test
observations, briefs, prior reports/PDFs, historical policy outputs, model caches,
model-authored rationales, private hashes/identifiers, credentials and Git history.
Private authority/receipt builders are not copied under renamed filenames:
without their excluded inputs they could not honestly provide working replay.

No third-party weights, tokenizer assets, datasets or source trees are vendored.
Assets without a clear redistribution basis are absent rather than assumed safe.

## Qualification

The offline evaluator and synthetic regression path are executable. Local model
backends are implementation exports with mocked contract tests; no real model
inference/server was qualified during release. The browser demo has separate
coordinator QA and merge status.
