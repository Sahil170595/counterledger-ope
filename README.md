# Counterledger OPE

An inspectable offline policy evaluator: patient-disjoint roles, backward fitted
Q evaluation (FQE), sequential doubly robust estimation (SDR), support audits,
paired patient bootstrap and conservative comparison gates.

This is the substantial Python evaluator, not the reduced browser calculation.
It includes conventional policies, cross-fitted value teachers, observation-only
distillation, support-aware probability transforms, frozen prompt/cache contracts
and optional local scoring adapters.

**Not clinical software.** The bundled inputs are freshly generated educational
trajectories. No patient records, original input assets, private model outputs,
provider credentials, model weights or earlier repository history are included.
The runnable example does not execute an LLM.

- [Public portfolio](https://chimeraforge.vercel.app/work)
- [Browser demo](https://chimeraforge.vercel.app/projects/reinforcement-learning/offline-policy-evaluation):
  importance-sampling estimates on a generated cohort, with a constant control, reward
  presets and a support audit. It is a separately reduced explanatory adaptation, not a
  replacement for this evaluator.
- [Method and limitations](docs/METHOD.md)
- [Release scope](docs/RELEASE_SCOPE.md)
- [Actual synthetic reference](examples/reference.json)

## Reproduce Offline

Supported and checked runtime: Python 3.13.1, NumPy 2.2.6, pandas 2.3.3,
scikit-learn 1.8.0, SciPy 1.15.3 and pytest 9.1.1. The core dependency versions
are pinned in pyproject.toml; optional model dependencies are not an exact
historical model-runtime lock.

With these packages already available, no installation or model download is needed:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python pipeline.py --seed 41 --out output/example
python -m pytest
ruff check --no-cache .
ruff format --check .
python scripts/check_release.py
```

PowerShell equivalent for the compute run:

```powershell
$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
python pipeline.py --seed 41 --out output/example
```

For a new environment on another machine, install the declared packages there;
do not install into a shared read-only runtime:

```sh
python -m venv .venv
# Activate the new environment using your shell's standard activation command.
python -m pip install -e ".[dev]"
```

The console entry point is `counterledger-example`. No original checkout is
needed. The source release was checked using an existing read-only runtime, with
isolated import paths and bytecode/cache writing disabled.

### Outputs

`output/example` contains:

| Artifact | Meaning |
| --- | --- |
| metrics.json | FQE/SDR, uncertainty, calibration, support, ablations and gates |
| comparison.csv | Four frozen policy surfaces on a common evaluation stack |
| bootstrap.npz | Ordered patient values, shared draws and paired contrasts |
| roles.csv | Synthetic patient role allocation and role checksum |
| policy_build.json | Adapter fit scope and fixed example-transform receipt |
| input_manifest.json | Generated-data hashes, counts, seed and runtime versions |
| config.json | Reward, models, support thresholds and bootstrap settings |
| synthetic_*.csv | Freshly generated train, validation and outcome-free test inputs |
| predictions.csv | Test probabilities and argmax actions, explicitly synthetic |

The example always reconstructs its inputs from a local seed. Regenerate the
committed small input tables and compact reference with:

```sh
python scripts/refresh_fixtures.py
```

Regeneration is an engineering regression check, not a new independent evaluation.
The tests repeat the entire pipeline and compare text artifacts and NPZ array
contents. ZIP container timestamps are not treated as scientific evidence.

## Architecture

```text
fresh synthetic train episodes
    |
    +-- SHA-256 patient role partition
    |      +-- policy_development: clone CV, cross-fit teacher, distiller, anchor
    |      +-- ope_nuisance: grouped calibrated propensity and target-specific FQE
    |
frozen policies + held-out validation episodes
    +-- policy observation allowlist (no IDs, clock index, actions or outcomes)
    +-- factual rewards from logged transitions only
    +-- backward time-indexed Q fits -> initial-state direct values
    +-- logged-action ratios -> raw and cumulative-cap SDR residual corrections
    +-- support/ESS/concentration/calibration/shortcut diagnostics
    +-- paired patient bootstrap -> simultaneous contrasts -> eligibility gates
    |
outcome-free synthetic test observations -> frozen-policy predictions
```

### Role Separation Is Implemented

The patient role splitter is shared by policy construction and OPE. It ranks
unique patient identifiers by a seeded SHA-256 digest, allocates 60% to
development, and assigns the remainder to nuisance fitting. Validation is
disjoint from both. Fit receipts bind exact patient sets and checksums.

The evaluator recursively audits declared policy components and immediate helpers
that expose fit receipts. The behavior clone must have exactly the development
role. The improved policy's distiller and support anchor must also expose exact
development fit receipts. This catches accidental cross-role fitting; it cannot
prove an arbitrary externally supplied policy has never seen validation.

### Two Different Kinds of Fitting

Development fitting includes a grouped-CV behavior clone, patient-out-of-fold
FQE teachers, centered action advantages, robust advantage scaling, a Ridge
distiller and a calibrated observation-only behavior anchor. Distillation prevents
the teacher's time index from becoming a served policy input.

Evaluation fitting is separate: calibrated multinomial propensities and one
backward Q model per time stage, reward specification, target policy and learner.
The primary Q family is histogram gradient boosting; Ridge with state/action
interactions is a sensitivity check. All fits use factual logged action/outcome
pairs. No factual reward is relabeled as an outcome for an unobserved action.

### Frozen Phi-4 Policies Are Not Trained Weights

The underlying system uses a frozen pretrained Phi-4 prompt policy with exact
restricted A/B/C next-token scoring. The llama.cpp path scores three cyclic
action-to-letter mappings, maps them back to canonical action order and averages
them. This reduces fixed letter-position bias. It is not a full-vocabulary
confidence estimate and does not generate a rationale.

Model identity, revision, tokenizer/template identity, scoring semantics and
temperature enter cache/fingerprint contracts. Cache-only replay fails closed
on a missing or incompatible record. The retained generic Transformers adapter
requires locally available files by default and refuses remote model code.

The improvement layer transforms frozen probabilities using temperature,
centered value advantages and estimated support, then can blend them with a
development-only behavior anchor. That changes the policy distribution without
fine-tuning Phi-4, training a language-model policy or running online RL.
Tabular Q/clone/distillation fitting is real and separate.

No original Phi-4 probability cache or rationale output is distributed here.
The offline runner substitutes `SyntheticProbabilityPolicy`: a documented
analytic fixture, not recorded model votes and not an imitation of model execution.
The legacy keys `llm_zero_shot` and `llm_improved` name evaluator slots; the
example's metadata explicitly declares their synthetic origin.

See the upstream [Phi-4 model card](https://huggingface.co/microsoft/phi-4) for
the model's own terms and limitations. No weights or tokenizer files are bundled.
Live model inference, hardware qualification and historical cache replay were
not performed for this release.

## What The Example Measures

Seed 41 creates 60 training, 24 validation and 12 test episodes, each three
steps long. There are 36 development and 24 nuisance episodes. Each generated
training/validation episode contains all three actions in a randomly permuted
order; this is deliberately artificial coverage, not measured clinical behavior.

The compact reference records actual new FQE and cap-10 SDR totals, paired
intervals, terminal raw ESS and the gate result. These values are scores of a
small synthetic mechanism under fitted nuisance models. They are not historical
model scores, clinical benefit, generalization evidence or ground-truth causal
effects. The fixture has history-dependent logging and simplifies state:
identification assumptions are not automatically established by its generator.

The example uses fixed temperature 2, a 50/50 base/behavior blend, 200 patient
bootstrap resamples and 25 boosting iterations. These are new inexpensive
regression settings, not the original frozen-policy selection. The full classes
support larger horizons and fits; the fixture is not a throughput benchmark.

Actual seed-41 totals under the checked runtime:

| Evaluator slot | Direct FQE | Cap-10 SDR | Terminal raw ESS |
| --- | ---: | ---: | ---: |
| Always-maintain | -0.225092 | 0.078477 | 0.000000 |
| Behavior clone | -0.134467 | -0.083270 | 9.913015 |
| Synthetic base (`llm_zero_shot`) | -0.141189 | -0.062751 | 9.561982 |
| Synthetic transform (`llm_improved`) | -0.134008 | -0.112879 | 8.949320 |

**No eligible policy** passes the example's gates; there is no robust winner.
All terminal raw ESS values are below the configured minimum of 10. In particular,
forced one-of-each-action episode coverage supplies no all-maintain trajectory,
so every baseline cumulative weight is zero at the final stage. This is an
inspectable support failure, not an improvement over the logging process.
Full intervals and gate flags are in the reference JSON.

## Findings And Failure Cases

- Backward FQE obeys terminal masking and retains temporal stage information in
  the evaluator, not in the deployed policy interface.
- SDR caps cumulative ratios at every horizon, rather than capping each ratio
  and allowing the effective weight to grow as cap-to-the-horizon.
- Zero target probability makes the subsequent trajectory weight zero. Large
  raw weights are reported as invalid when they overflow; they are not silently
  replaced with clipped values.
- Agreement with logged actions measures behavior agreement, not optimal action
  calibration. Descriptive reward differences by action are not treatment effects.
- A change in stochastic probabilities need not change the greedy action rule.
  This release measures greedy identity instead of importing a prior result.
- A highest point estimate is not a robust winner. Gates inspect fallback,
  overlap, low-support mass, terminal raw ESS and weight concentration. Robustness
  additionally requires simultaneous dominance under both primary estimators,
  alternative reward and learner stability.
- Patient-bootstrap intervals condition on frozen fitted nuisances. They do not
  include nuisance training, policy selection or model-runtime uncertainty.

## Tests And Scope

Tests include analytic two-step FQE/SDR examples, terminal/ordering guards,
role leakage and nested fit audits, numerical overflow and zero-weight cases,
paired/simultaneous bootstrap, winner vetoes, prompt-boundary injection, cache
tampering, mocked local backend contracts and full synthetic pipeline replay.
Optional Transformers tests skip when PyTorch is unavailable; mocked HTTP tests
are not qualification of a live inference server.

Excluded functionality is documented rather than left as a fake working path:
private submission/evidence packaging, historical tournaments and caches,
original PDFs and supplied archives, model-authored rationale passes, and prior
artifact-specific authority chains are not part of this public release.

No license grant is inferred from publication. Third-party components retain
their own licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
