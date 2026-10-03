# Evaluation Method

## Underlying Engineering

The underlying implementation separated a frozen language-model policy from
train-only tabular policy adapters and independently fitted evaluation nuisances.
Its orchestration tied policy construction, role manifests, probability caches,
selection metadata and prediction export together. The public edition retains
the evaluator and useful policy implementation but removes private artifact
authorities. It does not claim to replay the original selected policy.

The frozen policy path uses restricted next-token probabilities, not prose
self-reported confidence. Cyclic action/letter mappings in the native local
scorer address letter-position preference. Fingerprints include semantic model,
template, token and scoring settings; execution settings are recorded separately.
Cache conflicts and identity drift are failures rather than fallback successes.

The inspected original construction includes a temperature-softened frozen
Phi-4 component blended with an observation-only behavior anchor. The retained
generic transform also supports value/support ablations. No trained language-model
weights result from those transforms. This distinction matters: a new prompt,
temperature or mixture creates a new policy surface, not a new trained foundation
model.

Original validation was characterized as previously inspected and adaptively
reused in the inspected evaluation code. A disjoint final role partition cannot
retroactively erase that exposure. No original score or patient-level artifact
has been imported into the public example.

## Factual Transition Boundary

Rows consist of episode identity and time, pre-action state, logged action,
factual next-window outcomes and a terminal marker. The trajectory validator
requires unique episode/time keys, consecutive integer stages, one final
termination and aligned previous actions.

The policy interface receives only the declared pre-action observation fields.
Patient/site/provider identifiers, time index, logged action, terminal flags,
mortality fields and future/adverse outcomes are prohibited. Missing notes use
an explicit absence sentinel. Real caller-supplied note text is bounded,
normalized and JSON-encoded as untrusted data; marker and chat-token collisions
are rejected. These are structural protections, not proof of semantic resistance
to every possible prompt attack.

The environment implements `step_logged()`, not `step(action)`. An offline
dataset is not an action-conditioned simulator. The fresh data generator is a
separate fixture authoring mechanism and is not used to manufacture alternative
outcomes for an evaluator row.

## Finite-Horizon FQE

For stage t, the fit target is:

```text
y_t = r_t + gamma * (1 - terminal_t) * sum_a pi(a | s_(t+1)) Q_(t+1)(s_(t+1), a)
```

Stages are fitted backwards. Next-stage values are aligned by episode identity,
and rewards/probabilities are aligned to stable transition keys before fitting.
Only logged actions appear as training labels. The Q regressor predicts every
action at evaluation time; those unobserved-action predictions are model-based
extrapolation, not recovered factual outcomes.

The direct estimate is the average target-policy value at validation episode
starts. Statewise remaining-horizon values are retained for diagnostics. HGB is
the designated primary family in the example; Ridge uses explicit state/action
interactions as a model sensitivity check.

## Sequential Doubly Robust Estimation

```text
rho_t = pi(a_t | s_t) / behavior(a_t | s_t)
w_t = product_(k=0..t) rho_k
DR_episode = V_0 + sum_t gamma^t w_t
             * [r_t + gamma*(1-terminal_t)*V_(t+1) - Q_t(s_t,a_t)]
```

Cumulative weights are accumulated in log space. Each capped estimator uses
`min(w_t, cap)` for each residual. It does not multiply independently capped
ratios. Clipping introduces bias; raw and capped estimators are both kept.
The diagnostics report raw ESS, concentration, zero weights and linear-scale
overflow. An invalid raw return is unavailable, not a hidden clipped estimate.

The behavior model is a multinomial logistic pipeline with grouped regularization
selection, sigmoid calibration and patient-group out-of-fold diagnostics. Its
probabilities are floored and renormalized for numerical protection. Calibration
to the logging policy is not calibration to a treatment-effect or optimal-action
target.

Sequential exchangeability, adequate support and sufficiently appropriate
nuisance models are assumptions. Fitted models and extensive diagnostics cannot
establish those assumptions from logged data alone.

## Inference And Gates

One patient-resampling matrix is reused across policy estimates, differences,
estimators and reward/learner sensitivities. Intervals are conditional on already
fitted nuisance models. The simultaneous max-t family includes all six unordered
policy pairs across primary direct FQE and primary SDR: twelve contrasts.

Eligibility is a separate step from ranking. The example checks schema, fallback
rate, action overlap, low-support target mass, terminal raw ESS and maximum
normalized episode weight share. Thresholds are visible in config.json.
The gate is a research diagnostic, not a clinical safety guarantee.

A robust winner requires positive simultaneous lower bounds against every
eligible opponent under both primary estimators. It must also preserve its top
rank under the alternative reward and Ridge learner, and its primary contrast
must exceed the corresponding learner sensitivity. A failed gate or changed
ranking remains visible.

The evaluator also computes matched-marginal constant, row-permuted, uniform-base
and fully uniform controls. They are post-hoc state-dependence/extrapolation
diagnostics excluded from winner selection. Matching validation marginals uses
that same split and is explicitly descriptive, not a fresh competing policy.

## Reproduction Boundaries

The new input generator and deterministic analytic policy make offline tests
independent of model downloads or original assets. Published input CSVs reconcile
with generator values and canonical LF hashes. The manifest declares runtime
versions; all IDs start with SYN and are fabricated.

Same-process byte/array replay is tested under the pinned core stack. This is not
a claim of byte-identical cross-platform numerical libraries, cold GPU inference,
historical model-cache regeneration or unbiased held-out generalization.

The original profile's hyperparameters and cached outputs are not used as a
public benchmark. This example uses its own small fixed transform, shortened
horizon and low-cost fits. No clinical benefit, online policy learning, causal
ground truth, provider integration or live model result is inferred from it.
