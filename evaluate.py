"""Shared four-policy offline evaluation and command-line entry point.

All target policies must be completely frozen before :func:`evaluate_policies`
is called.  This module fits nuisance models only on the disjoint
``ope_nuisance`` training role and opens validation outcomes only for final
evaluation.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from contracts import ACTION_NAMES, POLICY_NAMES, portable_file_sha256
from diagnostics import (
    behavioral_calibration_summary,
    decision_rule_summary,
    paired_bootstrap_gap_summary,
    reward_shortcut_diagnostics,
)
from reward import RewardConfig
from value_model import (
    BehaviorPropensityModel,
    FiniteHorizonFQE,
    assert_disjoint_patient_sets,
    deterministic_training_roles,
    evaluate_sdr,
    factual_reward_vector,
    patient_level_paired_bootstrap,
    predict_policy_probabilities,
    support_diagnostics,
    validate_trajectory_frame,
)

BASELINE_POLICY = "always_maintain"
PRIMARY_REWARD_NAME = "primary"
ALTERNATIVE_REWARD_NAME = "alternative"
MAP_ABLATED_REWARD_NAME = "map_ablated"


@dataclass
class EvaluationBundle:
    """Generated metrics plus exact patient-level samples used to build them."""

    metrics: dict[str, Any]
    role_manifest: pd.DataFrame
    comparison: pd.DataFrame
    bootstrap_samples: dict[str, np.ndarray]


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _sha256_file(path: Path) -> str:
    return portable_file_sha256(path)


def _row_key_sha256(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for patient_id, time_step in frame.loc[:, ["patient_id", "time_step"]].itertuples(
        index=False,
        name=None,
    ):
        encoded = json.dumps(
            [str(patient_id), int(time_step)],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(encoded + b"\n")
    return digest.hexdigest()


def _probability_reference(frame: pd.DataFrame, probabilities: np.ndarray) -> dict[str, Any]:
    matrix = np.ascontiguousarray(np.asarray(probabilities, dtype="<f8"))
    if matrix.shape != (len(frame), len(ACTION_NAMES)):
        raise ValueError("Probability reference shape does not match its source frame")
    return {
        "shape": list(matrix.shape),
        "float64_sha256": hashlib.sha256(matrix.tobytes(order="C")).hexdigest(),
    }


def _policy_runtime_diagnostics(policy: Any, rows: int) -> dict[str, Any]:
    diagnostics = getattr(policy, "diagnostics", None)
    requests = int(
        getattr(
            diagnostics,
            "requests",
            getattr(policy, "total_requests", getattr(policy, "requests", rows)),
        )
    )
    fallbacks = int(
        getattr(
            diagnostics,
            "fallbacks",
            getattr(policy, "fallback_count", getattr(policy, "fallbacks", 0)),
        )
    )
    malformed = int(
        getattr(
            diagnostics,
            "malformed_outputs",
            getattr(policy, "malformed_count", getattr(policy, "malformed_outputs", 0)),
        )
    )
    fallback_rate = float(
        getattr(policy, "fallback_rate", fallbacks / requests if requests else 0.0)
    )
    failure_reasons = dict(getattr(policy, "failure_reasons", {}))
    payload = {
        "requests": requests,
        "fallbacks": fallbacks,
        "malformed_outputs": malformed,
        "fallback_rate": fallback_rate,
        "failure_reasons": failure_reasons,
        "cache_hits": int(getattr(diagnostics, "cache_hits", getattr(policy, "cache_hits", 0))),
        "cache_misses": int(
            getattr(diagnostics, "cache_misses", getattr(policy, "cache_misses", 0))
        ),
        "model_id": getattr(policy, "model_id", policy.__class__.__name__),
        "prompt_version": getattr(policy, "prompt_version", None),
    }
    execution_metadata = getattr(policy, "execution_metadata", None)
    if execution_metadata is not None:
        payload["execution_metadata"] = dict(execution_metadata)
    policy_metadata = getattr(policy, "policy_metadata", None)
    if isinstance(policy_metadata, Mapping):
        payload["policy_metadata"] = dict(policy_metadata)
    return payload


def _audit_policy_fit_scope(
    policy: Any,
    *,
    allowed_patient_ids: Sequence[str],
) -> dict[str, Any]:
    """Recursively verify any exposed policy-side fit receipts are development-only."""

    allowed = {str(patient_id) for patient_id in allowed_patient_ids}
    seen: set[int] = set()
    receipts: list[dict[str, Any]] = []

    def visit(component: Any, path: str) -> None:
        if component is None or id(component) in seen:
            return
        seen.add(id(component))
        fit_ids: tuple[str, ...] = ()
        for attribute in ("fit_patient_ids_", "fitted_patient_ids_"):
            values = getattr(component, attribute, ())
            if values:
                fit_ids = tuple(sorted({str(value) for value in values}))
                break
        if fit_ids:
            outside = set(fit_ids) - allowed
            if outside:
                raise RuntimeError(
                    f"Policy-side fit leakage at {path}; example={sorted(outside)[0]}"
                )
            checksum = getattr(
                component,
                "fit_patient_checksum_",
                getattr(component, "fitted_patient_checksum_", None),
            )
            computed_checksum = hashlib.sha256("\n".join(fit_ids).encode()).hexdigest()
            if checksum is not None and str(checksum) != computed_checksum:
                raise RuntimeError(f"Policy-side fit checksum mismatch at {path}")
            receipts.append(
                {
                    "component": path,
                    "type": component.__class__.__name__,
                    "patients": len(fit_ids),
                    "patient_checksum": computed_checksum,
                    "declared_checksum_verified": checksum is not None,
                    "development_only": True,
                }
            )
        declared_children = ("base_policy", "action_value_model", "support_model", "model")
        visited_children: set[str] = set()
        for child_name in declared_children:
            if hasattr(component, child_name):
                visit(getattr(component, child_name), f"{path}.{child_name}")
                visited_children.add(child_name)
        # Fail closed for an immediately attached fitted helper even when a new
        # composition attribute was not added to the declared traversal list.
        # We deliberately do not recurse through arbitrary sklearn internals;
        # only objects exposing our patient-fit receipt contract are followed.
        component_state = getattr(component, "__dict__", {})
        if isinstance(component_state, Mapping):
            for child_name, child in component_state.items():
                if child_name in visited_children or child is None:
                    continue
                if any(
                    hasattr(child, attribute)
                    for attribute in ("fit_patient_ids_", "fitted_patient_ids_")
                ):
                    visit(child, f"{path}.{child_name}")

    visit(policy, policy.__class__.__name__)
    all_discovered = all(receipt["development_only"] for receipt in receipts)
    return {
        "receipts": receipts,
        "discovered_fitted_components": len(receipts),
        "all_discovered_fit_receipts_development_only": all_discovered,
        "traversal_attributes": [
            "base_policy",
            "action_value_model",
            "support_model",
            "model",
            "any immediate attribute exposing fit_patient_ids_ or fitted_patient_ids_",
        ],
        "scope_note": (
            "claim covers fitted components reachable through declared policy-composition "
            "attributes plus any immediate helper exposing the fit-receipt contract"
        ),
    }


def _summary_to_dict(summary: Any, *, horizon: int) -> dict[str, Any]:
    return {
        "total_return": float(summary.point_estimate),
        "return_per_decision": float(summary.point_estimate / horizon),
        "confidence_interval_95_total": [float(value) for value in summary.confidence_interval],
        "confidence_interval_95_per_decision": [
            float(value / horizon) for value in summary.confidence_interval
        ],
        "bootstrap_unit": "patient",
        "nuisance_models_refit_in_bootstrap": False,
    }


def _initial_policy_values(
    frame: pd.DataFrame,
    fqe: FiniteHorizonFQE,
    probabilities: np.ndarray,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    state_values = fqe.predict_v(frame, probabilities)
    initial_mask = frame["time_step"].to_numpy(dtype=int) == 0
    initial_frame = frame.loc[initial_mask]
    patient_ids = tuple(initial_frame["patient_id"].astype(str))
    if patient_ids != tuple(sorted(patient_ids)):
        raise RuntimeError("Validation episodes must be stably sorted by patient")
    return patient_ids, state_values[initial_mask], state_values


def _reward_configs(config: Mapping[str, Any]) -> dict[str, RewardConfig]:
    rewards = {
        PRIMARY_REWARD_NAME: RewardConfig.from_mapping(config["reward"]),
        ALTERNATIVE_REWARD_NAME: RewardConfig.from_mapping(config["alternative_reward"]),
    }
    if "map_ablated_reward" in config:
        rewards[MAP_ABLATED_REWARD_NAME] = RewardConfig.from_mapping(config["map_ablated_reward"])
    return rewards


def _behavior_model(
    observation_columns: Sequence[str],
    config: Mapping[str, Any],
) -> BehaviorPropensityModel:
    experiment = config["experiment"]
    settings = config["behavior_model"]
    return BehaviorPropensityModel(
        observation_columns,
        c_grid=settings["c_grid"],
        group_folds=int(settings["group_folds"]),
        max_iter=int(settings["max_iter"]),
        probability_floor=float(experiment["behavior_probability_floor"]),
        seed=int(experiment["seed"]),
    )


def _bootstrap_estimator(
    values_by_policy: Mapping[str, np.ndarray],
    *,
    baseline: str,
    experiment: Mapping[str, Any],
    key: str,
    sample_store: dict[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, Any]]:
    estimates, differences, indices = patient_level_paired_bootstrap(
        values_by_policy,
        baseline=baseline,
        resamples=int(experiment["bootstrap_resamples"]),
        confidence=float(experiment["bootstrap_confidence"]),
        seed=int(experiment["seed"]),
    )
    sample_store.setdefault("paired_patient_indices", indices)
    estimate_dict: dict[str, Any] = {}
    difference_dict: dict[str, Any] = {}
    for policy_name in values_by_policy:
        sample_store[f"per_patient__{key}__{policy_name}"] = np.asarray(
            values_by_policy[policy_name],
            dtype=float,
        )
        sample_store[f"{key}__{policy_name}"] = estimates[policy_name].samples
        sample_store[f"{key}__difference_vs_{baseline}__{policy_name}"] = differences[
            policy_name
        ].samples
        estimate_dict[policy_name] = _summary_to_dict(
            estimates[policy_name],
            horizon=int(experiment["horizon"]),
        )
        difference_dict[policy_name] = _summary_to_dict(
            differences[policy_name],
            horizon=int(experiment["horizon"]),
        )
    return estimate_dict, difference_dict


def _rank(
    episode_values: Mapping[str, np.ndarray],
    eligible: Sequence[str] | None = None,
) -> list[str]:
    candidates = list(eligible) if eligible is not None else list(episode_values)
    return sorted(candidates, key=lambda name: float(np.mean(episode_values[name])), reverse=True)


def simultaneous_paired_bootstrap_intervals(
    estimator_values: Mapping[str, Mapping[str, np.ndarray]],
    *,
    bootstrap_indices: np.ndarray,
    confidence: float,
) -> dict[str, Any]:
    """Build one max-t family over every unordered policy pair and estimator.

    The patient indices are shared with every ordinary interval in the artifact.
    Studentization makes the joint maximum comparable across the direct and SDR
    scales.  A zero-variance contrast is treated as exactly known under the frozen
    patient-level bootstrap rather than divided by zero.
    """

    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    indices = np.asarray(bootstrap_indices)
    if indices.ndim != 2 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("bootstrap_indices must be a two-dimensional integer array")
    contrast_work: list[dict[str, Any]] = []
    standardized_errors: list[np.ndarray] = []
    for estimator_name, values_by_policy in estimator_values.items():
        if tuple(values_by_policy) != POLICY_NAMES:
            raise ValueError(f"Estimator {estimator_name} must contain policies in canonical order")
        arrays = {
            policy_name: np.asarray(values_by_policy[policy_name], dtype=float)
            for policy_name in POLICY_NAMES
        }
        patient_counts = {len(values) for values in arrays.values()}
        if len(patient_counts) != 1 or patient_counts != {indices.shape[1]}:
            raise ValueError("Every contrast must use the same ordered patients")
        if any(not np.all(np.isfinite(values)) for values in arrays.values()):
            raise ValueError("Simultaneous bootstrap values must be finite")
        for left_policy, right_policy in itertools.combinations(POLICY_NAMES, 2):
            paired_values = arrays[left_policy] - arrays[right_policy]
            point = float(paired_values.mean())
            samples = paired_values[indices].mean(axis=1)
            standard_error = float(np.std(samples, ddof=1))
            if standard_error == 0.0:
                standardized = np.zeros_like(samples)
            else:
                standardized = np.abs((samples - point) / standard_error)
            standardized_errors.append(standardized)
            contrast_work.append(
                {
                    "estimator": estimator_name,
                    "left_policy": left_policy,
                    "right_policy": right_policy,
                    "point_difference": point,
                    "bootstrap_standard_error": standard_error,
                }
            )
    if not contrast_work:
        raise ValueError("At least one simultaneous contrast is required")
    max_statistics = np.max(np.vstack(standardized_errors), axis=0)
    critical_value = float(np.quantile(max_statistics, float(confidence), method="higher"))
    contrasts: list[dict[str, Any]] = []
    for row in contrast_work:
        half_width = critical_value * float(row["bootstrap_standard_error"])
        point = float(row["point_difference"])
        contrasts.append(
            {
                **row,
                "simultaneous_confidence_interval_total": [
                    point - half_width,
                    point + half_width,
                ],
            }
        )
    return {
        "method": "paired-patient bootstrap max-t",
        "confidence_level": float(confidence),
        "family_size": len(contrasts),
        "family_definition": (
            "all six unordered policy pairs jointly across primary HGB-FQE direct "
            "and primary cap-10 sequential-DR"
        ),
        "shared_bootstrap_indices": True,
        "bootstrap_resamples": int(indices.shape[0]),
        "patient_count": int(indices.shape[1]),
        "studentization": "bootstrap standard deviation of each paired contrast",
        "critical_value": critical_value,
        "contrasts": contrasts,
    }


def _directed_simultaneous_contrast(
    simultaneous: Mapping[str, Any],
    *,
    estimator: str,
    winner: str,
    opponent: str,
) -> dict[str, Any]:
    for contrast in simultaneous["contrasts"]:
        if contrast["estimator"] != estimator:
            continue
        left = str(contrast["left_policy"])
        right = str(contrast["right_policy"])
        interval = list(map(float, contrast["simultaneous_confidence_interval_total"]))
        point = float(contrast["point_difference"])
        if left == winner and right == opponent:
            return {"point_difference": point, "interval": interval}
        if left == opponent and right == winner:
            return {
                "point_difference": -point,
                "interval": [-interval[1], -interval[0]],
            }
    raise KeyError(f"Missing simultaneous contrast: {estimator}/{winner}/{opponent}")


def _winner_assessment(
    *,
    metrics: Mapping[str, Any],
    values: Mapping[tuple[str, str, str], Mapping[str, np.ndarray]],
    experiment: Mapping[str, Any],
    designated_cap: float,
    simultaneous: Mapping[str, Any],
) -> dict[str, Any]:
    gates = experiment["winner_gates"]
    gate_results: dict[str, dict[str, bool]] = {}
    eligible: list[str] = []
    for policy_name in POLICY_NAMES:
        policy_metrics = metrics["policies"][policy_name]
        support = policy_metrics["support"]
        checks = {
            "schema_pass": bool(policy_metrics["schema_pass"]),
            "fallback_rate": float(policy_metrics["runtime"]["fallback_rate"])
            <= float(gates["max_fallback_rate"]),
            "mean_statewise_action_overlap": float(support["mean_statewise_action_overlap"])
            >= float(gates["min_mean_overlap"]),
            "low_support_mass": float(support["target_mass_below_behavior_propensity"]["0.01"])
            <= float(gates["max_low_support_mass"]),
            "terminal_raw_ess": float(support["cumulative_raw_ess_by_horizon"][-1])
            >= float(gates["min_terminal_raw_ess"]),
            "maximum_episode_weight_share": float(
                support["maximum_normalized_raw_episode_weight_share_by_horizon"][-1]
            )
            <= float(gates["max_episode_weight_share"]),
        }
        gate_results[policy_name] = checks
        if all(checks.values()):
            eligible.append(policy_name)

    primary_learner = str(experiment["primary_q_learner"])
    sensitivity_learner = str(experiment["q_learner_sensitivity"])
    direct_key = (PRIMARY_REWARD_NAME, primary_learner, "direct")
    direct_values = values[direct_key]
    all_ranking = _rank(direct_values)
    eligible_ranking = _rank(direct_values, eligible)
    if not eligible_ranking:
        return {
            "status": "no eligible policy",
            "best_performing_policy": None,
            "highest_unfiltered_point_estimate": all_ranking[0],
            "robust_winner": False,
            "eligible_policies": [],
            "ranking_all": all_ranking,
            "gate_results": gate_results,
        }

    winner = eligible_ranking[0]
    robust = False
    reasons: list[str] = []
    robustness_diagnostics: dict[str, Any] = {}
    runner = eligible_ranking[1] if len(eligible_ranking) > 1 else None
    if runner is None:
        reasons.append("fewer than two policies passed the prespecified eligibility gates")
    else:
        alternative_rank = _rank(
            values[(ALTERNATIVE_REWARD_NAME, primary_learner, "direct")],
            eligible,
        )
        sensitivity_rank = _rank(
            values[(PRIMARY_REWARD_NAME, sensitivity_learner, "direct")],
            eligible,
        )
        stable_alternative = alternative_rank[0] == winner
        stable_learner = sensitivity_rank[0] == winner
        sensitivity_values = values[(PRIMARY_REWARD_NAME, sensitivity_learner, "direct")]
        opponent_diagnostics: dict[str, Any] = {}
        all_simultaneous_lower_bounds_positive = True
        all_differences_exceed_learner_sensitivity = True
        for opponent in eligible_ranking[1:]:
            direct_contrast = _directed_simultaneous_contrast(
                simultaneous,
                estimator="primary_hgb_fqe_direct",
                winner=winner,
                opponent=opponent,
            )
            sdr_contrast = _directed_simultaneous_contrast(
                simultaneous,
                estimator="primary_hgb_sdr_cap_10",
                winner=winner,
                opponent=opponent,
            )
            primary_difference = float(np.mean(direct_values[winner] - direct_values[opponent]))
            sensitivity_difference = float(
                np.mean(sensitivity_values[winner] - sensitivity_values[opponent])
            )
            learner_sensitivity = abs(primary_difference - sensitivity_difference)
            exceeds_learner_sensitivity = abs(primary_difference) > learner_sensitivity
            direct_positive = float(direct_contrast["interval"][0]) > 0.0
            sdr_positive = float(sdr_contrast["interval"][0]) > 0.0
            all_simultaneous_lower_bounds_positive &= direct_positive and sdr_positive
            all_differences_exceed_learner_sensitivity &= exceeds_learner_sensitivity
            opponent_diagnostics[opponent] = {
                "primary_fqe_difference": primary_difference,
                "ridge_fqe_difference": sensitivity_difference,
                "absolute_q_learner_sensitivity_of_difference": learner_sensitivity,
                "difference_exceeds_q_learner_sensitivity": exceeds_learner_sensitivity,
                "primary_fqe_simultaneous_interval": direct_contrast["interval"],
                "primary_sdr_cap_10_simultaneous_interval": sdr_contrast["interval"],
                "both_simultaneous_lower_bounds_positive": direct_positive and sdr_positive,
            }
        robust = all(
            (
                all_simultaneous_lower_bounds_positive,
                stable_alternative,
                stable_learner,
                all_differences_exceed_learner_sensitivity,
            )
        )
        robustness_diagnostics = {
            "simultaneous_family_method": simultaneous["method"],
            "simultaneous_family_size": simultaneous["family_size"],
            "all_eligible_opponents_pass_both_estimators": (all_simultaneous_lower_bounds_positive),
            "alternative_reward_ranking": alternative_rank,
            "ridge_q_learner_ranking": sensitivity_rank,
            "top_rank_stable_under_alternative_reward": stable_alternative,
            "top_rank_stable_under_ridge_q_learner": stable_learner,
            "all_primary_differences_exceed_q_learner_sensitivity": (
                all_differences_exceed_learner_sensitivity
            ),
            "eligible_opponent_contrasts": opponent_diagnostics,
        }
        if not all_simultaneous_lower_bounds_positive:
            reasons.append(
                "a familywise simultaneous FQE or cap-10 SDR lower bound versus an "
                "eligible opponent was not positive"
            )
        if not stable_alternative:
            reasons.append("top rank changed under the alternative reward")
        if not stable_learner:
            reasons.append("top rank changed under the ridge Q-learner sensitivity")
        if not all_differences_exceed_learner_sensitivity:
            reasons.append("a primary difference did not exceed its Q-learner sensitivity")

    return {
        "status": "robust winner" if robust else "highest point estimate; no robust winner",
        "best_performing_policy": winner,
        "runner_up": runner,
        "robust_winner": robust,
        "eligible_policies": eligible_ranking,
        "ranking_all": all_ranking,
        "gate_results": gate_results,
        "non_robust_reasons": reasons,
        "robustness_diagnostics": robustness_diagnostics,
    }


def evaluate_policies(
    *,
    training_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    policies: Mapping[str, Any],
    observation_columns: Sequence[str],
    config: Mapping[str, Any],
    policy_development_ids: Sequence[str] | None = None,
    policy_role_manifest_checksum: str | None = None,
    run_provider_hospital_sensitivity: bool = True,
    run_seed_refit_sensitivity: bool = True,
) -> EvaluationBundle:
    """Evaluate the four already-frozen policies through the same OPE stack."""

    if set(policies) != set(POLICY_NAMES):
        raise ValueError(f"Expected exactly policies {POLICY_NAMES}, got {sorted(policies)}")
    experiment = config["experiment"]
    horizon = int(experiment["horizon"])
    gamma = float(experiment["gamma"])
    seed = int(experiment["seed"])
    caps = tuple(float(value) for value in experiment["importance_ratio_caps"])
    designated_cap = float(experiment["primary_importance_ratio_cap"])
    if designated_cap not in caps:
        raise ValueError("primary_importance_ratio_cap must be one of importance_ratio_caps")
    learners = (
        str(experiment["primary_q_learner"]),
        str(experiment["q_learner_sensitivity"]),
    )

    training = validate_trajectory_frame(training_frame, horizon=horizon)
    validation = validate_trajectory_frame(validation_frame, horizon=horizon)
    split = deterministic_training_roles(
        training["patient_id"].astype(str),
        policy_development_fraction=float(experiment["policy_development_fraction"]),
        seed=seed,
    )
    if policy_development_ids is not None and set(policy_development_ids) != set(
        split.policy_development_ids
    ):
        raise ValueError("Frozen policy role IDs do not match the canonical deterministic split")
    if (
        policy_role_manifest_checksum is not None
        and str(policy_role_manifest_checksum) != split.manifest_checksum
    ):
        raise ValueError("Frozen policy role checksum does not match the evaluator split")
    assert_disjoint_patient_sets(
        policy_development=split.policy_development_ids,
        ope_nuisance=split.ope_nuisance_ids,
        validation=validation["patient_id"].astype(str).unique(),
    )
    nuisance = training.loc[
        training["patient_id"].astype(str).isin(split.ope_nuisance_ids)
    ].reset_index(drop=True)
    development_rows = training["patient_id"].astype(str).isin(split.policy_development_ids)
    if int(development_rows.sum() + len(nuisance)) != len(training):
        raise RuntimeError("Training role rows do not exactly cover the training frame")

    target_probabilities: dict[str, dict[str, np.ndarray]] = {}
    runtime: dict[str, dict[str, Any]] = {}
    policy_fit_scope: dict[str, dict[str, Any]] = {}
    for policy_name in POLICY_NAMES:
        policy = policies[policy_name]
        policy_fit_scope[policy_name] = _audit_policy_fit_scope(
            policy,
            allowed_patient_ids=split.policy_development_ids,
        )
        if policy_name == "behavior_clone" and not policy_fit_scope[policy_name]["receipts"]:
            raise RuntimeError("Behavior clone did not expose a policy-development fit receipt")
        if policy_name == "behavior_clone":
            behavior_fit_ids = {str(value) for value in getattr(policy, "fitted_patient_ids_", ())}
            if behavior_fit_ids != set(split.policy_development_ids):
                raise RuntimeError(
                    "Behavior clone was not fit on exactly the policy-development patients"
                )
        if policy_name == "llm_improved":
            receipts = policy_fit_scope[policy_name]["receipts"]
            expected_checksum = hashlib.sha256(
                "\n".join(sorted(split.policy_development_ids)).encode()
            ).hexdigest()
            required_fragments = (".action_value_model", ".support_model")
            for fragment in required_fragments:
                matching = [
                    receipt for receipt in receipts if fragment in str(receipt["component"])
                ]
                if not matching:
                    raise RuntimeError(f"Improved policy lacks required fit provenance: {fragment}")
                if not any(
                    int(receipt["patients"]) == len(split.policy_development_ids)
                    and receipt["patient_checksum"] == expected_checksum
                    and bool(receipt["declared_checksum_verified"])
                    for receipt in matching
                ):
                    raise RuntimeError(
                        f"Improved policy fit receipt is not exact development scope: {fragment}"
                    )
        target_probabilities[policy_name] = {
            "nuisance": predict_policy_probabilities(
                policy,
                nuisance,
                observation_columns,
            ),
            "validation": predict_policy_probabilities(
                policy,
                validation,
                observation_columns,
            ),
        }
    for policy_name in POLICY_NAMES:
        runtime[policy_name] = _policy_runtime_diagnostics(
            policies[policy_name],
            len(nuisance) + len(validation),
        )

    behavior_columns = tuple(observation_columns) + ("time_step",)
    behavior_model = _behavior_model(behavior_columns, config).fit(nuisance)
    if set(behavior_model.fit_patient_ids_) != set(split.ope_nuisance_ids):
        raise RuntimeError("Behavior model did not fit exactly the OPE nuisance patient role")
    validation_behavior = behavior_model.predict_proba(validation)

    reward_configs = _reward_configs(config)
    nuisance_rewards = {
        name: factual_reward_vector(nuisance, reward_config)
        for name, reward_config in reward_configs.items()
    }
    validation_rewards = {
        name: factual_reward_vector(validation, reward_config)
        for name, reward_config in reward_configs.items()
    }

    improved_policy = policies["llm_improved"]
    support_component = getattr(improved_policy, "support_model", None)
    if support_component is None:
        raise RuntimeError("Improved policy does not expose its train-only behavior anchor")
    support_anchor = {
        "nuisance": predict_policy_probabilities(
            support_component,
            nuisance,
            observation_columns,
        ),
        "validation": predict_policy_probabilities(
            support_component,
            validation,
            observation_columns,
        ),
    }
    llm_weight = float(getattr(improved_policy, "behavior_blend_llm_weight", 1.0))
    if not 0.0 < llm_weight <= 1.0:
        raise RuntimeError("Improved policy exposes an invalid LLM blend weight")
    softened_base = (
        target_probabilities["llm_improved"]["validation"]
        - (1.0 - llm_weight) * support_anchor["validation"]
    ) / llm_weight
    softened_base = np.maximum(softened_base, 0.0)
    softened_base /= softened_base.sum(axis=1, keepdims=True)
    reconstructed_blend = (
        llm_weight * softened_base + (1.0 - llm_weight) * support_anchor["validation"]
    )
    if not np.allclose(
        reconstructed_blend,
        target_probabilities["llm_improved"]["validation"],
        atol=1e-12,
        rtol=0.0,
    ):
        raise RuntimeError("Could not exactly reconstruct the softened base blend component")
    training_marginal = np.asarray(
        [
            float(np.mean(training["observed_clinician_action"].astype(str) == action))
            for action in ACTION_NAMES
        ]
    )
    constant_training_marginal = np.tile(training_marginal, (len(validation), 1))
    uniform_validation = np.full(
        (len(validation), len(ACTION_NAMES)),
        1.0 / len(ACTION_NAMES),
    )
    calibration_probabilities = {
        "base_zero_shot": target_probabilities["llm_zero_shot"]["validation"],
        "base_component_pre_blend": softened_base,
        "improved_behavior_blend": target_probabilities["llm_improved"]["validation"],
        "train_only_behavior_anchor": support_anchor["validation"],
        "constant_training_action_marginal": constant_training_marginal,
        "uniform": uniform_validation,
    }
    logged_validation_actions = validation["observed_clinician_action"].astype(str).to_numpy()

    metrics: dict[str, Any] = {
        "schema_version": "counterledger-evaluation-v1",
        "evidence_boundary": {
            "handoff_notes_used": False,
            "external_challenge_set_used": False,
            "test_outcomes_used": False,
            "clinical_validation_claimed": False,
            "validation_previously_inspected": bool(
                config.get("evidence", {}).get("validation_previously_inspected", False)
            ),
            "target_policy_origin": config.get("evidence", {}).get(
                "target_policy_origin", "caller_supplied"
            ),
            "validation_status": (
                config.get("evidence", {}).get(
                    "validation_status", "caller_supplied_frozen_policy_characterization"
                )
            ),
        },
        "experiment": {
            "seed": seed,
            "gamma": gamma,
            "horizon": horizon,
            "validation_patients": int(validation["patient_id"].nunique()),
            "validation_rows": len(validation),
            "bootstrap_resamples": int(experiment["bootstrap_resamples"]),
            "bootstrap_confidence": float(experiment["bootstrap_confidence"]),
            "primary_q_learner": learners[0],
            "q_learner_sensitivity": learners[1],
            "importance_ratio_caps": list(caps),
            "primary_importance_ratio_cap": designated_cap,
            "bootstrap_scope": (
                "held-out validation-patient uncertainty with frozen nuisance models; "
                "training/FQE uncertainty is not included"
            ),
            "identification_assumptions": [
                "sequential exchangeability given observed state",
                "adequate behavior-policy support",
                "correct-enough behavior and Q nuisance models",
            ],
        },
        "training_roles": {
            "policy_development_patients": len(split.policy_development_ids),
            "ope_nuisance_patients": len(split.ope_nuisance_ids),
            "policy_development_rows": int(development_rows.sum()),
            "ope_nuisance_rows": len(nuisance),
            "manifest_sha256": split.manifest_checksum,
            "disjoint": True,
        },
        "behavior_model": {
            **behavior_model.calibration_diagnostics_,
            "feature_columns": list(behavior_columns),
            "uses_provider_or_hospital": False,
            "fit_patient_checksum": behavior_model.fit_patient_checksum_,
        },
        "target_policy_behavioral_calibration": {
            "label_source": "logged validation clinician actions",
            "caveat": (
                "behavior agreement is not calibration to optimal actions, treatment effect, "
                "or clinical utility"
            ),
            "predictors": {
                name: behavioral_calibration_summary(
                    logged_validation_actions,
                    probabilities,
                )
                for name, probabilities in calibration_probabilities.items()
            },
        },
        "reward_shortcut_diagnostics": {
            "scope": (
                "post-selection descriptive audit across supplied train and validation "
                "logged transitions; not used to tune the frozen policy"
            ),
            **reward_shortcut_diagnostics(
                pd.concat([training, validation], ignore_index=True),
                reward_configs[PRIMARY_REWARD_NAME],
            ),
        },
        "policies": {
            policy_name: {
                "schema_pass": True,
                "runtime": runtime[policy_name],
                "fit_scope": policy_fit_scope[policy_name],
                "estimates": {},
            }
            for policy_name in POLICY_NAMES
        },
    }
    bootstrap_samples: dict[str, np.ndarray] = {}
    patient_values: dict[tuple[str, str, str], dict[str, np.ndarray]] = {}
    patient_order: tuple[str, ...] | None = None
    support_inputs: dict[str, tuple[Any, np.ndarray]] = {}

    matched_marginal = target_probabilities["llm_improved"]["validation"].mean(axis=0)
    permutation_rng = np.random.default_rng(seed + 91_337)
    permuted_nuisance = target_probabilities["llm_improved"]["nuisance"][
        permutation_rng.permutation(len(nuisance))
    ]
    permuted_validation = target_probabilities["llm_improved"]["validation"][
        permutation_rng.permutation(len(validation))
    ]
    adversarial_target_probabilities = {
        "matched_marginal_constant": {
            "nuisance": np.tile(matched_marginal, (len(nuisance), 1)),
            "validation": np.tile(matched_marginal, (len(validation), 1)),
        },
        "uniform_llm_ablation": {
            "nuisance": (
                llm_weight * np.full((len(nuisance), len(ACTION_NAMES)), 1.0 / len(ACTION_NAMES))
                + (1.0 - llm_weight) * support_anchor["nuisance"]
            ),
            "validation": (
                llm_weight * uniform_validation + (1.0 - llm_weight) * support_anchor["validation"]
            ),
        },
        "row_permuted_selected": {
            "nuisance": permuted_nuisance,
            "validation": permuted_validation,
        },
        "full_uniform": {
            "nuisance": np.full(
                (len(nuisance), len(ACTION_NAMES)),
                1.0 / len(ACTION_NAMES),
            ),
            "validation": uniform_validation,
        },
    }
    adversarial_patient_values: dict[tuple[str, str, str], dict[str, np.ndarray]] = {}
    adversarial_support_inputs: dict[str, tuple[Any, np.ndarray]] = {}

    for reward_name in reward_configs:
        for learner in learners:
            learner_config = config["fqe"][learner]
            for policy_name in POLICY_NAMES:
                fqe = FiniteHorizonFQE(
                    observation_columns,
                    horizon=horizon,
                    gamma=gamma,
                    learner=learner,
                    learner_config=learner_config,
                    seed=seed,
                ).fit(
                    nuisance,
                    nuisance_rewards[reward_name],
                    target_probabilities[policy_name]["nuisance"],
                )
                if set(fqe.fit_patient_ids_) != set(split.ope_nuisance_ids):
                    raise RuntimeError("FQE fit escaped the OPE nuisance patient role")
                ids, direct_values, all_state_values = _initial_policy_values(
                    validation,
                    fqe,
                    target_probabilities[policy_name]["validation"],
                )
                if patient_order is None:
                    patient_order = ids
                elif ids != patient_order:
                    raise RuntimeError("Patient order changed across paired estimators")
                sdr = evaluate_sdr(
                    validation,
                    rewards=validation_rewards[reward_name],
                    target_probabilities=target_probabilities[policy_name]["validation"],
                    behavior_probabilities=validation_behavior,
                    fqe=fqe,
                    caps=caps,
                )
                if sdr.patient_ids != patient_order:
                    raise RuntimeError("SDR and FQE patient order do not match")
                patient_values.setdefault((reward_name, learner, "direct"), {})[policy_name] = (
                    direct_values
                )
                patient_values.setdefault((reward_name, learner, "sdr_raw"), {})[policy_name] = (
                    sdr.raw_episode_values
                )
                for cap in caps:
                    patient_values.setdefault(
                        (reward_name, learner, f"sdr_cap_{cap}"),
                        {},
                    )[policy_name] = sdr.episode_values_by_cap[cap]
                if reward_name == PRIMARY_REWARD_NAME and learner == learners[0]:
                    support_inputs[policy_name] = (sdr, all_state_values)

    for reward_name in reward_configs:
        for learner in learners:
            learner_config = config["fqe"][learner]
            for control_name, control_probabilities in adversarial_target_probabilities.items():
                fqe = FiniteHorizonFQE(
                    observation_columns,
                    horizon=horizon,
                    gamma=gamma,
                    learner=learner,
                    learner_config=learner_config,
                    seed=seed,
                ).fit(
                    nuisance,
                    nuisance_rewards[reward_name],
                    control_probabilities["nuisance"],
                )
                if set(fqe.fit_patient_ids_) != set(split.ope_nuisance_ids):
                    raise RuntimeError("Adversarial-control FQE escaped the nuisance role")
                ids, direct_values, all_state_values = _initial_policy_values(
                    validation,
                    fqe,
                    control_probabilities["validation"],
                )
                if patient_order is None or ids != patient_order:
                    raise RuntimeError("Adversarial-control patient order changed")
                sdr = evaluate_sdr(
                    validation,
                    rewards=validation_rewards[reward_name],
                    target_probabilities=control_probabilities["validation"],
                    behavior_probabilities=validation_behavior,
                    fqe=fqe,
                    caps=caps,
                )
                if sdr.patient_ids != patient_order:
                    raise RuntimeError("Adversarial-control SDR patient order changed")
                adversarial_patient_values.setdefault((reward_name, learner, "direct"), {})[
                    control_name
                ] = direct_values
                adversarial_patient_values.setdefault((reward_name, learner, "sdr_raw"), {})[
                    control_name
                ] = sdr.raw_episode_values
                for cap in caps:
                    adversarial_patient_values.setdefault(
                        (reward_name, learner, f"sdr_cap_{cap}"), {}
                    )[control_name] = sdr.episode_values_by_cap[cap]
                if reward_name == PRIMARY_REWARD_NAME and learner == learners[0]:
                    adversarial_support_inputs[control_name] = (sdr, all_state_values)

    if patient_order is None:
        raise RuntimeError("Evaluation produced no patient-level values")
    bootstrap_samples["patient_ids"] = np.asarray(patient_order, dtype=str)

    for (reward_name, learner, estimator), values_by_policy in patient_values.items():
        if not all(np.all(np.isfinite(values)) for values in values_by_policy.values()):
            if estimator != "sdr_raw":
                raise RuntimeError(
                    f"Non-finite patient values for {reward_name}/{learner}/{estimator}"
                )
            finite_values = {
                policy_name: values
                for policy_name, values in values_by_policy.items()
                if np.all(np.isfinite(values))
            }
            invalid_policies = set(values_by_policy) - set(finite_values)
            for policy_name in sorted(invalid_policies):
                values = values_by_policy[policy_name]
                reward_metrics = metrics["policies"][policy_name]["estimates"].setdefault(
                    reward_name, {}
                )
                reward_metrics.setdefault(learner, {})[estimator] = {
                    "available": False,
                    "reason": (
                        "unclipped raw importance weights produced a non-finite episode return; "
                        "the raw estimator is invalid and was not silently clipped"
                    ),
                    "finite_episode_fraction": float(np.isfinite(values).mean()),
                }
            if BASELINE_POLICY in finite_values:
                key = f"{reward_name}__{learner}__{estimator}__finite_only"
                estimates, differences = _bootstrap_estimator(
                    finite_values,
                    baseline=BASELINE_POLICY,
                    experiment=experiment,
                    key=key,
                    sample_store=bootstrap_samples,
                )
                for policy_name in finite_values:
                    estimator_metrics = estimates[policy_name]
                    estimator_metrics["available"] = True
                    estimator_metrics["difference_vs_always_maintain"] = differences[policy_name]
                    metrics["policies"][policy_name]["estimates"].setdefault(
                        reward_name, {}
                    ).setdefault(learner, {})[estimator] = estimator_metrics
            else:
                for policy_name, values in finite_values.items():
                    estimates, _, indices = patient_level_paired_bootstrap(
                        {policy_name: values},
                        baseline=policy_name,
                        resamples=int(experiment["bootstrap_resamples"]),
                        confidence=float(experiment["bootstrap_confidence"]),
                        seed=int(experiment["seed"]),
                    )
                    bootstrap_samples.setdefault("paired_patient_indices", indices)
                    estimator_metrics = _summary_to_dict(
                        estimates[policy_name],
                        horizon=horizon,
                    )
                    estimator_metrics["available"] = True
                    estimator_metrics["difference_vs_always_maintain"] = {
                        "available": False,
                        "reason": "baseline raw SDR was non-finite",
                    }
                    metrics["policies"][policy_name]["estimates"].setdefault(
                        reward_name, {}
                    ).setdefault(learner, {})[estimator] = estimator_metrics
            continue
        key = f"{reward_name}__{learner}__{estimator}"
        estimates, differences = _bootstrap_estimator(
            values_by_policy,
            baseline=BASELINE_POLICY,
            experiment=experiment,
            key=key,
            sample_store=bootstrap_samples,
        )
        for policy_name in POLICY_NAMES:
            policy_estimates = metrics["policies"][policy_name]["estimates"]
            estimator_metrics = estimates[policy_name]
            estimator_metrics["difference_vs_always_maintain"] = differences[policy_name]
            policy_estimates.setdefault(reward_name, {}).setdefault(learner, {})[estimator] = (
                estimator_metrics
            )

    paired_indices = np.asarray(bootstrap_samples["paired_patient_indices"])
    metrics["adversarial_controls"] = {
        "role": (
            "post-hoc state-dependence and extrapolation diagnostics; excluded from policy "
            "selection and winner assessment"
        ),
        "matched_marginal_constant": {
            "construction": (
                "one state-independent vector matched to the selected policy's validation "
                "mean probabilities; descriptive same-split decomposition"
            ),
            "probability_vector": {
                action: float(matched_marginal[index]) for index, action in enumerate(ACTION_NAMES)
            },
            "estimates": {},
        },
        "uniform_llm_ablation": {
            "construction": (
                "replace the base component with uniform probabilities while preserving the "
                "configured train-only behavior anchor"
            ),
            "estimates": {},
        },
        "row_permuted_selected": {
            "construction": (
                "deterministically permute the selected probability rows within each OPE "
                "split, preserving the exact marginal distribution while breaking its "
                "state alignment"
            ),
            "permutation_seed": seed + 91_337,
            "estimates": {},
        },
        "full_uniform": {
            "construction": (
                "fully uniform policy used only to expose Q-model extrapolation and support failure"
            ),
            "estimates": {},
        },
    }
    for (reward_name, learner, estimator), values_by_control in adversarial_patient_values.items():
        baseline_values = patient_values[(reward_name, learner, estimator)][BASELINE_POLICY]
        improved_values = patient_values[(reward_name, learner, estimator)]["llm_improved"]
        for control_name, control_values in values_by_control.items():
            destination = metrics["adversarial_controls"][control_name]["estimates"]
            if (
                not np.all(np.isfinite(control_values))
                or not np.all(np.isfinite(baseline_values))
                or not np.all(np.isfinite(improved_values))
            ):
                destination.setdefault(reward_name, {}).setdefault(learner, {})[estimator] = {
                    "available": False,
                    "reason": "non-finite raw importance-weight return",
                }
                continue
            summaries, differences = _bootstrap_estimator(
                {
                    BASELINE_POLICY: baseline_values,
                    control_name: control_values,
                },
                baseline=BASELINE_POLICY,
                experiment=experiment,
                key=f"control__{control_name}__{reward_name}__{learner}__{estimator}",
                sample_store=bootstrap_samples,
            )
            control_summary = summaries[control_name]
            control_summary["available"] = True
            control_summary["difference_vs_always_maintain"] = differences[control_name]
            control_summary["difference_vs_llm_improved"] = paired_bootstrap_gap_summary(
                control_values,
                improved_values,
                paired_indices,
                confidence=float(experiment["bootstrap_confidence"]),
                left_label=control_name,
                right_label="llm_improved",
            )
            destination.setdefault(reward_name, {}).setdefault(learner, {})[estimator] = (
                control_summary
            )

    for control_name, (sdr, state_values) in adversarial_support_inputs.items():
        control_target = adversarial_target_probabilities[control_name]["validation"]
        metrics["adversarial_controls"][control_name]["decision_rule"] = decision_rule_summary(
            control_target
        )
        metrics["adversarial_controls"][control_name]["support"] = support_diagnostics(
            validation,
            target_probabilities=control_target,
            behavior_probabilities=validation_behavior,
            sdr=sdr,
            thresholds=experiment["support_propensity_thresholds"],
            direct_policy_values=state_values,
        )

    selected_validation = target_probabilities["llm_improved"]["validation"]
    matched_validation = adversarial_target_probabilities["matched_marginal_constant"]["validation"]
    metrics["state_dependence_diagnostics"] = {
        "selected_vs_matched_marginal_constant": {
            "mean_total_variation": float(
                np.mean(0.5 * np.abs(selected_validation - matched_validation).sum(axis=1))
            ),
            "maximum_total_variation": float(
                np.max(0.5 * np.abs(selected_validation - matched_validation).sum(axis=1))
            ),
            "per_action_probability_standard_deviation": {
                action: float(selected_validation[:, index].std(ddof=1))
                for index, action in enumerate(ACTION_NAMES)
            },
        },
        "selected_vs_row_permuted": {
            "mean_total_variation": float(
                np.mean(0.5 * np.abs(selected_validation - permuted_validation).sum(axis=1))
            ),
            "maximum_total_variation": float(
                np.max(0.5 * np.abs(selected_validation - permuted_validation).sum(axis=1))
            ),
            "marginal_probabilities_preserved": bool(
                np.allclose(
                    selected_validation.mean(axis=0),
                    permuted_validation.mean(axis=0),
                    atol=5e-15,
                    rtol=0.0,
                )
            ),
            "marginal_probability_max_abs_difference": float(
                np.max(np.abs(selected_validation.mean(axis=0) - permuted_validation.mean(axis=0)))
            ),
        },
    }
    metrics["decision_rule_diagnostics"] = {
        policy_name: decision_rule_summary(target_probabilities[policy_name]["validation"])
        for policy_name in POLICY_NAMES
    }
    improved_argmax = np.argmax(selected_validation, axis=1)
    maintain_argmax = np.argmax(target_probabilities["always_maintain"]["validation"], axis=1)
    metrics["decision_rule_diagnostics"]["llm_improved"]["argmax_identical_to_always_maintain"] = (
        bool(np.array_equal(improved_argmax, maintain_argmax))
    )
    improved_greedy_targets = {
        split_name: np.eye(len(ACTION_NAMES), dtype=float)[
            np.argmax(target_probabilities["llm_improved"][split_name], axis=1)
        ]
        for split_name in ("nuisance", "validation")
    }
    always_targets = {
        split_name: target_probabilities["always_maintain"][split_name]
        for split_name in ("nuisance", "validation")
    }
    greedy_exact_on_all_ope_rows = all(
        np.array_equal(improved_greedy_targets[split_name], always_targets[split_name])
        for split_name in ("nuisance", "validation")
    )
    metrics["decision_rule_diagnostics"]["llm_improved"][
        "greedy_probability_matrix_identical_to_always_maintain_on_nuisance_and_validation"
    ] = greedy_exact_on_all_ope_rows
    metrics["decision_rule_diagnostics"]["llm_improved"]["greedy_ope_identity"] = {
        "policy": "always_maintain" if greedy_exact_on_all_ope_rows else None,
        "exact_on_nuisance_and_validation": greedy_exact_on_all_ope_rows,
        "reason": (
            "identical probability matrices imply identical OPE estimates"
            if greedy_exact_on_all_ope_rows
            else "greedy improved policy is not identical to always-maintain on all OPE rows"
        ),
    }

    for policy_name in POLICY_NAMES:
        sdr, state_values = support_inputs[policy_name]
        support = support_diagnostics(
            validation,
            target_probabilities=target_probabilities[policy_name]["validation"],
            behavior_probabilities=validation_behavior,
            sdr=sdr,
            thresholds=experiment["support_propensity_thresholds"],
            direct_policy_values=state_values,
        )
        target = target_probabilities[policy_name]["validation"]
        support["per_action_support"] = {
            action: {
                "target_mean_probability": float(target[:, action_index].mean()),
                "behavior_mean_probability": float(validation_behavior[:, action_index].mean()),
                "mean_overlap_contribution": float(
                    np.minimum(
                        target[:, action_index],
                        validation_behavior[:, action_index],
                    ).mean()
                ),
                "target_mass_below_behavior_propensity": {
                    str(float(threshold)): float(
                        np.mean(
                            target[:, action_index]
                            * (validation_behavior[:, action_index] < float(threshold))
                        )
                    )
                    for threshold in experiment["support_propensity_thresholds"]
                },
            }
            for action_index, action in enumerate(ACTION_NAMES)
        }
        metrics["policies"][policy_name]["support"] = support

    zero_shot_actions = np.argmax(target_probabilities["llm_zero_shot"]["validation"], axis=1)
    improved_actions = np.argmax(target_probabilities["llm_improved"]["validation"], axis=1)
    metrics["policy_agreement"] = {
        "llm_zero_shot_vs_improved_argmax_agreement": float(
            np.mean(zero_shot_actions == improved_actions)
        ),
        "disagreement_rows": int(np.sum(zero_shot_actions != improved_actions)),
        "mean_total_variation_distance": float(
            0.5
            * np.abs(
                target_probabilities["llm_zero_shot"]["validation"]
                - target_probabilities["llm_improved"]["validation"]
            )
            .sum(axis=1)
            .mean()
        ),
    }
    metrics["direct_vs_sdr"] = {
        "primary_q_learner": learners[0],
        "importance_ratio_cap": designated_cap,
        "interpretation": (
            "direct FQE has lower sampling variance but is vulnerable to Q-model "
            "extrapolation; SDR adds logged-action correction but is higher variance. "
            "The prespecified direct estimate is primary and SDR acts as a robustness veto."
        ),
        "policies": {
            policy_name: {
                "direct_total": float(
                    patient_values[(PRIMARY_REWARD_NAME, learners[0], "direct")][policy_name].mean()
                ),
                "sdr_total": float(
                    patient_values[
                        (
                            PRIMARY_REWARD_NAME,
                            learners[0],
                            f"sdr_cap_{designated_cap}",
                        )
                    ][policy_name].mean()
                ),
                "paired_gap": paired_bootstrap_gap_summary(
                    patient_values[
                        (
                            PRIMARY_REWARD_NAME,
                            learners[0],
                            f"sdr_cap_{designated_cap}",
                        )
                    ][policy_name],
                    patient_values[(PRIMARY_REWARD_NAME, learners[0], "direct")][policy_name],
                    paired_indices,
                    confidence=float(experiment["bootstrap_confidence"]),
                    left_label="sdr",
                    right_label="direct",
                ),
            }
            for policy_name in POLICY_NAMES
        },
    }
    metrics["reward_estimator_stress_grid"] = {
        reward_name: {
            learner: {
                estimator: {
                    "ranking": sorted(
                        POLICY_NAMES,
                        key=lambda name: metrics["policies"][name]["estimates"][reward_name][
                            learner
                        ][estimator]["total_return"],
                        reverse=True,
                    ),
                    "llm_improved_minus_always_maintain": float(
                        metrics["policies"]["llm_improved"]["estimates"][reward_name][learner][
                            estimator
                        ]["total_return"]
                        - metrics["policies"]["always_maintain"]["estimates"][reward_name][learner][
                            estimator
                        ]["total_return"]
                    ),
                }
                for estimator in ("direct", f"sdr_cap_{designated_cap}")
            }
            for learner in learners
        }
        for reward_name in reward_configs
    }
    metrics["probability_references"] = {
        "schema_version": "counterledger-probability-reference-v1",
        "float_encoding": "little-endian-float64-c-order",
        "row_key_encoding": "utf8-json-lines-[patient_id,time_step]",
        "validation": {
            "source_file": config.get("validation_source_name", "in_memory_validation"),
            "rows": int(len(validation)),
            "row_key_sha256": _row_key_sha256(validation),
            "policies": {
                policy_name: _probability_reference(
                    validation,
                    target_probabilities[policy_name]["validation"],
                )
                for policy_name in POLICY_NAMES
            },
        },
    }

    if run_provider_hospital_sensitivity:
        metrics["provider_hospital_nuisance_sensitivity"] = _provider_hospital_sensitivity(
            nuisance=nuisance,
            validation=validation,
            target_probabilities=target_probabilities,
            observation_columns=observation_columns,
            config=config,
            nuisance_rewards=nuisance_rewards[PRIMARY_REWARD_NAME],
            validation_rewards=validation_rewards[PRIMARY_REWARD_NAME],
            patient_order=patient_order or (),
            designated_cap=designated_cap,
        )
    else:
        metrics["provider_hospital_nuisance_sensitivity"] = {"run": False}

    if run_seed_refit_sensitivity:
        metrics["q_refit_seed_sensitivity"] = _seed_refit_sensitivity(
            nuisance=nuisance,
            validation=validation,
            target_probabilities=target_probabilities,
            observation_columns=observation_columns,
            config=config,
            nuisance_rewards=nuisance_rewards[PRIMARY_REWARD_NAME],
            reference_values=patient_values[(PRIMARY_REWARD_NAME, learners[0], "direct")],
        )
    else:
        metrics["q_refit_seed_sensitivity"] = {"run": False}

    paired_indices = np.asarray(bootstrap_samples["paired_patient_indices"])
    metrics["simultaneous_inference"] = simultaneous_paired_bootstrap_intervals(
        {
            "primary_hgb_fqe_direct": patient_values[(PRIMARY_REWARD_NAME, learners[0], "direct")],
            "primary_hgb_sdr_cap_10": patient_values[
                (PRIMARY_REWARD_NAME, learners[0], f"sdr_cap_{designated_cap}")
            ],
        },
        bootstrap_indices=paired_indices,
        confidence=float(experiment["bootstrap_confidence"]),
    )
    metrics["winner_assessment"] = _winner_assessment(
        metrics=metrics,
        values=patient_values,
        experiment=experiment,
        designated_cap=designated_cap,
        simultaneous=metrics["simultaneous_inference"],
    )

    comparison_rows: list[dict[str, Any]] = []
    for policy_name in POLICY_NAMES:
        primary = metrics["policies"][policy_name]["estimates"][PRIMARY_REWARD_NAME][learners[0]]
        comparison_rows.append(
            {
                "policy": policy_name,
                "eligible": policy_name
                in metrics["winner_assessment"].get("eligible_policies", []),
                "primary_fqe_total": primary["direct"]["total_return"],
                "primary_fqe_ci_low": primary["direct"]["confidence_interval_95_total"][0],
                "primary_fqe_ci_high": primary["direct"]["confidence_interval_95_total"][1],
                "primary_sdr_cap_10_total": primary[f"sdr_cap_{designated_cap}"]["total_return"],
                "alternative_fqe_total": metrics["policies"][policy_name]["estimates"][
                    ALTERNATIVE_REWARD_NAME
                ][learners[0]]["direct"]["total_return"],
                "ridge_primary_fqe_total": metrics["policies"][policy_name]["estimates"][
                    PRIMARY_REWARD_NAME
                ][learners[1]]["direct"]["total_return"],
                "mean_action_overlap": metrics["policies"][policy_name]["support"][
                    "mean_statewise_action_overlap"
                ],
                "terminal_raw_ess": metrics["policies"][policy_name]["support"][
                    "cumulative_raw_ess_by_horizon"
                ][-1],
            }
        )

    role_manifest = split.as_frame()
    role_manifest["manifest_sha256"] = split.manifest_checksum
    return EvaluationBundle(
        metrics=metrics,
        role_manifest=role_manifest,
        comparison=pd.DataFrame(comparison_rows),
        bootstrap_samples=bootstrap_samples,
    )


def _provider_hospital_sensitivity(
    *,
    nuisance: pd.DataFrame,
    validation: pd.DataFrame,
    target_probabilities: Mapping[str, Mapping[str, np.ndarray]],
    observation_columns: Sequence[str],
    config: Mapping[str, Any],
    nuisance_rewards: np.ndarray,
    validation_rewards: np.ndarray,
    patient_order: tuple[str, ...],
    designated_cap: float,
) -> dict[str, Any]:
    """Sensitivity only: add provider/hospital to nuisance models, never policies."""

    expanded_columns = tuple(observation_columns) + ("hospital_id", "provider_id")
    expanded_behavior_columns = tuple(observation_columns) + (
        "time_step",
        "hospital_id",
        "provider_id",
    )
    behavior = _behavior_model(expanded_behavior_columns, config).fit(nuisance)
    behavior_probabilities = behavior.predict_proba(validation)
    experiment = config["experiment"]
    learner = str(experiment["primary_q_learner"])
    output: dict[str, Any] = {
        "run": True,
        "label": (
            "provider/hospital included only in evaluation nuisance models; "
            "target policies never receive either identifier"
        ),
        "behavior_model": {
            **behavior.calibration_diagnostics_,
            "feature_columns": list(expanded_behavior_columns),
        },
        "policies": {},
    }
    for policy_name in POLICY_NAMES:
        fqe = FiniteHorizonFQE(
            expanded_columns,
            horizon=int(experiment["horizon"]),
            gamma=float(experiment["gamma"]),
            learner=learner,
            learner_config=config["fqe"][learner],
            seed=int(experiment["seed"]),
        ).fit(
            nuisance,
            nuisance_rewards,
            target_probabilities[policy_name]["nuisance"],
        )
        ids, direct_values, _ = _initial_policy_values(
            validation,
            fqe,
            target_probabilities[policy_name]["validation"],
        )
        if ids != patient_order:
            raise RuntimeError("Provider sensitivity changed patient ordering")
        sdr = evaluate_sdr(
            validation,
            rewards=validation_rewards,
            target_probabilities=target_probabilities[policy_name]["validation"],
            behavior_probabilities=behavior_probabilities,
            fqe=fqe,
            caps=[designated_cap],
        )
        output["policies"][policy_name] = {
            "fqe_total": float(np.mean(direct_values)),
            "fqe_per_decision": float(np.mean(direct_values) / int(experiment["horizon"])),
            "sdr_cap_10_total": float(np.mean(sdr.episode_values_by_cap[designated_cap])),
        }
    return output


def _seed_refit_sensitivity(
    *,
    nuisance: pd.DataFrame,
    validation: pd.DataFrame,
    target_probabilities: Mapping[str, Mapping[str, np.ndarray]],
    observation_columns: Sequence[str],
    config: Mapping[str, Any],
    nuisance_rewards: np.ndarray,
    reference_values: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    experiment = config["experiment"]
    learner = str(experiment["primary_q_learner"])
    alternate_seed = int(experiment["seed"]) + 1
    output: dict[str, Any] = {
        "run": True,
        "alternate_seed": alternate_seed,
        "note": (
            "HGB uses no early-stopping split or subsampling; this verifies "
            "deterministic refitting "
            "rather than claiming full nuisance-training uncertainty"
        ),
        "policies": {},
    }
    for policy_name in POLICY_NAMES:
        fqe = FiniteHorizonFQE(
            observation_columns,
            horizon=int(experiment["horizon"]),
            gamma=float(experiment["gamma"]),
            learner=learner,
            learner_config=config["fqe"][learner],
            seed=alternate_seed,
        ).fit(
            nuisance,
            nuisance_rewards,
            target_probabilities[policy_name]["nuisance"],
        )
        _, values, _ = _initial_policy_values(
            validation,
            fqe,
            target_probabilities[policy_name]["validation"],
        )
        difference = values - reference_values[policy_name]
        output["policies"][policy_name] = {
            "alternate_seed_fqe_total": float(np.mean(values)),
            "mean_difference": float(np.mean(difference)),
            "maximum_absolute_patient_difference": float(np.max(np.abs(difference))),
        }
    return output


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    return value


def write_evaluation_artifacts(
    bundle: EvaluationBundle,
    *,
    config: Mapping[str, Any],
    root: str | Path = ".",
) -> dict[str, Path]:
    root_path = Path(root)
    artifact_config = config["artifacts"]
    paths = {
        "metrics": root_path / artifact_config["metrics_json"],
        "comparison": root_path / artifact_config["comparison_csv"],
        "bootstrap": root_path / artifact_config["bootstrap_npz"],
        "roles": root_path / artifact_config["role_manifest"],
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    paths["metrics"].write_text(
        json.dumps(_json_safe(bundle.metrics), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\r\n",
    )
    bundle.comparison.to_csv(paths["comparison"], index=False, lineterminator="\r\n")
    bundle.role_manifest.to_csv(paths["roles"], index=False, lineterminator="\r\n")
    np.savez_compressed(paths["bootstrap"], **bundle.bootstrap_samples)
    return paths
