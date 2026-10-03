"""Auditable diagnostics for policy calibration, decisions, and reward shortcuts.

These helpers are descriptive.  Agreement with logged clinician actions is not
evidence that a policy is clinically correct, and factual outcome differences by
logged action are not causal treatment effects.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from contracts import ACTION_NAMES, validate_probability_matrix
from reward import RewardConfig


def _action_indices(actions: Sequence[str]) -> np.ndarray:
    action_to_index = {action: index for index, action in enumerate(ACTION_NAMES)}
    values = np.asarray(actions, dtype=str)
    unknown = sorted(set(values) - set(action_to_index))
    if unknown:
        raise ValueError(f"Unknown logged actions: {unknown}")
    return np.asarray([action_to_index[action] for action in values], dtype=int)


def behavioral_calibration_summary(
    logged_actions: Sequence[str],
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    *,
    bins: int = 10,
) -> dict[str, Any]:
    """Compare a policy distribution with logged actions at the same rows.

    This is deliberately named *behavioral* calibration: the labels are actions
    selected by the logging policy, not counterfactual optimal-action labels.
    """

    matrix = validate_probability_matrix(probabilities)
    targets = _action_indices(logged_actions)
    if len(targets) != len(matrix):
        raise ValueError("Calibration labels and probability rows differ")
    if bins <= 0:
        raise ValueError("bins must be positive")
    row_index = np.arange(len(matrix))
    clipped = np.clip(matrix[row_index, targets], 1e-15, 1.0)
    one_hot = np.eye(len(ACTION_NAMES), dtype=float)[targets]
    predictions = np.argmax(matrix, axis=1)
    confidence = np.max(matrix, axis=1)
    correct = (predictions == targets).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    bin_rows: list[dict[str, Any]] = []
    for index in range(bins):
        lower = float(edges[index])
        upper = float(edges[index + 1])
        selected = (confidence >= lower) & (
            confidence <= upper if index == bins - 1 else confidence < upper
        )
        count = int(selected.sum())
        if not count:
            continue
        mean_confidence = float(confidence[selected].mean())
        accuracy = float(correct[selected].mean())
        ece += (count / len(matrix)) * abs(accuracy - mean_confidence)
        bin_rows.append(
            {
                "lower": lower,
                "upper": upper,
                "rows": count,
                "mean_confidence": mean_confidence,
                "top1_accuracy": accuracy,
            }
        )
    return {
        "interpretation": (
            "agreement with logged clinician actions; not calibration to optimal actions "
            "or clinical utility"
        ),
        "rows": int(len(matrix)),
        "multiclass_log_loss": float(-np.log(clipped).mean()),
        "multiclass_brier": float(np.square(matrix - one_hot).sum(axis=1).mean()),
        "top_label_ece": float(ece),
        "top1_accuracy": float(correct.mean()),
        "bins": bin_rows,
    }


def decision_rule_summary(
    probabilities: Sequence[Sequence[float]] | np.ndarray,
) -> dict[str, Any]:
    """Summarize the deterministic argmax rule implied by a probability policy."""

    matrix = validate_probability_matrix(probabilities)
    choices = np.argmax(matrix, axis=1)
    ranked = np.partition(matrix, -2, axis=1)
    top = ranked[:, -1]
    second = ranked[:, -2]
    return {
        "rows": int(len(matrix)),
        "argmax_action_counts": {
            action: int(np.sum(choices == index)) for index, action in enumerate(ACTION_NAMES)
        },
        "argmax_action_distribution": {
            action: float(np.mean(choices == index)) for index, action in enumerate(ACTION_NAMES)
        },
        "minimum_top1_top2_margin": float(np.min(top - second)),
        "mean_top1_top2_margin": float(np.mean(top - second)),
    }


def paired_bootstrap_gap_summary(
    left: Sequence[float] | np.ndarray,
    right: Sequence[float] | np.ndarray,
    bootstrap_indices: np.ndarray,
    *,
    confidence: float = 0.95,
    left_label: str,
    right_label: str,
) -> dict[str, Any]:
    """Summarize a paired mean gap using pre-generated patient bootstrap rows."""

    left_values = np.asarray(left, dtype=float)
    right_values = np.asarray(right, dtype=float)
    indices = np.asarray(bootstrap_indices, dtype=int)
    if left_values.shape != right_values.shape or left_values.ndim != 1:
        raise ValueError("Paired values must be equal-length vectors")
    if indices.ndim != 2 or indices.shape[1] != len(left_values):
        raise ValueError("Bootstrap indices do not match the paired values")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    gaps = left_values - right_values
    draws = gaps[indices].mean(axis=1)
    alpha = 1.0 - confidence
    interval = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        "contrast": f"{left_label}_minus_{right_label}",
        "point_estimate": float(gaps.mean()),
        "confidence": float(confidence),
        "confidence_interval": [float(interval[0]), float(interval[1])],
        "bootstrap_standard_error": float(draws.std(ddof=1)),
        "interval_excludes_zero": bool(interval[0] > 0.0 or interval[1] < 0.0),
        "paired_patients": int(len(gaps)),
        "bootstrap_resamples": int(len(draws)),
    }


def _reward_components(frame: pd.DataFrame, config: RewardConfig) -> pd.DataFrame:
    required = {
        "observed_clinician_action",
        "map_mm_hg",
        "next_6h_map_delta",
        "next_6h_lactate_delta",
        "next_6h_deterioration",
        "adverse_hypotension_next_6h",
        "adverse_fluid_overload_next_6h",
        "adverse_tachyarrhythmia_next_6h",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Reward diagnostic frame is missing columns: {missing}")
    output = pd.DataFrame(index=frame.index)
    output["observed_clinician_action"] = frame["observed_clinician_action"].astype(str)
    output["map_mm_hg"] = frame["map_mm_hg"].astype(float)
    output["map_component"] = config.map_weight * np.tanh(
        frame["next_6h_map_delta"].astype(float) / config.map_scale
    )
    output["lactate_component"] = -config.lactate_weight * np.tanh(
        frame["next_6h_lactate_delta"].astype(float) / config.lactate_scale
    )
    output["deterioration_component"] = -config.deterioration_penalty * frame[
        "next_6h_deterioration"
    ].astype(float)
    output["hypotension_component"] = -config.hypotension_penalty * frame[
        "adverse_hypotension_next_6h"
    ].astype(float)
    output["fluid_overload_component"] = -config.fluid_overload_penalty * frame[
        "adverse_fluid_overload_next_6h"
    ].astype(float)
    output["tachyarrhythmia_component"] = -config.tachyarrhythmia_penalty * frame[
        "adverse_tachyarrhythmia_next_6h"
    ].astype(float)
    component_columns = [column for column in output if column.endswith("_component")]
    output["total_reward"] = output[component_columns].sum(axis=1)
    output["next_6h_deterioration"] = frame["next_6h_deterioration"].astype(float)
    output["adverse_tachyarrhythmia_next_6h"] = frame["adverse_tachyarrhythmia_next_6h"].astype(
        float
    )
    return output


def reward_shortcut_diagnostics(
    frame: pd.DataFrame,
    config: RewardConfig,
) -> dict[str, Any]:
    """Expose descriptive reward-component differences by logged action."""

    components = _reward_components(frame, config)

    def comparison(subset: pd.DataFrame) -> dict[str, Any]:
        grouped = subset.groupby("observed_clinician_action", observed=True).mean(numeric_only=True)
        if not {"maintain", "escalate_vasopressor"}.issubset(grouped.index):
            raise ValueError("Reward shortcut diagnostic needs maintain and escalate rows")
        gap = grouped.loc["escalate_vasopressor"] - grouped.loc["maintain"]
        total_gap = float(gap["total_reward"])
        map_gap = float(gap["map_component"])
        return {
            "rows": int(len(subset)),
            "logged_action_counts": {
                str(key): int(value)
                for key, value in subset["observed_clinician_action"].value_counts().items()
            },
            "escalate_minus_maintain": {
                "total_reward": total_gap,
                "map_component": map_gap,
                "map_share_of_total_gap": (
                    float(map_gap / total_gap) if not np.isclose(total_gap, 0.0) else None
                ),
                "deterioration_rate": float(gap["next_6h_deterioration"]),
                "tachyarrhythmia_rate": float(gap["adverse_tachyarrhythmia_next_6h"]),
            },
            "logged_action_means": {
                str(action): {
                    str(column): float(value) for column, value in row.items() if np.isfinite(value)
                }
                for action, row in grouped.iterrows()
            },
        }

    high_map = components.loc[components["map_mm_hg"] > 75.0]
    return {
        "interpretation": (
            "descriptive logged-action association; not a causal action effect or "
            "counterfactual reward comparison"
        ),
        "all_rows": comparison(components),
        "pre_action_map_above_75": comparison(high_map),
    }


def signed_probability_delta_summary(
    neutral: np.ndarray,
    perturbed: np.ndarray,
    *,
    bootstrap_seed: int,
    bootstrap_resamples: int = 10_000,
    group_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Report signed shifts with paired cluster-bootstrap intervals.

    ``group_ids`` identifies the independent sampling unit. The proxy-note
    analysis supplies patient IDs so repeated states from one patient are not
    treated as independent observations. If omitted, each row is its own group.
    """

    reference = validate_probability_matrix(neutral)
    candidate = validate_probability_matrix(perturbed, expected_rows=len(reference))
    delta = candidate - reference
    if group_ids is None:
        groups = np.asarray([str(index) for index in range(len(delta))])
        sampling_unit = "row"
    else:
        groups = np.asarray(group_ids, dtype=str)
        if groups.shape != (len(delta),):
            raise ValueError("group_ids must contain exactly one ID per probability row")
        if np.any(np.char.str_len(groups) == 0):
            raise ValueError("group_ids cannot contain empty IDs")
        sampling_unit = "patient"

    unique_groups, inverse = np.unique(groups, return_inverse=True)
    group_sums = np.zeros((len(unique_groups), len(ACTION_NAMES)), dtype=float)
    group_counts = np.zeros(len(unique_groups), dtype=float)
    np.add.at(group_sums, inverse, delta)
    np.add.at(group_counts, inverse, 1.0)
    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(
        0,
        len(unique_groups),
        size=(bootstrap_resamples, len(unique_groups)),
    )
    sampled_sums = group_sums[indices].sum(axis=1)
    sampled_counts = group_counts[indices].sum(axis=1, keepdims=True)
    draws = sampled_sums / sampled_counts
    return {
        "rows": int(len(delta)),
        "sampling_unit": sampling_unit,
        "sampling_groups": int(len(unique_groups)),
        "paired_bootstrap_resamples": int(bootstrap_resamples),
        "actions": {
            action: {
                "mean_delta": float(delta[:, index].mean()),
                "confidence_interval_95": [
                    float(value) for value in np.quantile(draws[:, index], [0.025, 0.975])
                ],
            }
            for index, action in enumerate(ACTION_NAMES)
        },
    }
