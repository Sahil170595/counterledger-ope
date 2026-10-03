from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from diagnostics import (
    behavioral_calibration_summary,
    decision_rule_summary,
    paired_bootstrap_gap_summary,
    reward_shortcut_diagnostics,
    signed_probability_delta_summary,
)
from reward import RewardConfig


def test_behavioral_calibration_summary_is_exact_for_perfect_predictions() -> None:
    actions = ["maintain", "iv_fluids", "escalate_vasopressor"]
    matrix = np.eye(3)
    summary = behavioral_calibration_summary(actions, matrix)
    assert summary["multiclass_log_loss"] == pytest.approx(0.0, abs=1e-12)
    assert summary["multiclass_brier"] == pytest.approx(0.0)
    assert summary["top_label_ece"] == pytest.approx(0.0)
    assert summary["top1_accuracy"] == pytest.approx(1.0)


def test_decision_rule_summary_exposes_constant_argmax_and_margin() -> None:
    matrix = np.asarray([[0.7, 0.2, 0.1], [0.6, 0.3, 0.1]])
    summary = decision_rule_summary(matrix)
    assert summary["argmax_action_counts"] == {
        "maintain": 2,
        "iv_fluids": 0,
        "escalate_vasopressor": 0,
    }
    assert summary["minimum_top1_top2_margin"] == pytest.approx(0.3)


def test_decision_rule_summary_uses_contract_order_for_exact_ties() -> None:
    summary = decision_rule_summary(np.asarray([[1.0 / 3.0] * 3]))
    assert summary["argmax_action_counts"]["maintain"] == 1
    assert summary["minimum_top1_top2_margin"] == pytest.approx(0.0)


def test_paired_bootstrap_gap_summary_uses_shared_patient_rows() -> None:
    left = np.asarray([1.0, 2.0, 3.0])
    right = np.asarray([0.0, 1.0, 2.0])
    indices = np.asarray([[0, 1, 2], [2, 2, 2], [0, 0, 0]])
    summary = paired_bootstrap_gap_summary(
        left,
        right,
        indices,
        left_label="left",
        right_label="right",
    )
    assert summary["point_estimate"] == pytest.approx(1.0)
    assert summary["confidence_interval"] == pytest.approx([1.0, 1.0])
    assert summary["interval_excludes_zero"] is True


def test_signed_probability_delta_summary_preserves_zero_sum() -> None:
    neutral = np.asarray([[0.7, 0.2, 0.1], [0.6, 0.3, 0.1]])
    perturbed = np.asarray([[0.6, 0.2, 0.2], [0.5, 0.3, 0.2]])
    summary = signed_probability_delta_summary(
        neutral,
        perturbed,
        bootstrap_seed=3,
        bootstrap_resamples=50,
    )
    deltas = [row["mean_delta"] for row in summary["actions"].values()]
    assert sum(deltas) == pytest.approx(0.0)
    assert summary["actions"]["escalate_vasopressor"]["mean_delta"] == pytest.approx(0.1)
    assert summary["sampling_unit"] == "row"
    assert summary["sampling_groups"] == 2


def test_signed_probability_delta_summary_bootstraps_patient_clusters() -> None:
    neutral = np.asarray([[0.7, 0.2, 0.1], [0.6, 0.3, 0.1], [0.5, 0.4, 0.1]])
    perturbed = np.asarray([[0.6, 0.2, 0.2], [0.5, 0.3, 0.2], [0.4, 0.4, 0.2]])
    summary = signed_probability_delta_summary(
        neutral,
        perturbed,
        bootstrap_seed=4,
        bootstrap_resamples=50,
        group_ids=["patient-a", "patient-a", "patient-b"],
    )
    assert summary["sampling_unit"] == "patient"
    assert summary["sampling_groups"] == 2
    assert "paired_t_statistic" not in summary["actions"]["maintain"]


def test_reward_shortcut_diagnostics_is_descriptive_and_component_reconciled() -> None:
    frame = pd.DataFrame(
        {
            "observed_clinician_action": ["maintain", "escalate_vasopressor"],
            "map_mm_hg": [80.0, 80.0],
            "next_6h_map_delta": [0.0, 5.0],
            "next_6h_lactate_delta": [0.0, 0.0],
            "next_6h_deterioration": [0, 0],
            "adverse_hypotension_next_6h": [0, 0],
            "adverse_fluid_overload_next_6h": [0, 0],
            "adverse_tachyarrhythmia_next_6h": [0, 1],
        }
    )
    summary = reward_shortcut_diagnostics(frame, RewardConfig())
    gap = summary["all_rows"]["escalate_minus_maintain"]
    expected = 0.45 * np.tanh(1.0) - 0.7
    assert gap["total_reward"] == pytest.approx(expected)
    assert "not a causal" in summary["interpretation"]
