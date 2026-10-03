from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from contracts import POLICY_NAMES
from evaluate import (
    _audit_policy_fit_scope,
    _json_safe,
    _winner_assessment,
    simultaneous_paired_bootstrap_intervals,
)
from value_model import (
    EvaluationDataError,
    FiniteHorizonFQE,
    assert_disjoint_patient_sets,
    build_fqe_targets,
    canonical_multiclass_log_loss,
    deterministic_training_roles,
    effective_sample_size_from_logs,
    make_policy_observations,
    maximum_normalized_weight_share,
    patient_level_paired_bootstrap,
    sequential_dr_returns,
    support_diagnostics,
    validate_trajectory_frame,
)


def test_json_safe_preserves_boolean_types() -> None:
    assert _json_safe({"python": False, "numpy": np.bool_(True)}) == {
        "python": False,
        "numpy": True,
    }


def _two_step_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "patient_id": "p1",
                "time_step": 0,
                "map_mm_hg": 62.0,
                "previous_action": "maintain",
                "observed_clinician_action": "iv_fluids",
                "terminal": 0,
            },
            {
                "patient_id": "p1",
                "time_step": 1,
                "map_mm_hg": 67.0,
                "previous_action": "iv_fluids",
                "observed_clinician_action": "maintain",
                "terminal": 1,
            },
        ]
    )


def test_training_roles_are_deterministic_disjoint_and_exhaustive() -> None:
    patients = [f"p{index:03d}" for index in range(10)]
    first = deterministic_training_roles(patients, seed=7)
    second = deterministic_training_roles(reversed(patients), seed=7)
    assert first == second
    development = set(first.policy_development_ids)
    nuisance = set(first.ope_nuisance_ids)
    assert len(development) == 6
    assert not development & nuisance
    assert development | nuisance == set(patients)
    assert len(first.manifest_checksum) == 64


def test_patient_role_leakage_guard_reports_pair() -> None:
    with pytest.raises(EvaluationDataError, match="policy_development and validation"):
        assert_disjoint_patient_sets(
            policy_development=["p1", "p2"],
            ope_nuisance=["p3"],
            validation=["p2", "p4"],
        )


def test_policy_component_fit_receipt_fails_closed_on_ope_patient() -> None:
    class FittedComponent:
        fit_patient_ids_ = ("dev_1", "ope_1")

    with pytest.raises(RuntimeError, match="Policy-side fit leakage"):
        _audit_policy_fit_scope(FittedComponent(), allowed_patient_ids=["dev_1"])


def test_policy_fit_audit_discovers_an_undeclared_fitted_helper() -> None:
    class HiddenFittedComponent:
        fit_patient_ids_ = ("ope_1",)

    class PolicyWrapper:
        def __init__(self) -> None:
            self.unlisted_helper = HiddenFittedComponent()

    with pytest.raises(RuntimeError, match="Policy-side fit leakage"):
        _audit_policy_fit_scope(PolicyWrapper(), allowed_patient_ids=["dev_1"])


def test_policy_observation_builder_excludes_identifiers_and_outcomes() -> None:
    frame = _two_step_frame().assign(next_6h_map_delta=[1.0, -1.0], provider_id="secret")
    observations = make_policy_observations(frame, ["map_mm_hg", "previous_action"])
    assert observations[0] == {
        "structured": {"map_mm_hg": 62.0, "previous_action": "maintain"},
        "handoff_note": "[NO HANDOFF NOTE]",
    }
    serialized = repr(observations)
    assert "patient_id" not in serialized
    assert "provider_id" not in serialized
    assert "next_6h" not in serialized


def test_trajectory_validation_rejects_fractional_time_and_cross_patient_history() -> None:
    fractional = _two_step_frame()
    fractional["time_step"] = fractional["time_step"].astype(float)
    fractional.loc[1, "time_step"] = 1.5
    with pytest.raises(EvaluationDataError, match="exact integers"):
        validate_trajectory_frame(fractional, horizon=2)

    bad_history = _two_step_frame()
    bad_history.loc[1, "previous_action"] = "escalate_vasopressor"
    with pytest.raises(EvaluationDataError, match="history is misaligned"):
        validate_trajectory_frame(bad_history, horizon=2)


def test_fqe_targets_zero_terminal_continuation() -> None:
    targets = build_fqe_targets(
        rewards=[1.0, 2.0],
        terminal=[0, 1],
        next_values=[10.0, 999.0],
        gamma=1.0,
    )
    assert np.allclose(targets, [11.0, 2.0])


def test_canonical_log_loss_does_not_use_lexicographic_action_order() -> None:
    probabilities = np.eye(3, dtype=float)
    loss = canonical_multiclass_log_loss(
        ["maintain", "iv_fluids", "escalate_vasopressor"],
        probabilities,
    )
    assert loss == pytest.approx(0.0)


def test_two_step_tabular_fqe_backward_recursion_is_analytic() -> None:
    actions = ("maintain", "iv_fluids", "escalate_vasopressor")
    rows: list[dict[str, object]] = []
    rewards: list[float] = []
    for patient_index in range(30):
        patient_id = f"p{patient_index:02d}"
        first_action = actions[patient_index % 3]
        terminal_action = actions[(patient_index // 3) % 3]
        rows.extend(
            [
                {
                    "patient_id": patient_id,
                    "time_step": 0,
                    "map_mm_hg": 60.0 + patient_index / 10.0,
                    "previous_action": "maintain",
                    "observed_clinician_action": first_action,
                    "terminal": 0,
                },
                {
                    "patient_id": patient_id,
                    "time_step": 1,
                    "map_mm_hg": 61.0 + patient_index / 10.0,
                    "previous_action": first_action,
                    "observed_clinician_action": terminal_action,
                    "terminal": 1,
                },
            ]
        )
        rewards.extend([0.0, float(actions.index(terminal_action) + 1)])
    frame = pd.DataFrame(rows)
    target = np.zeros((len(frame), 3), dtype=float)
    target[:, 0] = 1.0
    fqe = FiniteHorizonFQE(
        ["map_mm_hg", "previous_action"],
        horizon=2,
        gamma=1.0,
        learner="ridge",
        learner_config={"alpha": 1e-8},
    ).fit(frame, rewards, target)

    q_values = fqe.predict_q(frame)
    terminal_rows = frame["time_step"].to_numpy() == 1
    assert q_values[terminal_rows] == pytest.approx(
        np.tile([1.0, 2.0, 3.0], (terminal_rows.sum(), 1)),
        abs=1e-7,
    )
    initial_values = fqe.predict_v(frame, target)[~terminal_rows]
    assert initial_values == pytest.approx(np.ones(30), abs=1e-7)


def test_two_step_sdr_recovers_exact_value_with_exact_q() -> None:
    result = sequential_dr_returns(
        patient_ids=["p1", "p1"],
        time_steps=[0, 1],
        rewards=[1.0, 2.0],
        logged_q_values=[3.0, 2.0],
        current_policy_values=[3.0, 2.0],
        next_policy_values=[2.0, 0.0],
        terminal=[0, 1],
        target_logged_probabilities=[0.5, 0.5],
        behavior_logged_probabilities=[0.5, 0.5],
        horizon=2,
        gamma=1.0,
        caps=[5.0, 10.0, 20.0],
    )
    assert result.raw_episode_values == pytest.approx([3.0])
    for values in result.episode_values_by_cap.values():
        assert values == pytest.approx([3.0])
    assert np.allclose(result.raw_cumulative_log_weights, 0.0)


def test_two_step_sdr_reduces_to_return_when_q_is_zero() -> None:
    result = sequential_dr_returns(
        patient_ids=["p1", "p1"],
        time_steps=[0, 1],
        rewards=[1.0, 2.0],
        logged_q_values=[0.0, 0.0],
        current_policy_values=[0.0, 0.0],
        next_policy_values=[0.0, 0.0],
        terminal=[0, 1],
        target_logged_probabilities=[0.9, 0.9],
        behavior_logged_probabilities=[0.1, 0.1],
        horizon=2,
        gamma=1.0,
        caps=[5.0],
    )
    assert result.raw_episode_values == pytest.approx([171.0])
    assert result.episode_values_by_cap[5.0] == pytest.approx([15.0])
    assert np.exp(result.raw_cumulative_log_weights[0]) == pytest.approx([9.0, 81.0])
    assert np.exp(result.clipped_cumulative_log_weights_by_cap[5.0][0]) == pytest.approx([5.0, 5.0])


def test_zero_target_probability_produces_zero_cumulative_weight() -> None:
    result = sequential_dr_returns(
        patient_ids=["p1", "p1"],
        time_steps=[0, 1],
        rewards=[1.0, 2.0],
        logged_q_values=[0.0, 0.0],
        current_policy_values=[0.0, 0.0],
        next_policy_values=[0.0, 0.0],
        terminal=[0, 1],
        target_logged_probabilities=[0.0, 0.5],
        behavior_logged_probabilities=[0.5, 0.5],
        horizon=2,
        gamma=1.0,
        caps=[5.0],
    )
    assert np.all(np.isneginf(result.raw_cumulative_log_weights))
    assert result.raw_episode_values == pytest.approx([0.0])
    assert result.episode_values_by_cap[5.0] == pytest.approx([0.0])


def test_raw_sdr_overflow_is_invalid_while_clipped_sdr_remains_finite() -> None:
    result = sequential_dr_returns(
        patient_ids=["p1", "p1"],
        time_steps=[0, 1],
        rewards=[1.0, 1.0],
        logged_q_values=[0.0, 0.0],
        current_policy_values=[0.0, 0.0],
        next_policy_values=[0.0, 0.0],
        terminal=[0, 1],
        target_logged_probabilities=[1.0, 1.0],
        behavior_logged_probabilities=[1e-300, 1e-300],
        horizon=2,
        gamma=1.0,
        caps=[5.0],
    )
    assert np.isinf(result.raw_episode_values[0])
    assert result.episode_values_by_cap[5.0] == pytest.approx([10.0])
    frame = _two_step_frame()
    frame["observed_clinician_action"] = "maintain"
    target = np.tile([1.0, 0.0, 0.0], (2, 1))
    behavior = np.tile([1e-300, 0.5, 0.5 - 1e-300], (2, 1))
    diagnostics = support_diagnostics(
        frame,
        target_probabilities=target,
        behavior_probabilities=behavior,
        sdr=result,
    )
    assert diagnostics["raw_sdr_episode_return_valid"] is False
    assert (
        diagnostics["cumulative_raw_weight_quantiles_by_horizon"][-1]["linear_scale_valid"] is False
    )
    assert (
        "not clipped"
        in diagnostics["cumulative_raw_weight_quantiles_by_horizon"][-1]["invalid_reason"]
    )


def test_log_weight_diagnostics_are_scale_stable() -> None:
    logs = np.log(np.asarray([1.0, 2.0, 3.0])) + 700.0
    assert effective_sample_size_from_logs(logs) == pytest.approx(36.0 / 14.0)
    assert maximum_normalized_weight_share(logs) == pytest.approx(0.5)


def test_patient_bootstrap_is_paired() -> None:
    estimates, differences, indices = patient_level_paired_bootstrap(
        {
            "always_maintain": [1.0, 2.0, 3.0, 4.0],
            "candidate": [2.0, 3.0, 4.0, 5.0],
        },
        baseline="always_maintain",
        resamples=200,
        seed=11,
    )
    assert indices.shape == (200, 4)
    assert estimates["candidate"].point_estimate == pytest.approx(3.5)
    assert differences["candidate"].point_estimate == pytest.approx(1.0)
    assert differences["candidate"].confidence_interval == pytest.approx((1.0, 1.0))
    assert np.allclose(differences["candidate"].samples, 1.0)


def _ordered_policy_values(offsets: list[float]) -> dict[str, np.ndarray]:
    return {
        policy_name: np.asarray([offset, offset + 0.1, offset - 0.1])
        for policy_name, offset in zip(POLICY_NAMES, offsets, strict=True)
    }


def test_simultaneous_bootstrap_covers_all_pairs_and_both_estimators() -> None:
    indices = np.random.default_rng(19).integers(0, 3, size=(250, 3))
    result = simultaneous_paired_bootstrap_intervals(
        {
            "primary_hgb_fqe_direct": _ordered_policy_values([0.0, 1.0, 2.0, 3.0]),
            "primary_hgb_sdr_cap_10": _ordered_policy_values([0.0, 0.8, 1.7, 2.5]),
        },
        bootstrap_indices=indices,
        confidence=0.95,
    )
    assert result["method"] == "paired-patient bootstrap max-t"
    assert result["family_size"] == 12
    assert result["shared_bootstrap_indices"] is True
    keys = {
        (row["estimator"], row["left_policy"], row["right_policy"]) for row in result["contrasts"]
    }
    assert len(keys) == 12
    for row in result["contrasts"]:
        low, high = row["simultaneous_confidence_interval_total"]
        assert low <= row["point_difference"] <= high


def test_robust_winner_requires_simultaneous_dominance_in_both_estimators() -> None:
    direct = _ordered_policy_values([0.0, 1.0, 2.0, 3.0])
    sdr = _ordered_policy_values([0.0, 0.8, 1.7, 2.5])
    indices = np.random.default_rng(23).integers(0, 3, size=(250, 3))
    simultaneous = simultaneous_paired_bootstrap_intervals(
        {
            "primary_hgb_fqe_direct": direct,
            "primary_hgb_sdr_cap_10": sdr,
        },
        bootstrap_indices=indices,
        confidence=0.95,
    )
    metrics = {
        "policies": {
            policy_name: {
                "schema_pass": True,
                "runtime": {"fallback_rate": 0.0},
                "support": {
                    "mean_statewise_action_overlap": 0.9,
                    "target_mass_below_behavior_propensity": {"0.01": 0.0},
                    "cumulative_raw_ess_by_horizon": [200.0],
                    "maximum_normalized_raw_episode_weight_share_by_horizon": [0.01],
                },
            }
            for policy_name in POLICY_NAMES
        }
    }
    experiment = {
        "primary_q_learner": "hist_gradient_boosting",
        "q_learner_sensitivity": "ridge",
        "winner_gates": {
            "max_fallback_rate": 0.005,
            "min_mean_overlap": 0.8,
            "max_low_support_mass": 0.01,
            "min_terminal_raw_ess": 100.0,
            "max_episode_weight_share": 0.1,
        },
    }
    values = {
        ("primary", "hist_gradient_boosting", "direct"): direct,
        ("primary", "hist_gradient_boosting", "sdr_cap_10.0"): sdr,
        ("alternative", "hist_gradient_boosting", "direct"): direct,
        ("primary", "ridge", "direct"): direct,
    }
    assessment = _winner_assessment(
        metrics=metrics,
        values=values,
        experiment=experiment,
        designated_cap=10.0,
        simultaneous=simultaneous,
    )
    assert assessment["best_performing_policy"] == "llm_improved"
    assert assessment["status"] == "robust winner"

    conflicting_sdr = dict(sdr)
    conflicting_sdr["llm_improved"] = np.asarray([1.5, 1.6, 1.4])
    conflicting = simultaneous_paired_bootstrap_intervals(
        {
            "primary_hgb_fqe_direct": direct,
            "primary_hgb_sdr_cap_10": conflicting_sdr,
        },
        bootstrap_indices=indices,
        confidence=0.95,
    )
    assessment = _winner_assessment(
        metrics=metrics,
        values=values,
        experiment=experiment,
        designated_cap=10.0,
        simultaneous=conflicting,
    )
    assert assessment["robust_winner"] is False
    assert assessment["status"] == "highest point estimate; no robust winner"
