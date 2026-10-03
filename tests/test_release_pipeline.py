import json

import numpy as np
import pytest

from contracts import ACTION_NAMES, POLICY_NAMES
from pipeline import build_policies, run_example
from synthetic import generate_data


def test_fresh_data_are_seeded_complete_and_patient_disjoint():
    first = generate_data(seed=41)
    second = generate_data(seed=41)
    for split in first:
        assert first[split].equals(second[split])
        assert first[split].groupby("patient_id").size().eq(3).all()
    assert not set(first["train"].patient_id) & set(first["validation"].patient_id)
    assert not set(first["test"].patient_id) & set(first["train"].patient_id)
    assert not any(c.startswith("next_") for c in first["test"].columns)
    assert "observed_clinician_action" not in first["test"]


def test_policy_builder_is_independent_of_validation_outcomes():
    data = generate_data(seed=41)
    policies, receipt = build_policies(data["train"])
    assert tuple(policies) == POLICY_NAMES
    assert receipt["model_weights_trained"] is False
    assert receipt["base_policy_origin"] == "deterministic_synthetic_not_llm"
    assert receipt["selection_validation_rows"] == 0
    assert receipt["development_patients"] == 36
    assert receipt["nuisance_patients"] == 24
    assert len(receipt["role_manifest_sha256"]) == 64


def test_full_pipeline_reproduces_artifacts_and_outcome_free_test_prediction(tmp_path):
    first = run_example(tmp_path / "first")
    second = run_example(tmp_path / "second")
    assert first == second
    assert first["evidence_boundary"]["clinical_validation_claimed"] is False
    assert first["evidence_boundary"]["target_policy_origin"] == "deterministic_synthetic_not_llm"
    assert first["training_roles"]["disjoint"] is True
    assert first["experiment"]["validation_patients"] == 24
    assert first["simultaneous_inference"]["family_size"] == 12
    for filename in (
        "metrics.json",
        "comparison.csv",
        "roles.csv",
        "predictions.csv",
        "policy_build.json",
        "input_manifest.json",
    ):
        assert (tmp_path / "first" / filename).read_bytes() == (
            tmp_path / "second" / filename
        ).read_bytes()
    with np.load(tmp_path / "first" / "bootstrap.npz") as a:
        with np.load(tmp_path / "second" / "bootstrap.npz") as b:
            for key in a.files:
                assert np.array_equal(a[key], b[key])
    receipt = json.loads((tmp_path / "first" / "input_manifest.json").read_text())
    assert receipt["data_origin"] == "fresh_generator_v1"
    assert receipt["splits"]["train"]["rows"] == 180


def test_role_overlap_is_rejected_before_evaluation(tmp_path):
    data = generate_data(seed=41)
    data["validation"].loc[:, "patient_id"] = data["train"].patient_id.iloc[0]
    with pytest.raises(ValueError):
        run_example(tmp_path, data=data)


def test_pipeline_greedy_identity_is_observed_not_required(tmp_path):
    metrics = run_example(tmp_path)
    identity = metrics["decision_rule_diagnostics"]["llm_improved"]["greedy_ope_identity"]
    assert isinstance(identity["exact_on_nuisance_and_validation"], bool)
    assert tuple(ACTION_NAMES) == ("maintain", "iv_fluids", "escalate_vasopressor")
