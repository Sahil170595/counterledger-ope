"""Public synthetic runner around the full patient-disjoint evaluation stack."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from contracts import ACTION_NAMES, POLICY_NAMES, portable_file_sha256
from evaluate import _json_safe, evaluate_policies, write_evaluation_artifacts
from improvement import (
    AdvantageDistiller,
    ObservationSupportAdapter,
    centered_action_advantages,
    robust_advantage_scale,
)
from policy import AlwaysMaintainPolicy, BehaviorClonePolicy, SupportAwareImprovedPolicy
from reward import ALTERNATIVE_REWARD_CONFIG, PRIMARY_REWARD_CONFIG, RewardConfig
from synthetic import (
    DEFAULT_SEED,
    HORIZON,
    OBSERVATION_COLUMNS,
    SyntheticProbabilityPolicy,
    generate_data,
)
from value_model import (
    BehaviorPropensityModel,
    FiniteHorizonFQE,
    canonical_multiclass_log_loss,
    deterministic_training_roles,
    factual_reward_vector,
    predict_policy_probabilities,
    validate_trajectory_frame,
)


def example_config(seed: int = DEFAULT_SEED) -> dict[str, Any]:
    return {
        "validation_source_name": "synthetic_validation.csv",
        "experiment": {
            "seed": seed,
            "horizon": HORIZON,
            "gamma": 1.0,
            "policy_development_fraction": 0.6,
            "bootstrap_resamples": 200,
            "bootstrap_confidence": 0.95,
            "behavior_probability_floor": 0.001,
            "importance_ratio_caps": [2.0, 10.0, 50.0],
            "primary_importance_ratio_cap": 10.0,
            "primary_q_learner": "hist_gradient_boosting",
            "q_learner_sensitivity": "ridge",
            "support_propensity_thresholds": [0.01, 0.05],
            "winner_gates": {
                "max_fallback_rate": 0.02,
                "min_mean_overlap": 0.5,
                "max_low_support_mass": 0.05,
                "min_terminal_raw_ess": 10,
                "max_episode_weight_share": 0.2,
            },
        },
        "behavior_model": {"c_grid": [0.1, 1.0], "group_folds": 3, "max_iter": 500},
        "fqe": {
            "hist_gradient_boosting": {
                "max_iter": 25,
                "min_samples_leaf": 5,
                "max_leaf_nodes": 7,
                "learning_rate": 0.1,
                "l2_regularization": 1.0,
            },
            "ridge": {"alpha": 5.0},
        },
        "improvement": {"inner_group_folds": 3},
        "reward": asdict(PRIMARY_REWARD_CONFIG),
        "alternative_reward": asdict(ALTERNATIVE_REWARD_CONFIG),
        "artifacts": {
            "metrics_json": "metrics.json",
            "comparison_csv": "comparison.csv",
            "bootstrap_npz": "bootstrap.npz",
            "role_manifest": "roles.csv",
        },
        "evidence": {
            "validation_previously_inspected": True,
            "validation_status": "synthetic_regression_fixture_not_unbiased_generalization",
            "target_policy_origin": "deterministic_synthetic_not_llm",
        },
    }


def _behavior_model(
    observation_columns: Sequence[str],
    config: Mapping[str, Any],
    *,
    seed: int,
) -> BehaviorPropensityModel:
    behavior = config["behavior_model"]
    experiment = config["experiment"]
    return BehaviorPropensityModel(
        observation_columns,
        c_grid=behavior["c_grid"],
        group_folds=int(behavior["group_folds"]),
        max_iter=int(behavior["max_iter"]),
        probability_floor=float(experiment["behavior_probability_floor"]),
        seed=seed,
    )


def _fit_fqe(
    frame: pd.DataFrame,
    rewards: np.ndarray,
    probabilities: np.ndarray,
    observation_columns: Sequence[str],
    config: Mapping[str, Any],
    *,
    seed: int,
) -> FiniteHorizonFQE:
    experiment = config["experiment"]
    return FiniteHorizonFQE(
        observation_columns,
        horizon=int(experiment["horizon"]),
        gamma=float(experiment["gamma"]),
        learner="hist_gradient_boosting",
        learner_config=config["fqe"]["hist_gradient_boosting"],
        seed=seed,
    ).fit(frame, rewards, probabilities)


def _cross_fitted_teacher_advantages(
    frame: pd.DataFrame,
    base_probabilities: np.ndarray,
    observation_columns: Sequence[str],
    config: Mapping[str, Any],
    *,
    seed: int,
) -> np.ndarray:
    """Produce patient-out-of-fold HGB-FQE advantages for distillation."""

    groups = frame["patient_id"].astype(str).to_numpy()
    folds = min(int(config["improvement"]["inner_group_folds"]), len(np.unique(groups)))
    if folds < 2:
        raise ValueError("Cross-fitted teacher requires at least two patients")
    rewards = factual_reward_vector(frame, RewardConfig.from_mapping(config["reward"]))
    output = np.full((len(frame), len(ACTION_NAMES)), np.nan, dtype=float)
    for fold_index, (train_indices, heldout_indices) in enumerate(
        GroupKFold(n_splits=folds).split(frame, groups=groups)
    ):
        teacher = _fit_fqe(
            frame.iloc[train_indices].reset_index(drop=True),
            rewards[train_indices],
            base_probabilities[train_indices],
            observation_columns,
            config,
            seed=seed + 101 * (fold_index + 1),
        )
        q_values = teacher.predict_q(frame.iloc[heldout_indices])
        output[heldout_indices] = centered_action_advantages(
            q_values,
            base_probabilities[heldout_indices],
        )
    if not np.all(np.isfinite(output)):
        raise RuntimeError("Cross-fitted teacher failed to populate every development row")
    return output


def _select_behavior_clone_c(
    frame: pd.DataFrame,
    observation_columns: Sequence[str],
    config: Mapping[str, Any],
) -> tuple[float, list[dict[str, float]]]:
    behavior = config["behavior_model"]
    groups = frame["patient_id"].astype(str).to_numpy()
    labels = frame["observed_clinician_action"].astype(str).to_numpy()
    folds = min(int(behavior["group_folds"]), len(np.unique(groups)))
    results: list[dict[str, float]] = []
    for c_value in map(float, behavior["c_grid"]):
        losses: list[float] = []
        for train_indices, heldout_indices in GroupKFold(n_splits=folds).split(
            frame,
            labels,
            groups,
        ):
            model = BehaviorClonePolicy(
                observation_columns,
                random_state=int(config["experiment"]["seed"]),
                regularization_c=c_value,
            ).fit(frame.iloc[train_indices])
            probabilities = model.predict_proba(frame.iloc[heldout_indices])
            losses.append(canonical_multiclass_log_loss(labels[heldout_indices], probabilities))
        results.append(
            {
                "regularization_c": c_value,
                "grouped_cv_log_loss": float(np.mean(losses)),
            }
        )
    selected = min(
        results,
        key=lambda row: (row["grouped_cv_log_loss"], row["regularization_c"]),
    )
    return float(selected["regularization_c"]), results


def build_policies(
    training: pd.DataFrame,
    *,
    config: Mapping[str, Any] | None = None,
    base_policy: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit development-only adapters; accept a caller-frozen base policy."""

    config = example_config() if config is None else config
    experiment = config["experiment"]
    training = validate_trajectory_frame(training, horizon=int(experiment["horizon"]))
    roles = deterministic_training_roles(
        training["patient_id"].astype(str),
        seed=int(experiment["seed"]),
        policy_development_fraction=float(experiment["policy_development_fraction"]),
    )
    development = training.loc[training.patient_id.isin(roles.policy_development_ids)].reset_index(
        drop=True
    )
    base = SyntheticProbabilityPolicy() if base_policy is None else base_policy
    probabilities = predict_policy_probabilities(base, development, OBSERVATION_COLUMNS)
    c_value, c_results = _select_behavior_clone_c(development, OBSERVATION_COLUMNS, config)
    advantages = _cross_fitted_teacher_advantages(
        development, probabilities, OBSERVATION_COLUMNS, config, seed=int(experiment["seed"])
    )
    distiller = AdvantageDistiller(OBSERVATION_COLUMNS, alpha=5.0).fit(development, advantages)
    scale = robust_advantage_scale(
        centered_action_advantages(distiller.predict_action_values(development), probabilities)
    )
    support = ObservationSupportAdapter(
        _behavior_model(OBSERVATION_COLUMNS, config, seed=int(experiment["seed"])).fit(development),
        OBSERVATION_COLUMNS,
    )
    clone = BehaviorClonePolicy(
        OBSERVATION_COLUMNS, random_state=int(experiment["seed"]), regularization_c=c_value
    ).fit(development)
    parameters = {"base_temperature": 2.0, "advantage_weight": 0.3, "support_weight": 0.2}
    improved = SupportAwareImprovedPolicy(
        base,
        distiller,
        support,
        **parameters,
        advantage_scale=scale,
        selected_variant="behavior_blend_50",
        selected_parameters=parameters,
    )
    policies = dict(zip(POLICY_NAMES, (AlwaysMaintainPolicy(), clone, base, improved), strict=True))
    receipt = {
        "schema": "counterledger-policy-build-v1",
        "base_policy_origin": getattr(base, "policy_metadata", {}).get("origin", "caller_supplied"),
        "model_weights_trained": False,
        "tabular_adapters_fitted": True,
        "selection_validation_rows": 0,
        "selection_test_rows": 0,
        "development_patients": len(roles.policy_development_ids),
        "nuisance_patients": len(roles.ope_nuisance_ids),
        "role_manifest_sha256": roles.manifest_checksum,
        "development_ids": list(roles.policy_development_ids),
        "behavior_regularization_cv": c_results,
        "selected_behavior_c": c_value,
        "transform_parameters": parameters,
        "behavior_blend_base_weight": 0.5,
        "transform_selection": "fixed_example_parameters_not_a_validation_tournament",
        "advantage_scale": scale,
    }
    return policies, receipt


def run_example(
    out: str | Path,
    *,
    seed: int = DEFAULT_SEED,
    data: Mapping[str, pd.DataFrame] | None = None,
    sensitivity: bool = True,
) -> dict[str, Any]:
    """Run the fixture evaluator; caller-supplied frames have explicit provenance."""

    config = example_config(seed)
    data_origin = "fresh_generator_v1" if data is None else "caller_supplied_frames"
    data = generate_data(seed=seed) if data is None else data
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    policies, receipt = build_policies(data["train"], config=config)
    bundle = evaluate_policies(
        training_frame=data["train"],
        validation_frame=data["validation"],
        policies=policies,
        observation_columns=OBSERVATION_COLUMNS,
        config=config,
        policy_development_ids=receipt["development_ids"],
        policy_role_manifest_checksum=receipt["role_manifest_sha256"],
        run_provider_hospital_sensitivity=sensitivity,
        run_seed_refit_sensitivity=sensitivity,
    )
    bundle.metrics["public_example"] = {
        "data_origin": data_origin,
        "model_execution_performed": False,
    }
    write_evaluation_artifacts(bundle, config=config, root=out)
    for split, frame in data.items():
        frame.to_csv(out / f"synthetic_{split}.csv", index=False, lineterminator="\n")
    test_probabilities = predict_policy_probabilities(
        policies["llm_improved"], data["test"], OBSERVATION_COLUMNS
    )
    predictions = data["test"][["patient_id", "time_step"]].copy()
    for index, action in enumerate(ACTION_NAMES):
        predictions[f"prob_{action}"] = test_probabilities[:, index]
    predictions["chosen_action"] = np.asarray(ACTION_NAMES)[test_probabilities.argmax(axis=1)]
    predictions["policy_origin"] = "deterministic_synthetic_not_llm"
    predictions.to_csv(out / "predictions.csv", index=False, lineterminator="\n")
    manifest = {
        "schema": "counterledger-synthetic-input-v1",
        "seed": seed,
        "data_origin": data_origin,
        "splits": {
            split: {
                "rows": len(frame),
                "patients": int(frame.patient_id.nunique()),
                "canonical_csv_sha256": portable_file_sha256(out / f"synthetic_{split}.csv"),
            }
            for split, frame in data.items()
        },
        "runtime_versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scikit-learn", "scipy")
        },
        "test_outcomes_used": False,
    }
    for filename, value in (
        ("policy_build.json", receipt),
        ("input_manifest.json", manifest),
        ("config.json", config),
    ):
        (out / filename).write_text(
            json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return _json_safe(bundle.metrics)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path, default=Path("output/example"))
    args = parser.parse_args(argv)
    metrics = run_example(args.out, seed=args.seed)
    print(
        json.dumps(
            {
                "output": str(args.out),
                "origin": "synthetic_not_clinical",
                "winner_assessment": metrics["winner_assessment"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
