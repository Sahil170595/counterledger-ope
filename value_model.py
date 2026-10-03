"""Leakage-resistant nuisance models and offline-policy evaluation primitives.

The estimators in this module operate only on factual logged transitions.  A
target policy is evaluated with finite-horizon fitted Q evaluation (FQE) and
sequential doubly robust (SDR) estimation; logged outcomes are never reassigned
to a non-logged action.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from contracts import ACTION_NAMES, validate_policy_columns, validate_probability_matrix
from reward import RewardConfig, compute_reward

PolicyLike = Any
QLearnerName = Literal["hist_gradient_boosting", "ridge"]

DEFAULT_NOTE_SENTINEL = "[NO HANDOFF NOTE]"
ACTION_TO_INDEX = {action: index for index, action in enumerate(ACTION_NAMES)}


class EvaluationDataError(ValueError):
    """Raised when data violate the experiment's temporal or leakage contract."""


@dataclass(frozen=True)
class TrainingRoleSplit:
    """Deterministic, patient-disjoint training roles."""

    policy_development_ids: tuple[str, ...]
    ope_nuisance_ids: tuple[str, ...]
    manifest_checksum: str

    def as_frame(self) -> pd.DataFrame:
        records = [
            {"patient_id": patient_id, "role": "policy_development"}
            for patient_id in self.policy_development_ids
        ]
        records.extend(
            {"patient_id": patient_id, "role": "ope_nuisance"}
            for patient_id in self.ope_nuisance_ids
        )
        return pd.DataFrame(records).sort_values("patient_id", kind="stable").reset_index(drop=True)


def deterministic_training_roles(
    patient_ids: Sequence[str],
    *,
    policy_development_fraction: float = 0.60,
    seed: int = 41,
) -> TrainingRoleSplit:
    """Partition patients using the policy module's canonical SHA-256 rank."""

    unique_ids = tuple(sorted({str(patient_id) for patient_id in patient_ids}))
    if len(unique_ids) < 2:
        raise EvaluationDataError("At least two training patients are required")
    if not 0.0 < policy_development_fraction < 1.0:
        raise ValueError("policy_development_fraction must lie strictly between zero and one")

    # Import lazily to keep the evaluation primitives usable in isolation while
    # ensuring the policy and OPE sides share exactly one production splitter.
    try:
        from policy import deterministic_policy_development_ids

        development = set(
            deterministic_policy_development_ids(
                unique_ids,
                fraction=policy_development_fraction,
                seed=seed,
            )
        )
    except ImportError:
        ranked = sorted(
            unique_ids,
            key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest(),
        )
        count = int(round(len(ranked) * policy_development_fraction))
        count = min(max(count, 1), len(ranked) - 1)
        development = set(ranked[:count])

    nuisance = set(unique_ids) - development
    if not development or not nuisance or development & nuisance:
        raise EvaluationDataError("Training role partition must be non-empty and disjoint")
    if development | nuisance != set(unique_ids):
        raise EvaluationDataError("Training role partition does not cover every patient")

    rows = [
        f"{patient_id},{'policy_development' if patient_id in development else 'ope_nuisance'}"
        for patient_id in unique_ids
    ]
    checksum = hashlib.sha256("\n".join(rows).encode()).hexdigest()
    return TrainingRoleSplit(
        policy_development_ids=tuple(sorted(development)),
        ope_nuisance_ids=tuple(sorted(nuisance)),
        manifest_checksum=checksum,
    )


def assert_disjoint_patient_sets(**named_patient_sets: Sequence[str]) -> None:
    """Fail closed when any named patient role overlaps another."""

    normalized = {
        name: {str(value) for value in values} for name, values in named_patient_sets.items()
    }
    names = tuple(normalized)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            overlap = normalized[left_name] & normalized[right_name]
            if overlap:
                example = sorted(overlap)[0]
                raise EvaluationDataError(
                    f"Patient leakage between {left_name} and {right_name}; example={example}"
                )


def validate_trajectory_frame(frame: pd.DataFrame, *, horizon: int | None = None) -> pd.DataFrame:
    """Validate temporal keys and return a stable episode/time ordering."""

    required = {"patient_id", "time_step", "observed_clinician_action", "terminal"}
    missing = required - set(frame.columns)
    if missing:
        raise EvaluationDataError(f"Trajectory frame is missing columns: {sorted(missing)}")
    if frame.loc[:, list(required)].isna().any().any():
        raise EvaluationDataError(
            "Trajectory keys, logged actions, and terminal flags cannot be null"
        )
    if frame.duplicated(["patient_id", "time_step"]).any():
        raise EvaluationDataError("Duplicate patient/time transitions are not allowed")
    if not set(frame["observed_clinician_action"].dropna().astype(str)).issubset(ACTION_TO_INDEX):
        raise EvaluationDataError("Trajectory frame contains a non-canonical logged action")

    numeric_steps = pd.to_numeric(frame["time_step"], errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(numeric_steps)) or not np.array_equal(
        numeric_steps,
        np.rint(numeric_steps),
    ):
        raise EvaluationDataError("time_step values must be finite exact integers")
    numeric_terminal = pd.to_numeric(frame["terminal"], errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isin(numeric_terminal, (0.0, 1.0))):
        raise EvaluationDataError("terminal values must be exactly 0 or 1")

    ordered = frame.copy()
    ordered["time_step"] = numeric_steps.astype(int)
    ordered["terminal"] = numeric_terminal.astype(int)
    ordered = ordered.sort_values(["patient_id", "time_step"], kind="stable").reset_index(drop=True)
    for patient_id, episode in ordered.groupby("patient_id", sort=False):
        steps = episode["time_step"].to_numpy(dtype=int)
        if not np.array_equal(steps, np.arange(len(episode))):
            raise EvaluationDataError(f"Non-consecutive time steps for patient {patient_id}")
        terminals = episode["terminal"].to_numpy(dtype=int)
        expected = np.zeros(len(episode), dtype=int)
        expected[-1] = 1
        if not np.array_equal(terminals, expected):
            raise EvaluationDataError(f"Terminal flags are invalid for patient {patient_id}")
        if horizon is not None and len(episode) != horizon:
            raise EvaluationDataError(
                f"Patient {patient_id} has {len(episode)} steps, expected {horizon}"
            )
        if "previous_action" in episode.columns and len(episode) > 1:
            previous = episode["previous_action"].iloc[1:].astype(str).to_numpy()
            expected_previous = (
                episode["observed_clinician_action"].iloc[:-1].astype(str).to_numpy()
            )
            if not np.array_equal(previous, expected_previous):
                raise EvaluationDataError(
                    f"previous_action history is misaligned within patient {patient_id}"
                )
    return ordered


def make_policy_observations(
    frame: pd.DataFrame,
    observation_columns: Sequence[str],
    *,
    handoff_note: str = DEFAULT_NOTE_SENTINEL,
) -> list[dict[str, object]]:
    """Construct the only observation shape policy implementations may receive."""

    try:
        columns = validate_policy_columns(observation_columns)
    except ValueError as error:
        raise EvaluationDataError(str(error)) from error
    missing = set(columns) - set(frame.columns)
    if missing:
        raise EvaluationDataError(f"Missing policy observation columns: {sorted(missing)}")
    forbidden = {
        "patient_id",
        "time_step",
        "split",
        "hospital_id",
        "provider_id",
        "observed_clinician_action",
        "terminal",
    }
    forbidden.update(column for column in frame.columns if column.startswith("next_6h_"))
    forbidden.update(column for column in frame.columns if column.startswith("adverse_"))
    overlap = forbidden & set(columns)
    if overlap:
        raise EvaluationDataError(f"Forbidden policy inputs requested: {sorted(overlap)}")

    observations: list[dict[str, object]] = []
    for values in frame.loc[:, columns].itertuples(index=False, name=None):
        structured: dict[str, object] = {}
        for column, value in zip(columns, values, strict=True):
            if pd.isna(value):
                structured[column] = None
            elif isinstance(value, np.generic):
                structured[column] = value.item()
            else:
                structured[column] = value
        observations.append({"structured": structured, "handoff_note": handoff_note})
    return observations


def predict_policy_probabilities(
    policy: PolicyLike,
    frame: pd.DataFrame,
    observation_columns: Sequence[str],
    *,
    handoff_note: str = DEFAULT_NOTE_SENTINEL,
) -> np.ndarray:
    """Call a frozen policy and enforce the shared probability contract."""

    observations = make_policy_observations(
        frame,
        observation_columns,
        handoff_note=handoff_note,
    )
    probabilities = policy.predict_proba(observations)
    return validate_probability_matrix(probabilities, expected_rows=len(frame))


def factual_reward_vector(frame: pd.DataFrame, config: RewardConfig) -> np.ndarray:
    """Compute rewards only for factual rows containing logged outcomes."""

    return np.asarray(
        [compute_reward(row, config) for row in frame.to_dict(orient="records")],
        dtype=float,
    )


def _feature_types(observation_columns: Sequence[str]) -> tuple[list[str], list[str]]:
    categorical_candidates = {"sex", "previous_action", "hospital_id", "provider_id"}
    categorical = [column for column in observation_columns if column in categorical_candidates]
    numeric = [column for column in observation_columns if column not in categorical_candidates]
    return numeric, categorical


def _state_preprocessor(observation_columns: Sequence[str]) -> ColumnTransformer:
    numeric, categorical = _feature_types(observation_columns)
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "one_hot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical,
            )
        )
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)


def _ordered_class_probabilities(model: Any, frame: pd.DataFrame) -> np.ndarray:
    raw = np.asarray(model.predict_proba(frame), dtype=float)
    classes = tuple(str(value) for value in model.classes_)
    if set(classes) != set(ACTION_NAMES):
        raise EvaluationDataError(f"Behavior model classes are invalid: {classes}")
    return raw[:, [classes.index(action) for action in ACTION_NAMES]]


def _multiclass_brier(y_true: Sequence[str], probabilities: np.ndarray) -> float:
    encoded = np.zeros_like(probabilities)
    for row_index, action in enumerate(y_true):
        encoded[row_index, ACTION_TO_INDEX[str(action)]] = 1.0
    return float(np.mean(np.sum((probabilities - encoded) ** 2, axis=1)))


def canonical_multiclass_log_loss(
    y_true: Sequence[str],
    probabilities: np.ndarray,
) -> float:
    """Log loss in canonical action order, independent of sklearn label sorting."""

    matrix = validate_probability_matrix(probabilities, expected_rows=len(y_true))
    selected = logged_action_probabilities(matrix, y_true)
    return float(-np.mean(np.log(np.clip(selected, 1e-15, 1.0))))


def _top_label_ece(y_true: Sequence[str], probabilities: np.ndarray, bins: int = 10) -> float:
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    truth = np.asarray([ACTION_TO_INDEX[str(action)] for action in y_true])
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        include = (confidence > lower) & (confidence <= upper)
        if lower == 0.0:
            include |= confidence == 0.0
        if not include.any():
            continue
        accuracy = np.mean(prediction[include] == truth[include])
        ece += float(include.mean() * abs(accuracy - confidence[include].mean()))
    return ece


class BehaviorPropensityModel:
    """Unweighted, patient-group calibrated multinomial behavior model."""

    def __init__(
        self,
        observation_columns: Sequence[str],
        *,
        c_grid: Sequence[float] = (0.01, 0.1, 1.0, 10.0),
        group_folds: int = 3,
        max_iter: int = 2000,
        probability_floor: float = 1e-3,
        seed: int = 41,
    ) -> None:
        self.observation_columns = tuple(observation_columns)
        self.c_grid = tuple(float(value) for value in c_grid)
        self.group_folds = int(group_folds)
        self.max_iter = int(max_iter)
        self.probability_floor = float(probability_floor)
        self.seed = int(seed)
        if not 0.0 < self.probability_floor < 1.0 / len(ACTION_NAMES):
            raise ValueError("probability_floor must be positive and smaller than 1/3")
        self.model_: Any | None = None
        self.selected_c_: float | None = None
        self.calibration_diagnostics_: dict[str, float | int | list[float]] = {}
        self.fit_patient_ids_: tuple[str, ...] = ()
        self.fit_patient_checksum_: str | None = None
        self.fit_row_count_: int = 0

    def _base_pipeline(self, c_value: float) -> Pipeline:
        return Pipeline(
            [
                ("preprocess", _state_preprocessor(self.observation_columns)),
                (
                    "classifier",
                    LogisticRegression(
                        C=c_value,
                        solver="lbfgs",
                        max_iter=self.max_iter,
                        random_state=self.seed,
                    ),
                ),
            ]
        )

    @staticmethod
    def _fold_count(groups: np.ndarray, requested: int) -> int:
        return min(max(int(requested), 2), len(np.unique(groups)))

    def _select_c(self, features: pd.DataFrame, target: np.ndarray, groups: np.ndarray) -> float:
        folds = self._fold_count(groups, self.group_folds)
        splitter = GroupKFold(n_splits=folds)
        scores: list[float] = []
        for c_value in self.c_grid:
            fold_scores: list[float] = []
            for train_indices, test_indices in splitter.split(features, target, groups):
                model = self._base_pipeline(c_value)
                model.fit(features.iloc[train_indices], target[train_indices])
                probabilities = _ordered_class_probabilities(model, features.iloc[test_indices])
                fold_scores.append(
                    canonical_multiclass_log_loss(target[test_indices], probabilities)
                )
            scores.append(float(np.mean(fold_scores)))
        self.calibration_diagnostics_["c_grid"] = list(self.c_grid)
        self.calibration_diagnostics_["grouped_cv_log_loss"] = scores
        return self.c_grid[int(np.argmin(scores))]

    def _calibrated_model(
        self,
        features: pd.DataFrame,
        target: np.ndarray,
        groups: np.ndarray,
        c_value: float,
    ) -> Any:
        unique_groups = len(np.unique(groups))
        if unique_groups < 2:
            model = self._base_pipeline(c_value)
            model.fit(features, target)
            return model
        splits = list(
            GroupKFold(n_splits=self._fold_count(groups, self.group_folds)).split(
                features,
                target,
                groups,
            )
        )
        model = CalibratedClassifierCV(
            estimator=self._base_pipeline(c_value),
            method="sigmoid",
            cv=splits,
            ensemble=True,
        )
        model.fit(features, target)
        return model

    def _grouped_oof_probabilities(
        self,
        features: pd.DataFrame,
        target: np.ndarray,
        groups: np.ndarray,
        c_value: float,
    ) -> np.ndarray:
        folds = self._fold_count(groups, self.group_folds)
        output = np.full((len(features), len(ACTION_NAMES)), np.nan, dtype=float)
        for train_indices, test_indices in GroupKFold(n_splits=folds).split(
            features,
            target,
            groups,
        ):
            inner_features = features.iloc[train_indices]
            inner_target = target[train_indices]
            inner_groups = groups[train_indices]
            model = self._calibrated_model(
                inner_features,
                inner_target,
                inner_groups,
                c_value,
            )
            output[test_indices] = _ordered_class_probabilities(
                model,
                features.iloc[test_indices],
            )
        if not np.all(np.isfinite(output)):
            raise RuntimeError("Grouped behavior-model OOF predictions are incomplete")
        return output

    def fit(self, frame: pd.DataFrame) -> BehaviorPropensityModel:
        missing = set(self.observation_columns) - set(frame.columns)
        if missing:
            raise EvaluationDataError(f"Missing behavior-model columns: {sorted(missing)}")
        features = frame.loc[:, self.observation_columns]
        target = frame["observed_clinician_action"].astype(str).to_numpy()
        groups = frame["patient_id"].astype(str).to_numpy()
        if set(target) != set(ACTION_NAMES):
            raise EvaluationDataError("Behavior nuisance data must contain all canonical actions")
        self.selected_c_ = self._select_c(features, target, groups)
        self.fit_patient_ids_ = tuple(sorted(np.unique(groups)))
        self.fit_patient_checksum_ = hashlib.sha256(
            "\n".join(self.fit_patient_ids_).encode()
        ).hexdigest()
        self.fit_row_count_ = len(frame)
        oof = self._grouped_oof_probabilities(features, target, groups, self.selected_c_)
        self.calibration_diagnostics_.update(
            {
                "selected_c": self.selected_c_,
                "group_folds": self._fold_count(groups, self.group_folds),
                "oof_log_loss": canonical_multiclass_log_loss(target, oof),
                "oof_multiclass_brier": _multiclass_brier(target, oof),
                "oof_top_label_ece_10_bins": _top_label_ece(target, oof),
                "probability_floor": self.probability_floor,
                "samples": len(frame),
                "patients": len(np.unique(groups)),
            }
        )
        self.model_ = self._calibrated_model(
            features,
            target,
            groups,
            self.selected_c_,
        )
        return self

    def predict_raw_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("BehaviorPropensityModel must be fit before prediction")
        probabilities = _ordered_class_probabilities(
            self.model_,
            frame.loc[:, self.observation_columns],
        )
        return validate_probability_matrix(probabilities, expected_rows=len(frame))

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """Return numerically floored and renormalized behavior probabilities."""

        probabilities = self.predict_raw_proba(frame)
        probabilities = np.maximum(probabilities, self.probability_floor)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return validate_probability_matrix(probabilities, expected_rows=len(frame))


def action_indices(actions: Sequence[str]) -> np.ndarray:
    try:
        return np.asarray([ACTION_TO_INDEX[str(action)] for action in actions], dtype=int)
    except KeyError as error:
        raise EvaluationDataError(f"Unknown action: {error.args[0]}") from error


def logged_action_probabilities(probabilities: np.ndarray, actions: Sequence[str]) -> np.ndarray:
    indices = action_indices(actions)
    return np.asarray(probabilities, dtype=float)[np.arange(len(indices)), indices]


def _action_design(
    state_features: np.ndarray,
    actions: np.ndarray,
    *,
    include_linear_interactions: bool,
) -> np.ndarray:
    action_one_hot = np.eye(len(ACTION_NAMES), dtype=float)[actions]
    pieces = [np.asarray(state_features, dtype=float), action_one_hot]
    if include_linear_interactions:
        pieces.extend(
            state_features * action_one_hot[:, [index]] for index in range(len(ACTION_NAMES))
        )
    return np.hstack(pieces)


class _QStageModel:
    def __init__(
        self,
        observation_columns: Sequence[str],
        *,
        learner: QLearnerName,
        learner_config: Mapping[str, Any],
        seed: int,
    ) -> None:
        self.observation_columns = tuple(observation_columns)
        self.learner = learner
        self.learner_config = dict(learner_config)
        self.seed = int(seed)
        self.preprocessor = _state_preprocessor(self.observation_columns)
        self.regressor: Any | None = None

    def fit(self, frame: pd.DataFrame, actions: np.ndarray, targets: np.ndarray) -> _QStageModel:
        state = np.asarray(
            self.preprocessor.fit_transform(frame.loc[:, self.observation_columns]),
            dtype=float,
        )
        design = _action_design(
            state,
            actions,
            include_linear_interactions=self.learner == "ridge",
        )
        if self.learner == "hist_gradient_boosting":
            self.regressor = HistGradientBoostingRegressor(
                learning_rate=float(self.learner_config.get("learning_rate", 0.05)),
                max_iter=int(self.learner_config.get("max_iter", 150)),
                max_leaf_nodes=int(self.learner_config.get("max_leaf_nodes", 15)),
                min_samples_leaf=int(self.learner_config.get("min_samples_leaf", 50)),
                l2_regularization=float(self.learner_config.get("l2_regularization", 1.0)),
                random_state=self.seed,
                early_stopping=False,
            )
        elif self.learner == "ridge":
            self.regressor = Ridge(alpha=float(self.learner_config.get("alpha", 10.0)))
        else:
            raise ValueError(f"Unsupported Q learner: {self.learner}")
        self.regressor.fit(design, np.asarray(targets, dtype=float))
        return self

    def predict(self, frame: pd.DataFrame, actions: np.ndarray) -> np.ndarray:
        if self.regressor is None:
            raise RuntimeError("Q-stage model must be fit before prediction")
        state = np.asarray(
            self.preprocessor.transform(frame.loc[:, self.observation_columns]),
            dtype=float,
        )
        design = _action_design(
            state,
            actions,
            include_linear_interactions=self.learner == "ridge",
        )
        return np.asarray(self.regressor.predict(design), dtype=float)

    def predict_all_actions(self, frame: pd.DataFrame) -> np.ndarray:
        outputs = [
            self.predict(frame, np.full(len(frame), action_index, dtype=int))
            for action_index in range(len(ACTION_NAMES))
        ]
        return np.column_stack(outputs)


def build_fqe_targets(
    rewards: Sequence[float],
    terminal: Sequence[bool | int],
    next_values: Sequence[float],
    *,
    gamma: float,
) -> np.ndarray:
    """Construct Bellman targets while forcing terminal continuation to zero."""

    reward_values = np.asarray(rewards, dtype=float)
    terminal_values = np.asarray(terminal, dtype=bool)
    continuation = np.asarray(next_values, dtype=float)
    if not (reward_values.shape == terminal_values.shape == continuation.shape):
        raise ValueError("FQE target inputs must have identical one-dimensional shapes")
    if not np.all(np.isfinite(reward_values)) or not np.all(np.isfinite(continuation)):
        raise ValueError("FQE target inputs must be finite")
    return reward_values + float(gamma) * (~terminal_values) * continuation


class FiniteHorizonFQE:
    """Backward, time-indexed finite-horizon fitted Q evaluation."""

    def __init__(
        self,
        observation_columns: Sequence[str],
        *,
        horizon: int = 12,
        gamma: float = 1.0,
        learner: QLearnerName = "hist_gradient_boosting",
        learner_config: Mapping[str, Any] | None = None,
        seed: int = 41,
    ) -> None:
        self.observation_columns = tuple(observation_columns)
        self.horizon = int(horizon)
        self.gamma = float(gamma)
        self.learner = learner
        self.learner_config = dict(learner_config or {})
        self.seed = int(seed)
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must lie in [0, 1]")
        self.stage_models_: dict[int, _QStageModel] = {}
        self.fit_patient_ids_: tuple[str, ...] = ()
        self.fit_patient_checksum_: str | None = None
        self.fit_row_count_: int = 0

    @staticmethod
    def _stage_by_patient(frame: pd.DataFrame, time_step: int) -> pd.DataFrame:
        stage = frame.loc[frame["time_step"].astype(int) == time_step].copy()
        return stage.sort_values("patient_id", kind="stable")

    def fit(
        self,
        frame: pd.DataFrame,
        rewards: Sequence[float],
        target_probabilities: np.ndarray,
    ) -> FiniteHorizonFQE:
        ordered = validate_trajectory_frame(frame, horizon=self.horizon).copy()
        reward_values = np.asarray(rewards, dtype=float)
        probabilities = validate_probability_matrix(target_probabilities, expected_rows=len(frame))

        # The caller may pass an unsorted frame. Align rewards/probabilities to the
        # validated stable order using unique transition keys.
        source_keys = pd.MultiIndex.from_frame(frame[["patient_id", "time_step"]])
        ordered_keys = pd.MultiIndex.from_frame(ordered[["patient_id", "time_step"]])
        positions = source_keys.get_indexer(ordered_keys)
        if (positions < 0).any():
            raise EvaluationDataError("Could not align FQE inputs to trajectory keys")
        reward_values = reward_values[positions]
        probabilities = probabilities[positions]
        ordered["__reward"] = reward_values
        for action_index in range(len(ACTION_NAMES)):
            ordered[f"__pi_{action_index}"] = probabilities[:, action_index]

        self.stage_models_ = {}
        self.fit_patient_ids_ = tuple(sorted(ordered["patient_id"].astype(str).unique()))
        self.fit_patient_checksum_ = hashlib.sha256(
            "\n".join(self.fit_patient_ids_).encode()
        ).hexdigest()
        self.fit_row_count_ = len(ordered)
        next_values_by_patient: pd.Series | None = None

        for time_step in range(self.horizon - 1, -1, -1):
            stage = self._stage_by_patient(ordered, time_step)
            if len(stage) != len(self.fit_patient_ids_):
                raise EvaluationDataError(
                    f"FQE stage {time_step} does not cover every nuisance patient"
                )
            if next_values_by_patient is None:
                next_values = np.zeros(len(stage), dtype=float)
            else:
                next_values = stage["patient_id"].astype(str).map(next_values_by_patient).to_numpy()
                if not np.all(np.isfinite(next_values)):
                    raise EvaluationDataError("Missing next-stage values during FQE recursion")

            targets = build_fqe_targets(
                stage["__reward"].to_numpy(dtype=float),
                stage["terminal"].to_numpy(dtype=bool),
                next_values,
                gamma=self.gamma,
            )
            model = _QStageModel(
                self.observation_columns,
                learner=self.learner,
                learner_config=self.learner_config,
                seed=self.seed + time_step,
            ).fit(
                stage,
                action_indices(stage["observed_clinician_action"].astype(str)),
                targets,
            )
            self.stage_models_[time_step] = model
            q_values = model.predict_all_actions(stage)
            policy_values = np.sum(
                q_values
                * stage[[f"__pi_{index}" for index in range(len(ACTION_NAMES))]].to_numpy(),
                axis=1,
            )
            next_values_by_patient = pd.Series(
                policy_values,
                index=stage["patient_id"].astype(str),
            )
        return self

    def predict_q(self, frame: pd.DataFrame) -> np.ndarray:
        if len(self.stage_models_) != self.horizon:
            raise RuntimeError("FiniteHorizonFQE must be fit before prediction")
        output = np.full((len(frame), len(ACTION_NAMES)), np.nan, dtype=float)
        for time_step, model in self.stage_models_.items():
            mask = frame["time_step"].astype(int).to_numpy() == time_step
            if mask.any():
                output[mask] = model.predict_all_actions(frame.loc[mask])
        if not np.all(np.isfinite(output)):
            raise EvaluationDataError("Evaluation frame contains an unsupported time step")
        return output

    def predict_v(self, frame: pd.DataFrame, target_probabilities: np.ndarray) -> np.ndarray:
        probabilities = validate_probability_matrix(target_probabilities, expected_rows=len(frame))
        return np.sum(self.predict_q(frame) * probabilities, axis=1)


@dataclass(frozen=True)
class SDRResult:
    """Per-patient SDR values and importance-weight diagnostics."""

    patient_ids: tuple[str, ...]
    raw_episode_values: np.ndarray
    episode_values_by_cap: dict[float, np.ndarray]
    step_ratios: np.ndarray
    raw_cumulative_log_weights: np.ndarray
    clipped_cumulative_log_weights_by_cap: dict[float, np.ndarray]


def _safe_log(values: np.ndarray) -> np.ndarray:
    output = np.full_like(values, -np.inf, dtype=float)
    positive = values > 0.0
    output[positive] = np.log(values[positive])
    return output


def _safe_exp(log_values: np.ndarray) -> np.ndarray:
    values = np.asarray(log_values, dtype=float)
    output = np.zeros_like(values)
    maximum_log = float(np.log(np.finfo(float).max))
    representable = np.isfinite(values) & (values <= maximum_log)
    output[representable] = np.exp(np.maximum(values[representable], -745.0))
    output[np.isfinite(values) & (values > maximum_log)] = np.inf
    output[np.isposinf(values)] = np.inf
    return output


def sequential_dr_returns(
    *,
    patient_ids: Sequence[str],
    time_steps: Sequence[int],
    rewards: Sequence[float],
    logged_q_values: Sequence[float],
    current_policy_values: Sequence[float],
    next_policy_values: Sequence[float],
    terminal: Sequence[bool | int],
    target_logged_probabilities: Sequence[float],
    behavior_logged_probabilities: Sequence[float],
    horizon: int,
    gamma: float,
    caps: Sequence[float],
) -> SDRResult:
    """Compute finite-horizon SDR with cumulative ratios accumulated in log space."""

    if horizon <= 0 or not math.isfinite(gamma) or not 0 <= gamma <= 1:
        raise ValueError("SDR horizon must be positive and gamma finite in [0, 1]")
    numeric_inputs = (
        rewards,
        logged_q_values,
        current_policy_values,
        next_policy_values,
        target_logged_probabilities,
        behavior_logged_probabilities,
    )
    if any(not np.all(np.isfinite(np.asarray(values, dtype=float))) for values in numeric_inputs):
        raise ValueError("SDR inputs must be finite")
    if any(
        np.any(np.asarray(values) > 1)
        for values in (target_logged_probabilities, behavior_logged_probabilities)
    ):
        raise ValueError("SDR probabilities cannot exceed one")

    data = pd.DataFrame(
        {
            "patient_id": np.asarray(patient_ids, dtype=str),
            "time_step": np.asarray(time_steps, dtype=int),
            "reward": np.asarray(rewards, dtype=float),
            "q_logged": np.asarray(logged_q_values, dtype=float),
            "v_current": np.asarray(current_policy_values, dtype=float),
            "v_next": np.asarray(next_policy_values, dtype=float),
            "terminal": np.asarray(terminal, dtype=bool),
            "target_logged": np.asarray(target_logged_probabilities, dtype=float),
            "behavior_logged": np.asarray(behavior_logged_probabilities, dtype=float),
        }
    ).sort_values(["patient_id", "time_step"], kind="stable")
    if (data["behavior_logged"] <= 0).any():
        raise ValueError("Behavior logged-action probabilities must be positive")
    if (data["target_logged"] < 0).any():
        raise ValueError("Target logged-action probabilities must be non-negative")

    unique_patients = tuple(data["patient_id"].drop_duplicates())
    patient_count = len(unique_patients)
    ratios = np.full((patient_count, horizon), np.nan, dtype=float)
    raw_log_weights = np.full((patient_count, horizon), np.nan, dtype=float)
    raw_episode_values = np.full(patient_count, np.nan, dtype=float)
    clipped_logs = {
        float(cap): np.full((patient_count, horizon), np.nan, dtype=float) for cap in caps
    }
    episode_values = {float(cap): np.full(patient_count, np.nan, dtype=float) for cap in caps}

    for patient_index, (patient_id, episode) in enumerate(data.groupby("patient_id", sort=False)):
        steps = episode["time_step"].to_numpy(dtype=int)
        if len(episode) != horizon or not np.array_equal(steps, np.arange(horizon)):
            raise EvaluationDataError(
                f"SDR requires a complete {horizon}-step episode: {patient_id}"
            )
        step_ratios = episode["target_logged"].to_numpy(dtype=float) / episode[
            "behavior_logged"
        ].to_numpy(dtype=float)
        ratios[patient_index] = step_ratios
        raw_logs = np.cumsum(_safe_log(step_ratios))
        raw_log_weights[patient_index] = raw_logs

        rewards_array = episode["reward"].to_numpy(dtype=float)
        q_logged = episode["q_logged"].to_numpy(dtype=float)
        v_current = episode["v_current"].to_numpy(dtype=float)
        v_next = episode["v_next"].to_numpy(dtype=float)
        terminal_array = episode["terminal"].to_numpy(dtype=bool)
        residuals = rewards_array + gamma * (~terminal_array) * v_next - q_logged
        discounts = np.power(float(gamma), np.arange(horizon, dtype=float))
        with np.errstate(over="ignore", invalid="ignore"):
            raw_episode_values[patient_index] = float(
                v_current[0] + np.sum(discounts * _safe_exp(raw_logs) * residuals)
            )

        for cap in caps:
            cap_value = float(cap)
            if not math.isfinite(cap_value) or cap_value <= 0:
                raise ValueError("Every SDR cap must be finite and positive")
            # Clip the cumulative trajectory ratio at every horizon, rather
            # than clipping each step and allowing the effective bound to grow
            # to ``cap ** horizon``.  This makes the advertised cap a real
            # bound on every residual's importance weight.
            log_weights = np.minimum(raw_logs, math.log(cap_value))
            clipped_logs[cap_value][patient_index] = log_weights
            episode_values[cap_value][patient_index] = float(
                v_current[0] + np.sum(discounts * _safe_exp(log_weights) * residuals)
            )

    return SDRResult(
        patient_ids=unique_patients,
        raw_episode_values=raw_episode_values,
        episode_values_by_cap=episode_values,
        step_ratios=ratios,
        raw_cumulative_log_weights=raw_log_weights,
        clipped_cumulative_log_weights_by_cap=clipped_logs,
    )


def evaluate_sdr(
    frame: pd.DataFrame,
    *,
    rewards: np.ndarray,
    target_probabilities: np.ndarray,
    behavior_probabilities: np.ndarray,
    fqe: FiniteHorizonFQE,
    caps: Sequence[float],
) -> SDRResult:
    probabilities = validate_probability_matrix(target_probabilities, expected_rows=len(frame))
    behavior = validate_probability_matrix(behavior_probabilities, expected_rows=len(frame))
    q_values = fqe.predict_q(frame)
    policy_values = np.sum(q_values * probabilities, axis=1)
    logged_actions = frame["observed_clinician_action"].astype(str)
    logged_q = q_values[np.arange(len(frame)), action_indices(logged_actions)]

    ordered = frame.sort_values(["patient_id", "time_step"], kind="stable")
    source_keys = pd.MultiIndex.from_frame(frame[["patient_id", "time_step"]])
    ordered_keys = pd.MultiIndex.from_frame(ordered[["patient_id", "time_step"]])
    positions = source_keys.get_indexer(ordered_keys)
    aligned_v = policy_values[positions]
    next_v = (
        pd.Series(aligned_v, index=ordered.index)
        .groupby(ordered["patient_id"], sort=False)
        .shift(-1)
        .fillna(0.0)
        .to_numpy()
    )
    # Restore source row order so the public primitive performs one definitive sort.
    inverse = np.empty_like(positions)
    inverse[positions] = np.arange(len(positions))
    next_v_source_order = next_v[inverse]

    return sequential_dr_returns(
        patient_ids=frame["patient_id"].astype(str),
        time_steps=frame["time_step"].astype(int),
        rewards=rewards,
        logged_q_values=logged_q,
        current_policy_values=policy_values,
        next_policy_values=next_v_source_order,
        terminal=frame["terminal"].astype(bool),
        target_logged_probabilities=logged_action_probabilities(probabilities, logged_actions),
        behavior_logged_probabilities=logged_action_probabilities(behavior, logged_actions),
        horizon=fqe.horizon,
        gamma=fqe.gamma,
        caps=caps,
    )


def _logsumexp(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -np.inf
    maximum = float(np.max(finite))
    return maximum + float(np.log(np.exp(finite - maximum).sum()))


def effective_sample_size_from_logs(log_weights: np.ndarray) -> float:
    """Compute ESS without exponentiating potentially huge cumulative weights."""

    values = np.asarray(log_weights, dtype=float)
    if np.isposinf(values).any():
        return float(np.isposinf(values).sum())
    log_sum = _logsumexp(values)
    log_sum_squared = _logsumexp(2.0 * values)
    if not np.isfinite(log_sum) or not np.isfinite(log_sum_squared):
        return 0.0
    return float(np.exp(2.0 * log_sum - log_sum_squared))


def maximum_normalized_weight_share(log_weights: np.ndarray) -> float:
    values = np.asarray(log_weights, dtype=float)
    if np.isposinf(values).any():
        return 1.0 / float(np.isposinf(values).sum())
    log_total = _logsumexp(values)
    if not np.isfinite(log_total):
        return 0.0
    return float(np.exp(np.max(values[np.isfinite(values)]) - log_total))


def _finite_quantiles(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {key: float("nan") for key in ("p50", "p90", "p95", "p99", "max")}
    quantiles = np.quantile(finite, [0.50, 0.90, 0.95, 0.99, 1.00])
    return dict(zip(("p50", "p90", "p95", "p99", "max"), map(float, quantiles), strict=True))


def _raw_weight_horizon_summary(log_weights: np.ndarray) -> dict[str, Any]:
    values = np.asarray(log_weights, dtype=float)
    maximum_log = float(np.log(np.finfo(float).max))
    overflow = np.isposinf(values) | (values > maximum_log)
    positive_finite = values[np.isfinite(values)]
    summary: dict[str, Any] = {
        "linear_scale_valid": not bool(overflow.any()),
        "overflow_count": int(overflow.sum()),
        "zero_weight_fraction": float(np.isneginf(values).mean()),
        "positive_log_weight_quantiles": _finite_quantiles(positive_finite),
    }
    if overflow.any():
        summary["weight_quantiles"] = None
        summary["invalid_reason"] = (
            "one or more raw cumulative weights exceed finite float range; "
            "raw weights were not clipped"
        )
    else:
        summary["weight_quantiles"] = _finite_quantiles(_safe_exp(values))
    return summary


def _state_support_summary(
    target: np.ndarray,
    behavior: np.ndarray,
    *,
    thresholds: Sequence[float],
) -> dict[str, Any]:
    entropy_terms = np.zeros_like(target)
    positive = target > 0.0
    entropy_terms[positive] = target[positive] * np.log(target[positive])
    entropy = -np.sum(entropy_terms, axis=1)
    summary: dict[str, Any] = {
        "target_action_distribution": {
            action: float(target[:, index].mean()) for index, action in enumerate(ACTION_NAMES)
        },
        "mean_entropy_nats": float(entropy.mean()),
        "mean_statewise_action_overlap": float(np.minimum(target, behavior).sum(axis=1).mean()),
        "mean_target_weighted_behavior_probability": float(
            np.sum(target * behavior, axis=1).mean()
        ),
    }
    summary["target_mass_below_behavior_propensity"] = {
        str(float(threshold)): float(np.sum(target * (behavior < float(threshold)), axis=1).mean())
        for threshold in thresholds
    }
    return summary


def support_diagnostics(
    frame: pd.DataFrame,
    *,
    target_probabilities: np.ndarray,
    behavior_probabilities: np.ndarray,
    sdr: SDRResult,
    thresholds: Sequence[float] = (0.01, 0.05),
    map_threshold: float = 65.0,
    direct_policy_values: np.ndarray | None = None,
) -> dict[str, Any]:
    """Summarize action overlap, ratios, concentration, ESS, and MAP<65 support."""

    target = validate_probability_matrix(target_probabilities, expected_rows=len(frame))
    behavior = validate_probability_matrix(behavior_probabilities, expected_rows=len(frame))
    result = _state_support_summary(target, behavior, thresholds=thresholds)
    result["logged_action_ratio_quantiles_raw"] = _finite_quantiles(sdr.step_ratios)
    result["raw_sdr_episode_return_valid"] = bool(np.all(np.isfinite(sdr.raw_episode_values)))
    result["raw_sdr_episode_return_finite_fraction"] = float(
        np.isfinite(sdr.raw_episode_values).mean()
    )
    result["cumulative_raw_ess_by_horizon"] = [
        effective_sample_size_from_logs(sdr.raw_cumulative_log_weights[:, index])
        for index in range(sdr.raw_cumulative_log_weights.shape[1])
    ]
    result["cumulative_raw_weight_quantiles_by_horizon"] = [
        _raw_weight_horizon_summary(sdr.raw_cumulative_log_weights[:, index])
        for index in range(sdr.raw_cumulative_log_weights.shape[1])
    ]
    result["maximum_normalized_raw_episode_weight_share_by_horizon"] = [
        maximum_normalized_weight_share(sdr.raw_cumulative_log_weights[:, index])
        for index in range(sdr.raw_cumulative_log_weights.shape[1])
    ]
    result["by_cap"] = {}
    for cap, log_weights in sdr.clipped_cumulative_log_weights_by_cap.items():
        clipped_step_ratios = np.minimum(sdr.step_ratios, cap)
        result["by_cap"][str(cap)] = {
            "step_ratio_quantiles": _finite_quantiles(clipped_step_ratios),
            "cumulative_ess_by_horizon": [
                effective_sample_size_from_logs(log_weights[:, index])
                for index in range(log_weights.shape[1])
            ],
            "maximum_normalized_episode_weight_share_by_horizon": [
                maximum_normalized_weight_share(log_weights[:, index])
                for index in range(log_weights.shape[1])
            ],
        }

    map_mask = frame["map_mm_hg"].to_numpy(dtype=float) < float(map_threshold)
    if map_mask.any():
        result["map_below_65_slice"] = {
            "rows": int(map_mask.sum()),
            "patients": int(frame.loc[map_mask, "patient_id"].nunique()),
            "fraction": float(map_mask.mean()),
            **_state_support_summary(
                target[map_mask],
                behavior[map_mask],
                thresholds=thresholds,
            ),
            "logged_action_ratio_quantiles_raw": _finite_quantiles(
                logged_action_probabilities(
                    target[map_mask],
                    frame.loc[map_mask, "observed_clinician_action"],
                )
                / logged_action_probabilities(
                    behavior[map_mask], frame.loc[map_mask, "observed_clinician_action"]
                )
            ),
        }
        if direct_policy_values is not None:
            direct_values = np.asarray(direct_policy_values, dtype=float)
            if direct_values.shape != (len(frame),):
                raise ValueError("direct_policy_values must have one value per state")
            result["map_below_65_slice"]["model_based_mean_remaining_fqe_value"] = float(
                direct_values[map_mask].mean()
            )
            result["map_below_65_slice"]["fqe_value_label"] = (
                "model-based mean remaining-horizon value at MAP<65 states"
            )
    else:
        result["map_below_65_slice"] = {"rows": 0, "patients": 0, "fraction": 0.0}
    return result


@dataclass(frozen=True)
class BootstrapSummary:
    point_estimate: float
    confidence_interval: tuple[float, float]
    samples: np.ndarray


def patient_level_paired_bootstrap(
    values_by_policy: Mapping[str, Sequence[float]],
    *,
    baseline: str,
    resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 41,
) -> tuple[dict[str, BootstrapSummary], dict[str, BootstrapSummary], np.ndarray]:
    """Bootstrap patients once and reuse draws for every policy and difference."""

    if baseline not in values_by_policy:
        raise KeyError(f"Missing baseline policy: {baseline}")
    arrays = {name: np.asarray(values, dtype=float) for name, values in values_by_policy.items()}
    lengths = {len(values) for values in arrays.values()}
    if len(lengths) != 1:
        raise ValueError("Paired bootstrap inputs must contain the same patients in the same order")
    patient_count = lengths.pop()
    if patient_count == 0 or resamples <= 0:
        raise ValueError("Bootstrap requires patients and positive resamples")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    if any(not np.all(np.isfinite(values)) for values in arrays.values()):
        raise ValueError("Bootstrap values must be finite")

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, patient_count, size=(int(resamples), patient_count))
    lower_probability = (1.0 - confidence) / 2.0
    upper_probability = 1.0 - lower_probability

    def summarize(values: np.ndarray) -> BootstrapSummary:
        samples = values[indices].mean(axis=1)
        interval = np.quantile(samples, [lower_probability, upper_probability])
        return BootstrapSummary(
            point_estimate=float(values.mean()),
            confidence_interval=(float(interval[0]), float(interval[1])),
            samples=samples,
        )

    estimates = {name: summarize(values) for name, values in arrays.items()}
    baseline_values = arrays[baseline]
    differences = {name: summarize(values - baseline_values) for name, values in arrays.items()}
    return estimates, differences, indices
