"""Training-only policy-improvement utilities.

The value teacher may use ``time_step`` while fitting finite-horizon FQE, but the
deployed policy may not.  This module closes that boundary explicitly: it distils
cross-fitted, row-centred teacher advantages into a model that accepts only the
policy observation allow-list.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from contracts import ACTION_NAMES, validate_probability_matrix
from policy import PolicyFeaturePreprocessor


def centered_action_advantages(
    action_values: Sequence[Sequence[float]] | np.ndarray,
    reference_probabilities: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    """Remove each state's reference-policy value from every action value."""

    values = np.asarray(action_values, dtype=float)
    probabilities = validate_probability_matrix(reference_probabilities)
    if values.shape != probabilities.shape or not np.all(np.isfinite(values)):
        raise ValueError("Action values and probabilities must be finite aligned N x 3 matrices")
    baseline = np.sum(values * probabilities, axis=1, keepdims=True)
    return values - baseline


def robust_advantage_scale(
    advantages: Sequence[Sequence[float]] | np.ndarray,
    *,
    floor: float = 0.05,
) -> float:
    """Return a robust, strictly positive scale fitted on development data only."""

    values = np.asarray(advantages, dtype=float).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Advantages must be a non-empty finite matrix")
    if not np.isfinite(floor) or floor <= 0:
        raise ValueError("Scale floor must be finite and positive")
    median = float(np.median(values))
    mad_scale = 1.4826 * float(np.median(np.abs(values - median)))
    if not np.isfinite(mad_scale) or mad_scale < floor:
        standard_deviation = float(np.std(values))
        mad_scale = standard_deviation if np.isfinite(standard_deviation) else 0.0
    return max(float(floor), mad_scale)


class AdvantageDistiller:
    """Time-agnostic Ridge adapter over the exact policy observation fields."""

    def __init__(
        self,
        observation_columns: Sequence[str],
        *,
        alpha: float = 10.0,
    ) -> None:
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("Ridge alpha must be finite and positive")
        self.observation_columns = tuple(observation_columns)
        self.alpha = float(alpha)
        self.preprocessor = PolicyFeaturePreprocessor(self.observation_columns)
        self.regressor = Ridge(alpha=self.alpha)
        self.fit_patient_ids_: tuple[str, ...] = ()
        self.fit_patient_checksum_: str | None = None
        self.fit_row_count_: int = 0
        self.is_fitted = False

    def fit(
        self,
        frame: pd.DataFrame,
        centered_advantages: Sequence[Sequence[float]] | np.ndarray,
    ) -> AdvantageDistiller:
        if "patient_id" not in frame:
            raise ValueError("Distillation data must include patient_id for fit provenance")
        targets = np.asarray(centered_advantages, dtype=float)
        if targets.shape != (len(frame), len(ACTION_NAMES)):
            raise ValueError("Distillation targets must have shape [rows, 3]")
        if not np.all(np.isfinite(targets)):
            raise ValueError("Distillation targets must be finite")
        features = self.preprocessor.fit(frame).transform(frame)
        self.regressor.fit(features, targets)
        self.fit_patient_ids_ = tuple(sorted(frame["patient_id"].astype(str).unique()))
        self.fit_patient_checksum_ = hashlib.sha256(
            "\n".join(self.fit_patient_ids_).encode()
        ).hexdigest()
        self.fit_row_count_ = len(frame)
        self.is_fitted = True
        return self

    def predict_action_values(
        self,
        observations: Sequence[Mapping[str, Any]] | pd.DataFrame,
    ) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("AdvantageDistiller must be fit before prediction")
        features = self.preprocessor.transform(observations)
        output = np.asarray(self.regressor.predict(features), dtype=float)
        if output.shape != (len(features), len(ACTION_NAMES)) or not np.all(np.isfinite(output)):
            raise RuntimeError("Distiller produced an invalid action-value matrix")
        return output


class ObservationSupportAdapter:
    """Adapt a tabular behavior model to the policy observation interface."""

    def __init__(self, model: Any, observation_columns: Sequence[str]) -> None:
        self.model = model
        self.observation_columns = tuple(observation_columns)

    @property
    def fit_patient_ids_(self) -> tuple[str, ...]:
        return tuple(getattr(self.model, "fit_patient_ids_", ()))

    @property
    def fit_patient_checksum_(self) -> str | None:
        value = getattr(self.model, "fit_patient_checksum_", None)
        return None if value is None else str(value)

    @property
    def fit_row_count_(self) -> int:
        return int(getattr(self.model, "fit_row_count_", 0))

    def _frame(
        self,
        observations: Sequence[Mapping[str, Any]] | pd.DataFrame,
    ) -> pd.DataFrame:
        if isinstance(observations, pd.DataFrame):
            return observations.loc[:, list(self.observation_columns)].copy()
        records: list[dict[str, Any]] = []
        allowed = set(self.observation_columns)
        for observation in observations:
            if set(observation) - {"structured", "handoff_note"}:
                raise ValueError("Support-model observation has unsupported fields")
            structured = observation.get("structured")
            if not isinstance(structured, Mapping) or set(structured) != allowed:
                raise ValueError("Support-model structured fields do not match the manifest")
            records.append({column: structured[column] for column in self.observation_columns})
        return pd.DataFrame.from_records(records, columns=list(self.observation_columns))

    def predict_proba(
        self,
        observations: Sequence[Mapping[str, Any]] | pd.DataFrame,
    ) -> np.ndarray:
        frame = self._frame(observations)
        probabilities = self.model.predict_proba(frame)
        return validate_probability_matrix(probabilities, expected_rows=len(frame))
