"""Logged-transition-only environment for the synthetic ICU trajectories."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from contracts import ACTION_NAMES, validate_policy_columns
from reward import REQUIRED_OUTCOME_COLUMNS, compute_reward

POST_ACTION_COLUMNS = frozenset(
    {
        "observed_clinician_action",
        *REQUIRED_OUTCOME_COLUMNS,
        "terminal",
        "icu_mortality",
        "hospital_mortality",
    }
)


def load_observation_columns(path: str | Path = "observation_columns.json") -> list[str]:
    columns = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        not isinstance(columns, list)
        or not columns
        or not all(isinstance(column, str) for column in columns)
    ):
        raise ValueError("Observation-column manifest must be a non-empty string list")
    if len(columns) != len(set(columns)):
        raise ValueError("Observation-column manifest contains duplicates")
    leakage = sorted(set(columns) & POST_ACTION_COLUMNS)
    if leakage:
        raise ValueError(f"Post-action columns are forbidden observations: {leakage}")
    validate_policy_columns(columns)
    return columns


def _python_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


class OfflineClinicalEnv:
    """Replay factual logged transitions without inventing counterfactual outcomes."""

    def __init__(
        self,
        trajectories: pd.DataFrame,
        observation_columns: Sequence[str],
        *,
        reward_fn: Callable[[Mapping[str, Any]], float] = compute_reward,
        note_column: str = "handoff_note",
    ) -> None:
        self.observation_columns = list(validate_policy_columns(observation_columns))
        self.reward_fn = reward_fn
        self.note_column = note_column
        self._validate(trajectories)
        self.trajectories = trajectories.sort_values(
            ["patient_id", "time_step"], kind="stable"
        ).reset_index(drop=True)
        self._episodes = {
            str(patient_id): episode.reset_index(drop=False)
            for patient_id, episode in self.trajectories.groupby(
                "patient_id", sort=True, observed=True
            )
        }
        self._patient_ids = tuple(self._episodes)
        self._episode: pd.DataFrame | None = None
        self._patient_id: str | None = None
        self._cursor = 0

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        observation_columns: Sequence[str],
        **kwargs: Any,
    ) -> OfflineClinicalEnv:
        return cls(pd.read_csv(path), observation_columns, **kwargs)

    def _validate(self, trajectories: pd.DataFrame) -> None:
        if not isinstance(trajectories, pd.DataFrame) or trajectories.empty:
            raise ValueError("Trajectories must be a non-empty pandas DataFrame")
        leakage = sorted(set(self.observation_columns) & POST_ACTION_COLUMNS)
        if leakage:
            raise ValueError(f"Post-action columns are forbidden observations: {leakage}")

        required = {
            "patient_id",
            "time_step",
            "observed_clinician_action",
            "terminal",
            *self.observation_columns,
            *REQUIRED_OUTCOME_COLUMNS,
        }
        missing = sorted(required - set(trajectories.columns))
        if missing:
            raise ValueError(f"Trajectory table is missing required columns: {missing}")
        if trajectories[["patient_id", "time_step"]].duplicated().any():
            raise ValueError("Duplicate patient/time transitions are not allowed")
        if trajectories["patient_id"].isna().any():
            raise ValueError("patient_id must never be missing")

        numeric_steps = pd.to_numeric(trajectories["time_step"], errors="coerce")
        if numeric_steps.isna().any() or not np.all(
            numeric_steps.to_numpy(dtype=float) == numeric_steps.to_numpy(dtype=int)
        ):
            raise ValueError("time_step must contain exact integers")

        actions = trajectories["observed_clinician_action"]
        unknown_actions = sorted(set(actions.dropna().astype(str)) - set(ACTION_NAMES))
        if actions.isna().any() or unknown_actions:
            raise ValueError(f"Unknown or missing logged actions: {unknown_actions}")

        continuous_outcomes = ("next_6h_map_delta", "next_6h_lactate_delta")
        binary_outcomes = tuple(set(REQUIRED_OUTCOME_COLUMNS) - set(continuous_outcomes))
        for column in continuous_outcomes:
            values = pd.to_numeric(trajectories[column], errors="coerce").to_numpy(dtype=float)
            if not np.all(np.isfinite(values)):
                raise ValueError(f"Outcome column {column!r} must be finite")
        for column in binary_outcomes:
            values = pd.to_numeric(trajectories[column], errors="coerce").to_numpy(dtype=float)
            if not np.all(np.isin(values, (0.0, 1.0))):
                raise ValueError(f"Outcome column {column!r} must be finite and binary")

        for patient_id, raw_episode in trajectories.groupby(
            "patient_id", sort=False, observed=True
        ):
            episode = raw_episode.sort_values("time_step", kind="stable")
            steps = episode["time_step"].to_numpy(dtype=int)
            expected_steps = np.arange(len(episode), dtype=int)
            if not np.array_equal(steps, expected_steps):
                raise ValueError(
                    f"Episode {patient_id!r} time steps must be exactly 0..{len(episode) - 1}"
                )
            terminal = episode["terminal"].to_numpy(dtype=int)
            if not np.all(np.isin(terminal, (0, 1))):
                raise ValueError(f"Episode {patient_id!r} has non-binary terminal flags")
            expected = np.zeros(len(episode), dtype=int)
            expected[-1] = 1
            if not np.array_equal(terminal, expected):
                raise ValueError(f"Episode {patient_id!r} must terminate exactly on its final row")
            if "previous_action" in episode.columns:
                previous = episode["previous_action"].astype(str).to_numpy()
                logged = episode["observed_clinician_action"].astype(str).to_numpy()
                if previous[0] != "maintain" or not np.array_equal(previous[1:], logged[:-1]):
                    raise ValueError(f"Episode {patient_id!r} has inconsistent action history")

    def _observation(self, row: pd.Series) -> dict[str, Any]:
        structured = {column: _python_value(row[column]) for column in self.observation_columns}
        note = _python_value(row[self.note_column]) if self.note_column in row.index else None
        return {"structured": structured, "handoff_note": note}

    def reset(
        self, patient_id: str | None = None, seed: int | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Select an episode reproducibly and return its first observation."""

        if patient_id is None:
            rng = np.random.default_rng(seed)
            patient_id = self._patient_ids[int(rng.integers(len(self._patient_ids)))]
        patient_key = str(patient_id)
        if patient_key not in self._episodes:
            raise KeyError(f"Unknown patient_id: {patient_id}")
        self._patient_id = patient_key
        self._episode = self._episodes[patient_key]
        self._cursor = 0
        first = self._episode.iloc[0]
        metadata = {
            "patient_id": patient_key,
            "episode_length": len(self._episode),
            "has_handoff_note": self.note_column in self._episode.columns,
        }
        return self._observation(first), metadata

    def step_logged(
        self,
    ) -> tuple[
        dict[str, Any],
        str,
        float,
        dict[str, Any] | None,
        bool,
        dict[str, Any],
    ]:
        """Advance one factual logged transition.

        There is deliberately no ``step(action)`` method. The dataset contains
        no factual outcome for an action different from the logged action.
        """

        if self._episode is None or self._patient_id is None:
            raise RuntimeError("Call reset() before step_logged()")
        if self._cursor >= len(self._episode):
            raise RuntimeError("Episode has terminated; call reset()")

        row = self._episode.iloc[self._cursor]
        observation = self._observation(row)
        action = str(row["observed_clinician_action"])
        reward = self.reward_fn(row)
        terminated = bool(row["terminal"])
        next_observation = None
        if not terminated:
            next_observation = self._observation(self._episode.iloc[self._cursor + 1])

        info = {
            "patient_id": self._patient_id,
            "time_step": int(row["time_step"]),
            "source_row_index": int(row["index"]),
            "factual_logged_transition": True,
        }
        self._cursor += 1
        return observation, action, reward, next_observation, terminated, info
