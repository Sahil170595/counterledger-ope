"""Bounded reward specifications for logged six-hour outcomes.

The reward is an engineering objective for synthetic data, not a clinical
utility function. Continuous terms use tanh transforms so a single extreme
measurement cannot dominate the binary safety outcomes.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from math import isfinite, tanh
from typing import Any

REQUIRED_OUTCOME_COLUMNS = (
    "next_6h_map_delta",
    "next_6h_lactate_delta",
    "next_6h_deterioration",
    "adverse_hypotension_next_6h",
    "adverse_fluid_overload_next_6h",
    "adverse_tachyarrhythmia_next_6h",
)


@dataclass(frozen=True)
class RewardConfig:
    """Weights and scales for the synthetic short-horizon reward."""

    map_weight: float = 0.45
    map_scale: float = 5.0
    lactate_weight: float = 0.25
    lactate_scale: float = 1.0
    deterioration_penalty: float = 0.9
    hypotension_penalty: float = 0.8
    fluid_overload_penalty: float = 0.7
    tachyarrhythmia_penalty: float = 0.7

    def __post_init__(self) -> None:
        values = {field.name: float(getattr(self, field.name)) for field in fields(self)}
        if not all(isfinite(value) for value in values.values()):
            raise ValueError("Reward configuration must contain only finite values")
        if self.map_scale <= 0 or self.lactate_scale <= 0:
            raise ValueError("Continuous reward scales must be positive")
        penalty_names = (
            "deterioration_penalty",
            "hypotension_penalty",
            "fluid_overload_penalty",
            "tachyarrhythmia_penalty",
        )
        if any(getattr(self, name) < 0 for name in penalty_names):
            raise ValueError("Reward penalties must be non-negative")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> RewardConfig:
        return cls(**{field.name: values[field.name] for field in fields(cls)})


PRIMARY_REWARD_CONFIG = RewardConfig()

ALTERNATIVE_REWARD_CONFIG = RewardConfig(
    map_weight=0.0,
    lactate_weight=0.25,
    deterioration_penalty=1.20,
    hypotension_penalty=1.20,
    fluid_overload_penalty=1.20,
    tachyarrhythmia_penalty=1.20,
)


def _finite_float(row: Mapping[str, Any], name: str) -> float:
    if name not in row:
        raise KeyError(f"Missing reward input: {name}")
    value = float(row[name])
    if not isfinite(value):
        raise ValueError(f"Reward input {name!r} must be finite")
    return value


def _binary(row: Mapping[str, Any], name: str) -> float:
    value = _finite_float(row, name)
    if value not in (0.0, 1.0):
        raise ValueError(f"Reward input {name!r} must be binary, got {value}")
    return value


def compute_reward(
    row: Mapping[str, Any],
    config: RewardConfig = PRIMARY_REWARD_CONFIG,
) -> float:
    """Compute the bounded scalar reward for one factual logged outcome.

    Positive MAP change and falling lactate are rewarded. Deterioration and
    each adverse event are penalized. This function must never be evaluated on
    a counterfactual action using a factual outcome from another action.
    """

    map_delta = _finite_float(row, "next_6h_map_delta")
    lactate_delta = _finite_float(row, "next_6h_lactate_delta")

    reward = config.map_weight * tanh(map_delta / config.map_scale)
    reward -= config.lactate_weight * tanh(lactate_delta / config.lactate_scale)
    reward -= config.deterioration_penalty * _binary(row, "next_6h_deterioration")
    reward -= config.hypotension_penalty * _binary(row, "adverse_hypotension_next_6h")
    reward -= config.fluid_overload_penalty * _binary(row, "adverse_fluid_overload_next_6h")
    reward -= config.tachyarrhythmia_penalty * _binary(row, "adverse_tachyarrhythmia_next_6h")

    if not isfinite(reward):
        raise ValueError("Computed reward is not finite")
    return float(reward)
