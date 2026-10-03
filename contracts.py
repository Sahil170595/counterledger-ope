"""Machine-readable shared contracts for the Counterledger experiment."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ACTION_NAMES: tuple[str, ...] = (
    "maintain",
    "iv_fluids",
    "escalate_vasopressor",
)

POLICY_NAMES: tuple[str, ...] = (
    "always_maintain",
    "behavior_clone",
    "llm_zero_shot",
    "llm_improved",
)

PREDICTION_COLUMNS: tuple[str, ...] = (
    "patient_id",
    "time_step",
    "chosen_action",
    "prob_maintain",
    "prob_iv_fluids",
    "prob_escalate_vasopressor",
    "model_id",
    "prompt_version",
)

PROBABILITY_TOLERANCE = 1e-8


def validate_policy_columns(columns: Sequence[str]) -> tuple[str, ...]:
    """One pre-action allowlist boundary, separate from nuisance feature scope."""

    allowed = tuple(columns)
    forbidden = {
        "patient_id",
        "time_step",
        "split",
        "hospital_id",
        "provider_id",
        "observed_clinician_action",
        "terminal",
        "icu_mortality",
        "hospital_mortality",
        "handoff_note",
    }
    unsafe = {
        column
        for column in allowed
        if column in forbidden or column.startswith(("next_", "adverse_"))
    }
    if unsafe:
        raise ValueError(f"Forbidden policy inputs requested: {sorted(unsafe)}")
    if not allowed or len(set(allowed)) != len(allowed):
        raise ValueError("Policy columns must be non-empty and unique")
    return allowed


def portable_file_sha256(path: str | Path) -> str:
    """Hash text CSVs using LF canonicalization; other files are byte-exact."""

    resolved = Path(path)
    payload = resolved.read_bytes()
    if resolved.suffix.lower() == ".csv":
        payload = payload.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class PolicyRunDiagnostics:
    """Policy-side diagnostics consumed by the shared evaluator."""

    requests: int = 0
    fallbacks: int = 0
    malformed_outputs: int = 0
    cache_hits: int = 0
    cache_misses: int = 0

    @property
    def fallback_rate(self) -> float:
        return self.fallbacks / self.requests if self.requests else 0.0


def validate_probability_matrix(
    values: Sequence[Sequence[float]] | np.ndarray,
    *,
    expected_rows: int | None = None,
) -> np.ndarray:
    """Validate and return an immutable-shape policy probability matrix."""

    probabilities = np.asarray(values, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(ACTION_NAMES):
        raise ValueError(f"Expected an N x {len(ACTION_NAMES)} probability matrix")
    if expected_rows is not None and probabilities.shape[0] != expected_rows:
        raise ValueError(f"Expected {expected_rows} probability rows, got {probabilities.shape[0]}")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Policy probabilities must be finite")
    if np.any(probabilities < 0):
        raise ValueError("Policy probabilities must be non-negative")
    row_sums = probabilities.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=PROBABILITY_TOLERANCE, rtol=0.0):
        raise ValueError("Every policy probability row must sum to one")
    return probabilities
