"""Fresh educational trajectories. No imported data or model-generated output."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from contracts import ACTION_NAMES, validate_probability_matrix

OBSERVATION_COLUMNS = ("map_mm_hg", "lactate_mmol_l", "sex", "previous_action")
HORIZON = 3
DEFAULT_SEED = 41
SPLIT_PATIENTS = {"train": 60, "validation": 24, "test": 12}


def generate_data(*, seed: int = DEFAULT_SEED) -> dict[str, pd.DataFrame]:
    """Generate factual transitions from an intentionally simplified mechanism."""

    rng = np.random.default_rng(seed)
    output = {}
    for split, patients in SPLIT_PATIENTS.items():
        rows = []
        for patient in range(patients):
            previous = "maintain"
            pressure = float(rng.uniform(55, 95))
            lactate = float(rng.uniform(1, 5))
            sex = "F" if rng.random() < 0.5 else "M"
            for step in range(HORIZON):
                # Each episode contains all actions, with a random ordering.
                # This artificial coverage is disclosed; it is not real behavior data.
                if step == 0:
                    action_order = rng.permutation(len(ACTION_NAMES))
                action_index = int(action_order[step])
                action = ACTION_NAMES[action_index]
                delta = float(rng.normal((0.0, 1.5, 2.5)[action_index], 2))
                lactate_delta = float(rng.normal(-0.1 * action_index, 0.25))
                rows.append(
                    {
                        "patient_id": f"SYN-{split.upper()}-{patient:03d}",
                        "time_step": step,
                        "split": split,
                        "hospital_id": f"SYN-SITE-{patient % 3}",
                        "provider_id": f"SYN-OPERATOR-{patient % 4}",
                        "map_mm_hg": pressure,
                        "lactate_mmol_l": lactate,
                        "sex": sex,
                        "previous_action": previous,
                        "observed_clinician_action": action,
                        "terminal": int(step == HORIZON - 1),
                        "next_6h_map_delta": delta,
                        "next_6h_lactate_delta": lactate_delta,
                        "next_6h_deterioration": int(rng.random() < 0.08),
                        "adverse_hypotension_next_6h": int(rng.random() < 0.06),
                        "adverse_fluid_overload_next_6h": int(
                            action_index == 1 and rng.random() < 0.12
                        ),
                        "adverse_tachyarrhythmia_next_6h": int(
                            action_index == 2 and rng.random() < 0.14
                        ),
                    }
                )
                previous = action
                pressure = float(np.clip(pressure + delta, 45, 105))
                lactate = float(np.clip(lactate + lactate_delta, 0.2, 8))
        frame = pd.DataFrame(rows)
        if split == "test":
            frame = frame[["patient_id", "time_step", "split", *OBSERVATION_COLUMNS]]
        output[split] = frame
    return output


class SyntheticProbabilityPolicy:
    """Deterministic analytic fixture, NOT an LLM or a frozen model-output cache."""

    model_id = "deterministic-synthetic-fixture"
    prompt_version = "no-model-executed"
    policy_metadata = {"origin": "deterministic_synthetic_not_llm"}

    def predict_proba(self, observations: Sequence[Mapping[str, Any]]) -> np.ndarray:
        logits = []
        for observation in observations:
            state = observation["structured"]
            pressure = float(state["map_mm_hg"])
            lactate = float(state["lactate_mmol_l"])
            logits.append([0.5, (70 - pressure) / 15, (lactate - 3) / 2])
        values = np.asarray(logits, dtype=float).reshape(-1, len(ACTION_NAMES))
        values -= values.max(axis=1, keepdims=True)
        values = np.exp(values)
        values /= values.sum(axis=1, keepdims=True)
        return validate_probability_matrix(values, expected_rows=len(observations))
