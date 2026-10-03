"""Leakage-resistant policies for the four-way Counterledger comparison."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Any, Protocol

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from contracts import (
    ACTION_NAMES,
    PolicyRunDiagnostics,
    validate_policy_columns,
    validate_probability_matrix,
)
from llm_client import (
    CHOICE_TOKENS,
    ChoiceLogitScorer,
    JsonlTokenProbabilityCache,
    LLMClient,
    TokenProbabilityRecord,
    token_probability_cache_key,
)
from prompts import (
    CHOICE_PROMPT_VERSION,
    POLICY_OUTPUT_SCHEMA,
    PROMPT_VERSION,
    MissingHandoffNoteError,
    build_choice_messages,
    build_policy_messages,
)

DEFAULT_POLICY_SPLIT_SEED = 41
CATEGORICAL_OBSERVATION_COLUMNS = frozenset({"sex", "previous_action"})
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
SELECTED_VARIANT_NAMES = (
    "zero_shot",
    "temperature_only",
    "support_only",
    "value_only",
    "value_plus_support",
    "behavior_blend_25",
    "behavior_blend_50",
    "behavior_blend_75",
)
BEHAVIOR_BLEND_LLM_WEIGHTS = {
    "behavior_blend_25": 0.25,
    "behavior_blend_50": 0.50,
    "behavior_blend_75": 0.75,
}
_EXPLICIT_SELECTION_FIELDS = frozenset(
    {
        "candidate_id",
        "selected_variant",
        "scorer_semantic_fingerprint",
        "tournament_receipt",
        "selected_model_receipt",
    }
)
_SELECTED_PARAMETER_FIELDS = frozenset({"base_temperature", "advantage_weight", "support_weight"})


@dataclass(frozen=True)
class ReceiptReference:
    """Immutable path/digest reference to a promotion authority artifact."""

    path: str
    sha256: str

    def as_dict(self) -> dict[str, str]:
        return {"path": self.path, "sha256": self.sha256}


@dataclass(frozen=True)
class ExplicitModelSelection:
    """Strict model/variant promotion contract used by production construction."""

    candidate_id: str
    selected_variant: str
    scorer_semantic_fingerprint: str
    tournament_receipt: ReceiptReference
    selected_model_receipt: ReceiptReference
    selected_parameters: dict[str, float]

    @property
    def applied_parameters(self) -> dict[str, float]:
        return selected_variant_parameters(
            self.selected_variant,
            self.selected_parameters,
        )

    @property
    def receipt_metadata(self) -> dict[str, dict[str, str]]:
        return {
            "tournament": self.tournament_receipt.as_dict(),
            "selected_model": self.selected_model_receipt.as_dict(),
        }


def _receipt_reference(value: Any, *, field: str) -> ReceiptReference:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"llm.{field} must contain exactly path and sha256")
    path = value["path"]
    digest = value["sha256"]
    if not isinstance(path, str) or not path.strip() or path != path.strip():
        raise ValueError(f"llm.{field}.path must be a non-empty trimmed string")
    if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
        raise ValueError(f"llm.{field}.sha256 must be a lowercase SHA-256 digest")
    return ReceiptReference(path=path, sha256=digest)


def selected_variant_parameters(
    selected_variant: str,
    selected_parameters: Mapping[str, Any],
) -> dict[str, float]:
    """Map a selected ablation to the exact tournament transform parameters."""

    if selected_variant not in SELECTED_VARIANT_NAMES:
        raise ValueError("selected_variant must be one of " + ", ".join(SELECTED_VARIANT_NAMES))
    if not isinstance(selected_parameters, Mapping) or set(selected_parameters) != (
        _SELECTED_PARAMETER_FIELDS
    ):
        raise ValueError(
            "improvement.selected_parameters must contain exactly "
            "base_temperature, advantage_weight, and support_weight"
        )
    temperature = float(selected_parameters["base_temperature"])
    advantage = float(selected_parameters["advantage_weight"])
    support = float(selected_parameters["support_weight"])
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Selected base_temperature must be finite and positive")
    if not np.isfinite(advantage) or advantage < 0:
        raise ValueError("Selected advantage_weight must be finite and non-negative")
    if not np.isfinite(support) or support < 0:
        raise ValueError("Selected support_weight must be finite and non-negative")
    mappings = {
        "zero_shot": (1.0, 0.0, 0.0),
        "temperature_only": (temperature, 0.0, 0.0),
        "support_only": (1.0, 0.0, support),
        "value_only": (1.0, advantage, 0.0),
        "value_plus_support": (temperature, advantage, support),
        "behavior_blend_25": (temperature, advantage, support),
        "behavior_blend_50": (temperature, advantage, support),
        "behavior_blend_75": (temperature, advantage, support),
    }
    mapped_temperature, mapped_advantage, mapped_support = mappings[selected_variant]
    return {
        "base_temperature": float(mapped_temperature),
        "advantage_weight": float(mapped_advantage),
        "support_weight": float(mapped_support),
    }


def parse_explicit_model_selection(
    config: Mapping[str, Any],
) -> ExplicitModelSelection | None:
    """Return the strict promotion contract, or ``None`` for untouched legacy config."""

    llm = config.get("llm", {})
    improvement = config.get("improvement", {})
    if not isinstance(llm, Mapping):
        raise ValueError("config.llm must be an object")
    if not isinstance(improvement, Mapping):
        raise ValueError("config.improvement must be an object")
    present = _EXPLICIT_SELECTION_FIELDS.intersection(llm)
    has_parameters = "selected_parameters" in improvement
    if not present and not has_parameters:
        return None
    if present != _EXPLICIT_SELECTION_FIELDS:
        missing = sorted(_EXPLICIT_SELECTION_FIELDS - present)
        raise ValueError(
            "Explicit model selection is incomplete; missing llm fields: " + ", ".join(missing)
        )
    if not has_parameters:
        raise ValueError("Explicit model selection requires improvement.selected_parameters")
    candidate_id = llm["candidate_id"]
    selected_variant = llm["selected_variant"]
    fingerprint = llm["scorer_semantic_fingerprint"]
    if (
        not isinstance(candidate_id, str)
        or not candidate_id.strip()
        or candidate_id != candidate_id.strip()
    ):
        raise ValueError("llm.candidate_id must be a non-empty trimmed string")
    if not isinstance(selected_variant, str):
        raise ValueError("llm.selected_variant must be a string")
    if not isinstance(fingerprint, str) or _SHA256_PATTERN.fullmatch(fingerprint) is None:
        raise ValueError("llm.scorer_semantic_fingerprint must be a lowercase SHA-256 digest")
    raw_parameters = improvement["selected_parameters"]
    applied = selected_variant_parameters(selected_variant, raw_parameters)
    # Preserve the tournament-selected knobs separately from the applied ablation
    # mapping.  This lets receipts prove both which row won and how it is served.
    selected = {key: float(raw_parameters[key]) for key in sorted(_SELECTED_PARAMETER_FIELDS)}
    if any(not np.isfinite(value) for value in selected.values()):
        raise ValueError("improvement.selected_parameters must contain finite numbers")
    _ = applied
    return ExplicitModelSelection(
        candidate_id=candidate_id,
        selected_variant=selected_variant,
        scorer_semantic_fingerprint=fingerprint,
        tournament_receipt=_receipt_reference(
            llm["tournament_receipt"], field="tournament_receipt"
        ),
        selected_model_receipt=_receipt_reference(
            llm["selected_model_receipt"], field="selected_model_receipt"
        ),
        selected_parameters=selected,
    )


class Policy:
    def predict_proba(self, observations: Sequence[Mapping[str, Any]]) -> np.ndarray:
        """Return an ``[N, 3]`` array in canonical action order."""

        raise NotImplementedError


class AlwaysMaintainPolicy(Policy):
    def predict_proba(self, observations: Sequence[Mapping[str, Any]]) -> np.ndarray:
        output = np.zeros((len(observations), len(ACTION_NAMES)), dtype=float)
        output[:, 0] = 1.0
        return output


def deterministic_policy_development_ids(
    patient_ids: Sequence[str],
    *,
    fraction: float = 0.60,
    seed: int = DEFAULT_POLICY_SPLIT_SEED,
) -> frozenset[str]:
    """Select a stable patient-disjoint development role by salted SHA-256 rank."""

    if not 0 < fraction < 1:
        raise ValueError("fraction must be strictly between zero and one")
    unique_ids = sorted({str(patient_id) for patient_id in patient_ids})
    if len(unique_ids) < 2:
        raise ValueError("At least two unique patients are required for a role split")
    ranked = sorted(
        unique_ids,
        key=lambda patient_id: (
            hashlib.sha256(f"{seed}:{patient_id}".encode()).digest(),
            patient_id,
        ),
    )
    count = min(len(ranked) - 1, max(1, int(round(fraction * len(ranked)))))
    return frozenset(ranked[:count])


def _python_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def frame_to_observations(
    frame: pd.DataFrame,
    observation_columns: Sequence[str],
    *,
    note_column: str = "handoff_note",
) -> list[dict[str, Any]]:
    """Convert a table to strict nested policy observations in source row order."""

    validate_policy_columns(observation_columns)

    missing = [column for column in observation_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Observation frame is missing allowed fields: {missing}")
    forbidden = {
        "patient_id",
        "time_step",
        "split",
        "hospital_id",
        "provider_id",
        "observed_clinician_action",
        "terminal",
    }
    forbidden.update(column for column in observation_columns if column.startswith("next_6h_"))
    forbidden.update(column for column in observation_columns if column.startswith("adverse_"))
    overlap = forbidden & set(observation_columns)
    if overlap:
        raise ValueError(f"Forbidden policy inputs requested: {sorted(overlap)}")
    has_note = note_column in frame.columns
    observations: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        observations.append(
            {
                "structured": {
                    column: _python_value(row[column]) for column in observation_columns
                },
                "handoff_note": _python_value(row[note_column]) if has_note else None,
            }
        )
    return observations


def _observation_frame(
    observations: Sequence[Mapping[str, Any]] | pd.DataFrame,
    observation_columns: Sequence[str],
) -> pd.DataFrame:
    if isinstance(observations, pd.DataFrame):
        missing = [column for column in observation_columns if column not in observations]
        if missing:
            raise ValueError(f"Observation frame is missing allowed fields: {missing}")
        return observations.loc[:, list(observation_columns)].copy()
    records: list[dict[str, Any]] = []
    allowed = set(observation_columns)
    for observation in observations:
        if set(observation) - {"structured", "handoff_note"}:
            raise ValueError("Policy observation contains unsupported top-level fields")
        structured = observation.get("structured")
        if not isinstance(structured, Mapping):
            raise ValueError("Policy observation must contain a structured mapping")
        if set(structured) != allowed:
            raise ValueError("Structured policy fields do not match the observation manifest")
        records.append({column: structured[column] for column in observation_columns})
    return pd.DataFrame.from_records(records, columns=list(observation_columns))


class PolicyFeaturePreprocessor:
    """Role-local structured-state preprocessing shared by conventional policies."""

    def __init__(self, observation_columns: Sequence[str]) -> None:
        validate_policy_columns(observation_columns)
        self.observation_columns = tuple(observation_columns)
        self.categorical_columns = tuple(
            column
            for column in self.observation_columns
            if column in CATEGORICAL_OBSERVATION_COLUMNS
        )
        self.numeric_columns = tuple(
            column
            for column in self.observation_columns
            if column not in CATEGORICAL_OBSERVATION_COLUMNS
        )
        numeric = Pipeline(
            [
                (
                    "impute",
                    SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                ),
                ("scale", StandardScaler()),
            ]
        )
        categorical = Pipeline(
            [
                (
                    "impute",
                    SimpleImputer(
                        strategy="most_frequent", add_indicator=True, keep_empty_features=True
                    ),
                ),
                (
                    "one_hot",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )
        self.transformer = ColumnTransformer(
            [
                ("numeric", numeric, list(self.numeric_columns)),
                ("categorical", categorical, list(self.categorical_columns)),
            ],
            remainder="drop",
            verbose_feature_names_out=True,
        )
        self.is_fitted = False

    def _coerce(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.loc[:, list(self.observation_columns)].copy()
        for column in self.numeric_columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
        for column in self.categorical_columns:
            result[column] = result[column].astype("object")
            result.loc[result[column].isna(), column] = np.nan
        return result

    def fit(self, frame: pd.DataFrame) -> PolicyFeaturePreprocessor:
        selected = _observation_frame(frame, self.observation_columns)
        self.transformer.fit(self._coerce(selected))
        self.is_fitted = True
        return self

    def transform(self, observations: Sequence[Mapping[str, Any]] | pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("PolicyFeaturePreprocessor must be fit before transform")
        frame = _observation_frame(observations, self.observation_columns)
        return np.asarray(self.transformer.transform(self._coerce(frame)), dtype=float)


class BehaviorClonePolicy(Policy):
    """Multinomial behavior clone fit only on the caller-provided development rows."""

    def __init__(
        self,
        observation_columns: Sequence[str],
        *,
        random_state: int = DEFAULT_POLICY_SPLIT_SEED,
        regularization_c: float = 1.0,
    ) -> None:
        if regularization_c <= 0 or not np.isfinite(regularization_c):
            raise ValueError("regularization_c must be finite and positive")
        self.observation_columns = tuple(observation_columns)
        self.preprocessor = PolicyFeaturePreprocessor(self.observation_columns)
        self.classifier = LogisticRegression(
            C=float(regularization_c),
            max_iter=2_000,
            random_state=int(random_state),
            solver="lbfgs",
        )
        self.fitted_patient_ids_: tuple[str, ...] = ()
        self.fitted_patient_checksum_: str | None = None
        self.fit_row_count_ = 0
        self.is_fitted = False

    def fit(
        self,
        frame: pd.DataFrame,
        *,
        action_column: str = "observed_clinician_action",
    ) -> BehaviorClonePolicy:
        if action_column not in frame:
            raise ValueError(f"Training frame is missing {action_column!r}")
        labels = frame[action_column].astype(str).to_numpy()
        unknown = sorted(set(labels) - set(ACTION_NAMES))
        missing = sorted(set(ACTION_NAMES) - set(labels))
        if unknown or missing:
            raise ValueError(f"Behavior labels mismatch; unknown={unknown}, missing={missing}")
        features = self.preprocessor.fit(frame).transform(frame)
        self.classifier.fit(features, labels)
        self.fit_row_count_ = len(frame)
        if "patient_id" in frame:
            self.fitted_patient_ids_ = tuple(sorted(frame["patient_id"].astype(str).unique()))
            encoded = "\n".join(self.fitted_patient_ids_).encode()
            self.fitted_patient_checksum_ = hashlib.sha256(encoded).hexdigest()
        self.is_fitted = True
        return self

    def predict_proba(self, observations: Sequence[Mapping[str, Any]] | pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("BehaviorClonePolicy must be fit before prediction")
        features = self.preprocessor.transform(observations)
        raw = self.classifier.predict_proba(features)
        class_index = {str(label): index for index, label in enumerate(self.classifier.classes_)}
        ordered = raw[:, [class_index[action] for action in ACTION_NAMES]]
        return validate_probability_matrix(ordered, expected_rows=len(features))


@dataclass(frozen=True)
class ParsedPolicyOutput:
    probabilities: np.ndarray
    rationale: str


def normalize_probabilities(values: Sequence[float]) -> np.ndarray:
    probabilities = np.asarray(values, dtype=float)
    if probabilities.shape != (len(ACTION_NAMES),):
        raise ValueError(f"Expected {len(ACTION_NAMES)} action probabilities")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Probabilities must be finite")
    if np.any(probabilities < 0):
        raise ValueError("Probabilities must be non-negative")
    total = float(probabilities.sum())
    if total <= 0:
        raise ValueError("Probability mass must be positive")
    return probabilities / total


def parse_policy_output(raw: str | Mapping[str, Any]) -> ParsedPolicyOutput:
    payload = json.loads(raw) if isinstance(raw, str) else dict(raw)
    if set(payload) != {"probabilities", "rationale"}:
        raise ValueError("Output must contain exactly probabilities and rationale")
    probability_map = payload["probabilities"]
    if not isinstance(probability_map, Mapping) or set(probability_map) != set(ACTION_NAMES):
        raise ValueError("Probability object must contain exactly the canonical actions")
    rationale = payload["rationale"]
    if not isinstance(rationale, str) or len(rationale) > 400:
        raise ValueError("Rationale must be a string of at most 400 characters")
    probabilities = normalize_probabilities([probability_map[action] for action in ACTION_NAMES])
    return ParsedPolicyOutput(probabilities=probabilities, rationale=rationale)


class LLMPolicy(Policy):
    """Prompted policy with deterministic, observable fallback behavior."""

    def __init__(
        self,
        client: LLMClient,
        observation_columns: Sequence[str],
        *,
        model_id: str,
        prompt_version: str = PROMPT_VERSION,
        temperature: float = 0.0,
        fallback_probabilities: Sequence[float] = (1 / 3, 1 / 3, 1 / 3),
        require_note: bool = True,
    ) -> None:
        if prompt_version != PROMPT_VERSION:
            raise ValueError(f"Unsupported prompt version: {prompt_version}")
        if not isfinite(float(temperature)) or temperature < 0:
            raise ValueError("Temperature must be finite and non-negative")
        self.client = client
        self.observation_columns = tuple(observation_columns)
        self.model_id = model_id
        self.prompt_version = prompt_version
        self.temperature = float(temperature)
        self.fallback_probabilities = normalize_probabilities(fallback_probabilities)
        self.require_note = require_note
        self.total_requests = 0
        self.fallback_count = 0
        self.failure_reasons: dict[str, int] = {}
        self.last_rationales: list[str | None] = []

    @property
    def fallback_rate(self) -> float:
        return self.fallback_count / self.total_requests if self.total_requests else 0.0

    def _record_failure(self, error: Exception) -> None:
        name = type(error).__name__
        self.failure_reasons[name] = self.failure_reasons.get(name, 0) + 1
        self.fallback_count += 1

    def predict_proba(self, observations: Sequence[Mapping[str, Any]]) -> np.ndarray:
        rows: list[np.ndarray] = []
        self.last_rationales = []
        for observation in observations:
            # Missing required input is a dataset error, not malformed
            # model output, and must not be silently hidden by fallback behavior.
            messages = build_policy_messages(
                observation,
                self.observation_columns,
                require_note=self.require_note,
            )
            self.total_requests += 1
            try:
                raw = self.client.complete(
                    messages,
                    model_id=self.model_id,
                    response_schema=POLICY_OUTPUT_SCHEMA,
                    temperature=self.temperature,
                )
                parsed = parse_policy_output(raw)
            except MissingHandoffNoteError:
                raise
            except (KeyError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as error:
                self._record_failure(error)
                rows.append(self.fallback_probabilities.copy())
                self.last_rationales.append(None)
                continue
            rows.append(parsed.probabilities)
            self.last_rationales.append(parsed.rationale)
        if not rows:
            return np.empty((0, len(ACTION_NAMES)), dtype=float)
        return np.vstack(rows)


class CachedTokenLogitPolicy(Policy):
    """Model-neutral cached next-token policy over single-token A/B/C choices."""

    def __init__(
        self,
        observation_columns: Sequence[str],
        cache: JsonlTokenProbabilityCache,
        *,
        scorer: ChoiceLogitScorer | None = None,
        model_id: str,
        model_revision: str,
        prompt_version: str = CHOICE_PROMPT_VERSION,
        choice_tokens: Sequence[str] = CHOICE_TOKENS,
        logit_temperature: float = 1.0,
        fallback_probabilities: Sequence[float] = (1 / 3, 1 / 3, 1 / 3),
        batch_size: int = 32,
        inference_dtype: str = "bfloat16",
        require_note: bool = False,
        fail_on_cache_miss: bool = True,
        scorer_semantic_fingerprint: str | None = None,
        candidate_id: str | None = None,
        selected_variant: str = "zero_shot",
        selection_receipts: Mapping[str, Mapping[str, str]] | None = None,
        _bind_scorer_fingerprint: bool = True,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if logit_temperature <= 0 or not np.isfinite(logit_temperature):
            raise ValueError("logit_temperature must be finite and positive")
        if len(choice_tokens) != len(ACTION_NAMES) or len(set(choice_tokens)) != len(ACTION_NAMES):
            raise ValueError("choice_tokens must contain three distinct tokens")
        if scorer is not None and tuple(scorer.choice_tokens) != tuple(choice_tokens):
            raise ValueError("Policy choice tokens do not match the scorer tokenizer")
        if not model_id or not model_revision:
            raise ValueError("model_id and model_revision must be non-empty")
        configured_fingerprint = scorer_semantic_fingerprint
        scorer_fingerprint = (
            None if scorer is None else getattr(scorer, "semantic_fingerprint", None)
        )
        if configured_fingerprint is None and scorer is not None and _bind_scorer_fingerprint:
            configured_fingerprint = scorer_fingerprint
        if configured_fingerprint is not None and (
            not isinstance(configured_fingerprint, str)
            or _SHA256_PATTERN.fullmatch(configured_fingerprint) is None
        ):
            raise ValueError("scorer_semantic_fingerprint must be a lowercase SHA-256 digest")
        if configured_fingerprint is not None and scorer is not None:
            if scorer_fingerprint != configured_fingerprint:
                raise ValueError(
                    "Configured scorer semantic fingerprint does not match the live scorer"
                )
        if scorer is not None and _bind_scorer_fingerprint and configured_fingerprint is None:
            raise ValueError("Live cached scoring requires scorer.semantic_fingerprint")
        if candidate_id is not None and (
            not isinstance(candidate_id, str)
            or not candidate_id.strip()
            or candidate_id != candidate_id.strip()
        ):
            raise ValueError("candidate_id must be a non-empty trimmed string when set")
        if selected_variant not in SELECTED_VARIANT_NAMES:
            raise ValueError("selected_variant must be one of " + ", ".join(SELECTED_VARIANT_NAMES))
        receipts: dict[str, dict[str, str]] = {}
        if selection_receipts is not None:
            if not isinstance(selection_receipts, Mapping) or set(selection_receipts) != {
                "tournament",
                "selected_model",
            }:
                raise ValueError(
                    "selection_receipts must contain exactly tournament and selected_model"
                )
            receipts = {
                name: _receipt_reference(reference, field=name).as_dict()
                for name, reference in selection_receipts.items()
            }
        self.observation_columns = tuple(observation_columns)
        self.cache = cache
        self.scorer = scorer
        self.model_id = model_id
        self.model_revision = model_revision
        self.prompt_version = prompt_version
        self.choice_tokens = tuple(choice_tokens)
        self.logit_temperature = float(logit_temperature)
        self.fallback_probabilities = normalize_probabilities(fallback_probabilities)
        self.batch_size = int(batch_size)
        self.inference_dtype = str(inference_dtype)
        self.require_note = require_note
        self.fail_on_cache_miss = bool(fail_on_cache_miss)
        self.scorer_semantic_fingerprint = configured_fingerprint
        self.candidate_id = candidate_id
        self.selected_variant = selected_variant
        self.selection_receipts = receipts
        self.total_requests = 0
        self.fallback_count = 0
        self.malformed_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.failure_reasons: dict[str, int] = {}
        self.last_rationales: list[str | None] = []
        self._cached_execution_settings: set[tuple[int, str]] = set()

    @property
    def fallback_rate(self) -> float:
        return self.fallback_count / self.total_requests if self.total_requests else 0.0

    @property
    def diagnostics(self) -> PolicyRunDiagnostics:
        return PolicyRunDiagnostics(
            requests=self.total_requests,
            fallbacks=self.fallback_count,
            malformed_outputs=self.malformed_count,
            cache_hits=self.cache_hits,
            cache_misses=self.cache_misses,
        )

    @property
    def execution_metadata(self) -> dict[str, Any]:
        return {
            "configured_batch_size": self.batch_size,
            "configured_dtype": self.inference_dtype,
            "cached_generation_settings": [
                {"batch_size": batch_size, "dtype": dtype}
                for batch_size, dtype in sorted(self._cached_execution_settings)
            ],
        }

    @property
    def policy_metadata(self) -> dict[str, Any]:
        """Expose model-neutral identity and promotion receipts for reporting."""

        return {
            "policy_class": type(self).__name__,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "candidate_id": self.candidate_id,
            "selected_variant": self.selected_variant,
            "prompt_version": self.prompt_version,
            "scorer_semantic_fingerprint": self.scorer_semantic_fingerprint,
            "selection_receipts": {
                name: dict(reference) for name, reference in self.selection_receipts.items()
            },
        }

    def _request(self, observation: Mapping[str, Any]) -> tuple[str, list[dict[str, str]], str]:
        messages = build_choice_messages(
            observation,
            self.observation_columns,
            require_note=self.require_note,
            prompt_version=self.prompt_version,
        )
        serialized = json.dumps(
            messages,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        prompt_sha256 = hashlib.sha256(serialized.encode()).hexdigest()
        key = token_probability_cache_key(
            messages,
            model_id=self.model_id,
            model_revision=self.model_revision,
            prompt_version=self.prompt_version,
            choice_tokens=self.choice_tokens,
            logit_temperature=self.logit_temperature,
            scorer_semantic_fingerprint=self.scorer_semantic_fingerprint,
        )
        return key, messages, prompt_sha256

    def _require_live_scorer_fingerprint(self) -> None:
        """Fail closed if a fingerprinted live scorer changes after construction."""

        if self.scorer is None or self.scorer_semantic_fingerprint is None:
            return
        current = getattr(self.scorer, "semantic_fingerprint", None)
        if current != self.scorer_semantic_fingerprint:
            raise ValueError("Live scorer semantic fingerprint changed during policy use")

    def _record_failure(self, error: Exception, count: int) -> None:
        reason = type(error).__name__
        self.failure_reasons[reason] = self.failure_reasons.get(reason, 0) + count
        self.fallback_count += count

    def _validate_cached_record(
        self, record: TokenProbabilityRecord, *, prompt_sha256: str
    ) -> None:
        semantic_fields_match = (
            record.model_id == self.model_id
            and record.model_revision == self.model_revision
            and record.prompt_version == self.prompt_version
            and record.prompt_sha256 == prompt_sha256
            and record.choice_tokens == self.choice_tokens
            and record.scorer_semantic_fingerprint == self.scorer_semantic_fingerprint
            and np.isclose(
                record.logit_temperature,
                self.logit_temperature,
                atol=0.0,
                rtol=0.0,
            )
        )
        if not semantic_fields_match:
            raise ValueError("Cached record semantic metadata does not match the request")
        if not record.schema_valid or record.fallback_reason is not None:
            raise ValueError("Cached record is not a valid model-scored response")

    def predict_proba(self, observations: Sequence[Mapping[str, Any]]) -> np.ndarray:
        if not observations:
            return np.empty((0, len(ACTION_NAMES)), dtype=float)
        requests = [self._request(observation) for observation in observations]
        self.total_requests += len(requests)
        output: list[np.ndarray | None] = [None] * len(requests)
        rationales: list[str | None] = [None] * len(requests)
        missing_by_key: dict[str, tuple[list[dict[str, str]], str, list[int]]] = {}

        for index, (key, messages, prompt_sha256) in enumerate(requests):
            record = self.cache.get(key)
            if record is not None:
                self._validate_cached_record(record, prompt_sha256=prompt_sha256)
                self._cached_execution_settings.add(
                    (record.inference_batch_size, record.inference_dtype)
                )
                output[index] = np.asarray(record.probabilities, dtype=float)
                rationales[index] = record.rationale
                self.cache_hits += 1
                continue
            self.cache_misses += 1
            if key not in missing_by_key:
                missing_by_key[key] = (messages, prompt_sha256, [])
            missing_by_key[key][2].append(index)

        missing_items = list(missing_by_key.items())
        if self.scorer is None and missing_items:
            error = KeyError("Exact-token response is absent and no local scorer was supplied")
            missing_count = sum(len(item[1][2]) for item in missing_items)
            if self.fail_on_cache_miss:
                self._record_failure(error, missing_count)
                raise error
            self._record_failure(error, missing_count)
            for _, (_, _, indexes) in missing_items:
                for index in indexes:
                    output[index] = self.fallback_probabilities.copy()
        elif self.scorer is not None:
            for start in range(0, len(missing_items), self.batch_size):
                batch = missing_items[start : start + self.batch_size]
                self._require_live_scorer_fingerprint()
                try:
                    score_messages = [item[1][0] for item in batch]
                    if len(score_messages) < self.batch_size:
                        score_messages.extend(
                            [score_messages[-1]] * (self.batch_size - len(score_messages))
                        )
                    scores = self.scorer.score_messages(score_messages)
                    scores = validate_probability_matrix(scores, expected_rows=self.batch_size)[
                        : len(batch)
                    ]
                except (RuntimeError, TypeError, ValueError) as error:
                    affected = sum(len(item[1][2]) for item in batch)
                    self._record_failure(error, affected)
                    self.malformed_count += affected
                    for _, (_, _, indexes) in batch:
                        for index in indexes:
                            output[index] = self.fallback_probabilities.copy()
                    continue
                self._require_live_scorer_fingerprint()

                for (key, (_, prompt_sha256, indexes)), probabilities in zip(
                    batch, scores, strict=True
                ):
                    chosen = ACTION_NAMES[int(np.argmax(probabilities))]
                    rationale = (
                        "Code-generated audit annotation (not a model rationale): "
                        f"pinned token-logit scorer favored {chosen}; "
                        "probabilities are normalized only across A/B/C."
                    )
                    record = TokenProbabilityRecord(
                        key=key,
                        model_id=self.model_id,
                        model_revision=self.model_revision,
                        prompt_version=self.prompt_version,
                        prompt_sha256=prompt_sha256,
                        choice_tokens=self.choice_tokens,
                        logit_temperature=self.logit_temperature,
                        inference_batch_size=self.batch_size,
                        inference_dtype=self.inference_dtype,
                        probabilities=tuple(float(value) for value in probabilities),
                        rationale=rationale,
                        scorer_semantic_fingerprint=self.scorer_semantic_fingerprint,
                    )
                    self.cache.put(record)
                    self._cached_execution_settings.add(
                        (record.inference_batch_size, record.inference_dtype)
                    )
                    for index in indexes:
                        output[index] = probabilities.copy()
                        rationales[index] = rationale

        if any(row is None for row in output):
            raise RuntimeError("Internal error: a policy probability row was not populated")
        self.last_rationales = rationales
        return validate_probability_matrix(np.vstack(output), expected_rows=len(requests))


class ActionValueModel(Protocol):
    def predict_action_values(self, observations: Sequence[Mapping[str, Any]]) -> np.ndarray:
        """Return an N x 3 action-value matrix."""


class SupportModel(Protocol):
    def predict_proba(self, observations: Sequence[Mapping[str, Any]]) -> np.ndarray:
        """Return behavior probabilities in canonical action order."""


def support_aware_probabilities(
    base: Sequence[Sequence[float]] | np.ndarray,
    action_values: Sequence[Sequence[float]] | np.ndarray,
    support: Sequence[Sequence[float]] | np.ndarray,
    *,
    base_temperature: float,
    advantage_weight: float,
    support_weight: float,
    advantage_scale: float,
    support_floor: float,
    advantage_clip: float = 2.0,
    low_support_penalty: float = 0.0,
    probability_floor: float = 1e-12,
) -> np.ndarray:
    """Apply the production support-aware transform to already-frozen matrices.

    The pure function is also the single source of truth for train-only grouped OOF
    grid selection, preventing tuning/serving formula drift.
    """

    base_matrix = validate_probability_matrix(base)
    support_matrix = validate_probability_matrix(support, expected_rows=len(base_matrix))
    values = np.asarray(action_values, dtype=float)
    if values.shape != base_matrix.shape or not np.all(np.isfinite(values)):
        raise ValueError("Action values must be a finite N x 3 matrix")
    positive = (
        base_temperature,
        advantage_scale,
        advantage_clip,
        support_floor,
        probability_floor,
    )
    nonnegative = (advantage_weight, support_weight, low_support_penalty)
    if not all(np.isfinite(value) and value > 0 for value in positive):
        raise ValueError("Temperature, scales, clips, and floors must be positive")
    if not all(np.isfinite(value) and value >= 0 for value in nonnegative):
        raise ValueError("Reranking weights and penalties must be non-negative")
    if support_floor >= 1 or probability_floor >= 1:
        raise ValueError("Probability/support floors must be below one")

    baseline_value = np.sum(base_matrix * values, axis=1, keepdims=True)
    standardized_advantage = np.clip(
        (values - baseline_value) / advantage_scale,
        -advantage_clip,
        advantage_clip,
    )
    logits = (
        np.log(np.maximum(base_matrix, probability_floor)) / base_temperature
        + advantage_weight * standardized_advantage
        + support_weight * np.log(np.maximum(support_matrix, support_floor))
    )
    if low_support_penalty:
        support_shortfall = np.maximum(
            0.0,
            np.log(support_floor) - np.log(np.maximum(support_matrix, probability_floor)),
        )
        logits -= low_support_penalty * support_shortfall
    logits -= np.max(logits, axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return validate_probability_matrix(probabilities, expected_rows=len(base_matrix))


def behavior_blend_probabilities(
    llm_probabilities: Sequence[Sequence[float]] | np.ndarray,
    behavior_probabilities: Sequence[Sequence[float]] | np.ndarray,
    *,
    llm_weight: float,
) -> np.ndarray:
    """Convexly anchor an LLM policy to an observation-only behavior estimate.

    The mixture is intentionally global and transparent.  It is a conservative
    development policy class, not a claim of a formal safe-policy-improvement
    guarantee: the behavior policy is estimated and the OPE assumptions remain.
    """

    llm = validate_probability_matrix(llm_probabilities)
    behavior = validate_probability_matrix(behavior_probabilities, expected_rows=len(llm))
    if not np.isfinite(llm_weight) or not 0.0 < llm_weight < 1.0:
        raise ValueError("Behavior-blend LLM weight must be finite and strictly in (0, 1)")
    blended = float(llm_weight) * llm + (1.0 - float(llm_weight)) * behavior
    return validate_probability_matrix(blended, expected_rows=len(llm))


class SupportAwareImprovedPolicy(Policy):
    """Training-only Q reranker with an explicit low-behavior-support penalty."""

    def __init__(
        self,
        base_policy: Policy,
        action_value_model: ActionValueModel,
        support_model: SupportModel,
        *,
        base_temperature: float = 1.0,
        advantage_weight: float = 0.5,
        support_weight: float = 0.5,
        advantage_scale: float = 1.0,
        advantage_clip: float = 2.0,
        support_floor: float = 0.01,
        low_support_penalty: float = 0.0,
        probability_floor: float = 1e-12,
        selected_variant: str | None = None,
        selected_parameters: Mapping[str, Any] | None = None,
    ) -> None:
        if selected_variant is not None:
            if selected_parameters is None:
                raise ValueError(
                    "selected_parameters are required with an explicit selected_variant"
                )
            applied = selected_variant_parameters(selected_variant, selected_parameters)
            expected = {
                "base_temperature": float(base_temperature),
                "advantage_weight": float(advantage_weight),
                "support_weight": float(support_weight),
            }
            if applied != expected:
                raise ValueError(
                    "Runtime transform parameters do not match the selected variant contract"
                )
            if selected_variant not in {
                "support_only",
                "value_plus_support",
                *BEHAVIOR_BLEND_LLM_WEIGHTS,
            }:
                low_support_penalty = 0.0
        positive = (
            base_temperature,
            advantage_scale,
            advantage_clip,
            support_floor,
            probability_floor,
        )
        nonnegative = (advantage_weight, support_weight, low_support_penalty)
        if not all(np.isfinite(value) and value > 0 for value in positive):
            raise ValueError("Temperature, scales, clips, and floors must be positive")
        if not all(np.isfinite(value) and value >= 0 for value in nonnegative):
            raise ValueError("Reranking weights and penalties must be non-negative")
        if support_floor >= 1 or probability_floor >= 1:
            raise ValueError("Probability/support floors must be below one")
        self.base_policy = base_policy
        self.action_value_model = action_value_model
        self.support_model = support_model
        self.base_temperature = float(base_temperature)
        self.advantage_weight = float(advantage_weight)
        self.support_weight = float(support_weight)
        self.advantage_scale = float(advantage_scale)
        self.advantage_clip = float(advantage_clip)
        self.support_floor = float(support_floor)
        self.low_support_penalty = float(low_support_penalty)
        self.probability_floor = float(probability_floor)
        self.selected_variant = selected_variant
        self.behavior_blend_llm_weight = (
            1.0
            if selected_variant is None
            else float(BEHAVIOR_BLEND_LLM_WEIGHTS.get(selected_variant, 1.0))
        )
        self.selected_parameters = (
            None
            if selected_parameters is None
            else {
                key: float(selected_parameters[key]) for key in sorted(_SELECTED_PARAMETER_FIELDS)
            }
        )
        base_model_id = getattr(base_policy, "model_id", "llm")
        base_prompt_version = getattr(base_policy, "prompt_version", CHOICE_PROMPT_VERSION)
        if selected_variant is None:
            self.model_id = f"{base_model_id}-support-aware"
            self.prompt_version = f"{base_prompt_version}+q-support-v1"
        else:
            self.model_id = f"{base_model_id}::{selected_variant}"
            self.prompt_version = f"{base_prompt_version}+selected-variant-v1:{selected_variant}"

    @property
    def fallback_rate(self) -> float:
        return float(getattr(self.base_policy, "fallback_rate", 0.0))

    @property
    def diagnostics(self) -> PolicyRunDiagnostics:
        return getattr(self.base_policy, "diagnostics", PolicyRunDiagnostics())

    @property
    def execution_metadata(self) -> dict[str, Any]:
        return dict(getattr(self.base_policy, "execution_metadata", {}))

    @property
    def policy_metadata(self) -> dict[str, Any]:
        metadata = dict(getattr(self.base_policy, "policy_metadata", {}))
        metadata.update(
            {
                "policy_class": type(self).__name__,
                "model_id": self.model_id,
                "selected_variant": self.selected_variant,
                "selected_parameters": (
                    None if self.selected_parameters is None else dict(self.selected_parameters)
                ),
                "applied_parameters": {
                    "base_temperature": self.base_temperature,
                    "advantage_weight": self.advantage_weight,
                    "support_weight": self.support_weight,
                },
                "applied_low_support_penalty": self.low_support_penalty,
                "behavior_blend_llm_weight": self.behavior_blend_llm_weight,
            }
        )
        return metadata

    def predict_proba(self, observations: Sequence[Mapping[str, Any]]) -> np.ndarray:
        base = validate_probability_matrix(
            self.base_policy.predict_proba(observations), expected_rows=len(observations)
        )
        if self.selected_variant == "zero_shot":
            return base.copy()
        values = np.asarray(
            self.action_value_model.predict_action_values(observations), dtype=float
        )
        support = validate_probability_matrix(
            self.support_model.predict_proba(observations), expected_rows=len(observations)
        )
        improved = support_aware_probabilities(
            base,
            values,
            support,
            base_temperature=self.base_temperature,
            advantage_weight=self.advantage_weight,
            support_weight=self.support_weight,
            advantage_scale=self.advantage_scale,
            support_floor=self.support_floor,
            advantage_clip=self.advantage_clip,
            low_support_penalty=self.low_support_penalty,
            probability_floor=self.probability_floor,
        )
        if self.behavior_blend_llm_weight < 1.0:
            return behavior_blend_probabilities(
                improved,
                support,
                llm_weight=self.behavior_blend_llm_weight,
            )
        return improved
