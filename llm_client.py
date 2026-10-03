"""Provider-neutral LLM client and deterministic JSONL response cache."""

from __future__ import annotations

import hashlib
import inspect
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from contracts import ACTION_NAMES, validate_probability_matrix
from llm_provenance import configure_deterministic_algorithms

CHOICE_TOKENS = ("A", "B", "C")
CACHE_FORMAT_VERSION = 1
SCORER_FINGERPRINT_SCHEMA_VERSION = "transformers-choice-scorer-v1"


def _json_safe(value: Any) -> Any:
    """Return a deterministic JSON-compatible representation for fingerprints."""

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("Semantic fingerprint metadata must contain finite numbers")
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "to_dict"):
        return _json_safe(value.to_dict())
    return str(value)


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _json_safe(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: str, *, field: str) -> str:
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


class LLMClient(Protocol):
    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        model_id: str,
        response_schema: Mapping[str, Any],
        temperature: float,
    ) -> str:
        """Return the model's raw response text."""


def request_cache_key(
    messages: Sequence[Mapping[str, str]],
    *,
    model_id: str,
    prompt_version: str,
    response_schema: Mapping[str, Any],
    temperature: float,
) -> str:
    payload = {
        "messages": list(messages),
        "model_id": model_id,
        "prompt_version": prompt_version,
        "response_schema": response_schema,
        "temperature": temperature,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class JsonlResponseCache:
    """Append-only cache keyed by the full prompt/model/schema configuration."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._responses: dict[str, str] = {}
        if self.path.exists():
            for line_number, line in enumerate(
                self.path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if not line.strip():
                    continue
                record = json.loads(line)
                key = record.get("key")
                response = record.get("response")
                if not isinstance(key, str) or not isinstance(response, str):
                    raise ValueError(f"Invalid cache record on line {line_number}")
                previous = self._responses.get(key)
                if previous is not None and previous != response:
                    raise ValueError(f"Conflicting cached responses for key {key}")
                self._responses[key] = response

    def get(self, key: str) -> str | None:
        return self._responses.get(key)

    def put(self, key: str, response: str) -> None:
        previous = self._responses.get(key)
        if previous is not None:
            if previous != response:
                raise ValueError(f"Refusing to overwrite cached response for key {key}")
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = json.dumps(
            {"key": key, "response": response},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(record + "\n")
        self._responses[key] = response


class CachedLLMClient:
    """Require cache hits or populate them through an explicitly supplied client."""

    def __init__(
        self,
        cache: JsonlResponseCache,
        *,
        prompt_version: str,
        upstream: LLMClient | None = None,
    ) -> None:
        self.cache = cache
        self.prompt_version = prompt_version
        self.upstream = upstream
        self.hits = 0
        self.misses = 0

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        model_id: str,
        response_schema: Mapping[str, Any],
        temperature: float,
    ) -> str:
        key = request_cache_key(
            messages,
            model_id=model_id,
            prompt_version=self.prompt_version,
            response_schema=response_schema,
            temperature=temperature,
        )
        cached = self.cache.get(key)
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1
        if self.upstream is None:
            raise KeyError(f"No cached LLM response for request {key}")
        response = self.upstream.complete(
            messages,
            model_id=model_id,
            response_schema=response_schema,
            temperature=temperature,
        )
        self.cache.put(key, response)
        return response


class StaticLLMClient:
    """Deterministic test double; it is not an LLM and must be disclosed as such."""

    def __init__(self, responses: Sequence[str]) -> None:
        self._responses = list(responses)
        self.calls = 0

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        model_id: str,
        response_schema: Mapping[str, Any],
        temperature: float,
    ) -> str:
        del messages, model_id, response_schema, temperature
        if self.calls >= len(self._responses):
            raise RuntimeError("StaticLLMClient has no remaining responses")
        response = self._responses[self.calls]
        self.calls += 1
        return response


def token_probability_cache_key(
    messages: Sequence[Mapping[str, str]],
    *,
    model_id: str,
    model_revision: str,
    prompt_version: str,
    choice_tokens: Sequence[str],
    logit_temperature: float,
    scorer_semantic_fingerprint: str | None = None,
) -> str:
    """Hash every semantic/model input used by exact candidate-token scoring."""

    payload = {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "messages": list(messages),
        "model_id": model_id,
        "model_revision": model_revision,
        "prompt_version": prompt_version,
        "choice_tokens": list(choice_tokens),
        "decoding": {
            "method": "candidate_normalized_exact_next_token_logits",
            "logit_temperature": logit_temperature,
            "generation": False,
            "final_position_logits_only": True,
        },
        "actions": list(ACTION_NAMES),
    }
    if scorer_semantic_fingerprint is not None:
        payload["scorer_semantic_fingerprint"] = _require_sha256(
            scorer_semantic_fingerprint,
            field="scorer_semantic_fingerprint",
        )
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class TokenProbabilityRecord:
    """Validated replay record for one prompted observation.

    ``rationale`` is the legacy serialized field name. Restricted-token scoring does
    not generate prose; this value is a deterministic code-generated audit annotation,
    never a model-authored chain of thought or clinical explanation.
    """

    key: str
    model_id: str
    model_revision: str
    prompt_version: str
    prompt_sha256: str
    choice_tokens: tuple[str, ...]
    logit_temperature: float
    inference_batch_size: int
    inference_dtype: str
    probabilities: tuple[float, ...]
    rationale: str
    schema_valid: bool = True
    fallback_reason: str | None = None
    scorer_semantic_fingerprint: str | None = None

    @property
    def audit_annotation(self) -> str:
        """Expose the legacy ``rationale`` value with its accurate provenance name."""

        return self.rationale

    def as_dict(self) -> dict[str, Any]:
        decoding: dict[str, Any] = {
            "method": "candidate_normalized_exact_next_token_logits",
            "logit_temperature": self.logit_temperature,
            "generation": False,
            "configured_batch_size": self.inference_batch_size,
            "dtype": self.inference_dtype,
            "final_position_logits_only": True,
        }
        if self.scorer_semantic_fingerprint is not None:
            decoding["scorer_semantic_fingerprint"] = _require_sha256(
                self.scorer_semantic_fingerprint,
                field="scorer_semantic_fingerprint",
            )
        return {
            "cache_format_version": CACHE_FORMAT_VERSION,
            "key": self.key,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "prompt_version": self.prompt_version,
            "prompt_sha256": self.prompt_sha256,
            "choice_tokens": list(self.choice_tokens),
            "decoding": decoding,
            "schema": {"actions": list(ACTION_NAMES), "schema_valid": self.schema_valid},
            "probabilities": {
                action: probability
                for action, probability in zip(ACTION_NAMES, self.probabilities, strict=True)
            },
            "rationale": self.rationale,
            "fallback_reason": self.fallback_reason,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TokenProbabilityRecord:
        if payload.get("cache_format_version") != CACHE_FORMAT_VERSION:
            raise ValueError("Unsupported token-probability cache format")
        schema = payload.get("schema")
        if not isinstance(schema, Mapping) or schema.get("actions") != list(ACTION_NAMES):
            raise ValueError("Cached action schema does not match canonical actions")
        probability_map = payload.get("probabilities")
        if not isinstance(probability_map, Mapping) or set(probability_map) != set(ACTION_NAMES):
            raise ValueError("Cached probabilities do not match canonical actions")
        probabilities = validate_probability_matrix(
            [[probability_map[action] for action in ACTION_NAMES]], expected_rows=1
        )[0]
        decoding = payload.get("decoding")
        if not isinstance(decoding, Mapping):
            raise ValueError("Cached decoding metadata is missing")
        choice_tokens = payload.get("choice_tokens")
        if not isinstance(choice_tokens, list) or not all(
            isinstance(token, str) for token in choice_tokens
        ):
            raise ValueError("Cached choice tokens are invalid")
        required_strings = (
            "key",
            "model_id",
            "model_revision",
            "prompt_version",
            "prompt_sha256",
            "rationale",
        )
        if any(not isinstance(payload.get(field), str) for field in required_strings):
            raise ValueError("Cached record is missing required string metadata")
        fallback_reason = payload.get("fallback_reason")
        if fallback_reason is not None and not isinstance(fallback_reason, str):
            raise ValueError("Cached fallback reason must be null or a string")
        scorer_semantic_fingerprint = decoding.get("scorer_semantic_fingerprint")
        if scorer_semantic_fingerprint is not None:
            if not isinstance(scorer_semantic_fingerprint, str):
                raise ValueError("Cached scorer semantic fingerprint must be a string")
            _require_sha256(
                scorer_semantic_fingerprint,
                field="scorer_semantic_fingerprint",
            )
        return cls(
            key=str(payload["key"]),
            model_id=str(payload["model_id"]),
            model_revision=str(payload["model_revision"]),
            prompt_version=str(payload["prompt_version"]),
            prompt_sha256=str(payload["prompt_sha256"]),
            choice_tokens=tuple(choice_tokens),
            logit_temperature=float(decoding["logit_temperature"]),
            inference_batch_size=int(decoding["configured_batch_size"]),
            inference_dtype=str(decoding["dtype"]),
            probabilities=tuple(float(value) for value in probabilities),
            rationale=str(payload["rationale"]),
            schema_valid=bool(schema.get("schema_valid")),
            fallback_reason=fallback_reason,
            scorer_semantic_fingerprint=scorer_semantic_fingerprint,
        )


class JsonlTokenProbabilityCache:
    """Append-only, conflict-detecting cache for candidate-token probabilities."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._records: dict[str, TokenProbabilityRecord] = {}
        if not self.path.exists():
            return
        for line_number, line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                record = TokenProbabilityRecord.from_dict(json.loads(line))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"Invalid token cache record on line {line_number}") from error
            previous = self._records.get(record.key)
            if previous is not None and previous != record:
                raise ValueError(f"Conflicting cached token probabilities for {record.key}")
            self._records[record.key] = record

    def __len__(self) -> int:
        return len(self._records)

    def get(self, key: str) -> TokenProbabilityRecord | None:
        return self._records.get(key)

    def records(self) -> tuple[TokenProbabilityRecord, ...]:
        """Return an immutable snapshot for provenance validation."""

        return tuple(self._records.values())

    def put(self, record: TokenProbabilityRecord) -> None:
        previous = self._records.get(record.key)
        if previous is not None:
            if previous != record:
                raise ValueError(f"Refusing to overwrite cached record {record.key}")
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(
            record.as_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized + "\n")
        self._records[record.key] = record


class ChoiceLogitScorer(Protocol):
    """Batch scorer used by cached token-logit policies and deterministic test doubles."""

    choice_tokens: tuple[str, ...]
    semantic_fingerprint: str

    def score_messages(self, message_batches: Sequence[Sequence[Mapping[str, str]]]) -> np.ndarray:
        """Return candidate-normalized next-token probabilities in A/B/C order."""


class LocalTransformersChoiceScorer:
    """Exact next-token A/B/C scorer for a pinned public Transformers model.

    No text is generated. For each chat prompt, this class extracts final-position
    logits for three verified single-token choices and normalizes only across those
    candidates. It supports causal language models and the text-only path of
    image-text-to-text conditional-generation models.
    """

    def __init__(
        self,
        *,
        model_id: str,
        model_revision: str,
        backend: str = "auto",
        device: str = "auto",
        device_map: str | Mapping[str, Any] | None = None,
        quantization: str | Mapping[str, Any] | None = None,
        local_files_only: bool = True,
        logit_temperature: float = 1.0,
        max_input_tokens: int = 1024,
        dtype: str = "auto",
        attn_implementation: str | None = None,
    ) -> None:
        if not model_id or not model_revision:
            raise ValueError("model_id and model_revision must be non-empty")
        if logit_temperature <= 0 or not np.isfinite(logit_temperature):
            raise ValueError("logit_temperature must be finite and positive")
        if max_input_tokens <= 0:
            raise ValueError("max_input_tokens must be positive")
        if backend not in {"auto", "causal_lm", "image_text_to_text"}:
            raise ValueError(f"Unsupported Transformers backend: {backend}")
        if device_map is not None and device != "auto":
            raise ValueError("device must be 'auto' when device_map is configured")
        try:
            import torch
            import transformers
        except ImportError as error:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "Local model inference requires installed torch and transformers"
            ) from error

        self._torch = torch
        self.model_id = model_id
        self.model_revision = model_revision
        self.logit_temperature = float(logit_temperature)
        self.max_input_tokens = int(max_input_tokens)
        self.requested_backend = backend
        self.device_map = device_map
        self.attn_implementation = attn_implementation

        device_hint = device
        if device_hint == "auto":
            device_hint = "cuda" if torch.cuda.is_available() else "cpu"
        if device_hint.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        configured_device = torch.device(device_hint)

        dtype_options = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if dtype == "auto":
            model_dtype = torch.float32
            if configured_device.type == "cuda":
                model_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif dtype in dtype_options:
            model_dtype = dtype_options[dtype]
        else:
            raise ValueError(f"Unsupported model dtype: {dtype}")
        if device_map is None and configured_device.type == "cpu" and model_dtype == torch.float16:
            raise ValueError("float16 inference is unsupported on CPU")
        self.dtype_name = str(model_dtype).removeprefix("torch.")

        load_common = {
            "revision": model_revision,
            "local_files_only": local_files_only,
            "trust_remote_code": False,
        }
        config = transformers.AutoConfig.from_pretrained(model_id, **load_common)
        self.backend = self._resolve_backend(config, backend)

        self.processor: Any | None = None
        if self.backend == "image_text_to_text":
            processor_loader = getattr(transformers, "AutoProcessor", None)
            if processor_loader is None:
                raise RuntimeError(
                    "This Transformers version has no AutoProcessor for multimodal models"
                )
            self.processor = processor_loader.from_pretrained(model_id, **load_common)
            self.tokenizer = getattr(self.processor, "tokenizer", None)
            if self.tokenizer is None:
                raise RuntimeError("The multimodal processor does not expose a tokenizer")
            self._chat_template_owner = self.processor
        else:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, **load_common)
            self._chat_template_owner = self.tokenizer

        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token is None:
                raise RuntimeError("Tokenizer has neither a pad token nor an EOS token")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.choice_tokens, self.choice_token_ids = self._resolve_choice_tokens()
        chat_template = self._resolved_chat_template()
        tokenizer_semantic_sha256 = self._tokenizer_semantic_sha256()
        chat_template_sha256 = _canonical_sha256({"chat_template": chat_template})

        quantization_config, quantization_payload = self._build_quantization_config(
            transformers, quantization, model_dtype=model_dtype
        )
        effective_device_map = device_map
        if quantization_config is not None and effective_device_map is None:
            effective_device_map = "auto"

        model_kwargs: dict[str, Any] = {
            **load_common,
            "config": config,
            "dtype": model_dtype,
        }
        if effective_device_map is not None:
            model_kwargs["device_map"] = effective_device_map
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        if self.backend == "causal_lm":
            model_loader = transformers.AutoModelForCausalLM
        else:
            model_loader = getattr(transformers, "AutoModelForImageTextToText", None)
            if model_loader is None:
                model_loader = getattr(transformers, "AutoModelForVision2Seq", None)
            if model_loader is None:
                raise RuntimeError(
                    "This Transformers version cannot load image-text-to-text models"
                )
        self.model = model_loader.from_pretrained(model_id, **model_kwargs)
        if effective_device_map is None:
            self.model.to(configured_device)
        self.model.eval()
        self.device = self._resolve_input_device(configured_device)
        self._logit_slice_argument = self._resolve_logit_slice_argument()
        if self._logit_slice_argument is None and self.model.get_output_embeddings() is None:
            raise RuntimeError(
                "Model cannot guarantee final-position-only logits: no output head is exposed"
            )
        configure_deterministic_algorithms(torch)

        actual_quantization = getattr(self.model.config, "quantization_config", None)
        payload = {
            "schema_version": SCORER_FINGERPRINT_SCHEMA_VERSION,
            "model": {
                "id": model_id,
                "revision": model_revision,
                "class": type(self.model).__name__,
            },
            "transformers": {
                "version": str(getattr(transformers, "__version__", "unknown")),
                "backend": self.backend,
                "trust_remote_code": False,
                "text_only": self.backend == "image_text_to_text",
                "attention_implementation": attn_implementation,
            },
            "tokenizer": {
                "class": type(self.tokenizer).__name__,
                "processor_class": (
                    None if self.processor is None else type(self.processor).__name__
                ),
                "semantic_sha256": tokenizer_semantic_sha256,
                "chat_template_sha256": chat_template_sha256,
                "padding_side": self.tokenizer.padding_side,
            },
            "scoring": {
                "method": "candidate_normalized_exact_next_token_logits",
                "generation": False,
                "final_position_logits_only": True,
                "choice_tokens": list(self.choice_tokens),
                "choice_token_ids": list(self.choice_token_ids),
                "logit_temperature": self.logit_temperature,
            },
            "numeric": {
                "requested_dtype": dtype,
                "load_dtype": self.dtype_name,
                "observed_parameter_dtype": self._observed_parameter_dtype(),
                "quantization_requested": quantization_payload,
                "quantization_resolved": _json_safe(actual_quantization),
            },
        }
        self.semantic_fingerprint_payload = _json_safe(payload)
        self.semantic_fingerprint = _canonical_sha256(payload)
        self.scorer_semantic_fingerprint = self.semantic_fingerprint

    @staticmethod
    def _resolve_backend(config: Any, requested: str) -> str:
        if requested != "auto":
            return requested
        architectures = tuple(getattr(config, "architectures", None) or ())
        if any("CausalLM" in architecture for architecture in architectures):
            return "causal_lm"
        multimodal_markers = ("ConditionalGeneration", "Vision2Seq", "ImageTextToText")
        if any(
            any(marker in architecture for marker in multimodal_markers)
            for architecture in architectures
        ):
            return "image_text_to_text"
        raise RuntimeError(
            "Could not infer a supported model backend from config.architectures; "
            "set backend explicitly"
        )

    @staticmethod
    def _build_quantization_config(
        transformers_module: Any,
        quantization: str | Mapping[str, Any] | None,
        *,
        model_dtype: Any,
    ) -> tuple[Any | None, Mapping[str, Any]]:
        if quantization is None:
            return None, {"method": "none"}
        if isinstance(quantization, str):
            if quantization == "4bit":
                options: dict[str, Any] = {
                    "load_in_4bit": True,
                    "bnb_4bit_compute_dtype": model_dtype,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_use_double_quant": True,
                }
            elif quantization == "8bit":
                options = {"load_in_8bit": True}
            else:
                raise ValueError(f"Unsupported quantization preset: {quantization}")
        elif isinstance(quantization, Mapping):
            options = dict(quantization)
            method = options.pop("method", "bitsandbytes")
            if method != "bitsandbytes":
                raise ValueError(f"Unsupported quantization method: {method}")
            requested_compute_dtype = options.get("bnb_4bit_compute_dtype")
            if isinstance(requested_compute_dtype, str):
                import torch

                resolved_dtype = getattr(torch, requested_compute_dtype, None)
                if resolved_dtype is None:
                    raise ValueError(
                        f"Unsupported bnb_4bit_compute_dtype: {requested_compute_dtype}"
                    )
                options["bnb_4bit_compute_dtype"] = resolved_dtype
        else:
            raise TypeError("quantization must be null, a preset string, or a mapping")
        if bool(options.get("load_in_4bit")) == bool(options.get("load_in_8bit")):
            raise ValueError("Quantization must enable exactly one of 4-bit or 8-bit loading")
        config_class = getattr(transformers_module, "BitsAndBytesConfig", None)
        if config_class is None:
            raise RuntimeError("BitsAndBytesConfig is unavailable in this Transformers build")
        config = config_class(**options)
        return config, {"method": "bitsandbytes", "config": _json_safe(config)}

    def _resolved_chat_template(self) -> Any:
        for owner in (self._chat_template_owner, self.tokenizer):
            getter = getattr(owner, "get_chat_template", None)
            if getter is not None:
                try:
                    template = getter()
                except (KeyError, TypeError, ValueError):
                    template = None
                if template:
                    return template
            template = getattr(owner, "chat_template", None)
            if template:
                return template
        raise RuntimeError("The pinned tokenizer/processor has no usable chat template")

    def _tokenizer_semantic_sha256(self) -> str:
        backend_tokenizer = getattr(self.tokenizer, "backend_tokenizer", None)
        if backend_tokenizer is not None and hasattr(backend_tokenizer, "to_str"):
            tokenizer_definition: Any = backend_tokenizer.to_str()
        else:
            get_vocab = getattr(self.tokenizer, "get_vocab", None)
            if get_vocab is None:
                raise RuntimeError("Tokenizer exposes neither a fast backend nor a vocabulary")
            tokenizer_definition = get_vocab()
        return _canonical_sha256(
            {
                "tokenizer_class": type(self.tokenizer).__name__,
                "definition": tokenizer_definition,
                "special_tokens_map": getattr(self.tokenizer, "special_tokens_map", {}),
                "padding_side": self.tokenizer.padding_side,
            }
        )

    def _resolve_input_device(self, fallback: Any) -> Any:
        input_embeddings = self.model.get_input_embeddings()
        weight = None if input_embeddings is None else getattr(input_embeddings, "weight", None)
        candidates = (getattr(weight, "device", None), getattr(self.model, "device", None))
        for candidate in candidates:
            if candidate is None:
                continue
            resolved = self._torch.device(candidate)
            if resolved.type != "meta":
                return resolved
        return fallback

    def _resolve_logit_slice_argument(self) -> str | None:
        try:
            parameters = inspect.signature(self.model.forward).parameters
        except (TypeError, ValueError):
            return None
        for candidate in ("logits_to_keep", "num_logits_to_keep"):
            if candidate in parameters:
                return candidate
        return None

    def _observed_parameter_dtype(self) -> str:
        for parameter in self.model.parameters():
            if parameter.is_floating_point():
                return str(parameter.dtype).removeprefix("torch.")
        return "no-floating-parameters"

    def _resolve_choice_tokens(self) -> tuple[tuple[str, ...], tuple[int, ...]]:
        candidates = (CHOICE_TOKENS, (" A", " B", " C"))
        for candidate_set in candidates:
            tokenized = [
                self.tokenizer.encode(candidate, add_special_tokens=False)
                for candidate in candidate_set
            ]
            if all(len(token_ids) == 1 for token_ids in tokenized):
                ids = tuple(int(token_ids[0]) for token_ids in tokenized)
                if len(set(ids)) == len(ACTION_NAMES):
                    return tuple(candidate_set), ids
        raise RuntimeError("A/B/C are not distinct single tokens for the pinned tokenizer")

    def _format_messages(self, message_batches: Sequence[Sequence[Mapping[str, str]]]) -> list[str]:
        return [
            self._chat_template_owner.apply_chat_template(
                list(messages), tokenize=False, add_generation_prompt=True
            )
            for messages in message_batches
        ]

    def _encode_texts(self, texts: Sequence[str]) -> Mapping[str, Any]:
        if self.processor is None:
            encoded = self.tokenizer(
                list(texts), padding=True, return_tensors="pt", truncation=False
            )
        else:
            encoded = self.processor(
                text=list(texts), padding=True, return_tensors="pt", truncation=False
            )
        if "input_ids" not in encoded:
            raise RuntimeError("Tokenizer/processor did not return input_ids")
        return encoded

    def _forward_final_logits(self, encoded: Mapping[str, Any]) -> Any:
        model_kwargs = {**encoded, "use_cache": False}
        if self._logit_slice_argument is not None:
            model_kwargs[self._logit_slice_argument] = 1
            output = self.model(**model_kwargs)
        else:
            output_head = self.model.get_output_embeddings()
            if output_head is None:
                raise RuntimeError("Model has no output head for final-position logit slicing")
            head_was_sliced = False

            def slice_head_input(module: Any, args: tuple[Any, ...]) -> tuple[Any, ...]:
                del module
                nonlocal head_was_sliced
                if not args or getattr(args[0], "ndim", 0) != 3:
                    raise RuntimeError("Output head did not receive batched sequence states")
                head_was_sliced = True
                return (args[0][:, -1:, :], *args[1:])

            hook = output_head.register_forward_pre_hook(slice_head_input)
            try:
                output = self.model(**model_kwargs)
            finally:
                hook.remove()
            if not head_was_sliced:
                raise RuntimeError("Model bypassed its exposed output head; scoring aborted")
        logits = getattr(output, "logits", None)
        if logits is None or logits.ndim != 3 or int(logits.shape[1]) != 1:
            raise RuntimeError(
                "Model did not honor final-position-only logit scoring; scoring aborted"
            )
        return logits[:, 0, :]

    def score_messages(self, message_batches: Sequence[Sequence[Mapping[str, str]]]) -> np.ndarray:
        if not message_batches:
            return np.empty((0, len(ACTION_NAMES)), dtype=float)
        texts = self._format_messages(message_batches)
        encoded = self._encode_texts(texts)
        width = int(encoded["input_ids"].shape[1])
        if width > self.max_input_tokens:
            raise ValueError(
                f"Prompt has {width} tokens, exceeding the {self.max_input_tokens}-token limit"
            )
        device_encoded = {
            name: tensor.to(self.device) if hasattr(tensor, "to") else tensor
            for name, tensor in encoded.items()
            if tensor is not None
        }
        with self._torch.inference_mode():
            logits = self._forward_final_logits(device_encoded)
            candidate_logits = logits[:, list(self.choice_token_ids)].float()
            probabilities = self._torch.softmax(candidate_logits / self.logit_temperature, dim=-1)
        result = probabilities.cpu().numpy().astype(np.float64)
        result /= result.sum(axis=1, keepdims=True)
        return validate_probability_matrix(result, expected_rows=len(texts))
