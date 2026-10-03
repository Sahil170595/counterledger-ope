"""Fail-closed A/B/C probability scoring through a pinned llama.cpp server.

The scorer deliberately uses the native llama.cpp endpoints rather than assuming
that an OpenAI-compatible endpoint exposes raw candidate probabilities.  The live
tokenizer and chat template are checked against a caller-supplied fingerprint before
inference.  Each request applies the same large logit bias to the three candidate
tokens, which preserves their relative logits while making exactly those tokens the
reported probability set.  No grammar mask is used: observed llama.cpp builds can
report post-grammar ``top_probs`` whose mass is less than one.

Every observation is scored under all three cyclic action-to-letter mappings.  The
three distributions are converted back to canonical action order before averaging,
so a model's position-independent A/B/C preference cannot determine the result.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from collections.abc import Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

import numpy as np

from contracts import ACTION_NAMES, validate_probability_matrix

CHOICE_LABELS: tuple[str, ...] = ("A", "B", "C")
ACTION_DESCRIPTIONS: tuple[str, ...] = (
    "maintain current treatment",
    "administer IV fluids",
    "escalate vasopressor",
)
CYCLIC_LABEL_TO_ACTION: tuple[tuple[int, ...], ...] = (
    (0, 1, 2),
    (1, 2, 0),
    (2, 0, 1),
)
CANDIDATE_LOGIT_BIAS = 100.0
SCORING_CONTRACT_VERSION = "llama-cpp-equal-bias-cyclic-abc-v2"
SCORER_DIAGNOSTICS_CONTRACT_VERSION = "llama-cpp-scoring-diagnostics-v1"
SCORER_DIAGNOSTIC_KEYS: tuple[str, ...] = (
    "mapping_attempts",
    "mapping_successes",
    "mapping_failures",
    "request_attempts",
    "request_responses",
    "request_successes",
    "request_failures",
    "template_request_attempts",
    "template_request_responses",
    "template_request_successes",
    "template_request_failures",
    "completion_request_attempts",
    "completion_request_responses",
    "completion_request_successes",
    "completion_request_failures",
    "transport_failures",
    "protocol_failures",
    "truncation_failures",
    "malformed_response_failures",
    "interrupted_failures",
    "unexpected_failures",
    "fallback_count",
)

_CANONICAL_CANDIDATE_BLOCK = "\n".join(
    f"{label} = {description}"
    for label, description in zip(CHOICE_LABELS, ACTION_DESCRIPTIONS, strict=True)
)
_SHA256_HEX_LENGTH = 64
_PROBABILITY_SUM_TOLERANCE = 1e-6
_MAX_WORKERS_LIMIT = 64


class LlamaCppError(RuntimeError):
    """Base class for fail-closed llama.cpp client errors."""


class LlamaCppTransportError(LlamaCppError):
    """The server could not be reached or returned a non-JSON HTTP response."""


class LlamaCppProtocolError(LlamaCppError):
    """The server response did not satisfy the scoring contract."""


class LlamaCppTruncationError(LlamaCppProtocolError):
    """The server explicitly reported that a scoring prompt was truncated."""


class LlamaCppFingerprintError(LlamaCppProtocolError):
    """The running backend or model did not match its pinned fingerprint."""


def _is_sha256(value: str) -> bool:
    return len(value) == _SHA256_HEX_LENGTH and all(
        character in "0123456789abcdef" for character in value
    )


def _validated_max_workers(value: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= _MAX_WORKERS_LIMIT
    ):
        raise ValueError(f"max_workers must be an integer from 1 to {_MAX_WORKERS_LIMIT}")
    return value


def _canonical_absolute_path(value: str | Path, *, field: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise ValueError(f"{field} must be an absolute filesystem path")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{field} must be an absolute filesystem path")
    return path.resolve(strict=False)


@dataclass(frozen=True)
class LlamaCppModelFingerprint:
    """Immutable identity for one llama.cpp/GGUF scoring environment.

    ``backend_version`` must exactly match the server's ``/props`` ``build_info``.
    llama.cpp embeds its source revision in that value; requiring
    ``backend_repository_revision`` to be a substring makes the relationship
    explicit in the cache identity.  The scorer requires an expected absolute GGUF
    path and verifies exact canonical equality with the path reported by the server.
    The Stage 0 runner hashes that exact served file.
    """

    backend_version: str
    backend_repository_revision: str
    model_repository: str
    model_repository_revision: str
    gguf_filename: str
    gguf_sha256: str
    quantization: str
    chat_template_sha256: str
    choice_token_ids: tuple[int, int, int]

    def __post_init__(self) -> None:
        text_fields = {
            "backend_version": self.backend_version,
            "backend_repository_revision": self.backend_repository_revision,
            "model_repository": self.model_repository,
            "model_repository_revision": self.model_repository_revision,
            "gguf_filename": self.gguf_filename,
            "quantization": self.quantization,
        }
        for field, value in text_fields.items():
            if not isinstance(value, str) or not value.strip() or value != value.strip():
                raise ValueError(f"{field} must be a non-empty, trimmed string")
        if self.backend_repository_revision not in self.backend_version:
            raise ValueError("backend_repository_revision must occur in the pinned backend_version")
        if "/" in self.gguf_filename or "\\" in self.gguf_filename:
            raise ValueError("gguf_filename must be a basename, not a path")
        if not self.gguf_filename.lower().endswith(".gguf"):
            raise ValueError("gguf_filename must name a GGUF file")
        if not _is_sha256(self.gguf_sha256):
            raise ValueError("gguf_sha256 must be a lowercase SHA-256 digest")
        if not _is_sha256(self.chat_template_sha256):
            raise ValueError("chat_template_sha256 must be a lowercase SHA-256 digest")
        if (
            len(self.choice_token_ids) != len(CHOICE_LABELS)
            or any(
                isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0
                for token_id in self.choice_token_ids
            )
            or len(set(self.choice_token_ids)) != len(CHOICE_LABELS)
        ):
            raise ValueError("choice_token_ids must contain three distinct non-negative ints")


@dataclass(frozen=True)
class UnrestrictedAuditRecord:
    """Reporting-only natural next-token diagnostics for one canonical prompt."""

    row_index: int
    generated_token_id: int
    generated_token: str
    generated_is_exact_choice: bool
    generated_choice_label: str | None
    candidate_logprobs: tuple[float, float, float]
    candidate_ranks: tuple[int, int, int]
    candidate_mass: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "row_index": self.row_index,
            "generated_token_id": self.generated_token_id,
            "generated_token": self.generated_token,
            "generated_is_exact_choice": self.generated_is_exact_choice,
            "generated_choice_label": self.generated_choice_label,
            "candidate_logprobs": dict(zip(CHOICE_LABELS, self.candidate_logprobs, strict=True)),
            "candidate_ranks": dict(zip(CHOICE_LABELS, self.candidate_ranks, strict=True)),
            "candidate_mass": self.candidate_mass,
        }


def cache_identity_payload(
    fingerprint: LlamaCppModelFingerprint,
    *,
    cache_prompt: bool = False,
    max_workers: int = 1,
) -> dict[str, Any]:
    """Return the complete canonical identity bound into downstream cache keys."""

    if not isinstance(cache_prompt, bool):
        raise ValueError("cache_prompt must be a bool")
    validated_workers = _validated_max_workers(max_workers)

    return {
        "scoring_contract_version": SCORING_CONTRACT_VERSION,
        "backend": {
            "name": "llama.cpp",
            "version": fingerprint.backend_version,
            "repository_revision": fingerprint.backend_repository_revision,
        },
        "model": {
            "repository": fingerprint.model_repository,
            "repository_revision": fingerprint.model_repository_revision,
            "gguf_filename": fingerprint.gguf_filename,
            "gguf_sha256": fingerprint.gguf_sha256,
            "quantization": fingerprint.quantization,
        },
        "prompt_rendering": {
            "endpoint": "/apply-template",
            "add_generation_prompt": True,
            "chat_template_sha256": fingerprint.chat_template_sha256,
            "choice_labels": list(CHOICE_LABELS),
            "choice_token_ids": list(fingerprint.choice_token_ids),
        },
        "scoring": {
            "endpoint": "/completion",
            "n_predict": 1,
            "candidate_logit_bias": CANDIDATE_LOGIT_BIAS,
            "post_sampling_probs": True,
            "candidate_probability_sum_required": True,
            "cyclic_label_to_action": [list(mapping) for mapping in CYCLIC_LABEL_TO_ACTION],
            "cache_prompt": cache_prompt,
            "max_workers": validated_workers,
        },
    }


def cache_identity_sha256(
    fingerprint: LlamaCppModelFingerprint,
    *,
    cache_prompt: bool = False,
    max_workers: int = 1,
) -> str:
    """Hash the backend, model file, template, tokenizer, and scoring contract."""

    encoded = json.dumps(
        cache_identity_payload(
            fingerprint,
            cache_prompt=cache_prompt,
            max_workers=max_workers,
        ),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class JsonTransport(Protocol):
    """Minimal injectable transport used by the scorer and unit tests."""

    def request_json(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        """Issue one request and return a decoded JSON object."""


class _RejectRedirects(HTTPRedirectHandler):
    def redirect_request(  # type: ignore[override]
        self,
        req: Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        del req, fp, code, msg, headers, newurl
        return None


class UrllibJsonTransport:
    """Small bounded JSON transport with redirects disabled."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 120.0,
        max_response_bytes: int = 8 * 1024 * 1024,
        api_key: str | None = None,
    ) -> None:
        parsed = urlsplit(base_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
        ):
            raise ValueError("base_url must be an origin-only HTTP(S) URL without credentials")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        if isinstance(max_response_bytes, bool) or max_response_bytes <= 0:
            raise ValueError("max_response_bytes must be positive")
        if api_key is not None and (not api_key or "\r" in api_key or "\n" in api_key):
            raise ValueError("api_key must be non-empty and cannot contain newlines")
        self.base_url = f"{parsed.scheme}://{parsed.netloc}"
        self.timeout_seconds = float(timeout_seconds)
        self.max_response_bytes = int(max_response_bytes)
        self.api_key = api_key
        self._opener = build_opener(_RejectRedirects())

    def request_json(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        normalized_method = method.upper()
        if normalized_method not in {"GET", "POST"}:
            raise ValueError("Only GET and POST are supported")
        if not path.startswith("/") or "?" in path or "#" in path or ".." in path:
            raise ValueError("path must be an absolute endpoint path without traversal/query")
        if normalized_method == "GET" and payload is not None:
            raise ValueError("GET requests cannot contain a JSON payload")

        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(
                dict(payload),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if self.api_key is not None:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(
            self.base_url + path,
            data=body,
            headers=headers,
            method=normalized_method,
        )
        try:
            with self._opener.open(request, timeout=self.timeout_seconds) as response:
                status = int(response.getcode())
                if not 200 <= status < 300:
                    raise LlamaCppTransportError(
                        f"{normalized_method} {path} returned HTTP {status}"
                    )
                content_length = response.headers.get("Content-Length")
                if content_length is not None:
                    try:
                        declared_length = int(content_length)
                    except ValueError as error:
                        raise LlamaCppTransportError(
                            f"{normalized_method} {path} returned invalid Content-Length"
                        ) from error
                    if declared_length > self.max_response_bytes:
                        raise LlamaCppTransportError(
                            f"{normalized_method} {path} response exceeds byte limit"
                        )
                content_type = response.headers.get_content_type()
                if content_type != "application/json" and not content_type.endswith("+json"):
                    raise LlamaCppTransportError(f"{normalized_method} {path} did not return JSON")
                raw = response.read(self.max_response_bytes + 1)
        except HTTPError as error:
            raise LlamaCppTransportError(
                f"{normalized_method} {path} returned HTTP {error.code}"
            ) from error
        except TimeoutError as error:
            raise LlamaCppTransportError(f"{normalized_method} {path} timed out") from error
        except URLError as error:
            raise LlamaCppTransportError(
                f"{normalized_method} {path} could not reach llama.cpp"
            ) from error

        if len(raw) > self.max_response_bytes:
            raise LlamaCppTransportError(f"{normalized_method} {path} response exceeds byte limit")
        try:
            decoded = json.loads(raw.decode("utf-8", errors="strict"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise LlamaCppTransportError(
                f"{normalized_method} {path} returned invalid JSON"
            ) from error
        if not isinstance(decoded, Mapping):
            raise LlamaCppTransportError(f"{normalized_method} {path} must return a JSON object")
        return decoded


def _require_string(payload: Mapping[str, Any], field: str, *, context: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise LlamaCppProtocolError(f"{context} must contain non-empty string {field!r}")
    return value


def _server_model_basename(model_path: str) -> str:
    return model_path.replace("\\", "/").rsplit("/", maxsplit=1)[-1]


def _validated_messages(
    messages: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    if isinstance(messages, (str, bytes)) or not messages:
        raise ValueError("messages must be a non-empty sequence")
    normalized: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping) or set(message) != {"role", "content"}:
            raise ValueError(f"message {index} must contain only role and content")
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"message {index} has an unsupported role")
        if not isinstance(content, str) or not content:
            raise ValueError(f"message {index} content must be a non-empty string")
        normalized.append({"role": role, "content": content})
    if normalized[-1]["role"] != "user":
        raise ValueError("the final message must have role 'user'")
    occurrences = sum(
        message["content"].count(_CANONICAL_CANDIDATE_BLOCK) for message in normalized
    )
    if occurrences != 1 or _CANONICAL_CANDIDATE_BLOCK not in normalized[-1]["content"]:
        raise ValueError("messages must contain exactly one canonical candidate block at the end")
    return normalized


def cyclic_choice_messages(
    messages: Sequence[Mapping[str, str]], mapping_index: int
) -> list[dict[str, str]]:
    """Return a copy with one of the three cyclic label/action mappings."""

    if isinstance(mapping_index, bool) or not 0 <= mapping_index < len(CYCLIC_LABEL_TO_ACTION):
        raise ValueError("mapping_index must be 0, 1, or 2")
    normalized = _validated_messages(messages)
    label_to_action = CYCLIC_LABEL_TO_ACTION[mapping_index]
    replacement = "\n".join(
        f"{label} = {ACTION_DESCRIPTIONS[action_index]}"
        for label, action_index in zip(CHOICE_LABELS, label_to_action, strict=True)
    )
    normalized[-1]["content"] = normalized[-1]["content"].replace(
        _CANONICAL_CANDIDATE_BLOCK, replacement, 1
    )
    return normalized


class LlamaCppChoiceScorer:
    """Cyclically debiased scorer with bounded-repeatability runtime controls."""

    choice_tokens = CHOICE_LABELS
    fallback_capable = False

    def __init__(
        self,
        base_url: str,
        fingerprint: LlamaCppModelFingerprint,
        *,
        timeout_seconds: float = 120.0,
        max_response_bytes: int = 8 * 1024 * 1024,
        api_key: str | None = None,
        transport: JsonTransport | None = None,
        preflight: bool = True,
        max_workers: int = 1,
        cache_prompt: bool = False,
        expected_model_path: str | Path,
    ) -> None:
        if not isinstance(cache_prompt, bool):
            raise ValueError("cache_prompt must be a bool")
        self.fingerprint = fingerprint
        self.max_workers = _validated_max_workers(max_workers)
        self.cache_prompt = cache_prompt
        self.expected_model_path = _canonical_absolute_path(
            expected_model_path,
            field="expected_model_path",
        )
        self.transport = transport or UrllibJsonTransport(
            base_url,
            timeout_seconds=timeout_seconds,
            max_response_bytes=max_response_bytes,
            api_key=api_key,
        )
        self._preflight_complete = False
        self._served_model_path: Path | None = None
        self._diagnostic_lock = threading.Lock()
        self._diagnostics = {key: 0 for key in SCORER_DIAGNOSTIC_KEYS}
        if preflight:
            self.preflight()

    @property
    def cache_identity(self) -> str:
        return cache_identity_sha256(
            self.fingerprint,
            cache_prompt=self.cache_prompt,
            max_workers=self.max_workers,
        )

    @property
    def semantic_fingerprint(self) -> str:
        """Alias used by policy caches to bind semantic scoring configuration."""

        return self.cache_identity

    @property
    def execution_metadata(self) -> dict[str, Any]:
        """Return runtime settings that are also bound into semantic identity."""

        return {"backend": "llama.cpp-http", "max_workers": self.max_workers}

    def diagnostic_snapshot(self) -> dict[str, int]:
        """Return an atomic copy of cumulative Stage 1 scoring diagnostics.

        Preflight and unrestricted-audit traffic is intentionally excluded.  The
        counters cover only calls made by :meth:`score_messages`, so a generator can
        take before/after snapshots and persist exact deltas across resumable chunks.
        """

        with self._diagnostic_lock:
            return dict(self._diagnostics)

    def _increment_diagnostics(self, *names: str) -> None:
        with self._diagnostic_lock:
            for name in names:
                self._diagnostics[name] += 1

    def _record_scoring_request_failure(self, kind: str, error: BaseException) -> None:
        names = ["request_failures", f"{kind}_request_failures"]
        if isinstance(error, LlamaCppTruncationError):
            names.extend(("protocol_failures", "truncation_failures"))
        elif isinstance(error, LlamaCppProtocolError):
            names.extend(("protocol_failures", "malformed_response_failures"))
        elif isinstance(
            error,
            (LlamaCppTransportError, ConnectionError, TimeoutError, OSError),
        ):
            names.append("transport_failures")
        elif isinstance(error, (KeyboardInterrupt, SystemExit)):
            names.append("interrupted_failures")
        else:
            names.append("unexpected_failures")
        self._increment_diagnostics(*names)

    def _scoring_request(
        self,
        kind: str,
        path: str,
        payload: Mapping[str, Any],
        parser: Any,
    ) -> Any:
        """Issue and parse one instrumented scoring request, failing closed."""

        self._increment_diagnostics("request_attempts", f"{kind}_request_attempts")
        try:
            response = self.transport.request_json("POST", path, payload)
        except BaseException as error:
            self._record_scoring_request_failure(kind, error)
            raise
        self._increment_diagnostics("request_responses", f"{kind}_request_responses")
        try:
            parsed = parser(response)
        except BaseException as error:
            self._record_scoring_request_failure(kind, error)
            raise
        self._increment_diagnostics("request_successes", f"{kind}_request_successes")
        return parsed

    @property
    def served_model_path(self) -> Path | None:
        """Return the canonical server-reported GGUF path after successful preflight."""

        return self._served_model_path

    def preflight(self) -> None:
        """Verify server build, model file, template hash, and live label token IDs."""

        props = self.transport.request_json("GET", "/props")
        build_info = _require_string(props, "build_info", context="/props response")
        if build_info != self.fingerprint.backend_version:
            raise LlamaCppFingerprintError(
                "Running llama.cpp build_info does not match the pinned backend version"
            )
        model_path = _require_string(props, "model_path", context="/props response")
        try:
            served_model_path = _canonical_absolute_path(
                model_path,
                field="/props model_path",
            )
        except ValueError as error:
            raise LlamaCppFingerprintError(
                "Running llama.cpp model_path is not an absolute canonical path"
            ) from error
        if served_model_path != self.expected_model_path:
            raise LlamaCppFingerprintError(
                "Running llama.cpp model path does not exactly match the expected GGUF path"
            )
        if _server_model_basename(model_path) != self.fingerprint.gguf_filename:
            raise LlamaCppFingerprintError(
                "Running llama.cpp model filename does not match the pinned GGUF"
            )
        chat_template = _require_string(props, "chat_template", context="/props response")
        live_template_sha256 = hashlib.sha256(chat_template.encode("utf-8")).hexdigest()
        if live_template_sha256 != self.fingerprint.chat_template_sha256:
            raise LlamaCppFingerprintError(
                "Running llama.cpp chat template does not match the pinned hash"
            )

        live_ids: list[int] = []
        for label in CHOICE_LABELS:
            tokenized = self.transport.request_json(
                "POST",
                "/tokenize",
                {
                    "content": label,
                    "add_special": False,
                    "parse_special": False,
                    "with_pieces": True,
                },
            )
            tokens = tokenized.get("tokens")
            if not isinstance(tokens, list) or len(tokens) != 1:
                raise LlamaCppFingerprintError(
                    f"Choice label {label!r} is not exactly one live tokenizer token"
                )
            token = tokens[0]
            if isinstance(token, bool):
                raise LlamaCppFingerprintError("Tokenizer returned an invalid boolean token ID")
            if isinstance(token, int):
                token_id = token
            elif isinstance(token, Mapping):
                token_id = token.get("id")
                piece = token.get("piece")
                if piece is not None and piece != label:
                    raise LlamaCppFingerprintError(
                        f"Tokenizer piece for {label!r} is not the exact canonical label"
                    )
            else:
                raise LlamaCppFingerprintError("Tokenizer returned an unsupported token record")
            if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
                raise LlamaCppFingerprintError("Tokenizer returned an invalid token ID")
            live_ids.append(token_id)
        if len(set(live_ids)) != len(CHOICE_LABELS):
            raise LlamaCppFingerprintError("A/B/C are not distinct live tokenizer tokens")
        if tuple(live_ids) != self.fingerprint.choice_token_ids:
            raise LlamaCppFingerprintError(
                "Live A/B/C token IDs do not match the pinned tokenizer fingerprint"
            )
        self._served_model_path = served_model_path
        self._preflight_complete = True

    def _render_prompt(self, messages: Sequence[Mapping[str, str]]) -> str:
        response = self.transport.request_json(
            "POST",
            "/apply-template",
            {"messages": list(messages), "add_generation_prompt": True},
        )
        return _require_string(response, "prompt", context="/apply-template response")

    def _completion_payload(self, prompt: str) -> dict[str, Any]:
        return {
            "prompt": prompt,
            "n_predict": 1,
            "stream": False,
            "return_tokens": True,
            "cache_prompt": self.cache_prompt,
            "seed": 0,
            "temperature": 1.0,
            "dynatemp_range": 0.0,
            "top_k": 0,
            "top_p": 1.0,
            "min_p": 0.0,
            "typical_p": 1.0,
            "xtc_probability": 0.0,
            "repeat_last_n": 0,
            "repeat_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "dry_multiplier": 0.0,
            "mirostat": 0,
            "samplers": ["top_k", "typ_p", "top_p", "min_p", "xtc", "temperature"],
            "logit_bias": [
                [token_id, CANDIDATE_LOGIT_BIAS] for token_id in self.fingerprint.choice_token_ids
            ],
            "n_probs": len(CHOICE_LABELS),
            "min_keep": len(CHOICE_LABELS),
            "post_sampling_probs": True,
        }

    def _parse_label_probabilities(self, response: Mapping[str, Any]) -> np.ndarray:
        if response.get("truncated") is True:
            raise LlamaCppTruncationError("llama.cpp truncated the scoring prompt")
        tokens = response.get("tokens")
        if (
            not isinstance(tokens, list)
            or len(tokens) != 1
            or isinstance(tokens[0], bool)
            or tokens[0] not in self.fingerprint.choice_token_ids
        ):
            raise LlamaCppProtocolError("Completion must generate exactly one pinned A/B/C token")
        content = response.get("content")
        generated_index = self.fingerprint.choice_token_ids.index(tokens[0])
        if content != CHOICE_LABELS[generated_index]:
            raise LlamaCppProtocolError("Completion text does not match its returned token ID")

        probability_keys = {key for key in ("completion_probabilities", "probs") if key in response}
        if len(probability_keys) != 1:
            raise LlamaCppProtocolError(
                "Completion must contain exactly one recognized probability field"
            )
        # b7524 and current llama.cpp call this ``completion_probabilities``.
        # ``probs`` is retained solely as a fail-closed legacy wire-format alias.
        steps = response[next(iter(probability_keys))]
        if not isinstance(steps, list) or len(steps) != 1 or not isinstance(steps[0], Mapping):
            raise LlamaCppProtocolError("Completion must contain one probability step")
        top_probabilities = steps[0].get("top_probs")
        if not isinstance(top_probabilities, list):
            raise LlamaCppProtocolError(
                "Completion must return post-sampling top_probs, not raw top_logprobs"
            )

        by_id: dict[int, float] = {}
        expected_ids = set(self.fingerprint.choice_token_ids)
        for entry in top_probabilities:
            if not isinstance(entry, Mapping):
                raise LlamaCppProtocolError("top_probs entries must be JSON objects")
            token_id = entry.get("id")
            probability = entry.get("prob")
            if (
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or isinstance(probability, bool)
                or not isinstance(probability, (int, float))
                or not math.isfinite(float(probability))
                or float(probability) < 0.0
                or float(probability) > 1.0
                or token_id in by_id
            ):
                raise LlamaCppProtocolError("Invalid or duplicate top_probs entry")
            by_id[token_id] = float(probability)
        if set(by_id) != expected_ids:
            raise LlamaCppProtocolError(
                "top_probs token IDs must be exactly the pinned A/B/C token IDs"
            )
        values = np.asarray(
            [by_id[token_id] for token_id in self.fingerprint.choice_token_ids],
            dtype=np.float64,
        )
        total = float(values.sum())
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=_PROBABILITY_SUM_TOLERANCE):
            raise LlamaCppProtocolError(
                "A/B/C top_probs must contain the complete normalized probability mass"
            )
        values /= total
        return values

    def _unrestricted_completion_payload(self, prompt: str) -> dict[str, Any]:
        payload = self._completion_payload(prompt)
        del payload["logit_bias"]
        payload.update(
            {
                "n_probs": 50,
                "min_keep": 1,
                "post_sampling_probs": False,
            }
        )
        return payload

    def _parse_unrestricted_audit(
        self, response: Mapping[str, Any], *, row_index: int
    ) -> UnrestrictedAuditRecord:
        if response.get("truncated") is True:
            raise LlamaCppProtocolError("llama.cpp truncated an unrestricted audit prompt")
        tokens = response.get("tokens")
        content = response.get("content")
        if (
            not isinstance(tokens, list)
            or len(tokens) != 1
            or isinstance(tokens[0], bool)
            or not isinstance(tokens[0], int)
            or tokens[0] < 0
            or not isinstance(content, str)
        ):
            raise LlamaCppProtocolError(
                "Unrestricted audit must return one natural token ID and token text"
            )

        steps = response.get("completion_probabilities")
        if not isinstance(steps, list) or len(steps) != 1 or not isinstance(steps[0], Mapping):
            raise LlamaCppProtocolError(
                "Unrestricted audit must contain one completion_probabilities step"
            )
        step = steps[0]
        if step.get("id") != tokens[0] or step.get("token") != content:
            raise LlamaCppProtocolError(
                "Unrestricted audit token metadata disagrees with the completion"
            )
        top_logprobs = step.get("top_logprobs")
        if not isinstance(top_logprobs, list) or not 1 <= len(top_logprobs) <= 50:
            raise LlamaCppProtocolError(
                "Unrestricted audit must return between 1 and 50 top_logprobs"
            )

        entries: dict[int, tuple[int, float, str]] = {}
        previous_logprob = math.inf
        for rank, entry in enumerate(top_logprobs, start=1):
            if not isinstance(entry, Mapping):
                raise LlamaCppProtocolError("Unrestricted top_logprobs entries must be objects")
            token_id = entry.get("id")
            token = entry.get("token")
            logprob = entry.get("logprob")
            if (
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or token_id < 0
                or not isinstance(token, str)
                or isinstance(logprob, bool)
                or not isinstance(logprob, (int, float))
                or not math.isfinite(float(logprob))
                or float(logprob) > 0.0
                or token_id in entries
                or float(logprob) > previous_logprob + 1e-12
            ):
                raise LlamaCppProtocolError("Invalid, duplicate, or unsorted top_logprobs entry")
            previous_logprob = float(logprob)
            entries[token_id] = (rank, float(logprob), token)

        missing = set(self.fingerprint.choice_token_ids) - set(entries)
        if missing:
            raise LlamaCppProtocolError("Unrestricted top-50 must contain every pinned A/B/C token")
        candidate_ranks: list[int] = []
        candidate_logprobs: list[float] = []
        for label, token_id in zip(CHOICE_LABELS, self.fingerprint.choice_token_ids, strict=True):
            rank, logprob, token = entries[token_id]
            if token != label:
                raise LlamaCppProtocolError(
                    "Unrestricted candidate token text does not match its pinned label"
                )
            candidate_ranks.append(rank)
            candidate_logprobs.append(logprob)
        candidate_mass = math.fsum(math.exp(value) for value in candidate_logprobs)
        if not math.isfinite(candidate_mass) or not 0.0 < candidate_mass <= 1.0 + 1e-9:
            raise LlamaCppProtocolError("Unrestricted candidate mass is invalid")

        generated_choice_label = None
        if tokens[0] in self.fingerprint.choice_token_ids:
            candidate_index = self.fingerprint.choice_token_ids.index(tokens[0])
            if content == CHOICE_LABELS[candidate_index]:
                generated_choice_label = CHOICE_LABELS[candidate_index]
        return UnrestrictedAuditRecord(
            row_index=row_index,
            generated_token_id=tokens[0],
            generated_token=content,
            generated_is_exact_choice=generated_choice_label is not None,
            generated_choice_label=generated_choice_label,
            candidate_logprobs=tuple(candidate_logprobs),
            candidate_ranks=tuple(candidate_ranks),
            candidate_mass=candidate_mass,
        )

    def audit_unrestricted_messages(
        self, message_batches: Sequence[Sequence[Mapping[str, str]]]
    ) -> tuple[UnrestrictedAuditRecord, ...]:
        """Audit natural canonical next tokens without changing policy probabilities.

        This intentionally runs sequentially over a small probe set.  It does not
        participate in the semantic scoring fingerprint or the three-mapping policy
        average.
        """

        if not self._preflight_complete:
            raise LlamaCppFingerprintError("preflight must succeed before any audit request")
        records: list[UnrestrictedAuditRecord] = []
        normalized_batches = [_validated_messages(messages) for messages in message_batches]
        for row_index, messages in enumerate(normalized_batches):
            canonical_messages = cyclic_choice_messages(messages, 0)
            prompt = self._render_prompt(canonical_messages)
            response = self.transport.request_json(
                "POST", "/completion", self._unrestricted_completion_payload(prompt)
            )
            records.append(self._parse_unrestricted_audit(response, row_index=row_index))
        return tuple(records)

    def _score_one_mapping(self, messages: Sequence[Mapping[str, str]]) -> np.ndarray:
        self._increment_diagnostics("mapping_attempts")
        try:
            prompt = self._scoring_request(
                "template",
                "/apply-template",
                {"messages": list(messages), "add_generation_prompt": True},
                lambda response: _require_string(
                    response,
                    "prompt",
                    context="/apply-template response",
                ),
            )
            probabilities = self._scoring_request(
                "completion",
                "/completion",
                self._completion_payload(prompt),
                self._parse_label_probabilities,
            )
        except BaseException:
            self._increment_diagnostics("mapping_failures")
            raise
        self._increment_diagnostics("mapping_successes")
        return probabilities

    def _score_tasks_parallel(
        self,
        normalized_batches: Sequence[Sequence[Mapping[str, str]]],
        scores: np.ndarray,
    ) -> None:
        """Run a bounded request window and store results by deterministic indices."""

        tasks = (
            (row_index, mapping_index, cyclic_choice_messages(messages, mapping_index))
            for row_index, messages in enumerate(normalized_batches)
            for mapping_index in range(len(CYCLIC_LABEL_TO_ACTION))
        )
        executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="llama-cpp-score",
        )
        pending: dict[Future[np.ndarray], tuple[int, int]] = {}

        def submit_next() -> bool:
            try:
                row_index, mapping_index, variant = next(tasks)
            except StopIteration:
                return False
            pending[executor.submit(self._score_one_mapping, variant)] = (
                row_index,
                mapping_index,
            )
            return True

        try:
            for _ in range(self.max_workers):
                if not submit_next():
                    break
            while pending:
                completed, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
                for future in completed:
                    row_index, mapping_index = pending.pop(future)
                    scores[row_index, mapping_index, :] = future.result()
                    submit_next()
        except BaseException:
            for future in pending:
                future.cancel()
            raise
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

    def score_messages(self, message_batches: Sequence[Sequence[Mapping[str, str]]]) -> np.ndarray:
        """Return cyclically debiased probabilities in canonical action order."""

        if not self._preflight_complete:
            raise LlamaCppFingerprintError("preflight must succeed before any scoring request")
        if not message_batches:
            return np.empty((0, len(ACTION_NAMES)), dtype=np.float64)

        # Validate the entire batch before the first inference request.  Parallel
        # completions may finish in any order; fixed array indices preserve caller
        # order and cyclic-mapping order deterministically.
        normalized_batches = [_validated_messages(messages) for messages in message_batches]
        label_scores = np.empty(
            (
                len(normalized_batches),
                len(CYCLIC_LABEL_TO_ACTION),
                len(CHOICE_LABELS),
            ),
            dtype=np.float64,
        )
        if self.max_workers == 1:
            for row_index, messages in enumerate(normalized_batches):
                for mapping_index in range(len(CYCLIC_LABEL_TO_ACTION)):
                    variant = cyclic_choice_messages(messages, mapping_index)
                    label_scores[row_index, mapping_index, :] = self._score_one_mapping(variant)
        else:
            self._score_tasks_parallel(normalized_batches, label_scores)

        results: list[np.ndarray] = []
        for row_index in range(len(normalized_batches)):
            action_sum = np.zeros(len(ACTION_NAMES), dtype=np.float64)
            for mapping_index, label_to_action in enumerate(CYCLIC_LABEL_TO_ACTION):
                action_probabilities = np.zeros(len(ACTION_NAMES), dtype=np.float64)
                for label_index, action_index in enumerate(label_to_action):
                    action_probabilities[action_index] = label_scores[
                        row_index, mapping_index, label_index
                    ]
                action_sum += action_probabilities
            results.append(action_sum / len(CYCLIC_LABEL_TO_ACTION))
        return validate_probability_matrix(np.vstack(results), expected_rows=len(results))
