"""Canonical runtime provenance for local model cache generation and replay.

The module deliberately imports PyTorch lazily.  Cache-only policy replay therefore
does not require the optional LLM dependencies unless provenance is being captured.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import re
from collections.abc import Mapping
from typing import Any

RUNTIME_PROVENANCE_SCHEMA_VERSION = "counterledger-local-runtime-v1"
RUNTIME_FINGERPRINT_CANONICALIZATION = "json-sort-keys-compact-utf8-v1"


def _canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _fingerprint_payload(provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Return the documented, order-independent runtime identity payload."""

    return {
        "schema_version": provenance.get("schema_version"),
        "python": provenance.get("python"),
        "packages": provenance.get("packages"),
        "platform": provenance.get("platform"),
        "cuda": provenance.get("cuda"),
        "deterministic_algorithms": provenance.get("deterministic_algorithms"),
    }


def runtime_fingerprint_sha256(provenance: Mapping[str, Any]) -> str:
    """Hash the canonical v1 runtime identity fields."""

    return _canonical_json_sha256(_fingerprint_payload(provenance))


def configure_deterministic_algorithms(torch_module: Any) -> None:
    """Apply the deterministic-algorithm mode used by local cache generation."""

    torch_module.use_deterministic_algorithms(True, warn_only=True)


def capture_runtime_provenance(*, capture_context: str) -> dict[str, Any]:
    """Capture package/device state without loading a model or tokenizer."""

    if capture_context not in {"cache_generation", "cache_only_finalization"}:
        raise ValueError(f"Unsupported runtime provenance context: {capture_context}")
    try:
        import torch
    except ImportError as error:  # pragma: no cover - depends on optional environment
        raise RuntimeError(
            "Runtime provenance capture requires the optional 'llm' dependencies"
        ) from error

    configure_deterministic_algorithms(torch)
    cuda_available = bool(torch.cuda.is_available())
    gpu_name: str | None = None
    gpu_capability: list[int] | None = None
    if cuda_available:
        device_index = int(torch.cuda.current_device())
        gpu_name = str(torch.cuda.get_device_name(device_index))
        capability = torch.cuda.get_device_capability(device_index)
        gpu_capability = [int(capability[0]), int(capability[1])]

    provenance: dict[str, Any] = {
        "schema_version": RUNTIME_PROVENANCE_SCHEMA_VERSION,
        "capture_context": capture_context,
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "packages": {
            "torch": str(torch.__version__),
            "transformers": importlib.metadata.version("transformers"),
            "safetensors": importlib.metadata.version("safetensors"),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "cuda": {
            "available": cuda_available,
            "runtime_version": (None if torch.version.cuda is None else str(torch.version.cuda)),
            "gpu_name": gpu_name,
            "compute_capability": gpu_capability,
        },
        "deterministic_algorithms": {
            "enabled": bool(torch.are_deterministic_algorithms_enabled()),
            "warn_only": bool(torch.is_deterministic_algorithms_warn_only_enabled()),
        },
        "fingerprint_canonicalization": RUNTIME_FINGERPRINT_CANONICALIZATION,
    }
    provenance["runtime_fingerprint_sha256"] = runtime_fingerprint_sha256(provenance)
    return provenance


def validate_runtime_provenance(provenance: Mapping[str, Any]) -> str:
    """Fail closed on incomplete or internally inconsistent runtime provenance."""

    required_keys = {
        "schema_version",
        "capture_context",
        "python",
        "packages",
        "platform",
        "cuda",
        "deterministic_algorithms",
        "fingerprint_canonicalization",
        "runtime_fingerprint_sha256",
    }
    if set(provenance) != required_keys:
        raise ValueError("Runtime provenance keys do not match the v1 schema")
    if provenance.get("schema_version") != RUNTIME_PROVENANCE_SCHEMA_VERSION:
        raise ValueError("Unsupported runtime provenance schema")
    if provenance.get("capture_context") not in {
        "cache_generation",
        "cache_only_finalization",
    }:
        raise ValueError("Invalid runtime provenance capture context")
    if provenance.get("fingerprint_canonicalization") != RUNTIME_FINGERPRINT_CANONICALIZATION:
        raise ValueError("Unsupported runtime fingerprint canonicalization")

    python = provenance.get("python")
    if not isinstance(python, Mapping) or set(python) != {"implementation", "version"}:
        raise ValueError("Runtime Python provenance is incomplete")
    if not all(isinstance(python.get(key), str) and python[key] for key in python):
        raise ValueError("Runtime Python provenance values must be non-empty strings")

    packages = provenance.get("packages")
    expected_packages = {"torch", "transformers", "safetensors"}
    if not isinstance(packages, Mapping) or set(packages) != expected_packages:
        raise ValueError("Runtime package provenance is incomplete")
    if not all(
        isinstance(packages.get(name), str) and packages[name] for name in expected_packages
    ):
        raise ValueError("Runtime package versions must be non-empty strings")

    runtime_platform = provenance.get("platform")
    expected_platform = {"system", "release", "machine"}
    if not isinstance(runtime_platform, Mapping) or set(runtime_platform) != expected_platform:
        raise ValueError("Runtime platform provenance is incomplete")
    if not all(
        isinstance(runtime_platform.get(name), str) and runtime_platform[name]
        for name in expected_platform
    ):
        raise ValueError("Runtime platform values must be non-empty strings")

    cuda = provenance.get("cuda")
    expected_cuda = {"available", "runtime_version", "gpu_name", "compute_capability"}
    if not isinstance(cuda, Mapping) or set(cuda) != expected_cuda:
        raise ValueError("Runtime CUDA provenance is incomplete")
    if not isinstance(cuda.get("available"), bool):
        raise ValueError("Runtime CUDA availability must be Boolean")
    runtime_version = cuda.get("runtime_version")
    if runtime_version is not None and (
        not isinstance(runtime_version, str) or not runtime_version
    ):
        raise ValueError("CUDA runtime version must be null or a non-empty string")
    if cuda["available"]:
        capability = cuda.get("compute_capability")
        if not isinstance(runtime_version, str) or not runtime_version:
            raise ValueError("CUDA runtime provenance requires a runtime version")
        if not isinstance(cuda.get("gpu_name"), str) or not cuda["gpu_name"]:
            raise ValueError("CUDA runtime provenance requires a GPU name")
        if (
            not isinstance(capability, list)
            or len(capability) != 2
            or not all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0
                for value in capability
            )
        ):
            raise ValueError("CUDA runtime provenance requires a two-integer capability")
    elif cuda.get("gpu_name") is not None or cuda.get("compute_capability") is not None:
        raise ValueError("CPU runtime provenance must not claim a CUDA GPU")

    deterministic = provenance.get("deterministic_algorithms")
    if not isinstance(deterministic, Mapping) or set(deterministic) != {
        "enabled",
        "warn_only",
    }:
        raise ValueError("Deterministic-algorithm provenance is incomplete")
    if deterministic != {"enabled": True, "warn_only": True}:
        raise ValueError("LLM runtime must enable deterministic algorithms in warn-only mode")

    fingerprint = provenance.get("runtime_fingerprint_sha256")
    if not isinstance(fingerprint, str) or re.fullmatch(r"[0-9a-f]{64}", fingerprint) is None:
        raise ValueError("Runtime fingerprint is not a lowercase SHA-256 digest")
    expected_fingerprint = runtime_fingerprint_sha256(provenance)
    if fingerprint != expected_fingerprint:
        raise ValueError("Runtime fingerprint does not reconcile with provenance fields")
    return fingerprint
