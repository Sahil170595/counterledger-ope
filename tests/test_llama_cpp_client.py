from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from llama_cpp_client import (
    ACTION_DESCRIPTIONS,
    CANDIDATE_LOGIT_BIAS,
    CHOICE_LABELS,
    CYCLIC_LABEL_TO_ACTION,
    LlamaCppChoiceScorer,
    LlamaCppFingerprintError,
    LlamaCppModelFingerprint,
    LlamaCppProtocolError,
    LlamaCppTransportError,
    LlamaCppTruncationError,
    UrllibJsonTransport,
    cache_identity_payload,
    cache_identity_sha256,
    cyclic_choice_messages,
)

CHAT_TEMPLATE = "{% for message in messages %}{{ message.role }}: {{ message.content }}{% endfor %}"
TOKEN_IDS = {"A": 101, "B": 102, "C": 103}
MODEL_PATH = (Path.cwd() / "mock-models" / "model-Q6_K.gguf").resolve()


def _fingerprint() -> LlamaCppModelFingerprint:
    return LlamaCppModelFingerprint(
        backend_version="b7000-deadbeef",
        backend_repository_revision="deadbeef",
        model_repository="example/model-GGUF",
        model_repository_revision="a" * 40,
        gguf_filename="model-Q6_K.gguf",
        gguf_sha256="b" * 64,
        quantization="Q6_K",
        chat_template_sha256=hashlib.sha256(CHAT_TEMPLATE.encode()).hexdigest(),
        choice_token_ids=tuple(TOKEN_IDS.values()),
    )


def _messages() -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "Select one synthetic benchmark action."},
        {
            "role": "user",
            "content": (
                "CANDIDATES\n"
                "A = maintain current treatment\n"
                "B = administer IV fluids\n"
                "C = escalate vasopressor\n"
                "Answer with exactly A, B, or C.\nCHOICE:"
            ),
        },
    ]


def _completion(probabilities: list[float]) -> dict[str, Any]:
    maximum = int(np.argmax(probabilities))
    # Deliberately scramble top_probs order: production parsing must bind by token ID.
    entries = [
        {"id": TOKEN_IDS[label], "token": label, "prob": probabilities[index]}
        for index, label in enumerate(CHOICE_LABELS)
    ]
    return {
        "content": CHOICE_LABELS[maximum],
        "tokens": [TOKEN_IDS[CHOICE_LABELS[maximum]]],
        "truncated": False,
        "completion_probabilities": [{"top_probs": [entries[2], entries[0], entries[1]]}],
    }


def _unrestricted_completion() -> dict[str, Any]:
    entries = [
        {"id": 999, "token": "\n", "logprob": -0.1},
        {"id": TOKEN_IDS["A"], "token": "A", "logprob": -1.0},
        {"id": TOKEN_IDS["B"], "token": "B", "logprob": -2.0},
        {"id": TOKEN_IDS["C"], "token": "C", "logprob": -3.0},
    ]
    return {
        "content": "\n",
        "tokens": [999],
        "truncated": False,
        "completion_probabilities": [
            {
                "id": 999,
                "token": "\n",
                "logprob": -0.1,
                "top_logprobs": entries,
            }
        ],
    }


class _FakeTransport:
    def __init__(
        self,
        completions: list[Mapping[str, Any]] | None = None,
        *,
        model_path: Path = MODEL_PATH,
    ) -> None:
        self.completions = list(completions or [])
        self.model_path = model_path
        self.calls: list[tuple[str, str, Mapping[str, Any] | None]] = []
        self.rendered_messages: list[list[dict[str, str]]] = []

    def request_json(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        self.calls.append((method, path, payload))
        if path == "/props":
            return {
                "build_info": "b7000-deadbeef",
                "model_path": str(self.model_path),
                "chat_template": CHAT_TEMPLATE,
            }
        if path == "/tokenize":
            assert payload is not None
            label = payload["content"]
            return {"tokens": [{"id": TOKEN_IDS[label], "piece": label}]}
        if path == "/apply-template":
            assert payload is not None
            messages = payload["messages"]
            self.rendered_messages.append(messages)
            return {"prompt": f"rendered prompt {len(self.rendered_messages)}"}
        if path == "/completion":
            if not self.completions:
                raise AssertionError("test provided too few completion responses")
            return self.completions.pop(0)
        raise AssertionError(f"unexpected endpoint: {path}")


def test_fingerprint_identity_binds_every_runtime_and_model_field() -> None:
    original = _fingerprint()
    original_digest = cache_identity_sha256(original)
    variations = [
        replace(original, backend_version="b7001-deadbeef"),
        replace(original, backend_repository_revision="dead"),
        replace(original, model_repository="example/other-GGUF"),
        replace(original, model_repository_revision="c" * 40),
        replace(original, gguf_filename="other-Q6_K.gguf"),
        replace(original, gguf_sha256="d" * 64),
        replace(original, quantization="Q8_0"),
        replace(original, chat_template_sha256="e" * 64),
        replace(original, choice_token_ids=(201, 202, 203)),
    ]
    assert all(cache_identity_sha256(item) != original_digest for item in variations)
    payload = cache_identity_payload(original)
    assert payload["model"]["gguf_sha256"] == "b" * 64
    assert payload["prompt_rendering"]["choice_token_ids"] == [101, 102, 103]
    assert payload["scoring"]["candidate_logit_bias"] == CANDIDATE_LOGIT_BIAS
    assert payload["scoring"]["cache_prompt"] is False
    assert payload["scoring"]["max_workers"] == 1
    assert cache_identity_sha256(original, cache_prompt=True) != original_digest
    assert cache_identity_sha256(original, max_workers=2) != original_digest
    assert cache_identity_payload(original, cache_prompt=True)["scoring"]["cache_prompt"] is True


def test_preflight_checks_props_and_exact_single_token_labels() -> None:
    transport = _FakeTransport()
    scorer = LlamaCppChoiceScorer(
        "http://127.0.0.1:8080",
        _fingerprint(),
        transport=transport,
        expected_model_path=MODEL_PATH,
    )
    assert scorer.cache_identity == cache_identity_sha256(_fingerprint())
    assert scorer.semantic_fingerprint == scorer.cache_identity
    assert scorer.expected_model_path == MODEL_PATH
    assert scorer.served_model_path == MODEL_PATH
    assert [(method, path) for method, path, _ in transport.calls] == [
        ("GET", "/props"),
        ("POST", "/tokenize"),
        ("POST", "/tokenize"),
        ("POST", "/tokenize"),
    ]
    token_payloads = [payload for _, path, payload in transport.calls if path == "/tokenize"]
    assert [payload["content"] for payload in token_payloads if payload is not None] == [
        "A",
        "B",
        "C",
    ]
    assert all(payload["add_special"] is False for payload in token_payloads if payload)


def test_preflight_rejects_same_filename_from_a_different_canonical_path() -> None:
    different_path = (MODEL_PATH.parent / "shadow" / MODEL_PATH.name).resolve()
    with pytest.raises(LlamaCppFingerprintError, match="exactly match"):
        LlamaCppChoiceScorer(
            "http://127.0.0.1:8080",
            _fingerprint(),
            transport=_FakeTransport(model_path=different_path),
            expected_model_path=MODEL_PATH,
        )


def test_expected_model_path_must_be_absolute() -> None:
    with pytest.raises(ValueError, match="absolute filesystem path"):
        LlamaCppChoiceScorer(
            "http://127.0.0.1:8080",
            _fingerprint(),
            transport=_FakeTransport(),
            expected_model_path=Path("model-Q6_K.gguf"),
        )


def test_cyclic_matrix_is_converted_back_to_canonical_action_order() -> None:
    # Each permutation represents the same action distribution [0.6, 0.3, 0.1].
    transport = _FakeTransport(
        [_completion([0.6, 0.3, 0.1]), _completion([0.3, 0.1, 0.6]), _completion([0.1, 0.6, 0.3])]
    )
    scorer = LlamaCppChoiceScorer(
        "http://127.0.0.1:8080",
        _fingerprint(),
        transport=transport,
        expected_model_path=MODEL_PATH,
    )
    actual = scorer.score_messages([_messages()])
    assert np.allclose(actual, [[0.6, 0.3, 0.1]])
    diagnostics = scorer.diagnostic_snapshot()
    assert diagnostics["mapping_attempts"] == 3
    assert diagnostics["mapping_successes"] == 3
    assert diagnostics["mapping_failures"] == 0
    assert diagnostics["request_attempts"] == 6
    assert diagnostics["request_responses"] == 6
    assert diagnostics["request_successes"] == 6
    assert diagnostics["request_failures"] == 0
    assert diagnostics["template_request_attempts"] == 3
    assert diagnostics["completion_request_attempts"] == 3
    assert diagnostics["fallback_count"] == 0

    for messages, label_to_action in zip(
        transport.rendered_messages, CYCLIC_LABEL_TO_ACTION, strict=True
    ):
        content = messages[-1]["content"]
        for label, action_index in zip(CHOICE_LABELS, label_to_action, strict=True):
            assert f"{label} = {ACTION_DESCRIPTIONS[action_index]}" in content

    completion_payloads = [payload for _, path, payload in transport.calls if path == "/completion"]
    assert len(completion_payloads) == 3
    for payload in completion_payloads:
        assert payload is not None
        assert payload["n_predict"] == 1
        assert payload["cache_prompt"] is False
        assert payload["post_sampling_probs"] is True
        assert payload["n_probs"] == 3
        assert payload["temperature"] == 1.0
        assert payload["top_k"] == 0
        assert payload["top_p"] == 1.0
        assert payload["min_p"] == 0.0
        assert payload["repeat_penalty"] == 1.0
        assert payload["logit_bias"] == [[101, 100.0], [102, 100.0], [103, 100.0]]
        assert "grammar" not in payload
    apply_payloads = [payload for _, path, payload in transport.calls if path == "/apply-template"]
    assert all(payload["add_generation_prompt"] is True for payload in apply_payloads if payload)


def test_equal_letter_preference_cancels_under_cyclic_average() -> None:
    transport = _FakeTransport([_completion([0.8, 0.15, 0.05])] * 3)
    scorer = LlamaCppChoiceScorer(
        "http://127.0.0.1:8080",
        _fingerprint(),
        transport=transport,
        expected_model_path=MODEL_PATH,
    )
    assert np.allclose(scorer.score_messages([_messages()]), [[1 / 3, 1 / 3, 1 / 3]])


def test_cache_prompt_is_emitted_and_bound_into_semantic_fingerprint() -> None:
    enabled_transport = _FakeTransport(
        [
            _completion([0.6, 0.3, 0.1]),
            _completion([0.3, 0.1, 0.6]),
            _completion([0.1, 0.6, 0.3]),
        ]
    )
    enabled = LlamaCppChoiceScorer(
        "http://127.0.0.1:8080",
        _fingerprint(),
        transport=enabled_transport,
        cache_prompt=True,
        expected_model_path=MODEL_PATH,
    )
    disabled = LlamaCppChoiceScorer(
        "http://127.0.0.1:8080",
        _fingerprint(),
        transport=_FakeTransport(),
        expected_model_path=MODEL_PATH,
    )
    assert np.allclose(enabled.score_messages([_messages()]), [[0.6, 0.3, 0.1]])
    completion_payloads = [
        payload for _, path, payload in enabled_transport.calls if path == "/completion"
    ]
    assert all(payload["cache_prompt"] is True for payload in completion_payloads if payload)
    assert enabled.semantic_fingerprint == enabled.cache_identity
    assert enabled.semantic_fingerprint != disabled.semantic_fingerprint
    assert enabled.semantic_fingerprint == cache_identity_sha256(_fingerprint(), cache_prompt=True)


def test_unrestricted_audit_reports_natural_token_candidate_ranks_and_mass() -> None:
    transport = _FakeTransport([_unrestricted_completion()])
    scorer = LlamaCppChoiceScorer(
        "http://127.0.0.1:8080",
        _fingerprint(),
        transport=transport,
        cache_prompt=True,
        expected_model_path=MODEL_PATH,
    )
    identity_before = scorer.semantic_fingerprint
    records = scorer.audit_unrestricted_messages([_messages()])
    assert scorer.semantic_fingerprint == identity_before
    assert len(records) == 1
    record = records[0]
    assert record.row_index == 0
    assert record.generated_token_id == 999
    assert record.generated_token == "\n"
    assert record.generated_is_exact_choice is False
    assert record.generated_choice_label is None
    assert record.candidate_logprobs == (-1.0, -2.0, -3.0)
    assert record.candidate_ranks == (2, 3, 4)
    assert record.candidate_mass == pytest.approx(np.exp(-1.0) + np.exp(-2.0) + np.exp(-3.0))
    assert record.as_dict()["candidate_ranks"] == {"A": 2, "B": 3, "C": 4}

    completion_payload = next(
        payload for _, path, payload in transport.calls if path == "/completion"
    )
    assert completion_payload["cache_prompt"] is True
    assert completion_payload["n_probs"] == 50
    assert completion_payload["min_keep"] == 1
    assert completion_payload["post_sampling_probs"] is False
    assert "logit_bias" not in completion_payload


def test_unrestricted_audit_fails_closed_when_a_candidate_is_missing() -> None:
    malformed = _unrestricted_completion()
    top_logprobs = malformed["completion_probabilities"][0]["top_logprobs"]
    malformed["completion_probabilities"][0]["top_logprobs"] = [
        entry for entry in top_logprobs if entry["id"] != TOKEN_IDS["C"]
    ]
    scorer = LlamaCppChoiceScorer(
        "http://127.0.0.1:8080",
        _fingerprint(),
        transport=_FakeTransport([malformed]),
        expected_model_path=MODEL_PATH,
    )
    with pytest.raises(LlamaCppProtocolError, match="every pinned A/B/C"):
        scorer.audit_unrestricted_messages([_messages()])


def test_parallel_scoring_preserves_sequential_values_and_row_order() -> None:
    action_probabilities = ([0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [0.1, 0.2, 0.7])

    class _IndexedTransport(_FakeTransport):
        def request_json(self, method, path, payload=None):
            if path == "/apply-template":
                content = payload["messages"][-1]["content"]
                row_index = int(content.split("ROW:", 1)[1].split("\n", 1)[0])
                mapping_index = next(
                    index
                    for index, mapping in enumerate(CYCLIC_LABEL_TO_ACTION)
                    if all(
                        f"{label} = {ACTION_DESCRIPTIONS[action_index]}" in content
                        for label, action_index in zip(CHOICE_LABELS, mapping, strict=True)
                    )
                )
                return {"prompt": f"{row_index}:{mapping_index}"}
            if path == "/completion":
                row_index, mapping_index = (int(value) for value in payload["prompt"].split(":"))
                # Reverse completion latency so parallel futures do not finish in
                # submission order.
                time.sleep(0.002 * (3 - mapping_index))
                mapping = CYCLIC_LABEL_TO_ACTION[mapping_index]
                label_probabilities = [
                    action_probabilities[row_index][action_index] for action_index in mapping
                ]
                return _completion(label_probabilities)
            return super().request_json(method, path, payload)

    batches = []
    for row_index in range(len(action_probabilities)):
        messages = _messages()
        messages[-1]["content"] = f"ROW:{row_index}\n" + messages[-1]["content"]
        batches.append(messages)

    sequential = LlamaCppChoiceScorer(
        "http://127.0.0.1:8080",
        _fingerprint(),
        transport=_IndexedTransport(),
        max_workers=1,
        expected_model_path=MODEL_PATH,
    )
    parallel = LlamaCppChoiceScorer(
        "http://127.0.0.1:8080",
        _fingerprint(),
        transport=_IndexedTransport(),
        max_workers=4,
        expected_model_path=MODEL_PATH,
    )
    expected = np.asarray(action_probabilities)
    assert np.allclose(sequential.score_messages(batches), expected)
    assert np.array_equal(parallel.score_messages(batches), sequential.score_messages(batches))
    assert sequential.cache_identity != parallel.cache_identity
    assert sequential.cache_identity == cache_identity_sha256(_fingerprint(), max_workers=1)
    assert parallel.cache_identity == cache_identity_sha256(_fingerprint(), max_workers=4)
    assert sequential.execution_metadata == {"backend": "llama.cpp-http", "max_workers": 1}
    assert parallel.execution_metadata == {"backend": "llama.cpp-http", "max_workers": 4}


@pytest.mark.parametrize("workers", [True, 0, 65, 1.5])
def test_max_workers_validation(workers: object) -> None:
    with pytest.raises(ValueError, match="max_workers"):
        LlamaCppChoiceScorer(
            "http://127.0.0.1:8080",
            _fingerprint(),
            transport=_FakeTransport(),
            max_workers=workers,
            expected_model_path=MODEL_PATH,
        )
    with pytest.raises(ValueError, match="max_workers"):
        cache_identity_sha256(_fingerprint(), max_workers=workers)


@pytest.mark.parametrize("cache_prompt", [0, 1, "true", None])
def test_cache_prompt_requires_an_exact_bool(cache_prompt: object) -> None:
    with pytest.raises(ValueError, match="cache_prompt"):
        LlamaCppChoiceScorer(
            "http://127.0.0.1:8080",
            _fingerprint(),
            transport=_FakeTransport(),
            cache_prompt=cache_prompt,
            expected_model_path=MODEL_PATH,
        )
    with pytest.raises(ValueError, match="cache_prompt"):
        cache_identity_sha256(_fingerprint(), cache_prompt=cache_prompt)


def test_scorer_rejects_incomplete_candidate_id_set() -> None:
    malformed = _completion([0.6, 0.3, 0.1])
    step = malformed["completion_probabilities"][0]
    step["top_probs"] = step["top_probs"][:2]
    transport = _FakeTransport([malformed])
    scorer = LlamaCppChoiceScorer(
        "http://127.0.0.1:8080",
        _fingerprint(),
        transport=transport,
        expected_model_path=MODEL_PATH,
    )
    with pytest.raises(LlamaCppProtocolError, match="exactly the pinned A/B/C"):
        scorer.score_messages([_messages()])
    diagnostics = scorer.diagnostic_snapshot()
    assert diagnostics["mapping_attempts"] == 1
    assert diagnostics["mapping_failures"] == 1
    assert diagnostics["request_attempts"] == 2
    assert diagnostics["request_failures"] == 1
    assert diagnostics["protocol_failures"] == 1
    assert diagnostics["truncation_failures"] == 0
    assert diagnostics["malformed_response_failures"] == 1
    assert diagnostics["fallback_count"] == 0


def test_scorer_separately_counts_explicit_truncation() -> None:
    truncated = _completion([0.6, 0.3, 0.1])
    truncated["truncated"] = True
    scorer = LlamaCppChoiceScorer(
        "http://127.0.0.1:8080",
        _fingerprint(),
        transport=_FakeTransport([truncated]),
        expected_model_path=MODEL_PATH,
    )

    with pytest.raises(LlamaCppTruncationError, match="truncated"):
        scorer.score_messages([_messages()])
    diagnostics = scorer.diagnostic_snapshot()
    assert diagnostics["mapping_attempts"] == 1
    assert diagnostics["mapping_failures"] == 1
    assert diagnostics["request_attempts"] == 2
    assert diagnostics["request_responses"] == 2
    assert diagnostics["request_successes"] == 1
    assert diagnostics["request_failures"] == 1
    assert diagnostics["protocol_failures"] == 1
    assert diagnostics["truncation_failures"] == 1
    assert diagnostics["malformed_response_failures"] == 0
    assert diagnostics["fallback_count"] == 0


def test_scorer_counts_raw_connection_reset_as_transport_failure() -> None:
    class _ResetTransport(_FakeTransport):
        def request_json(self, method, path, payload=None):
            if path == "/completion":
                raise ConnectionResetError("injected reset")
            return super().request_json(method, path, payload)

    scorer = LlamaCppChoiceScorer(
        "http://127.0.0.1:8080",
        _fingerprint(),
        transport=_ResetTransport(),
        expected_model_path=MODEL_PATH,
    )

    with pytest.raises(ConnectionResetError, match="injected reset"):
        scorer.score_messages([_messages()])
    diagnostics = scorer.diagnostic_snapshot()
    assert diagnostics["mapping_attempts"] == 1
    assert diagnostics["mapping_failures"] == 1
    assert diagnostics["request_attempts"] == 2
    assert diagnostics["request_responses"] == 1
    assert diagnostics["request_successes"] == 1
    assert diagnostics["request_failures"] == 1
    assert diagnostics["transport_failures"] == 1
    assert diagnostics["protocol_failures"] == 0
    assert diagnostics["unexpected_failures"] == 0
    assert diagnostics["fallback_count"] == 0


def test_scorer_rejects_three_candidate_probabilities_with_missing_mass() -> None:
    # Regression for a live grammar/post_sampling_probs response observed at 0.8825 mass.
    malformed = _completion([0.50, 0.25, 0.1325])
    transport = _FakeTransport([malformed])
    scorer = LlamaCppChoiceScorer(
        "http://127.0.0.1:8080",
        _fingerprint(),
        transport=transport,
        expected_model_path=MODEL_PATH,
    )
    with pytest.raises(LlamaCppProtocolError, match="complete normalized probability mass"):
        scorer.score_messages([_messages()])


def test_scorer_accepts_unambiguous_legacy_probs_alias() -> None:
    responses = []
    for probabilities in ([0.6, 0.3, 0.1], [0.3, 0.1, 0.6], [0.1, 0.6, 0.3]):
        response = _completion(list(probabilities))
        response["probs"] = response.pop("completion_probabilities")
        responses.append(response)
    scorer = LlamaCppChoiceScorer(
        "http://127.0.0.1:8080",
        _fingerprint(),
        transport=_FakeTransport(responses),
        expected_model_path=MODEL_PATH,
    )
    assert np.allclose(scorer.score_messages([_messages()]), [[0.6, 0.3, 0.1]])


def test_scorer_rejects_ambiguous_probability_fields() -> None:
    malformed = _completion([0.6, 0.3, 0.1])
    malformed["probs"] = malformed["completion_probabilities"]
    scorer = LlamaCppChoiceScorer(
        "http://127.0.0.1:8080",
        _fingerprint(),
        transport=_FakeTransport([malformed]),
        expected_model_path=MODEL_PATH,
    )
    with pytest.raises(LlamaCppProtocolError, match="exactly one recognized"):
        scorer.score_messages([_messages()])


def test_preflight_rejects_live_tokenizer_drift() -> None:
    class _DriftedTransport(_FakeTransport):
        def request_json(self, method, path, payload=None):
            response = super().request_json(method, path, payload)
            if path == "/tokenize" and payload["content"] == "B":
                return {"tokens": [{"id": 999, "piece": "B"}]}
            return response

    with pytest.raises(LlamaCppFingerprintError, match="do not match"):
        LlamaCppChoiceScorer(
            "http://127.0.0.1:8080",
            _fingerprint(),
            transport=_DriftedTransport(),
            expected_model_path=MODEL_PATH,
        )


def test_cyclic_rewrite_fails_closed_without_exact_canonical_block() -> None:
    messages = _messages()
    messages[-1]["content"] = messages[-1]["content"].replace("vasopressor", "pressors")
    with pytest.raises(ValueError, match="exactly one canonical candidate block"):
        cyclic_choice_messages(messages, 0)


@pytest.mark.parametrize(
    "url",
    [
        "ftp://localhost:8080",
        "http://user:secret@localhost:8080",
        "http://localhost:8080/v1",
        "http://localhost:8080?x=1",
    ],
)
def test_http_transport_rejects_unsafe_or_ambiguous_base_urls(url: str) -> None:
    with pytest.raises(ValueError, match="origin-only"):
        UrllibJsonTransport(url)


def test_http_transport_converts_timeout_to_sanitized_domain_error() -> None:
    class _TimeoutOpener:
        def open(self, request, timeout):
            del request, timeout
            raise TimeoutError("low-level detail")

    transport = UrllibJsonTransport("http://127.0.0.1:8080")
    transport._opener = _TimeoutOpener()
    with pytest.raises(LlamaCppTransportError, match="GET /props timed out") as caught:
        transport.request_json("GET", "/props")
    assert "low-level detail" not in str(caught.value)


def test_scoring_is_blocked_when_explicit_preflight_has_not_run() -> None:
    scorer = LlamaCppChoiceScorer(
        "http://127.0.0.1:8080",
        _fingerprint(),
        transport=_FakeTransport(),
        preflight=False,
        expected_model_path=MODEL_PATH,
    )
    with pytest.raises(LlamaCppFingerprintError, match="preflight must succeed"):
        scorer.score_messages([_messages()])
