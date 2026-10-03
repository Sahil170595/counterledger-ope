from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="local-transformers backend is an optional extra")

from llm_client import (  # noqa: E402
    CHOICE_TOKENS,
    LocalTransformersChoiceScorer,
    TokenProbabilityRecord,
    token_probability_cache_key,
)


class _BackendTokenizer:
    def __init__(self, definition: str) -> None:
        self.definition = definition

    def to_str(self) -> str:
        return self.definition


class _FakeTokenizer:
    def __init__(self, *, chat_template: str = "chat-v1", invalid_choices: bool = False) -> None:
        self.chat_template = chat_template
        self.invalid_choices = invalid_choices
        self.pad_token_id = 0
        self.pad_token = "<pad>"
        self.eos_token = "</s>"
        self.padding_side = "right"
        self.special_tokens_map = {"pad_token": self.pad_token, "eos_token": self.eos_token}
        self.backend_tokenizer = _BackendTokenizer("stable-tokenizer-definition")

    def get_chat_template(self) -> str:
        return self.chat_template

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        label = text.strip()
        if label not in {"A", "B", "C"}:
            return [7]
        if self.invalid_choices:
            return [1, 2]
        return {"A": [1], "B": [2], "C": [3]}[label]

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        return "|".join(message["content"] for message in messages)

    def __call__(
        self,
        texts: list[str],
        *,
        padding: bool,
        return_tensors: str,
        truncation: bool,
    ) -> dict[str, torch.Tensor]:
        assert padding is True
        assert return_tensors == "pt"
        assert truncation is False
        rows = [[4, 5, 6, 7] for _ in texts]
        return {
            "input_ids": torch.tensor(rows, dtype=torch.long),
            "attention_mask": torch.ones((len(rows), 4), dtype=torch.long),
        }


class _FakeProcessor:
    def __init__(self, tokenizer: _FakeTokenizer) -> None:
        self.tokenizer = tokenizer
        self.chat_template = tokenizer.chat_template
        self.encode_calls = 0

    def get_chat_template(self) -> str:
        return self.chat_template

    def apply_chat_template(self, *args: Any, **kwargs: Any) -> str:
        return self.tokenizer.apply_chat_template(*args, **kwargs)

    def __call__(self, *, text: list[str], **kwargs: Any) -> dict[str, torch.Tensor]:
        self.encode_calls += 1
        return self.tokenizer(text, **kwargs)


class _RecordingHead(torch.nn.Linear):
    def __init__(self) -> None:
        super().__init__(4, 8, bias=False)
        self.last_sequence_width: int | None = None
        with torch.no_grad():
            self.weight.copy_(torch.arange(32, dtype=torch.float32).reshape(8, 4) / 32.0)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.last_sequence_width = int(hidden_states.shape[1])
        return super().forward(hidden_states)


class _FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_embeddings = torch.nn.Embedding(16, 4)
        self.lm_head = _RecordingHead()
        self.config = SimpleNamespace(quantization_config=None)

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.input_embeddings

    def get_output_embeddings(self) -> torch.nn.Module:
        return self.lm_head

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        use_cache: bool,
    ) -> SimpleNamespace:
        del attention_mask
        assert use_cache is False
        hidden_states = self.input_embeddings(input_ids)
        return SimpleNamespace(logits=self.lm_head(hidden_states))


class _FakeBitsAndBytesConfig:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def to_dict(self) -> dict[str, Any]:
        return self.kwargs


def _fake_transformers(
    *,
    architecture: str,
    tokenizer: _FakeTokenizer,
    model: _FakeModel,
    processor: _FakeProcessor | None = None,
) -> tuple[types.ModuleType, dict[str, Any]]:
    calls: dict[str, Any] = {"causal": 0, "multimodal": 0, "tokenizer": 0}
    module = types.ModuleType("transformers")
    module.__version__ = "test-transformers"

    class AutoConfig:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: Any) -> SimpleNamespace:
            calls["config_kwargs"] = kwargs
            assert model_id == "org/model"
            return SimpleNamespace(architectures=[architecture])

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: Any) -> _FakeTokenizer:
            calls["tokenizer"] += 1
            calls["tokenizer_kwargs"] = kwargs
            assert model_id == "org/model"
            return tokenizer

    class AutoProcessor:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: Any) -> _FakeProcessor:
            calls["processor_kwargs"] = kwargs
            assert model_id == "org/model"
            assert processor is not None
            return processor

    class AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: Any) -> _FakeModel:
            calls["causal"] += 1
            calls["model_kwargs"] = kwargs
            assert model_id == "org/model"
            return model

    class AutoModelForImageTextToText:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: Any) -> _FakeModel:
            calls["multimodal"] += 1
            calls["model_kwargs"] = kwargs
            assert model_id == "org/model"
            return model

    module.AutoConfig = AutoConfig
    module.AutoTokenizer = AutoTokenizer
    module.AutoProcessor = AutoProcessor
    module.AutoModelForCausalLM = AutoModelForCausalLM
    module.AutoModelForImageTextToText = AutoModelForImageTextToText
    module.BitsAndBytesConfig = _FakeBitsAndBytesConfig
    return module, calls


def _messages() -> list[list[dict[str, str]]]:
    return [
        [{"role": "user", "content": "first"}],
        [{"role": "user", "content": "second"}],
    ]


def test_cache_key_and_record_bind_scorer_semantic_fingerprint() -> None:
    kwargs = {
        "messages": _messages()[0],
        "model_id": "org/model",
        "model_revision": "a" * 40,
        "prompt_version": "prompt-v1",
        "choice_tokens": CHOICE_TOKENS,
        "logit_temperature": 1.0,
    }
    first = "1" * 64
    second = "2" * 64
    assert token_probability_cache_key(**kwargs, scorer_semantic_fingerprint=first) != (
        token_probability_cache_key(**kwargs, scorer_semantic_fingerprint=second)
    )
    with pytest.raises(ValueError, match="SHA-256"):
        token_probability_cache_key(**kwargs, scorer_semantic_fingerprint="invalid")

    record = TokenProbabilityRecord(
        key="k",
        model_id="org/model",
        model_revision="a" * 40,
        prompt_version="prompt-v1",
        prompt_sha256="3" * 64,
        choice_tokens=CHOICE_TOKENS,
        logit_temperature=1.0,
        inference_batch_size=2,
        inference_dtype="float32",
        probabilities=(0.6, 0.3, 0.1),
        rationale="code-generated annotation",
        scorer_semantic_fingerprint=first,
    )
    assert TokenProbabilityRecord.from_dict(record.as_dict()) == record


def test_causal_scorer_fingerprints_semantics_and_slices_before_output_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = _FakeTokenizer()
    model = _FakeModel()
    fake_transformers, calls = _fake_transformers(
        architecture="FakeForCausalLM", tokenizer=tokenizer, model=model
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    scorer = LocalTransformersChoiceScorer(
        model_id="org/model",
        model_revision="a" * 40,
        dtype="float32",
        device="cpu",
    )
    probabilities = scorer.score_messages(_messages())

    assert scorer.backend == "causal_lm"
    assert scorer.choice_tokens == CHOICE_TOKENS
    assert len(scorer.semantic_fingerprint) == 64
    assert scorer.semantic_fingerprint_payload["numeric"]["quantization_requested"] == {
        "method": "none"
    }
    assert calls["causal"] == 1 and calls["multimodal"] == 0
    assert model.lm_head.last_sequence_width == 1
    assert probabilities.shape == (2, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_multimodal_backend_uses_processor_in_text_only_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = _FakeTokenizer()
    processor = _FakeProcessor(tokenizer)
    model = _FakeModel()
    fake_transformers, calls = _fake_transformers(
        architecture="FakeForConditionalGeneration",
        tokenizer=tokenizer,
        model=model,
        processor=processor,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    scorer = LocalTransformersChoiceScorer(
        model_id="org/model",
        model_revision="b" * 40,
        dtype="float32",
        device="cpu",
    )
    probabilities = scorer.score_messages(_messages()[:1])

    assert scorer.backend == "image_text_to_text"
    assert scorer.semantic_fingerprint_payload["transformers"]["text_only"] is True
    assert calls["causal"] == 0 and calls["multimodal"] == 1
    assert calls["tokenizer"] == 0
    assert processor.encode_calls == 1
    assert probabilities.shape == (1, 3)


def test_choice_token_preflight_fails_closed_before_model_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = _FakeTokenizer(invalid_choices=True)
    fake_transformers, calls = _fake_transformers(
        architecture="FakeForCausalLM", tokenizer=tokenizer, model=_FakeModel()
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    with pytest.raises(RuntimeError, match="not distinct single tokens"):
        LocalTransformersChoiceScorer(
            model_id="org/model",
            model_revision="c" * 40,
            dtype="float32",
            device="cpu",
        )
    assert calls["causal"] == 0


def test_four_bit_preset_is_explicit_and_passed_to_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = _FakeTokenizer()
    fake_transformers, calls = _fake_transformers(
        architecture="FakeForCausalLM", tokenizer=tokenizer, model=_FakeModel()
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    scorer = LocalTransformersChoiceScorer(
        model_id="org/model",
        model_revision="d" * 40,
        dtype="float32",
        quantization="4bit",
    )

    model_kwargs = calls["model_kwargs"]
    assert model_kwargs["device_map"] == "auto"
    assert isinstance(model_kwargs["quantization_config"], _FakeBitsAndBytesConfig)
    requested = scorer.semantic_fingerprint_payload["numeric"]["quantization_requested"]
    assert requested["method"] == "bitsandbytes"
    assert requested["config"]["load_in_4bit"] is True
