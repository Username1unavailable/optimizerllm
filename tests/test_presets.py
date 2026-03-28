"""Tests for the presets API."""

import pytest
from optimizerllm.presets import preset, list_presets, rag_optimizer, chat_optimizer
from optimizerllm.optimizer import ContextOptimizer
from optimizerllm.core.block import CompressionMode


DOCS = [
    "Paris is the capital of France.",
    "Berlin is the capital of Germany.",
    "Tokyo is the capital of Japan.",
]
MESSAGES = [{"role": "user", "content": "What is the capital of France?"}]


class TestPresetFactory:
    def test_returns_context_optimizer(self):
        opt = preset("rag")
        assert isinstance(opt, ContextOptimizer)

    def test_all_preset_names_valid(self):
        for name in ["rag", "chat", "agent", "long_context_qa", "summarization"]:
            opt = preset(name)
            assert isinstance(opt, ContextOptimizer)

    def test_invalid_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            preset("nonexistent_preset")

    def test_provider_override(self):
        opt = preset("rag", provider="anthropic", model="claude-3-5-sonnet")
        assert opt.provider == "anthropic"
        assert opt.model == "claude-3-5-sonnet"

    def test_param_override(self):
        opt = preset("rag", selector_threshold=0.99)
        assert opt.selector_threshold == 0.99


class TestListPresets:
    def test_returns_list(self):
        presets = list_presets()
        assert isinstance(presets, list)
        assert len(presets) == 5

    def test_each_entry_has_required_keys(self):
        for p in list_presets():
            assert "name" in p
            assert "mode" in p
            assert "description" in p


class TestConvenienceFunctions:
    def test_rag_optimizer(self):
        opt = rag_optimizer(provider="openai")
        assert isinstance(opt, ContextOptimizer)
        assert opt.mode == CompressionMode.CONSERVATIVE

    def test_chat_optimizer(self):
        opt = chat_optimizer(provider="openai")
        assert isinstance(opt, ContextOptimizer)

    def test_rag_preset_runs_end_to_end(self):
        opt = rag_optimizer(provider="openai", model="gpt-4o", max_input_tokens=8192)
        result = opt.optimize(
            system="You are a helpful assistant.",
            documents=DOCS,
            messages=MESSAGES,
            query="capital of France",
        )
        assert result.telemetry.tokens_after > 0
        assert isinstance(result.rendered, list)

    def test_summarization_preset_keeps_all_docs(self):
        opt = preset("summarization", provider="openai", model="gpt-4o", max_input_tokens=128_000)
        result = opt.optimize(
            system="Summarize the following documents.",
            documents=DOCS,
            messages=MESSAGES,
        )
        # Lossless mode — no docs should be dropped for relevance
        relevance_drops = [
            r for r in result.telemetry.blocks_removed
            if "score_below_threshold" in r.reason
        ]
        assert len(relevance_drops) == 0

    def test_chat_preset_longer_recency_tail(self):
        history = [
            {"role": "user", "content": f"Message {i}"}
            for i in range(10)
        ]
        chat_opt = chat_optimizer(provider="openai", model="gpt-4o", max_input_tokens=128_000)
        rag_opt = rag_optimizer(provider="openai", model="gpt-4o", max_input_tokens=128_000)

        chat_result = chat_opt.optimize(history=history, messages=MESSAGES)
        rag_result = rag_opt.optimize(history=history, messages=MESSAGES)

        # Chat preset has longer recency tail so retains more history
        chat_history = sum(1 for b in chat_result.blocks if b.block_type.value == "conversation_history")
        rag_history = sum(1 for b in rag_result.blocks if b.block_type.value == "conversation_history")
        assert chat_history >= rag_history
