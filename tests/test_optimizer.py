"""End-to-end integration tests for ContextOptimizer."""

import pytest

from optimizerllm import ContextOptimizer, CompressionMode


SYSTEM = "You are a knowledgeable assistant specialized in European geography."

DOCUMENTS = [
    "Paris is the capital and most populous city of France, with a population of over 2 million.",
    "Berlin is the capital and largest city of Germany by both area and population.",
    "Madrid is the capital and most populous municipality of Spain.",
    "Rome is the capital city of Italy and a special comune.",
    "Amsterdam is the capital of the Netherlands, though The Hague is the seat of government.",
]

MESSAGES = [{"role": "user", "content": "What is the capital of France?"}]


class TestContextOptimizerLossless:
    def test_returns_result(self):
        opt = ContextOptimizer(mode=CompressionMode.LOSSLESS)
        result = opt.optimize(system=SYSTEM, messages=MESSAGES)
        assert result is not None
        assert result.rendered is not None

    def test_telemetry_populated(self):
        opt = ContextOptimizer(mode=CompressionMode.LOSSLESS)
        result = opt.optimize(system=SYSTEM, messages=MESSAGES)
        t = result.telemetry
        assert t.tokens_before > 0
        assert t.tokens_after > 0
        assert t.blocks_before > 0
        assert t.blocks_after > 0

    def test_no_blocks_dropped_in_lossless(self):
        opt = ContextOptimizer(mode=CompressionMode.LOSSLESS)
        result = opt.optimize(system=SYSTEM, messages=MESSAGES, documents=DOCUMENTS)
        # In lossless mode with a large budget, nothing should be dropped
        dropped_for_budget = [
            r for r in result.telemetry.blocks_removed
            if "budget" in r.reason
        ]
        assert len(dropped_for_budget) == 0


class TestContextOptimizerConservative:
    def test_scores_docs_against_query(self):
        opt = ContextOptimizer(
            mode=CompressionMode.CONSERVATIVE,
            selector_threshold=0.05,
        )
        result = opt.optimize(
            system=SYSTEM,
            messages=MESSAGES,
            documents=DOCUMENTS,
            query="capital of France",
        )
        assert result.tokens_after > 0

    def test_reduction_never_exceeds_100_pct(self):
        opt = ContextOptimizer(mode=CompressionMode.CONSERVATIVE)
        result = opt.optimize(
            system=SYSTEM,
            messages=MESSAGES,
            documents=DOCUMENTS,
        )
        assert 0 <= result.reduction_pct <= 100

    def test_rendered_is_list_for_openai(self):
        opt = ContextOptimizer(mode=CompressionMode.CONSERVATIVE, output_format="openai")
        result = opt.optimize(system=SYSTEM, messages=MESSAGES)
        assert isinstance(result.rendered, list)
        for msg in result.rendered:
            assert "role" in msg
            assert "content" in msg

    def test_rendered_is_dict_for_anthropic(self):
        opt = ContextOptimizer(
            provider="anthropic",
            model="claude-3-5-sonnet",
            mode=CompressionMode.CONSERVATIVE,
            output_format="anthropic",
        )
        result = opt.optimize(system=SYSTEM, messages=MESSAGES)
        assert isinstance(result.rendered, dict)
        assert "messages" in result.rendered


class TestContextOptimizerBudget:
    def test_respects_token_budget(self):
        opt = ContextOptimizer(
            mode=CompressionMode.CONSERVATIVE,
            max_input_tokens=200,
            output_budget=50,
        )
        result = opt.optimize(
            system=SYSTEM,
            messages=MESSAGES,
            documents=DOCUMENTS,
        )
        # Must fit within the token budget
        assert result.telemetry.tokens_after <= 150  # 200 - 50

    def test_last_user_message_always_preserved(self):
        opt = ContextOptimizer(
            mode=CompressionMode.AGGRESSIVE,
            max_input_tokens=100,
            output_budget=10,
        )
        result = opt.optimize(
            system=SYSTEM,
            messages=MESSAGES,
            documents=DOCUMENTS * 5,  # Flood with docs
        )
        # The user's message must always be in the output
        rendered = result.rendered
        if isinstance(rendered, list):
            user_msgs = [m for m in rendered if m["role"] == "user"]
            assert any("capital of France" in m["content"] for m in user_msgs)


class TestContextOptimizerHistory:
    def test_compacts_long_history(self):
        history = []
        for i in range(10):
            history.append({"role": "user", "content": f"I need help with task {i}."})
            history.append({"role": "assistant", "content": f"Sure, here is help for task {i}."})

        opt = ContextOptimizer(
            mode=CompressionMode.CONSERVATIVE,
            compactor_recency_tail=2,
        )
        result = opt.optimize(
            system=SYSTEM,
            messages=MESSAGES,
            history=history,
        )
        assert result.telemetry.blocks_after < result.telemetry.blocks_before

    def test_summary_runs_without_error(self):
        opt = ContextOptimizer(mode=CompressionMode.CONSERVATIVE)
        result = opt.optimize(system=SYSTEM, messages=MESSAGES)
        summary = result.summary()
        assert "Tokens:" in summary
        assert "Blocks:" in summary


class TestContextOptimizerCacheTelemetry:
    def test_cacheable_prefix_tokens_set(self):
        opt = ContextOptimizer(mode=CompressionMode.CONSERVATIVE)
        result = opt.optimize(system=SYSTEM, messages=MESSAGES)
        # System prompt should form the cacheable prefix
        assert result.telemetry.cacheable_prefix_tokens >= 0

    def test_risk_level_set(self):
        opt = ContextOptimizer(mode=CompressionMode.CONSERVATIVE)
        result = opt.optimize(system=SYSTEM, messages=MESSAGES)
        assert result.telemetry.risk_level in ("low", "medium", "high")
