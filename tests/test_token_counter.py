"""Tests for the token accounting layer."""

import pytest

from optimizerllm.token_accounting.counter import TokenCounter, TokenBudget


class TestTokenCounter:
    def test_count_nonempty_string(self):
        counter = TokenCounter()
        count = counter.count("Hello, world!", provider="openai", model="gpt-4o")
        assert count > 0

    def test_count_empty_string(self):
        counter = TokenCounter()
        assert counter.count("", provider="openai") == 0

    def test_count_scales_by_provider(self):
        counter = TokenCounter()
        text = "The quick brown fox jumps over the lazy dog." * 10
        openai_count = counter.count(text, provider="openai", model="gpt-4o")
        anthropic_count = counter.count(text, provider="anthropic", model="claude-3-5-sonnet")
        # Anthropic applies a scaling factor > 1.0
        assert anthropic_count >= openai_count

    def test_count_blocks_fills_token_count(self, system_block, user_block):
        counter = TokenCounter()
        blocks = counter.count_blocks([system_block, user_block])
        for b in blocks:
            assert b.token_count is not None
            assert b.token_count > 0

    def test_total_tokens(self, system_block, user_block):
        counter = TokenCounter()
        blocks = counter.count_blocks([system_block, user_block])
        total = counter.total_tokens(blocks)
        expected = sum(b.token_count for b in blocks)
        assert total == expected

    def test_cost_estimate_known_model(self):
        counter = TokenCounter()
        estimate = counter.estimate_cost(
            10_000,
            output_tokens=1_000,
            provider="openai",
            model="gpt-4o",
        )
        assert estimate.input_cost_usd > 0
        assert estimate.total_usd > 0

    def test_cost_estimate_unknown_model(self):
        counter = TokenCounter()
        estimate = counter.estimate_cost(
            10_000,
            provider="openai",
            model="unknown-model-xyz",
        )
        # Should not crash, returns zeros
        assert estimate.total_usd == 0.0

    def test_cost_with_cache_prefix(self):
        counter = TokenCounter()
        no_cache = counter.estimate_cost(
            10_000, provider="openai", model="gpt-4o"
        )
        with_cache = counter.estimate_cost(
            10_000,
            cacheable_prefix_tokens=8_000,
            provider="openai",
            model="gpt-4o",
        )
        # Cached reads should be cheaper
        assert with_cache.total_usd < no_cache.total_usd


class TestTokenBudget:
    def test_available_tokens(self):
        budget = TokenBudget(
            max_input_tokens=128_000,
            output_reserved=4_096,
            provider="openai",
            model="gpt-4o",
        )
        assert budget.available_input_tokens == 128_000 - 4_096

    def test_cache_min_tokens(self):
        budget = TokenBudget(
            max_input_tokens=128_000,
            output_reserved=4_096,
            provider="openai",
            model="gpt-4o",
        )
        assert budget.cache_min_tokens == 1024
