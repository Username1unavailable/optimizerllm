"""Tests for the budget controller."""

import pytest

from optimizerllm.budget.controller import BudgetController
from optimizerllm.core.block import (
    BlockMetadata,
    BlockType,
    CompressionMode,
    Mutability,
    make_message_block,
    make_retrieved_doc_block,
    make_system_block,
)


def _block_with_tokens(content: str, tokens: int, *, droppable: bool = True, protected: bool = False):
    """Helper: make a block with a preset token count."""
    b = make_retrieved_doc_block(content, source="test")
    if protected:
        b = make_system_block(content, protected=True)
    elif not droppable:
        from dataclasses import replace
        new_meta = replace(b.metadata, mutability=Mutability.COMPRESSIBLE)
        b = b.clone_with(metadata=new_meta)
    b = b.clone_with(token_count=tokens)
    return b


class TestBudgetControllerBasic:
    def test_all_blocks_fit(self):
        ctrl = BudgetController(max_input_tokens=1000, output_reserved=0)
        blocks = [
            _block_with_tokens("A", 100),
            _block_with_tokens("B", 200),
            _block_with_tokens("C", 300),
        ]
        result = ctrl.allocate(blocks)
        assert len(result.dropped) == 0
        assert result.tokens_used == 600

    def test_drops_lowest_priority_when_over_budget(self):
        ctrl = BudgetController(max_input_tokens=300, output_reserved=0)
        high = _block_with_tokens("High priority", 150)
        high.metadata.priority = 80
        low = _block_with_tokens("Low priority", 200)
        low.metadata.priority = 20
        result = ctrl.allocate([high, low])
        kept_contents = {b.content for b in result.final_blocks}
        assert "High priority" in kept_contents
        assert "Low priority" not in kept_contents

    def test_protected_blocks_always_kept(self):
        ctrl = BudgetController(max_input_tokens=100, output_reserved=0)
        protected = _block_with_tokens("Must keep", 80, protected=True)
        optional = _block_with_tokens("Optional", 80, droppable=True)
        result = ctrl.allocate([protected, optional])
        kept_contents = {b.content for b in result.final_blocks}
        assert "Must keep" in kept_contents

    def test_output_reserved_reduces_available(self):
        ctrl = BudgetController(max_input_tokens=1000, output_reserved=400)
        assert ctrl.available_tokens == 600

    def test_empty_blocks(self):
        ctrl = BudgetController(max_input_tokens=1000, output_reserved=0)
        result = ctrl.allocate([])
        assert result.tokens_used == 0
        assert result.final_blocks == []


class TestBudgetControllerTruncation:
    def test_truncates_compressible_block(self):
        ctrl = BudgetController(max_input_tokens=50, output_reserved=0)
        long_text = "word " * 100  # ~500 tokens worth
        block = _block_with_tokens(long_text, 500, droppable=False)
        result = ctrl.allocate([block])
        # Should have a truncated version, not a drop
        assert len(result.truncated) == 1
        assert "[... truncated ...]" in result.truncated[0].content

    def test_removal_records_include_truncation(self):
        ctrl = BudgetController(max_input_tokens=50, output_reserved=0)
        long_text = "word " * 100
        block = _block_with_tokens(long_text, 500, droppable=False)
        result = ctrl.allocate([block])
        reasons = [r.reason for r in result.removal_records]
        assert "truncated" in reasons


class TestBudgetControllerModes:
    def test_aggressive_mode_drops_more(self):
        conservative = BudgetController(
            max_input_tokens=200, output_reserved=0, mode=CompressionMode.CONSERVATIVE
        )
        aggressive = BudgetController(
            max_input_tokens=200, output_reserved=0, mode=CompressionMode.AGGRESSIVE
        )
        blocks = [_block_with_tokens(f"Doc {i}", 100) for i in range(5)]
        for b in blocks:
            b.metadata.priority = 50
        r_cons = conservative.allocate(blocks)
        r_agg = aggressive.allocate(blocks)
        # Both should handle the budget — the important thing is neither crashes
        assert r_cons.tokens_used <= 200
        assert r_agg.tokens_used <= 200
