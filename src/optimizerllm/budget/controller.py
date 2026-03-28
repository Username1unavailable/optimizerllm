"""Budget controller — a knapsack policy that fits blocks into a token budget."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from optimizerllm.core.block import Block

from optimizerllm.core.block import Block, BlockType, CompressionMode, Mutability
from optimizerllm.core.result import RemovalRecord


_TRUNCATION_SUFFIX = "\n[... truncated ...]"

# Minimum tokens to keep in a truncated block (must have something useful)
_MIN_TRUNCATED_TOKENS = 10


@dataclass
class BudgetResult:
    """Output of the budget controller."""

    kept: list["Block"] = field(default_factory=list)
    dropped: list["Block"] = field(default_factory=list)
    truncated: list["Block"] = field(default_factory=list)       # The truncated versions
    removal_records: list[RemovalRecord] = field(default_factory=list)
    tokens_used: int = 0
    tokens_budget: int = 0

    @property
    def final_blocks(self) -> list["Block"]:
        """Kept + truncated blocks, in original position order."""
        all_b = self.kept + self.truncated
        return sorted(all_b, key=lambda b: (b.metadata.position or 0))


class BudgetController:
    """
    Selects and trims blocks to fit within a token budget.

    Algorithm:
      1. Always include protected / immutable blocks (they count against the budget).
      2. Sort remaining blocks by effective_priority() descending.
      3. Greedily include blocks until the budget is saturated.
      4. For compressible blocks that don't fit whole, attempt truncation.
      5. Droppable blocks that don't fit are dropped outright.

    The mode influences which blocks are eligible for truncation vs. dropping.
    """

    def __init__(
        self,
        max_input_tokens: int,
        output_reserved: int = 1024,
        mode: CompressionMode = CompressionMode.CONSERVATIVE,
    ) -> None:
        self.max_input_tokens = max_input_tokens
        self.output_reserved = output_reserved
        self.mode = mode

    @property
    def available_tokens(self) -> int:
        return self.max_input_tokens - self.output_reserved

    def allocate(self, blocks: list["Block"]) -> BudgetResult:
        """
        Fit blocks into the available token budget.

        Returns a BudgetResult with kept/dropped/truncated blocks and records.
        """
        result = BudgetResult(tokens_budget=self.available_tokens)
        tokens_used = 0

        # --- Phase 1: Protected / immutable blocks (must include) ---
        protected: list["Block"] = []
        carvable: list["Block"] = []

        for b in blocks:
            if b.metadata.protected or b.metadata.mutability == Mutability.IMMUTABLE:
                protected.append(b)
            else:
                carvable.append(b)

        for b in protected:
            tokens_used += b.token_count or 0
            result.kept.append(b)

        # If protected blocks alone exceed budget, nothing else we can do
        if tokens_used >= self.available_tokens:
            for b in carvable:
                record = self._drop_record(b, "budget_exceeded_by_protected")
                result.dropped.append(b.clone_with(drop_reason="budget_exceeded_by_protected"))
                result.removal_records.append(record)
            result.tokens_used = tokens_used
            return result

        # --- Phase 2: Sort carvable blocks by priority ---
        carvable.sort(key=lambda b: b.effective_priority(), reverse=True)

        remaining_budget = self.available_tokens - tokens_used

        for b in carvable:
            block_tokens = b.token_count or 0

            if block_tokens == 0:
                result.kept.append(b)
                continue

            if block_tokens <= remaining_budget:
                # Block fits whole
                result.kept.append(b)
                tokens_used += block_tokens
                remaining_budget -= block_tokens
            else:
                # Block doesn't fit — try truncation or drop
                outcome = self._handle_overflow(b, remaining_budget)
                if outcome == "drop":
                    record = self._drop_record(b, "budget_exceeded")
                    result.dropped.append(b.clone_with(drop_reason="budget_exceeded"))
                    result.removal_records.append(record)
                elif outcome == "keep_truncated":
                    truncated, used = self._truncate(b, remaining_budget)
                    result.truncated.append(truncated)
                    result.removal_records.append(
                        RemovalRecord(
                            block_id=b.id,
                            block_type=b.block_type.value,
                            token_count=block_tokens - used,
                            reason="truncated",
                            score=b.score,
                            priority=b.metadata.priority,
                        )
                    )
                    tokens_used += used
                    remaining_budget -= used
                # If outcome is "skip" (nothing fits), just drop
                else:
                    record = self._drop_record(b, "budget_no_space")
                    result.dropped.append(b.clone_with(drop_reason="budget_no_space"))
                    result.removal_records.append(record)

        result.tokens_used = tokens_used
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _handle_overflow(
        self, block: "Block", remaining: int
    ) -> str:
        """Decide what to do when a block doesn't fit. Returns "drop" | "keep_truncated" | "skip"."""
        if remaining <= 0:
            return "skip"

        mutability = block.metadata.mutability

        if mutability == Mutability.DROPPABLE:
            if self.mode == CompressionMode.AGGRESSIVE:
                # Try to truncate even droppable blocks in aggressive mode
                return "keep_truncated" if remaining >= _MIN_TRUNCATED_TOKENS else "drop"
            return "drop"

        if mutability == Mutability.COMPRESSIBLE:
            return "keep_truncated" if remaining >= _MIN_TRUNCATED_TOKENS else "drop"

        # IMMUTABLE — should never reach here (caught in phase 1)
        return "drop"

    def _truncate(
        self, block: "Block", token_budget: int
    ) -> tuple["Block", int]:
        """
        Truncate block content to fit within token_budget tokens.

        Uses a character-based heuristic: ~4 chars per token.
        Returns (truncated_block, tokens_used).
        """
        chars_budget = max(token_budget * 4 - len(_TRUNCATION_SUFFIX), 50)
        content = block.content
        if len(content) > chars_budget:
            content = content[:int(chars_budget)] + _TRUNCATION_SUFFIX

        # Rough token estimate for the truncated content
        tokens_used = max(1, len(content) // 4)
        return block.clone_with(content=content), min(tokens_used, token_budget)

    @staticmethod
    def _drop_record(block: "Block", reason: str) -> RemovalRecord:
        return RemovalRecord(
            block_id=block.id,
            block_type=block.block_type.value,
            token_count=block.token_count or 0,
            reason=reason,
            score=block.score,
            priority=block.metadata.priority,
        )
