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


# ---------------------------------------------------------------------------
# BudgetPlannerV2 — family-aware pre-allocation
# ---------------------------------------------------------------------------

# Map block type values to budget families
_FAMILY_MAP: dict[str, str] = {
    "system_prompt": "schema_core",
    "tool_schema": "schema_core",
    "retrieved_doc": "evidence",
    "few_shot_example": "evidence",
    "structured_data": "constraints",
    "conversation_history": "memory",
    "user_message": "background",
    "assistant_message": "background",
    "tool_result": "constraints",
}

# Hard floors: guaranteed regardless of content size (schema_core only).
# Soft targets: reserved UP TO this fraction, but only if the family actually
# has that much content — unused fraction flows back to other families.
_FAMILY_HARD_FLOORS: dict[str, float] = {
    "schema_core": 0.10,   # Hard floor — schemas must survive
}
_FAMILY_SOFT_TARGETS: dict[str, float] = {
    "schema_core": 0.20,   # Soft target for schema (beyond the hard floor)
    "evidence": 0.30,       # Soft target for retrieved docs / examples
    "constraints": 0.08,    # Soft target for structured data
    "memory": 0.08,         # Soft target for history
    "background": 0.05,     # Soft target for messages
}
# Back-compat alias used elsewhere in the class
_FAMILY_MINIMUMS = _FAMILY_SOFT_TARGETS


class BudgetPlannerV2:
    """
    Family-aware budget planner that wraps BudgetController.

    Instead of purely greedy priority-based selection, it:
      1. Guarantees minimum budget fractions for each block family.
      2. Distributes remaining budget greedily by priority.
      3. Delegates overflow handling to BudgetController internals.

    This prevents evidence blocks from being starved by a very large
    schema_core block, and vice versa.
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
        self._ctrl = BudgetController(
            max_input_tokens=max_input_tokens,
            output_reserved=output_reserved,
            mode=mode,
        )

    @property
    def available_tokens(self) -> int:
        return self._ctrl.available_tokens

    def allocate(self, blocks: list["Block"]) -> BudgetResult:
        """
        Fit blocks into the available token budget using family-aware pre-allocation.

        Algorithm:
          1. Group blocks by family.
          2. Compute minimum token budget per family.
          3. First pass: fill family minimums (high-priority blocks first).
          4. Second pass: remaining budget distributed greedily by priority.
          5. Delegate actual keep/drop/truncate decisions to BudgetController.
        """
        available = self.available_tokens

        # --- Phase 0: Protected / immutable blocks (must include, skip family logic) ---
        protected: list["Block"] = []
        carvable: list["Block"] = []

        for b in blocks:
            if b.metadata.protected or b.metadata.mutability == Mutability.IMMUTABLE:
                protected.append(b)
            else:
                carvable.append(b)

        protected_tokens = sum(b.token_count or 0 for b in protected)
        remaining_after_protected = max(0, available - protected_tokens)

        if remaining_after_protected == 0:
            # Nothing left for carvable — fall back to base controller
            return self._ctrl.allocate(blocks)

        # --- Phase 1: Group carvable blocks by family ---
        families: dict[str, list["Block"]] = {family: [] for family in _FAMILY_MINIMUMS}
        for b in carvable:
            family = _FAMILY_MAP.get(b.block_type.value, "background")
            families[family].append(b)

        # Sort each family by effective_priority desc
        for family in families:
            families[family].sort(key=lambda b: b.effective_priority(), reverse=True)

        # --- Phase 2: Compute SOFT minimum budgets ---
        # Reservation is capped to actual content in the family — unused quota
        # flows back to the general pool so it doesn't choke overall compression.
        family_min_tokens: dict[str, int] = {}
        for family, fraction in _FAMILY_SOFT_TARGETS.items():
            target = int(remaining_after_protected * fraction)
            # Cap to actual tokens available in this family
            actual = sum(b.token_count or 0 for b in families[family])
            # Hard floor: schema_core gets at least the hard floor if content exists
            hard = int(remaining_after_protected * _FAMILY_HARD_FLOORS.get(family, 0.0))
            family_min_tokens[family] = max(min(target, actual), hard)

        # --- Phase 3: First pass — fill minimums ---
        selected_in_pass1: list["Block"] = []
        tokens_committed_pass1 = 0

        for family in _FAMILY_SOFT_TARGETS:
            family_budget = family_min_tokens[family]
            family_tokens_used = 0
            for b in families[family]:
                bt = b.token_count or 0
                if family_tokens_used + bt <= family_budget:
                    selected_in_pass1.append(b)
                    family_tokens_used += bt
                    tokens_committed_pass1 += bt

        selected_ids_pass1 = {id(b) for b in selected_in_pass1}

        # --- Phase 4: Second pass — remaining budget, greedy by priority ---
        remaining_budget = remaining_after_protected - tokens_committed_pass1
        remaining_carvable = [b for b in carvable if id(b) not in selected_ids_pass1]
        remaining_carvable.sort(key=lambda b: b.effective_priority(), reverse=True)

        selected_in_pass2: list["Block"] = []
        for b in remaining_carvable:
            bt = b.token_count or 0
            if bt <= remaining_budget:
                selected_in_pass2.append(b)
                remaining_budget -= bt

        # --- Phase 5: Delegate to BudgetController with ordered blocks ---
        # Re-order: protected + pass1 + pass2, everything else marked to drop
        final_candidate_ids = (
            {id(b) for b in protected}
            | {id(b) for b in selected_in_pass1}
            | {id(b) for b in selected_in_pass2}
        )

        # Build ordered list for BudgetController
        ordered_blocks: list["Block"] = []
        # Protected first (they'll be picked up in phase 1 of BudgetController)
        ordered_blocks.extend(protected)
        # Then pass1 selected (highest-priority blocks from each family minimum)
        ordered_blocks.extend(selected_in_pass1)
        # Then pass2 selected (remaining budget, greedy)
        ordered_blocks.extend(selected_in_pass2)
        # Then unselected blocks (they'll likely be dropped)
        unselected = [b for b in carvable if id(b) not in final_candidate_ids]
        # Sort unselected by priority so the controller can still try to fit them
        unselected.sort(key=lambda b: b.effective_priority(), reverse=True)
        ordered_blocks.extend(unselected)

        return self._ctrl.allocate(ordered_blocks)
