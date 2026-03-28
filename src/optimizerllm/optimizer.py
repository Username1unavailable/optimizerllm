"""ContextOptimizer — the main public entry point for the library."""

from __future__ import annotations

from typing import Any, Sequence

from optimizerllm.budget.controller import BudgetController
from optimizerllm.cache.optimizer import CacheOptimizer
from optimizerllm.compression.compactor import ConversationCompactor
from optimizerllm.compression.granular_selector import GranularSelector
from optimizerllm.compression.planner import CompressionPlanner
from optimizerllm.compression.selector import QueryAwareSelector
from optimizerllm.core.block import Block, BlockType, CompressionMode
from optimizerllm.core.result import OptimizationResult, RemovalRecord, TelemetryReport
from optimizerllm.ingestion.ingester import Ingester
from optimizerllm.normalization.normalizer import Normalizer, TextMinifier
from optimizerllm.recover.recover import Recover, RecoveryRecord
from optimizerllm.rendering.renderer import Renderer
from optimizerllm.slot_guard.guard import SlotGuard, SlotIndex
from optimizerllm.task_synth.synth import TaskDescriptor, TaskSynth
from optimizerllm.token_accounting.counter import TokenBudget, TokenCounter


class ContextOptimizer:
    """
    One-stop context optimization engine.

    Usage::

        optimizer = ContextOptimizer(
            provider="openai",
            model="gpt-4o",
            max_input_tokens=128_000,
            output_budget=4_096,
            mode=CompressionMode.CONSERVATIVE,
        )

        result = optimizer.optimize(
            messages=[{"role": "user", "content": "What is the capital of France?"}],
            system="You are a helpful assistant.",
            documents=["Paris is the capital of France."],
            query="capital of France",
        )

        # result.rendered  →  OpenAI-compatible messages list
        # result.telemetry →  full audit report
        print(result.summary())

    Pipeline order (Phase 1 + 2 + 3):
      1.  Ingest → flat list of Block objects
      2.  Normalize → whitespace, JSON canonicalization, deduplication
      3.  [conservative/aggressive] Minify prose
      4.  Token count every block (pre-normalization snapshot for telemetry)
      5.  Plan compression strategy per block
      5b. [optional] TaskSynth → infer task type and retention policy
      5c. [optional] SlotGuard → extract and index brittle facts (slots)
      6.  [conservative/aggressive] Score blocks against query
      7.  [conservative/aggressive] Filter low-relevance blocks
      8.  [conservative/aggressive] Compact conversation history (slot-aware)
      9.  [Phase 2] Granular trimming → section and sentence level (slot-aware)
      10. Cache-optimize prefix ordering
      11. Budget control → fit blocks into token limit
      11b.[optional] Recover → detect and re-inject lost slots
      12. Render to target format
      13. Collect and finalize telemetry
    """

    def __init__(
        self,
        *,
        provider: str = "openai",
        model: str = "gpt-4o",
        max_input_tokens: int = 128_000,
        output_budget: int = 4_096,
        mode: CompressionMode | str = CompressionMode.CONSERVATIVE,
        output_format: str = "openai",
        selector_threshold: float = 0.12,
        compactor_recency_tail: int = 3,
        # Phase 2 — granular selection
        section_level: bool = True,
        sentence_level: bool = True,
        section_threshold: float = 0.10,
        sentence_threshold: float = 0.08,
        # Phase 3 — slot guard / task synthesis / recovery
        enable_slot_guard: bool = True,
        enable_task_synth: bool = True,
        enable_recover: bool = True,
        recover_max_slots: int = 8,
    ) -> None:
        """
        Args:
            provider: "openai" | "anthropic" | "gemini"
            model: Model name for token counting and cost estimation.
            max_input_tokens: Hard limit on input token budget.
            output_budget: Tokens reserved for the model's output.
            mode: CompressionMode (lossless / conservative / aggressive).
            output_format: "openai" | "anthropic" | "text" | "raw".
            selector_threshold: Minimum block-level relevance score (conservative+).
            compactor_recency_tail: Verbatim history turns to preserve (conservative+).
            section_level: Enable section-level trimming within blocks (Phase 2).
            sentence_level: Enable sentence-level trimming within blocks (Phase 2).
            section_threshold: Min score for a section to survive trimming.
            sentence_threshold: Min score for a sentence to survive trimming.
            enable_slot_guard: Extract brittle facts and protect them from trimming.
            enable_task_synth: Infer task type to bias retention weights.
            enable_recover: Inject recovery note if critical slots were dropped.
            recover_max_slots: Max number of lost slots to recover.
        """
        if isinstance(mode, str):
            mode = CompressionMode(mode)

        self.provider = provider
        self.model = model
        self.mode = mode
        self.output_format = output_format
        self.selector_threshold = selector_threshold
        self.section_level = section_level
        self.sentence_level = sentence_level
        self.enable_slot_guard = enable_slot_guard
        self.enable_task_synth = enable_task_synth
        self.enable_recover = enable_recover

        self._budget = TokenBudget(
            max_input_tokens=max_input_tokens,
            output_reserved=output_budget,
            provider=provider,
            model=model,
        )
        self._ingester = Ingester()
        self._normalizer = Normalizer()
        self._minifier = TextMinifier()
        self._counter = TokenCounter()
        self._planner = CompressionPlanner(mode=mode)
        self._selector = QueryAwareSelector()
        self._granular = GranularSelector(
            section_threshold=section_threshold,
            sentence_threshold=sentence_threshold,
        )
        self._compactor = ConversationCompactor(recency_tail=compactor_recency_tail)
        self._cache_opt = CacheOptimizer(provider=provider)
        self._budget_ctrl = BudgetController(
            max_input_tokens=max_input_tokens,
            output_reserved=output_budget,
            mode=mode,
        )
        self._renderer = Renderer()
        self._task_synth = TaskSynth()
        self._slot_guard = SlotGuard()
        self._recover = Recover(max_slots=recover_max_slots)

    def optimize(
        self,
        *,
        messages: Sequence[dict[str, Any]] | None = None,
        system: str | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
        documents: Sequence[str | dict[str, Any]] | None = None,
        examples: Sequence[str | dict[str, Any]] | None = None,
        history: Sequence[dict[str, Any]] | None = None,
        structured: dict[str, Any] | None = None,
        raw_blocks: Sequence[Block] | None = None,
        query: str | None = None,
    ) -> OptimizationResult:
        """
        Run the full optimization pipeline.

        Args:
            messages: Current-turn messages (OpenAI format).
            system: System prompt string.
            tools: Tool-schema dicts.
            documents: Retrieved documents (strings or {"content":..., "source":...}).
            examples: Few-shot examples.
            history: Older conversation turns (will be compacted if conservative+).
            structured: Arbitrary JSON payload.
            raw_blocks: Pre-built Block objects (bypass ingestion).
            query: Active user query for relevance scoring (defaults to last user message).

        Returns:
            OptimizationResult with rendered output and telemetry.
        """
        telemetry = TelemetryReport(
            compression_mode=self.mode.value,
            provider=self.provider,
            model=self.model,
        )
        all_removed: list[Block] = []

        # ------------------------------------------------------------------
        # Step 1: Ingest
        # ------------------------------------------------------------------
        blocks = self._ingester.ingest(
            messages=messages,
            system=system,
            tools=tools,
            documents=documents,
            examples=examples,
            history=history,
            structured=structured,
            raw_blocks=raw_blocks,
        )
        telemetry.blocks_before = len(blocks)

        # Snapshot raw token count BEFORE any compression (including dedup)
        # so telemetry accurately reflects savings from normalization too.
        raw_counted = self._counter.count_blocks(blocks, provider=self.provider, model=self.model)
        telemetry.tokens_before = self._counter.total_tokens(raw_counted)

        # ------------------------------------------------------------------
        # Step 2: Normalize (lossless — always applied)
        # ------------------------------------------------------------------
        blocks, removed_norm = self._normalizer.run(blocks)
        all_removed.extend(removed_norm)

        # ------------------------------------------------------------------
        # Step 3: Minify prose (conservative and aggressive)
        # ------------------------------------------------------------------
        if self.mode in (CompressionMode.CONSERVATIVE, CompressionMode.AGGRESSIVE):
            blocks = self._minifier.minify_blocks(blocks)

        # ------------------------------------------------------------------
        # Step 4: Token counting (post-normalization)
        # ------------------------------------------------------------------
        blocks = self._counter.count_blocks(blocks, provider=self.provider, model=self.model)
        total_before = self._counter.total_tokens(blocks)

        # Cost estimate before optimization (uses raw pre-normalization count)
        cost_before = self._counter.estimate_cost(
            telemetry.tokens_before,
            provider=self.provider,
            model=self.model,
        )
        telemetry.cost_before_usd = cost_before.total_usd if cost_before.total_usd else None

        # ------------------------------------------------------------------
        # Step 5: Compression planner
        # ------------------------------------------------------------------
        blocks = self._planner.plan(blocks)

        active_query = query or self._extract_last_user_query(blocks)

        # ------------------------------------------------------------------
        # Step 5b: TaskSynth — infer task type and retention policy
        # ------------------------------------------------------------------
        task_descriptor: TaskDescriptor | None = None
        if self.enable_task_synth:
            task_descriptor = self._task_synth.synthesize(blocks, query=active_query)
            telemetry.extra["task_type"] = task_descriptor.task_type.value
            telemetry.extra["task_confidence"] = round(task_descriptor.confidence, 3)

        # ------------------------------------------------------------------
        # Step 5c: SlotGuard — extract brittle facts from all blocks
        # ------------------------------------------------------------------
        slot_index: SlotIndex | None = None
        if self.enable_slot_guard:
            slot_index = self._slot_guard.extract(blocks)
            telemetry.extra["slots_extracted"] = len(slot_index)

        # ------------------------------------------------------------------
        # Step 6 & 7: Query-aware scoring and filtering (conservative+)
        # ------------------------------------------------------------------
        if self.mode in (CompressionMode.CONSERVATIVE, CompressionMode.AGGRESSIVE):
            if active_query:
                blocks = self._selector.score_blocks(
                    blocks,
                    active_query,
                    task_descriptor=task_descriptor,
                )
                kept, removed_sel = self._selector.filter_by_threshold(
                    blocks, threshold=self.selector_threshold
                )
                all_removed.extend(removed_sel)
                blocks = kept

        # ------------------------------------------------------------------
        # Step 8: Conversation compaction (conservative+, slot-aware)
        # ------------------------------------------------------------------
        if self.mode in (CompressionMode.CONSERVATIVE, CompressionMode.AGGRESSIVE):
            history_blocks = [b for b in blocks if b.block_type == BlockType.CONVERSATION_HISTORY]
            other_blocks = [b for b in blocks if b.block_type != BlockType.CONVERSATION_HISTORY]

            if history_blocks:
                tail, state_block = self._compactor.compact(
                    history_blocks, slot_index=slot_index
                )
                blocks = other_blocks
                if state_block is not None:
                    # Re-count the synthetic state block
                    state_block = self._counter.count_blocks([state_block], provider=self.provider, model=self.model)[0]
                    blocks.append(state_block)
                blocks.extend(tail)

        # ------------------------------------------------------------------
        # Step 9: Granular trimming — section and sentence level (slot-aware)
        # ------------------------------------------------------------------
        if self.mode in (CompressionMode.CONSERVATIVE, CompressionMode.AGGRESSIVE):
            if active_query and (self.section_level or self.sentence_level):
                blocks = self._granular.trim_blocks(
                    blocks,
                    active_query,
                    section_level=self.section_level,
                    sentence_level=self.sentence_level,
                    slot_index=slot_index,
                )
                # Re-count tokens after granular trimming (content may have shrunk)
                blocks = self._counter.count_blocks(blocks, provider=self.provider, model=self.model)

        # ------------------------------------------------------------------
        # Step 10: Cache optimization
        # ------------------------------------------------------------------
        blocks, cache_analysis = self._cache_opt.optimize(blocks)
        telemetry.cacheable_prefix_tokens = cache_analysis.cacheable_prefix_tokens

        # ------------------------------------------------------------------
        # Step 11: Budget control
        # ------------------------------------------------------------------
        budget_result = self._budget_ctrl.allocate(blocks)
        final_blocks = budget_result.final_blocks

        for dropped_b in budget_result.dropped:
            all_removed.append(dropped_b)

        telemetry.blocks_removed = [
            *[
                RemovalRecord(
                    block_id=b.id,
                    block_type=b.block_type.value,
                    token_count=b.token_count or 0,
                    reason=b.drop_reason or "removed",
                )
                for b in all_removed
            ],
            *budget_result.removal_records,
        ]

        # ------------------------------------------------------------------
        # Step 11b: Recover — re-inject lost slots after budget control
        # ------------------------------------------------------------------
        if self.enable_recover and slot_index is not None and len(slot_index) > 0:
            final_blocks, recovery_records = self._recover.run(final_blocks, slot_index)
            telemetry.extra["slots_recovered"] = len(recovery_records)

        # ------------------------------------------------------------------
        # Step 12: Render
        # ------------------------------------------------------------------
        rendered_with_ids = self._renderer.render(
            final_blocks, fmt=self.output_format, include_block_ids=True
        )

        # Inject provider cache hints (Anthropic only in v1)
        if self.provider == "anthropic" and isinstance(rendered_with_ids, dict):
            msgs = rendered_with_ids.get("messages", [])
            annotated = self._cache_opt.inject_cache_hints(final_blocks, msgs)
            rendered_with_ids["messages"] = annotated
            rendered = rendered_with_ids
        elif isinstance(rendered_with_ids, list):
            # Clean internal _block_id field for OpenAI format
            rendered = [{k: v for k, v in m.items() if k != "_block_id"} for m in rendered_with_ids]
        else:
            rendered = rendered_with_ids

        # ------------------------------------------------------------------
        # Step 13: Finalize telemetry
        # ------------------------------------------------------------------
        tokens_after = sum(b.token_count or 0 for b in final_blocks)
        telemetry.tokens_after = tokens_after
        telemetry.blocks_after = len(final_blocks)

        cost_after = self._counter.estimate_cost(
            tokens_after,
            cacheable_prefix_tokens=cache_analysis.cacheable_prefix_tokens,
            provider=self.provider,
            model=self.model,
        )
        telemetry.cost_after_usd = cost_after.total_usd if cost_after.total_usd else None
        telemetry.finalize()

        return OptimizationResult(
            blocks=final_blocks,
            rendered=rendered,
            telemetry=telemetry,
            provider=self.provider,
            model=self.model,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_last_user_query(blocks: list[Block]) -> str:
        """Find the last user message to use as the active query."""
        for block in reversed(blocks):
            if block.block_type == BlockType.USER_MESSAGE and block.role == "user":
                return block.content
        return ""
