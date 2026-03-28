"""Compression planner — a rule engine that decides per-block strategy."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from optimizerllm.core.block import Block

from optimizerllm.core.block import (
    BlockType,
    CompressionMode,
    CompressionPolicy,
    Mutability,
)


class CompressionPlanner:
    """
    Decides the compression strategy for each block based on:
      - The global compression mode (lossless / conservative / aggressive)
      - The block's semantic type
      - The block's explicit metadata overrides

    This is a rule engine — no model calls, fully deterministic.
    It does NOT modify content; it sets policy flags that downstream
    components (selector, compactor, budget controller) respect.
    """

    def __init__(self, mode: CompressionMode = CompressionMode.CONSERVATIVE) -> None:
        self.mode = mode

    def plan(self, blocks: list["Block"]) -> list["Block"]:
        """Return blocks with updated metadata reflecting planned strategy."""
        return [self._plan_block(b) for b in blocks]

    def _plan_block(self, block: "Block") -> "Block":
        if block.metadata.protected:
            # Protected blocks are never touched
            return block

        # Explicit policy overrides take precedence
        if block.metadata.compression_policy not in (
            CompressionPolicy.DEFAULT,
        ):
            return block

        # Apply type-based rules
        policy, mutability = self._type_rules(block.block_type)

        # Escalate based on mode
        if self.mode == CompressionMode.LOSSLESS:
            # In lossless mode nothing becomes droppable
            if mutability == Mutability.DROPPABLE:
                mutability = Mutability.COMPRESSIBLE
            policy = CompressionPolicy.LOSSLESS_ONLY

        elif self.mode == CompressionMode.AGGRESSIVE:
            # In aggressive mode compressible becomes droppable (except system/tool)
            if (
                mutability == Mutability.COMPRESSIBLE
                and block.block_type
                not in (BlockType.SYSTEM_PROMPT, BlockType.TOOL_SCHEMA)
            ):
                mutability = Mutability.DROPPABLE

        meta = block.metadata
        # Only update if the planner has something new to say
        if (
            meta.compression_policy == CompressionPolicy.DEFAULT
            and (
                meta.mutability != mutability
                or meta.compression_policy != policy
            )
        ):
            from dataclasses import replace as dc_replace
            new_meta = dc_replace(meta, mutability=mutability, compression_policy=policy)
            return block.clone_with(metadata=new_meta)

        return block

    def _type_rules(
        self, block_type: BlockType
    ) -> tuple[CompressionPolicy, Mutability]:
        """
        Return (compression_policy, mutability) for a block type.

        Priority of types from safest to most compressible:
          system_prompt < tool_schema < few_shot_example
          < structured_data < retrieved_doc < conversation_history
          < user/assistant messages
        """
        rules: dict[BlockType, tuple[CompressionPolicy, Mutability]] = {
            BlockType.SYSTEM_PROMPT: (
                CompressionPolicy.CACHE_ANCHOR,
                Mutability.IMMUTABLE,
            ),
            BlockType.TOOL_SCHEMA: (
                CompressionPolicy.CACHE_ANCHOR,
                Mutability.COMPRESSIBLE,
            ),
            BlockType.FEW_SHOT_EXAMPLE: (
                CompressionPolicy.CACHE_ANCHOR,
                Mutability.COMPRESSIBLE,
            ),
            BlockType.STRUCTURED_DATA: (
                CompressionPolicy.DEFAULT,
                Mutability.COMPRESSIBLE,
            ),
            BlockType.RETRIEVED_DOC: (
                CompressionPolicy.AGGRESSIVE,
                Mutability.DROPPABLE,
            ),
            BlockType.CONVERSATION_HISTORY: (
                CompressionPolicy.AGGRESSIVE,
                Mutability.DROPPABLE,
            ),
            BlockType.USER_MESSAGE: (
                CompressionPolicy.PRIORITY_KEEP,
                Mutability.COMPRESSIBLE,
            ),
            BlockType.ASSISTANT_MESSAGE: (
                CompressionPolicy.DEFAULT,
                Mutability.DROPPABLE,
            ),
            BlockType.TOOL_RESULT: (
                CompressionPolicy.DEFAULT,
                Mutability.DROPPABLE,
            ),
            BlockType.CODE_SNIPPET: (
                CompressionPolicy.LOSSLESS_ONLY,
                Mutability.COMPRESSIBLE,
            ),
        }
        return rules.get(
            block_type,
            (CompressionPolicy.DEFAULT, Mutability.COMPRESSIBLE),
        )
