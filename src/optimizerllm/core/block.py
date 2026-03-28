"""Core block data model — the fundamental unit of context optimization."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class BlockType(str, Enum):
    """Semantic type of a context block."""

    SYSTEM_PROMPT = "system_prompt"
    USER_MESSAGE = "user_message"
    ASSISTANT_MESSAGE = "assistant_message"
    TOOL_SCHEMA = "tool_schema"
    TOOL_RESULT = "tool_result"
    RETRIEVED_DOC = "retrieved_doc"
    FEW_SHOT_EXAMPLE = "few_shot_example"
    CONVERSATION_HISTORY = "conversation_history"
    STRUCTURED_DATA = "structured_data"
    CODE_SNIPPET = "code_snippet"
    UNKNOWN = "unknown"


class Mutability(str, Enum):
    """How the block may be modified during compression."""

    IMMUTABLE = "immutable"      # Must not be changed at all
    COMPRESSIBLE = "compressible"  # May be shortened but not dropped
    DROPPABLE = "droppable"      # May be removed if budget demands it


class CompressionPolicy(str, Enum):
    """Block-level compression policy hint."""

    DEFAULT = "default"          # Use mode-level policy
    LOSSLESS_ONLY = "lossless_only"  # Only safe normalization
    PRIORITY_KEEP = "priority_keep"  # Try hard to keep this block
    AGGRESSIVE = "aggressive"    # Compress/drop aggressively
    CACHE_ANCHOR = "cache_anchor"  # Part of the stable cache prefix


class CompressionMode(str, Enum):
    """Global compression aggressiveness."""

    LOSSLESS = "lossless"        # Only remove provable waste
    CONSERVATIVE = "conservative"  # + query-aware filtering, state compaction
    AGGRESSIVE = "aggressive"    # + deeper trimming, tighter budgets


@dataclass
class BlockMetadata:
    """Metadata attached to every block, driving compression decisions."""

    source: str | None = None
    priority: int = 50              # 0 (lowest) to 100 (highest)
    mutability: Mutability = Mutability.COMPRESSIBLE
    cacheability: bool = True
    compression_policy: CompressionPolicy = CompressionPolicy.DEFAULT
    protected: bool = False         # Alias for immutable + top priority
    position: int | None = None     # Original insertion order
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.protected:
            self.mutability = Mutability.IMMUTABLE
            self.priority = max(self.priority, 90)


@dataclass
class Block:
    """
    A single semantic unit of context.

    Each block carries its content, type, and metadata that drive all
    downstream compression, scoring, and rendering decisions.
    """

    content: str
    block_type: BlockType
    metadata: BlockMetadata = field(default_factory=BlockMetadata)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    role: str | None = None         # "system" | "user" | "assistant" for messages
    token_count: int | None = None  # Filled in by TokenCounter
    score: float | None = None      # Query-relevance score (0–1)
    drop_reason: str | None = None  # Set when block is removed

    def is_removable(self) -> bool:
        """True when the block can be dropped by the budget controller."""
        return (
            not self.metadata.protected
            and self.metadata.mutability == Mutability.DROPPABLE
        )

    def is_compressible(self) -> bool:
        """True when the block can be shortened but not removed."""
        return (
            not self.metadata.protected
            and self.metadata.mutability in (Mutability.COMPRESSIBLE, Mutability.DROPPABLE)
        )

    def effective_priority(self) -> float:
        """Combined priority that factors in relevance score."""
        base = float(self.metadata.priority)
        if self.score is not None:
            return base * (0.6 + 0.4 * self.score)
        return base

    def clone_with(self, **kwargs: Any) -> "Block":
        """Return a shallow copy with selected fields overridden."""
        from dataclasses import replace
        return replace(self, **kwargs)


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

def make_system_block(
    content: str,
    *,
    protected: bool = True,
    priority: int = 95,
    source: str | None = None,
) -> Block:
    return Block(
        content=content,
        block_type=BlockType.SYSTEM_PROMPT,
        role="system",
        metadata=BlockMetadata(
            source=source,
            priority=priority,
            protected=protected,
            mutability=Mutability.IMMUTABLE if protected else Mutability.COMPRESSIBLE,
            cacheability=True,
            compression_policy=CompressionPolicy.CACHE_ANCHOR,
        ),
    )


def make_message_block(
    role: str,
    content: str,
    *,
    priority: int = 50,
    block_type: BlockType | None = None,
    source: str | None = None,
) -> Block:
    if block_type is None:
        mapping = {
            "user": BlockType.USER_MESSAGE,
            "assistant": BlockType.ASSISTANT_MESSAGE,
            "system": BlockType.SYSTEM_PROMPT,
        }
        block_type = mapping.get(role, BlockType.UNKNOWN)
    return Block(
        content=content,
        block_type=block_type,
        role=role,
        metadata=BlockMetadata(
            source=source,
            priority=priority,
            mutability=Mutability.DROPPABLE,
            cacheability=False,
        ),
    )


def make_tool_schema_block(content: str, *, source: str | None = None) -> Block:
    return Block(
        content=content,
        block_type=BlockType.TOOL_SCHEMA,
        metadata=BlockMetadata(
            source=source,
            priority=80,
            mutability=Mutability.COMPRESSIBLE,
            cacheability=True,
            compression_policy=CompressionPolicy.CACHE_ANCHOR,
        ),
    )


def make_retrieved_doc_block(
    content: str,
    *,
    source: str | None = None,
    priority: int = 40,
) -> Block:
    return Block(
        content=content,
        block_type=BlockType.RETRIEVED_DOC,
        metadata=BlockMetadata(
            source=source,
            priority=priority,
            mutability=Mutability.DROPPABLE,
            cacheability=False,
            compression_policy=CompressionPolicy.AGGRESSIVE,
        ),
    )


def make_example_block(content: str, *, source: str | None = None) -> Block:
    return Block(
        content=content,
        block_type=BlockType.FEW_SHOT_EXAMPLE,
        metadata=BlockMetadata(
            source=source,
            priority=70,
            mutability=Mutability.DROPPABLE,
            cacheability=True,
            compression_policy=CompressionPolicy.CACHE_ANCHOR,
        ),
    )
