"""Conversation-state compactor — converts raw history into a compact state object."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from optimizerllm.core.block import Block
    from optimizerllm.slot_guard.guard import SlotIndex

from optimizerllm.core.block import (
    Block,
    BlockMetadata,
    BlockType,
    CompressionPolicy,
    Mutability,
)


@dataclass
class ConversationState:
    """Structured summary of a conversation's history."""

    active_goals: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    decisions_made: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    facts: list[str] = field(default_factory=list)
    tool_results_summary: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not any([
            self.active_goals, self.constraints, self.decisions_made,
            self.open_questions, self.facts, self.tool_results_summary,
        ])

    def render(self) -> str:
        """Render state as a compact text block."""
        sections: list[str] = []

        def _section(title: str, items: list[str]) -> None:
            if items:
                bullet_items = "\n".join(f"- {item}" for item in items)
                sections.append(f"[{title}]\n{bullet_items}")

        _section("GOALS", self.active_goals)
        _section("CONSTRAINTS", self.constraints)
        _section("DECISIONS", self.decisions_made)
        _section("OPEN", self.open_questions)
        _section("FACTS", self.facts)
        _section("TOOL_RESULTS", self.tool_results_summary)

        return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Pattern matchers for state extraction (all deterministic / regex-based)
# ---------------------------------------------------------------------------

_GOAL_PATTERNS = [
    re.compile(r"\b(I want to|I need to|my goal is|please|help me|can you)\b", re.I),
]
_CONSTRAINT_PATTERNS = [
    re.compile(r"\b(must|should|cannot|don't|never|always|only|require)\b", re.I),
]
_DECISION_PATTERNS = [
    re.compile(r"\b(I decided|we agreed|let's use|I chose|going with|we will)\b", re.I),
]
_QUESTION_PATTERNS = [
    re.compile(r"\?$", re.MULTILINE),
]
_FACT_PATTERNS = [
    re.compile(r"\b(the (current|existing|actual)|as of|note that|FYI|for context)\b", re.I),
]


def _classify_sentence(sentence: str) -> str | None:
    """Return the state bucket name for a sentence, or None if unclassifiable."""
    s = sentence.strip()
    if not s or len(s.split()) < 3:
        return None
    for pat in _GOAL_PATTERNS:
        if pat.search(s):
            return "goal"
    for pat in _CONSTRAINT_PATTERNS:
        if pat.search(s):
            return "constraint"
    for pat in _DECISION_PATTERNS:
        if pat.search(s):
            return "decision"
    for pat in _QUESTION_PATTERNS:
        if pat.search(s):
            return "question"
    for pat in _FACT_PATTERNS:
        if pat.search(s):
            return "fact"
    return None


def _split_sentences(text: str) -> list[str]:
    """Rough sentence splitter — good enough for state extraction."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


class ConversationCompactor:
    """
    Compacts conversation history blocks into a structured state object
    plus a short recency tail.

    This is rule-driven, not freeform LLM summarization. It:
      1. Classifies sentences from history into state buckets.
      2. Returns the compact state as a synthetic block.
      3. Retains only the most recent N turns verbatim (recency tail).
    """

    def __init__(self, recency_tail: int = 3) -> None:
        """
        recency_tail: how many of the most recent history blocks to keep verbatim.
        """
        self.recency_tail = recency_tail

    def compact(
        self,
        history_blocks: list["Block"],
        slot_index: "SlotIndex | None" = None,
    ) -> tuple[list["Block"], "Block | None"]:
        """
        Compact history blocks.

        If slot_index is provided, any block that contains a protected slot is
        moved into the recency tail (kept verbatim) rather than being compacted,
        to avoid losing brittle facts during state extraction.

        Returns:
          (recency_tail_blocks, state_block_or_None)

        The caller should prepend the state_block before the tail blocks.
        """
        if not history_blocks:
            return [], None

        # Sort by position to ensure chronological order
        sorted_blocks = sorted(
            history_blocks,
            key=lambda b: (b.metadata.position or 0),
        )

        tail = sorted_blocks[-self.recency_tail :]
        older = sorted_blocks[: -self.recency_tail] if len(sorted_blocks) > self.recency_tail else []

        # Slot-aware: promote blocks with protected slots from older→tail
        if slot_index is not None and older:
            promoted: list["Block"] = []
            remaining_older: list["Block"] = []
            for b in older:
                if slot_index.block_has_slot(b.id):
                    promoted.append(b)
                else:
                    remaining_older.append(b)
            tail = promoted + tail
            older = remaining_older

        if not older:
            # Nothing old enough to compact
            return tail, None

        state = self._extract_state(older)
        if state.is_empty():
            return tail, None

        state_block = Block(
            content=state.render(),
            block_type=BlockType.CONVERSATION_HISTORY,
            role="system",
            metadata=BlockMetadata(
                source="compacted_history",
                priority=55,
                mutability=Mutability.COMPRESSIBLE,
                cacheability=False,
                compression_policy=CompressionPolicy.DEFAULT,
                protected=False,
            ),
        )
        return tail, state_block

    def _extract_state(self, blocks: list["Block"]) -> ConversationState:
        state = ConversationState()

        # Deduplicate extracted items
        seen: set[str] = set()

        def _add(bucket: list[str], item: str) -> None:
            norm = item.lower().strip()
            if norm not in seen and len(norm) > 10:
                seen.add(norm)
                bucket.append(item.strip())

        for block in blocks:
            # Tool results get special treatment
            if block.block_type == BlockType.TOOL_RESULT:
                summary = block.content[:200].replace("\n", " ")
                _add(state.tool_results_summary, summary)
                continue

            for sentence in _split_sentences(block.content):
                bucket_name = _classify_sentence(sentence)
                if bucket_name == "goal":
                    _add(state.active_goals, sentence)
                elif bucket_name == "constraint":
                    _add(state.constraints, sentence)
                elif bucket_name == "decision":
                    _add(state.decisions_made, sentence)
                elif bucket_name == "question":
                    _add(state.open_questions, sentence)
                elif bucket_name == "fact":
                    _add(state.facts, sentence)

        # Limit each bucket to avoid runaway growth
        state.active_goals = state.active_goals[:5]
        state.constraints = state.constraints[:5]
        state.decisions_made = state.decisions_made[:5]
        state.open_questions = state.open_questions[:3]
        state.facts = state.facts[:5]
        state.tool_results_summary = state.tool_results_summary[:3]

        return state
