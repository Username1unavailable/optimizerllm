"""
TaskSynth — infer the real task from context to guide compression decisions.

Inspired by TPC (Task-agnostic Prompt Compression with Context-aware Sentence
Embedding and Reward-guided Task Descriptor). This deterministic implementation
produces a task descriptor from signal in the system prompt, query structure,
tool schemas, and conversation history — no LLM call required.

The descriptor is consumed by:
  - QueryAwareSelector: adjusts relevance score weights per task
  - GranularSelector: adjusts trimming aggressiveness per task
  - SlotGuard: selects which slot types are most critical per task
  - ConversationCompactor: adjusts recency tail per task
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from optimizerllm.core.block import Block, BlockType


class TaskType(str, Enum):
    RAG             = "rag"             # Answer question from retrieved docs
    MATH            = "math"            # Arithmetic / quantitative reasoning
    CODE            = "code"            # Write, explain, or debug code
    SUMMARIZATION   = "summarization"   # Distil long content into short form
    CHAT            = "chat"            # Multi-turn conversational assistance
    TOOL_USE        = "tool_use"        # Agent with tool/function calls
    CLASSIFICATION  = "classification"  # Categorise, label, or route
    EXTRACTION      = "extraction"      # Pull structured data from text
    GENERAL         = "general"         # Fallback when signal is mixed


@dataclass
class RetentionPolicy:
    """
    Per-task weights that bias the selector and granular trimmer.

    Each value is a multiplier applied to base relevance scores.
    Values > 1.0 mean "be more conservative about dropping this type."
    Values < 1.0 mean "it's OK to compress this type more aggressively."
    """
    doc_weight: float = 1.0          # Retrieved documents
    history_weight: float = 1.0      # Conversation history turns
    system_weight: float = 1.0       # System prompt sections
    example_weight: float = 1.0      # Few-shot examples
    tool_weight: float = 1.0         # Tool schemas
    numeric_slots: bool = True       # Must protect numbers
    name_slots: bool = True          # Must protect person names
    api_slots: bool = True           # Must protect API/schema fields
    constraint_slots: bool = True    # Must protect constraint sentences


_TASK_POLICIES: dict[TaskType, RetentionPolicy] = {
    TaskType.RAG: RetentionPolicy(
        doc_weight=1.5,       # Documents are the core signal
        history_weight=0.6,   # History is mostly noise for point-in-time QA
        example_weight=0.8,
        numeric_slots=True,
        name_slots=True,
        api_slots=False,
        constraint_slots=False,
    ),
    TaskType.MATH: RetentionPolicy(
        doc_weight=1.3,
        history_weight=0.7,
        numeric_slots=True,   # Critical — never drop numbers
        name_slots=False,
        api_slots=False,
        constraint_slots=True,
    ),
    TaskType.CODE: RetentionPolicy(
        doc_weight=1.2,
        history_weight=0.8,
        example_weight=1.3,   # Examples are very useful for code
        tool_weight=1.4,
        api_slots=True,       # Function names / parameter names are critical
        numeric_slots=True,
        name_slots=False,
        constraint_slots=True,
    ),
    TaskType.SUMMARIZATION: RetentionPolicy(
        doc_weight=2.0,       # Must keep the entire source document
        history_weight=0.5,
        example_weight=0.5,
        numeric_slots=True,
        name_slots=True,
        api_slots=False,
        constraint_slots=False,
    ),
    TaskType.CHAT: RetentionPolicy(
        doc_weight=1.0,
        history_weight=1.4,   # Chat depends on history
        example_weight=0.7,
        numeric_slots=True,
        name_slots=True,      # Names are critical in chat
        api_slots=False,
        constraint_slots=True,
    ),
    TaskType.TOOL_USE: RetentionPolicy(
        doc_weight=1.0,
        history_weight=0.9,
        tool_weight=2.0,      # Tool schemas must never be dropped
        api_slots=True,       # API field names are critical
        numeric_slots=True,
        name_slots=False,
        constraint_slots=True,
    ),
    TaskType.CLASSIFICATION: RetentionPolicy(
        doc_weight=1.2,
        history_weight=0.6,
        example_weight=1.4,   # Label examples matter
        numeric_slots=False,
        name_slots=False,
        api_slots=False,
        constraint_slots=False,
    ),
    TaskType.EXTRACTION: RetentionPolicy(
        doc_weight=1.5,
        history_weight=0.5,
        api_slots=True,
        numeric_slots=True,
        name_slots=True,
        constraint_slots=False,
    ),
    TaskType.GENERAL: RetentionPolicy(),  # All defaults at 1.0
}


@dataclass
class TaskDescriptor:
    """Output of TaskSynth — consumed throughout the pipeline."""
    task_type: TaskType
    description: str                        # Short human-readable summary
    retention_policy: RetentionPolicy
    confidence: float = 0.0                 # 0–1 how confident we are
    signals: list[str] = field(default_factory=list)  # Which signals fired


# ---------------------------------------------------------------------------
# Signal matchers
# ---------------------------------------------------------------------------

_RAG_SIGNALS = re.compile(
    r"\b(retriev|document|passage|context|given\s+the\s+(follow|text|article)|"
    r"based\s+on\s+(the\s+)?(document|text|follow|context|passage|article)|"
    r"answer\s+(the\s+)?question|use\s+(only\s+)?(the\s+)?provided)\b",
    re.IGNORECASE,
)
_MATH_SIGNALS = re.compile(
    r"\b(calculat|comput|solv|arithmetic|equation|formula|result|total|sum|"
    r"how\s+many|how\s+much|what\s+is\s+\d|percent|percent|ratio|average)\b",
    re.IGNORECASE,
)
_CODE_SIGNALS = re.compile(
    r"\b(code|function|implement|class|method|debug|refactor|algorithm|"
    r"python|javascript|typescript|java|rust|golang|sql|script|program|"
    r"def\s+\w|return|import\s+\w|```)\b",
    re.IGNORECASE,
)
_SUMM_SIGNALS = re.compile(
    r"\b(summar|tldr|tl;dr|abstract|overview|brief|distill|condense|"
    r"what\s+(are\s+the\s+)?(key|main)\s+(point|idea|topic)|in\s+short)\b",
    re.IGNORECASE,
)
_CHAT_SIGNALS = re.compile(
    r"\b(assistant|conversational|helpful|chat|talk|discuss|feel free|"
    r"hi|hello|thanks|please\s+help|can\s+you\s+(help|tell)|i\s+(am|have|want))\b",
    re.IGNORECASE,
)
_TOOL_SIGNALS = re.compile(
    r"\b(tool|function\s+call|plugin|api|endpoint|invoke|use\s+the\s+\w+\s+tool|"
    r"call\s+\w+\s+with|agentic|agent|action|execute)\b",
    re.IGNORECASE,
)
_CLASS_SIGNALS = re.compile(
    r"\b(classif|categor|label|tag|route|assign|determine\s+(the\s+)?(category|type|class)|"
    r"is\s+(this|the)\s+(a|an|the)?\s+\w+\?|sentiment|intent)\b",
    re.IGNORECASE,
)
_EXTRACT_SIGNALS = re.compile(
    r"\b(extract|parse|identify|pull\s+out|find\s+(all\s+)?the|list\s+(all\s+)?the|"
    r"what\s+entities|structured\s+(data|output)|json\s+output|fill\s+(in\s+)?the)\b",
    re.IGNORECASE,
)

_SIGNAL_MAP = [
    (TaskType.RAG,            _RAG_SIGNALS),
    (TaskType.MATH,           _MATH_SIGNALS),
    (TaskType.CODE,           _CODE_SIGNALS),
    (TaskType.SUMMARIZATION,  _SUMM_SIGNALS),
    (TaskType.TOOL_USE,       _TOOL_SIGNALS),
    (TaskType.CLASSIFICATION, _CLASS_SIGNALS),
    (TaskType.EXTRACTION,     _EXTRACT_SIGNALS),
    (TaskType.CHAT,           _CHAT_SIGNALS),
]


class TaskSynth:
    """
    Infer the task type from context without an LLM call.

    Usage::

        synth = TaskSynth()
        descriptor = synth.synthesize(blocks, query="How do I run async tasks?")
    """

    def synthesize(
        self,
        blocks: list["Block"],
        query: str | None = None,
    ) -> TaskDescriptor:
        from optimizerllm.core.block import BlockType

        # Gather text to analyze
        system_text = " ".join(
            b.content for b in blocks if b.block_type == BlockType.SYSTEM_PROMPT
        )
        has_tools = any(b.block_type == BlockType.TOOL_SCHEMA for b in blocks)
        has_docs = any(b.block_type == BlockType.RETRIEVED_DOC for b in blocks)
        has_history = any(b.block_type == BlockType.CONVERSATION_HISTORY for b in blocks)
        history_len = sum(1 for b in blocks if b.block_type == BlockType.CONVERSATION_HISTORY)

        analysis_text = " ".join(filter(None, [system_text, query or ""]))

        # Score each task type
        scores: dict[TaskType, float] = {t: 0.0 for t, _ in _SIGNAL_MAP}

        for task_type, pattern in _SIGNAL_MAP:
            matches = pattern.findall(analysis_text)
            scores[task_type] += len(matches) * 1.0

        # Structural bonuses
        if has_tools:
            scores[TaskType.TOOL_USE] += 3.0
        if has_docs:
            scores[TaskType.RAG] += 2.0
        if has_history and history_len >= 4:
            scores[TaskType.CHAT] += 2.0
        if has_history and not has_docs and not has_tools:
            scores[TaskType.CHAT] += 1.0

        # Pick the winner
        best = max(scores, key=lambda t: scores[t])
        best_score = scores[best]
        total = sum(scores.values()) or 1.0
        confidence = min(best_score / total, 1.0) if best_score > 0 else 0.0

        if best_score == 0:
            best = TaskType.GENERAL
            confidence = 0.0

        fired = [t.value for t, _ in _SIGNAL_MAP if scores[t] > 0]

        policy = _TASK_POLICIES[best]
        description = self._describe(best, has_docs, has_tools, has_history)

        return TaskDescriptor(
            task_type=best,
            description=description,
            retention_policy=policy,
            confidence=confidence,
            signals=fired,
        )

    @staticmethod
    def _describe(
        task: TaskType,
        has_docs: bool,
        has_tools: bool,
        has_history: bool,
    ) -> str:
        parts = [f"Task: {task.value}"]
        if has_docs:
            parts.append("with retrieved documents")
        if has_tools:
            parts.append("with tool schemas")
        if has_history:
            parts.append("in a multi-turn session")
        return " ".join(parts)
