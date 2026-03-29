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


# ---------------------------------------------------------------------------
# MemoryGraph data structures
# ---------------------------------------------------------------------------

@dataclass
class MemoryNode:
    node_id: str
    node_type: str    # "goal" | "constraint" | "decision" | "entity" | "question" | "fact"
    content: str
    turn: int         # which conversation turn this came from
    confidence: float = 1.0


@dataclass
class MemoryEdge:
    from_id: str
    to_id: str
    edge_type: str   # "updates" | "supersedes" | "contradicts" | "supports"


@dataclass
class MemoryGraph:
    nodes: list[MemoryNode] = field(default_factory=list)
    edges: list[MemoryEdge] = field(default_factory=list)

    def add_node(self, node: MemoryNode) -> None:
        """
        Add a node, deduplicating by content similarity.

        If a node of the same type exists with > 60% token overlap, keep the
        more recent one and add an "updates" edge from newer → older.
        """
        for existing in self.nodes:
            if existing.node_type != node.node_type:
                continue
            similarity = _token_overlap(existing.content, node.content)
            if similarity > 0.60:
                if node.turn >= existing.turn:
                    # New node supersedes old one — add updates edge, remove old
                    self.nodes.remove(existing)
                    self.nodes.append(node)
                    self.edges.append(
                        MemoryEdge(
                            from_id=node.node_id,
                            to_id=existing.node_id,
                            edge_type="updates",
                        )
                    )
                # else: existing is newer, don't add
                return
        self.nodes.append(node)

    def add_edge(self, edge: MemoryEdge) -> None:
        self.edges.append(edge)

    def resolve_contradictions(self) -> None:
        """
        For pairs of nodes where edge_type="contradicts", remove the older
        node (lower turn) and keep the newer one.
        """
        contradicts_edges = [e for e in self.edges if e.edge_type == "contradicts"]
        to_remove: set[str] = set()

        node_map = {n.node_id: n for n in self.nodes}

        for edge in contradicts_edges:
            from_node = node_map.get(edge.from_id)
            to_node = node_map.get(edge.to_id)
            if from_node is None or to_node is None:
                continue
            # Remove the older node
            if from_node.turn >= to_node.turn:
                to_remove.add(to_node.node_id)
            else:
                to_remove.add(from_node.node_id)

        self.nodes = [n for n in self.nodes if n.node_id not in to_remove]

    def render(self) -> str:
        """
        Output the resolved graph as structured text.

        Format:
          [GOALS]
          - goal 1
          - goal 2 (updates: goal 0)

          [CONSTRAINTS]
          - constraint 1

          [DECISIONS]
          - decision 1 (supersedes: earlier decision)
        """
        # Build edge lookup: from_id → list of (to_id, edge_type)
        edge_lookup: dict[str, list[tuple[str, str]]] = {}
        for edge in self.edges:
            edge_lookup.setdefault(edge.from_id, []).append((edge.to_id, edge.edge_type))

        # Group nodes by type
        type_order = ["goal", "constraint", "decision", "question", "fact", "entity"]
        grouped: dict[str, list[MemoryNode]] = {t: [] for t in type_order}
        for node in self.nodes:
            nt = node.node_type if node.node_type in grouped else "fact"
            grouped[nt].append(node)

        sections: list[str] = []

        type_labels = {
            "goal": "GOALS",
            "constraint": "CONSTRAINTS",
            "decision": "DECISIONS",
            "question": "OPEN",
            "fact": "FACTS",
            "entity": "ENTITIES",
        }

        for node_type in type_order:
            nodes = grouped[node_type]
            if not nodes:
                continue
            lines = [f"[{type_labels[node_type]}]"]
            for node in nodes:
                edges_from = edge_lookup.get(node.node_id, [])
                if edges_from:
                    edge_notes = ", ".join(f"{et}: {tid}" for tid, et in edges_from)
                    lines.append(f"- {node.content} ({edge_notes})")
                else:
                    lines.append(f"- {node.content}")
            sections.append("\n".join(lines))

        return "\n\n".join(sections)


def _token_overlap(a: str, b: str) -> float:
    """Token-level Jaccard similarity between two strings."""
    tokens_a = set(re.findall(r'\b[a-z0-9_]+\b', a.lower()))
    tokens_b = set(re.findall(r'\b[a-z0-9_]+\b', b.lower()))
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def _detect_contradiction(sent_a: str, sent_b: str) -> bool:
    """
    Simple heuristic: two sentences are contradictory if they share a subject
    token but one has a negation word and the other doesn't, around the same topic.
    """
    negation_words = frozenset(["not", "never", "cannot", "no", "don't", "doesn't", "isn't", "aren't", "won't"])
    tokens_a = set(re.findall(r'\b[a-z0-9_]+\b', sent_a.lower()))
    tokens_b = set(re.findall(r'\b[a-z0-9_]+\b', sent_b.lower()))

    shared = tokens_a & tokens_b - negation_words
    if len(shared) < 3:
        return False

    a_has_neg = bool(tokens_a & negation_words)
    b_has_neg = bool(tokens_b & negation_words)
    return a_has_neg != b_has_neg  # One has negation, the other doesn't


# ---------------------------------------------------------------------------
# Legacy ConversationState (kept as fallback)
# ---------------------------------------------------------------------------

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

    Uses a MemoryGraph as the primary path for structured state extraction,
    with ConversationState as a fallback.

    This is rule-driven, not freeform LLM summarization. It:
      1. Builds a MemoryGraph from older blocks.
      2. Adds nodes for each classified sentence.
      3. Detects "updates" relationships (newer sentence with similar content).
      4. Detects "contradicts" (sentences with opposite polarity around same subject).
      5. Resolves contradictions.
      6. Renders the graph to a state block.
      7. Retains only the most recent N turns verbatim (recency tail).
    """

    def __init__(self, recency_tail: int = 3) -> None:
        """
        recency_tail: how many of the most recent history blocks to keep verbatim.
        """
        self.recency_tail = recency_tail
        self._node_counter = 0

    def _next_node_id(self) -> str:
        self._node_counter += 1
        return f"node_{self._node_counter}"

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

        # Primary path: MemoryGraph
        graph = self._build_memory_graph(older)
        graph.resolve_contradictions()

        rendered = graph.render()
        if not rendered.strip():
            # Fallback to legacy ConversationState
            state = self._extract_state(older)
            if state.is_empty():
                return tail, None
            rendered = state.render()

        state_block = Block(
            content=rendered,
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

    def _build_memory_graph(self, blocks: list["Block"]) -> MemoryGraph:
        """Build a MemoryGraph from a list of history blocks."""
        graph = MemoryGraph()

        for turn_idx, block in enumerate(blocks):
            if block.block_type == BlockType.TOOL_RESULT:
                # Tool results become fact nodes
                summary = block.content[:200].replace("\n", " ")
                node = MemoryNode(
                    node_id=self._next_node_id(),
                    node_type="fact",
                    content=f"[Tool result] {summary}",
                    turn=turn_idx,
                )
                graph.add_node(node)
                continue

            sentences = _split_sentences(block.content)
            prev_nodes_in_block: list[MemoryNode] = []

            for sentence in sentences:
                bucket = _classify_sentence(sentence)
                if bucket is None:
                    continue

                new_node = MemoryNode(
                    node_id=self._next_node_id(),
                    node_type=bucket,
                    content=sentence.strip()[:300],
                    turn=turn_idx,
                )

                # Check for contradictions with previous nodes in same bucket
                for prev_node in graph.nodes:
                    if prev_node.node_type != bucket:
                        continue
                    if _detect_contradiction(prev_node.content, new_node.content):
                        graph.add_edge(
                            MemoryEdge(
                                from_id=new_node.node_id,
                                to_id=prev_node.node_id,
                                edge_type="contradicts",
                            )
                        )

                graph.add_node(new_node)
                prev_nodes_in_block.append(new_node)

        return graph

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
