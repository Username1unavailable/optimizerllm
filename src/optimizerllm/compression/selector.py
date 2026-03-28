"""Query-aware selector — scores blocks by relevance to the active query."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from optimizerllm.core.block import Block
    from optimizerllm.task_synth.synth import TaskDescriptor

# Block types that carry retrieved or historical content
_SELECTABLE_TYPES = {
    "retrieved_doc",
    "conversation_history",
    "few_shot_example",
    "structured_data",
}


def _tokenize(text: str) -> list[str]:
    """Simple word tokenizer — lowercase, strip punctuation."""
    return re.findall(r"\b[a-z0-9_]+\b", text.lower())


# ---------------------------------------------------------------------------
# TF-IDF (implemented without sklearn to keep deps minimal)
# ---------------------------------------------------------------------------

def _tfidf_cosine(query_tokens: list[str], doc_token_lists: list[list[str]]) -> list[float]:
    """
    Compute cosine similarity between a query and each document using TF-IDF.
    Returns a list of scores in [0, 1].
    """
    all_token_lists = [query_tokens] + doc_token_lists
    N = len(doc_token_lists)

    # Build vocab
    vocab: set[str] = set(query_tokens)
    for tl in doc_token_lists:
        vocab.update(tl)
    vocab_list = list(vocab)

    # IDF over documents only (not the query)
    idf: dict[str, float] = {}
    for term in vocab_list:
        df = sum(1 for tl in doc_token_lists if term in set(tl))
        idf[term] = math.log((N + 1) / (df + 1)) + 1.0

    def tfidf_vec(tokens: list[str]) -> dict[str, float]:
        counts = Counter(tokens)
        total = max(len(tokens), 1)
        return {t: (counts.get(t, 0) / total) * idf.get(t, 0.0) for t in vocab_list}

    query_vec = tfidf_vec(query_tokens)
    q_norm = math.sqrt(sum(v * v for v in query_vec.values()))
    if q_norm == 0:
        return [0.0] * N

    scores: list[float] = []
    for tl in doc_token_lists:
        doc_vec = tfidf_vec(tl)
        dot = sum(query_vec[t] * doc_vec[t] for t in vocab_list)
        d_norm = math.sqrt(sum(v * v for v in doc_vec.values()))
        if d_norm == 0:
            scores.append(0.0)
        else:
            scores.append(dot / (q_norm * d_norm))
    return scores


# ---------------------------------------------------------------------------
# BM25 (via rank_bm25 when available, fallback otherwise)
# ---------------------------------------------------------------------------

def _bm25_scores(query_tokens: list[str], doc_token_lists: list[list[str]]) -> list[float]:
    try:
        from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

        bm25 = BM25Okapi(doc_token_lists)
        raw = bm25.get_scores(query_tokens)
        # Normalize to [0, 1]
        max_s = max(raw) if len(raw) > 0 else 1.0
        if max_s == 0:
            return [0.0] * len(doc_token_lists)
        return [float(s / max_s) for s in raw]
    except ImportError:
        # rank_bm25 not available — fall back to simple term overlap
        return _term_overlap(query_tokens, doc_token_lists)


def _term_overlap(query_tokens: list[str], doc_token_lists: list[list[str]]) -> list[float]:
    query_set = set(query_tokens)
    scores: list[float] = []
    for tokens in doc_token_lists:
        if not tokens:
            scores.append(0.0)
            continue
        doc_set = set(tokens)
        overlap = len(query_set & doc_set)
        scores.append(overlap / max(len(query_set), 1))
    return scores


# ---------------------------------------------------------------------------
# Recency / priority bonus
# ---------------------------------------------------------------------------

def _recency_score(position: int | None, total_blocks: int) -> float:
    """Blocks that appear later in the conversation get a recency bonus."""
    if position is None or total_blocks <= 1:
        return 0.5
    return position / (total_blocks - 1)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

class QueryAwareSelector:
    """
    Scores a list of blocks against the active query using:
      - BM25 (keyword overlap weighted by IDF)
      - TF-IDF cosine similarity
      - Recency weighting
      - Source priority weighting

    The combined score drives the budget controller's keep/drop decisions.
    """

    def __init__(
        self,
        bm25_weight: float = 0.50,
        tfidf_weight: float = 0.30,
        recency_weight: float = 0.10,
        priority_weight: float = 0.10,
    ) -> None:
        total = bm25_weight + tfidf_weight + recency_weight + priority_weight
        self.bm25_w = bm25_weight / total
        self.tfidf_w = tfidf_weight / total
        self.recency_w = recency_weight / total
        self.priority_w = priority_weight / total

    def score_blocks(
        self,
        blocks: list["Block"],
        query: str,
        *,
        score_all_types: bool = False,
        task_descriptor: "TaskDescriptor | None" = None,
    ) -> list["Block"]:
        """
        Attach a relevance score (0–1) to each block.

        By default only selectable block types are scored;
        others receive score=1.0 to ensure they are preserved.
        If score_all_types=True, everything is scored.

        If task_descriptor is provided, the RetentionPolicy weights are applied
        as multipliers on the final score before clamping, biasing the selector
        towards block types that are important for the inferred task.
        """
        if not query.strip():
            # No query — assign neutral scores
            return [b.clone_with(score=1.0) for b in blocks]

        query_tokens = _tokenize(query)
        total_blocks = len(blocks)

        # Separate blocks to score from those to preserve
        to_score_idx: list[int] = []
        to_preserve_idx: list[int] = []

        for i, b in enumerate(blocks):
            if score_all_types or b.block_type.value in _SELECTABLE_TYPES:
                to_score_idx.append(i)
            else:
                to_preserve_idx.append(i)

        result = list(blocks)

        # Preserve blocks get score=1.0
        for i in to_preserve_idx:
            result[i] = blocks[i].clone_with(score=1.0)

        if not to_score_idx:
            return result

        doc_token_lists = [_tokenize(blocks[i].content) for i in to_score_idx]

        bm25 = _bm25_scores(query_tokens, doc_token_lists)
        tfidf = _tfidf_cosine(query_tokens, doc_token_lists)

        # Build a per-block-type weight multiplier from the retention policy
        policy = task_descriptor.retention_policy if task_descriptor else None

        def _policy_weight(block: "Block") -> float:
            if policy is None:
                return 1.0
            bt = block.block_type.value
            if bt == "retrieved_doc":
                return policy.doc_weight
            if bt == "conversation_history":
                return policy.history_weight
            if bt == "system_prompt":
                return policy.system_weight
            if bt == "few_shot_example":
                return policy.example_weight
            if bt == "tool_schema":
                return policy.tool_weight
            return 1.0

        for rank, i in enumerate(to_score_idx):
            b = blocks[i]
            recency = _recency_score(b.metadata.position, total_blocks)
            priority_norm = b.metadata.priority / 100.0

            combined = (
                self.bm25_w * bm25[rank]
                + self.tfidf_w * tfidf[rank]
                + self.recency_w * recency
                + self.priority_w * priority_norm
            )
            # Apply task-aware retention weight
            combined = combined * _policy_weight(b)
            # Clamp to [0, 1]
            result[i] = b.clone_with(score=min(1.0, max(0.0, combined)))

        return result

    def filter_by_threshold(
        self, blocks: list["Block"], threshold: float = 0.15
    ) -> tuple[list["Block"], list["Block"]]:
        """
        Split blocks into (kept, dropped) based on minimum score.

        Protected blocks are always kept regardless of score.
        """
        kept: list["Block"] = []
        dropped: list["Block"] = []
        for b in blocks:
            if b.metadata.protected or (b.score is not None and b.score >= threshold):
                kept.append(b)
            else:
                score_str = f"{b.score:.3f}" if b.score is not None else "unscored"
                dropped.append(
                    b.clone_with(drop_reason=f"score_below_threshold:{score_str}")
                )
        return kept, dropped
