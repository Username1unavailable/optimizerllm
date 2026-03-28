"""
Granular selector — sub-block content trimming.

Phase 1 keeps or drops whole blocks.
Phase 2 adds finer control: score sections and sentences *within* a block,
then rebuild a trimmed version that preserves only the relevant parts.

This runs after block-level scoring and before the budget controller.
"""

from __future__ import annotations

from dataclasses import replace as dc_replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from optimizerllm.core.block import Block
    from optimizerllm.slot_guard.guard import SlotIndex

from optimizerllm.compression.selector import _bm25_scores, _tfidf_cosine, _tokenize
from optimizerllm.compression.splitter import (
    ContentSplitter,
    Section,
    Sentence,
)
from optimizerllm.core.block import BlockType

# Block types eligible for granular trimming
_GRANULAR_ELIGIBLE = {
    BlockType.RETRIEVED_DOC,
    BlockType.CONVERSATION_HISTORY,
    BlockType.STRUCTURED_DATA,
    BlockType.ASSISTANT_MESSAGE,
}

# Minimum character length a block must have before section splitting is useful
_MIN_CHARS_FOR_SECTION_SPLIT = 300

# Minimum character length before sentence splitting is useful
_MIN_CHARS_FOR_SENTENCE_SPLIT = 150

_ELLIPSIS = "[…]"


def _score_texts(query_tokens: list[str], texts: list[str]) -> list[float]:
    """Combined BM25 + TF-IDF score, normalised to [0, 1]."""
    if not texts or not query_tokens:
        return [0.0] * len(texts)

    token_lists = [_tokenize(t) for t in texts]
    bm25 = _bm25_scores(query_tokens, token_lists)
    tfidf = _tfidf_cosine(query_tokens, token_lists)
    return [0.6 * b + 0.4 * t for b, t in zip(bm25, tfidf)]


class GranularSelector:
    """
    Trims individual blocks at the section and sentence level.

    Pipeline per eligible block:
      1. If block is long enough → split into sections → score each section.
         Drop sections below `section_threshold`.
      2. If any remaining section is still long enough → split into sentences
         → score each sentence. Keep top-k sentences (with 1-sentence context
         buffer on each side to avoid orphaned fragments).
      3. Reassemble the kept content and return a trimmed block.

    Slot-awareness: if a SlotIndex is provided, any sentence that contains a
    protected slot is always kept regardless of its relevance score.

    Blocks that become empty after trimming are left unchanged (better to
    keep something than return nothing).
    """

    def __init__(
        self,
        section_threshold: float = 0.10,
        sentence_threshold: float = 0.08,
        min_sentences_to_keep: int = 1,
        context_buffer: int = 1,
    ) -> None:
        self.section_threshold = section_threshold
        self.sentence_threshold = sentence_threshold
        self.min_sentences = min_sentences_to_keep
        self.context_buffer = context_buffer
        self._splitter = ContentSplitter()

    def trim_blocks(
        self,
        blocks: list["Block"],
        query: str,
        *,
        section_level: bool = True,
        sentence_level: bool = True,
        slot_index: "SlotIndex | None" = None,
    ) -> list["Block"]:
        """
        Apply granular trimming to all eligible blocks.

        Returns new block objects (originals untouched).
        """
        if not query.strip():
            return blocks

        query_tokens = _tokenize(query)
        result: list["Block"] = []

        for block in blocks:
            if block.metadata.protected or block.block_type not in _GRANULAR_ELIGIBLE:
                result.append(block)
                continue

            # Slot-aware trimming: sentence-level slot protection handles individual
            # slot sentences. We only skip the entire block for non-document types
            # (user messages, system prompts) that are outside _GRANULAR_ELIGIBLE
            # anyway, so no additional block-level skip is needed here.
            trimmed_content = block.content

            if section_level and len(trimmed_content) >= _MIN_CHARS_FOR_SECTION_SPLIT:
                trimmed_content = self._trim_sections(
                    trimmed_content, query_tokens, slot_index=slot_index
                )

            if sentence_level and len(trimmed_content) >= _MIN_CHARS_FOR_SENTENCE_SPLIT:
                trimmed_content = self._trim_sentences(
                    trimmed_content, query_tokens, slot_index=slot_index
                )

            if trimmed_content.strip() and trimmed_content != block.content:
                result.append(block.clone_with(content=trimmed_content))
            else:
                result.append(block)

        return result

    # ------------------------------------------------------------------
    # Section-level trimming
    # ------------------------------------------------------------------

    def _trim_sections(
        self,
        text: str,
        query_tokens: list[str],
        slot_index: "SlotIndex | None" = None,
    ) -> str:
        sections = self._splitter.split_sections(text)

        if len(sections) <= 1:
            # Nothing to split — skip
            return text

        bodies = [s.body for s in sections]
        scores = _score_texts(query_tokens, bodies)

        kept: list[Section] = []
        for section, score in zip(sections, scores):
            section.score = score
            # Always keep sections that contain a protected slot
            has_slot = slot_index is not None and slot_index.sentence_has_slot(section.body)
            if score >= self.section_threshold or has_slot:
                kept.append(section)

        if not kept:
            # Nothing passed — return original to avoid empty block
            return text

        if len(kept) == len(sections):
            # Nothing was pruned — skip reassembly overhead
            return text

        return self._splitter.reassemble_sections(kept)

    # ------------------------------------------------------------------
    # Sentence-level trimming
    # ------------------------------------------------------------------

    def _trim_sentences(
        self,
        text: str,
        query_tokens: list[str],
        slot_index: "SlotIndex | None" = None,
    ) -> str:
        sentences = self._splitter.split_sentences(text)

        if len(sentences) <= 2:
            return text

        texts = [s.text for s in sentences]
        scores = _score_texts(query_tokens, texts)

        for sentence, score in zip(sentences, scores):
            sentence.score = score

        # Determine which indices to keep
        n = len(sentences)
        keep_set: set[int] = set()

        # Always keep the highest-scoring sentence
        max_idx = max(range(n), key=lambda i: scores[i])
        keep_set.add(max_idx)

        # Add all sentences that cross the threshold
        for i, score in enumerate(scores):
            if score >= self.sentence_threshold:
                keep_set.add(i)

        # Always keep sentences that contain a protected slot
        if slot_index is not None:
            for i, sent in enumerate(sentences):
                if slot_index.sentence_has_slot(sent.text):
                    keep_set.add(i)

        # Enforce minimum
        if len(keep_set) < self.min_sentences:
            ranked = sorted(range(n), key=lambda i: scores[i], reverse=True)
            for idx in ranked:
                keep_set.add(idx)
                if len(keep_set) >= self.min_sentences:
                    break

        # Add context buffer (one sentence before/after each kept sentence)
        buffered: set[int] = set()
        for idx in keep_set:
            for offset in range(-self.context_buffer, self.context_buffer + 1):
                neighbour = idx + offset
                if 0 <= neighbour < n:
                    buffered.add(neighbour)

        if len(buffered) == n:
            # Nothing was pruned
            return text

        # Reconstruct in order, inserting ellipsis for gaps
        kept_sentences: list[str] = []
        prev_idx: int | None = None

        for idx in sorted(buffered):
            if prev_idx is not None and idx > prev_idx + 1:
                kept_sentences.append(_ELLIPSIS)
            kept_sentences.append(sentences[idx].text)
            prev_idx = idx

        if not kept_sentences:
            return text

        return " ".join(kept_sentences)
