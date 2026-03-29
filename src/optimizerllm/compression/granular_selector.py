"""
Granular selector — sub-block content trimming.

Phase 1 keeps or drops whole blocks.
Phase 2 adds finer control: score sections and sentences *within* a block,
then rebuild a trimmed version that preserves only the relevant parts.

This runs after block-level scoring and before the budget controller.
"""

from __future__ import annotations

import re
from dataclasses import replace as dc_replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from optimizerllm.core.block import Block
    from optimizerllm.slot_guard.guard import SlotIndex

from optimizerllm.compression.selector import _bm25_scores, _tfidf_cosine, _tokenize
from optimizerllm.compression.span_critic import SpanCritic
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

# Block types eligible for span-level trimming
_SPAN_ELIGIBLE = {
    BlockType.RETRIEVED_DOC,
    BlockType.STRUCTURED_DATA,
}

# Minimum character length a block must have before section splitting is useful
_MIN_CHARS_FOR_SECTION_SPLIT = 300

# Minimum character length before sentence splitting is useful.
# 350 chars ≈ 4–6 typical sentences.  Below this, any single dropped
# sentence is a large fraction of the block — not worth the risk.
_MIN_CHARS_FOR_SENTENCE_SPLIT = 350

# Minimum sentence length before span trimming is useful
_MIN_CHARS_FOR_SPAN_TRIM = 150

_ELLIPSIS = "[…]"
_SPAN_ELLIPSIS = " … "


def _score_texts(query_tokens: list[str], texts: list[str]) -> list[float]:
    """Combined BM25 + TF-IDF score, normalised to [0, 1]."""
    if not texts or not query_tokens:
        return [0.0] * len(texts)

    token_lists = [_tokenize(t) for t in texts]
    bm25 = _bm25_scores(query_tokens, token_lists)
    tfidf = _tfidf_cosine(query_tokens, token_lists)
    return [0.6 * b + 0.4 * t for b, t in zip(bm25, tfidf)]


def _has_number(text: str) -> bool:
    """Return True if text contains a digit."""
    return bool(re.search(r'\d', text))


def _extract_spans(text: str, query_tokens: set[str]) -> str:
    """
    Trim a long sentence to its most relevant clauses.

    If sentence is longer than 150 chars:
      - Split into comma-separated or semicolon-separated clauses
      - Score each clause by overlap with query_tokens
      - Keep clauses scoring > 0 or those with numbers
      - Rejoin with " … " if clauses were dropped

    If sentence is shorter than 150 chars: return as-is.
    """
    if len(text) <= _MIN_CHARS_FOR_SPAN_TRIM:
        return text

    clauses = re.split(r'[,;]', text)
    if len(clauses) <= 1:
        return text

    kept_clauses: list[str] = []
    dropped_any = False

    for clause in clauses:
        clause_stripped = clause.strip()
        if not clause_stripped:
            continue
        clause_tokens = set(_tokenize(clause_stripped))
        overlap = len(clause_tokens & query_tokens)
        if overlap > 0 or _has_number(clause_stripped):
            kept_clauses.append(clause_stripped)
        else:
            dropped_any = True

    if not kept_clauses:
        # Nothing passed — return original to avoid empty sentence
        return text

    if not dropped_any:
        return text

    return _SPAN_ELLIPSIS.join(kept_clauses)


class GranularSelector:
    """
    Trims individual blocks at the section, sentence, and optionally span level.

    Pipeline per eligible block:
      1. If block is long enough → split into sections → score each section.
         Drop sections below `section_threshold`.
      2. If any remaining section is still long enough → split into sentences
         → score each sentence. When a SpanCritic is provided, MMR-style
         redundancy-aware selection is used; otherwise falls back to simple
         threshold filtering.
      3. [Optional] If span_level=True and block is RETRIEVED_DOC or STRUCTURED_DATA:
         apply span-level trimming to each surviving sentence.
      4. Reassemble the kept content and return a trimmed block.

    Slot-awareness: if a SlotIndex is provided, any sentence that contains a
    protected slot is always kept regardless of its relevance score.

    Blocks that become empty after trimming are left unchanged (better to
    keep something than return nothing).
    """

    # Class-level capture buffer for training data collection.
    # None means capture is off.  Enable via start_capture() / stop_capture().
    _capture: "list[dict] | None" = None
    _block_seq: int = 0  # monotone counter — unique block_id across a capture session

    @classmethod
    def start_capture(cls) -> None:
        """Enable sentence-level decision logging (for SpanCritic training)."""
        cls._capture = []
        cls._block_seq = 0

    @classmethod
    def stop_capture(cls) -> "list[dict]":
        """Disable logging and return the collected decision records."""
        records = cls._capture or []
        cls._capture = None
        return records

    def __init__(
        self,
        section_threshold: float = 0.10,
        sentence_threshold: float = 0.08,
        min_sentences_to_keep: int = 1,
        context_buffer: int = 1,
        span_level: bool = False,
    ) -> None:
        self.section_threshold = section_threshold
        self.sentence_threshold = sentence_threshold
        self.min_sentences = min_sentences_to_keep
        self.context_buffer = context_buffer
        self.span_level = span_level
        self._splitter = ContentSplitter()
        self._span_critic: SpanCritic | None = None

    def set_span_critic(self, critic: SpanCritic) -> None:
        """Attach a SpanCritic for MMR-style sentence selection."""
        self._span_critic = critic

    def trim_blocks(
        self,
        blocks: list["Block"],
        query: str,
        *,
        section_level: bool = True,
        sentence_level: bool = True,
        span_level: bool | None = None,
        slot_index: "SlotIndex | None" = None,
        sentence_keep_fraction: float | None = None,
    ) -> list["Block"]:
        """
        Apply granular trimming to all eligible blocks.

        Args:
            span_level: If provided, overrides self.span_level for this call.
                        None → use self.span_level (the instance default).
            sentence_keep_fraction: When set (0.0–1.0), switch sentence selection
                from threshold-based to budget-based: keep the top
                ceil(n * sentence_keep_fraction) sentences ranked by SpanCritic
                net_keep_score. Force-keep sentences (fragility_prob >= 0.45) are
                always included outside the budget. None = threshold mode (default).

        Returns new block objects (originals untouched).
        """
        if not query.strip():
            return blocks

        effective_span_level = self.span_level if span_level is None else span_level
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
                if self._span_critic is not None:
                    trimmed_content = self._trim_sentences_with_critic(
                        trimmed_content, query_tokens, slot_index=slot_index,
                        keep_fraction=sentence_keep_fraction,
                    )
                else:
                    trimmed_content = self._trim_sentences(
                        trimmed_content, query_tokens, slot_index=slot_index
                    )

            # Span-level trimming (only when enabled and block type is eligible)
            if (
                effective_span_level
                and block.block_type in _SPAN_ELIGIBLE
                and len(trimmed_content) >= _MIN_CHARS_FOR_SENTENCE_SPLIT
            ):
                trimmed_content = self._trim_spans(
                    trimmed_content, query_tokens, slot_index
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

        # Emit to capture buffer (heuristic path — no SpanCritic scores available)
        if GranularSelector._capture is not None:
            block_id = GranularSelector._block_seq
            GranularSelector._block_seq += 1
            for i in range(n):
                GranularSelector._capture.append({
                    "text": texts[i],
                    "block_id": block_id,
                    "by_selection": i in keep_set,
                    "kept": i in buffered,
                    # No SpanCritic scores — fill with heuristic proxies
                    "keep_prob": float(scores[i]),
                    "fragility_prob": 0.0,
                    "redundancy_prob": 0.0,
                    "position": i,
                    "total": n,
                    "loss_type_probs": {},
                    "query_tokens": query_tokens,
                })

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

    # ------------------------------------------------------------------
    # Sentence-level trimming — SpanCritic path (MMR-style)
    # ------------------------------------------------------------------

    def _trim_sentences_with_critic(
        self,
        text: str,
        query_tokens: list[str],
        slot_index: "SlotIndex | None" = None,
        keep_fraction: float | None = None,
    ) -> str:
        """
        MMR-style sentence selection using SpanCritic.

        Two modes depending on keep_fraction:

        Threshold mode (keep_fraction=None, default):
          1. Score all sentences via SpanCritic (initial scores, redundancy=0)
          2. Seed keep_set with the highest keep_prob sentence
          3. Keep if net_keep_score >= threshold OR force_keep OR slot sentence
          4. Enforce minimum, add context buffer, reconstruct

        Budget mode (keep_fraction set):
          1. Score all sentences
          2. Force-keep: fragility_prob >= 0.45 OR slot sentences (always kept)
          3. Fill remaining budget = ceil(n * keep_fraction) - n_forced
             from highest net_keep_score downward
          4. Enforce minimum, reconstruct (no context buffer — budget is the budget)
        """
        assert self._span_critic is not None

        sentences = self._splitter.split_sentences(text)
        if len(sentences) <= 2:
            return text

        n = len(sentences)
        texts = [s.text for s in sentences]

        # Step 1: score all sentences
        scores = self._span_critic.score_batch(
            texts, query_tokens, slot_index=slot_index
        )

        # Only use budget mode for genuinely long blocks. Short documents
        # (< 3000 chars) have few sentences so dropping even one can remove
        # critical factual content that isn't lexically close to the query.
        _MIN_CHARS_FOR_BUDGET_MODE = 3000
        effective_keep_fraction = (
            keep_fraction if (keep_fraction is not None and len(text) >= _MIN_CHARS_FOR_BUDGET_MODE)
            else None
        )

        if effective_keep_fraction is not None:
            keep_fraction = effective_keep_fraction  # use local alias below
            # ── Budget mode ────────────────────────────────────────────────
            import math as _math

            # Compute full redundancy for all sentences against the best-scoring one
            max_idx = max(range(n), key=lambda i: scores[i].keep_prob)
            selected_texts: list[str] = [texts[max_idx]]
            updated_scores: list = list(scores)
            for i in range(n):
                if i == max_idx:
                    continue
                updated_scores[i] = self._span_critic.update_redundancy(
                    scores[i], texts[i], selected_texts
                )

            # Forced-keep set: fragility >= 0.45 or slot sentence
            forced: set[int] = set()
            for i in range(n):
                is_slot = slot_index is not None and slot_index.sentence_has_slot(texts[i])
                if updated_scores[i].should_force_keep or is_slot:
                    forced.add(i)

            # Budget: at most ceil(n * keep_fraction) total, but at least min_sentences
            budget = max(
                self.min_sentences,
                _math.ceil(n * keep_fraction),
            )

            # Fill budget with highest net_keep_score non-forced sentences
            non_forced_ranked = sorted(
                [i for i in range(n) if i not in forced],
                key=lambda i: updated_scores[i].net_keep_score,
                reverse=True,
            )
            remaining = max(0, budget - len(forced))
            selected_non_forced = set(non_forced_ranked[:remaining])

            keep_set = forced | selected_non_forced
            buffered = keep_set  # no context buffer in budget mode

        else:
            # ── Threshold mode (original) ──────────────────────────────────
            # Step 2: seed with highest keep_prob sentence
            max_idx = max(range(n), key=lambda i: scores[i].keep_prob)
            keep_set = {max_idx}
            selected_texts = [texts[max_idx]]

            # Step 3: greedy MMR selection over remaining sentences
            updated_scores = list(scores)  # will hold updated redundancy
            for i in range(n):
                if i in keep_set:
                    continue
                updated = self._span_critic.update_redundancy(
                    scores[i], texts[i], selected_texts
                )
                updated_scores[i] = updated
                is_slot_sentence = (
                    slot_index is not None and slot_index.sentence_has_slot(texts[i])
                )
                should_keep = (
                    updated.net_keep_score >= self.sentence_threshold
                    or updated.should_force_keep
                    or is_slot_sentence
                )
                if should_keep:
                    keep_set.add(i)
                    selected_texts.append(texts[i])

            # Step 4: enforce minimum
            if len(keep_set) < self.min_sentences:
                ranked = sorted(range(n), key=lambda i: scores[i].keep_prob, reverse=True)
                for idx in ranked:
                    keep_set.add(idx)
                    if len(keep_set) >= self.min_sentences:
                        break

            # Add context buffer (threshold mode only)
            buffered: set[int] = set()
            for idx in keep_set:
                for offset in range(-self.context_buffer, self.context_buffer + 1):
                    neighbour = idx + offset
                    if 0 <= neighbour < n:
                        buffered.add(neighbour)

        # Emit to capture buffer (training data collection).
        # Done AFTER buffer expansion so `kept` reflects the true final output.
        if GranularSelector._capture is not None:
            block_id = GranularSelector._block_seq
            GranularSelector._block_seq += 1
            for i in range(n):
                sc = updated_scores[i]
                GranularSelector._capture.append({
                    "text": texts[i],
                    "block_id": block_id,
                    "by_selection": i in keep_set,   # chosen by MMR logic (clean label signal)
                    "kept": i in buffered,            # final output (includes context buffer)
                    "keep_prob": sc.keep_prob,
                    "fragility_prob": sc.fragility_prob,
                    "redundancy_prob": sc.redundancy_prob,
                    "position": i,
                    "total": n,
                    "loss_type_probs": sc.loss_type_probs,
                    "query_tokens": query_tokens,
                })

        if len(buffered) == n:
            return text

        kept_sentences: list[str] = []
        prev_idx: int | None = None
        for idx in sorted(buffered):
            if prev_idx is not None and idx > prev_idx + 1:
                kept_sentences.append(_ELLIPSIS)
            kept_sentences.append(texts[idx])
            prev_idx = idx

        if not kept_sentences:
            return text

        return " ".join(kept_sentences)

    # ------------------------------------------------------------------
    # Span-level trimming
    # ------------------------------------------------------------------

    def _trim_spans(
        self,
        text: str,
        query_tokens: list[str],
        slot_index: "SlotIndex | None",
    ) -> str:
        """
        Apply span-level trimming to each sentence in text.

        For each sentence longer than 150 chars that is NOT a slot sentence,
        trim it to the most relevant comma/semicolon-separated clauses.
        """
        sentences = self._splitter.split_sentences(text)
        if not sentences:
            return text

        query_token_set = set(query_tokens)
        trimmed_parts: list[str] = []

        for sent in sentences:
            sent_text = sent.text
            # Do not trim slot-containing sentences
            is_slot_sentence = (
                slot_index is not None and slot_index.sentence_has_slot(sent_text)
            )
            if is_slot_sentence or len(sent_text) <= _MIN_CHARS_FOR_SPAN_TRIM:
                trimmed_parts.append(sent_text)
            else:
                trimmed_parts.append(_extract_spans(sent_text, query_token_set))

        return " ".join(trimmed_parts)
