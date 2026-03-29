"""
Granular selector — sub-block content trimming.

Three-stage pipeline per eligible block:
  1. Section level: score and drop whole low-relevance sections (fast, safe).
  2. Sentence level: score and drop whole low-relevance sentences via SpanCritic
     budget mode (restored from Phase 2C — gave 0.792 on meetingbank, 0.836 on
     longbench).
  3. Token level: compress tokens WITHIN every surviving sentence (new).
     Multiplied on top of sentence-level: 0.792 × 0.55 ≈ 0.436 on meetingbank.

Hard guards that can never be overridden by any fraction:
  - Numbers (4.2, 12%, $500M): changing them changes facts.
  - Negation words (not, no, nor, never, none): dropping them inverts meaning.
  - Proper nouns (capitalized mid-sentence): dropping them loses entity references.
  - Query tokens: always kept regardless of IDF.
  - Slot-containing sentences: keep all tokens.
  - Fragile sentences (SpanCritic fragility >= 0.45): keep all tokens.

Everything else — articles, prepositions, auxiliary verbs, non-query adjectives —
is fair game regardless of how aggressive the token_keep_fraction is.
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from optimizerllm.core.block import Block
    from optimizerllm.slot_guard.guard import SlotIndex

from optimizerllm.compression.selector import _bm25_scores, _tfidf_cosine, _tokenize
from optimizerllm.compression.span_critic import SpanCritic
from optimizerllm.compression.splitter import ContentSplitter, Section, Sentence
from optimizerllm.core.block import BlockType

# ---------------------------------------------------------------------------
# Block eligibility
# ---------------------------------------------------------------------------

_GRANULAR_ELIGIBLE = {
    BlockType.RETRIEVED_DOC,
    BlockType.CONVERSATION_HISTORY,
    BlockType.STRUCTURED_DATA,
    BlockType.ASSISTANT_MESSAGE,
}

# ---------------------------------------------------------------------------
# Size thresholds
# ---------------------------------------------------------------------------

_MIN_CHARS_FOR_SECTION_SPLIT  = 300   # minimum block size for section splitting
_MIN_CHARS_FOR_SENTENCE_SPLIT = 350   # minimum block size for sentence-level trimming
_MIN_CHARS_FOR_TOKEN_TRIM     = 350   # minimum block size for token-level trimming
_MIN_SENT_CHARS_FOR_TOKEN_TRIM = 25   # minimum sentence size to bother compressing
_MIN_CHARS_FOR_BUDGET_MODE    = 3000  # minimum block size for sentence budget mode

_ELLIPSIS = "[…]"

# ---------------------------------------------------------------------------
# Token-level compression constants
# ---------------------------------------------------------------------------

# Words safe to drop when not in the query.
# Intentionally excludes negation/causal/concessive words — those carry
# truth-value information.
_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "on", "at", "by", "for", "with", "about",
    "against", "between", "into", "through", "during", "before", "after",
    "above", "below", "from", "up", "down", "out", "off", "over", "under",
    "again", "further", "then", "once", "that", "this", "these", "those",
    "it", "its", "itself", "they", "them", "their", "what", "which", "who",
    "whom", "when", "where", "why", "how", "all", "both", "each", "few",
    "more", "most", "other", "some", "such", "only", "own", "same", "so",
    "than", "too", "very", "just", "also", "as", "but", "and", "or", "if",
    "while", "although", "though", "unless", "until", "whether",
    "i", "he", "she", "we", "you", "my", "your", "his", "her", "our", "any",
    "there",
})

# Negation words: hard-kept like numbers — dropping them inverts truth value.
_NEGATION_WORDS = frozenset({
    "not", "no", "nor", "never", "none", "nothing", "nobody", "nowhere",
    "without", "neither", "hardly", "barely", "scarcely",
})

# Tokenizer that preserves numeric units as single tokens
_TOKEN_RE = re.compile(
    r'[$£€]?\d[\d,]*\.?\d*[%BMKbmk]?'  # $4.2B, 12%, 1,000
    r"|\b\w[\w']*\b"                     # words
    r'|[^\w\s]'                          # punctuation
)
_NUMBER_RE = re.compile(r'\d')
_PUNCT_ONLY_RE = re.compile(r'^[^\w]+$')


def _build_sentence_idf(sentences: list[str]) -> dict[str, float]:
    """IDF over a sentence list — used as local corpus for token scoring."""
    N = max(len(sentences), 1)
    df: dict[str, int] = {}
    for s in sentences:
        for t in set(re.findall(r"\b\w+\b", s.lower())):
            df[t] = df.get(t, 0) + 1
    return {t: math.log((N + 1) / (d + 1)) + 1.0 for t, d in df.items()}


def _compress_sentence_tokens(
    sentence: str,
    query_token_set: frozenset,
    idf: dict[str, float],
    keep_fraction: float,
) -> str:
    """
    Keep the most important `keep_fraction` of tokens in a sentence.

    Hard keeps (score 85–100, never dropped regardless of fraction):
      numbers, negation words, query tokens, proper nouns (capitalized mid-sentence)

    Soft scores: content words by IDF, stop words near-zero.

    Returns original sentence if result would be empty.
    """
    raw_tokens = _TOKEN_RE.findall(sentence)
    if len(raw_tokens) <= 3:
        return sentence

    n = len(raw_tokens)
    target_keep = max(3, math.ceil(n * keep_fraction))
    scores: list[float] = []

    for pos, tok in enumerate(raw_tokens):
        lower = tok.lower()

        if _NUMBER_RE.search(tok):
            scores.append(100.0)
        elif lower in _NEGATION_WORDS:
            scores.append(95.0)
        elif lower in query_token_set:
            scores.append(90.0)
        elif tok[0].isupper() and pos > 0:
            scores.append(85.0)
        elif _PUNCT_ONLY_RE.match(tok):
            scores.append(-1.0)
        elif lower in _STOP_WORDS:
            scores.append(0.5)
        else:
            scores.append(idf.get(lower, 1.0))

    keep_set = set(
        sorted(range(n), key=lambda i: scores[i], reverse=True)[:target_keep]
    )
    kept = [raw_tokens[i] for i in range(n) if i in keep_set]
    return " ".join(kept) if kept else sentence


# ---------------------------------------------------------------------------
# Section / sentence scoring helper
# ---------------------------------------------------------------------------

def _score_texts(query_tokens: list[str], texts: list[str]) -> list[float]:
    if not texts or not query_tokens:
        return [0.0] * len(texts)
    token_lists = [_tokenize(t) for t in texts]
    bm25  = _bm25_scores(query_tokens, token_lists)
    tfidf = _tfidf_cosine(query_tokens, token_lists)
    return [0.6 * b + 0.4 * t for b, t in zip(bm25, tfidf)]


# ---------------------------------------------------------------------------
# GranularSelector
# ---------------------------------------------------------------------------

class GranularSelector:
    """
    Three-stage granular trimming: section → sentence → token.

    Each stage multiplies the compression of the previous one.
    SpanCritic guides both sentence selection and per-sentence token fraction.
    """

    _capture: "list[dict] | None" = None
    _block_seq: int = 0

    @classmethod
    def start_capture(cls) -> None:
        cls._capture = []
        cls._block_seq = 0

    @classmethod
    def stop_capture(cls) -> "list[dict]":
        records = cls._capture or []
        cls._capture = None
        return records

    def __init__(
        self,
        section_threshold: float = 0.10,
        sentence_threshold: float = 0.08,
        min_sentences_to_keep: int = 1,
        context_buffer: int = 1,
    ) -> None:
        self.section_threshold  = section_threshold
        self.sentence_threshold = sentence_threshold
        self.min_sentences      = min_sentences_to_keep
        self.context_buffer     = context_buffer
        self._splitter          = ContentSplitter()
        self._span_critic: SpanCritic | None = None

    def set_span_critic(self, critic: SpanCritic) -> None:
        self._span_critic = critic

    def trim_blocks(
        self,
        blocks: list["Block"],
        query: str,
        *,
        section_level: bool = True,
        sentence_level: bool = True,
        span_level: bool | None = None,          # ignored, kept for API compat
        slot_index: "SlotIndex | None" = None,
        sentence_keep_fraction: float | None = None,
        token_keep_fraction: float | None = None,
    ) -> list["Block"]:
        """
        Apply three-stage trimming to all eligible blocks.

        Args:
            section_level: Enable section dropping.
            sentence_level: Enable sentence dropping + token compression.
            sentence_keep_fraction: Sentence budget fraction (0–1). When set,
                SpanCritic budget mode keeps top ceil(n × frac) sentences.
                None → threshold mode (keep sentences above sentence_threshold).
            token_keep_fraction: Within-sentence token budget (0–1). Applied to
                every sentence that survives sentence-level trimming.
                None → no token-level compression.
        """
        if not query.strip():
            return blocks

        query_tokens = _tokenize(query)
        result: list["Block"] = []

        for block in blocks:
            if block.metadata.protected or block.block_type not in _GRANULAR_ELIGIBLE:
                result.append(block)
                continue

            content = block.content

            # Stage 1: section-level
            if section_level and len(content) >= _MIN_CHARS_FOR_SECTION_SPLIT:
                content = self._trim_sections(content, query_tokens, slot_index)

            # Stage 2: sentence-level
            if sentence_level and len(content) >= _MIN_CHARS_FOR_SENTENCE_SPLIT:
                if self._span_critic is not None:
                    content = self._trim_sentences_with_critic(
                        content, query_tokens, slot_index,
                        keep_fraction=sentence_keep_fraction,
                    )
                else:
                    content = self._trim_sentences(content, query_tokens, slot_index)

            # Stage 3: token-level (runs on sentence-level output)
            if (
                sentence_level
                and token_keep_fraction is not None
                and len(content) >= _MIN_CHARS_FOR_TOKEN_TRIM
            ):
                content = self._trim_tokens(
                    content, query_tokens, slot_index, token_keep_fraction
                )

            if content.strip() and content != block.content:
                result.append(block.clone_with(content=content))
            else:
                result.append(block)

        return result

    # ------------------------------------------------------------------
    # Stage 1: Section-level
    # ------------------------------------------------------------------

    def _trim_sections(
        self,
        text: str,
        query_tokens: list[str],
        slot_index: "SlotIndex | None",
    ) -> str:
        sections = self._splitter.split_sections(text)
        if len(sections) <= 1:
            return text

        bodies  = [s.body for s in sections]
        scores  = _score_texts(query_tokens, bodies)
        kept: list[Section] = []

        for section, score in zip(sections, scores):
            section.score = score
            has_slot = slot_index is not None and slot_index.sentence_has_slot(section.body)
            if score >= self.section_threshold or has_slot:
                kept.append(section)

        if not kept or len(kept) == len(sections):
            return text
        return self._splitter.reassemble_sections(kept)

    # ------------------------------------------------------------------
    # Stage 2a: Sentence-level — SpanCritic path
    # ------------------------------------------------------------------

    def _trim_sentences_with_critic(
        self,
        text: str,
        query_tokens: list[str],
        slot_index: "SlotIndex | None",
        keep_fraction: float | None,
    ) -> str:
        assert self._span_critic is not None
        sentences = self._splitter.split_sentences(text)
        if len(sentences) <= 2:
            return text

        n     = len(sentences)
        texts = [s.text for s in sentences]
        scores = self._span_critic.score_batch(texts, query_tokens, slot_index=slot_index)

        effective_frac = (
            keep_fraction
            if (keep_fraction is not None and len(text) >= _MIN_CHARS_FOR_BUDGET_MODE)
            else None
        )

        if effective_frac is not None:
            # Budget mode
            max_idx = max(range(n), key=lambda i: scores[i].keep_prob)
            selected = [texts[max_idx]]
            updated  = list(scores)
            for i in range(n):
                if i == max_idx:
                    continue
                updated[i] = self._span_critic.update_redundancy(scores[i], texts[i], selected)

            forced = {
                i for i in range(n)
                if updated[i].should_force_keep
                or (slot_index is not None and slot_index.sentence_has_slot(texts[i]))
            }
            budget      = max(self.min_sentences, math.ceil(n * effective_frac))
            non_forced  = sorted(
                [i for i in range(n) if i not in forced],
                key=lambda i: updated[i].net_keep_score,
                reverse=True,
            )
            keep_set = forced | set(non_forced[: max(0, budget - len(forced))])
            buffered = keep_set

        else:
            # Threshold mode
            max_idx    = max(range(n), key=lambda i: scores[i].keep_prob)
            keep_set   = {max_idx}
            selected   = [texts[max_idx]]
            updated    = list(scores)

            for i in range(n):
                if i in keep_set:
                    continue
                upd = self._span_critic.update_redundancy(scores[i], texts[i], selected)
                updated[i] = upd
                is_slot = slot_index is not None and slot_index.sentence_has_slot(texts[i])
                if upd.net_keep_score >= self.sentence_threshold or upd.should_force_keep or is_slot:
                    keep_set.add(i)
                    selected.append(texts[i])

            if len(keep_set) < self.min_sentences:
                for idx in sorted(range(n), key=lambda i: scores[i].keep_prob, reverse=True):
                    keep_set.add(idx)
                    if len(keep_set) >= self.min_sentences:
                        break

            buffered: set[int] = set()
            for idx in keep_set:
                for off in range(-self.context_buffer, self.context_buffer + 1):
                    nb = idx + off
                    if 0 <= nb < n:
                        buffered.add(nb)

        # Capture for SpanCritic training
        if GranularSelector._capture is not None:
            block_id = GranularSelector._block_seq
            GranularSelector._block_seq += 1
            for i in range(n):
                sc = updated[i]
                GranularSelector._capture.append({
                    "text": texts[i], "block_id": block_id,
                    "by_selection": i in keep_set, "kept": i in buffered,
                    "keep_prob": sc.keep_prob, "fragility_prob": sc.fragility_prob,
                    "redundancy_prob": sc.redundancy_prob,
                    "position": i, "total": n,
                    "loss_type_probs": sc.loss_type_probs,
                    "query_tokens": query_tokens,
                })

        if len(buffered) == n:
            return text

        parts: list[str] = []
        prev: int | None = None
        for idx in sorted(buffered):
            if prev is not None and idx > prev + 1:
                parts.append(_ELLIPSIS)
            parts.append(texts[idx])
            prev = idx

        return " ".join(parts) if parts else text

    # ------------------------------------------------------------------
    # Stage 2b: Sentence-level — heuristic fallback (no SpanCritic)
    # ------------------------------------------------------------------

    def _trim_sentences(
        self,
        text: str,
        query_tokens: list[str],
        slot_index: "SlotIndex | None",
    ) -> str:
        sentences = self._splitter.split_sentences(text)
        if len(sentences) <= 2:
            return text

        n      = len(sentences)
        texts  = [s.text for s in sentences]
        scores = _score_texts(query_tokens, texts)

        max_idx  = max(range(n), key=lambda i: scores[i])
        keep_set = {max_idx}

        for i, score in enumerate(scores):
            if score >= self.sentence_threshold:
                keep_set.add(i)
        if slot_index is not None:
            for i, s in enumerate(sentences):
                if slot_index.sentence_has_slot(s.text):
                    keep_set.add(i)
        while len(keep_set) < self.min_sentences:
            for idx in sorted(range(n), key=lambda i: scores[i], reverse=True):
                keep_set.add(idx)
                if len(keep_set) >= self.min_sentences:
                    break

        buffered: set[int] = set()
        for idx in keep_set:
            for off in range(-self.context_buffer, self.context_buffer + 1):
                nb = idx + off
                if 0 <= nb < n:
                    buffered.add(nb)

        if GranularSelector._capture is not None:
            block_id = GranularSelector._block_seq
            GranularSelector._block_seq += 1
            for i in range(n):
                GranularSelector._capture.append({
                    "text": texts[i], "block_id": block_id,
                    "by_selection": i in keep_set, "kept": i in buffered,
                    "keep_prob": float(scores[i]), "fragility_prob": 0.0,
                    "redundancy_prob": 0.0, "position": i, "total": n,
                    "loss_type_probs": {}, "query_tokens": query_tokens,
                })

        if len(buffered) == n:
            return text

        parts: list[str] = []
        prev: int | None = None
        for idx in sorted(buffered):
            if prev is not None and idx > prev + 1:
                parts.append(_ELLIPSIS)
            parts.append(texts[idx])
            prev = idx

        return " ".join(parts) if parts else text

    # ------------------------------------------------------------------
    # Stage 3: Token-level
    # ------------------------------------------------------------------

    def _trim_tokens(
        self,
        text: str,
        query_tokens: list[str],
        slot_index: "SlotIndex | None",
        token_keep_fraction: float,
    ) -> str:
        """
        Compress tokens within every sentence in text.

        Slot sentences and SpanCritic-fragile sentences keep all tokens.
        Per-sentence fraction is modulated by SpanCritic keep_prob (±0.20)
        so important sentences retain slightly more tokens.

        Hard guards (negations, numbers, proper nouns) can never be dropped.
        """
        sentences = self._splitter.split_sentences(text)
        if len(sentences) <= 1:
            return text

        texts           = [s.text for s in sentences]
        query_token_set = frozenset(query_tokens)
        idf             = _build_sentence_idf(texts)

        critic_scores = None
        if self._span_critic is not None:
            try:
                critic_scores = self._span_critic.score_batch(
                    texts, query_tokens, slot_index=slot_index
                )
            except Exception:
                pass

        result: list[str] = []
        for i, sent_text in enumerate(texts):
            is_slot    = slot_index is not None and slot_index.sentence_has_slot(sent_text)
            is_fragile = critic_scores is not None and critic_scores[i].should_force_keep
            force_keep = is_slot or is_fragile

            if force_keep or len(sent_text) < _MIN_SENT_CHARS_FOR_TOKEN_TRIM:
                result.append(sent_text)
                continue

            # Modulate fraction by sentence importance
            if critic_scores is not None:
                adj      = (critic_scores[i].keep_prob - 0.5) * 0.40
                per_frac = max(0.30, min(1.0, token_keep_fraction + adj))
            else:
                per_frac = max(0.30, token_keep_fraction)

            result.append(
                _compress_sentence_tokens(sent_text, query_token_set, idf, per_frac)
            )

        joined = " ".join(result)
        return joined if joined != text else text
