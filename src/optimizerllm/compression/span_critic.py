"""
SpanCritic — multi-signal sentence/span scorer for GranularSelector.

Phase 2 module. Augments the simple BM25+TF-IDF scorer with:
  - Position bonus (topic/conclusion sentences get extra weight)
  - Entity/numeric density bonus (fact-dense sentences are more valuable)
  - MMR-style redundancy tracking (redundant sentences lose score)
  - Fragility detection (reasoning connectors + slot presence → force-keep)
  - Loss-type estimation (which type of information loss if removed)

The critic supports a greedy MMR selection loop:
  1. Call score_batch() to get initial scores (no redundancy applied yet)
  2. Iterate: call update_redundancy() after each selection, recompute net_keep_score
  3. Keep sentences where net_keep_score >= threshold OR should_force_keep is True

This allows redundant content to be dropped more aggressively while unique
and fragile content is protected — driving better compression without quality loss.
"""

from __future__ import annotations

import math
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from optimizerllm.slot_guard.guard import SlotIndex

from optimizerllm.compression.selector import _bm25_scores, _tfidf_cosine, _tokenize

# Default model path — same directory as this file
_DEFAULT_MODEL_PATH = Path(__file__).parent / "span_critic_model.pkl"

# Feature names in extraction order (must match _extract_feature_vector)
FEATURE_NAMES = [
    "lexical_score",
    "position_norm",
    "is_first",
    "is_last",
    "len_tokens_log",
    "numeric_count",
    "numeric_density",
    "connector_count",
    "connector_density",
    "has_slot",
    "has_schema",
    "has_coref",
    "has_temporal",
    "query_overlap_frac",
    "type_token_ratio",
    "avg_word_length",
    "capital_ratio",
    "has_question",
    "has_parentheses",
    "len_chars_log",
]


# Reasoning-chain connectors → fragility signal
_CONNECTOR_PATTERN = re.compile(
    r'\b(because|therefore|since|thus|hence|'
    r'if\s+\w[^,]{0,30}then|'
    r'it\s+follows|given\s+that|suppose|assume|'
    r'first|second|third|then|finally|step\s+\d+|'
    r'as\s+a\s+result|in\s+order\s+to|provided\s+that)\b',
    re.IGNORECASE,
)

# Temporal ordering markers → chronology_break loss type
_TEMPORAL_PATTERN = re.compile(
    r'\b(first|then|after\s+(that|which)|before|finally|next|subsequently|'
    r'previously|at\s+\d{1,2}:\d{2}|on\s+\w+day|by\s+\w+\s+\d{4})\b',
    re.IGNORECASE,
)

# Coreference markers → coref_break loss type
_COREF_PATTERN = re.compile(
    r'\b(he|she|they|it|this|that|these|those|the\s+former|the\s+latter)\s+'
    r'(is|are|was|were|has|have|had|will|would|can|could|should)\b',
    re.IGNORECASE,
)

# API / schema patterns → schema_loss loss type
_SCHEMA_PATTERN = re.compile(
    r'[a-z_][a-z0-9_]*\s*[:=]\s*\S|'   # key: value  or  key=value
    r'"[a-z_][a-z0-9_]+"\s*:|'          # JSON "field":
    r'\$[A-Z_]{2,}|'                     # environment variable $VAR
    r'`[^`]{2,}`',                        # backtick code reference
    re.IGNORECASE,
)

# Numeric / quantity patterns → fact_loss loss type
_NUMERIC_PATTERN = re.compile(r'\b\d+(?:[.,]\d+)?(?:\s*[%$€£]|\s*[kmgb]+)?\b')

# Minimum token set size to compute meaningful Jaccard overlap
_MIN_TOKENS_REDUNDANCY = 5

# Redundancy threshold: overlap at or above this → redundancy_prob = 1.0
_REDUNDANCY_SATURATION = 0.50


@dataclass
class SpanScore:
    """
    Per-sentence / per-span scores from SpanCritic.

    keep_prob:       0–1, probability this sentence adds unique value
    fragility_prob:  0–1, probability removing this sentence breaks something
    redundancy_prob: 0–1, how much this sentence overlaps already-selected spans
    loss_type_probs: probability of each information loss type if removed
    """
    keep_prob: float
    fragility_prob: float
    redundancy_prob: float = 0.0
    loss_type_probs: dict[str, float] = field(default_factory=dict)

    @property
    def net_keep_score(self) -> float:
        """
        MMR-adjusted score.

        Redundancy penalty weight is 0.40 so that marginally relevant sentences
        that are highly redundant still drop, while clearly relevant sentences
        (keep_prob > threshold) survive even at moderate redundancy.
        """
        return self.keep_prob * (1.0 - 0.40 * self.redundancy_prob)

    @property
    def should_force_keep(self) -> bool:
        """True if this span must be kept regardless of relevance score."""
        return self.fragility_prob >= 0.45


class SpanCritic:
    """
    Feature-based sentence scorer with MMR-style redundancy tracking.

    Phase 2A (no trained model): uses hand-crafted heuristics.
    Phase 2B+ (trained model loaded): replaces keep_prob with model prediction
    while retaining heuristic fragility_prob and redundancy_prob.

    Usage::

        critic = SpanCritic()
        critic.load_model()        # loads span_critic_model.pkl if present

        # Step 1: compute initial scores
        scores = critic.score_batch(sentences, query_tokens, slot_index=slot_index)

        # Step 2: MMR greedy loop
        selected: list[str] = []
        keep_set: set[int] = set()
        for i, (sent, score) in enumerate(zip(sentences, scores)):
            updated = critic.update_redundancy(score, sent, selected)
            if updated.net_keep_score >= threshold or updated.should_force_keep:
                keep_set.add(i)
                selected.append(sent)
    """

    def __init__(self, auto_load: bool = True) -> None:
        self._model: "Any | None" = None  # sklearn estimator when trained
        # Auto-load if model file exists (skip in training scripts)
        if auto_load and _DEFAULT_MODEL_PATH.exists():
            try:
                self.load_model()
            except Exception:
                pass  # corrupt / version-mismatch pkl → start heuristic

    # ------------------------------------------------------------------
    # Model I/O
    # ------------------------------------------------------------------

    def load_model(self, path: str | Path | None = None) -> None:
        """Load a trained sklearn model from disk (versioned format)."""
        target = Path(path) if path is not None else _DEFAULT_MODEL_PATH
        with open(target, "rb") as f:
            payload = pickle.load(f)
        # Support both legacy (raw model) and versioned ({model, feature_version}) formats
        if isinstance(payload, dict) and "model" in payload:
            saved_version = payload.get("feature_version", -1)
            if saved_version != len(FEATURE_NAMES):
                raise RuntimeError(
                    f"SpanCritic model version mismatch: "
                    f"saved={saved_version} features, current={len(FEATURE_NAMES)}. "
                    f"Retrain with train_span_critic.py."
                )
            self._model = payload["model"]
        else:
            # Legacy format — accept without version check
            self._model = payload

    def save_model(self, path: str | Path | None = None) -> None:
        """Persist the trained model to disk with version stamp."""
        if self._model is None:
            raise RuntimeError("No model to save — call train_model() first.")
        target = Path(path) if path is not None else _DEFAULT_MODEL_PATH
        payload = {"model": self._model, "feature_version": len(FEATURE_NAMES)}
        with open(target, "wb") as f:
            pickle.dump(payload, f)

    def train_model(self, X: "np.ndarray", y: "np.ndarray", sample_weight: "np.ndarray | None" = None) -> None:
        """
        Fit a GradientBoostingClassifier on labelled sentence feature vectors.

        X: (n_samples, n_features) — from extract_feature_matrix()
        y: (n_samples,) — 1 = must keep, 0 = safe to drop
        sample_weight: optional per-sample importance weights
        """
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("gbc", GradientBoostingClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                min_samples_leaf=5,
                max_features="sqrt",
                random_state=42,
            )),
        ])
        clf.fit(X, y, gbc__sample_weight=sample_weight)
        self._model = clf

    @property
    def has_trained_model(self) -> bool:
        return self._model is not None

    # ------------------------------------------------------------------
    # Feature extraction (used both at inference and training time)
    # ------------------------------------------------------------------

    @staticmethod
    def extract_feature_matrix(
        sentences: list[str],
        query_tokens: list[str],
        *,
        slot_index: "SlotIndex | None" = None,
    ) -> "np.ndarray":
        """
        Build an (n_sentences, n_features) feature matrix.

        Used by train_span_critic.py to build training data, and by
        score_batch() when a trained model is loaded.
        """
        if not sentences:
            return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)

        n = len(sentences)
        token_lists = [_tokenize(s) for s in sentences]
        query_set = set(query_tokens)

        # Lexical scores
        if query_tokens:
            bm25 = _bm25_scores(query_tokens, token_lists)
            tfidf = _tfidf_cosine(query_tokens, token_lists)
            lexical = [0.6 * b + 0.4 * t for b, t in zip(bm25, tfidf)]
        else:
            lexical = [0.0] * n

        rows: list[list[float]] = []
        for i, sent in enumerate(sentences):
            toks = token_lists[i]
            n_toks = max(len(toks), 1)
            numeric_hits = _NUMERIC_PATTERN.findall(sent)
            connector_hits = _CONNECTOR_PATTERN.findall(sent)
            n_num = len(numeric_hits)
            n_conn = len(connector_hits)
            words = sent.split()
            n_words = max(len(words), 1)

            # Structural
            pos_norm = i / max(n - 1, 1)
            is_first = float(i == 0)
            is_last = float(i == n - 1)
            len_toks_log = math.log1p(n_toks)
            len_chars_log = math.log1p(len(sent))

            # Numeric / entity
            numeric_density = n_num / n_toks
            entity_bonus = min(0.12, n_num * 0.04)  # same as heuristic

            # Reasoning connectors
            connector_density = n_conn / n_toks

            # Slots / schema / coref / temporal
            has_slot = float(slot_index is not None and slot_index.sentence_has_slot(sent))
            has_schema = float(bool(_SCHEMA_PATTERN.search(sent)))
            has_coref = float(bool(_COREF_PATTERN.search(sent)))
            has_temporal = float(bool(_TEMPORAL_PATTERN.search(sent)))

            # Query overlap
            sent_set = set(toks)
            query_overlap = len(sent_set & query_set) / max(len(query_set), 1)

            # Lexical diversity
            type_token_ratio = len(sent_set) / n_toks

            # Surface
            avg_word_len = sum(len(w) for w in words) / n_words
            cap_words = sum(1 for w in words if w and w[0].isupper())
            capital_ratio = cap_words / n_words
            has_question = float("?" in sent)
            has_parens = float("(" in sent or "[" in sent)

            rows.append([
                lexical[i],         # 0  lexical_score
                pos_norm,           # 1  position_norm
                is_first,           # 2  is_first
                is_last,            # 3  is_last
                len_toks_log,       # 4  len_tokens_log
                float(n_num),       # 5  numeric_count
                numeric_density,    # 6  numeric_density
                float(n_conn),      # 7  connector_count
                connector_density,  # 8  connector_density
                has_slot,           # 9  has_slot
                has_schema,         # 10 has_schema
                has_coref,          # 11 has_coref
                has_temporal,       # 12 has_temporal
                query_overlap,      # 13 query_overlap_frac
                type_token_ratio,   # 14 type_token_ratio
                avg_word_len,       # 15 avg_word_length
                capital_ratio,      # 16 capital_ratio
                has_question,       # 17 has_question
                has_parens,         # 18 has_parentheses
                len_chars_log,      # 19 len_chars_log
            ])

        return np.array(rows, dtype=np.float32)

    def score_batch(
        self,
        sentences: list[str],
        query_tokens: list[str],
        *,
        slot_index: "SlotIndex | None" = None,
    ) -> list[SpanScore]:
        """
        Compute initial SpanScore for every sentence in the block.

        When a trained model is loaded, keep_prob comes from the model.
        Otherwise falls back to the hand-crafted heuristic.

        redundancy_prob is 0.0 for all returned scores; call update_redundancy()
        inside the MMR selection loop to fill it in.
        """
        if not sentences:
            return []

        n = len(sentences)

        if self._model is not None:
            # --- Trained model path ---
            X = self.extract_feature_matrix(sentences, query_tokens, slot_index=slot_index)
            keep_probs = self._model.predict_proba(X)[:, 1].tolist()  # P(keep)
        else:
            # --- Heuristic path (Phase 2A) ---
            token_lists = [_tokenize(s) for s in sentences]
            if query_tokens:
                bm25 = _bm25_scores(query_tokens, token_lists)
                tfidf = _tfidf_cosine(query_tokens, token_lists)
                lexical = [0.6 * b + 0.4 * t for b, t in zip(bm25, tfidf)]
            else:
                lexical = [0.0] * n
            keep_probs = None  # computed per sentence below

        scores: list[SpanScore] = []
        for i, sent in enumerate(sentences):
            if keep_probs is not None:
                # Model-predicted keep_prob; fragility still from heuristics
                score = self._score_single(
                    sent,
                    keep_probs[i],  # use model output directly as lexical proxy
                    position=i,
                    total=n,
                    slot_index=slot_index,
                    model_keep_prob=keep_probs[i],
                )
            else:
                score = self._score_single(
                    sent,
                    lexical[i],  # type: ignore[possibly-undefined]
                    position=i,
                    total=n,
                    slot_index=slot_index,
                )
            scores.append(score)

        return scores

    def update_redundancy(
        self,
        score: SpanScore,
        sentence: str,
        selected_sentences: list[str],
    ) -> SpanScore:
        """
        Return a new SpanScore with redundancy_prob recomputed against selected_sentences.

        Checks against the last 8 selected sentences (recency heuristic) for efficiency.
        """
        if not selected_sentences:
            return score

        sent_tokens = set(_tokenize(sentence))
        if len(sent_tokens) < _MIN_TOKENS_REDUNDANCY:
            return score

        max_overlap = 0.0
        for sel in selected_sentences[-8:]:
            sel_tokens = set(_tokenize(sel))
            if len(sel_tokens) < _MIN_TOKENS_REDUNDANCY:
                continue
            union = sent_tokens | sel_tokens
            if not union:
                continue
            overlap = len(sent_tokens & sel_tokens) / len(union)
            max_overlap = max(max_overlap, overlap)

        # Scale so that max_overlap == _REDUNDANCY_SATURATION → redundancy_prob == 1.0
        redundancy_prob = min(1.0, max_overlap / _REDUNDANCY_SATURATION)

        return SpanScore(
            keep_prob=score.keep_prob,
            fragility_prob=score.fragility_prob,
            redundancy_prob=round(redundancy_prob, 3),
            loss_type_probs=score.loss_type_probs,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _score_single(
        sentence: str,
        lexical_score: float,
        *,
        position: int,
        total: int,
        slot_index: "SlotIndex | None",
        model_keep_prob: float | None = None,
    ) -> SpanScore:
        """Compute all non-redundancy signals for a single sentence.

        When model_keep_prob is provided (trained model path), it replaces the
        heuristic keep_prob computation.  Fragility and loss-type signals always
        use heuristics — they are protective guards, not learned scores.
        """

        # Always-needed signals (used for fragility + loss types regardless of path)
        numeric_hits = _NUMERIC_PATTERN.findall(sentence)
        connector_hits = _CONNECTOR_PATTERN.findall(sentence)
        n_nums = len(numeric_hits)
        n_connectors = len(connector_hits)

        if model_keep_prob is not None:
            keep_prob = float(np.clip(model_keep_prob, 0.0, 1.0))
        else:
            # --- Position bonus ---
            position_bonus = 0.0
            if total > 2:
                if position == 0:
                    position_bonus = 0.15
                elif position == total - 1:
                    position_bonus = 0.05

            entity_bonus = min(0.12, n_nums * 0.04)
            keep_prob = min(1.0, lexical_score + position_bonus + entity_bonus)

        # --- Fragility signals ---
        has_slot = slot_index is not None and slot_index.sentence_has_slot(sentence)

        fragility_prob = min(
            1.0,
            n_connectors * 0.18           # each reasoning connector adds 18%
            + (0.55 if has_slot else 0.0)  # known brittle fact → strong fragility
        )

        # --- Loss-type probabilities ---
        has_schema = bool(_SCHEMA_PATTERN.search(sentence))
        has_coref = bool(_COREF_PATTERN.search(sentence))
        has_temporal = bool(_TEMPORAL_PATTERN.search(sentence))

        loss_type_probs = {
            "fact_loss":        round(min(1.0, 0.30 + n_nums * 0.15), 3),
            "schema_loss":      0.85 if has_schema else 0.05,
            "reasoning_break":  round(min(1.0, n_connectors * 0.25), 3),
            "coref_break":      0.65 if has_coref else 0.08,
            "chronology_break": 0.55 if has_temporal else 0.08,
        }

        return SpanScore(
            keep_prob=round(keep_prob, 3),
            fragility_prob=round(fragility_prob, 3),
            redundancy_prob=0.0,
            loss_type_probs=loss_type_probs,
        )
