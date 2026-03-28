"""Tests for the query-aware selector."""

import pytest

from optimizerllm.compression.selector import QueryAwareSelector, _tokenize, _tfidf_cosine
from optimizerllm.core.block import make_retrieved_doc_block, make_system_block


class TestTokenize:
    def test_basic(self):
        assert _tokenize("Hello World") == ["hello", "world"]

    def test_strips_punctuation(self):
        tokens = _tokenize("hello, world!")
        assert "hello" in tokens
        assert "world" in tokens
        assert "," not in tokens

    def test_empty(self):
        assert _tokenize("") == []


class TestTfidfCosine:
    def test_identical_query_and_doc(self):
        tokens = ["python", "programming"]
        scores = _tfidf_cosine(tokens, [tokens])
        assert scores[0] > 0.9

    def test_unrelated_docs_score_low(self):
        query = ["python", "programming"]
        docs = [["cooking", "recipes", "food"], ["car", "engine", "motor"]]
        scores = _tfidf_cosine(query, docs)
        assert all(s < 0.3 for s in scores)

    def test_returns_correct_length(self):
        query = ["hello"]
        docs = [["hello"], ["world"], ["foo"]]
        scores = _tfidf_cosine(query, docs)
        assert len(scores) == 3

    def test_empty_query_returns_zeros(self):
        scores = _tfidf_cosine([], [["hello", "world"]])
        assert scores[0] == 0.0


class TestQueryAwareSelector:
    def test_scores_all_blocks(self, doc_blocks):
        selector = QueryAwareSelector()
        query = "capital of France"
        # Add token counts (selector doesn't need them but blocks need positions)
        for i, b in enumerate(doc_blocks):
            b.metadata.position = i
        scored = selector.score_blocks(doc_blocks, query)
        assert all(b.score is not None for b in scored)
        assert all(0.0 <= b.score <= 1.0 for b in scored)

    def test_france_doc_scores_highest(self, doc_blocks):
        selector = QueryAwareSelector()
        query = "capital of France Paris"
        for i, b in enumerate(doc_blocks):
            b.metadata.position = i
        scored = selector.score_blocks(doc_blocks, query)
        france_score = scored[0].score  # Paris doc
        other_scores = [scored[1].score, scored[2].score]
        assert france_score > max(other_scores)

    def test_empty_query_assigns_ones(self, doc_blocks):
        selector = QueryAwareSelector()
        scored = selector.score_blocks(doc_blocks, "")
        assert all(b.score == 1.0 for b in scored)

    def test_system_prompt_always_preserved(self, system_block):
        selector = QueryAwareSelector()
        system_block.metadata.position = 0
        scored = selector.score_blocks([system_block], "some query")
        # Non-selectable types get score=1.0
        assert scored[0].score == 1.0

    def test_filter_by_threshold_removes_low_scores(self, doc_blocks):
        selector = QueryAwareSelector()
        # Manually set scores
        doc_blocks[0] = doc_blocks[0].clone_with(score=0.8)
        doc_blocks[1] = doc_blocks[1].clone_with(score=0.05)
        doc_blocks[2] = doc_blocks[2].clone_with(score=0.4)

        kept, dropped = selector.filter_by_threshold(doc_blocks, threshold=0.15)
        assert len(kept) == 2
        assert len(dropped) == 1
        assert dropped[0].score == 0.05

    def test_protected_blocks_always_kept(self):
        selector = QueryAwareSelector()
        from optimizerllm.core.block import make_system_block
        block = make_system_block("Important system instructions.")
        block = block.clone_with(score=0.0)  # Force zero score
        kept, dropped = selector.filter_by_threshold([block], threshold=0.99)
        assert len(kept) == 1
        assert len(dropped) == 0
