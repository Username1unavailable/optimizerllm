"""
SemanticReorder — reorder retrieved documents by marginal information gain.

Instead of passing documents to the granular selector in arbitrary order,
this module reorders them so the most relevant (and most diverse) documents
appear first. This improves:
  - Budget control: higher-value docs survive truncation
  - Cache prefix stability: stable high-value ordering
  - Granular selector: earlier docs set the relevance baseline

Algorithm: greedy MMR selection
  1. Score all docs by query relevance (BM25+TF-IDF)
  2. Iteratively select the doc maximizing: relevance - λ × max_sim(doc, selected)
  3. Return docs in selected order; non-doc blocks retain their original positions

Only RETRIEVED_DOC blocks are reordered. All other blocks preserve their
original relative order.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from optimizerllm.core.block import Block

from optimizerllm.compression.selector import (
    _bm25_scores,
    _jaccard,
    _tfidf_cosine,
    _tokenize,
)


class SemanticReorder:
    """
    Reorder RETRIEVED_DOC blocks by greedy MMR to put the most relevant
    and diverse documents first.

    All other block types retain their original relative positions.
    """

    def __init__(self, mmr_lambda: float = 0.4) -> None:
        self.mmr_lambda = mmr_lambda

    def reorder(self, blocks: list["Block"], query: str) -> list["Block"]:
        """
        Reorder RETRIEVED_DOC blocks within the block list using greedy MMR.

        Steps:
        1. Separate RETRIEVED_DOC blocks from all other blocks.
        2. If fewer than 2 retrieved docs, return unchanged.
        3. Score all docs with BM25+TF-IDF relevance to query.
        4. Greedily select docs using MMR (iterative).
        5. Reinsert the reordered docs at the positions where retrieved_docs
           originally appeared.
        6. Return the block list with all other blocks at their original positions.
        """
        from optimizerllm.core.block import BlockType

        if not query.strip():
            return blocks

        # Collect indices and blocks of RETRIEVED_DOC blocks
        retrieved_indices: list[int] = []
        for i, b in enumerate(blocks):
            if b.block_type == BlockType.RETRIEVED_DOC:
                retrieved_indices.append(i)

        retrieved_blocks = [blocks[i] for i in retrieved_indices]

        if len(retrieved_blocks) < 2:
            return blocks

        # Score all retrieved docs by relevance to query
        query_tokens = _tokenize(query)
        doc_token_lists = [_tokenize(b.content) for b in retrieved_blocks]
        bm25 = _bm25_scores(query_tokens, doc_token_lists)
        tfidf = _tfidf_cosine(query_tokens, doc_token_lists)
        relevance = [0.6 * b + 0.4 * t for b, t in zip(bm25, tfidf)]

        # Pre-compute token sets for Jaccard similarity
        token_sets = [set(tl) for tl in doc_token_lists]

        n = len(retrieved_blocks)
        selected_order: list[int] = []
        remaining: set[int] = set(range(n))

        while remaining:
            best_idx: int | None = None
            best_score = float("-inf")

            for i in remaining:
                rel = relevance[i]
                if selected_order:
                    max_sim = max(
                        _jaccard(token_sets[i], token_sets[j])
                        for j in selected_order
                    )
                else:
                    max_sim = 0.0

                mmr_score = rel - self.mmr_lambda * max_sim
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i

            if best_idx is None:
                break
            selected_order.append(best_idx)
            remaining.remove(best_idx)

        # Build reordered retrieved docs list
        reordered_retrieved = [retrieved_blocks[i] for i in selected_order]

        # Reinsert at the original positions
        result = list(blocks)
        for pos, original_idx in enumerate(retrieved_indices):
            result[original_idx] = reordered_retrieved[pos]

        return result
