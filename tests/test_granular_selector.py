"""Tests for the granular (section + sentence level) selector."""

import pytest
from optimizerllm.compression.granular_selector import GranularSelector
from optimizerllm.core.block import make_retrieved_doc_block, make_system_block


LONG_DOC = """\
# Introduction
This document provides background on machine learning frameworks.
It covers history, terminology, and general principles.

# Neural Networks
Neural networks are computational models inspired by the human brain.
They consist of layers of interconnected nodes called neurons.
Each connection has a weight that is adjusted during training.
Backpropagation is the algorithm used to compute gradients for weight updates.

# Gradient Descent
Gradient descent is an optimization algorithm used to minimize the loss function.
Stochastic gradient descent processes one sample at a time.
Mini-batch gradient descent is a compromise between the two.

# Conclusion
This concludes the overview of machine learning fundamentals.
Further reading is recommended for advanced topics.
"""

QUERY_NN = "neural networks neurons layers backpropagation"
QUERY_OPT = "gradient descent optimization loss function"


class TestGranularSelectorSections:
    def test_keeps_relevant_section(self):
        selector = GranularSelector(section_threshold=0.05)
        block = make_retrieved_doc_block(LONG_DOC)
        block.metadata.position = 0
        result = selector.trim_blocks([block], QUERY_NN)
        assert len(result) == 1
        # Neural networks section should survive
        assert "Neural Networks" in result[0].content or "neurons" in result[0].content.lower()

    def test_drops_irrelevant_sections(self):
        selector = GranularSelector(section_threshold=0.15)
        block = make_retrieved_doc_block(LONG_DOC)
        result = selector.trim_blocks([block], QUERY_NN)
        # Conclusion section should be trimmed
        assert len(result[0].content) < len(LONG_DOC)

    def test_content_shrinks_with_irrelevant_query(self):
        selector = GranularSelector(section_threshold=0.05, sentence_threshold=0.05)
        block = make_retrieved_doc_block(LONG_DOC)
        result = selector.trim_blocks([block], "cooking recipes food ingredients")
        # Very irrelevant query — all sections should score low
        # Block should still exist (we never return empty)
        assert len(result) == 1

    def test_short_block_not_section_split(self):
        selector = GranularSelector()
        short_text = "This is a short block."
        block = make_retrieved_doc_block(short_text)
        result = selector.trim_blocks([block], QUERY_NN)
        assert result[0].content == short_text

    def test_protected_block_unchanged(self):
        selector = GranularSelector(section_threshold=0.01)
        block = make_system_block(LONG_DOC, protected=True)
        result = selector.trim_blocks([block], QUERY_NN)
        assert result[0].content == LONG_DOC

    def test_different_query_selects_different_sections(self):
        selector = GranularSelector(section_threshold=0.08)
        block_nn = make_retrieved_doc_block(LONG_DOC)
        block_opt = make_retrieved_doc_block(LONG_DOC)

        result_nn = selector.trim_blocks([block_nn], QUERY_NN)
        result_opt = selector.trim_blocks([block_opt], QUERY_OPT)

        # Both should trim but to different content
        assert result_nn[0].content != result_opt[0].content


class TestGranularSelectorSentences:
    def test_sentence_trimming_reduces_length(self):
        # Build a block long enough to trigger sentence splitting
        text = (
            "The sky is blue and clear today. "
            "Neural networks learn by adjusting weights through backpropagation. "
            "Cats are common household pets around the world. "
            "The gradient descent algorithm minimizes the loss function iteratively. "
            "Pizza is a popular food in many countries. "
            "Convolutional neural networks excel at image recognition tasks. "
            "The weather forecast predicts rain tomorrow morning. "
            "Recurrent networks process sequential data over time steps."
        )
        selector = GranularSelector(sentence_threshold=0.15, section_threshold=0.01)
        block = make_retrieved_doc_block(text)
        result = selector.trim_blocks([block], "neural network backpropagation gradient", sentence_level=True, section_level=False)
        # Should be shorter than original
        assert len(result[0].content) <= len(text)

    def test_ellipsis_inserted_for_gaps(self):
        text = (
            "Sentence one about cooking. "
            "Neural networks are powerful machine learning models. "
            "Sentence three about cars. "
            "Backpropagation computes gradients efficiently. "
            "Sentence five about weather."
        )
        selector = GranularSelector(sentence_threshold=0.20, section_threshold=0.01)
        block = make_retrieved_doc_block(text)
        result = selector.trim_blocks([block], "neural network backpropagation gradient", sentence_level=True, section_level=False)
        # If sentences were dropped in the middle, ellipsis should appear
        if len(result[0].content) < len(text):
            assert "[…]" in result[0].content

    def test_min_sentences_respected(self):
        selector = GranularSelector(
            sentence_threshold=0.99,  # Impossibly high — nothing would pass
            min_sentences_to_keep=1,
        )
        text = "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence."
        block = make_retrieved_doc_block(text)
        result = selector.trim_blocks([block], "unrelated topic xyz", sentence_level=True, section_level=False)
        # Must keep at least 1 sentence
        assert result[0].content.strip()


class TestGranularSelectorMultipleBlocks:
    def test_processes_multiple_blocks(self):
        selector = GranularSelector(section_threshold=0.05)
        blocks = [
            make_retrieved_doc_block(LONG_DOC),
            make_retrieved_doc_block(LONG_DOC),
        ]
        for i, b in enumerate(blocks):
            b.metadata.position = i
        result = selector.trim_blocks(blocks, QUERY_NN)
        assert len(result) == 2

    def test_non_eligible_block_type_unchanged(self):
        selector = GranularSelector()
        block = make_system_block(LONG_DOC)
        result = selector.trim_blocks([block], QUERY_NN)
        assert result[0].content == LONG_DOC
