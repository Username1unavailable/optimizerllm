"""Tests for the content splitter."""

import pytest
from optimizerllm.compression.splitter import ContentSplitter


class TestSectionSplitting:
    def test_splits_markdown_headings(self):
        text = "# Introduction\nSome intro text.\n\n## Methods\nMethod details.\n\n## Results\nResult data."
        splitter = ContentSplitter()
        sections = splitter.split_sections(text)
        assert len(sections) == 3
        assert sections[0].heading == "Introduction"
        assert sections[1].heading == "Methods"
        assert sections[2].heading == "Results"

    def test_heading_level_captured(self):
        text = "# H1\ntext\n\n## H2\ntext\n\n### H3\ntext"
        splitter = ContentSplitter()
        sections = splitter.split_sections(text)
        levels = [s.level for s in sections]
        assert levels == [1, 2, 3]

    def test_preamble_before_first_heading(self):
        text = "Some preamble text.\n\n# Heading\nBody text."
        splitter = ContentSplitter()
        sections = splitter.split_sections(text)
        assert sections[0].heading is None
        assert "preamble" in sections[0].body
        assert sections[1].heading == "Heading"

    def test_no_headings_splits_by_paragraph(self):
        text = "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph here."
        splitter = ContentSplitter()
        sections = splitter.split_sections(text)
        assert len(sections) == 3
        assert all(s.heading is None for s in sections)

    def test_single_paragraph_returns_one_section(self):
        text = "Just one paragraph with no breaks."
        splitter = ContentSplitter()
        sections = splitter.split_sections(text)
        assert len(sections) == 1

    def test_empty_text_returns_empty(self):
        splitter = ContentSplitter()
        assert splitter.split_sections("") == []
        assert splitter.split_sections("   ") == []

    def test_full_text_property(self):
        text = "## Methods\nHere are the methods."
        splitter = ContentSplitter()
        sections = splitter.split_sections(text)
        assert sections[0].heading == "Methods"
        assert "##" in sections[0].full_text
        assert "Methods" in sections[0].full_text


class TestSentenceSplitting:
    def test_splits_on_period_capital(self):
        text = "First sentence. Second sentence. Third sentence."
        splitter = ContentSplitter()
        sentences = splitter.split_sentences(text)
        assert len(sentences) >= 2

    def test_bullet_list_treated_as_sentences(self):
        text = "- Item one\n- Item two\n- Item three\n- Item four"
        splitter = ContentSplitter()
        sentences = splitter.split_sentences(text)
        assert len(sentences) == 4

    def test_empty_returns_empty(self):
        splitter = ContentSplitter()
        assert splitter.split_sentences("") == []

    def test_indices_assigned(self):
        text = "First. Second. Third."
        splitter = ContentSplitter()
        sentences = splitter.split_sentences(text)
        indices = [s.index for s in sentences]
        assert indices == list(range(len(indices)))


class TestReassembly:
    def test_reassemble_sections(self):
        text = "# Section A\nContent A.\n\n# Section B\nContent B."
        splitter = ContentSplitter()
        sections = splitter.split_sections(text)
        reassembled = splitter.reassemble_sections(sections)
        assert "Section A" in reassembled
        assert "Content A" in reassembled
        assert "Section B" in reassembled

    def test_reassemble_subset(self):
        text = "# A\nText A.\n\n# B\nText B.\n\n# C\nText C."
        splitter = ContentSplitter()
        sections = splitter.split_sections(text)
        # Keep only A and C
        kept = [sections[0], sections[2]]
        result = splitter.reassemble_sections(kept)
        assert "Text A" in result
        assert "Text C" in result
        assert "Text B" not in result

    def test_reassemble_sentences(self):
        splitter = ContentSplitter()
        from optimizerllm.compression.splitter import Sentence
        sentences = [Sentence(text="Hello.", index=0), Sentence(text="World.", index=1)]
        result = splitter.reassemble_sentences(sentences)
        assert "Hello." in result
        assert "World." in result
