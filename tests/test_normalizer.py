"""Tests for the normalization layer."""

import json
import pytest

from optimizerllm.core.block import make_retrieved_doc_block, make_system_block
from optimizerllm.normalization.normalizer import Normalizer, TextMinifier


class TestNormalizerWhitespace:
    def test_strips_trailing_spaces(self):
        normalizer = Normalizer()
        from optimizerllm.core.block import make_message_block
        block = make_message_block("user", "hello   \nworld   \n")
        kept, _ = normalizer.run([block])
        assert kept[0].content == "hello\nworld"

    def test_collapses_many_blank_lines(self):
        normalizer = Normalizer()
        from optimizerllm.core.block import make_message_block
        block = make_message_block("user", "line1\n\n\n\n\nline2")
        kept, _ = normalizer.run([block])
        assert kept[0].content == "line1\n\nline2"

    def test_strips_leading_trailing_whitespace(self):
        normalizer = Normalizer()
        from optimizerllm.core.block import make_message_block
        block = make_message_block("user", "   hello world   ")
        kept, _ = normalizer.run([block])
        assert kept[0].content == "hello world"


class TestNormalizerJSON:
    def test_canonicalizes_json_keys(self):
        normalizer = Normalizer()
        payload = '{"z": 1, "a": 2, "m": 3}'
        from optimizerllm.core.block import make_message_block
        block = make_message_block("user", payload)
        kept, _ = normalizer.run([block])
        parsed = json.loads(kept[0].content)
        assert list(parsed.keys()) == sorted(parsed.keys())

    def test_strips_null_values(self):
        normalizer = Normalizer()
        payload = '{"key": null, "other": "value", "empty": ""}'
        from optimizerllm.core.block import make_message_block
        block = make_message_block("user", payload)
        kept, _ = normalizer.run([block])
        parsed = json.loads(kept[0].content)
        assert "key" not in parsed
        assert "empty" not in parsed
        assert parsed["other"] == "value"

    def test_non_json_unchanged(self):
        normalizer = Normalizer()
        from optimizerllm.core.block import make_message_block
        text = "plain text, not json"
        block = make_message_block("user", text)
        kept, _ = normalizer.run([block])
        assert kept[0].content == text


class TestNormalizerDedup:
    def test_removes_exact_duplicates(self, duplicate_blocks):
        normalizer = Normalizer()
        kept, removed = normalizer.run(duplicate_blocks)
        # b1 and b2 have identical content; one should be removed
        assert len(kept) == 2
        assert len(removed) == 1
        assert removed[0].drop_reason == "exact_duplicate"

    def test_keeps_different_content(self, duplicate_blocks):
        normalizer = Normalizer()
        kept, _ = normalizer.run(duplicate_blocks)
        contents = {b.content for b in kept}
        assert "Same content here." in contents
        assert "Different content." in contents

    def test_no_dedup_when_disabled(self, duplicate_blocks):
        normalizer = Normalizer(deduplicate=False)
        kept, removed = normalizer.run(duplicate_blocks)
        assert len(kept) == 3
        assert len(removed) == 0


class TestNormalizerEmpty:
    def test_removes_empty_blocks(self):
        normalizer = Normalizer()
        from optimizerllm.core.block import make_message_block
        block = make_message_block("user", "   \n  \n  ")
        kept, removed = normalizer.run([block])
        assert len(kept) == 0
        assert len(removed) == 1
        assert removed[0].drop_reason == "empty_after_normalization"


class TestTextMinifier:
    def test_removes_repeated_punctuation(self):
        minifier = TextMinifier()
        assert minifier.minify("Hello!!!") == "Hello!"
        assert minifier.minify("Really???") == "Really?"

    def test_removes_separator_lines(self):
        minifier = TextMinifier()
        text = "Section 1\n----\nSection 2"
        result = minifier.minify(text)
        assert "----" not in result
        assert "Section 1" in result
        assert "Section 2" in result

    def test_removes_empty_xml_tags(self):
        minifier = TextMinifier()
        text = "Some text <note></note> more text"
        result = minifier.minify(text)
        assert "<note></note>" not in result

    def test_skips_tool_schema_blocks(self):
        minifier = TextMinifier()
        from optimizerllm.core.block import make_tool_schema_block
        tool_block = make_tool_schema_block('{"name": "test!!!"}')
        result = minifier.minify_blocks([tool_block])
        assert result[0].content == '{"name": "test!!!"}'
