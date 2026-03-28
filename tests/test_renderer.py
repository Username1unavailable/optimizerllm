"""Tests for the rendering layer."""

import pytest

from optimizerllm.core.block import make_message_block, make_system_block, make_retrieved_doc_block
from optimizerllm.rendering.renderer import Renderer


class TestOpenAIRenderer:
    def test_basic_messages(self):
        renderer = Renderer()
        blocks = [
            make_system_block("You are helpful."),
            make_message_block("user", "Hello"),
            make_message_block("assistant", "Hi there!"),
        ]
        result = renderer.render(blocks, fmt="openai")
        assert isinstance(result, list)
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert result[2]["role"] == "assistant"

    def test_merges_consecutive_same_role(self):
        renderer = Renderer()
        blocks = [
            make_retrieved_doc_block("Doc 1"),
            make_retrieved_doc_block("Doc 2"),
            make_message_block("user", "My question"),
        ]
        # Both doc blocks render as "user" role — should be merged
        result = renderer.render(blocks, fmt="openai")
        user_messages = [m for m in result if m["role"] == "user"]
        # Should be merged into one
        assert len(user_messages) == 1
        assert "Doc 1" in user_messages[0]["content"]
        assert "Doc 2" in user_messages[0]["content"]

    def test_no_block_ids_in_default_output(self):
        renderer = Renderer()
        blocks = [make_message_block("user", "Hello")]
        result = renderer.render(blocks, fmt="openai")
        assert "_block_id" not in result[0]

    def test_block_ids_when_requested(self):
        renderer = Renderer()
        blocks = [make_message_block("user", "Hello")]
        result = renderer.render(blocks, fmt="openai", include_block_ids=True)
        assert "_block_id" in result[0]


class TestAnthropicRenderer:
    def test_separates_system_from_messages(self):
        renderer = Renderer()
        blocks = [
            make_system_block("System instructions here."),
            make_message_block("user", "Hello"),
            make_message_block("assistant", "Hi!"),
        ]
        result = renderer.render(blocks, fmt="anthropic")
        assert isinstance(result, dict)
        assert "system" in result
        assert "messages" in result
        assert "System instructions" in result["system"]
        assert len(result["messages"]) == 2

    def test_no_system_prompt(self):
        renderer = Renderer()
        blocks = [make_message_block("user", "Hello")]
        result = renderer.render(blocks, fmt="anthropic")
        assert "system" not in result
        assert len(result["messages"]) == 1


class TestTextRenderer:
    def test_produces_string(self):
        renderer = Renderer()
        blocks = [
            make_system_block("Instructions."),
            make_message_block("user", "Question?"),
        ]
        result = renderer.render(blocks, fmt="text")
        assert isinstance(result, str)
        assert "Instructions." in result
        assert "Question?" in result

    def test_separates_blocks(self):
        renderer = Renderer()
        blocks = [
            make_message_block("user", "Part 1"),
            make_message_block("assistant", "Part 2"),
        ]
        result = renderer.render(blocks, fmt="text")
        assert "---" in result


class TestRendererUnknownFormat:
    def test_raises_on_unknown_format(self):
        renderer = Renderer()
        blocks = [make_message_block("user", "hello")]
        with pytest.raises(ValueError, match="Unknown render format"):
            renderer.render(blocks, fmt="invalid_format")
