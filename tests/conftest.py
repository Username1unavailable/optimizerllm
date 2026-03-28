"""Shared test fixtures."""

import pytest

from optimizerllm.core.block import (
    Block,
    BlockMetadata,
    BlockType,
    Mutability,
    make_message_block,
    make_retrieved_doc_block,
    make_system_block,
)


@pytest.fixture
def system_block():
    return make_system_block("You are a helpful assistant.", source="test")


@pytest.fixture
def user_block():
    return make_message_block("user", "What is the capital of France?", source="test")


@pytest.fixture
def assistant_block():
    return make_message_block("assistant", "The capital of France is Paris.", source="test")


@pytest.fixture
def doc_blocks():
    return [
        make_retrieved_doc_block(
            "Paris is the capital and most populous city of France.",
            source="wiki-paris",
        ),
        make_retrieved_doc_block(
            "Berlin is the capital and largest city of Germany.",
            source="wiki-berlin",
        ),
        make_retrieved_doc_block(
            "Tokyo is the capital of Japan and one of the world's largest cities.",
            source="wiki-tokyo",
        ),
    ]


@pytest.fixture
def sample_messages():
    return [
        {"role": "user", "content": "Hello, I need help with Python."},
        {"role": "assistant", "content": "Sure, what do you need?"},
        {"role": "user", "content": "How do I read a file?"},
    ]


@pytest.fixture
def duplicate_blocks():
    b1 = make_retrieved_doc_block("Same content here.", source="a")
    b2 = make_retrieved_doc_block("Same content here.", source="b")
    b3 = make_retrieved_doc_block("Different content.", source="c")
    # Manually set positions
    b1.metadata.position = 0
    b2.metadata.position = 1
    b3.metadata.position = 2
    return [b1, b2, b3]
