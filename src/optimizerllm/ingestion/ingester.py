"""Ingestion layer — converts raw user inputs into typed Block objects."""

from __future__ import annotations

import json
from typing import Any, Sequence

from optimizerllm.core.block import (
    Block,
    BlockMetadata,
    BlockType,
    CompressionPolicy,
    Mutability,
    make_example_block,
    make_message_block,
    make_retrieved_doc_block,
    make_system_block,
    make_tool_schema_block,
)


class Ingester:
    """
    Converts heterogeneous inputs into a flat list of Block objects.

    Accepted inputs:
    - messages: OpenAI-style list of {"role": ..., "content": ...}
    - system: plain string system prompt
    - tools: list of tool-schema dicts
    - documents: list of strings or {"content": ..., "source": ...} dicts
    - examples: list of {"input": ..., "output": ...} or plain strings
    - history: alias for messages (tagged as CONVERSATION_HISTORY)
    - structured: arbitrary JSON payload
    """

    def ingest(
        self,
        *,
        messages: Sequence[dict[str, Any]] | None = None,
        system: str | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
        documents: Sequence[str | dict[str, Any]] | None = None,
        examples: Sequence[str | dict[str, Any]] | None = None,
        history: Sequence[dict[str, Any]] | None = None,
        structured: dict[str, Any] | Sequence[dict[str, Any]] | None = None,
        raw_blocks: Sequence[Block] | None = None,
    ) -> list[Block]:
        """
        Ingest all input sources and return an ordered list of Blocks.

        Ordering (from most stable to most dynamic, drives cache prefix):
          1. system prompt
          2. tool schemas
          3. few-shot examples
          4. retrieved documents
          5. conversation history
          6. current-turn messages
          7. structured data
          8. raw blocks (passed through as-is)
        """
        blocks: list[Block] = []
        position = 0

        def _stamp(b: Block) -> Block:
            nonlocal position
            b.metadata.position = position
            position += 1
            return b

        # 1. System prompt
        if system is not None and system.strip():
            blocks.append(_stamp(make_system_block(system, source="system")))

        # 2. Tool schemas
        if tools:
            for i, tool in enumerate(tools):
                content = self._serialize_tool(tool)
                b = make_tool_schema_block(content, source=f"tool:{i}")
                blocks.append(_stamp(b))

        # 3. Few-shot examples
        if examples:
            for i, ex in enumerate(examples):
                content = self._serialize_example(ex)
                b = make_example_block(content, source=f"example:{i}")
                blocks.append(_stamp(b))

        # 4. Retrieved documents
        if documents:
            for i, doc in enumerate(documents):
                content, source = self._parse_doc(doc, i)
                b = make_retrieved_doc_block(content, source=source)
                blocks.append(_stamp(b))

        # 5. Conversation history (older turns, lower priority than current messages)
        if history:
            for i, msg in enumerate(history):
                b = self._message_to_block(msg, source=f"history:{i}")
                b.block_type = BlockType.CONVERSATION_HISTORY
                b.metadata.priority = 35
                b.metadata.mutability = Mutability.DROPPABLE
                b.metadata.cacheability = False
                blocks.append(_stamp(b))

        # 6. Current-turn messages
        if messages:
            for i, msg in enumerate(messages):
                b = self._message_to_block(msg, source=f"message:{i}")
                # Last user message is protected — it is the active query
                if i == len(messages) - 1 and msg.get("role") == "user":
                    b.metadata.protected = True
                    b.metadata.priority = 100
                    b.metadata.mutability = Mutability.IMMUTABLE
                blocks.append(_stamp(b))

        # 7. Structured data
        if structured is not None:
            content = self._serialize_structured(structured)
            b = Block(
                content=content,
                block_type=BlockType.STRUCTURED_DATA,
                metadata=BlockMetadata(
                    source="structured",
                    priority=60,
                    mutability=Mutability.COMPRESSIBLE,
                    cacheability=False,
                    compression_policy=CompressionPolicy.DEFAULT,
                ),
            )
            blocks.append(_stamp(b))

        # 8. Pass-through raw blocks
        if raw_blocks:
            for b in raw_blocks:
                blocks.append(_stamp(b))

        return blocks

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _message_to_block(self, msg: dict[str, Any], source: str) -> Block:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if not isinstance(content, str):
            # Content may be a list of content parts (vision etc.) — serialize
            content = json.dumps(content, ensure_ascii=False)
        return make_message_block(role, content, source=source)

    def _serialize_tool(self, tool: dict[str, Any]) -> str:
        return json.dumps(tool, indent=2, ensure_ascii=False, sort_keys=True)

    def _serialize_example(self, ex: str | dict[str, Any]) -> str:
        if isinstance(ex, str):
            return ex
        # Expect {"input": ..., "output": ...} or arbitrary dict
        return json.dumps(ex, indent=2, ensure_ascii=False)

    def _parse_doc(
        self, doc: str | dict[str, Any], index: int
    ) -> tuple[str, str]:
        if isinstance(doc, str):
            return doc, f"doc:{index}"
        content = doc.get("content") or doc.get("text") or json.dumps(doc, ensure_ascii=False)
        source = doc.get("source") or doc.get("title") or f"doc:{index}"
        return str(content), str(source)

    def _serialize_structured(
        self, data: dict[str, Any] | Sequence[dict[str, Any]]
    ) -> str:
        return json.dumps(data, indent=2, ensure_ascii=False)
