"""Assembly and rendering layer — converts final blocks into provider-ready output."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from optimizerllm.core.block import Block

from optimizerllm.core.block import Block, BlockType


# ---------------------------------------------------------------------------
# Role mapping for message-based providers
# ---------------------------------------------------------------------------

_ROLE_MAP: dict[BlockType, str] = {
    BlockType.SYSTEM_PROMPT: "system",
    BlockType.USER_MESSAGE: "user",
    BlockType.ASSISTANT_MESSAGE: "assistant",
    BlockType.TOOL_RESULT: "tool",
    BlockType.TOOL_SCHEMA: "system",          # Tools injected into system in OpenAI format
    BlockType.RETRIEVED_DOC: "user",          # RAG docs injected as user context
    BlockType.FEW_SHOT_EXAMPLE: "user",       # Inline examples — simplified
    BlockType.CONVERSATION_HISTORY: "user",   # Compacted history
    BlockType.STRUCTURED_DATA: "user",
    BlockType.CODE_SNIPPET: "user",
    BlockType.UNKNOWN: "user",
}


def _role(block: "Block") -> str:
    """Resolve final role for a block."""
    if block.role:
        return block.role
    return _ROLE_MAP.get(block.block_type, "user")


class Renderer:
    """
    Renders the final ordered list of blocks into provider-ready formats.

    Supported output formats:
      - "openai"    → list[dict]  (messages with role/content)
      - "anthropic" → dict with "system" key + "messages" list
      - "text"      → plain concatenated string
      - "raw"       → list[Block] (pass-through)
    """

    def render(
        self,
        blocks: list["Block"],
        fmt: str = "openai",
        *,
        include_block_ids: bool = False,
    ) -> Any:
        """
        Render blocks to the target format.

        include_block_ids: embed _block_id in each rendered message
          (used by CacheOptimizer.inject_cache_hints).
        """
        if fmt == "openai":
            return self._render_openai(blocks, include_block_ids=include_block_ids)
        if fmt == "anthropic":
            return self._render_anthropic(blocks, include_block_ids=include_block_ids)
        if fmt == "text":
            return self._render_text(blocks)
        if fmt == "raw":
            return blocks
        raise ValueError(f"Unknown render format: {fmt!r}. Use 'openai', 'anthropic', 'text', or 'raw'.")

    # ------------------------------------------------------------------
    # OpenAI format
    # ------------------------------------------------------------------

    def _render_openai(
        self, blocks: list["Block"], *, include_block_ids: bool
    ) -> list[dict[str, Any]]:
        """
        Produce an OpenAI-compatible messages list.

        Consecutive messages with the same role are merged (OpenAI requires
        alternating roles for most models).
        """
        messages: list[dict[str, Any]] = []

        for block in blocks:
            role = _role(block)
            content = block.content

            msg: dict[str, Any] = {"role": role, "content": content}
            if include_block_ids:
                msg["_block_id"] = block.id

            # Merge with previous message if same role (except system)
            if (
                messages
                and role != "system"
                and messages[-1]["role"] == role
                and not include_block_ids
            ):
                sep = "\n\n"
                messages[-1]["content"] = messages[-1]["content"] + sep + content
            else:
                messages.append(msg)

        return messages

    # ------------------------------------------------------------------
    # Anthropic format
    # ------------------------------------------------------------------

    def _render_anthropic(
        self, blocks: list["Block"], *, include_block_ids: bool
    ) -> dict[str, Any]:
        """
        Produce an Anthropic-compatible payload.

        Returns:
          {
            "system": "<merged system prompt>",
            "messages": [{"role": ..., "content": ...}, ...]
          }
        """
        system_parts: list[str] = []
        messages: list[dict[str, Any]] = []

        for block in blocks:
            role = _role(block)
            content = block.content

            if role == "system":
                system_parts.append(content)
                continue

            msg: dict[str, Any] = {"role": role, "content": content}
            if include_block_ids:
                msg["_block_id"] = block.id

            # Merge same-role adjacent messages (Anthropic requires alternating)
            if (
                messages
                and messages[-1]["role"] == role
                and not include_block_ids
            ):
                messages[-1]["content"] += "\n\n" + content
            else:
                messages.append(msg)

        result: dict[str, Any] = {"messages": messages}
        if system_parts:
            result["system"] = "\n\n".join(system_parts)
        return result

    # ------------------------------------------------------------------
    # Plain text format
    # ------------------------------------------------------------------

    def _render_text(self, blocks: list["Block"]) -> str:
        parts: list[str] = []
        for block in blocks:
            role = _role(block)
            header = f"[{role.upper()}]"
            parts.append(f"{header}\n{block.content}")
        return "\n\n---\n\n".join(parts)
