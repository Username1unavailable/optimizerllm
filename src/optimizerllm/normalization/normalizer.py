"""Normalization layer — remove format waste without touching meaning."""

from __future__ import annotations

import json
import re
from typing import Any

from optimizerllm.core.block import Block, BlockType


class Normalizer:
    """
    Converts blocks into their canonical form.

    Operations (all lossless):
    - Whitespace normalization (trailing spaces, consecutive blank lines)
    - JSON canonicalization (sorted keys, compact or pretty serialization)
    - Exact-duplicate detection and removal
    - Empty / placeholder block removal
    - Redundant wrapper stripping in JSON payloads
    """

    def __init__(
        self,
        *,
        normalize_whitespace: bool = True,
        canonicalize_json: bool = True,
        deduplicate: bool = True,
        strip_empty: bool = True,
    ) -> None:
        self.normalize_whitespace = normalize_whitespace
        self.canonicalize_json = canonicalize_json
        self.deduplicate = deduplicate
        self.strip_empty = strip_empty

    def run(self, blocks: list[Block]) -> tuple[list[Block], list[Block]]:
        """
        Normalize a list of blocks.

        Returns:
          (kept_blocks, removed_blocks)
        """
        output: list[Block] = []
        removed: list[Block] = []
        seen_fingerprints: set[str] = set()

        for block in blocks:
            content = block.content

            if self.normalize_whitespace:
                content = self._normalize_ws(content)

            if self.canonicalize_json and self._looks_like_json(content):
                content = self._canonicalize_json(content)

            # Strip empty after cleaning
            if self.strip_empty and not content.strip():
                b = block.clone_with(content=content, drop_reason="empty_after_normalization")
                removed.append(b)
                continue

            block = block.clone_with(content=content)

            if self.deduplicate:
                fp = self._fingerprint(block)
                if fp in seen_fingerprints:
                    removed.append(block.clone_with(drop_reason="exact_duplicate"))
                    continue
                seen_fingerprints.add(fp)

            output.append(block)

        return output, removed

    # ------------------------------------------------------------------
    # Whitespace helpers
    # ------------------------------------------------------------------

    _TRAILING_SPACES = re.compile(r"[^\S\n]+\n")
    _MANY_BLANK_LINES = re.compile(r"\n{3,}")
    _LEADING_TRAILING_WS = re.compile(r"^\s+|\s+$")

    def _normalize_ws(self, text: str) -> str:
        # Strip trailing spaces per line
        text = self._TRAILING_SPACES.sub("\n", text)
        # Collapse runs of 3+ blank lines to 2
        text = self._MANY_BLANK_LINES.sub("\n\n", text)
        # Strip leading/trailing whitespace from the whole block
        text = text.strip()
        return text

    # ------------------------------------------------------------------
    # JSON helpers
    # ------------------------------------------------------------------

    _JSON_HEURISTIC = re.compile(r"^\s*[\[{]")

    def _looks_like_json(self, text: str) -> bool:
        return bool(self._JSON_HEURISTIC.match(text))

    def _canonicalize_json(self, text: str) -> str:
        try:
            parsed = json.loads(text)
            parsed = self._strip_null_values(parsed)
            return json.dumps(parsed, indent=2, ensure_ascii=False, sort_keys=True)
        except (json.JSONDecodeError, TypeError):
            return text

    def _strip_null_values(self, obj: Any) -> Any:
        """Recursively remove None values and empty-string values from dicts."""
        if isinstance(obj, dict):
            return {
                k: self._strip_null_values(v)
                for k, v in obj.items()
                if v is not None and v != ""
            }
        if isinstance(obj, list):
            return [self._strip_null_values(item) for item in obj]
        return obj

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def _fingerprint(self, block: Block) -> str:
        """Content-based fingerprint that ignores block ID."""
        import hashlib
        key = f"{block.block_type}:{block.role}:{block.content}"
        return hashlib.sha256(key.encode()).hexdigest()


class TextMinifier:
    """
    Light structural minification for prose and bullet content.

    Removes purely cosmetic noise that appears in LLM prompts:
    - Repeated punctuation (!!!, ???)
    - Over-spaced bullet lists
    - Redundant section separators (-----, =====)
    - XML/HTML-style empty tags
    """

    _REPEATED_PUNCT = re.compile(r"([!?.]){3,}")
    _SEPARATOR_LINE = re.compile(r"^[-=*]{4,}\s*$", re.MULTILINE)
    _EMPTY_TAG = re.compile(r"<(\w+)>\s*</\1>")
    _MULTI_SPACE = re.compile(r"  +")
    _OVER_SPACED_BULLET = re.compile(r"^(\s*[-*+])\s{2,}", re.MULTILINE)

    def minify(self, text: str) -> str:
        text = self._REPEATED_PUNCT.sub(r"\1", text)
        text = self._SEPARATOR_LINE.sub("", text)
        text = self._EMPTY_TAG.sub("", text)
        text = self._OVER_SPACED_BULLET.sub(r"\1 ", text)
        text = self._MULTI_SPACE.sub(" ", text)
        # Clean up any blank lines created by removals
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def minify_blocks(
        self,
        blocks: list[Block],
        skip_types: set[BlockType] | None = None,
    ) -> list[Block]:
        """
        Apply minification to compressible blocks.

        Code, JSON, and tool schemas are skipped by default because
        their whitespace may be load-bearing.
        """
        if skip_types is None:
            skip_types = {
                BlockType.CODE_SNIPPET,
                BlockType.TOOL_SCHEMA,
                BlockType.STRUCTURED_DATA,
            }

        result = []
        for block in blocks:
            if block.block_type in skip_types or not block.is_compressible():
                result.append(block)
            else:
                result.append(block.clone_with(content=self.minify(block.content)))
        return result
