"""Cache optimizer — shapes the prompt prefix for maximum provider cache hits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from optimizerllm.core.block import Block

from optimizerllm.core.block import Block, BlockType, CompressionPolicy
from optimizerllm.token_accounting.counter import CACHE_MIN_TOKENS


# ---------------------------------------------------------------------------
# Cache-stable block types (content that rarely changes turn-to-turn)
# ---------------------------------------------------------------------------
_CACHE_STABLE_TYPES = {
    BlockType.SYSTEM_PROMPT,
    BlockType.TOOL_SCHEMA,
    BlockType.FEW_SHOT_EXAMPLE,
}

# Provider-specific cache annotation keys
_ANTHROPIC_CACHE_CONTROL = {"type": "ephemeral"}


@dataclass
class CacheAnalysis:
    """Report of how the optimized prompt will interact with provider caching."""

    provider: str
    cacheable_prefix_tokens: int
    cache_eligible: bool                  # Meets provider minimum
    cache_min_tokens: int
    cache_hint_blocks: list[str]          # IDs of blocks marked as cache anchors
    expected_hit_potential: str           # "high" | "medium" | "low" | "none"


class CacheOptimizer:
    """
    Reorders blocks to maximize provider cache hits and generates
    provider-specific cache hints.

    Strategy:
      1. Move all cache-stable blocks (system, tools, examples) to the front.
         Within that group, preserve their original relative order.
      2. Dynamic blocks (docs, history, current messages) follow,
         also in their original relative order.
      3. The current user turn is always last.
      4. For Anthropic: inject cache_control breakpoints at the boundary
         between stable and dynamic content.
      5. Report how many tokens are in the cacheable prefix.
    """

    def __init__(self, provider: str = "openai") -> None:
        self.provider = provider

    def optimize(
        self,
        blocks: list["Block"],
        *,
        token_budget_min: int | None = None,
    ) -> tuple[list["Block"], CacheAnalysis]:
        """
        Return (reordered_blocks, cache_analysis).

        token_budget_min: if provided, override the provider's minimum.
        """
        stable, dynamic = self._split(blocks)

        # Sort stable by original position, dynamic by original position
        stable.sort(key=lambda b: b.metadata.position or 0)
        dynamic.sort(key=lambda b: b.metadata.position or 0)

        reordered = stable + dynamic

        prefix_tokens = sum(b.token_count or 0 for b in stable)
        min_tokens = token_budget_min or CACHE_MIN_TOKENS.get(self.provider, 1024)
        eligible = prefix_tokens >= min_tokens

        if eligible:
            hit_potential = self._hit_potential(prefix_tokens, sum(b.token_count or 0 for b in reordered))
        else:
            hit_potential = "none"

        analysis = CacheAnalysis(
            provider=self.provider,
            cacheable_prefix_tokens=prefix_tokens,
            cache_eligible=eligible,
            cache_min_tokens=min_tokens,
            cache_hint_blocks=[b.id for b in stable],
            expected_hit_potential=hit_potential,
        )

        return reordered, analysis

    def inject_cache_hints(
        self,
        blocks: list["Block"],
        rendered_messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Annotate rendered messages with provider-specific cache hints.

        Only modifies the last message in the cache-stable prefix
        (the "cache breakpoint") for providers that support explicit hints.
        """
        if self.provider == "anthropic":
            return self._inject_anthropic_hints(blocks, rendered_messages)
        # OpenAI: automatic caching — no annotation needed
        # Gemini: TTL-based explicit caching — not yet implemented in v1
        return rendered_messages

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _split(
        self, blocks: list["Block"]
    ) -> tuple[list["Block"], list["Block"]]:
        stable: list["Block"] = []
        dynamic: list["Block"] = []
        for b in blocks:
            if (
                b.block_type in _CACHE_STABLE_TYPES
                or b.metadata.compression_policy == CompressionPolicy.CACHE_ANCHOR
                or b.metadata.cacheability
            ):
                stable.append(b)
            else:
                dynamic.append(b)
        return stable, dynamic

    @staticmethod
    def _hit_potential(prefix_tokens: int, total_tokens: int) -> str:
        if total_tokens == 0:
            return "none"
        ratio = prefix_tokens / total_tokens
        if ratio >= 0.6:
            return "high"
        if ratio >= 0.3:
            return "medium"
        return "low"

    def _inject_anthropic_hints(
        self,
        blocks: list["Block"],
        rendered: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        For Anthropic, mark the last stable-prefix message with
        cache_control: {type: "ephemeral"}.

        This tells Claude to cache everything up to and including that turn.
        """
        # Find the last stable block index in the rendered list
        stable_ids = {
            b.id
            for b in blocks
            if b.block_type in _CACHE_STABLE_TYPES
            or b.metadata.compression_policy == CompressionPolicy.CACHE_ANCHOR
        }

        # Walk rendered messages backwards to find last cache-stable position
        # We use a simple heuristic: mark the last "system" role message
        last_stable_idx = -1
        for i, msg in enumerate(rendered):
            if msg.get("_block_id") in stable_ids:
                last_stable_idx = i

        if last_stable_idx == -1:
            return rendered

        result = list(rendered)
        msg = dict(result[last_stable_idx])

        # Wrap content with cache_control if it's a string
        content = msg.get("content", "")
        if isinstance(content, str):
            msg["content"] = [
                {"type": "text", "text": content, "cache_control": _ANTHROPIC_CACHE_CONTROL}
            ]
        elif isinstance(content, list) and content:
            last_part = dict(content[-1])
            last_part["cache_control"] = _ANTHROPIC_CACHE_CONTROL
            msg["content"] = list(content[:-1]) + [last_part]

        result[last_stable_idx] = msg
        return result
