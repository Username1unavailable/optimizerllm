"""Token accounting layer — provider-aware token counting and cost estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from optimizerllm.core.block import Block


# ---------------------------------------------------------------------------
# Provider token-count scaling factors relative to cl100k_base (tiktoken)
# ---------------------------------------------------------------------------
# These are empirical approximations. Anthropic's tokenizer tends to produce
# slightly different counts; Gemini's SentencePiece tokenizer also differs.
_PROVIDER_SCALE: dict[str, float] = {
    "openai": 1.00,
    "anthropic": 1.05,  # Conservative — slightly longer in practice
    "gemini": 1.10,
    "unknown": 1.00,
}

# Input cost per 1 M tokens (USD) — approximate mid-2025 values
# Format: {"model_prefix": (input_per_1M, output_per_1M, cache_read_per_1M)}
_COST_TABLE: dict[str, tuple[float, float, float]] = {
    "gpt-4o": (5.00, 15.00, 0.50),
    "gpt-4o-mini": (0.15, 0.60, 0.075),
    "gpt-4-turbo": (10.00, 30.00, 0.0),
    "gpt-3.5-turbo": (0.50, 1.50, 0.0),
    "claude-3-5-sonnet": (3.00, 15.00, 0.30),
    "claude-3-5-haiku": (0.80, 4.00, 0.08),
    "claude-3-opus": (15.00, 75.00, 1.50),
    "claude-sonnet-4": (3.00, 15.00, 0.30),
    "claude-haiku-4": (0.80, 4.00, 0.08),
    "gemini-1.5-pro": (3.50, 10.50, 0.0),
    "gemini-1.5-flash": (0.35, 1.05, 0.0),
    "gemini-2.0-flash": (0.10, 0.40, 0.0),
}

# Minimum tokens required for provider caching to activate
CACHE_MIN_TOKENS: dict[str, int] = {
    "openai": 1024,
    "anthropic": 1024,
    "gemini": 32768,
    "unknown": 1024,
}


@dataclass
class TokenBudget:
    """Tracks budget allocation for a single optimization run."""

    max_input_tokens: int
    output_reserved: int
    provider: str
    model: str

    @property
    def available_input_tokens(self) -> int:
        return self.max_input_tokens - self.output_reserved

    @property
    def cache_min_tokens(self) -> int:
        return CACHE_MIN_TOKENS.get(self.provider, 1024)


@dataclass
class CostEstimate:
    """Per-run cost breakdown."""

    input_tokens: int
    output_tokens_reserved: int
    cacheable_prefix_tokens: int
    provider: str
    model: str
    input_cost_usd: float
    cache_read_cost_usd: float
    total_usd: float

    def without_cache(self) -> float:
        """Baseline cost if no caching applied."""
        return self.input_cost_usd + self.cache_read_cost_usd


class TokenCounter:
    """
    Estimates token counts per provider using tiktoken as the base tokenizer.

    OpenAI: exact count via cl100k_base / o200k_base.
    Anthropic / Gemini: tiktoken approximation with a scaling factor.
    """

    def __init__(self) -> None:
        self._enc_cache: dict[str, object] = {}

    def _encoder(self, model: str, provider: str) -> object:
        """Return (and cache) the tiktoken encoder for this model."""
        import tiktoken

        key = f"{provider}:{model}"
        if key not in self._enc_cache:
            try:
                enc = tiktoken.encoding_for_model(model)
            except KeyError:
                # Fall back to cl100k_base for unknown models
                enc = tiktoken.get_encoding("cl100k_base")
            self._enc_cache[key] = enc
        return self._enc_cache[key]

    def count(self, text: str, provider: str = "openai", model: str = "gpt-4o") -> int:
        """Return estimated token count for a single string."""
        if not text:
            return 0
        enc = self._encoder(model, provider)
        raw = len(enc.encode(text))  # type: ignore[arg-type]
        scale = _PROVIDER_SCALE.get(provider, 1.0)
        return max(1, round(raw * scale))

    def count_blocks(
        self,
        blocks: list["Block"],
        provider: str = "openai",
        model: str = "gpt-4o",
    ) -> list["Block"]:
        """Fill in token_count on every block, return them."""
        result = []
        for block in blocks:
            tc = self.count(block.content, provider=provider, model=model)
            result.append(block.clone_with(token_count=tc))
        return result

    def total_tokens(self, blocks: list["Block"]) -> int:
        return sum(b.token_count or 0 for b in blocks)

    def estimate_cost(
        self,
        input_tokens: int,
        *,
        output_tokens: int = 0,
        cacheable_prefix_tokens: int = 0,
        provider: str = "openai",
        model: str = "gpt-4o",
    ) -> CostEstimate:
        """
        Estimate USD cost for an API call.

        cacheable_prefix_tokens: tokens in the cache-hit prefix (billed at reduced rate).
        """
        per_1m = self._lookup_cost(model)
        if per_1m is None:
            return CostEstimate(
                input_tokens=input_tokens,
                output_tokens_reserved=output_tokens,
                cacheable_prefix_tokens=cacheable_prefix_tokens,
                provider=provider,
                model=model,
                input_cost_usd=0.0,
                cache_read_cost_usd=0.0,
                total_usd=0.0,
            )

        input_usd_pm, _output_usd_pm, cache_read_usd_pm = per_1m

        non_cached = max(0, input_tokens - cacheable_prefix_tokens)
        input_cost = (non_cached / 1_000_000) * input_usd_pm
        cache_read_cost = (cacheable_prefix_tokens / 1_000_000) * cache_read_usd_pm
        total = input_cost + cache_read_cost

        return CostEstimate(
            input_tokens=input_tokens,
            output_tokens_reserved=output_tokens,
            cacheable_prefix_tokens=cacheable_prefix_tokens,
            provider=provider,
            model=model,
            input_cost_usd=input_cost,
            cache_read_cost_usd=cache_read_cost,
            total_usd=total,
        )

    def _lookup_cost(
        self, model: str
    ) -> tuple[float, float, float] | None:
        model_lower = model.lower()
        for prefix, costs in _COST_TABLE.items():
            if model_lower.startswith(prefix):
                return costs
        return None
