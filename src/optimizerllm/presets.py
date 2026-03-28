"""
Pre-tuned ContextOptimizer presets for common use cases.

Usage::

    from optimizerllm.presets import preset

    optimizer = preset("rag", provider="openai", model="gpt-4o")
    result = optimizer.optimize(documents=[...], messages=[...])

Or use the named factory functions directly::

    from optimizerllm.presets import rag_optimizer, chat_optimizer

    optimizer = rag_optimizer(provider="anthropic", model="claude-3-5-sonnet")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from optimizerllm.core.block import CompressionMode


@dataclass(frozen=True)
class PresetConfig:
    """All tunable parameters for a single preset."""

    name: str
    description: str
    mode: CompressionMode
    selector_threshold: float
    compactor_recency_tail: int
    section_level: bool
    sentence_level: bool
    section_threshold: float
    sentence_threshold: float
    output_budget: int


# ---------------------------------------------------------------------------
# Preset definitions
# ---------------------------------------------------------------------------

_PRESETS: dict[str, PresetConfig] = {
    # ── RAG ─────────────────────────────────────────────────────────────────
    # Retrieved docs are noisy. Be aggressive about dropping irrelevant chunks
    # and trimming inside the ones we keep. Short recency tail — history
    # usually matters less than retrieved evidence in QA.
    "rag": PresetConfig(
        name="rag",
        description="RAG / document QA: aggressive doc filtering and section trimming",
        mode=CompressionMode.CONSERVATIVE,
        selector_threshold=0.14,
        compactor_recency_tail=2,
        section_level=True,
        sentence_level=True,
        section_threshold=0.10,
        sentence_threshold=0.08,
        output_budget=2048,
    ),

    # ── CHAT ────────────────────────────────────────────────────────────────
    # Multi-turn conversation. History preservation matters more.
    # Sentence trimming is off — chopping mid-conversation context is risky.
    # Longer recency tail, lower threshold so more history survives.
    "chat": PresetConfig(
        name="chat",
        description="Multi-turn chat: history-aware, light compression",
        mode=CompressionMode.CONSERVATIVE,
        selector_threshold=0.08,
        compactor_recency_tail=6,
        section_level=False,
        sentence_level=False,
        section_threshold=0.10,
        sentence_threshold=0.08,
        output_budget=2048,
    ),

    # ── AGENT ────────────────────────────────────────────────────────────────
    # Tool-heavy agentic loops. Tool schemas must be preserved intact.
    # Section/sentence trimming off — structured outputs can be fragile.
    # Focus is on cache shaping: stable prefix = tools + instructions.
    "agent": PresetConfig(
        name="agent",
        description="Agentic / tool-use: schema-safe, cache-optimised prefix",
        mode=CompressionMode.CONSERVATIVE,
        selector_threshold=0.12,
        compactor_recency_tail=4,
        section_level=False,
        sentence_level=False,
        section_threshold=0.10,
        sentence_threshold=0.08,
        output_budget=4096,
    ),

    # ── LONG CONTEXT QA ──────────────────────────────────────────────────────
    # Very long documents (books, reports, transcripts). Must trim hard inside
    # each document — can't just drop blocks because the answer might be anywhere.
    # Aggressive section + sentence trimming, low sentence threshold.
    "long_context_qa": PresetConfig(
        name="long_context_qa",
        description="Long-context QA: deep section and sentence trimming",
        mode=CompressionMode.AGGRESSIVE,
        selector_threshold=0.10,
        compactor_recency_tail=2,
        section_level=True,
        sentence_level=True,
        section_threshold=0.08,
        sentence_threshold=0.06,
        output_budget=1024,
    ),

    # ── SUMMARIZATION ────────────────────────────────────────────────────────
    # The full document content matters — don't drop sections or sentences.
    # Only safe lossless normalization + dedup. Budget is generous.
    "summarization": PresetConfig(
        name="summarization",
        description="Summarization / extraction: lossless, preserve full docs",
        mode=CompressionMode.LOSSLESS,
        selector_threshold=0.0,
        compactor_recency_tail=3,
        section_level=False,
        sentence_level=False,
        section_threshold=0.10,
        sentence_threshold=0.08,
        output_budget=4096,
    ),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preset(
    name: str,
    *,
    provider: str = "openai",
    model: str = "gpt-4o",
    max_input_tokens: int = 128_000,
    output_format: str = "openai",
    **overrides: Any,
) -> "ContextOptimizer":
    """
    Build a ContextOptimizer from a named preset.

    Args:
        name: One of "rag", "chat", "agent", "long_context_qa", "summarization".
        provider: LLM provider ("openai", "anthropic", "gemini").
        model: Model name for token counting and cost estimation.
        max_input_tokens: Token budget for the provider/model.
        output_format: "openai", "anthropic", or "text".
        **overrides: Any ContextOptimizer param to override from the preset.

    Returns:
        A configured ContextOptimizer instance.
    """
    from optimizerllm.optimizer import ContextOptimizer

    cfg = _get(name)

    params: dict[str, Any] = dict(
        provider=provider,
        model=model,
        max_input_tokens=max_input_tokens,
        output_budget=cfg.output_budget,
        mode=cfg.mode,
        output_format=output_format,
        selector_threshold=cfg.selector_threshold,
        compactor_recency_tail=cfg.compactor_recency_tail,
        section_level=cfg.section_level,
        sentence_level=cfg.sentence_level,
        section_threshold=cfg.section_threshold,
        sentence_threshold=cfg.sentence_threshold,
    )
    params.update(overrides)
    return ContextOptimizer(**params)


def list_presets() -> list[dict[str, str]]:
    """Return a summary of all available presets."""
    return [
        {"name": cfg.name, "mode": cfg.mode.value, "description": cfg.description}
        for cfg in _PRESETS.values()
    ]


def _get(name: str) -> PresetConfig:
    if name not in _PRESETS:
        available = ", ".join(_PRESETS)
        raise ValueError(f"Unknown preset {name!r}. Available: {available}")
    return _PRESETS[name]


# ---------------------------------------------------------------------------
# Convenience factory functions
# ---------------------------------------------------------------------------

def rag_optimizer(**kwargs: Any) -> "ContextOptimizer":
    """ContextOptimizer tuned for RAG / document QA."""
    return preset("rag", **kwargs)


def chat_optimizer(**kwargs: Any) -> "ContextOptimizer":
    """ContextOptimizer tuned for multi-turn chat."""
    return preset("chat", **kwargs)


def agent_optimizer(**kwargs: Any) -> "ContextOptimizer":
    """ContextOptimizer tuned for agentic / tool-use loops."""
    return preset("agent", **kwargs)


def long_context_qa_optimizer(**kwargs: Any) -> "ContextOptimizer":
    """ContextOptimizer tuned for long-context document QA."""
    return preset("long_context_qa", **kwargs)


def summarization_optimizer(**kwargs: Any) -> "ContextOptimizer":
    """ContextOptimizer tuned for summarization tasks."""
    return preset("summarization", **kwargs)
