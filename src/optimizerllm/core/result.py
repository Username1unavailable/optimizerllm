"""OptimizationResult and TelemetryReport — what every optimize() call returns."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from optimizerllm.core.block import Block, CompressionMode


@dataclass
class RemovalRecord:
    """Records why a block was removed or truncated."""

    block_id: str
    block_type: str
    token_count: int
    reason: str                     # "budget_exceeded" | "deduplication" | "low_score" | ...
    score: float | None = None
    priority: int = 0


@dataclass
class TelemetryReport:
    """Full audit trail for one optimization run."""

    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)

    # Token counts
    tokens_before: int = 0
    tokens_after: int = 0
    reduction_pct: float = 0.0
    cacheable_prefix_tokens: int = 0

    # Block counts
    blocks_before: int = 0
    blocks_after: int = 0
    blocks_removed: list[RemovalRecord] = field(default_factory=list)

    # Cost (USD) — None if provider pricing is unknown
    cost_before_usd: float | None = None
    cost_after_usd: float | None = None

    # Settings used
    compression_mode: str = CompressionMode.CONSERVATIVE
    provider: str = "unknown"
    model: str = "unknown"

    # Risk assessment
    risk_level: str = "low"        # "low" | "medium" | "high"
    risk_reasons: list[str] = field(default_factory=list)

    # Extension bag — populated by optional pipeline modules
    extra: dict[str, Any] = field(default_factory=dict)

    def finalize(self) -> None:
        """Compute derived fields after all blocks are processed."""
        if self.tokens_before > 0:
            self.reduction_pct = round(
                (1 - self.tokens_after / self.tokens_before) * 100, 2
            )
        self._assess_risk()

    def _assess_risk(self) -> None:
        reasons: list[str] = []
        if self.reduction_pct > 60:
            reasons.append("reduction_over_60pct")
        if self.blocks_removed and any(
            r.reason == "budget_exceeded" for r in self.blocks_removed
        ):
            reasons.append("forced_budget_drops")
        if self.tokens_after == 0:
            reasons.append("empty_output")
        self.risk_reasons = reasons
        if len(reasons) >= 2:
            self.risk_level = "high"
        elif len(reasons) == 1:
            self.risk_level = "medium"
        else:
            self.risk_level = "low"

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "reduction_pct": self.reduction_pct,
            "cacheable_prefix_tokens": self.cacheable_prefix_tokens,
            "blocks_before": self.blocks_before,
            "blocks_after": self.blocks_after,
            "blocks_removed": [
                {
                    "block_id": r.block_id,
                    "block_type": r.block_type,
                    "token_count": r.token_count,
                    "reason": r.reason,
                    "score": r.score,
                    "priority": r.priority,
                }
                for r in self.blocks_removed
            ],
            "cost_before_usd": self.cost_before_usd,
            "cost_after_usd": self.cost_after_usd,
            "compression_mode": self.compression_mode,
            "provider": self.provider,
            "model": self.model,
            "risk_level": self.risk_level,
            "risk_reasons": self.risk_reasons,
            "extra": self.extra,
        }


@dataclass
class OptimizationResult:
    """The output of a single ContextOptimizer.optimize() call."""

    blocks: list[Block]
    rendered: Any                   # Provider-specific rendered form
    telemetry: TelemetryReport
    provider: str = "unknown"
    model: str = "unknown"

    @property
    def tokens_before(self) -> int:
        return self.telemetry.tokens_before

    @property
    def tokens_after(self) -> int:
        return self.telemetry.tokens_after

    @property
    def tokens_saved(self) -> int:
        return self.telemetry.tokens_before - self.telemetry.tokens_after

    @property
    def reduction_pct(self) -> float:
        return self.telemetry.reduction_pct

    def summary(self) -> str:
        t = self.telemetry
        lines = [
            f"Run {t.run_id[:8]}",
            f"  Tokens: {t.tokens_before:,} → {t.tokens_after:,} ({t.reduction_pct:.1f}% reduction)",
            f"  Blocks: {t.blocks_before} → {t.blocks_after} ({len(t.blocks_removed)} removed)",
            f"  Cacheable prefix: {t.cacheable_prefix_tokens:,} tokens",
            f"  Risk: {t.risk_level}",
        ]
        if t.cost_before_usd is not None:
            savings = (t.cost_before_usd or 0) - (t.cost_after_usd or 0)
            lines.append(
                f"  Cost: ${t.cost_before_usd:.4f} → ${t.cost_after_usd:.4f}"
                f" (save ${savings:.4f})"
            )
        return "\n".join(lines)
