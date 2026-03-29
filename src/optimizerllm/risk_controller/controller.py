"""
RiskController — predict compression risk before applying any compression.

Analyzes the context structure to predict:
  - risk_level: how risky it is to compress this context ("low"/"medium"/"high"/"critical")
  - reasoning_fragility: how much the task depends on multi-hop reasoning (0.0–1.0)
  - schema_sensitivity: how much schema integrity matters (0.0–1.0)
  - target_compression_band: (min_rate, max_rate) recommended compression range

These predictions feed back into:
  - BudgetPlannerV2: tighten/loosen family minimums
  - AdaptiveFilter: adjust evidence quotas
  - SelectiveExpand: decide whether to recover aggressively

All signals are deterministic — no LLM calls.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from optimizerllm.core.block import Block
    from optimizerllm.task_synth.synth import TaskDescriptor

# Patterns for chain-of-thought markers
_COT_PATTERN = re.compile(
    r'\b(step\s+\d|therefore|because|if\s+.{1,30}\s+then)\b',
    re.IGNORECASE,
)

# Patterns for logical dependency / reasoning-chain markers within a block
_REASONING_CHAIN_PATTERN = re.compile(
    r'\b(if|unless|before|after|exactly|neither|only if|provided that|'
    r'given that|it follows|suppose|assume|let\s+\w+\s*=|'
    r'therefore|thus|hence|so\s+that|in\s+order\s+to|'
    r'first|second|third|then|finally|step\s+\d|'
    r'because|since|due\s+to|as\s+a\s+result)\b',
    re.IGNORECASE,
)

# Threshold above which sentence trimming is bypassed
_SKIP_TRIMMING_FRAGILITY_THRESHOLD = 0.4


@dataclass
class RiskAssessment:
    risk_level: str                              # "low" | "medium" | "high" | "critical"
    reasoning_fragility: float                   # 0.0–1.0
    schema_sensitivity: float                    # 0.0–1.0
    target_compression_band: tuple[float, float] # (min_rate, max_rate) e.g. (0.3, 0.7)
    signals: list[str]                           # Which risk signals fired
    skip_sentence_trimming: bool = False         # True → bypass sentence/span trimming


class RiskController:
    """
    Deterministic risk assessor for context compression.

    Usage::

        controller = RiskController()
        assessment = controller.assess(blocks, task_descriptor=task_descriptor)
    """

    def assess(
        self,
        blocks: list["Block"],
        task_descriptor: "TaskDescriptor | None" = None,
    ) -> RiskAssessment:
        """
        Analyze the block list and return a RiskAssessment.

        Risk signals:
          1. has_tool_schema: +0.3 schema_sensitivity if any TOOL_SCHEMA blocks
          2. has_multi_hop_math: +0.3 reasoning_fragility if MATH task and > 3 numeric slots
          3. has_chain_of_thought_markers: +0.2 reasoning_fragility if content has CoT markers
          4. has_long_doc: +0.1 if any single block > 2000 tokens
          5. has_many_constraints: +0.2 if > 5 CONSTRAINT slots
          6. few_documents: if RETRIEVED_DOC count > 5, reduce target max_rate
          7. task_is_code: +0.2 schema_sensitivity if CODE task
        """
        from optimizerllm.core.block import BlockType

        reasoning_fragility = 0.0
        schema_sensitivity = 0.0
        signals: list[str] = []

        # --- Signal 1: has_tool_schema ---
        has_tool_schema = any(b.block_type == BlockType.TOOL_SCHEMA for b in blocks)
        if has_tool_schema:
            schema_sensitivity += 0.3
            signals.append("has_tool_schema")

        # --- Signal 2: has_multi_hop_math ---
        task_type_value = (
            task_descriptor.task_type.value if task_descriptor is not None else ""
        )
        if task_type_value == "math":
            # Count numeric-looking tokens across all blocks
            numeric_count = 0
            for b in blocks:
                numeric_count += len(re.findall(r'\b\d+(?:[.,]\d+)?\b', b.content))
            if numeric_count > 3:
                reasoning_fragility += 0.3
                signals.append("has_multi_hop_math")

        # --- Signal 3: has_chain_of_thought_markers ---
        all_content = " ".join(b.content for b in blocks)
        cot_matches = _COT_PATTERN.findall(all_content)
        if cot_matches:
            reasoning_fragility += 0.2
            signals.append("has_chain_of_thought_markers")

        # --- Signal 3b: dense dependency markers (BBH-style ordering/logic) ---
        dep_matches = _REASONING_CHAIN_PATTERN.findall(all_content)
        if len(dep_matches) >= 6:
            # Many logical connectors → strong reasoning chain, boost fragility
            reasoning_fragility += min(0.3, len(dep_matches) * 0.03)
            signals.append(f"dense_dependency_markers:{len(dep_matches)}")

        # --- Signal 4: has_long_doc ---
        for b in blocks:
            token_count = b.token_count if b.token_count is not None else len(b.content) // 4
            if token_count > 2000:
                reasoning_fragility += 0.1
                signals.append("has_long_doc")
                break  # Only fire once

        # --- Signal 5: has_many_constraints ---
        # Count constraint-like patterns across all content
        constraint_count = len(re.findall(
            r'\b(must|cannot|shall not|required|forbidden|prohibited|not allowed)\b',
            all_content,
            re.IGNORECASE,
        ))
        if constraint_count > 5:
            reasoning_fragility += 0.2
            signals.append("has_many_constraints")

        # --- Signal 6: few_documents (many docs → reduce max compression) ---
        retrieved_doc_count = sum(
            1 for b in blocks if b.block_type == BlockType.RETRIEVED_DOC
        )
        many_docs = retrieved_doc_count > 5

        # --- Signal 7: task_is_code ---
        if task_type_value == "code":
            schema_sensitivity += 0.2
            signals.append("task_is_code")

        # --- Clamp values ---
        reasoning_fragility = min(1.0, reasoning_fragility)
        schema_sensitivity = min(1.0, schema_sensitivity)

        # --- Compute overall risk score ---
        risk_score = (reasoning_fragility + schema_sensitivity) / 2.0

        # --- Determine risk level and band ---
        if risk_score <= 0.2:
            risk_level = "low"
            band: tuple[float, float] = (0.3, 0.7)
        elif risk_score <= 0.4:
            risk_level = "medium"
            band = (0.4, 0.6)
        elif risk_score <= 0.6:
            risk_level = "high"
            band = (0.5, 0.7)
        else:
            risk_level = "critical"
            band = (0.6, 0.8)

        # Tighten max_rate when there are many docs (don't compress too hard)
        if many_docs:
            min_rate, max_rate = band
            band = (min_rate, min(max_rate, 0.5))

        skip_sentence_trimming = reasoning_fragility >= _SKIP_TRIMMING_FRAGILITY_THRESHOLD

        return RiskAssessment(
            risk_level=risk_level,
            reasoning_fragility=round(reasoning_fragility, 3),
            schema_sensitivity=round(schema_sensitivity, 3),
            target_compression_band=band,
            signals=signals,
            skip_sentence_trimming=skip_sentence_trimming,
        )
