"""
TargetPredictor — calibrated per-input compression target.

Phase 2 module. Sits after RiskController (step 5a) and enriches its
global risk assessment with task-type-aware trimming permissions and
evidence redundancy analysis.

Key insight: RiskController fires skip_sentence_trimming based on connector
density. That is correct for reasoning tasks (BBH, math), but wrong for
retrieval / summarization tasks where source documents naturally contain
logical text. The TargetPredictor splits this single global flag into
per-task-type permissions.

Secondary: for evidence-heavy tasks with overlapping retrieved documents,
it computes a redundancy estimate that widens the compression band, enabling
the AdaptiveEvidenceFilter and SpanCritic to be more aggressive.

Outputs a TargetPrediction consumed by:
  - optimizer.py step 9: overrides section_level / sentence_level / span_level
  - optimizer.py step 7a: adjusts per-family evidence quotas
  - GranularSelector: passes span_level override per call
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from optimizerllm.core.block import Block
    from optimizerllm.risk_controller.controller import RiskAssessment
    from optimizerllm.task_synth.synth import TaskDescriptor

from optimizerllm.compression.selector import _tokenize


# Reasoning-intensive tasks: sentence/section trimming must obey fragility threshold
_REASONING_TASKS = frozenset({"math", "general"})

# Retrieval / distillation tasks: sentence trimming is always safe
# (we are condensing retrieved content, not following a logical chain)
_SAFE_TRIM_TASKS = frozenset({"summarization", "rag", "extraction", "classification"})

# Tasks where span-level (clause-level) trimming is also beneficial
_SPAN_TRIM_TASKS = frozenset({"summarization", "rag", "extraction"})

# Fragility thresholds — split into sentence and section independently
_SENTENCE_TRIM_MAX_FRAGILITY = 0.40   # same as RiskController baseline
_SECTION_TRIM_MAX_FRAGILITY  = 0.50   # tighter than before (was 0.60); fixes BBH

# Minimum token set size for Jaccard redundancy computation
_MIN_TOKENS_REDUNDANCY = 8

# Characters to sample from each block for redundancy analysis (speed cap)
_REDUNDANCY_CONTENT_CAP = 2500


@dataclass
class TargetPrediction:
    """Per-input compression target and granularity permissions."""

    # Compression band (keep rates): (min_keep, max_keep)
    # e.g. (0.70, 0.85) means "keep 70–85% of tokens"
    compression_band: tuple[float, float]

    # Granularity permissions — override optimizer.py's global flags
    allow_section_trim: bool = True
    allow_sentence_trim: bool = True
    allow_span_trim: bool = False    # clause-level trimming inside sentences

    # Families where SpanCritic / AdaptiveFilter should apply extra pressure
    aggressive_families: set[str] = field(default_factory=set)

    # Per-family budget overrides (family → fraction of total budget).
    # Empty dict → use BudgetPlannerV2 soft-target defaults unchanged.
    per_family_budgets: dict[str, float] = field(default_factory=dict)

    # Per-block redundancy estimates (block_id → 0.0–1.0).
    # Higher = this block is more redundant given the other evidence blocks.
    block_redundancy: dict[str, float] = field(default_factory=dict)

    # Sentence-level budget fraction: SpanCritic budget mode keeps top
    # ceil(n × frac) sentences per block.
    # None = threshold mode (keep every sentence above sentence_threshold).
    # Phase 2C values: 0.55 (summarization/rag), 0.62 (extraction), 0.70 (tool_use)
    sentence_keep_fraction: float | None = None

    # Token-level keep fraction: within each surviving sentence, keep this
    # fraction of tokens (e.g. 0.55 = keep 55% of tokens per sentence).
    # Force-keep sentences (fragility >= 0.45, slot sentences) always retain
    # all their tokens regardless of this value.
    # None = skip token-level trimming entirely (conservative / lossless).
    token_keep_fraction: float | None = None

    # Human-readable log of which signals fired
    rationale: list[str] = field(default_factory=list)


class TargetPredictor:
    """
    Deterministic compression-target predictor.

    Usage::

        predictor = TargetPredictor()
        prediction = predictor.predict(
            blocks,
            risk_assessment=risk_assessment,
            task_descriptor=task_descriptor,
        )
    """

    def predict(
        self,
        blocks: list["Block"],
        *,
        risk_assessment: "RiskAssessment | None" = None,
        task_descriptor: "TaskDescriptor | None" = None,
    ) -> TargetPrediction:
        from optimizerllm.core.block import BlockType

        task_type = (
            task_descriptor.task_type.value if task_descriptor is not None else "general"
        )
        reasoning_fragility = (
            risk_assessment.reasoning_fragility if risk_assessment is not None else 0.0
        )
        schema_sensitivity = (
            risk_assessment.schema_sensitivity if risk_assessment is not None else 0.0
        )
        base_band: tuple[float, float] = (
            risk_assessment.target_compression_band
            if risk_assessment is not None
            else (0.30, 0.70)
        )
        retrieval_noise = (
            task_descriptor.retrieval_noise_profile
            if task_descriptor is not None
            else "low"
        )

        rationale: list[str] = []

        # ------------------------------------------------------------------
        # 1. Trimming permissions: task-type × fragility
        # ------------------------------------------------------------------
        allow_section_trim, allow_sentence_trim, allow_span_trim = (
            self._trimming_permissions(task_type, reasoning_fragility, rationale)
        )

        # ------------------------------------------------------------------
        # 2. Evidence redundancy analysis
        # ------------------------------------------------------------------
        evidence_blocks = [
            b for b in blocks if b.block_type == BlockType.RETRIEVED_DOC
        ]
        block_redundancy = self._estimate_evidence_redundancy(evidence_blocks)
        avg_redundancy = (
            sum(block_redundancy.values()) / len(block_redundancy)
            if block_redundancy
            else 0.0
        )

        # ------------------------------------------------------------------
        # 3. Compression band refinement
        # ------------------------------------------------------------------
        min_rate, max_rate = base_band

        if avg_redundancy > 0.50:
            # High inter-doc redundancy → can drop more aggressively
            max_rate = min(max_rate, 0.72)
            rationale.append(
                f"high_evidence_redundancy:{avg_redundancy:.2f}:tighten_max_rate"
            )
        elif avg_redundancy > 0.30:
            rationale.append(f"moderate_evidence_redundancy:{avg_redundancy:.2f}")

        if task_type == "summarization":
            min_rate = max(min_rate, 0.60)
            rationale.append("summarization:band_conservative_min")

        if schema_sensitivity >= 0.50:
            min_rate = max(min_rate, 0.65)
            rationale.append(
                f"schema_sensitivity:{schema_sensitivity:.2f}:min_rate_raised"
            )

        # ------------------------------------------------------------------
        # 4. Aggressive families
        # ------------------------------------------------------------------
        aggressive_families: set[str] = set()

        if retrieval_noise == "high" or len(evidence_blocks) >= 5:
            aggressive_families.add("evidence")
            rationale.append(
                f"retrieval_noise:{retrieval_noise}:evidence_aggressive"
            )

        if task_type in _SAFE_TRIM_TASKS | {"tool_use"}:
            aggressive_families.add("background")
            rationale.append(f"{task_type}:background_aggressive")

        if task_type in ("math", "code") and reasoning_fragility < 0.30:
            aggressive_families.update({"background", "memory"})

        # ------------------------------------------------------------------
        # 5. Per-family budget overrides
        # ------------------------------------------------------------------
        per_family_budgets = self._family_budgets(task_type)
        if per_family_budgets:
            rationale.append(f"{task_type}:custom_family_budgets")

        # ------------------------------------------------------------------
        # 6. Sentence-level budget fraction
        #    Controls what fraction of sentences to keep per block in budget mode.
        #    Now always active for safe tasks (fragility gate removed).
        #    Fragility SCALES the fraction rather than disabling it entirely:
        #      low frag  (<0.40): aggressive — meeting transcripts are safe to trim
        #      mod frag  (0.40–0.60): moderate — some reasoning present
        #      high frag (>=0.60): conservative — dense logical connectors
        # ------------------------------------------------------------------
        sentence_keep_fraction: float | None = None
        if allow_sentence_trim:
            if task_type in ("summarization", "rag"):
                if reasoning_fragility < 0.40:
                    sentence_keep_fraction = 0.35
                    rationale.append(f"sentence_budget:0.35:{task_type}:low_frag")
                elif reasoning_fragility < 0.60:
                    sentence_keep_fraction = 0.48
                    rationale.append(f"sentence_budget:0.48:{task_type}:mod_frag")
                else:
                    sentence_keep_fraction = 0.62
                    rationale.append(f"sentence_budget:0.62:{task_type}:high_frag")
            elif task_type == "extraction":
                sentence_keep_fraction = 0.42 if reasoning_fragility < 0.40 else 0.58
                rationale.append(f"sentence_budget:{sentence_keep_fraction:.2f}:extraction")
            elif task_type == "tool_use":
                sentence_keep_fraction = 0.55
                rationale.append("sentence_budget:0.55:tool_use")
            elif task_type in ("math", "code") and reasoning_fragility < 0.30:
                sentence_keep_fraction = 0.75
                rationale.append(f"sentence_budget:0.75:{task_type}:low_fragility")
            # classification, general, chat → None (threshold mode)

        # ------------------------------------------------------------------
        # 7. Token-level keep fraction
        #    Runs AFTER sentence-level. Combined math:
        #      sentence_stage × token_stage ≈ total compression floor.
        #    Now uses multiplicative sentence-importance weighting in
        #    GranularSelector._trim_tokens (low-importance sentences can go
        #    as low as base × 0.20, high-importance up to base × 1.80).
        #    Also active for tool_use and math/code at low fragility.
        #
        #    Base fractions (scaled up by fragility for safe tasks):
        #      low redundancy    → 0.50  (base)
        #      moderate          → 0.45
        #      high redundancy   → 0.40
        #    Fragility scale-up (prevents over-trimming connector-dense docs):
        #      frag >= 0.60      → +0.15 to base
        #      frag 0.40–0.60    → +0.08 to base
        # ------------------------------------------------------------------
        token_keep_fraction: float | None = None
        if allow_sentence_trim and task_type in _SAFE_TRIM_TASKS:
            # Base token fraction from redundancy
            if avg_redundancy > 0.30:
                base_tok = 0.40
                rationale.append(f"token_budget:0.40:high_redundancy:{avg_redundancy:.2f}")
            elif avg_redundancy > 0.10:
                base_tok = 0.45
                rationale.append(f"token_budget:0.45:moderate_redundancy:{avg_redundancy:.2f}")
            else:
                base_tok = 0.50
                rationale.append(f"token_budget:0.50:{task_type}:low_redundancy")
            # Scale up conservatively if fragility is high
            if reasoning_fragility >= 0.60:
                base_tok = min(0.75, base_tok + 0.15)
                rationale.append(f"token_budget:frag_scale_up_0.15:{base_tok:.2f}")
            elif reasoning_fragility >= 0.40:
                base_tok = min(0.65, base_tok + 0.08)
                rationale.append(f"token_budget:frag_scale_up_0.08:{base_tok:.2f}")
            token_keep_fraction = base_tok
        elif task_type == "tool_use":
            # Mild token compression for tool_use (schemas are protected by slots)
            token_keep_fraction = 0.65
            rationale.append("token_budget:0.65:tool_use")
        elif task_type in ("math", "code") and reasoning_fragility < 0.30:
            # Very light token compression for low-fragility reasoning tasks
            token_keep_fraction = 0.72
            rationale.append(f"token_budget:0.72:{task_type}:low_fragility")

        return TargetPrediction(
            compression_band=(round(min_rate, 3), round(max_rate, 3)),
            allow_section_trim=allow_section_trim,
            allow_sentence_trim=allow_sentence_trim,
            allow_span_trim=allow_span_trim,
            aggressive_families=aggressive_families,
            per_family_budgets=per_family_budgets,
            block_redundancy=block_redundancy,
            sentence_keep_fraction=sentence_keep_fraction,
            token_keep_fraction=token_keep_fraction,
            rationale=rationale,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _trimming_permissions(
        task_type: str,
        reasoning_fragility: float,
        rationale: list[str],
    ) -> tuple[bool, bool, bool]:
        """Return (allow_section_trim, allow_sentence_trim, allow_span_trim)."""

        if task_type in _REASONING_TASKS:
            if reasoning_fragility >= _SECTION_TRIM_MAX_FRAGILITY:
                # Fragility is high enough to block even section-level trimming.
                # Note: threshold is 0.50 (was 0.60) — fixes BBH quality gap.
                rationale.append(
                    f"reasoning_task:{task_type}:fragility={reasoning_fragility:.2f}"
                    f":section+sentence_trim_disabled"
                )
                return False, False, False
            elif reasoning_fragility >= _SENTENCE_TRIM_MAX_FRAGILITY:
                rationale.append(
                    f"reasoning_task:{task_type}:fragility={reasoning_fragility:.2f}"
                    f":sentence_trim_disabled"
                )
                return True, False, False
            else:
                rationale.append(f"reasoning_task:{task_type}:low_fragility:all_trim_allowed")
                return True, True, False

        elif task_type in _SAFE_TRIM_TASKS:
            # For retrieval / summarization tasks, NEVER gate trimming based on
            # fragility. Meeting transcripts and RAG docs naturally contain
            # connector words (because/therefore/since) that inflate the fragility
            # score, but these are background connectors in retrieved content —
            # safe to trim. Fragility adjusts HOW aggressive the fractions are
            # (see sections 6/7 of predict()), not WHETHER we trim at all.
            rationale.append(
                f"safe_trim_task:{task_type}:fragility={reasoning_fragility:.2f}"
                f":always_trim_allowed"
            )
            return True, True, False

        elif task_type == "tool_use":
            # Tool use: trim background, but protect tool schemas — no span trim
            rationale.append("tool_use:section+sentence_allowed:no_span_trim")
            return True, True, False

        else:
            # code, chat, general fallback — respect fragility thresholds
            if reasoning_fragility >= _SECTION_TRIM_MAX_FRAGILITY:
                rationale.append(
                    f"fallback:{task_type}:fragility={reasoning_fragility:.2f}"
                    f":section+sentence_disabled"
                )
                return False, False, False
            elif reasoning_fragility >= _SENTENCE_TRIM_MAX_FRAGILITY:
                rationale.append(
                    f"fallback:{task_type}:fragility={reasoning_fragility:.2f}"
                    f":sentence_disabled"
                )
                return True, False, False
            else:
                rationale.append(f"fallback:{task_type}:low_fragility:all_trim_allowed")
                return True, True, False

    @staticmethod
    def _estimate_evidence_redundancy(
        evidence_blocks: list["Block"],
    ) -> dict[str, float]:
        """
        Per-block redundancy relative to all other evidence blocks.

        Returns {block_id: jaccard_max_similarity} where higher means
        the block is more redundant given the other blocks.

        O(n²) over token sets — fast enough for n ≤ 20 blocks.
        """
        if len(evidence_blocks) < 2:
            return {}

        token_sets: list[set[str]] = []
        for b in evidence_blocks:
            sample = b.content[:_REDUNDANCY_CONTENT_CAP]
            toks = set(_tokenize(sample))
            token_sets.append(toks)

        n = len(evidence_blocks)
        result: dict[str, float] = {}

        for i, block in enumerate(evidence_blocks):
            if not token_sets[i]:
                result[block.id] = 0.0
                continue

            max_sim = 0.0
            for j in range(n):
                if i == j:
                    continue
                if not token_sets[j]:
                    continue
                union = token_sets[i] | token_sets[j]
                if not union:
                    continue
                sim = len(token_sets[i] & token_sets[j]) / len(union)
                max_sim = max(max_sim, sim)

            result[block.id] = round(max_sim, 3)

        return result

    @staticmethod
    def _family_budgets(task_type: str) -> dict[str, float]:
        """
        Per-task-type soft budget overrides for BudgetPlannerV2.

        Returns empty dict for tasks where defaults are fine.
        """
        if task_type == "summarization":
            return {
                "schema_core": 0.10,
                "evidence": 0.55,
                "constraints": 0.10,
                "memory": 0.08,
                "background": 0.02,
            }
        if task_type == "rag":
            return {
                "schema_core": 0.10,
                "evidence": 0.45,
                "constraints": 0.10,
                "memory": 0.10,
                "background": 0.04,
            }
        if task_type == "tool_use":
            return {
                "schema_core": 0.25,
                "evidence": 0.25,
                "constraints": 0.15,
                "memory": 0.15,
                "background": 0.03,
            }
        return {}
