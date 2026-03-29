"""
AdaptiveEvidenceFilter — evidence-quota-based block filtering.

Replaces a single global relevance threshold with per-block-family quotas.
Each block family (schema_core, evidence, constraints, memory, background)
has a minimum keep count and a maximum budget share.

This prevents over-dropping in one family while another family wastes budget.

Block families:
  - schema_core:  system_prompt, tool_schema (always keep, min_keep=None/all)
  - evidence:     retrieved_doc, few_shot_example (keep top-K by score)
  - constraints:  structured_data (keep all protected, drop low-score)
  - memory:       conversation_history (keep top-K by recency × score)
  - background:   assistant_message, user_message (keep by score threshold)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from optimizerllm.core.block import Block
    from optimizerllm.task_synth.synth import TaskDescriptor

# Map block type values to families
_FAMILY_MAP: dict[str, str] = {
    "system_prompt": "schema_core",
    "tool_schema": "schema_core",
    "retrieved_doc": "evidence",
    "few_shot_example": "evidence",
    "structured_data": "constraints",
    "conversation_history": "memory",
    "user_message": "background",
    "assistant_message": "background",
    "tool_result": "constraints",
}

_CONSTRAINTS_MIN_SCORE = 0.05


class AdaptiveEvidenceFilter:
    """
    Evidence-quota-based block filter.

    Instead of a single global relevance threshold, each block family has
    its own quota. This prevents over-dropping in one family while another
    family wastes budget.
    """

    def __init__(
        self,
        evidence_quota: int = 10,
        memory_quota: int = 5,
        background_threshold: float = 0.10,
    ) -> None:
        self.evidence_quota = evidence_quota
        self.memory_quota = memory_quota
        self.background_threshold = background_threshold

    def filter(
        self,
        blocks: list["Block"],
        *,
        task_descriptor: "TaskDescriptor | None" = None,
    ) -> tuple[list["Block"], list["Block"]]:
        """
        Filter blocks by per-family quotas.

        Returns (kept_blocks, dropped_blocks).

        Family rules:
          - schema_core: always kept
          - evidence: sort by score, keep top evidence_quota
          - memory: sort by (position desc, score desc), keep top memory_quota
          - constraints: keep all protected; drop those with score < 0.05
          - background: keep those with score >= background_threshold
          - unscored blocks (score is None): assign score=0.5 and keep
        """
        # Task-aware quota adjustment
        evidence_quota = self.evidence_quota
        memory_quota = self.memory_quota
        background_threshold = self.background_threshold

        if task_descriptor is not None:
            policy = task_descriptor.retention_policy
            task_type = task_descriptor.task_type.value
            if task_type == "summarization":
                # Summarization: must keep ALL evidence — the whole doc is the signal.
                # Disable evidence filtering entirely by setting a very high quota.
                evidence_quota = 999
            elif policy.doc_weight > 1.2:
                # RAG: keep more docs
                evidence_quota = int(evidence_quota * 1.5)
            if policy.history_weight > 1.2:
                # Chat: keep more history
                memory_quota = int(memory_quota * 1.5)
            if task_type == "tool_use":
                # Tool use: keep full schema, compress background aggressively
                background_threshold = 0.25
                evidence_quota = min(evidence_quota, 5)
            if task_type in ("math", "general"):
                # Reasoning tasks: keep more evidence to avoid chain fragmentation
                evidence_quota = int(evidence_quota * 1.5)
                background_threshold = 0.05  # be lenient — don't drop reasoning context

        # Group blocks by family
        families: dict[str, list[tuple[int, "Block"]]] = {
            "schema_core": [],
            "evidence": [],
            "constraints": [],
            "memory": [],
            "background": [],
        }

        for i, b in enumerate(blocks):
            family = _FAMILY_MAP.get(b.block_type.value, "background")
            families[family].append((i, b))

        kept: list["Block"] = []
        dropped: list["Block"] = []

        # --- schema_core: always keep ---
        for _, b in families["schema_core"]:
            # If task is TOOL_USE, always keep tool_schema regardless of score
            kept.append(b)

        # --- evidence: keep top-K by score ---
        evidence_pairs = families["evidence"]
        # Assign default score for unscored blocks
        evidence_sorted = sorted(
            evidence_pairs,
            key=lambda pair: pair[1].score if pair[1].score is not None else 0.5,
            reverse=True,
        )
        for rank, (_, b) in enumerate(evidence_sorted):
            if b.score is None:
                # Unscored → keep with neutral score
                kept.append(b)
            elif rank < evidence_quota:
                kept.append(b)
            else:
                dropped.append(b.clone_with(drop_reason="adaptive_filter:evidence_quota_exceeded"))

        # --- memory: keep top-K by (position desc, score desc) ---
        memory_pairs = families["memory"]

        def _memory_sort_key(pair: tuple[int, "Block"]) -> tuple[int, float]:
            _, b = pair
            pos = b.metadata.position if b.metadata.position is not None else 0
            score = b.score if b.score is not None else 0.5
            return (pos, score)

        memory_sorted = sorted(memory_pairs, key=_memory_sort_key, reverse=True)
        for rank, (_, b) in enumerate(memory_sorted):
            if b.score is None:
                kept.append(b)
            elif rank < memory_quota:
                kept.append(b)
            else:
                dropped.append(b.clone_with(drop_reason="adaptive_filter:memory_quota_exceeded"))

        # --- constraints: keep all protected; drop low-score ---
        for _, b in families["constraints"]:
            if b.metadata.protected:
                kept.append(b)
            elif b.score is None:
                # Unscored → keep
                kept.append(b)
            elif b.score < _CONSTRAINTS_MIN_SCORE:
                dropped.append(b.clone_with(drop_reason=f"adaptive_filter:constraint_score_too_low:{b.score:.3f}"))
            else:
                kept.append(b)

        # --- background: keep by score threshold (task-adjusted) ---
        for _, b in families["background"]:
            if b.score is None:
                # Unscored → keep
                kept.append(b)
            elif b.score >= background_threshold:
                kept.append(b)
            else:
                dropped.append(b.clone_with(drop_reason=f"adaptive_filter:background_score_too_low:{b.score:.3f}"))

        return kept, dropped
