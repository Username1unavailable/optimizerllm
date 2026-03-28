"""
Recover — restore critical slots lost during aggressive compression.

After budget control drops blocks, some brittle facts (names, numbers, dates,
constraints) may no longer appear in the final context. Recover detects these
missing slots and injects a compact recovery note so the model still has access
to the critical information.

This is a lightweight, deterministic post-pass — it never re-adds entire blocks,
only the minimum sentences needed to carry the missing slot values.

Inspired by:
  - LongLLMLingua subsequence recovery
  - AttnComp confidence-aware adaptive retention
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from optimizerllm.core.block import Block
    from optimizerllm.slot_guard.guard import Slot, SlotIndex


@dataclass
class RecoveryRecord:
    """Audit trail entry for each recovery action."""
    slot_type: str
    slot_text: str
    recovered_sentence: str
    injected_into_block_id: str | None


class Recover:
    """
    Post-compression slot recovery pass.

    Usage::

        recover = Recover(max_slots=5)
        final_blocks, records = recover.run(final_blocks, slot_index)
    """

    def __init__(
        self,
        *,
        max_slots: int = 8,          # Max number of lost slots to recover
        max_sentence_chars: int = 300,
        recovery_note_prefix: str = "[Context note] ",
    ) -> None:
        self.max_slots = max_slots
        self.max_sentence_chars = max_sentence_chars
        self.recovery_note_prefix = recovery_note_prefix

    def run(
        self,
        final_blocks: list["Block"],
        slot_index: "SlotIndex",
    ) -> tuple[list["Block"], list[RecoveryRecord]]:
        """
        Check for lost slots and inject recovery notes into the nearest
        appropriate block (the last user message, or a new recovery block).

        Returns updated block list and list of RecoveryRecord for telemetry.
        """
        if not slot_index or len(slot_index) == 0:
            return final_blocks, []

        lost_slots = slot_index.audit(final_blocks)
        if not lost_slots:
            return final_blocks, []

        records: list[RecoveryRecord] = []
        # Deduplicate: only recover each distinct slot text once
        seen: set[str] = set()
        to_recover: list[Slot] = []
        for slot in lost_slots[: self.max_slots]:
            if slot.normalized not in seen:
                to_recover.append(slot)
                seen.add(slot.normalized)

        if not to_recover:
            return final_blocks, []

        # Build a compact recovery note
        recovery_lines = []
        for slot in to_recover:
            sentence = slot.sentence.strip()[: self.max_sentence_chars]
            recovery_lines.append(f"• [{slot.slot_type.value}] {sentence}")

        note_content = (
            self.recovery_note_prefix
            + "The following key details were compressed out of earlier context:\n"
            + "\n".join(recovery_lines)
        )

        recovery_block = self._make_recovery_block(note_content)

        # Inject just before the last user message so it is in the live context
        insertion_idx = self._find_insertion_point(final_blocks)
        final_blocks = (
            final_blocks[:insertion_idx]
            + [recovery_block]
            + final_blocks[insertion_idx:]
        )

        for slot in to_recover:
            records.append(
                RecoveryRecord(
                    slot_type=slot.slot_type.value,
                    slot_text=slot.text,
                    recovered_sentence=slot.sentence[: self.max_sentence_chars],
                    injected_into_block_id=recovery_block.id,
                )
            )

        return final_blocks, records

    # ------------------------------------------------------------------

    @staticmethod
    def _make_recovery_block(content: str) -> "Block":
        from optimizerllm.core.block import Block, BlockMetadata, BlockType, Mutability

        return Block(
            content=content,
            block_type=BlockType.STRUCTURED_DATA,
            role="user",
            metadata=BlockMetadata(
                source="recover",
                priority=85,
                protected=True,
                mutability=Mutability.IMMUTABLE,
                cacheability=False,
            ),
        )

    @staticmethod
    def _find_insertion_point(blocks: list["Block"]) -> int:
        """Return index just before the last user message."""
        from optimizerllm.core.block import BlockType

        for i in range(len(blocks) - 1, -1, -1):
            if blocks[i].block_type == BlockType.USER_MESSAGE:
                return i
        return len(blocks)  # append to end if no user message found
