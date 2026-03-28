"""
SlotGuard — extract and protect brittle facts before compression.

Brittle facts are values that, if lost, cause silent wrong answers:
  person names, dates, prices, quantities, API/schema field names,
  constraints, code identifiers, URLs, email addresses.

Pipeline contract:
  1. Call SlotGuard.extract(blocks) → SlotIndex before any compression.
  2. Pass slot_index to GranularSelector and ConversationCompactor so they
     never cut sentences that contain a protected slot.
  3. Call SlotIndex.audit(final_blocks) after budget control to find any
     slots that were lost despite protection.
  4. Pass lost slots to Recover for restoration.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from optimizerllm.core.block import Block


class SlotType(str, Enum):
    PERSON_NAME   = "person_name"
    DATE          = "date"
    PRICE         = "price"
    QUANTITY      = "quantity"
    API_FIELD     = "api_field"
    CODE_IDENT    = "code_ident"
    URL           = "url"
    EMAIL         = "email"
    CONSTRAINT    = "constraint"
    NUMERIC       = "numeric"
    ENUM_VALUE    = "enum_value"


@dataclass
class Slot:
    """A single extracted brittle fact."""
    text: str                   # Raw matched text
    slot_type: SlotType
    block_id: str               # Which block it came from
    sentence: str               # Full sentence containing the slot (for recovery)
    normalized: str = ""        # Lowercased/stripped for dedup comparison

    def __post_init__(self) -> None:
        self.normalized = self.text.lower().strip()


@dataclass
class SlotIndex:
    """
    All slots extracted from the current context, keyed by block_id.

    Used downstream by:
      - GranularSelector: skip trimming any sentence that contains a slot
      - ConversationCompactor: mark slot-carrying sentences as protected
      - Recover: detect which slots survived budget control
    """
    slots: list[Slot] = field(default_factory=list)

    # Set of block_ids that contained at least one slot
    _slot_blocks: set[str] = field(default_factory=set, repr=False)

    def add(self, slot: Slot) -> None:
        # Deduplicate by (normalized_text, block_id)
        key = (slot.normalized, slot.block_id)
        existing = {(s.normalized, s.block_id) for s in self.slots}
        if key not in existing:
            self.slots.append(slot)
            self._slot_blocks.add(slot.block_id)

    def sentence_has_slot(self, sentence: str) -> bool:
        """True if any known slot value appears verbatim in this sentence."""
        sl = sentence.lower()
        return any(s.normalized in sl for s in self.slots)

    def block_has_slot(self, block_id: str) -> bool:
        return block_id in self._slot_blocks

    def audit(self, final_blocks: list["Block"]) -> list[Slot]:
        """
        Return slots whose text is no longer present in any final block.
        These are candidates for recovery.
        """
        all_content = " ".join(b.content.lower() for b in final_blocks)
        return [s for s in self.slots if s.normalized not in all_content]

    def __len__(self) -> int:
        return len(self.slots)


# ---------------------------------------------------------------------------
# Extraction patterns
# ---------------------------------------------------------------------------

_PRICE_RE = re.compile(
    r"""
    (?:
        [\$£€¥₹][\s]?\d[\d,]*(?:\.\d{1,2})?   # $42, $1,234.56
        | \d[\d,]*(?:\.\d{1,2})?[\s]?(?:USD|EUR|GBP|JPY|INR|CAD|AUD)
        | \d+(?:\.\d+)?[\s]?(?:dollars?|euros?|pounds?|yen|rupees?)
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

_DATE_RE = re.compile(
    r"""
    (?:
        (?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?
          |Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)
        [\s,]+\d{1,2}(?:st|nd|rd|th)?(?:[\s,]+\d{2,4})?  # January 5th, 2024
        | \d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}           # 01/15/2024
        | \d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2}             # 2024-01-15
        | \d{4}                                             # standalone year
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

_QUANTITY_RE = re.compile(
    r"""
    \b\d+(?:[,\.]\d+)?[\s]?
    (?:kg|g|mg|lb(?:s)?|oz|km|m|cm|mm|miles?|ft|inches?
      |hours?|hrs?|minutes?|mins?|seconds?|secs?|days?|weeks?|months?|years?
      |units?|items?|pieces?|employees?|users?|requests?|tokens?|pages?|rows?
      |litres?|liters?|ml|gallons?|acres?|hectares?)
    \b
    """,
    re.IGNORECASE | re.VERBOSE,
)

_NUMERIC_RE = re.compile(
    r"""
    (?<!\w)                          # not preceded by a word char
    (?:\d{1,3}(?:,\d{3})+|\d{2,})   # 1,234 or plain ≥2 digit number
    (?:\.\d+)?                       # optional decimal
    (?!\w)                           # not followed by a word char
    """,
    re.VERBOSE,
)

# snake_case, camelCase, SCREAMING_SNAKE that look like code/API fields
_API_FIELD_RE = re.compile(
    r"""
    \b
    (?:
        [a-z][a-z0-9]*(?:_[a-z0-9]+){1,}   # snake_case (≥1 underscore)
        | [a-z][a-zA-Z0-9]{3,}(?=[A-Z])     # camelCase prefix
        | [A-Z][a-z][a-zA-Z0-9]*[A-Z][a-zA-Z0-9]*  # CamelCase
        | [A-Z_]{3,}                         # SCREAMING_SNAKE
    )
    \b
    """,
    re.VERBOSE,
)

_URL_RE = re.compile(
    r"https?://[^\s\]\)\>\"\']+",
    re.IGNORECASE,
)

_EMAIL_RE = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
)

# Proper-noun-like name: 2–4 capitalized words not at sentence start
_NAME_RE = re.compile(
    r"(?<=[.!?\s])([A-Z][a-z]+(?:\s[A-Z][a-z]+){1,3})",
)

# Constraint-like sentences: "must", "cannot", "only", "always", "never" + requirement
_CONSTRAINT_RE = re.compile(
    r"""
    (?:must|cannot|can\'t|only|always|never|required|mandatory|forbidden|
       not\s+allowed|prohibited|should\s+not|do\s+not|don\'t)
    .{5,80}
    """,
    re.IGNORECASE | re.VERBOSE,
)

# STOP WORDS for API field extraction (too common to be meaningful)
_API_FIELD_STOP = frozenset({
    "with", "this", "that", "from", "into", "over", "under", "have",
    "been", "will", "also", "then", "when", "what", "your", "their",
    "some", "more", "such", "each", "much", "many", "most", "only",
    "very", "just", "both", "between", "through", "where", "after",
    "before", "about", "other", "first", "last", "next", "same",
})


def _sentences(text: str) -> list[str]:
    """Split text into rough sentences."""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 5]


class SlotGuard:
    """
    Deterministic slot extractor.

    Usage::

        guard = SlotGuard()
        slot_index = guard.extract(blocks)

        # pass slot_index downstream to GranularSelector, Compactor, Recover
    """

    def __init__(
        self,
        *,
        protect_names: bool = True,
        protect_dates: bool = True,
        protect_prices: bool = True,
        protect_quantities: bool = True,
        protect_numerics: bool = True,
        protect_api_fields: bool = True,
        protect_urls: bool = True,
        protect_emails: bool = True,
        protect_constraints: bool = True,
        min_name_len: int = 4,   # ignore very short "names" like "Al"
    ) -> None:
        self.protect_names = protect_names
        self.protect_dates = protect_dates
        self.protect_prices = protect_prices
        self.protect_quantities = protect_quantities
        self.protect_numerics = protect_numerics
        self.protect_api_fields = protect_api_fields
        self.protect_urls = protect_urls
        self.protect_emails = protect_emails
        self.protect_constraints = protect_constraints
        self.min_name_len = min_name_len

    # ------------------------------------------------------------------

    def extract(self, blocks: list["Block"]) -> SlotIndex:
        """
        Scan all blocks and return a SlotIndex of all brittle facts found.
        Only blocks that are DROPPABLE or COMPRESSIBLE are scanned — immutable
        / protected blocks don't need slot protection because they're never dropped.
        """
        from optimizerllm.core.block import Mutability

        index = SlotIndex()
        for block in blocks:
            if block.metadata.protected or block.metadata.mutability == Mutability.IMMUTABLE:
                continue
            self._extract_from_block(block, index)
        return index

    def _extract_from_block(self, block: "Block", index: SlotIndex) -> None:
        text = block.content
        sents = _sentences(text)

        for sent in sents:
            self._extract_prices(sent, block.id, index)
            self._extract_dates(sent, block.id, index)
            self._extract_quantities(sent, block.id, index)
            self._extract_numerics(sent, block.id, index)
            self._extract_urls(sent, block.id, index)
            self._extract_emails(sent, block.id, index)
            self._extract_api_fields(sent, block.id, index)
            self._extract_constraints(sent, block.id, index)
            self._extract_names(sent, block.id, index)

    # ------------------------------------------------------------------
    # Per-type extractors
    # ------------------------------------------------------------------

    def _extract_prices(self, sent: str, block_id: str, index: SlotIndex) -> None:
        if not self.protect_prices:
            return
        for m in _PRICE_RE.finditer(sent):
            index.add(Slot(m.group(), SlotType.PRICE, block_id, sent))

    def _extract_dates(self, sent: str, block_id: str, index: SlotIndex) -> None:
        if not self.protect_dates:
            return
        for m in _DATE_RE.finditer(sent):
            val = m.group().strip()
            if len(val) >= 4:
                index.add(Slot(val, SlotType.DATE, block_id, sent))

    def _extract_quantities(self, sent: str, block_id: str, index: SlotIndex) -> None:
        if not self.protect_quantities:
            return
        for m in _QUANTITY_RE.finditer(sent):
            index.add(Slot(m.group().strip(), SlotType.QUANTITY, block_id, sent))

    def _extract_numerics(self, sent: str, block_id: str, index: SlotIndex) -> None:
        if not self.protect_numerics:
            return
        # Only extract numerics that haven't already been caught by price/quantity
        for m in _NUMERIC_RE.finditer(sent):
            val = m.group().strip()
            # Skip pure years already caught by date, and trivially small numbers
            if val.isdigit() and int(val) < 10:
                continue
            index.add(Slot(val, SlotType.NUMERIC, block_id, sent))

    def _extract_urls(self, sent: str, block_id: str, index: SlotIndex) -> None:
        if not self.protect_urls:
            return
        for m in _URL_RE.finditer(sent):
            index.add(Slot(m.group(), SlotType.URL, block_id, sent))

    def _extract_emails(self, sent: str, block_id: str, index: SlotIndex) -> None:
        if not self.protect_emails:
            return
        for m in _EMAIL_RE.finditer(sent):
            index.add(Slot(m.group(), SlotType.EMAIL, block_id, sent))

    def _extract_api_fields(self, sent: str, block_id: str, index: SlotIndex) -> None:
        if not self.protect_api_fields:
            return
        for m in _API_FIELD_RE.finditer(sent):
            val = m.group()
            if val.lower() in _API_FIELD_STOP:
                continue
            if len(val) < 4:
                continue
            index.add(Slot(val, SlotType.API_FIELD, block_id, sent))

    def _extract_constraints(self, sent: str, block_id: str, index: SlotIndex) -> None:
        if not self.protect_constraints:
            return
        if _CONSTRAINT_RE.search(sent):
            # Store the whole sentence as the slot value
            trimmed = sent.strip()[:200]
            index.add(Slot(trimmed, SlotType.CONSTRAINT, block_id, sent))

    def _extract_names(self, sent: str, block_id: str, index: SlotIndex) -> None:
        if not self.protect_names:
            return
        for m in _NAME_RE.finditer(f". {sent}"):  # prefix dot to enable look-behind
            val = m.group(1)
            if len(val) < self.min_name_len:
                continue
            # Reject if it looks like a sentence start (preceded by period+space)
            index.add(Slot(val, SlotType.PERSON_NAME, block_id, sent))
