"""Content splitter — breaks block text into sections and sentences."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Sentence:
    text: str
    index: int          # Position within the parent section
    score: float = 0.0


@dataclass
class Section:
    heading: str | None         # None for body-level sections without a heading
    body: str
    index: int                  # Position within the parent block
    level: int = 0              # Heading depth: 0 = no heading, 1 = #, 2 = ##, etc.
    score: float = 0.0
    sentences: list[Sentence] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        if self.heading:
            prefix = "#" * self.level + " " if self.level > 0 else ""
            return f"{prefix}{self.heading}\n{self.body}".strip()
        return self.body.strip()

    @property
    def is_empty(self) -> bool:
        return not self.body.strip() and not self.heading


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Markdown headings: # H1, ## H2, ### H3
_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

# Double-newline paragraph break (used when no headings present)
_PARAGRAPH_BREAK = re.compile(r"\n{2,}")

# Sentence boundary: end-of-sentence punctuation followed by whitespace + capital
_SENTENCE_BOUNDARY = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z\"\'\(])"
    r"|(?<=\n)(?=[A-Z\"\'\(])"      # newline also breaks sentences
)

# Bullet list item — treat each bullet as its own sentence
_BULLET_LINE = re.compile(r"^[ \t]*[-*+•]\s+.+$", re.MULTILINE)


# ---------------------------------------------------------------------------
# ContentSplitter
# ---------------------------------------------------------------------------

class ContentSplitter:
    """
    Splits block content into a hierarchy of sections → sentences.

    Strategy:
    - If the text contains markdown headings, split on them.
    - Otherwise split on double-newlines (paragraph mode).
    - Each section is further split into sentences for fine-grained scoring.
    """

    def split_sections(self, text: str) -> list[Section]:
        """Return a list of Section objects for the given text."""
        text = text.strip()
        if not text:
            return []

        if self._has_headings(text):
            return self._split_by_headings(text)
        return self._split_by_paragraphs(text)

    def split_sentences(self, text: str) -> list[Sentence]:
        """Return a flat list of Sentence objects for the given text."""
        text = text.strip()
        if not text:
            return []

        # Bullet lines are atomic — don't split them further
        bullets = _BULLET_LINE.findall(text)
        if len(bullets) >= 3:
            # Mostly a bullet list — treat each line as a sentence
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            return [Sentence(text=ln, index=i) for i, ln in enumerate(lines)]

        raw = _SENTENCE_BOUNDARY.split(text)
        sentences = []
        for i, chunk in enumerate(raw):
            chunk = chunk.strip()
            if chunk:
                sentences.append(Sentence(text=chunk, index=i))
        return sentences

    def split_full(self, text: str) -> list[Section]:
        """Split into sections and populate each section's sentences."""
        sections = self.split_sections(text)
        for section in sections:
            section.sentences = self.split_sentences(section.body)
        return sections

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _has_headings(text: str) -> bool:
        return bool(_MD_HEADING.search(text))

    def _split_by_headings(self, text: str) -> list[Section]:
        sections: list[Section] = []
        matches = list(_MD_HEADING.finditer(text))

        # Text before the first heading becomes section 0 (no heading)
        if matches and matches[0].start() > 0:
            preamble = text[: matches[0].start()].strip()
            if preamble:
                sections.append(Section(heading=None, body=preamble, index=0, level=0))

        for i, match in enumerate(matches):
            level = len(match.group(1))
            heading = match.group(2).strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[start:end].strip()
            sections.append(
                Section(heading=heading, body=body, index=len(sections), level=level)
            )

        return sections or [Section(heading=None, body=text, index=0)]

    def _split_by_paragraphs(self, text: str) -> list[Section]:
        parts = _PARAGRAPH_BREAK.split(text)
        sections = []
        for i, part in enumerate(parts):
            part = part.strip()
            if part:
                sections.append(Section(heading=None, body=part, index=i))
        return sections or [Section(heading=None, body=text, index=0)]

    # ------------------------------------------------------------------
    # Reassembly
    # ------------------------------------------------------------------

    @staticmethod
    def reassemble_sections(sections: list[Section], separator: str = "\n\n") -> str:
        """Reconstruct text from a filtered list of sections."""
        return separator.join(s.full_text for s in sections if not s.is_empty)

    @staticmethod
    def reassemble_sentences(sentences: list[Sentence], separator: str = " ") -> str:
        """Reconstruct text from a filtered list of sentences."""
        return separator.join(s.text for s in sentences if s.text.strip())
