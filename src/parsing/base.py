"""Resume parsing abstract interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.models.document import Document
from src.models.resume import ParsedResume


class ResumeParser(ABC):
    """Abstract base for resume section parsing and structuring."""

    @abstractmethod
    def parse(self, document: Document) -> ParsedResume:
        """Parse a document into a structured resume."""

    def parse_text(self, text: str) -> ParsedResume:
        """Convenience method for plain-text resumes."""
        from src.models.document import Document, DocumentSourceType

        doc = Document(source_type=DocumentSourceType.PLAIN_TEXT, raw_text=text)
        return self.parse(doc)
