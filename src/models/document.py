"""Document abstraction — format-agnostic representation after ingestion."""

from __future__ import annotations

from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class DocumentSourceType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    PLAIN_TEXT = "plain_text"


class DocumentPage(BaseModel):
    """Single page or logical chunk of extracted text."""

    page_number: int
    text: str


class Document(BaseModel):
    """
    Common document representation consumed by parsers.

    The rest of the pipeline does not need to know the original file format.
    """

    document_id: str = Field(default_factory=lambda: str(uuid4()))
    source_type: DocumentSourceType
    filename: Optional[str] = None
    raw_text: str
    pages: list[DocumentPage] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)
    extraction_warnings: list[str] = Field(default_factory=list)

    @property
    def extracted_text(self) -> str:
        """Alias for raw_text — explicit name for downstream parsers."""
        return self.raw_text

    @property
    def is_empty(self) -> bool:
        return not self.raw_text.strip()

    @property
    def char_count(self) -> int:
        return len(self.raw_text)

    @property
    def page_count(self) -> int:
        return len(self.pages) if self.pages else 1
