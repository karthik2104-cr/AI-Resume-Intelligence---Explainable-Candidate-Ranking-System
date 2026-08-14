"""Document ingestion abstract interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import BinaryIO, Union

from src.models.document import Document, DocumentSourceType

FileInput = Union[str, Path, BinaryIO, bytes]

# Re-export domain errors from errors module for backward compatibility.
from src.ingestion.errors import (  # noqa: E402
    CorruptedFileError,
    DocumentIngestionError,
    EmptyDocumentError,
    ExtractionFailureError,
    FileTooLargeError,
    UnsupportedFileTypeError,
)

__all__ = [
    "DocumentIngester",
    "DocumentIngestionError",
    "UnsupportedFileTypeError",
    "FileTooLargeError",
    "CorruptedFileError",
    "ExtractionFailureError",
    "EmptyDocumentError",
    "FileInput",
]


class DocumentIngester(ABC):
    """Abstract base for format-specific document ingestion."""

    @property
    @abstractmethod
    def supported_types(self) -> list[DocumentSourceType]:
        """Return supported document source types."""

    @abstractmethod
    def ingest(self, source: FileInput, filename: str | None = None) -> Document:
        """Extract text and metadata from a document source."""

    def validate_extension(self, filename: str, allowed: list[str]) -> None:
        suffix = Path(filename).suffix.lower()
        if suffix not in allowed:
            raise UnsupportedFileTypeError(
                f"Unsupported file extension '{suffix}'. Allowed: {allowed}"
            )
