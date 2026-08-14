"""Domain-level errors for document ingestion."""

from __future__ import annotations


class DocumentIngestionError(Exception):
    """Base exception for document ingestion failures."""


class UnsupportedFileTypeError(DocumentIngestionError):
    """Raised when the file extension or type is not supported."""


class FileTooLargeError(DocumentIngestionError):
    """Raised when the file exceeds the configured size limit."""


class CorruptedFileError(DocumentIngestionError):
    """Raised when the file is malformed or cannot be parsed."""


class ExtractionFailureError(DocumentIngestionError):
    """Raised when text extraction fails for a supported format."""


class EmptyDocumentError(DocumentIngestionError):
    """Raised when the document or extracted text is empty."""
