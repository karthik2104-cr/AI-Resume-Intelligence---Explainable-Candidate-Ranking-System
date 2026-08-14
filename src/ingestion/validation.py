"""Configurable validation for uploaded documents."""

from __future__ import annotations

import logging
import mimetypes
from pathlib import Path

from src.ingestion.errors import (
    EmptyDocumentError,
    FileTooLargeError,
    UnsupportedFileTypeError,
)
from src.models.document import Document
from src.utils.config import IngestionConfig, get_settings

logger = logging.getLogger(__name__)

EXTENSION_TO_MIME: dict[str, list[str]] = {
    ".pdf": ["application/pdf"],
    ".docx": [
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ],
    ".txt": ["text/plain"],
}

MAGIC_SIGNATURES: dict[str, list[bytes]] = {
    ".pdf": [b"%PDF"],
    ".docx": [b"PK\x03\x04"],
}


class DocumentValidator:
    """Validate file inputs before and after extraction."""

    def __init__(self, config: IngestionConfig | None = None) -> None:
        self._config = config or get_settings().ingestion

    @property
    def max_bytes(self) -> int:
        return self._config.max_upload_size_mb * 1024 * 1024

    def validate_filename(self, filename: str | None) -> str:
        if not filename or not filename.strip():
            raise UnsupportedFileTypeError("Filename is required for document ingestion.")

        suffix = Path(filename).suffix.lower()
        if suffix not in self._config.allowed_extensions:
            raise UnsupportedFileTypeError(
                f"Unsupported file extension '{suffix}'. "
                f"Allowed extensions: {self._config.allowed_extensions}"
            )
        return suffix

    def validate_size(self, data: bytes, filename: str | None = None) -> None:
        if len(data) == 0:
            raise EmptyDocumentError("Uploaded file is empty.")

        if len(data) > self.max_bytes:
            raise FileTooLargeError(
                f"File '{filename or 'upload'}' exceeds maximum size of "
                f"{self._config.max_upload_size_mb} MB."
            )

    def validate_magic_bytes(self, data: bytes, extension: str) -> None:
        if not self._config.validate_magic_bytes:
            return

        signatures = MAGIC_SIGNATURES.get(extension)
        if not signatures:
            return

        if not any(data.startswith(sig) for sig in signatures):
            raise UnsupportedFileTypeError(
                f"File content does not match expected format for '{extension}'."
            )

    def validate_mime_type(self, filename: str, extension: str) -> None:
        guessed, _ = mimetypes.guess_type(filename)
        if not guessed:
            return

        allowed = EXTENSION_TO_MIME.get(extension, [])
        if allowed and guessed not in allowed:
            logger.warning(
                "MIME type '%s' for '%s' does not match expected %s; proceeding on extension.",
                guessed,
                filename,
                allowed,
            )

    def validate_before_ingestion(self, data: bytes, filename: str) -> str:
        extension = self.validate_filename(filename)
        self.validate_size(data, filename)
        self.validate_magic_bytes(data, extension)
        self.validate_mime_type(filename, extension)
        return extension

    def validate_extracted_document(self, document: Document) -> None:
        if not self._config.reject_empty_extraction:
            return

        if document.is_empty:
            raise EmptyDocumentError(
                f"No extractable text found in '{document.filename or 'document'}'."
            )
