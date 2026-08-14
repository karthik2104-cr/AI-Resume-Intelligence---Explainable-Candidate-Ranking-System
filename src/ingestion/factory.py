"""Ingestion factory and registry."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.ingestion.base import DocumentIngester, FileInput
from src.ingestion.docx_ingester import DocxIngester
from src.ingestion.errors import UnsupportedFileTypeError
from src.ingestion.pdf_ingester import PdfIngester
from src.ingestion.txt_ingester import TxtIngester
from src.ingestion.validation import DocumentValidator
from src.models.document import Document, DocumentSourceType
from src.utils.config import IngestionConfig, get_settings

EXTENSION_REGISTRY: dict[str, type[DocumentIngester]] = {
    ".pdf": PdfIngester,
    ".docx": DocxIngester,
    ".txt": TxtIngester,
}


class IngestionFactory:
    """Select and invoke the correct ingester based on file extension."""

    def __init__(self, config: Optional[IngestionConfig] = None) -> None:
        self._config = config or get_settings().ingestion
        self._validator = DocumentValidator(self._config)
        self._ingesters: dict[str, DocumentIngester] = {}

    def resolve_extension(self, filename: str) -> str:
        return self._validator.validate_filename(filename)

    def get_ingester(self, extension: str) -> DocumentIngester:
        normalized = extension.lower()
        if normalized not in EXTENSION_REGISTRY:
            raise UnsupportedFileTypeError(
                f"No ingester registered for extension '{normalized}'. "
                f"Supported extensions: {list(EXTENSION_REGISTRY)}"
            )

        if normalized not in self._ingesters:
            self._ingesters[normalized] = EXTENSION_REGISTRY[normalized](
                config=self._config,
                validator=self._validator,
            )
        return self._ingesters[normalized]

    def get_ingester_for_filename(self, filename: str) -> DocumentIngester:
        extension = self.resolve_extension(filename)
        return self.get_ingester(extension)

    def ingest(self, source: FileInput, filename: str) -> Document:
        ingester = self.get_ingester_for_filename(filename)
        return ingester.ingest(source, filename=filename)

    def ingest_path(self, path: str | Path) -> Document:
        file_path = Path(path)
        return self.ingest(file_path.read_bytes(), filename=file_path.name)


def ingest_document(
    source: FileInput,
    filename: str,
    config: Optional[IngestionConfig] = None,
) -> Document:
    """Convenience function for one-shot document ingestion."""
    return IngestionFactory(config=config).ingest(source, filename)


def get_source_type_for_extension(extension: str) -> DocumentSourceType:
    mapping = {
        ".pdf": DocumentSourceType.PDF,
        ".docx": DocumentSourceType.DOCX,
        ".txt": DocumentSourceType.TXT,
    }
    try:
        return mapping[extension.lower()]
    except KeyError as exc:
        raise UnsupportedFileTypeError(
            f"Unsupported extension '{extension}'."
        ) from exc
