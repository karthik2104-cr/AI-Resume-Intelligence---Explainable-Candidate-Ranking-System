"""PDF document ingestion."""

from __future__ import annotations

import logging
from typing import Optional

from pypdf import PdfReader
from pypdf.errors import PdfReadError, PdfStreamError

from src.ingestion.base import DocumentIngester, FileInput
from src.ingestion.errors import CorruptedFileError, ExtractionFailureError
from src.ingestion.io_helpers import bytes_to_stream, normalize_extracted_text, read_bytes
from src.ingestion.validation import DocumentValidator
from src.models.document import Document, DocumentPage, DocumentSourceType
from src.utils.config import IngestionConfig, get_settings

logger = logging.getLogger(__name__)


class PdfIngester(DocumentIngester):
    """Extract text from PDF files with per-page boundaries."""

    def __init__(
        self,
        config: Optional[IngestionConfig] = None,
        validator: Optional[DocumentValidator] = None,
    ) -> None:
        self._config = config or get_settings().ingestion
        self._validator = validator or DocumentValidator(self._config)

    @property
    def supported_types(self) -> list[DocumentSourceType]:
        return [DocumentSourceType.PDF]

    def ingest(self, source: FileInput, filename: str | None = None) -> Document:
        if not filename:
            raise CorruptedFileError("PDF ingestion requires a filename.")

        data = read_bytes(source)
        self._validator.validate_before_ingestion(data, filename)

        try:
            reader = PdfReader(bytes_to_stream(data), strict=False)
        except (PdfReadError, PdfStreamError, OSError, ValueError) as exc:
            logger.exception("Failed to read PDF '%s'", filename)
            raise CorruptedFileError(
                f"Unable to read PDF file '{filename}'. The file may be corrupted or invalid."
            ) from exc

        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as exc:
                logger.warning("Encrypted PDF '%s' could not be decrypted: %s", filename, exc)
                raise ExtractionFailureError(
                    f"PDF '{filename}' is encrypted and cannot be processed."
                ) from exc

        pages: list[DocumentPage] = []
        warnings: list[str] = []

        try:
            page_objects = list(reader.pages)
        except (PdfReadError, PdfStreamError) as exc:
            logger.exception("Failed to iterate PDF pages for '%s'", filename)
            raise CorruptedFileError(
                f"Unable to read pages from PDF file '{filename}'."
            ) from exc

        if not page_objects:
            warnings.append("PDF contains no pages.")

        for index, page in enumerate(page_objects, start=1):
            try:
                page_text = page.extract_text() or ""
            except Exception as exc:
                logger.warning("Text extraction failed on page %d of '%s': %s", index, filename, exc)
                page_text = ""
                warnings.append(f"Page {index} could not be extracted: {exc}")

            if not page_text.strip():
                warnings.append(f"Page {index} yielded no extractable text.")

            pages.append(DocumentPage(page_number=index, text=page_text))

        raw_text = normalize_extracted_text("\n\n".join(page.text for page in pages))
        metadata = {
            "page_count": str(len(pages)),
            "pdf_encrypted": str(getattr(reader, "is_encrypted", False)),
        }

        document = Document(
            source_type=DocumentSourceType.PDF,
            filename=filename,
            raw_text=raw_text,
            pages=pages,
            metadata=metadata,
            extraction_warnings=warnings,
        )
        self._validator.validate_extracted_document(document)
        return document
