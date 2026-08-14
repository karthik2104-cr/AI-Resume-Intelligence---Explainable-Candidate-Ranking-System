"""Plain-text document ingestion."""

from __future__ import annotations

import logging
from typing import Optional

from src.ingestion.base import DocumentIngester, FileInput
from src.ingestion.errors import ExtractionFailureError
from src.ingestion.io_helpers import normalize_extracted_text, read_bytes
from src.ingestion.validation import DocumentValidator
from src.models.document import Document, DocumentPage, DocumentSourceType
from src.utils.config import IngestionConfig, get_settings

logger = logging.getLogger(__name__)


class TxtIngester(DocumentIngester):
    """Extract text from UTF-8 and common legacy encodings."""

    def __init__(
        self,
        config: Optional[IngestionConfig] = None,
        validator: Optional[DocumentValidator] = None,
    ) -> None:
        self._config = config or get_settings().ingestion
        self._validator = validator or DocumentValidator(self._config)

    @property
    def supported_types(self) -> list[DocumentSourceType]:
        return [DocumentSourceType.TXT]

    def _decode_text(self, data: bytes, filename: str) -> tuple[str, str]:
        for encoding in self._config.txt_encodings:
            try:
                return data.decode(encoding), encoding
            except UnicodeDecodeError:
                continue

        logger.error("Unable to decode TXT file '%s' with configured encodings.", filename)
        raise ExtractionFailureError(
            f"Unable to decode text file '{filename}' using supported encodings: "
            f"{self._config.txt_encodings}"
        )

    def ingest(self, source: FileInput, filename: str | None = None) -> Document:
        if not filename:
            raise ExtractionFailureError("TXT ingestion requires a filename.")

        data = read_bytes(source)
        self._validator.validate_before_ingestion(data, filename)

        try:
            raw_text, encoding = self._decode_text(data, filename)
        except ExtractionFailureError:
            raise
        except Exception as exc:
            logger.exception("Unexpected TXT decode failure for '%s'", filename)
            raise ExtractionFailureError(
                f"Failed to extract content from text file '{filename}'."
            ) from exc

        normalized = normalize_extracted_text(raw_text)
        pages = [DocumentPage(page_number=1, text=normalized)] if normalized else []
        warnings: list[str] = []
        if not normalized:
            warnings.append("Text file contains no non-whitespace content.")

        document = Document(
            source_type=DocumentSourceType.TXT,
            filename=filename,
            raw_text=normalized,
            pages=pages,
            metadata={"encoding": encoding},
            extraction_warnings=warnings,
        )
        self._validator.validate_extracted_document(document)
        return document
