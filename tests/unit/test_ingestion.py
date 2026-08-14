"""Unit tests for document ingestion."""

from __future__ import annotations

from io import BytesIO

import pytest

from src.ingestion.docx_ingester import DocxIngester
from src.ingestion.errors import (
    CorruptedFileError,
    EmptyDocumentError,
    ExtractionFailureError,
    FileTooLargeError,
    UnsupportedFileTypeError,
)
from src.ingestion.factory import IngestionFactory, ingest_document
from src.ingestion.pdf_ingester import PdfIngester
from src.ingestion.txt_ingester import TxtIngester
from src.ingestion.validation import DocumentValidator
from src.models.document import DocumentSourceType
from src.utils.config import IngestionConfig
from tests.fixtures.document_builders import build_docx, build_simple_pdf, build_txt


@pytest.fixture
def ingestion_config() -> IngestionConfig:
    return IngestionConfig(
        allowed_extensions=[".pdf", ".docx", ".txt"],
        max_upload_size_mb=1,
        validate_magic_bytes=True,
        reject_empty_extraction=True,
        txt_encodings=["utf-8", "utf-8-sig", "latin-1", "cp1252"],
    )


class TestPdfIngester:
    def test_valid_pdf(self, ingestion_config: IngestionConfig):
        data = build_simple_pdf(["Python Developer Resume"])
        ingester = PdfIngester(config=ingestion_config)
        doc = ingester.ingest(data, filename="resume.pdf")

        assert doc.source_type == DocumentSourceType.PDF
        assert "Python Developer Resume" in doc.raw_text
        assert doc.filename == "resume.pdf"
        assert doc.document_id
        assert doc.char_count > 0
        assert doc.page_count == 1

    def test_multi_page_pdf(self, ingestion_config: IngestionConfig):
        data = build_simple_pdf(["Page One Content", "Page Two Content"])
        doc = PdfIngester(config=ingestion_config).ingest(data, filename="multi.pdf")

        assert doc.page_count == 2
        assert len(doc.pages) == 2
        assert doc.pages[0].page_number == 1
        assert doc.pages[1].page_number == 2
        assert "Page One Content" in doc.pages[0].text
        assert "Page Two Content" in doc.pages[1].text

    def test_empty_pdf_raises(self, ingestion_config: IngestionConfig):
        data = build_simple_pdf([""])
        with pytest.raises(EmptyDocumentError):
            PdfIngester(config=ingestion_config).ingest(data, filename="empty.pdf")

    def test_corrupted_pdf_raises(self, ingestion_config: IngestionConfig):
        corrupted = b"%PDF-1.4\n% corrupted truncated content without valid objects"
        with pytest.raises(CorruptedFileError):
            PdfIngester(config=ingestion_config).ingest(corrupted, filename="bad.pdf")

    def test_invalid_pdf_magic_bytes(self, ingestion_config: IngestionConfig):
        with pytest.raises(UnsupportedFileTypeError):
            PdfIngester(config=ingestion_config).ingest(b"hello world", filename="fake.pdf")

    def test_missing_filename_raises(self, ingestion_config: IngestionConfig):
        with pytest.raises(CorruptedFileError):
            PdfIngester(config=ingestion_config).ingest(b"%PDF", filename=None)


class TestDocxIngester:
    def test_valid_docx(self, ingestion_config: IngestionConfig):
        data = build_docx(["Summary", "Experience in Python and SQL"])
        doc = DocxIngester(config=ingestion_config).ingest(data, filename="resume.docx")

        assert doc.source_type == DocumentSourceType.DOCX
        assert "Python and SQL" in doc.raw_text
        assert doc.extracted_text == doc.raw_text
        assert doc.metadata["paragraph_count"]

    def test_docx_with_table(self, ingestion_config: IngestionConfig):
        data = build_docx(
            ["Skills"],
            table_rows=[["Python", "Expert"], ["SQL", "Advanced"]],
        )
        doc = DocxIngester(config=ingestion_config).ingest(data, filename="table.docx")

        assert "Python | Expert" in doc.raw_text
        assert "SQL | Advanced" in doc.raw_text
        assert doc.metadata["table_count"] == "1"

    def test_empty_docx_raises(self, ingestion_config: IngestionConfig):
        data = build_docx([])
        with pytest.raises(EmptyDocumentError):
            DocxIngester(config=ingestion_config).ingest(data, filename="empty.docx")

    def test_corrupted_docx_raises(self, ingestion_config: IngestionConfig):
        corrupted = b"PK\x03\x04" + b"invalid zip payload"
        with pytest.raises(CorruptedFileError):
            DocxIngester(config=ingestion_config).ingest(corrupted, filename="bad.docx")


class TestTxtIngester:
    def test_valid_txt_utf8(self, ingestion_config: IngestionConfig):
        data = build_txt("Python developer with Django experience.")
        doc = TxtIngester(config=ingestion_config).ingest(data, filename="resume.txt")

        assert doc.source_type == DocumentSourceType.TXT
        assert "Django" in doc.raw_text
        assert doc.metadata["encoding"] == "utf-8"

    def test_empty_txt_raises(self, ingestion_config: IngestionConfig):
        with pytest.raises(EmptyDocumentError):
            TxtIngester(config=ingestion_config).ingest(b"   \n\t  ", filename="empty.txt")

    def test_latin1_encoding(self, ingestion_config: IngestionConfig):
        data = build_txt("Resume with latin text: cafe", encoding="latin-1")
        doc = TxtIngester(config=ingestion_config).ingest(data, filename="latin.txt")
        assert "cafe" in doc.raw_text

    def test_cp1252_encoding(self, ingestion_config: IngestionConfig):
        data = build_txt("Windows encoding test", encoding="cp1252")
        doc = TxtIngester(config=ingestion_config).ingest(data, filename="win.txt")
        assert "Windows encoding test" in doc.raw_text

    def test_unsupported_encoding_raises(self):
        strict_config = IngestionConfig(
            allowed_extensions=[".txt"],
            txt_encodings=["utf-8"],
            reject_empty_extraction=True,
        )
        data = "résumé".encode("utf-16")
        with pytest.raises(ExtractionFailureError):
            TxtIngester(config=strict_config).ingest(data, filename="utf16.txt")


class TestValidation:
    def test_unsupported_extension(self, ingestion_config: IngestionConfig):
        validator = DocumentValidator(ingestion_config)
        with pytest.raises(UnsupportedFileTypeError):
            validator.validate_filename("resume.exe")

    def test_oversized_file(self, ingestion_config: IngestionConfig):
        validator = DocumentValidator(ingestion_config)
        oversized = b"x" * (2 * 1024 * 1024)
        with pytest.raises(FileTooLargeError):
            validator.validate_size(oversized, filename="big.txt")

    def test_zero_byte_file(self, ingestion_config: IngestionConfig):
        validator = DocumentValidator(ingestion_config)
        with pytest.raises(EmptyDocumentError):
            validator.validate_size(b"", filename="zero.txt")

    def test_reject_empty_extraction_can_be_disabled(self):
        config = IngestionConfig(reject_empty_extraction=False)
        validator = DocumentValidator(config)
        from src.models.document import Document

        doc = Document(source_type=DocumentSourceType.TXT, raw_text="   ")
        validator.validate_extracted_document(doc)  # should not raise


class TestIngestionFactory:
    def test_pdf_routes_to_pdf_ingester(self, ingestion_config: IngestionConfig):
        factory = IngestionFactory(config=ingestion_config)
        ingester = factory.get_ingester_for_filename("resume.pdf")
        assert isinstance(ingester, PdfIngester)

    def test_docx_routes_to_docx_ingester(self, ingestion_config: IngestionConfig):
        factory = IngestionFactory(config=ingestion_config)
        ingester = factory.get_ingester_for_filename("resume.docx")
        assert isinstance(ingester, DocxIngester)

    def test_txt_routes_to_txt_ingester(self, ingestion_config: IngestionConfig):
        factory = IngestionFactory(config=ingestion_config)
        ingester = factory.get_ingester_for_filename("resume.txt")
        assert isinstance(ingester, TxtIngester)

    def test_unsupported_format_raises(self, ingestion_config: IngestionConfig):
        factory = IngestionFactory(config=ingestion_config)
        with pytest.raises(UnsupportedFileTypeError):
            factory.ingest(b"data", filename="file.xlsx")

    def test_ingest_document_convenience(self, ingestion_config: IngestionConfig):
        data = build_txt("Factory convenience test")
        doc = ingest_document(data, filename="note.txt", config=ingestion_config)
        assert "Factory convenience test" in doc.raw_text

    def test_ingest_from_binary_stream(self, ingestion_config: IngestionConfig):
        data = build_txt("Stream ingestion test")
        stream = BytesIO(data)
        doc = IngestionFactory(config=ingestion_config).ingest(stream, filename="stream.txt")
        assert "Stream ingestion test" in doc.raw_text

    def test_factory_reuses_ingester_instances(self, ingestion_config: IngestionConfig):
        factory = IngestionFactory(config=ingestion_config)
        first = factory.get_ingester(".pdf")
        second = factory.get_ingester(".pdf")
        assert first is second
