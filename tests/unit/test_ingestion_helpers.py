"""Tests for ingestion I/O helpers."""

from src.ingestion.io_helpers import normalize_extracted_text, read_bytes


def test_normalize_extracted_text_collapses_blank_lines():
    text = "Line one\r\n\r\n\r\nLine two\n\nLine three"
    result = normalize_extracted_text(text)
    assert result == "Line one\n\nLine two\n\nLine three"


def test_read_bytes_from_path(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_bytes(b"hello")
    assert read_bytes(file_path) == b"hello"
