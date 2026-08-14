"""Shared I/O helpers for document ingesters."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import BinaryIO

from src.ingestion.base import FileInput


def read_bytes(source: FileInput) -> bytes:
    """Read a file path, bytes payload, or binary stream into memory."""
    if isinstance(source, bytes):
        return source

    if isinstance(source, (str, Path)):
        path = Path(source)
        return path.read_bytes()

    if hasattr(source, "read"):
        stream: BinaryIO = source  # type: ignore[assignment]
        position = stream.tell() if hasattr(stream, "tell") else None
        data = stream.read()
        if position is not None and hasattr(stream, "seek"):
            stream.seek(position)
        return data

    raise TypeError(f"Unsupported source type: {type(source)!r}")


def normalize_extracted_text(text: str) -> str:
    """Normalize whitespace while preserving paragraph breaks where possible."""
    lines = [line.strip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    cleaned_lines: list[str] = []
    blank_run = 0
    for line in lines:
        if not line:
            blank_run += 1
            if blank_run <= 1:
                cleaned_lines.append("")
            continue
        blank_run = 0
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def bytes_to_stream(data: bytes) -> BytesIO:
    return BytesIO(data)
