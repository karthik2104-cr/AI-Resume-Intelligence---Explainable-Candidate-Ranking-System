"""Shared document ingest + parse helpers for API and Streamlit.

Keeps presentation layers thin and avoids duplicating ingestion/parsing logic.
"""
from __future__ import annotations

from typing import BinaryIO

from src.ingestion.factory import IngestionFactory
from src.models.document import Document
from src.models.job import ParsedJobDescription
from src.models.resume import ParsedResume
from src.parsing.job_parser import parse_job_description
from src.parsing.resume_parser import HeuristicResumeParser


def ingest_bytes(content: bytes, filename: str) -> Document:
    """Ingest raw file bytes into a Document."""
    return IngestionFactory().ingest(content, filename=filename)


def parse_resume_bytes(content: bytes, filename: str, *, strip_pii: bool = True) -> ParsedResume:
    """Ingest and parse a resume; optionally clear email/phone for presentation."""
    doc = ingest_bytes(content, filename)
    parsed = HeuristicResumeParser().parse(doc)
    if strip_pii:
        parsed.email = None
        parsed.phone = None
    return parsed


def parse_resume_upload(upload: BinaryIO, filename: str, *, strip_pii: bool = True) -> ParsedResume:
    """Ingest and parse a file-like upload."""
    content = upload.read()
    return parse_resume_bytes(content, filename, strip_pii=strip_pii)


def parse_job(job_text: str, title: str | None = None) -> ParsedJobDescription:
    """Parse a job description string."""
    return parse_job_description(job_text, title=title)
