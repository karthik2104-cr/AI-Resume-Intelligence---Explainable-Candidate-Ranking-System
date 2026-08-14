"""End-to-end ingest + parse pipeline helpers."""

from __future__ import annotations

from typing import Optional

from src.ingestion.base import FileInput
from src.ingestion.factory import ingest_document
from src.models.document import Document
from src.models.job import JobDescription, ParsedJobDescription
from src.models.resume import ParsedResume
from src.parsing.job_parser import HeuristicJobDescriptionParser, parse_job_description
from src.parsing.resume_parser import HeuristicResumeParser
from src.utils.config import JobParsingConfig, ParsingConfig, get_settings


def parse_document(document: Document, config: Optional[ParsingConfig] = None) -> ParsedResume:
    """Parse an ingested Document into a structured resume."""
    return HeuristicResumeParser(config=config or get_settings().parsing).parse(document)


def ingest_and_parse_resume(
    source: FileInput,
    filename: str,
    config: Optional[ParsingConfig] = None,
) -> ParsedResume:
    """
    Full pipeline: file bytes/path → Document → ParsedResume.

    Example:
        parsed = ingest_and_parse_resume(file_bytes, filename="resume.pdf")
    """
    document = ingest_document(source, filename)
    return parse_document(document, config=config)


def parse_job(job: JobDescription, config: Optional[JobParsingConfig] = None) -> ParsedJobDescription:
    """Parse a JobDescription into structured form."""
    return HeuristicJobDescriptionParser(config=config or get_settings().job_parsing).parse(job)


def parse_job_from_document(
    document: Document,
    config: Optional[JobParsingConfig] = None,
) -> ParsedJobDescription:
    """Parse ingested document text as a job description."""
    return parse_job(
        JobDescription(raw_text=document.extracted_text, source=document.filename),
        config=config,
    )
