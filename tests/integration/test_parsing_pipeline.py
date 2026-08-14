"""Integration tests: ingestion → parsing pipeline."""

from src.ingestion.factory import ingest_document
from src.parsing.pipeline import ingest_and_parse_resume, parse_document
from src.parsing.resume_parser import HeuristicResumeParser
from tests.fixtures.document_builders import build_docx, build_txt
from tests.fixtures.resume_samples import SAMPLE_RESUME


def test_parse_document_from_ingested_txt():
    data = build_txt(SAMPLE_RESUME)
    document = ingest_document(data, filename="resume.txt")
    parsed = parse_document(document)

    assert parsed.name == "John Smith"
    assert "Python" in parsed.skills
    assert document.source_type.value == "txt"


def test_ingest_and_parse_resume_docx():
    data = build_docx(SAMPLE_RESUME.split("\n"))
    parsed = ingest_and_parse_resume(data, filename="resume.docx")

    assert parsed.name == "John Smith"
    assert len(parsed.experience) >= 1
    assert parsed.parsing_quality in {"high", "medium"}


def test_parser_independent_of_ingestion_format():
    """Same text via TXT and DOCX should produce equivalent structure."""
    txt_parsed = ingest_and_parse_resume(build_txt(SAMPLE_RESUME), filename="resume.txt")
    docx_parsed = ingest_and_parse_resume(
        build_docx([line for line in SAMPLE_RESUME.split("\n") if line.strip()]),
        filename="resume.docx",
    )

    assert txt_parsed.name == docx_parsed.name
    assert txt_parsed.email == docx_parsed.email
    assert set(txt_parsed.skills) == set(docx_parsed.skills)
