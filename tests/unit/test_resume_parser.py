"""Unit tests for heuristic resume parser."""

from __future__ import annotations

import pytest

from src.models.document import Document, DocumentSourceType
from src.models.resume import ResumeSectionType
from src.parsing.resume_parser import HeuristicResumeParser
from src.parsing.section_splitter import split_into_sections
from src.parsing.quality import assess_parsing_quality
from tests.fixtures.resume_samples import (
    INTERNSHIP_RESUME,
    MARKDOWN_HEADINGS_RESUME,
    MINIMAL_RESUME,
    SAMPLE_RESUME,
    UNSTRUCTURED_RESUME,
)


@pytest.fixture
def parser() -> HeuristicResumeParser:
    return HeuristicResumeParser()


def _doc(text: str, source: DocumentSourceType = DocumentSourceType.PLAIN_TEXT) -> Document:
    return Document(source_type=source, raw_text=text, filename="resume.txt")


class TestSectionSplitter:
    def test_splits_standard_sections(self):
        header, sections = split_into_sections(SAMPLE_RESUME)
        types = {s.section_type for s in sections}
        assert ResumeSectionType.SKILLS in types
        assert ResumeSectionType.EXPERIENCE in types
        assert ResumeSectionType.EDUCATION in types
        assert "John Smith" in header

    def test_markdown_headings(self):
        _, sections = split_into_sections(MARKDOWN_HEADINGS_RESUME)
        types = {s.section_type for s in sections}
        assert ResumeSectionType.SUMMARY in types
        assert ResumeSectionType.SKILLS in types


class TestHeuristicResumeParser:
    def test_parses_full_resume(self, parser: HeuristicResumeParser):
        parsed = parser.parse(_doc(SAMPLE_RESUME))

        assert parsed.name == "John Smith"
        assert parsed.email == "john.smith@email.com"
        assert parsed.phone is not None
        assert parsed.summary is not None
        assert "Python" in parsed.skills
        assert len(parsed.experience) >= 2
        assert len(parsed.education) >= 1
        assert len(parsed.projects) >= 1
        assert len(parsed.certifications) >= 1
        assert len(parsed.sections) >= 5
        assert parsed.parsing_quality in {"high", "medium"}
        assert parsed.years_experience is not None

    def test_parses_minimal_resume(self, parser: HeuristicResumeParser):
        parsed = parser.parse(_doc(MINIMAL_RESUME))
        assert parsed.name == "Jane Doe"
        assert "Java" in parsed.skills
        assert parsed.parsing_quality in {"medium", "low"}

    def test_unstructured_resume(self, parser: HeuristicResumeParser):
        parsed = parser.parse(_doc(UNSTRUCTURED_RESUME))
        assert parsed.email == "dev@example.com"
        assert parsed.sections == []
        assert "no section headings" in parsed.parsing_warnings[0].lower() or any(
            "section" in w.lower() for w in parsed.parsing_warnings
        )
        assert parsed.parsing_quality == "low"

    def test_markdown_headings_resume(self, parser: HeuristicResumeParser):
        parsed = parser.parse(_doc(MARKDOWN_HEADINGS_RESUME))
        assert parsed.name == "Alex Chen"
        assert "Python" in parsed.skills
        assert len(parsed.experience) >= 1

    def test_internship_experience(self, parser: HeuristicResumeParser):
        parsed = parser.parse(_doc(INTERNSHIP_RESUME))
        assert len(parsed.experience) == 1
        assert parsed.experience[0].is_internship is True

    def test_empty_document(self, parser: HeuristicResumeParser):
        parsed = parser.parse(_doc("   "))
        assert parsed.parsing_quality == "low"
        assert not parsed.raw_text.strip()

    def test_format_agnostic_source_types(self, parser: HeuristicResumeParser):
        for source in (DocumentSourceType.PDF, DocumentSourceType.DOCX, DocumentSourceType.TXT):
            parsed = parser.parse(_doc(SAMPLE_RESUME, source=source))
            assert parsed.name == "John Smith"
            assert "Python" in parsed.skills

    def test_full_text_for_matching(self, parser: HeuristicResumeParser):
        parsed = parser.parse(_doc(SAMPLE_RESUME))
        matching_text = parsed.full_text_for_matching
        assert "Python" in matching_text
        assert len(matching_text) > len(parsed.summary or "")


class TestParsingQuality:
    def test_high_quality_resume(self, parser: HeuristicResumeParser):
        parsed = parser.parse(_doc(SAMPLE_RESUME))
        assert assess_parsing_quality(parsed) in {"high", "medium"}

    def test_low_quality_unstructured(self):
        from src.models.resume import ParsedResume

        parsed = ParsedResume(raw_text="some text")
        assert assess_parsing_quality(parsed) == "low"
